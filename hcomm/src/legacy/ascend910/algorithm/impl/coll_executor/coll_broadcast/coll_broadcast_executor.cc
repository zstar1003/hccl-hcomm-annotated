/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_executor.h"

namespace hccl {

CollBroadcastExecutor::CollBroadcastExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{}

HcclResult CollBroadcastExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclResult ret = HCCL_SUCCESS;

    // 由于bcast/allgather/reducescatter/reduce/send/recv暂不支持server间ring，需继续使用HD或NHR
    if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR)
        && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1)
        && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
        HCCL_WARNING("[BroadCastOperator][Broadcast] do not support ring in AlgoLevel1 yet, reset algType_=HD.");
    }

    tag_ = param.tag;
    algResResp_ = &algRes;
    bool needLaunchAtTheEnd = true; // 是否需要在Orchestrate()结束时launch任务
    /*  ------------执行算法-------------- */
    HcclUs startut = TIME_NOW();

    // 图模式和单卡场景下不需要Loop
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.inputPtr;
    HCCL_INFO(
        "Orchestrate UserRank[%u], devicePhyId[%u], inputPtr[%p], outputPtr[%p], root[%u]", topoAttr_.userRank,
        topoAttr_.devicePhyId, param.inputPtr, param.outputPtr, param.root);
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) { // 图模式直接调KernelRun接口
        HCCL_DEBUG("[CollBroadcastExecutor][Orchestrate]ops kernel broadcast");
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        if (scratchMemFlag_) {
            execMem.scratchMem = algRes.scratchMem;
        }
        ret = KernelRun(param, execMem);
    } else if (topoAttr_.userRankSize == 1) { // 单卡
        HCCL_DEBUG("[CollBroadcastExecutor][Orchestrate]1 rank broadcast");
        return HCCL_SUCCESS;
    } else if (desc_.isZeroCopy) {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        ret = KernelRunIntraServerPre(param, execMem);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollBroadcastExecutor][Orchestrate]errNo[0x%016llx]Broadcast executor level0 failed",
                HCCL_ERROR_CODE(ret)),
            ret);

        // 在Level1和Level2执行RunLoop
        if (topoAttr_.serverNum > 1) {
            ret = RunLoop(param, algRes);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollBroadcastExecutor][Orchestrate]errNo[0x%016llx]Broadcast executor runloop failed. RunLoop",
                    HCCL_ERROR_CODE(ret)),
                ret);
        } else { // 单机场景，数据直接从UserInput搬到UserOutput
            std::vector<Slice> level0Datalices;
            CHK_RET(AlgTemplateBase::PrepareSliceData(
                param.DataDes.count, SIZE_TABLE[param.DataDes.dataType], topoAttr_.deviceNumPerAggregation, 0,
                level0Datalices));
            u32 level0Rank = topoAttr_.userRank % topoAttr_.deviceNumPerAggregation;
            const Slice& slice = level0Datalices[level0Rank];
            DeviceMem dstMem
                = DeviceMem::create(static_cast<u8*>(algRes.paramOutputMem.ptr()) + slice.offset, slice.size);
            DeviceMem srcMem
                = DeviceMem::create(static_cast<u8*>(algRes.paramInputMem.ptr()) + slice.offset, slice.size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
        }

        ret = KernelRunIntraServerPost(param, execMem);
    } else {
        ret = RunLoop(param, algRes);
        needLaunchAtTheEnd = false;
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollBroadcastExecutor][Orchestrate]errNo[0x%016llx]broadcast executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);

    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    if (needLaunchAtTheEnd) {
        HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }

    HCCL_INFO(
        "tag[%s], Broadcast executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastExecutor::RunLoop(OpParam& param, AlgResourceResponse& algRes)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curOutputPtr = static_cast<u8*>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);
    u64 maxCountPerLoop = CalcLoopMaxCount(algRes.cclInputMem.size(), unitSize);

    HCCL_DEBUG(
        "[CollBroadcastExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop is [%llu].", param.tag.c_str(),
        topoAttr_.userRankSize, maxCountPerLoop);

    u64 totalCount;
    if (desc_.isZeroCopy) { // 对零拷贝场景而言，只在Server间通信切循环
        std::vector<Slice> level0Datalices;
        CHK_RET(AlgTemplateBase::PrepareSliceData(
            param.DataDes.count, unitSize, topoAttr_.deviceNumPerAggregation, 0, level0Datalices));
        u32 level0Rank = topoAttr_.userRank % topoAttr_.deviceNumPerAggregation;
        totalCount = level0Datalices[level0Rank].size / unitSize;
    } else {
        totalCount = param.DataDes.count;
    }

    for (u64 countLeft = totalCount, curCount = 0, inputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclInputMem; // broadcast只用一块CCL buffer
        // 使用当前Loop偏移到的地址作为当前的inputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curInputPtr;

        HCCL_DEBUG(
            "[CollBroadcastExecutor] RunLoop tag[%s], inputOffset[%llu], "
            "curInputPtr[%p], sendCount[%llu], sendSize[%llu], dataType[%s], realUserRank[%u]",
            param.tag.c_str(), inputOffset, curInputPtr, curCount, curSize,
            GetDataTypeEnumStr(param.DataDes.dataType).c_str(), topoAttr_.realUserRank);

        CHK_RET(RunLoopInner(param, execMem));

        inputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastExecutor::RunLoopInner(OpParam& param, ExecMem& execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 totalSize = unitSize * param.DataDes.count;
    bool isRootRank = param.root == topoAttr_.realUserRank ? true : false;
    u64 curSize = execMem.count * unitSize; // 单位：字节
    auto inCCLbufferSize = execMem.inputMem.size();
    u8* curPtr = static_cast<u8*>(execMem.inputPtr);
    auto originalAlgTypeLevel0 = algType_.algoLevel0;
    bool isDMATopoOn91093 = originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING
                            || originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    bool isDMAreduceOn91093 = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
                               && (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) && isDMATopoOn91093)
                              && DMAReduceFlag_;
    HCCL_DEBUG(
        "[CollBroadcastExecutor][RunLoopInner]inputMem[%p], outputMem[%p] "
        "inputPtr[%p], curCount[%llu], curSize[%llu]",
        execMem.inputMem.ptr(), execMem.outputMem.ptr(), execMem.inputPtr, execMem.count, curSize);
    CHK_PRT_RET(
        (execMem.count == 0), HCCL_ERROR("[CollBroadcastExecutor][RunLoop]In OP_BASE curCount is zero."), HCCL_E_PARA);

    bool hugeData
        = (inCCLbufferSize / topoAttr_.deviceNumPerAggregation > RDMA_SEND_MAX_SIZE) || (curSize > SDMA_SEND_MAX_SIZE);
    bool isSmallData = IsBroadcastSmallData(curSize, totalSize);
    u64 sliceNum = 0;
    CHK_RET(GetSliceNum(curSize, isSmallData, sliceNum));
    CopyPattern copy = DMAReduceFlag_ ? CopyPattern::ZCOPY : CopyPattern::BCOPY;
    auto meta = HcclOpMetaInfo::GetOneForBroadcast(isRootRank, param.root, hugeData, isSmallData, sliceNum, copy);
    CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));
    HCCL_INFO(
        "RunLoopInner:curPtr[%p], curCount[%llu], curSize[%llu], isSmallData[%u],"
        "deviceNumPerAggregation[%u]",
        curPtr, execMem.count, curSize, isSmallData, topoAttr_.deviceNumPerAggregation);

    // 执行
    HcclResult ret;

    // isDMAreduceOn91093场景
    if (isDMAreduceOn91093) {
        if (desc_.isZeroCopy) {
            ret = KernelRunInterServer(param, execMem);
        } else {
            ret = KernelRun(param, execMem);
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollBroadcastExecutor][RunLoop]errNo[0x%016llx] DMA reduce 91093, tag[%s]", HCCL_ERROR_CODE(ret),
                tag_.c_str()),
            ret);
    } else {
        // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
        DeviceMem inCommMem = execMem.inputMem.range(0, curSize);
        DeviceMem inMem(execMem.inputPtr, curSize);
        if (topoAttr_.userRank == param.root) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, param.stream));
        }
        HCCL_DEBUG("[CollBroadcastExecutor][RunLoop]copy from user in to ccl in.");

        ret = KernelRun(param, execMem);
        if (topoAttr_.realUserRank != param.root) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inMem, inCommMem, param.stream));
        }

        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollBroadcastExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], "
                "inputMem ptr[%p], count[%llu], dataType[%d]",
                HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.count, param.DataDes.dataType),
            ret);
    }

    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    return ret;
}

u64 CollBroadcastExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / unitSize;
    HCCL_WARNING(
        "[CollBroadcastExecutor][CalcLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.",
        maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollBroadcastExecutor::GetSliceNum(const u64 size, const bool isSmallData, u64& sliceNum)
{
    u64 actualSize = 0;
    u32 actualRankSize = 0;

    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED) {
        // level0算法配null走单层拓扑场景
        actualSize = size;
        actualRankSize = topoAttr_.userRankSize;
    } else {
        // 非单层拓扑场景
        const u32 localRankSize = topoAttr_.deviceNumPerAggregation;
        const u32 localRank = topoAttr_.userRank % localRankSize;
        const u64 tempPerSlice = (size + localRankSize - 1) / localRankSize;
        const u64 sizePerSlice
            = ((tempPerSlice + (HCCL_MIN_SLICE_ALIGN - 1)) / HCCL_MIN_SLICE_ALIGN) * HCCL_MIN_SLICE_ALIGN;

        if ((localRank + 1) * sizePerSlice < size) {
            actualSize = sizePerSlice;
        } else if (localRank * sizePerSlice < size) {
            actualSize = size - localRank * sizePerSlice;
        }

        actualRankSize = topoAttr_.userRankSize / localRankSize;
    }

    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        u64 sliceSize = (actualSize + (actualRankSize - 1)) / actualRankSize;
        u64 sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSize, HCCL_MIN_SLICE_ALIGN);
        sliceNum = isSmallData ? 1 : static_cast<u64>(std::ceil(actualSize * 1.0f / sliceSizeAligned));
    }
    return HCCL_SUCCESS;
}

bool CollBroadcastExecutor::IsBroadcastSmallData(u64 size, u64 totalSize)
{
    u64 actualSize;
    u64 actualRankSize;

    if ((topoAttr_.serverNum == 1) && (topoAttr_.deviceType == DevType::DEV_TYPE_910_93)) {
        return totalSize <= topoAttr_.userRankSize * HCCL_SMALL_COUNT_2_MB;
    }

    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED
        || (topoAttr_.deviceType == DevType::DEV_TYPE_910_93 && DMAReduceFlag_ == false)) {
        // level0算法配null走单层拓扑场景
        actualSize = size;
        actualRankSize = topoAttr_.userRankSize;
    } else {
        // 非单层拓扑场景
        actualSize = size / topoAttr_.deviceNumPerAggregation;
        actualRankSize = topoAttr_.userRankSize / topoAttr_.deviceNumPerAggregation;
    }

    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        return actualSize <= NHR_BCAST_SMALL_SIZE;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        return ShouldUseBinaryBroadcastOfNB(
            actualSize, actualRankSize, topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation);
    }
    return false;
}

HcclResult CollBroadcastExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_INPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_INPUT;
    }
    HCCL_INFO(
        "[CollBroadcastExecutor][CalcTransportMemType] tag[%s] inputType[%d] outputType[%d]", tag_.c_str(), inputType,
        outputType);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastExecutor::GetRankSliceSize(
    HcclDataType dataType, const u64 count, const u32 rankSize, std::vector<Slice>& sliceList) const
{
    if (rankSize <= 0) {
        HCCL_ERROR(
            "[Get][RankSliceSize]errNo[0x%016llx] rankSize[%u] is invalid", HCCL_ERROR_CODE(HCCL_E_PARA), rankSize);
        return HCCL_E_PARA;
    }

    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    u64 align = (count * perDataSize) / rankSize; // 按128字节对齐整除均分
    if ((count % rankSize) > 0) {
        align += 1;
    }

    u64 sliceSize = AlgTemplateBase::RoundUpWithDivisor(align, HCCL_MIN_SLICE_ALIGN);
    u64 residueSize = count * perDataSize;

    for (u32 i = 0; i < rankSize; i++) {
        Slice slice;
        slice.size = sliceSize < residueSize ? sliceSize : residueSize;
        slice.offset = (slice.size == 0) ? 0 : (i * sliceSize);
        residueSize -= slice.size;

        // 将cout转换为字节数
        sliceList.push_back(slice);
    }

    return HCCL_SUCCESS;
}
} // namespace hccl
