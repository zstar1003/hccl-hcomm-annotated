/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_executor.h"
#include <numeric>

namespace hccl {

CollReduceScatterExecutor::CollReduceScatterExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{}

HcclResult CollReduceScatterExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    ParseParam(param);
    tag_ = param.tag;
    algResResp_ = &algRes;
    const u64 count = param.GetDataCount(topoAttr_.userRank);
    const HcclDataType dataType = param.GetDataType();
    HcclResult ret = HCCL_SUCCESS;
    bool needLaunchAtTheEnd = !is310P3Common_; // 是否需要在Orchestrate()结束时launch任务
    // 图模式和单卡场景下不需要Loop
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        ExecMem execMem;
        execMem.count = count;
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        ret = KernelRun(param, execMem);
        if (algOpContext_.opRetryHandler.isPostSync == true) {
            // post Sync
            CHK_RET(RetryPostSync(param, execMem));
        }
    } else if (topoAttr_.userRankSize == 1) {
        ExecMem execMem;
        execMem.count = count;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
        needLaunchAtTheEnd = false;
    } else if (desc_.isZeroCopy) {
        // 在Level0执行KernelRun
        ExecMem execMem;
        execMem.count = count;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.paramInputMem;
        ret = KernelRunIntraServerPre(param, execMem);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollReduceScatterExecutor][Orchestrate]errNo[0x%016llx]ReduceScatter executor "
                "KernelRunIntraServerPre failed",
                HCCL_ERROR_CODE(ret)),
            ret);
        if (algOpContext_.opRetryHandler.isPostSync == true) {
            // post Sync
            CHK_RET(RetryPostSync(param, execMem));
        }
        // 在Level1和Level2执行RunLoop
        if (topoAttr_.serverNum > 1) {
            ret = RunLoop(param, algRes);
        } else { // 单机场景，数据直接从UserInput搬到UserOutput
            u64 totalSize = count * SIZE_TABLE[dataType];
            DeviceMem srcMem = DeviceMem::create(
                static_cast<u8*>(algRes.paramInputMem.ptr()) + totalSize * topoAttr_.userRank, totalSize);
            DeviceMem dstMem = DeviceMem::create(algRes.paramOutputMem.ptr(), totalSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
        }
    } else {
        if (algOpContext_.opRetryHandler.isInplacePreSync == true) {
            /*当重执行场景，UserInMem > CCLBuffer时，需要在reduce scatter算子前增加一个PreSync函数，提升重执行成功概率*/
            ExecMem execMem;
            execMem.count = count;
            execMem.inputPtr = param.inputPtr;
            execMem.outputPtr = param.outputPtr;
            execMem.inputMem = algRes.cclInputMem;
            execMem.outputMem = algRes.cclOutputMem;
            execMem.scratchMem = algRes.scratchMem;
            ret = InplaceOpSync(param, execMem);
        } else if (isReduceScatterV_) {
            ret = RunLoopV(param, algRes);
            needLaunchAtTheEnd = false;
        } else {
            ret = RunLoop(param, algRes);
            needLaunchAtTheEnd = false;
        }
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterExecutor][Orchestrate]errNo[0x%016llx]executor kernel run failed", HCCL_ERROR_CODE(ret)),
        ret);

    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    if (needLaunchAtTheEnd) {
        HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }

    HCCL_INFO(
        "tag[%s], ReduceScatter executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

u64 CollReduceScatterExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop
        = inCCLbufferSize_ / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    HCCL_INFO(
        "[CollReduceScatterExecutor][CalcLoopMaxCount]using default maxCountPerLoop[%llu] as "
        "CCLBuffSize / (userRankSize * unitSize). rsv[%u]",
        maxCountPerLoop, isReduceScatterV_);
    return maxCountPerLoop;
}

bool CollReduceScatterExecutor::IsHugeData(const u64 curSize, [[maybe_unused]] OpParam* param)
{
    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE)
                    || (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

bool CollReduceScatterExecutor::IsSmallData([[maybe_unused]] const u64 totalSize, [[maybe_unused]] const u64 curSize)
{
    HCCL_INFO("[CollReduceScatterExecutor][IsSmallData]opMeta is using the default option: not small data.");
    return false;
}

HcclResult CollReduceScatterExecutor::RunLoop(OpParam& param, AlgResourceResponse& algRes)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType
        = ((param.reduceType != HCCL_REDUCE_PROD) && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
              ReduceType::INLINE_REDUCE :
              ReduceType::TBE_REDUCE;

    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curOutputPtr = static_cast<u8*>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(unitSize);
    CHK_PRT_RET(
        maxCountPerLoop == 0, HCCL_ERROR("[CollReduceScatterExecutor][RunLoop]maxCountPerLoop is zero."),
        HCCL_E_INTERNAL);
    HCCL_DEBUG(
        "[CollReduceScatterExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop is [%llu].",
        param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop);
    HcclResult ret;
    for (u64 countLeft = param.DataDes.count, curCount = 0, inputOffset = 0, outputOffset = 0; countLeft > 0;
         countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        HCCL_DEBUG(
            "[CollReduceScatterExecutor][RunLoop]tag[%s], inputOffset[%llu], outputOffset[%llu], "
            "sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%d].",
            param.tag.c_str(), inputOffset, outputOffset, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        if (scratchMemFlag_) {
            execMem.scratchMem = algRes.scratchMem;
        } else {
            execMem.scratchMem = algRes.cclOutputMem; // 不需要申请则传入outputmem为scratchmem
        }
        HCCL_DEBUG("[CollReduceScatterExecutor][RunLoop]scratchMem address [%p]", execMem.scratchMem.ptr());

        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        ret = RunLoopInner(param, reduceType, execMem);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollReduceScatterExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s]", HCCL_ERROR_CODE(ret),
                param.tag.c_str()),
            ret);

        inputOffset = curSize;
        outputOffset = curSize;
    }
    if (algOpContext_.opRetryHandler.isPostSync == true) {
        ExecMem execMem;
        execMem.count = param.DataDes.count;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        CHK_RET(RetryPostSync(param, execMem));
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterExecutor::RunLoopInner(OpParam& param, const ReduceType& reduceType, ExecMem& execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    CHK_PRT_RET(
        (execMem.count == 0), HCCL_ERROR("[CollReduceScatterExecutor][RunLoopInner]In OP_BASE curCount is zero."),
        HCCL_E_PARA);

    // 不开启dma消减，且通信buffer足够大时，将user in到ccl的拷贝任务合并成一个
    const bool preloadCopyOpt = IsPreloadCopyOptimizeCondition(param, execMem);

    if (!is310P3Common_) {
        /* 设置子图复用标志 */
        auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
        bool hugeData = IsHugeData(curSize, &param);
        bool smallData = IsSmallData(param.DataDes.count * unitSize, curSize);
        bool dataSplit = false;
        u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
        auto opMeta = HcclOpMetaInfo::GetOneForReduceScatter(
            autoSelectedAlgTypeLevel1, param.DataDes.dataType, reduceType, hugeData, smallData, CopyPattern::BCOPY,
            dataSplit, deterministic, false, preloadCopyOpt);

        CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
    }

    if (CCLMemSlice_) {
        u32 sliceNum = desc_.isZeroCopy ? topoAttr_.serverNum : topoAttr_.userRankSize;
        execMem.inputMem = execMem.inputMem.range(0, curSize * sliceNum);
        execMem.outputMem = execMem.outputMem.range(0, curSize);
        if (scratchMemFlag_) {
            execMem.scratchMem = execMem.scratchMem.range(0, curSize * topoAttr_.userRankSize);
        }
    }

    // 执行
    if (!DMAReduceFlag_) { // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
        DeviceMem dstMem;
        DeviceMem srcMem;
        if (preloadCopyOpt) {
            // 中转内存大小足够时，一次性搬完
            const u64 copySize = param.DataDes.count * unitSize * topoAttr_.userRankSize;
            dstMem = execMem.inputMem.range(0, copySize);
            srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), copySize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
        } else {
            for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
                // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
                dstMem = execMem.inputMem.range(curSize * i, curSize);
                srcMem = DeviceMem::create(
                    static_cast<u8*>(execMem.inputPtr) + param.DataDes.count * unitSize * i, curSize);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
            }
        }
    }

    HcclResult ret = HCCL_SUCCESS;
    if (!desc_.isZeroCopy) {
        ret = KernelRun(param, execMem);
    } else {
        ret = KernelRunInterServer(param, execMem);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterExecutor][RunLoopInner]errNo[0x%016llx]kernel run error, tag[%s], "
            "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d], preloadCopyOpt[%d]",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(), execMem.count,
            param.DataDes.dataType, param.reduceType, preloadCopyOpt),
        ret);

    if (!DMAReduceFlag_) {
        // 如果使用CCL buffer，需要将CCL buffer out中的结果拷贝到user buffer out
        DeviceMem srcMem = execMem.outputMem.range(0, curSize);
        DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
    }
    HCCL_DEBUG(
        "[CollReduceScatterExecutor][RunLoopInner]inputMem ptr is [%p], outputMem ptr is [%p]", execMem.inputMem.ptr(),
        execMem.outputMem.ptr());

    if (!is310P3Common_) {
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return ret;
}

HcclResult CollReduceScatterExecutor::RunLoopV(OpParam& param, AlgResourceResponse& algRes)
{
    // 每轮loop需要重新计算counts和displs
    const auto* countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsLeft = std::vector<u64>(countsPtr, countsPtr + topoAttr_.userRankSize);
    const auto* displsPtr = static_cast<const u64*>(param.VDataDes.displs);
    auto displs = std::vector<u64>(displsPtr, displsPtr + topoAttr_.userRankSize);

    const HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curOutputPtr = static_cast<u8*>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);

    if (UNLIKELY(countsLeft[topoAttr_.userRank] == 0 && curOutputPtr == nullptr)) {
        // 若本rank的output count为0，此时允许curOutputPtr传入空指针，为保证后续流程正常执行，赋值为cclout的地址
        curOutputPtr = static_cast<u8*>(algRes.cclOutputMem.ptr());
        HCCL_DEBUG(
            "[CollReduceScatterExecutor][RunLoopV]Since the output count is 0, set curOutputPtr to "
            "ccl output[%p]",
            curOutputPtr);
    }
    CHK_PTR_NULL(curOutputPtr);

    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) && (dataType != HCCL_DATA_TYPE_INT64)) ?
                                ReduceType::INLINE_REDUCE :
                                ReduceType::TBE_REDUCE;

    // 计算MaxCountPerLoop
    const u64 maxCountPerLoop = CalcLoopMaxCount(unitSize);

    HcclResult ret;
    bool finished = false;
    while (!finished) {
        // 每个块尽可能平分，以均衡利用带宽
        auto curCounts = std::vector<u64>();
        auto curDispls = std::vector<u64>();
        finished = CalcCurCountsAndCurDispls(maxCountPerLoop, countsLeft, displs, curCounts, curDispls, unitSize);
        // 打印调测信息
        PrintCurCountAndCurDispls(curCounts, curDispls);

        OpParam curParam = param;
        curParam.VDataDes.counts = curCounts.data();
        curParam.VDataDes.displs = curDispls.data();
        curParam.VDataDes.dataType = dataType;

        ExecMem execMem;
        execMem.count = curCounts[topoAttr_.userRank];
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        if (scratchMemFlag_) {
            execMem.scratchMem = algRes.scratchMem;
        } else {
            execMem.scratchMem = algRes.cclOutputMem; // 不需要申请则传入outputmem为scratchmem
        }
        ret = RunLoopInnerV(curParam, reduceType, execMem);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollReduceScatterExecutor][RunLoopV]errNo[0x%016llx]kernel run error, tag[%s]", HCCL_ERROR_CODE(ret),
                curParam.tag.c_str()),
            ret);

        const auto outputSize = curCounts[topoAttr_.userRank] * unitSize;
        curOutputPtr += outputSize;
        // ReduceScatterV curInputPtr不需要偏移，input的偏移由displs计算
        HCCL_DEBUG("[CollReduceScatterExecutor][RunLoopV]kernel run, finished[%u]", finished);
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterExecutor::RunLoopInnerV(OpParam& param, const ReduceType& reduceType, ExecMem& execMem)
{
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    u64 count = counts[topoAttr_.userRank];
    HcclDataType dataType = param.VDataDes.dataType;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = count * unitSize; // 单位：字节;

    /* 设置子图复用标志 */
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = IsHugeData(curSize, &param);
    u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
    auto opMeta = HcclOpMetaInfo::GetOneForReduceScatterV(
        autoSelectedAlgTypeLevel1, dataType, reduceType, hugeData, false, CopyPattern::BCOPY, false, deterministic);

    CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));

    if (CCLMemSlice_) {
        const u64 inputCounts = std::accumulate(counts, counts + topoAttr_.userRankSize, 0ULL);
        execMem.inputMem = execMem.inputMem.range(0, inputCounts * unitSize);
        execMem.outputMem = execMem.outputMem.range(0, curSize);
        if (scratchMemFlag_) {
            execMem.scratchMem = execMem.scratchMem.range(0, inputCounts * unitSize);
        }
    }

    // 执行
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterExecutor][RunLoopInnerV]errNo[0x%016llx]kernel run error, tag[%s], "
            "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d]",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(), execMem.count,
            dataType, param.reduceType),
        ret);

    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    return ret;
}

bool CollReduceScatterExecutor::CalcCurCountsAndCurDispls(
    const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
    std::vector<u64>& curDispls, u32 unitSize) const
{
    bool finished = false;

    curCounts = std::vector<u64>(countsLeft.size(), 0);
    curDispls = std::vector<u64>(displs.size(), 0);
    auto allocatableCount = maxTotalCount;

    // 先设置本轮的displacements，等于入参displs
    std::copy(displs.begin(), displs.end(), curDispls.begin());

    // 分配本轮的counts，如果CCLbuffer空间还没完全利用，则再进行分配
    while (allocatableCount > 0) {
        // 计算现在还有几个rank还有数据需要去通信(countsLeft不为0)
        const auto nonZeroCount = std::count_if(countsLeft.begin(), countsLeft.end(), [](const u64 count) {
            return count != 0;
        });
        if (nonZeroCount == 0) {
            finished = true;
            break;
        }

        // 计算每个rank可以分到多少count
        auto perRankCount = allocatableCount / nonZeroCount;
        if (perRankCount == 0) {
            break;
        }

        const u64 perRankSize = perRankCount * unitSize;
        if (perRankSize > HCCL_MIN_SLICE_ALIGN) {
            perRankCount = perRankSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize; // align for perf
        } else if ((perRankSize < HCCL_MIN_SLICE_ALIGN) && (allocatableCount != maxTotalCount)) {
            break;
        }

        // 分配好每个rank的counts
        for (auto i = 0U; i < countsLeft.size(); ++i) {
            const auto curCount = countsLeft[i] < perRankCount ? countsLeft[i] : perRankCount;
            allocatableCount -= curCount;
            curCounts[i] += curCount;
            countsLeft[i] -= curCount;
            displs[i] += curCount;
        }
    }
    return finished;
}

void CollReduceScatterExecutor::PrintCurCountAndCurDispls(
    const std::vector<u64>& curCounts, const std::vector<u64>& curDispls) const
{
    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        std::ostringstream curLoopInfo;
        curLoopInfo << "counts[ ";
        for (auto count : curCounts) {
            curLoopInfo << count << " ";
        }
        curLoopInfo << "], displs[ ";
        for (auto displ : curDispls) {
            curLoopInfo << displ << " ";
        }
        curLoopInfo << "]";
        HCCL_DEBUG(
            "[CollReduceScatterExecutor][PrintCurCountAndCurDispls] Current loop info: %s", curLoopInfo.str().c_str());
    }
}

std::vector<std::vector<Slice>> CollReduceScatterExecutor::ReduceScatterRingSlicePrepare(
    u32 ringNum, u32 sliceNum, bool useInlineReduce, const DeviceMem& outputMem, std::vector<Slice>& dataSegsSlice,
    const std::string& tag)
{
    std::vector<std::vector<Slice>> multiStreamSlice;
    u64 outputMemSize = outputMem.size();
    dataSegsSlice.clear();
    Slice sliceTemp;
    for (u32 i = 0; i < sliceNum; i++) { // 根据数据量算每个环上数据的偏移和大小
        sliceTemp.size = outputMemSize;
        sliceTemp.offset = outputMemSize * i;
        dataSegsSlice.push_back(sliceTemp);
    }
    bool ARSFlag = topoMatcher_->GetARSFlag();
    auto nicList = topoAttr_.nicList;
    if (ARSFlag) {
        std::vector<u32> mockNicList;
        for (u32 i = 0; i < sliceNum; i++) {
            mockNicList.push_back(i);
        }
        nicList = mockNicList;
    }

    // 再将每个 slice 划分为 ringNum 份
    if (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) {
        if (useInlineReduce) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag);
        } else if (outputMem.size() % CCE_REDUCE_ALIGN_SIZE == 0) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag);
        } else {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, true);
        }
    } else if (ringNum == LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE) {
        // 双环场景，需要传入正确的 niclist (不涉及网口裁剪)
        if (useInlineReduce) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList);
        } else if (outputMem.size() % CCE_REDUCE_ALIGN_SIZE == 0) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList);
        } else {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, true, nicList);
        }
    } else {
        multiStreamSlice.push_back(dataSegsSlice);
    }

    return multiStreamSlice;
}

HcclResult CollReduceScatterExecutor::PrepareAivBuffers(
    u32 rankSize, u32 rankId, u32 rankOffset, DeviceMem& inputMem, DeviceMem& outputMem, std::vector<LINK>& links,
    void** dataBuffers, void** flagBuffers, UserMemType dataMemType, UserMemType flagMemType, u32 dataMemOffset,
    u32 flagMemOffset) const
{
    void* tmpCCLBufferData = nullptr;
    void* tmpCCLBufferFlag = nullptr;
    for (u32 i = 0; i < rankSize; i++) {
        if (i != rankId) {
            if (links[i + rankOffset] != nullptr) {
                CHK_RET(links[i + rankOffset]->GetRemoteMem(dataMemType, &(tmpCCLBufferData)));
                CHK_RET(links[i + rankOffset]->GetRemoteMem(flagMemType, &(tmpCCLBufferFlag)));
                dataBuffers[i] = static_cast<u8*>(tmpCCLBufferData) + dataMemOffset;
                flagBuffers[i] = static_cast<u8*>(tmpCCLBufferFlag) + flagMemOffset;
            }
        } else {
            dataBuffers[i] = static_cast<u8*>(inputMem.ptr()) + dataMemOffset;
            flagBuffers[i] = static_cast<u8*>(outputMem.ptr()) + flagMemOffset;
        }
    }
    return HCCL_SUCCESS;
}

std::vector<std::vector<Slice>> CollReduceScatterExecutor::AnyPathReduceScatterRingSlicePrepare(
    u32 ringNum, u32 sliceNum, bool useInlineReduce, DeviceMem& outputMem, std::vector<Slice>& dataSegsSlice,
    const std::string& tag)
{
    std::vector<std::vector<Slice>> multiStreamSlice;
    u64 outputMenSize = outputMem.size();
    dataSegsSlice.clear();
    Slice sliceTemp;
    for (u32 i = 0; i < sliceNum; i++) { // 根据数据量算每个环上数据的偏移和大小
        sliceTemp.size = outputMenSize;
        sliceTemp.offset = outputMenSize * i;
        dataSegsSlice.push_back(sliceTemp);
    }

    // 再将每个 slice 划分为 ringNum 份
    if (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) {
        if (useInlineReduce) {
            multiStreamSlice = AnyPathPrepareMultiRingSlice(dataSegsSlice, tag);
        } else if (outputMem.size() % CCE_REDUCE_ALIGN_SIZE == 0) {
            multiStreamSlice = AnyPathPrepareMultiRingSlice(dataSegsSlice, tag);
        } else {
            multiStreamSlice = AnyPathPrepareMultiRingSlice(dataSegsSlice, tag, true);
        }
    } else if (ringNum == LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE) {
        // 双环场景，需要传入正确的 niclist (不涉及网口裁剪)
        if (useInlineReduce) {
            multiStreamSlice = AnyPathPrepareMultiRingSlice(dataSegsSlice, tag, false, topoAttr_.nicList);
        } else if (outputMem.size() % CCE_REDUCE_ALIGN_SIZE == 0) {
            multiStreamSlice = AnyPathPrepareMultiRingSlice(dataSegsSlice, tag, false, topoAttr_.nicList);
        } else {
            multiStreamSlice = AnyPathPrepareMultiRingSlice(dataSegsSlice, tag, true, topoAttr_.nicList);
        }
    } else {
        multiStreamSlice.push_back(dataSegsSlice);
    }

    return multiStreamSlice;
}

HcclResult CollReduceScatterExecutor::RetryPostSync(OpParam& param, ExecMem& execMem)
{
    if ((algResResp_->slaveStreams).size() == 0) {
        CHK_RET(PostSyncWithoutSubstream(param, execMem));
    } else {
        PrepareData postSyncPrepareData;
        postSyncPrepareData.subStreamsPtr = &algResResp_->slaveStreams;
        postSyncPrepareData.signalPtr = &algResResp_->notifiesMain;
        postSyncPrepareData.signalAuxPtr = &algResResp_->notifiesAux;
        postSyncPrepareData.stream = param.stream;
        CHK_RET(PostSyncWithSubstream(param, execMem, postSyncPrepareData));
    }
    return HCCL_SUCCESS;
}

bool CollReduceScatterExecutor::IsPreloadCopyOptimizeCondition(const OpParam& param, ExecMem& execMem)
{
    // 不开启dma消减，且通信buffer足够大时，将user in到ccl的拷贝任务合并成一个
    return (!DMAReduceFlag_) && (param.DataDes.count == execMem.count);
}
} // namespace hccl
