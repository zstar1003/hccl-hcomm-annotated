/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_v_executor.h"

namespace hccl {

CollReduceScatterVExecutor::CollReduceScatterVExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{}

HcclResult CollReduceScatterVExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    ParseParam(param);
    tag_ = param.tag;
    algResResp_ = &algRes;
    u64 count = static_cast<u64*>(param.VDataDes.counts)[topoAttr_.userRank];
    HcclResult ret = HCCL_SUCCESS;
    // 图模式场景下不需要Loop
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        ExecMem execMem;
        execMem.count = count;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
    } else {
        ret = RunLoop(param, algRes);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterVExecutor][Orchestrate]errNo[0x%016llx]executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO(
        "tag[%s], ReduceScatterV executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVExecutor::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    (void)algRes;
    (void)adjInfo;
    return HCCL_SUCCESS;
}

u64 CollReduceScatterVExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，这里不除以RankSize，因为每次循环可能会减少需要参与通信的Rank
    u64 maxCountPerLoop = inCCLbufferSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    HCCL_INFO(
        "[CollReduceScatterVExecutor][CalcLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.",
        maxCountPerLoop);
    return maxCountPerLoop;
}

bool CollReduceScatterVExecutor::IsHugeData(const u64 curSize, const OpParam& param)
{
    (void)param;
    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE)
                    || (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

HcclResult CollReduceScatterVExecutor::CalcCurCountsAndCurDispls(
    [[maybe_unused]] const u64 maxTotalCount, [[maybe_unused]] std::vector<u64>& countsLeft,
    [[maybe_unused]] std::vector<u64>& displs, [[maybe_unused]] std::vector<u64>& curCounts,
    [[maybe_unused]] std::vector<u64>& curDispls, [[maybe_unused]] bool& finished)
{
    HCCL_DEBUG("[CollReduceScatterVExecutor][CalcCurCountsAndCurDispls]default func called.");
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVExecutor::RunLoop(OpParam& param, AlgResourceResponse& algRes)
{
    // 每轮loop需要重新计算counts和displs
    const auto* countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsLeft = std::vector<u64>(countsPtr, countsPtr + topoAttr_.userRankSize);
    const auto* displsPtr = static_cast<const u64*>(param.VDataDes.displs);
    auto displs = std::vector<u64>(displsPtr, displsPtr + topoAttr_.userRankSize);

    const HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];
    HCCL_DEBUG("[CollReduceScatterVExecutor][RunLoop]unitSize is %u", unitSize);
    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curOutputPtr = static_cast<u8*>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);

    if (UNLIKELY(countsLeft[topoAttr_.userRank] == 0 && curOutputPtr == nullptr)) {
        // 若本rank的output count为0，此时允许curOutputPtr传入空指针，为保证后续流程正常执行，赋值为cclout的地址
        curOutputPtr = static_cast<u8*>(algRes.cclOutputMem.ptr());
        HCCL_DEBUG("Since the output count is 0, set curOutputPtr to ccl output[%p]", curOutputPtr);
    } else {
        CHK_PTR_NULL(curOutputPtr);
    }

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
        CHK_RET(CalcCurCountsAndCurDispls(maxCountPerLoop, countsLeft, displs, curCounts, curDispls, finished));
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
        ret = RunLoopInner(curParam, reduceType, execMem);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollReduceScatterVExecutor][RunLoopForVaringCounts]errNo[0x%016llx]kernel run error, tag[%s]",
                HCCL_ERROR_CODE(ret), curParam.tag.c_str()),
            ret);
        curOutputPtr += curCounts[topoAttr_.userRank] * unitSize;
        // ReduceScatterV curInputPtr不需要偏移，input的偏移由displs计算
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVExecutor::RunLoopInner(OpParam& param, const ReduceType& reduceType, ExecMem& execMem)
{
    u64 count = static_cast<u64*>(param.VDataDes.counts)[topoAttr_.userRank];
    HcclDataType dataType = param.VDataDes.dataType;

    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = count * unitSize; // 单位：字节;

    if (!is310P3Common_) {
        /* 设置子图复用标志 */
        auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
        bool hugeData = IsHugeData(curSize, param);
        u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
        auto opMeta = HcclOpMetaInfo::GetOneForReduceScatterV(
            autoSelectedAlgTypeLevel1, dataType, reduceType, hugeData, false, CopyPattern::BCOPY, false, deterministic);

        CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
    }

    if (CCLMemSlice_) {
        auto inputCounts = 0ULL;
        for (auto rank = 0U; rank < topoAttr_.userRankSize; ++rank) {
            auto count = static_cast<u64*>(param.VDataDes.counts)[rank];
            inputCounts += count;
        }
        execMem.inputMem = execMem.inputMem.range(0, inputCounts * unitSize);
        execMem.outputMem = execMem.outputMem.range(0, inputCounts * unitSize);
        if (scratchMemFlag_) {
            execMem.scratchMem = execMem.scratchMem.range(0, inputCounts * unitSize);
        }
    }

    if (!DMAReduceFlag_) {
        // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
        auto cclOffset = 0ULL;
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
            const auto offset = static_cast<u64*>(param.VDataDes.displs)[i] * unitSize;
            const auto size = static_cast<u64*>(param.VDataDes.counts)[i] * unitSize;
            DeviceMem dstMem = execMem.inputMem.range(cclOffset, size);
            DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(param.inputPtr) + offset, size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
            cclOffset += size;
        }
        HCCL_DEBUG("[CollReduceScatterVExecutor][RunLoopInner]copy from user in to ccl in.");
    }

    // 执行
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterVExecutor][RunLoopInner]errNo[0x%016llx]kernel run error, tag[%s], "
            "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d]",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(), execMem.count,
            dataType, param.reduceType),
        ret);

    if (!DMAReduceFlag_) {
        // CO->UO
        DeviceMem srcMem = execMem.outputMem.range(0, curSize);
        DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
    }
    if (!is310P3Common_) {
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return ret;
}

void CollReduceScatterVExecutor::PrintCurCountAndCurDispls(
    const std::vector<u64>& curCounts, const std::vector<u64>& curDispls) const
{
    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        std::ostringstream curLoopInfo;
        curLoopInfo << "Counts[ ";
        for (auto count : curCounts) {
            curLoopInfo << count << " ";
        }
        curLoopInfo << "], displs[ ";
        for (auto displ : curDispls) {
            curLoopInfo << displ << " ";
        }
        curLoopInfo << "]";
        HCCL_DEBUG(
            "[CollReduceScatterVExecutor][PrintCurCountAndCurDispls] Current loop info: %s", curLoopInfo.str().c_str());
    }
}

} // namespace hccl
