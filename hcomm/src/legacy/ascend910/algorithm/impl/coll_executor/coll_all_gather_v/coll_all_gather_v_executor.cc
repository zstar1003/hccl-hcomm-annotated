/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_v_executor.h"

namespace hccl {
CollAllGatherVExecutor::CollAllGatherVExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{}

HcclResult CollAllGatherVExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    u64 count = static_cast<u64*>(param.VDataDes.counts)[topoAttr_.userRank];
    HcclResult ret = HCCL_SUCCESS;
    // 图模式场景下不需要Loop
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        ExecMem execMem;
        execMem.count = count;
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        HCCL_DEBUG(
            "[CollAllGatherVExecutor][Orchestrate]offload inputMem[%p][%llu], outputMem[%p][%llu],"
            "scratchMem[%p][%llu], inputPtr[%p] outputPtr[%p], count[%llu]",
            execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
            execMem.scratchMem.ptr(), execMem.scratchMem.size(), execMem.inputPtr, execMem.outputPtr, execMem.count);
        ret = KernelRun(param, execMem);
    } else {
        ret = RunLoop(param, algRes);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllGatherVExecutor][Orchestrate]errNo[0x%016llx]AllGatherV executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO(
        "tag[%s], AllgatherV executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVExecutor::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    (void)algRes;
    (void)adjInfo;
    return HCCL_SUCCESS;
}

u64 CollAllGatherVExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;

    HCCL_WARNING(
        "[CollAllGatherVExecutor][CalcLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.",
        maxCountPerLoop);
    return maxCountPerLoop;
}

bool CollAllGatherVExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE
                    || curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollAllGatherVExecutor::CalcCurCountsAndCurDispls(
    [[maybe_unused]] const u64 maxTotalCount, [[maybe_unused]] std::vector<u64>& countsLeft,
    [[maybe_unused]] std::vector<u64>& displs, [[maybe_unused]] std::vector<u64>& curCounts,
    [[maybe_unused]] std::vector<u64>& curDispls, [[maybe_unused]] bool& finished)
{
    HCCL_DEBUG("[CollAllGatherVExecutor][CalcCurCountsAndCurDispls]default func called.");
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVExecutor::RunLoop(OpParam& param, AlgResourceResponse& algRes)
{
    // 每轮loop需要重新计算counts和displs
    const auto* countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsLeft = std::vector<u64>(countsPtr, countsPtr + topoAttr_.userRankSize);
    const auto* displsPtr = static_cast<const u64*>(param.VDataDes.displs);
    auto displs = std::vector<u64>(displsPtr, displsPtr + topoAttr_.userRankSize);

    const HcclDataType dataType = param.VDataDes.dataType;
    u32 unitSize = SIZE_TABLE[dataType];

    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curOutputPtr = static_cast<u8*>(param.outputPtr);
    u8* commInputPtr = static_cast<u8*>(algRes.cclInputMem.ptr());
    u8* commOutputPtr = static_cast<u8*>(algRes.cclOutputMem.ptr());

    if (UNLIKELY(countsLeft[topoAttr_.userRank] == 0 && curInputPtr == nullptr)) {
        // 若本rank的input count为0，此时允许curInputPtr传入空指针，为保证后续流程正常执行，赋值为cclin的地址
        curInputPtr = commInputPtr;
        HCCL_DEBUG("Since the input count is 0, set curInputPtr to ccl input[%p]", curInputPtr);
    } else {
        CHK_PTR_NULL(curInputPtr);
    }

    CHK_PTR_NULL(curOutputPtr);
    CHK_PTR_NULL(commInputPtr);
    CHK_PTR_NULL(commOutputPtr);

    // 计算MaxCountPerLoop
    u64 maxCountPerLoop = CalcLoopMaxCount(algRes.cclInputMem.size(), unitSize); // override
    CHK_PRT_RET(
        maxCountPerLoop == 0,
        HCCL_ERROR(
            "[CollAllGatherVExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop is [%llu].",
            param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop),
        HCCL_E_PARA);

    bool finished = false;
    while (!finished) {
        // 每个块尽可能平分，以均衡利用带宽
        auto curCounts = std::vector<u64>();
        auto curDispls = std::vector<u64>();
        CHK_RET(CalcCurCountsAndCurDispls(maxCountPerLoop, countsLeft, displs, curCounts, curDispls, finished));

        // 打印调测信息
        PrintCurCountAndCurDispls(curCounts, curDispls);
        u64 totalCount = 0;
        CHK_RET(CalcTotalCount(curCounts, totalCount));
        u64 OutputSize = totalCount * unitSize; // 单位：字节

        u64 curCount = curCounts[topoAttr_.userRank];
        u64 InputSize = curCount * unitSize;
        u64 curMaxCount = *std::max_element(curCounts.begin(), curCounts.end());
        u64 curMaxSize = curMaxCount * unitSize;
        HCCL_DEBUG(
            "[CollAllGatherVExecutor][RunLoop]tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%d], "
            "OutputSize[%llu], curMaxSize[%llu].",
            param.tag.c_str(), curInputPtr, curOutputPtr, curCount, dataType, OutputSize, curMaxSize);

        if (!is310P3Common_) {
            /* 设置子图复用标志 */
            auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
            bool hugeData = IsHugeData(curMaxSize); // override
            auto opMeta = HcclOpMetaInfo::GetOneForAllGatherV(
                autoSelectedAlgTypeLevel1, hugeData, false, CopyPattern::BCOPY, false);
            CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        }

        // 执行
        if (!DMAReduceFlag_) {
            // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
            DeviceMem srcMem = DeviceMem::create(curInputPtr, InputSize);
            DeviceMem dstMem = DeviceMem::create(commInputPtr, InputSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
            HCCL_DEBUG("[CollAllGatherVExecutor][RunLoop]copy from user in to ccl in.");
        }

        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        OpParam curParam = param;
        curParam.VDataDes.counts = curCounts.data();
        curParam.VDataDes.displs = curDispls.data();
        curParam.VDataDes.dataType = dataType;
        HcclResult ret = KernelRun(curParam, execMem);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollAllGatherVExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], "
                "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d]",
                HCCL_ERROR_CODE(ret), param.tag.c_str(), commInputPtr, commOutputPtr, curCount, dataType),
            ret);

        if (!DMAReduceFlag_) {
            u64 offSetCount = 0;
            // 如果使用CCL buffer，需要将CCL buffer out中的结果拷贝到user buffer out
            for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
                // 拷贝中转output上每个slice的数据到output内存，目的端中每个slice的size固定为output的size
                DeviceMem dstMem = DeviceMem::create(curOutputPtr + curDispls[i] * unitSize, curCounts[i] * unitSize);
                DeviceMem srcMem = DeviceMem::create(commOutputPtr + offSetCount * unitSize, curCounts[i] * unitSize);
                offSetCount += curCounts[i];
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
            }
        }

        if (!is310P3Common_) {
            CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
        }

        curInputPtr += InputSize;
        // AllGatherV curOutputPtr不需要偏移，output的偏移由displs计算
    }
    return HCCL_SUCCESS;
}

void CollAllGatherVExecutor::PrintCurCountAndCurDispls(
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
            "[CollAllGatherVExecutor][PrintCurCountAndCurDispls] Current loop info: %s", curLoopInfo.str().c_str());
    }
}

HcclResult CollAllGatherVExecutor::CalcTotalCount(std::vector<u64> curCounts, u64& totalCount)
{
    for (u64 i = 0; i < topoAttr_.userRankSize; i++) {
        totalCount += curCounts[i];
    }

    return HCCL_SUCCESS;
}

} // namespace hccl
