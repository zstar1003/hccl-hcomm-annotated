/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_nhr.h"
#include "alg_template_register.h"

namespace hccl {
BroadcastNHR::BroadcastNHR(const HcclDispatcher dispatcher) : NHRBase(dispatcher) {}

BroadcastNHR::~BroadcastNHR() {}

HcclResult
BroadcastNHR::RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("[BroadcastNHR][RunAsync] run: rank[%u] totalrank[%u] count[%llu]", rank, rankSize, count_);

    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR("[BroadcastNHR][RunAsync] rank[%u] linksize[%llu] is less than rank size", rank, links.size()),
        HCCL_E_INTERNAL);

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(
        unitSize == 0, HCCL_ERROR("[BroadcastNHR][RunAsync] rank[%u] unit data size is zero", rank), HCCL_E_INTERNAL);

    DeviceMem srcMem = inputMem_.range(baseOffset_, count_ * unitSize);
    DeviceMem dstMem = outputMem_.range(baseOffset_, count_ * unitSize);

    if (inputMem_ != outputMem_ && rank == root_) {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream_));
    }

    // 准备slice
    PrepareSlice(rank, rankSize);

    // scatter
    std::unique_ptr<AlgTemplateBase> tempAlgScatter
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NHR, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlgScatter);
    CHK_RET(tempAlgScatter->Prepare(true));
    tempAlgScatter->CloseBarrier();
    CHK_RET(tempAlgScatter->Prepare(
        srcMem, srcMem, srcMem, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(
        tempAlgScatter->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));
    CHK_RET(tempAlgScatter->RunAsync(rank, rankSize, links));

    // allgather
    std::unique_ptr<AlgTemplateBase> tempAlgAllgather
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlgAllgather);
    CHK_RET(tempAlgAllgather->Prepare(true));
    CHK_RET(tempAlgAllgather->Prepare(
        srcMem, srcMem, srcMem, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(
        tempAlgAllgather->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));
    CHK_RET(tempAlgAllgather->RunAsync(rank, rankSize, links));

    HCCL_INFO("[BroadcastNHR][RunAsync] finished: rank[%u] end count[%llu]", rank, count_);
    return HCCL_SUCCESS;
}

HcclResult BroadcastNHR::PrepareSlice(const u32 rank, const u32 rankSize)
{
    u32 unitSize = DataUnitSize(dataType_);

    //  所有的数据平均到每个rank上
    u64 sizeAvg = ((count_ + rankSize - 1) / rankSize) * unitSize;
    u64 sizePerSlice = AlgTemplateBase::RoundUpWithDivisor(sizeAvg, HCCL_MIN_SLICE_ALIGN);
    HCCL_DEBUG(
        "[BroadcastNHR][PrepareSlice] bcast total count[%llu] sizeAverage[%llu] sizePerSlice after aligns[%llu]",
        count_, sizeAvg, sizePerSlice);

    // 准备slice
    slices_.resize(rankSize);
    u64 sizeResidue = count_ * unitSize;
    u64 sizePerRound = 0;

    for (u32 i = 0; i < rankSize; i++) {
        sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
        slices_[i].offset = count_ * unitSize - sizeResidue;
        slices_[i].size = sizePerRound;

        sizeResidue -= sizePerRound;
        HCCL_DEBUG(
            "[BroadcastNHR][PrepareSlice] rank[%u] default slice[%u]: offset: [%llu] size[%llu]", rank, i,
            slices_[i].offset, slices_[i].size);
    }
    return HCCL_SUCCESS;
}

HcclResult
BroadcastNHR::GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }
    if (links.size() < rankSize) {
        return HCCL_SUCCESS;
    }
    u32 nSteps = 0;
    for (u32 temp = rankSize - 1; temp != 0; temp >>= 1, ++nSteps) {
    }

    u32 deltaRoot = (rankSize - rank) % rankSize;
    // 先执行 scatter 流程
    for (u32 step = 0; step < nSteps; step++) {
        u32 deltaRankPair = 1 << step;
        u32 nRanks = 0;
        bool isPerfect = (rankSize & (rankSize - 1)) == 0;
        if (!isPerfect && step == nSteps - 1) {
            nRanks = rankSize - deltaRankPair;
        } else {
            nRanks = deltaRankPair;
        }
        if (deltaRoot >= nRanks) {
            continue;
        }
        u32 sendTo = (rank + rankSize - deltaRankPair) % rankSize;
        LINK linkRight = links[sendTo];
        CHK_SMART_PTR_NULL(linkRight);

        NslbDpAdjInfo adjInfoStep = {};
        adjInfoStep.dstLocalRankId = linkRight->GetRemoteRank();
        adjInfoStep.phaseId = step + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    u32 begin = nSteps;
    // 后续执行AllGather的NHR流程
    for (u32 step = 0; step < nSteps; step++) {
        u32 deltaRank = 1 << (nSteps - 1 - step);
        u32 sendTo = (rank + deltaRank) % rankSize;
        LINK linkRight = links[sendTo];
        CHK_SMART_PTR_NULL(linkRight);

        NslbDpAdjInfo allGatherInfoStep = {};
        allGatherInfoStep.dstLocalRankId = linkRight->GetRemoteRank();
        allGatherInfoStep.phaseId = step + begin + 1;
        allGatherInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(allGatherInfoStep);
    }
    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_BROADCAST_NHR, BroadcastNHR);
} // namespace hccl
