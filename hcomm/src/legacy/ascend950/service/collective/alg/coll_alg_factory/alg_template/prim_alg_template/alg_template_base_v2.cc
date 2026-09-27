/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_base_v2.h"
#include "log.h"

namespace Hccl {
AlgTemplateBase::AlgTemplateBase(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : myRank_(virtualRank),
      tempRankSize_(tempRankSize),
      tempVTopo_(tempVTopo),
      tempVirtRankMap_(tempVirtRankMap)
{}

AlgTemplateBase::~AlgTemplateBase() {}

void AlgTemplateBase::SetDmaMode(const DmaMode dmaMode)
{
    dmaMode_ = dmaMode;
    return;
}

HcclResult AlgTemplateBase::PostCopyOpbase(const UsrData& usrData, std::vector<PrimQuePtr>& tempPrimQues) const
{
    for (u32 i = 0; i < usrData.scratchOutSlices.size(); i++) {
        std::unique_ptr<Primitive> primLocalCopy
            = std::make_unique<PrimLocalCopy>(usrData.scratchOutSlices[i], usrData.usrOutSlices[i]);
        tempPrimQues[0]->Append(std::move(primLocalCopy));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::PreCopyOpbase(const UsrData& usrData, std::vector<PrimQuePtr>& tempPrimQues) const
{
    for (u32 i = 0; i < usrData.usrInSlices.size(); i++) {
        std::unique_ptr<Primitive> primLocalCopy
            = std::make_unique<PrimLocalCopy>(usrData.usrInSlices[i], usrData.scratchInSlices[i]);
        tempPrimQues[0]->Append(std::move(primLocalCopy));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::CalcSliceInfo(
    const AllignInfo& allignInfo, const bool forAllReduce, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    (void)allignInfo;
    (void)forAllReduce;
    (void)dataSize;
    (void)sliceInfoVec;
    HCCL_ERROR("[CollAlgFactory] Unsupported interface of slice info calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult AlgTemplateBase::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    (void)allignInfo;
    (void)dataSize;
    (void)sliceInfoVec;
    HCCL_ERROR("[CollAlgFactory] Unsupported interface of slice info calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult AlgTemplateBase::CalcRes(const bool forAllReduce, AlgTempResReq& tempResReq, u32& requiredScratchMultiplier)
{
    (void)forAllReduce;
    (void)tempResReq;
    (void)requiredScratchMultiplier;
    HCCL_ERROR("[CollAlgFactory] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult AlgTemplateBase::CalcResDetour(
    const bool forAllReduce, const RankGraph* rankGraph, AlgTempResReq& tempResReq, u32& requiredScratchMultiplier)
{
    (void)forAllReduce;
    (void)tempResReq;
    (void)rankGraph;
    (void)requiredScratchMultiplier;
    HCCL_ERROR("[CollAlgFactory] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult AlgTemplateBase::CalcResDetour(
    const bool forAllReduce, ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq, u32& requiredScratchMultiplier)
{
    (void)forAllReduce;
    (void)linkMgr;
    (void)tempResReq;
    (void)requiredScratchMultiplier;
    HCCL_ERROR("[CollAlgFactory] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult AlgTemplateBase::CalcRes(AlgTempResReq& tempResReq)
{
    (void)tempResReq;
    HCCL_ERROR("[CollAlgFactory] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult AlgTemplateBase::CalcResDetour(const RankGraph* rankGraph, AlgTempResReq& tempResReq)
{
    (void)rankGraph;
    (void)tempResReq;
    HCCL_ERROR("[CollAlgFactory] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult AlgTemplateBase::CalcResDetour(ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq)
{
    (void)linkMgr;
    (void)tempResReq;
    HCCL_ERROR("[CollAlgFactory] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

void AlgTemplateBase::InitReduceInfo(const ReduceOp& redOp, const DataType& dataType)
{
    redOp_ = redOp;
    dataType_ = dataType;

    return;
}

void AlgTemplateBase::SetDataType(const DataType& dataType)
{
    dataType_ = dataType;

    return;
}

HcclResult AlgTemplateBase::PreSync(const u32 queIdx, std::vector<PrimQuePtr>& syncPrimQues) const
{
    PrimQuePtr currPrimQue = syncPrimQues[queIdx];
    if (queIdx == 0) {
        // Semaphore Post
        for (u32 qidx = 1; qidx < syncPrimQues.size(); qidx++) {
            std::unique_ptr<Primitive> primPostTo = std::make_unique<PrimPostTo>(syncPrimQues[qidx]);
            CHK_PTR_NULL(primPostTo);
            currPrimQue->Append(std::move(primPostTo));
        }
    } else {
        // Semaphore Wait
        std::unique_ptr<Primitive> primWaitFrom = std::make_unique<PrimWaitFrom>(syncPrimQues[0]);
        CHK_PTR_NULL(primWaitFrom);
        currPrimQue->Append(std::move(primWaitFrom));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::PostSync(const u32 queIdx, std::vector<PrimQuePtr>& syncPrimQues) const
{
    PrimQuePtr currPrimQue = syncPrimQues[queIdx];
    if (queIdx == 0) {
        // Semaphore Wait
        if (enableCounterNotify_) {
            std::unique_ptr<PrimWaitGroup> primWaitGroup = std::make_unique<PrimWaitGroup>();
            for (u32 qidx = 1; qidx < syncPrimQues.size(); qidx++) {
                primWaitGroup->Append(syncPrimQues[qidx]);
            }
            CHK_PTR_NULL(primWaitGroup);
            currPrimQue->Append(std::move(primWaitGroup));
        } else {
            for (u32 qidx = 1; qidx < syncPrimQues.size(); qidx++) {
                std::unique_ptr<Primitive> primWaitFrom = std::make_unique<PrimWaitFrom>(syncPrimQues[qidx]);
                CHK_PTR_NULL(primWaitFrom);
                currPrimQue->Append(std::move(primWaitFrom));
            }
        }
    } else {
        // Semaphore Post
        if (enableCounterNotify_) {
            std::unique_ptr<Primitive> primPostTo = std::make_unique<PrimPostTo>(syncPrimQues[0], NotifyType::COUNTER);
            CHK_PTR_NULL(primPostTo);
            currPrimQue->Append(std::move(primPostTo));
        } else {
            std::unique_ptr<Primitive> primPostTo = std::make_unique<PrimPostTo>(syncPrimQues[0]);
            CHK_PTR_NULL(primPostTo);
            currPrimQue->Append(std::move(primPostTo));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::PreSyncInterQueues(std::vector<PrimQuePtr>& syncPrimQues) const
{
    for (u32 queIdx = 0; queIdx < syncPrimQues.size(); queIdx++) {
        CHK_PRT_RET(
            PreSync(queIdx, syncPrimQues) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollAlgFactory] Rank [%d], Que [%u], Semaphore Synchronization Failed.", myRank_,
                syncPrimQues[queIdx]->GetId()),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AlgTemplateBase::PostSyncInterQueues(std::vector<PrimQuePtr>& syncPrimQues) const
{
    for (u32 queIdx = 0; queIdx < syncPrimQues.size(); queIdx++) {
        CHK_PRT_RET(
            PostSync(queIdx, syncPrimQues) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollAlgFactory] Rank [%d], Que [%u], Semaphore Synchronization Failed.", myRank_,
                syncPrimQues[queIdx]->GetId()),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
