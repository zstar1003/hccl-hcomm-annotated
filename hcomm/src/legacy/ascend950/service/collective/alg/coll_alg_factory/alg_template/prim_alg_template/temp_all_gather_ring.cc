/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"

#include "temp_all_gather_ring.h"

namespace Hccl {
TempAllGatherRing::TempAllGatherRing(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

TempAllGatherRing::~TempAllGatherRing() {}

HcclResult TempAllGatherRing::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = tempVTopo_.size();

    CHK_PRT_RET(
        CalcResLinksRing(myRank_, tempRankSize_, tempVTopo_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [TempAllGatherRing] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
TempAllGatherRing::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);
    // for reduce scatter, dataSize = chunkSize
    CHK_RET(CalcRsAgSliceInfoRing(myRank_, tempVTopo_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempAllGatherRing::GenPrimQue(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<PrimQuePtr>& tempPrimQues)
{
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    buffInfo_ = buffInfo;

    queNum_ = tempVTopo_.size();
    CHK_PRT_RET(
        queNum_ != tempPrimQues.size(),
        HCCL_ERROR("[CollAlgFactory] [TempAllGatherRing] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // Local Copy from Input to Scratch Buffer for OPBASE
    if ((opMode_ == OpMode::OPBASE) && tempFuncs.isForepart && !tempFuncs.forAllReduce) {
        CHK_RET(PreCopyOpbase(tempFuncs.usrData, tempPrimQues));
    }

    // Local Copy from Input to Output Buffer for OFFLOAD
    if ((opMode_ == OpMode::OFFLOAD) && (!tempFuncs.forAlgSeqComb)) {
        CHK_RET(PreCopyOffload(sliceInfoVec, tempFuncs.forAllReduce, tempPrimQues));
    }

    stepNum_ = tempRankSize_ - 1;
    for (u32 queIdx = 0; queIdx < tempVTopo_.size(); queIdx++) {
        // semaphore sync for standAlone AllReduce
        if (!tempFuncs.forAllReduce && (queNum_ > 1)) {
            CHK_PRT_RET(
                PreSync(queIdx, tempPrimQues) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollAlgFactory] [TempAllGatherRing] Rank [%d], Que [%u], Semaphore Synchronization Failed.",
                    myRank_, tempPrimQues[queIdx]->GetId()),
                HcclResult::HCCL_E_INTERNAL);
        }

        PrimQuePtr currPrimQue = tempPrimQues[queIdx];
        CHK_RET(RunIndividualRing(queIdx, sliceInfoVec, tempLinks, currPrimQue));

        // semaphore sync
        if (queNum_ > 1) {
            CHK_PRT_RET(
                PostSync(queIdx, tempPrimQues) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollAlgFactory] [TempAllGatherRing] Rank [%d], Unable to synchronize all queues.", myRank_),
                HcclResult::HCCL_E_INTERNAL);
        }
    }

    // LocalCopy: from scratch to output for opbase
    if ((opMode_ == OpMode::OPBASE) && tempFuncs.isBottom) {
        CHK_RET(PostCopyOpbase(tempFuncs.usrData, tempPrimQues));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempAllGatherRing::PreCopyOffload(
    const RankSliceInfo& sliceInfoVec, const bool forAllReduce, std::vector<PrimQuePtr>& tempPrimQues)
{
    if (!forAllReduce) {
        u64 srcOffset = 0;
        u64 srcSize = sliceInfoVec[tempVirtRankMap_[myRank_]][queNum_ - 1].offset
                      - sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset
                      + sliceInfoVec[tempVirtRankMap_[myRank_]][queNum_ - 1].size;
        u64 dstOffset = sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset;
        DataSlice srcSlice = DataSlice(buffInfo_.inBuffType, srcOffset + buffInfo_.inBuffBaseOff, srcSize);
        DataSlice dstSlice = DataSlice(buffInfo_.outBuffType, dstOffset + buffInfo_.outBuffBaseOff, srcSize);
        std::unique_ptr<Primitive> primLocalCopy = std::make_unique<PrimLocalCopy>(srcSlice, dstSlice);
        tempPrimQues[0]->Append(std::move(primLocalCopy));
    } else { // allreduce: split local copy per queue to eliminate the notify/wait
        for (u32 qIdx = 0; qIdx < queNum_; qIdx++) {
            u64 queSliceOff = sliceInfoVec[tempVirtRankMap_[myRank_]][qIdx].offset;
            u64 queSliceSize = sliceInfoVec[tempVirtRankMap_[myRank_]][qIdx].size;
            DataSlice srcSlice = DataSlice(buffInfo_.inBuffType, queSliceOff + buffInfo_.inBuffBaseOff, queSliceSize);
            DataSlice dstSlice = DataSlice(buffInfo_.outBuffType, queSliceOff + buffInfo_.outBuffBaseOff, queSliceSize);
            std::unique_ptr<Primitive> primLocalCopy = std::make_unique<PrimLocalCopy>(srcSlice, dstSlice);
            tempPrimQues[qIdx]->Append(std::move(primLocalCopy));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempAllGatherRing::RunIndividualRing(
    const u32 queIdx, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, PrimQuePtr currPrimQue)
{
    // locate myRank in tempVTopo -> algRank
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[queIdx], myAlgRank));

    // find neighbors -> virtualRank
    RankId sendToRank = tempVTopo_[queIdx][(myAlgRank + 1) % tempRankSize_];
    RankId recvFromRank = tempVTopo_[queIdx][(myAlgRank - 1 + tempRankSize_) % tempRankSize_]; // virtualRank

    // Link
    LinkData sendLinkData = tempLinks.at(sendToRank)[0];
    LinkData recvLinkData = tempLinks.at(recvFromRank)[0];

    // run stepNum steps to complete the ring
    for (u32 step = 0; step < stepNum_; step++) {
        u32 sendChunkIdx = tempVirtRankMap_[tempVTopo_[queIdx][(myAlgRank - step + tempRankSize_) % tempRankSize_]];
        u64 sendOffset = sliceInfoVec[sendChunkIdx][queIdx].offset;
        u64 sendSize = sliceInfoVec[sendChunkIdx][queIdx].size;
        u32 recvChunkIdx = tempVirtRankMap_[tempVTopo_[queIdx][(myAlgRank - 1 - step + tempRankSize_) % tempRankSize_]];

        u64 recvOffset = sliceInfoVec[recvChunkIdx][queIdx].offset;
        u64 recvSize = sliceInfoVec[recvChunkIdx][queIdx].size;

        // PrimGroup
        std::unique_ptr<PrimGroup> primGroup = std::make_unique<PrimGroup>();

        // Send
        DataSlice sendLocSlice = DataSlice(buffInfo_.outBuffType, sendOffset + buffInfo_.outBuffBaseOff, sendSize);
        DataSlice sendRemSlice = DataSlice(buffInfo_.outBuffType, sendOffset + buffInfo_.outBuffBaseOff, sendSize);
        std::unique_ptr<Primitive> primSend
            = std::make_unique<PrimSend>(sendToRank, sendLinkData, sendLocSlice, sendRemSlice, dmaMode_);

        primGroup->Append(std::move(primSend));

        // Recv
        DataSlice recvRemSlice = DataSlice(buffInfo_.outBuffType, recvOffset + buffInfo_.outBuffBaseOff, recvSize);
        DataSlice recvLocSlice = DataSlice(buffInfo_.outBuffType, recvOffset + buffInfo_.outBuffBaseOff, recvSize);
        std::unique_ptr<Primitive> primRecv
            = std::make_unique<PrimRecv>(recvFromRank, recvLinkData, recvLocSlice, recvRemSlice, dmaMode_);

        primGroup->Append(std::move(primRecv));

        currPrimQue->Append(std::move(primGroup));
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
