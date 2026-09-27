/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>

#include "log.h"

#include "temp_reduce_scatter_ring.h"

namespace Hccl {
TempReduceScatterRing::TempReduceScatterRing(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

TempReduceScatterRing::~TempReduceScatterRing() {}

HcclResult
TempReduceScatterRing::CalcRes(const bool forAllReduce, AlgTempResReq& tempResReq, u32& requiredScratchMultiplier)
{
    tempResReq.queNum = tempVTopo_.size();
    requiredScratchMultiplier = forAllReduce ? tempRankSize_ : 0;

    CHK_PRT_RET(
        CalcResLinksRing(myRank_, tempRankSize_, tempVTopo_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [TempReduceScatterRing] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

/*
dataSize / (rankSize) --> chunkSize
dataSize / (rankSize * queNum) --> sliceSize

SliceInfoVecforRing: [1st chunk: [1st Slice, 2nd Slice, ...], 2nd chunk: [1st Slice, 2nd Slice, ...], ...]
*/
HcclResult TempReduceScatterRing::CalcSliceInfo(
    const AllignInfo& allignInfo, const bool forAllReduce, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    if (forAllReduce) {
        // for allreduce, dataSize = total dataSize
        CHK_RET(CalcSliceInfoAllReduce(allignInfo, dataSize, sliceInfoVec));
    } else {
        // for reduce scatter, dataSize = chunkSize
        CHK_RET(CalcRsAgSliceInfoRing(myRank_, tempVTopo_, allignInfo, dataSize, sliceInfoVec));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterRing::CalcSliceInfoAllReduce(
    const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    u32 queNum = tempVTopo_.size();
    u64 unitAlignSize;
    CHK_RET(GetUnitAllignSize(allignInfo, unitAlignSize));

    u64 queDataSize = RoundUp(dataSize, (queNum * unitAlignSize)) * unitAlignSize;

    u64 resDataSize = dataSize;
    std::vector<u64> resQueData;
    std::vector<u64> queChunkSize;
    for (u32 queIdx = 0; queIdx < queNum; queIdx++) {
        // split data on queues
        u64 currQueDataSize = (resDataSize > queDataSize) ? queDataSize : resDataSize;
        resQueData.push_back(currQueDataSize);
        resDataSize -= currQueDataSize;

        // support ReduceScatterV and AllGatherV for better data alignment when enable Data Align
        u64 currQueChunkSize = RoundUp(currQueDataSize, (tempRankSize_ * unitAlignSize)) * unitAlignSize;
        queChunkSize.push_back(currQueChunkSize);
    }
    CHK_PRT_RET(
        resDataSize != 0,
        HCCL_ERROR("[CollAlgFactory] [TempReduceScatterRing] Rank [%d], SliceInfo calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
        for (u32 queIdx = 0; queIdx < queNum; queIdx++) {
            u64 currSliceSize = (resQueData[queIdx] > queChunkSize[queIdx]) ? queChunkSize[queIdx] : resQueData[queIdx];
            SliceInfo currSlice = {accumOff, currSliceSize};
            resQueData[queIdx] -= currSliceSize;
            accumOff += currSliceSize;
            sliceInfoVec[rankIdx][queIdx] = currSlice;
        }
    }

    CHK_PRT_RET(
        ((sliceInfoVec[tempRankSize_ - 1][queNum - 1].offset + sliceInfoVec[tempRankSize_ - 1][queNum - 1].size
          != dataSize)
         || (accumulate(resQueData.begin(), resQueData.end(), 0) != 0)),
        HCCL_ERROR("[CollAlgFactory] [TempReduceScatterRing] Rank [%d], SliceInfo calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterRing::GenPrimQue(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<PrimQuePtr>& tempPrimQues)
{
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    buffInfo_ = buffInfo;

    queNum_ = tempVTopo_.size();
    CHK_PRT_RET(
        queNum_ != tempPrimQues.size(),
        HCCL_ERROR("[CollAlgFactory] [TempReduceScatterRing] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // LocalCopy: from input to scratch In Buffer for OPBASE
    if (tempFuncs.isForepart) {
        CHK_RET(PreCopyOpbase(tempFuncs.usrData, tempPrimQues));
    }

    stepNum_ = tempRankSize_ - 1;
    for (u32 queIdx = 0; queIdx < tempVTopo_.size(); queIdx++) {
        // semaphore sync
        if (queNum_ > 1) {
            CHK_PRT_RET(
                PreSync(queIdx, tempPrimQues) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollAlgFactory] [TempReduceScatterRing] Rank [%d], Unable to synchronize all queues.", myRank_),
                HcclResult::HCCL_E_INTERNAL);
        }

        PrimQuePtr currPrimQue = tempPrimQues[queIdx];
        CHK_RET(RunIndividualRing(queIdx, tempFuncs.forAllReduce, sliceInfoVec, tempLinks, currPrimQue));

        if ((!tempFuncs.forAllReduce) && (queNum_ > 1)) {
            // semaphore sync for standalone reducescatter
            CHK_PRT_RET(
                PostSync(queIdx, tempPrimQues) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollAlgFactory] [TempReduceScatterRing] Rank [%d], Unable to synchronize all queues.", myRank_),
                HcclResult::HCCL_E_INTERNAL);
        }
    }

    // LocalCopy for standalone reducescatter in Offload Mode
    if ((opMode_ == OpMode::OFFLOAD) && !tempFuncs.forAllReduce && !tempFuncs.forAlgSeqComb) {
        CHK_RET(PostCopyOffload(sliceInfoVec, tempPrimQues));
    }

    // LocalCopy from scratch to output for Opbase
    if (tempFuncs.isBottom && !tempFuncs.forAllReduce) {
        CHK_RET(PostCopyOpbase(tempFuncs.usrData, tempPrimQues));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterRing::RunIndividualRing(
    const u32 queIdx, const bool& forAllReduce, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
    PrimQuePtr currPrimQue)
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
        u32 sendChunkIdx = tempVirtRankMap_[tempVTopo_[queIdx][(myAlgRank - 1 - step + tempRankSize_) % tempRankSize_]];
        u64 sendOffset = sliceInfoVec[sendChunkIdx][queIdx].offset;
        u64 sendSize = sliceInfoVec[sendChunkIdx][queIdx].size;
        u64 tmpScratchSendOff = forAllReduce ? sendOffset : (sendOffset - sliceInfoVec[sendChunkIdx][0].offset);

        u32 recvChunkIdx = tempVirtRankMap_[tempVTopo_[queIdx][(myAlgRank - 2 - step + tempRankSize_) % tempRankSize_]];
        u64 recvOffset = sliceInfoVec[recvChunkIdx][queIdx].offset;
        u64 recvSize = sliceInfoVec[recvChunkIdx][queIdx].size;
        u64 tmpScratchRecvOff = forAllReduce ? recvOffset : (recvOffset - sliceInfoVec[recvChunkIdx][0].offset);

        // PrimGroup
        std::unique_ptr<PrimGroup> primGroup = std::make_unique<PrimGroup>();

        // SendReduce
        DataSlice sendLocSlice = DataSlice(buffInfo_.inBuffType, sendOffset + buffInfo_.inBuffBaseOff, sendSize);
        DataSlice sendRemSrcSlice
            = DataSlice(buffInfo_.outBuffType, tmpScratchSendOff + buffInfo_.outBuffBaseOff, sendSize);
        DataSlice sendRemDstSlice = DataSlice(buffInfo_.inBuffType, sendOffset + buffInfo_.inBuffBaseOff, sendSize);
        std::unique_ptr<Primitive> primSendReduce = std::make_unique<PrimSendReduce>(
            sendToRank, sendLinkData, sendLocSlice, sendRemSrcSlice, sendRemDstSlice, dataType_, redOp_, dmaMode_);

        primGroup->Append(std::move(primSendReduce));

        // RecvReduce
        DataSlice recvRemSlice = DataSlice(buffInfo_.inBuffType, recvOffset + buffInfo_.inBuffBaseOff, recvSize);
        DataSlice recvLocSrcSlice
            = DataSlice(buffInfo_.outBuffType, tmpScratchRecvOff + buffInfo_.outBuffBaseOff, recvSize);
        DataSlice recvLocDstSlice = DataSlice(buffInfo_.inBuffType, recvOffset + buffInfo_.inBuffBaseOff, recvSize);
        std::unique_ptr<Primitive> primRecvReduce = std::make_unique<PrimRecvReduce>(
            recvFromRank, recvLinkData, recvRemSlice, recvLocSrcSlice, recvLocDstSlice, dataType_, redOp_, dmaMode_);

        primGroup->Append(std::move(primRecvReduce));

        currPrimQue->Append(std::move(primGroup));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
TempReduceScatterRing::PostCopyOffload(const RankSliceInfo& sliceInfoVec, std::vector<PrimQuePtr>& tempPrimQues)
{
    u64 srcOffset = sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset;
    u64 srcSize = sliceInfoVec[tempVirtRankMap_[myRank_]][queNum_ - 1].offset
                  - sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset
                  + sliceInfoVec[tempVirtRankMap_[myRank_]][queNum_ - 1].size;
    u64 dstOffset = 0;
    DataSlice srcSlice = DataSlice(buffInfo_.inBuffType, srcOffset + buffInfo_.inBuffBaseOff, srcSize);
    DataSlice dstSlice = DataSlice(buffInfo_.outBuffType, dstOffset + buffInfo_.outBuffBaseOff, srcSize);
    std::unique_ptr<Primitive> primLocalCopy = std::make_unique<PrimLocalCopy>(srcSlice, dstSlice);
    tempPrimQues[0]->Append(std::move(primLocalCopy));

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
