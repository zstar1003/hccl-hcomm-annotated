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

#include "temp_reduce_scatter_mesh.h"

namespace Hccl {
TempReduceScatterMesh::TempReduceScatterMesh(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

TempReduceScatterMesh::~TempReduceScatterMesh() {}

HcclResult
TempReduceScatterMesh::CalcRes(const bool forAllReduce, AlgTempResReq& tempResReq, u32& requiredScratchMultiplier)
{
    (void)forAllReduce;
    tempResReq.queNum = tempVTopo_[0].size() - 1;
    requiredScratchMultiplier = tempRankSize_;

    CHK_RET(CalcResLinks(tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterMesh::CalcResDetour(
    const bool forAllReduce, const RankGraph* rankGraph, AlgTempResReq& tempResReq, u32& requiredScratchMultiplier)
{
    (void)forAllReduce;

    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));

    linkNumBtwPeers_ = GetLinkNum(rankGraph, myRank_, tempVTopo_[0][(myAlgRank + 1) % tempRankSize_]);
    if (linkNumBtwPeers_ == 1) {
        HCCL_INFO(
            "[CollAlgFactory] [TempReduceScatterMesh] [WARNING] Rank [%d], linkNum between rank [%d] and rank [%d] "
            "equals 1, not able to detour",
            myRank_, myRank_, tempVTopo_[0][(myAlgRank + 1) % tempRankSize_]);
        enableDetour_ = false;
    } else {
        enableDetour_ = true;
    }

    queNumPerNeighbor_ = (linkNumBtwPeers_ + 1) >> 1;
    tempResReq.queNum = (tempVTopo_[0].size() - 1) * queNumPerNeighbor_;

    requiredScratchMultiplier = tempRankSize_;

    CHK_RET(CalcResLinks(tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterMesh::CalcResDetour(
    const bool forAllReduce, ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq, u32& requiredScratchMultiplier)
{
    (void)forAllReduce;

    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));

    linkNumBtwPeers_ = (linkMgr->GetLinks(tempVTopo_[0][(myAlgRank + 1) % tempRankSize_])).size();

    enableDetour_ = (linkNumBtwPeers_ == 1) ? false : true;

    queNumPerNeighbor_ = (linkNumBtwPeers_ + 1) >> 1;
    tempResReq.queNum = (tempVTopo_[0].size() - 1) * queNumPerNeighbor_;

    requiredScratchMultiplier = tempRankSize_;

    CHK_RET(CalcResLinks(tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterMesh::CalcResLinks(AlgTempResReq& tempResReq)
{
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));

    for (u32 queIdx = 0; queIdx < tempVTopo_[0].size() - 1; queIdx++) {
        // find neighbors -> virtualRank
        RankId neighborRank = tempVTopo_[0][(myAlgRank + 1 + queIdx) % tempRankSize_];

        // LinkNum
        tempResReq.links[neighborRank] = linkNumBtwPeers_;
    }

    return HcclResult::HCCL_SUCCESS;
}

/*
dataSize / (rankSize) --> chunkSize
dataSize / (rankSize * queNum) --> sliceSize

SliceInfoVecforRing: [1st chunk: [1st Slice, 2nd Slice, ...], 2nd chunk: [1st Slice, 2nd Slice, ...], ...]
*/
HcclResult TempReduceScatterMesh::CalcSliceInfo(
    const AllignInfo& allignInfo, const bool forAllReduce, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(1);
    sliceInfoVec.resize(tempRankSize_, tmp);

    if (forAllReduce) {
        // for allreduce, dataSize = total dataSize
        CHK_RET(CalcSliceInfoAllReduce(allignInfo, dataSize, sliceInfoVec));
    } else {
        // for reduce scatter, dataSize = chunkSize
        CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterMesh::CalcSliceInfoAllReduce(
    const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec) const
{
    u64 unitAlignSize;
    CHK_RET(GetUnitAllignSize(allignInfo, unitAlignSize));

    u64 chunkSize = RoundUp(dataSize, (tempRankSize_ * unitAlignSize)) * unitAlignSize;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
        u64 currChunkSize = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice = {accumOff, currChunkSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize_ - 1][0].offset + sliceInfoVec[tempRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR("[CollAlgFactory] [TempReduceScatterMesh] Rank [%d], SliceInfo calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterMesh::GenPrimQue(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<PrimQuePtr>& tempPrimQues)
{
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    buffInfo_ = buffInfo;

    auto linkIter = tempLinks.begin();
    linkNumBtwPeers_ = linkIter->second.size();
    HCCL_INFO(
        "[CollAlgFactory] [TempReduceScatterMesh] Rank [%d], linkNumBtwPeers equals to [%u].", myRank_,
        linkNumBtwPeers_);
    queNumPerNeighbor_ = (linkNumBtwPeers_ + 1) >> 1;
    HCCL_INFO(
        "[CollAlgFactory] [TempReduceScatterMesh] Rank [%d], queNumPerNeighbor equals to [%u].", myRank_,
        queNumPerNeighbor_);
    enableDetour_ = (linkNumBtwPeers_ == 1) ? false : true;

    majorQueNum_ = tempVTopo_[0].size() - 1;
    CHK_PRT_RET(
        majorQueNum_ * queNumPerNeighbor_ != tempPrimQues.size(),
        HCCL_ERROR(
            "[CollAlgFactory] [TempReduceScatterMesh] Rank [%d], requiredQueNum [%u] not equals to "
            "templateQueNum [%u].",
            myRank_, majorQueNum_ * queNumPerNeighbor_, tempPrimQues.size()),
        HcclResult::HCCL_E_INTERNAL);

    // queue arrangement
    std::vector<PrimQuePtr> mainPrimQues;
    for (u32 queIdx = 0; queIdx < majorQueNum_; queIdx++) {
        mainPrimQues.push_back(tempPrimQues[queIdx * queNumPerNeighbor_]);
    }

    // LocalCopy: from input to scratch In Buffer for OPBASE
    if ((opMode_ == OpMode::OPBASE) && tempFuncs.isForepart) {
        CHK_RET(PreCopyOpbase(tempFuncs.usrData, mainPrimQues));
    }

    // semaphore sync
    if (majorQueNum_ > 1) {
        CHK_RET(PreSyncInterQueues(mainPrimQues));
    }

    // locate myRank in tempVTopo -> algRank
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));

    // run Mesh
    CHK_PRT_RET(
        RunMesh(myAlgRank, tempVTopo_[0], sliceInfoVec, tempLinks, tempPrimQues) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [TempReduceScatterMesh] Rank [%d], unable to run the mesh algorithm.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // semaphore sync
    if (majorQueNum_ > 1) {
        CHK_RET(PostSyncInterQueues(mainPrimQues));
    }

    // LocalCopy for standalone reducescatter in Offload Mode
    if ((opMode_ == OpMode::OFFLOAD) && !tempFuncs.forAllReduce && !tempFuncs.forAlgSeqComb) {
        CHK_RET(PostCopyOffload(sliceInfoVec, mainPrimQues));
    }

    // LocalCopy from scratch to output for Opbase
    if (tempFuncs.isBottom && !tempFuncs.forAllReduce) {
        CHK_RET(PostCopyOpbase(tempFuncs.usrData, mainPrimQues));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterMesh::RunMesh(
    const u32 myAlgRank, const std::vector<RankId>& vTopo, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
    std::vector<PrimQuePtr>& tempPrimQues)
{
    for (u32 queIdx = 0; queIdx < vTopo.size() - 1; queIdx++) {
        // find neighbors -> virtualRank
        RankId neighborRank = vTopo[(myAlgRank + 1 + queIdx) % tempRankSize_];

        u32 recvChunkIdx = tempVirtRankMap_[myRank_];
        u32 sendChunkIdx = tempVirtRankMap_[neighborRank];

        // queue assignment
        if (enableDetour_) {
            std::vector<PrimQuePtr> detourPrimQues;
            for (u32 detourIdx = 0; detourIdx < queNumPerNeighbor_; detourIdx++) {
                detourPrimQues.push_back(tempPrimQues[queIdx * queNumPerNeighbor_ + detourIdx]);
            }
            HCCL_INFO("[CollAlgFactory] [TempReduceScatterMesh] Rank [%d], Run Mesh with Detour.", myRank_);
            CHK_RET(RunIndividualPeerDetour(
                neighborRank, sliceInfoVec[sendChunkIdx][0], sliceInfoVec[recvChunkIdx][0], tempLinks, detourPrimQues));
        } else {
            PrimQuePtr currQue = tempPrimQues[queIdx];
            LinkData neighborLinkData = tempLinks.at(neighborRank)[0];
            HCCL_INFO("[CollAlgFactory] [TempReduceScatterMesh] Rank [%d], Run Mesh without Detour.", myRank_);
            CHK_RET(RunIndividualPeer(
                neighborRank, neighborLinkData, sliceInfoVec[sendChunkIdx][0], sliceInfoVec[recvChunkIdx][0], currQue));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterMesh::RunIndividualPeerDetour(
    const RankId neighborRank, const SliceInfo& sendReduceSlice, const SliceInfo& recvReduceSlice,
    const ResLinks& tempLinks, std::vector<PrimQuePtr>& detourPrimQues)
{
    CHK_RET(PreSyncInterQueues(detourPrimQues));

    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    u64 unitRecvReduceSize = RoundUp(recvReduceSlice.size, queNumPerNeighbor_ * dataSizePerVolume) * dataSizePerVolume;
    u64 resRecvReduceSize = recvReduceSlice.size;
    u64 currRecvReduceOff = recvReduceSlice.offset;

    u64 unitSendReduceSize = RoundUp(sendReduceSlice.size, queNumPerNeighbor_ * dataSizePerVolume) * dataSizePerVolume;
    u64 resSendReduceSize = sendReduceSlice.size;
    u64 currSendReduceOff = sendReduceSlice.offset;

    std::vector<std::vector<LinkDataIterator>> sendRecvRedLinks;
    CHK_RET(GetSendRecvRedLinks(neighborRank, tempLinks, sendRecvRedLinks));

    for (u32 detourIdx = 0; detourIdx < queNumPerNeighbor_; detourIdx++) {
        u64 currSendReduceSize = resSendReduceSize > unitSendReduceSize ? unitSendReduceSize : resSendReduceSize;
        u64 currRecvReduceSize = resRecvReduceSize > unitRecvReduceSize ? unitRecvReduceSize : resRecvReduceSize;
        SliceInfo currSendReduceSlice = {currSendReduceOff, currSendReduceSize};
        SliceInfo currRecvReduceSlice = {currRecvReduceOff, currRecvReduceSize};

        std::unique_ptr<PrimGroup> primGroup = RunSendRecvReduce(
            neighborRank, (*sendRecvRedLinks[detourIdx][0]), (*sendRecvRedLinks[detourIdx][1]), currSendReduceSlice,
            currRecvReduceSlice);
        detourPrimQues[detourIdx]->Append(std::move(primGroup));

        resRecvReduceSize -= currRecvReduceSize;
        resSendReduceSize -= currSendReduceSize;
        currRecvReduceOff += currRecvReduceSize;
        currSendReduceOff += currSendReduceSize;
    }

    CHK_RET(PostSyncInterQueues(detourPrimQues));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterMesh::GetSendRecvRedLinks(
    const RankId neighborRank, const ResLinks& tempLinks,
    std::vector<std::vector<LinkDataIterator>>& sendRecvLinks) const
{
    CHK_PRT_RET(
        ((queNumPerNeighbor_ != NUM_TWO) || (tempRankSize_ != NUM_TWO)),
        HCCL_ERROR(
            "[CollAlgFactory] [TempReduceScatterMesh] Rank [%d], detouring is supported only in 2P Mesh in 4P topo.",
            myRank_),
        HcclResult::HCCL_E_INTERNAL);

    std::vector<LinkDataIterator> tmpLinks(NUM_TWO);
    sendRecvLinks.resize(queNumPerNeighbor_, tmpLinks);

    CHK_PRT_RET(
        GetDetourSendRecvLinksIn4P(myRank_, neighborRank, tempLinks, sendRecvLinks),
        HCCL_ERROR(
            "[InsCollAlgFactory] [TempReduceScatterMesh] Rank [%d], get send recv links in 2P Mesh in 4P topo.",
            myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterMesh::RunIndividualPeer(
    const RankId neighborRank, const LinkData& neighborLinkData, const SliceInfo& sendReduceSlice,
    const SliceInfo& recvReduceSlice, PrimQuePtr currQue) const
{
    std::unique_ptr<PrimGroup> primGroup
        = RunSendRecvReduce(neighborRank, neighborLinkData, neighborLinkData, sendReduceSlice, recvReduceSlice);
    currQue->Append(std::move(primGroup));
    return HcclResult::HCCL_SUCCESS;
}

std::unique_ptr<PrimGroup> TempReduceScatterMesh::RunSendRecvReduce(
    const RankId neighborRank, const LinkData& sendLinkData, const LinkData& recvLinkData,
    const SliceInfo& currSendReduceSlice, const SliceInfo& currRecvReduceSlice) const
{
    // PrimGroup
    std::unique_ptr<PrimGroup> primGroup = std::make_unique<PrimGroup>();

    // RecvReduce
    u64 recvOffset = currRecvReduceSlice.offset;
    u64 recvSize = currRecvReduceSlice.size;
    DataSlice recvRemSlice = DataSlice(buffInfo_.inBuffType, recvOffset + buffInfo_.inBuffBaseOff, recvSize);
    DataSlice recvLocSrcSlice = DataSlice(buffInfo_.scratBuffType, recvOffset + buffInfo_.scratchBuffBaseOff, recvSize);
    DataSlice recvLocDstSlice = DataSlice(buffInfo_.inBuffType, recvOffset + buffInfo_.inBuffBaseOff, recvSize);
    std::unique_ptr<Primitive> primRecvReduce = std::make_unique<PrimRecvReduce>(
        neighborRank, recvLinkData, recvRemSlice, recvLocSrcSlice, recvLocDstSlice, dataType_, redOp_, dmaMode_);

    primGroup->Append(std::move(primRecvReduce));

    // SendReduce
    u64 sendOffset = currSendReduceSlice.offset;
    u64 sendSize = currSendReduceSlice.size;
    DataSlice sendLocSlice = DataSlice(buffInfo_.inBuffType, sendOffset + buffInfo_.inBuffBaseOff, sendSize);
    DataSlice sendRemSrcSlice = DataSlice(buffInfo_.scratBuffType, sendOffset + buffInfo_.scratchBuffBaseOff, sendSize);
    DataSlice sendRemDstSlice = DataSlice(buffInfo_.inBuffType, sendOffset + buffInfo_.inBuffBaseOff, sendSize);
    std::unique_ptr<Primitive> primSendReduce = std::make_unique<PrimSendReduce>(
        neighborRank, sendLinkData, sendLocSlice, sendRemSrcSlice, sendRemDstSlice, dataType_, redOp_, dmaMode_);

    primGroup->Append(std::move(primSendReduce));

    return primGroup;
}

HcclResult
TempReduceScatterMesh::PostCopyOffload(const RankSliceInfo& sliceInfoVec, std::vector<PrimQuePtr>& tempPrimQues)
{
    u64 srcOffset = sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset;
    u64 srcSize = sliceInfoVec[tempVirtRankMap_[myRank_]][0].size;
    u64 dstOffset = 0;

    DataSlice srcSlice = DataSlice(buffInfo_.inBuffType, srcOffset + buffInfo_.inBuffBaseOff, srcSize);
    DataSlice dstSlice = DataSlice(buffInfo_.outBuffType, dstOffset + buffInfo_.outBuffBaseOff, srcSize);
    std::unique_ptr<Primitive> primLocalCopy = std::make_unique<PrimLocalCopy>(srcSlice, dstSlice);
    tempPrimQues[0]->Append(std::move(primLocalCopy));

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
