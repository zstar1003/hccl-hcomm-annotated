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

#include "temp_all_gather_mesh.h"

namespace Hccl {
TempAllGatherMesh::TempAllGatherMesh(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

TempAllGatherMesh::~TempAllGatherMesh() {}

HcclResult TempAllGatherMesh::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = tempVTopo_[0].size() - 1;

    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempAllGatherMesh::CalcResDetour(const RankGraph* rankGraph, AlgTempResReq& tempResReq)
{
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));

    linkNumBtwPeers_ = GetLinkNum(rankGraph, myRank_, tempVTopo_[0][(myAlgRank + 1) % tempRankSize_]);
    if (linkNumBtwPeers_ == 1) {
        HCCL_INFO(
            "[CollAlgFactory] [TempAllGatherMesh] [WARNING] Rank [%d], linkNum between rank [%d] and rank [%d] "
            "equals 1, not able to detour",
            myRank_, myRank_, tempVTopo_[0][(myAlgRank + 1) % tempRankSize_]);
        enableDetour_ = false;
    } else {
        enableDetour_ = true;
    }

    queNumPerNeighbor_ = (linkNumBtwPeers_ + 1) >> 1;
    tempResReq.queNum = (tempVTopo_[0].size() - 1) * queNumPerNeighbor_;

    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempAllGatherMesh::CalcResDetour(ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq)
{
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[0], myAlgRank));

    linkNumBtwPeers_ = (linkMgr->GetLinks(tempVTopo_[0][(myAlgRank + 1) % tempRankSize_])).size();

    enableDetour_ = (linkNumBtwPeers_ == 1) ? false : true;

    queNumPerNeighbor_ = (linkNumBtwPeers_ + 1) >> 1;
    tempResReq.queNum = (tempVTopo_[0].size() - 1) * queNumPerNeighbor_;

    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

/*
dataSize / (rankSize) --> chunkSize
dataSize / (rankSize * dimNum) --> sliceSize

SliceInfoVecforConcurrMesh: [1st chunk: [1st Slice, 2nd Slice], 2nd chunk: [1st Slice, 2nd Slice], ...]
*/
HcclResult
TempAllGatherMesh::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(1);
    sliceInfoVec.resize(tempRankSize_, tmp);

    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempAllGatherMesh::GenPrimQue(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<PrimQuePtr>& tempPrimQues)
{
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    buffInfo_ = buffInfo;

    auto linkIter = tempLinks.begin();
    linkNumBtwPeers_ = linkIter->second.size();
    HCCL_INFO(
        "[CollAlgFactory] [TempAllGatherMesh] Rank [%d], linkNumBtwPeers equals to [%u].", myRank_, linkNumBtwPeers_);
    queNumPerNeighbor_ = (linkNumBtwPeers_ + 1) >> 1;
    HCCL_INFO(
        "[CollAlgFactory] [TempAllGatherMesh] Rank [%d], queNumPerNeighbor equals to [%u].", myRank_,
        queNumPerNeighbor_);
    enableDetour_ = (linkNumBtwPeers_ == 1) ? false : true;

    majorQueNum_ = tempVTopo_[0].size() - 1;
    CHK_PRT_RET(
        majorQueNum_ * queNumPerNeighbor_ != tempPrimQues.size(),
        HCCL_ERROR(
            "[CollAlgFactory] [TempAllGatherMesh] Rank [%d], requiredQueNum [%u] not equals to templateQueNum [%u].",
            myRank_, majorQueNum_ * queNumPerNeighbor_, tempPrimQues.size()),
        HcclResult::HCCL_E_INTERNAL);

    // queue arrangement
    std::vector<PrimQuePtr> mainPrimQues;
    for (u32 queIdx = 0; queIdx < majorQueNum_; queIdx++) {
        mainPrimQues.push_back(tempPrimQues[queIdx * queNumPerNeighbor_]);
    }

    // Local Copy from Input to Scratch Buffer for OPBASE
    if ((opMode_ == OpMode::OPBASE) && tempFuncs.isForepart && !tempFuncs.forAllReduce) {
        CHK_RET(PreCopyOpbase(tempFuncs.usrData, mainPrimQues));
    }

    // Local Copy from Input to Output Buffer for OFFLOAD
    if ((opMode_ == OpMode::OFFLOAD) && (!tempFuncs.forAlgSeqComb)) {
        CHK_RET(PreCopyOffload(sliceInfoVec, tempFuncs.forAllReduce, mainPrimQues));
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
        HCCL_ERROR("[CollAlgFactory] [TempAllGatherMesh] Rank [%d], unable to run the mesh algorithm.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // semaphore sync
    if (majorQueNum_ > 1) {
        CHK_RET(PostSyncInterQueues(mainPrimQues));
    }

    // LocalCopy: from scratch to output for opbase
    if ((opMode_ == OpMode::OPBASE) && tempFuncs.isBottom) {
        CHK_RET(PostCopyOpbase(tempFuncs.usrData, mainPrimQues));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempAllGatherMesh::RunMesh(
    const u32 myAlgRank, const std::vector<RankId>& vTopo, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
    std::vector<PrimQuePtr>& tempPrimQues)
{
    for (u32 queIdx = 0; queIdx < vTopo.size() - 1; queIdx++) {
        // find neighbors -> virtualRank
        RankId neighborRank = vTopo[(myAlgRank + 1 + queIdx) % tempRankSize_];

        u32 recvChunkIdx = tempVirtRankMap_[neighborRank];
        u32 sendChunkIdx = tempVirtRankMap_[myRank_];

        // queue assignment
        if (enableDetour_) {
            std::vector<PrimQuePtr> detourPrimQues;
            for (u32 detourIdx = 0; detourIdx < queNumPerNeighbor_; detourIdx++) {
                detourPrimQues.push_back(tempPrimQues[queIdx * queNumPerNeighbor_ + detourIdx]);
            }

            CHK_RET(RunIndividualPeerDetour(
                neighborRank, sliceInfoVec[sendChunkIdx][0], sliceInfoVec[recvChunkIdx][0], tempLinks, detourPrimQues));
        } else {
            PrimQuePtr currQue = tempPrimQues[queIdx];
            LinkData neighborLinkData = tempLinks.at(neighborRank)[0];
            CHK_RET(RunIndividualPeer(
                neighborRank, neighborLinkData, sliceInfoVec[sendChunkIdx][0], sliceInfoVec[recvChunkIdx][0], currQue));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempAllGatherMesh::RunIndividualPeerDetour(
    const RankId neighborRank, const SliceInfo& sendSlice, const SliceInfo& recvSlice, const ResLinks& tempLinks,
    std::vector<PrimQuePtr>& detourPrimQues)
{
    CHK_RET(PreSyncInterQueues(detourPrimQues));

    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    u64 unitRecvSize = RoundUp(recvSlice.size, queNumPerNeighbor_ * dataSizePerVolume) * dataSizePerVolume;
    u64 resRecvSize = recvSlice.size;
    u64 currRecvOff = recvSlice.offset + buffInfo_.outBuffBaseOff;

    u64 unitSendSize = RoundUp(sendSlice.size, queNumPerNeighbor_ * dataSizePerVolume) * dataSizePerVolume;
    u64 resSendSize = sendSlice.size;
    u64 currSendOff = sendSlice.offset + buffInfo_.outBuffBaseOff;

    std::vector<std::vector<LinkDataIterator>> sendRecvLinks;
    CHK_RET(GetSendRecvLinks(neighborRank, tempLinks, sendRecvLinks));

    for (u32 detourIdx = 0; detourIdx < queNumPerNeighbor_; detourIdx++) {
        u64 currRecvSize = resRecvSize > unitRecvSize ? unitRecvSize : resRecvSize;
        u64 currSendSize = resSendSize > unitSendSize ? unitSendSize : resSendSize;
        SliceInfo currSendSlice = {currSendOff, currSendSize};
        SliceInfo currRecvSlice = {currRecvOff, currRecvSize};

        std::unique_ptr<PrimGroup> primGroup = RunSendRecv(
            neighborRank, (*sendRecvLinks[detourIdx][0]), (*sendRecvLinks[detourIdx][1]), currSendSlice, currRecvSlice);

        detourPrimQues[detourIdx]->Append(std::move(primGroup));

        resRecvSize -= currRecvSize;
        resSendSize -= currSendSize;
        currRecvOff += currRecvSize;
        currSendOff += currSendSize;
    }

    CHK_RET(PostSyncInterQueues(detourPrimQues));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempAllGatherMesh::GetSendRecvLinks(
    const RankId neighborRank, const ResLinks& tempLinks,
    std::vector<std::vector<LinkDataIterator>>& sendRecvLinks) const
{
    CHK_PRT_RET(
        ((queNumPerNeighbor_ != NUM_TWO) || (tempRankSize_ != NUM_TWO)),
        HCCL_ERROR(
            "[CollAlgFactory] [TempAllGatherMesh] Rank [%d], detouring is supported only in 2P Mesh in 4P topo.",
            myRank_),
        HcclResult::HCCL_E_INTERNAL);

    std::vector<LinkDataIterator> tmpLinks(NUM_TWO);
    sendRecvLinks.resize(queNumPerNeighbor_, tmpLinks);

    CHK_PRT_RET(
        GetDetourSendRecvLinksIn4P(myRank_, neighborRank, tempLinks, sendRecvLinks),
        HCCL_ERROR(
            "[InsCollAlgFactory] [TempAllGatherMesh] Rank [%d], get send recv links in 2P Mesh in 4P topo.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempAllGatherMesh::RunIndividualPeer(
    const RankId neighborRank, const LinkData& neighborLinkData, const SliceInfo& sendSlice, const SliceInfo& recvSlice,
    PrimQuePtr currQue)
{
    SliceInfo currSendSlice = {sendSlice.offset + buffInfo_.outBuffBaseOff, sendSlice.size};
    SliceInfo currRecvSlice = {recvSlice.offset + buffInfo_.outBuffBaseOff, recvSlice.size};

    std::unique_ptr<PrimGroup> primGroup
        = RunSendRecv(neighborRank, neighborLinkData, neighborLinkData, currSendSlice, currRecvSlice);
    currQue->Append(std::move(primGroup));
    return HcclResult::HCCL_SUCCESS;
}

std::unique_ptr<PrimGroup> TempAllGatherMesh::RunSendRecv(
    const RankId neighborRank, const LinkData& sendLinkData, const LinkData& recvLinkData,
    const SliceInfo& currSendSlice, const SliceInfo& currRecvSlice) const
{
    // PrimGroup
    std::unique_ptr<PrimGroup> primGroup = std::make_unique<PrimGroup>();

    // Recv
    DataSlice recvRemSlice = DataSlice(buffInfo_.outBuffType, currRecvSlice.offset, currRecvSlice.size);
    DataSlice recvLocSlice = DataSlice(buffInfo_.outBuffType, currRecvSlice.offset, currRecvSlice.size);
    std::unique_ptr<Primitive> primRecv
        = std::make_unique<PrimRecv>(neighborRank, recvLinkData, recvLocSlice, recvRemSlice, dmaMode_);

    primGroup->Append(std::move(primRecv));

    // Send
    DataSlice sendLocSlice = DataSlice(buffInfo_.outBuffType, currSendSlice.offset, currSendSlice.size);
    DataSlice sendRemSlice = DataSlice(buffInfo_.outBuffType, currSendSlice.offset, currSendSlice.size);
    std::unique_ptr<Primitive> primSend
        = std::make_unique<PrimSend>(neighborRank, sendLinkData, sendLocSlice, sendRemSlice, dmaMode_);

    primGroup->Append(std::move(primSend));

    return primGroup;
}

HcclResult TempAllGatherMesh::PreCopyOffload(
    const RankSliceInfo& sliceInfoVec, const bool forAllReduce, std::vector<PrimQuePtr>& tempPrimQues)
{
    u64 srcOffset = forAllReduce ? sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset : 0;
    u64 srcSize = sliceInfoVec[tempVirtRankMap_[myRank_]][0].size;
    u64 dstOffset = sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset;
    DataSlice srcSlice = DataSlice(buffInfo_.inBuffType, srcOffset + buffInfo_.inBuffBaseOff, srcSize);
    DataSlice dstSlice = DataSlice(buffInfo_.outBuffType, dstOffset + buffInfo_.outBuffBaseOff, srcSize);
    std::unique_ptr<Primitive> primLocalCopy = std::make_unique<PrimLocalCopy>(srcSlice, dstSlice);
    tempPrimQues[0]->Append(std::move(primLocalCopy));

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
