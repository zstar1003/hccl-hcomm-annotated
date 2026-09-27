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

#include "temp_reduce_scatter_concurr_mesh.h"

namespace Hccl {
TempReduceScatterConcurrMesh::TempReduceScatterConcurrMesh(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

TempReduceScatterConcurrMesh::~TempReduceScatterConcurrMesh() {}

HcclResult TempReduceScatterConcurrMesh::CalcRes(
    const bool forAllReduce, AlgTempResReq& tempResReq, u32& requiredScratchMultiplier)
{
    (void)forAllReduce;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        tempResReq.queNum += tempVTopo_[dim].size() - 1;
    }
    requiredScratchMultiplier = tempRankSize_;

    u32 myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            // find neighbors -> virtualRank
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
            RankId neighborRank = tempVTopo_[dim][neighborAlgRank];
            HCCL_INFO(
                "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], Dim [%u], NeighborRank [%d].", myRank_, dim,
                neighborRank);

            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

/*
dataSize / (rankSize) --> chunkSize
dataSize / (rankSize * dimNum) --> sliceSize

SliceInfoVecforConcurrMesh: [1st chunk: [1st Slice, 2nd Slice], 2nd chunk: [1st Slice, 2nd Slice], ...]
*/
HcclResult TempReduceScatterConcurrMesh::CalcSliceInfo(
    const AllignInfo& allignInfo, const bool forAllReduce, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    u32 dimSize = 0;
    for (u32 dimIdx = 0; dimIdx < tempVTopo_.size(); dimIdx++) {
        if (tempVTopo_[dimIdx].size() != 1) {
            dimSize += 1;
        }
    }
    std::vector<SliceInfo> tmp(dimSize);
    sliceInfoVec.resize(tempRankSize_, tmp);

    if (forAllReduce) {
        // for allreduce, dataSize = total dataSize
        CHK_RET(CalcSliceInfoAllReduce(allignInfo, dataSize, sliceInfoVec));
    } else {
        // for reduce scatter, dataSize = chunkSize
        if (sliceInfoVec[0].size() == 1) {
            // one-dimensional mesh
            CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));
        } else {
            // multi-dimensional mesh
            CHK_RET(CalcRsAgSliceInfoConcurrMesh(myRank_, tempVTopo_, allignInfo, dataSize, sliceInfoVec));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterConcurrMesh::CalcSliceInfoAllReduce(
    const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    u64 unitAlignSize;
    CHK_RET(GetUnitAllignSize(allignInfo, unitAlignSize));

    u64 rankDataSize = RoundUp(dataSize, (tempRankSize_ * unitAlignSize)) * unitAlignSize;

    if (sliceInfoVec[0].size() == 1) {
        // one dimensional mesh
        u64 resDataSize = dataSize;
        for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
            u64 currChunkSize = (resDataSize > rankDataSize) ? rankDataSize : resDataSize;
            SliceInfo slice = {dataSize - resDataSize, currChunkSize};
            sliceInfoVec[rankIdx][0] = slice;
            resDataSize -= currChunkSize;
        }

        CHK_PRT_RET(
            (sliceInfoVec[tempRankSize_ - 1][0].offset + sliceInfoVec[tempRankSize_ - 1][0].size != dataSize),
            HCCL_ERROR(
                "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], SliceInfo calculation error for "
                "AllReduce ConcurrMesh!",
                myRank_),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        u32 dimSize0 = tempVTopo_[0].size();
        u32 dimSize1 = tempVTopo_[1].size();

        u64 resDataSize = dataSize;
        for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
            u64 currChunkSize = (resDataSize > rankDataSize) ? rankDataSize : resDataSize;
            u64 sliceSize0 = min(
                currChunkSize,
                RoundUp(currChunkSize, (dimSize0 + dimSize1) * unitAlignSize) * dimSize0 * unitAlignSize);
            SliceInfo slice0 = {dataSize - resDataSize, sliceSize0};
            sliceInfoVec[rankIdx][0] = slice0;
            resDataSize -= sliceSize0;

            u64 sliceSize1 = currChunkSize - sliceSize0;
            SliceInfo slice1 = {dataSize - resDataSize, sliceSize1};
            sliceInfoVec[rankIdx][1] = slice1;
            resDataSize -= sliceSize1;
        }

        CHK_PRT_RET(
            (sliceInfoVec[tempRankSize_ - 1][1].offset + sliceInfoVec[tempRankSize_ - 1][1].size != dataSize),
            HCCL_ERROR(
                "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], SliceInfo calculation error "
                "for AllReduce ConcurrMesh!",
                myRank_),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterConcurrMesh::GenPrimQue(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<PrimQuePtr>& tempPrimQues)
{
    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;
    buffInfo_ = buffInfo;
    HCCL_INFO(
        "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], EnableCounterNotify [%d].", myRank_,
        enableCounterNotify_);

    queNum_ = 0;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        queNum_ += tempVTopo_[dim].size() - 1;
    }
    CHK_PRT_RET(
        queNum_ != tempPrimQues.size(),
        HCCL_ERROR("[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // LocalCopy: from input to scratch In Buffer for OPBASE
    if ((opMode_ == OpMode::OPBASE) && tempFuncs.isForepart) {
        CHK_RET(PreCopyOpbase(tempFuncs.usrData, tempPrimQues));
    }

    if (sliceInfoVec[0].size() == 1) {
        CHK_RET(RunOneDimMesh(sliceInfoVec, tempLinks, tempPrimQues));
    } else {
        CHK_RET(RunConcurrMesh(sliceInfoVec, tempLinks, tempPrimQues));
    }

    // LocalCopy for standalone reducescatter in Offload Mode
    if ((opMode_ == OpMode::OFFLOAD) && !tempFuncs.forAllReduce && !tempFuncs.forAlgSeqComb) {
        CHK_RET(PostCopyOffload(sliceInfoVec, tempPrimQues));
    }

    // LocalCopy from scratch to output for Opbase
    if ((opMode_ == OpMode::OPBASE) && tempFuncs.isBottom && !tempFuncs.forAllReduce) {
        CHK_RET(PostCopyOpbase(tempFuncs.usrData, tempPrimQues));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterConcurrMesh::RunOneDimMesh(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<PrimQuePtr>& tempPrimQues)
{
    // semaphore sync
    if (queNum_ > 1) {
        CHK_RET(PreSyncInterQueues(tempPrimQues));
    }

    // locate myRank in tempVTopo -> algRank
    u32 myAlgRank;
    u32 validDim = (tempVTopo_[0].size() == 1) ? 1 : 0;
    HCCL_INFO("[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], valid Dim [%u].", myRank_, validDim);
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[validDim], myAlgRank));

    // runMesh
    CHK_PRT_RET(
        RunMesh(myAlgRank, tempVTopo_[validDim], sliceInfoVec, tempLinks, tempPrimQues) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], unable to run the mesh algorithm.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // semaphore sync
    if (queNum_ > 1) {
        CHK_RET(PostSyncInterQueues(tempPrimQues));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterConcurrMesh::RunMesh(
    const u32 myAlgRank, const std::vector<RankId>& vTopo, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
    std::vector<PrimQuePtr>& tempPrimQues)
{
    for (u32 queIdx = 0; queIdx < tempPrimQues.size(); queIdx++) {
        // find neighbors -> virtualRank
        RankId neighborRank = vTopo[(myAlgRank + 1 + queIdx) % tempRankSize_];
        // Link
        LinkData neighborLinkData = tempLinks.at(neighborRank)[0];

        u32 sendChunkIdx = tempVirtRankMap_[neighborRank];
        u64 sendOffset = sliceInfoVec[sendChunkIdx][0].offset;
        u64 sendSize = sliceInfoVec[sendChunkIdx][0].size;
        u32 recvChunkIdx = tempVirtRankMap_[myRank_];
        u64 recvOffset = sliceInfoVec[recvChunkIdx][0].offset;
        u64 recvSize = sliceInfoVec[recvChunkIdx][0].size;

        // PrimGroup
        std::unique_ptr<PrimGroup> primGroup = std::make_unique<PrimGroup>();

        // SendReduce
        DataSlice sendLocSlice = DataSlice(buffInfo_.inBuffType, sendOffset + buffInfo_.inBuffBaseOff, sendSize);
        DataSlice sendRemSrcSlice
            = DataSlice(buffInfo_.scratBuffType, sendOffset + buffInfo_.scratchBuffBaseOff, sendSize);
        DataSlice sendRemDstSlice = DataSlice(buffInfo_.inBuffType, sendOffset + buffInfo_.inBuffBaseOff, sendSize);
        std::unique_ptr<Primitive> primSendReduce = std::make_unique<PrimSendReduce>(
            neighborRank, neighborLinkData, sendLocSlice, sendRemSrcSlice, sendRemDstSlice, dataType_, redOp_,
            dmaMode_);

        primGroup->Append(std::move(primSendReduce));

        // RecvReduce
        DataSlice recvRemSlice = DataSlice(buffInfo_.inBuffType, recvOffset + buffInfo_.inBuffBaseOff, recvSize);
        DataSlice recvLocSrcSlice
            = DataSlice(buffInfo_.scratBuffType, recvOffset + buffInfo_.scratchBuffBaseOff, recvSize);
        DataSlice recvLocDstSlice = DataSlice(buffInfo_.inBuffType, recvOffset + buffInfo_.inBuffBaseOff, recvSize);
        std::unique_ptr<Primitive> primRecvReduce = std::make_unique<PrimRecvReduce>(
            neighborRank, neighborLinkData, recvRemSlice, recvLocSrcSlice, recvLocDstSlice, dataType_, redOp_,
            dmaMode_);

        primGroup->Append(std::move(primRecvReduce));

        tempPrimQues[queIdx]->Append(std::move(primGroup));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterConcurrMesh::RunConcurrMesh(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<PrimQuePtr>& tempPrimQues)
{
    std::vector<std::vector<PrimQuePtr>> dimQues;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        // assign queues
        std::vector<PrimQuePtr> tmpQue;
        for (u32 idx = 0; idx < tempVTopo_[dim].size() - 1; idx++) {
            if (dim == 0) {
                tmpQue.push_back(tempPrimQues[idx]);
            } else {
                tmpQue.push_back(tempPrimQues[tempVTopo_[0].size() - 1 + idx]);
            }
        }
        dimQues.push_back(tmpQue);
    }

    std::vector<PrimQuePtr> majorDimQue = {tempPrimQues[0], tempPrimQues[tempVTopo_[0].size() - 1]};

    // semaphore sync inter dimensions
    CHK_RET(PreSyncInterQueues(majorDimQue));

    // run concurrent Mesh Step 0
    u32 step = 0;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(RunSingleDimension(step, dim, sliceInfoVec, tempLinks, dimQues[dim]));
    }

    // semaphore sync
    CHK_RET(PostSyncInterQueues(majorDimQue));

    // semaphore sync inter dimensions
    CHK_RET(PreSyncInterQueues(majorDimQue));

    // run concurrent Mesh Step 1
    step = 1;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(RunSingleDimension(step, dim, sliceInfoVec, tempLinks, dimQues[dim]));
    }

    // semaphore sync
    CHK_RET(PostSyncInterQueues(majorDimQue));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TempReduceScatterConcurrMesh::RunSingleDimension(
    const u32& step, const u32& dim, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
    std::vector<PrimQuePtr>& dimPrimQues)
{
    CHK_PRT_RET(
        dim > 1,
        HCCL_ERROR("[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], invalid dim [%u].", myRank_, dim),
        HcclResult::HCCL_E_INTERNAL);

    // locate myRank in tempVTopo -> algRank
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));

    for (u32 queIdx = 0; queIdx < dimPrimQues.size(); queIdx++) {
        // semaphore sync
        if (dimPrimQues.size() > 1) {
            CHK_PRT_RET(
                PreSync(queIdx, dimPrimQues) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], Que [%u], Semaphore "
                    "Synchronization Failed.",
                    myRank_, dimPrimQues[queIdx]->GetId()),
                HcclResult::HCCL_E_INTERNAL);
        }

        // find neighbors -> virtualRank
        u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
        RankId neighborRank = tempVTopo_[dim][neighborAlgRank];

        // link
        LinkData neighborLinkData = tempLinks.at(neighborRank)[0];
        HCCL_INFO(
            "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], [%u]-th Que, queId [%u], neighborRank [%d].",
            myRank_, queIdx, dimPrimQues[queIdx]->GetId(), neighborRank);

        // PrimGroup
        std::unique_ptr<PrimGroup> primGroup = std::make_unique<PrimGroup>();

        std::vector<u32> sendChunkIdxs;
        std::vector<u32> recvChunkIdxs;

        if (step == 0) {
            for (u32 chunkIdx = 0; chunkIdx < tempVTopo_[1 - dim].size(); chunkIdx++) {
                u32 sendChunkIdx = (dim == 0) ? (neighborAlgRank + chunkIdx * tempVTopo_[0].size()) :
                                                (neighborAlgRank * tempVTopo_[0].size() + chunkIdx);
                sendChunkIdxs.push_back(sendChunkIdx);
                u32 recvChunkIdx = (dim == 0) ? (myAlgRank + chunkIdx * tempVTopo_[0].size()) :
                                                (myAlgRank * tempVTopo_[0].size() + chunkIdx);
                recvChunkIdxs.push_back(recvChunkIdx);
            }
        } else {
            sendChunkIdxs.push_back(tempVirtRankMap_[neighborRank]);
            recvChunkIdxs.push_back(tempVirtRankMap_[myRank_]);
        }

        // SendReduce
        u32 sliceIdx = (step == 0) ? dim : (1 - dim);
        std::unique_ptr<PrimSendReduce> primSendReduce
            = RunSendReduce(sliceInfoVec, sendChunkIdxs, sliceIdx, neighborRank, neighborLinkData);
        primGroup->Append(std::move(primSendReduce));

        // RecvReduce
        std::unique_ptr<PrimRecvReduce> primRecvReduce
            = RunRecvReduce(sliceInfoVec, recvChunkIdxs, sliceIdx, neighborRank, neighborLinkData);
        primGroup->Append(std::move(primRecvReduce));

        dimPrimQues[queIdx]->Append(std::move(primGroup));

        // semaphore sync
        if (dimPrimQues.size() > 1) {
            CHK_PRT_RET(
                PostSync(queIdx, dimPrimQues) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], Que [%u], Semaphore "
                    "Synchronization Failed.",
                    myRank_, dimPrimQues[queIdx]->GetId()),
                HcclResult::HCCL_E_INTERNAL);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

std::unique_ptr<PrimSendReduce> TempReduceScatterConcurrMesh::RunSendReduce(
    const RankSliceInfo& sliceInfoVec, const std::vector<u32>& sendChunkIdxs, const u32& sliceIdx,
    const RankId& neighborRank, const LinkData& priorLinkData)
{
    std::unique_ptr<PrimSendReduce> primSendReduce;
    u64 tmpSendOff;
    u64 tmpSendSize;
    for (u32 chunkIdx = 0; chunkIdx < sendChunkIdxs.size(); chunkIdx++) {
        if (chunkIdx == 0) {
            // first slice
            tmpSendOff = sliceInfoVec[sendChunkIdxs[chunkIdx]][sliceIdx].offset;
            tmpSendSize = sliceInfoVec[sendChunkIdxs[chunkIdx]][sliceIdx].size;
        } else if (tmpSendOff + tmpSendSize == sliceInfoVec[sendChunkIdxs[chunkIdx]][sliceIdx].offset) {
            // consequent slice
            tmpSendSize += sliceInfoVec[sendChunkIdxs[chunkIdx]][sliceIdx].size;
        } else {
            DataSlice sendLocSlice = DataSlice(buffInfo_.inBuffType, tmpSendOff + buffInfo_.inBuffBaseOff, tmpSendSize);
            DataSlice sendRemSrcSlice
                = DataSlice(buffInfo_.scratBuffType, tmpSendOff + buffInfo_.scratchBuffBaseOff, tmpSendSize);
            DataSlice sendRemDstSlice
                = DataSlice(buffInfo_.inBuffType, tmpSendOff + buffInfo_.inBuffBaseOff, tmpSendSize);
            if (!primSendReduce) {
                primSendReduce.reset(new PrimSendReduce(
                    neighborRank, priorLinkData, sendLocSlice, sendRemSrcSlice, sendRemDstSlice, dataType_, redOp_,
                    dmaMode_));
            } else {
                primSendReduce->Append(sendLocSlice, sendRemSrcSlice, sendRemDstSlice);
            }
            tmpSendOff = sliceInfoVec[sendChunkIdxs[chunkIdx]][sliceIdx].offset;
            tmpSendSize = sliceInfoVec[sendChunkIdxs[chunkIdx]][sliceIdx].size;
        }

        if (chunkIdx == (sendChunkIdxs.size() - 1)) {
            HCCL_INFO("[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], last chunk.", myRank_);
            DataSlice sendLocSlice = DataSlice(buffInfo_.inBuffType, tmpSendOff + buffInfo_.inBuffBaseOff, tmpSendSize);
            DataSlice sendRemSrcSlice
                = DataSlice(buffInfo_.scratBuffType, tmpSendOff + buffInfo_.scratchBuffBaseOff, tmpSendSize);
            DataSlice sendRemDstSlice
                = DataSlice(buffInfo_.inBuffType, tmpSendOff + buffInfo_.inBuffBaseOff, tmpSendSize);
            if (!primSendReduce) {
                HCCL_INFO(
                    "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], last chunk is a non-consecutive chunk.",
                    myRank_);
                primSendReduce.reset(new PrimSendReduce(
                    neighborRank, priorLinkData, sendLocSlice, sendRemSrcSlice, sendRemDstSlice, dataType_, redOp_,
                    dmaMode_));
            } else {
                HCCL_INFO(
                    "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], last chunk is consecutive.", myRank_);
                primSendReduce->Append(sendLocSlice, sendRemSrcSlice, sendRemDstSlice);
            }
        }
    }

    return primSendReduce;
}

std::unique_ptr<PrimRecvReduce> TempReduceScatterConcurrMesh::RunRecvReduce(
    const RankSliceInfo& sliceInfoVec, const std::vector<u32>& recvChunkIdxs, const u32& sliceIdx,
    const RankId& neighborRank, const LinkData& priorLinkData)
{
    std::unique_ptr<PrimRecvReduce> primRecvReduce;
    u64 tmpRecvOff;
    u64 tmpRecvSize;
    for (u32 chunkIdx = 0; chunkIdx < recvChunkIdxs.size(); chunkIdx++) {
        if (chunkIdx == 0) {
            // first slice
            tmpRecvOff = sliceInfoVec[recvChunkIdxs[chunkIdx]][sliceIdx].offset;
            tmpRecvSize = sliceInfoVec[recvChunkIdxs[chunkIdx]][sliceIdx].size;
        } else if (tmpRecvOff + tmpRecvSize == sliceInfoVec[recvChunkIdxs[chunkIdx]][sliceIdx].offset) {
            // consequent slice
            tmpRecvSize += sliceInfoVec[recvChunkIdxs[chunkIdx]][sliceIdx].size;
        } else {
            DataSlice recvRemSlice = DataSlice(buffInfo_.inBuffType, tmpRecvOff + buffInfo_.inBuffBaseOff, tmpRecvSize);
            DataSlice recvLocSrcSlice
                = DataSlice(buffInfo_.scratBuffType, tmpRecvOff + buffInfo_.scratchBuffBaseOff, tmpRecvSize);
            DataSlice recvLocDstSlice
                = DataSlice(buffInfo_.inBuffType, tmpRecvOff + buffInfo_.inBuffBaseOff, tmpRecvSize);

            if (!primRecvReduce) {
                HCCL_INFO(
                    "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], last chunk is a non-consecutive chunk.",
                    myRank_);
                primRecvReduce.reset(new PrimRecvReduce(
                    neighborRank, priorLinkData, recvRemSlice, recvLocSrcSlice, recvLocDstSlice, dataType_, redOp_,
                    dmaMode_));
            } else {
                HCCL_INFO(
                    "[CollAlgFactory] [TempReduceScatterConcurrMesh] Rank [%d], last chunk is consecutive.", myRank_);
                primRecvReduce->Append(recvRemSlice, recvLocSrcSlice, recvLocDstSlice);
            }
            tmpRecvOff = sliceInfoVec[recvChunkIdxs[chunkIdx]][sliceIdx].offset;
            tmpRecvSize = sliceInfoVec[recvChunkIdxs[chunkIdx]][sliceIdx].size;
        }

        if (chunkIdx == (recvChunkIdxs.size() - 1)) {
            DataSlice recvRemSlice = DataSlice(buffInfo_.inBuffType, tmpRecvOff + buffInfo_.inBuffBaseOff, tmpRecvSize);
            DataSlice recvLocSrcSlice
                = DataSlice(buffInfo_.scratBuffType, tmpRecvOff + buffInfo_.scratchBuffBaseOff, tmpRecvSize);
            DataSlice recvLocDstSlice
                = DataSlice(buffInfo_.inBuffType, tmpRecvOff + buffInfo_.inBuffBaseOff, tmpRecvSize);

            if (!primRecvReduce) {
                primRecvReduce.reset(new PrimRecvReduce(
                    neighborRank, priorLinkData, recvRemSlice, recvLocSrcSlice, recvLocDstSlice, dataType_, redOp_,
                    dmaMode_));
            } else {
                primRecvReduce->Append(recvRemSlice, recvLocSrcSlice, recvLocDstSlice);
            }
        }
    }
    return primRecvReduce;
}

HcclResult
TempReduceScatterConcurrMesh::PostCopyOffload(const RankSliceInfo& sliceInfoVec, std::vector<PrimQuePtr>& tempPrimQues)
{
    u64 srcOffset = sliceInfoVec[tempVirtRankMap_[myRank_]][0].offset;
    u64 srcSize = 0;
    for (u32 dimIdx = 0; dimIdx < sliceInfoVec[0].size(); dimIdx++) {
        srcSize += sliceInfoVec[tempVirtRankMap_[myRank_]][dimIdx].size;
    }
    u64 dstOffset = 0;
    DataSlice srcSlice = DataSlice(buffInfo_.inBuffType, srcOffset + buffInfo_.inBuffBaseOff, srcSize);
    DataSlice dstSlice = DataSlice(buffInfo_.outBuffType, dstOffset + buffInfo_.outBuffBaseOff, srcSize);
    std::unique_ptr<Primitive> primLocalCopy = std::make_unique<PrimLocalCopy>(srcSlice, dstSlice);
    tempPrimQues[0]->Append(std::move(primLocalCopy));

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
