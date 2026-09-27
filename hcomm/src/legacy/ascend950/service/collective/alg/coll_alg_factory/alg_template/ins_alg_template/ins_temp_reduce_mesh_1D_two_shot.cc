/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_mesh_1D_two_shot.h"

#include "log.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {

InsTempReduceMesh1DTwoShot::InsTempReduceMesh1DTwoShot(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    idxToRankMap_.assign(this->tempRankSize_, -1);
    for (const auto& pair : tempVirtRankMap_) {
        if (pair.second < this->tempRankSize_) {
            idxToRankMap_[pair.second] = pair.first;
        }
    }
    HCCL_INFO("[InsTempReduceMesh1DTwoShot] Init.");
}

InsTempReduceMesh1DTwoShot::~InsTempReduceMesh1DTwoShot() { HCCL_INFO("[InsTempReduceMesh1DTwoShot] exit."); }

HcclResult InsTempReduceMesh1DTwoShot::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = tempRankSize_;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);

    CHK_PRT_RET(
        CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[InsTempReduceMesh1DTwoShot] Rank [%d], resLinks calculation error!", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempReduceMesh1DTwoShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}

HcclResult InsTempReduceMesh1DTwoShot::CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    u32 unitAlignSize = DataTypeSizeGet(dataType_);
    if (unitAlignSize == 0) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    u64 totalElements = dataSize / unitAlignSize;
    u64 baseElements = totalElements / tempRankSize_;
    u64 remainder = totalElements % tempRankSize_;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
        u64 currSize = 0;

        if (rankIdx < remainder) {
            currSize = (baseElements + 1) * unitAlignSize;
        } else {
            currSize = baseElements * unitAlignSize;
        }

        sliceInfoVec[rankIdx][0] = {accumOff, currSize};
        accumOff += currSize;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1DTwoShot::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    if (tempAlgParams.sliceSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    opMode_ = tempFuncs.opMode;
    enableCounterNotify_ = tempFuncs.enableCounterNotify;

    auto it = tempVirtRankMap_.find(myRank_);
    if (it == tempVirtRankMap_.end()) {
        HCCL_ERROR("[InsTempReduceMesh1DTwoShot] myRank [%d] not found in tempVirtRankMap.", myRank_);
        return HcclResult::HCCL_E_INTERNAL;
    }
    myIdx_ = it->second;

    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSlice(tempAlgParams.sliceSize, sliceInfoVec));

    CHK_RET(RunReduceScatter(sliceInfoVec, tempLinks, tempInsQues, tempAlgParams));

    CHK_RET(RunGatherToRoot(sliceInfoVec, tempLinks, tempInsQues, tempAlgParams));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1DTwoShot::RunReduceScatter(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
    const TemplateDataParams& tempAlgParams)
{
    u64 inOff = tempAlgParams.buffInfo.inBuffBaseOff;
    u64 scOff = tempAlgParams.buffInfo.scratchBuffBaseOff;

    PreSyncInterQueues(tempInsQues);

    const u64 mySliceSize = sliceInfoVec[myIdx_][0].size;
    const u64 mySliceOffset = sliceInfoVec[myIdx_][0].offset;

    for (u32 rankId = 0; rankId < tempRankSize_; rankId++) {
        u64 sliceSize = sliceInfoVec[rankId][0].size;
        u64 sliceOffset = sliceInfoVec[rankId][0].offset;

        DataSlice sendSrcSlice(tempAlgParams.buffInfo.inBuffType, sliceOffset + inOff, sliceSize);
        DataSlice sendDstSlice(
            tempAlgParams.buffInfo.scratBuffType, static_cast<u64>(myIdx_) * sliceSize + scOff, sliceSize);

        if (rankId == myIdx_) {
            if (sliceSize != 0) {
                CHK_RET(LocalCopy(tempInsQues[rankId], sendSrcSlice, sendDstSlice));
            }
        } else {
            DataSlice recvSrcSlice(tempAlgParams.buffInfo.inBuffType, mySliceOffset + inOff, mySliceSize);
            DataSlice recvDstSlice(
                tempAlgParams.buffInfo.scratBuffType, static_cast<u64>(rankId) * mySliceSize + scOff, mySliceSize);

            RankId targetRank = GetRankFromMap(rankId);
            if (targetRank == -1 || tempLinks.find(targetRank) == tempLinks.end()) {
                HCCL_ERROR(
                    "[InsTempReduceMesh1DTwoShot] Invalid rank [%u] mapped to [%d] or link not found.", rankId,
                    targetRank);
                return HcclResult::HCCL_E_INTERNAL;
            }

            const auto& link = tempLinks.at(targetRank)[0];
            TxRxLinks links(link, link);

            SlicesList sendSList({sendSrcSlice}, {sendDstSlice});
            SlicesList recvSList({recvSrcSlice}, {recvDstSlice});
            TxRxSlicesList txRxSList(sendSList, recvSList);

            CHK_RET(SendRecv(SendRecvInfo(links, txRxSList), tempInsQues[rankId], 0, true, DmaMode::PUT));
        }
    }

    PostSyncInterQueues(tempInsQues);

    if (mySliceSize != 0) {
        u64 destOffset = static_cast<u64>(myIdx_) * mySliceSize + scOff;
        DataSlice finalDest(tempAlgParams.buffInfo.scratBuffType, destOffset, mySliceSize);

        for (u32 i = 0; i < tempRankSize_; i++) {
            if (i == myIdx_) {
                continue;
            }
            DataSlice currentSrc(
                tempAlgParams.buffInfo.scratBuffType, static_cast<u64>(i) * mySliceSize + scOff, mySliceSize);
            CHK_RET(LocalReduce(tempInsQues[0], currentSrc, finalDest, dataType_, redOp_));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1DTwoShot::RunGatherToRoot(
    const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
    const TemplateDataParams& tempAlgParams)
{
    u64 scOff = tempAlgParams.buffInfo.scratchBuffBaseOff;
    u64 outOff = tempAlgParams.buffInfo.outBuffBaseOff;

    PreSyncInterQueues(tempInsQues);

    if (static_cast<u32>(myRank_) == root_) {
        for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
            u64 curSize = sliceInfoVec[rankIdx][0].size;
            if (curSize == 0)
                continue;

            if (rankIdx == myIdx_) {
                u64 srcOffset = static_cast<u64>(myIdx_) * curSize + scOff;
                DataSlice src(tempAlgParams.buffInfo.scratBuffType, srcOffset, curSize);
                DataSlice dst(tempAlgParams.buffInfo.outBuffType, sliceInfoVec[rankIdx][0].offset + outOff, curSize);
                CHK_RET(LocalCopy(tempInsQues[rankIdx], src, dst));
            } else {
                u64 remoteSrcOffset = static_cast<u64>(rankIdx) * curSize + scOff;
                DataSlice rsrc(tempAlgParams.buffInfo.scratBuffType, remoteSrcOffset, curSize);
                DataSlice rdest(tempAlgParams.buffInfo.outBuffType, sliceInfoVec[rankIdx][0].offset + outOff, curSize);

                RankId targetRank = GetRankFromMap(rankIdx);
                if (targetRank == -1 || tempLinks.find(targetRank) == tempLinks.end()) {
                    HCCL_ERROR(
                        "[InsTempReduceMesh1DTwoShot] Gather root: Invalid rank [%u] mapped to [%d] or link not found.",
                        rankIdx, targetRank);
                    return HcclResult::HCCL_E_INTERNAL;
                }

                const auto& link = tempLinks.at(targetRank)[0];
                SlicesList sliceList({rsrc}, {rdest});

                CHK_RET(Recv(DataInfo(link, sliceList), tempInsQues[rankIdx], 1, true, DmaMode::GET));
            }
        }
    } else {
        u32 rankIdx = myIdx_;
        u64 curSize = sliceInfoVec[rankIdx][0].size;

        if (curSize != 0) {
            DataSlice ssrc(tempAlgParams.buffInfo.scratBuffType, static_cast<u64>(rankIdx) * curSize + scOff, curSize);
            DataSlice sdest(tempAlgParams.buffInfo.outBuffType, sliceInfoVec[rankIdx][0].offset + outOff, curSize);

            if (tempLinks.find(root_) == tempLinks.end()) {
                HCCL_ERROR("[InsTempReduceMesh1DTwoShot] Gather non-root: Root rank [%u] link not found.", root_);
                return HcclResult::HCCL_E_INTERNAL;
            }

            const auto& link = tempLinks.at(root_)[0];
            SlicesList sliceList({ssrc}, {sdest});

            auto rootIt = tempVirtRankMap_.find(root_);
            if (rootIt == tempVirtRankMap_.end()) {
                HCCL_ERROR("[InsTempReduceMesh1DTwoShot] root_ [%u] not found in tempVirtRankMap.", root_);
                return HcclResult::HCCL_E_INTERNAL;
            }

            CHK_RET(Send(DataInfo(link, sliceList), tempInsQues[rootIt->second], 1, true, DmaMode::GET));
        }
    }

    PostSyncInterQueues(tempInsQues);
    return HcclResult::HCCL_SUCCESS;
}

RankId InsTempReduceMesh1DTwoShot::GetRankFromMap(const u32 rankIdx)
{
    if (static_cast<size_t>(rankIdx) >= idxToRankMap_.size()) {
        return -1;
    }
    return idxToRankMap_[rankIdx];
}
} // namespace Hccl
