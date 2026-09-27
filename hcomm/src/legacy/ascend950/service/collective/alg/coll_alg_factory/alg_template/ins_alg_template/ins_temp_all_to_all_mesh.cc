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

#include "ins_temp_all_to_all_mesh.h"

namespace Hccl {
InsTempAlltoAllMesh::InsTempAlltoAllMesh(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    if (tempRankSize_ == 0) {
        THROW<InvalidParamsException>(StringFormat("[InsTempAlltoAllMesh] Invalid tempRankSize[%u].", tempRankSize_));
    }
}

InsTempAlltoAllMesh::~InsTempAlltoAllMesh() {}

HcclResult InsTempAlltoAllMesh::CalcRes(AlgTempResReq& tempResReq)
{
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));

    auto& linkReq = tempResReq.links;
    for (auto resReqIter = linkReq.begin(); resReqIter != linkReq.end(); resReqIter++) {
        auto remoteRank = resReqIter->first;
        if (rank2PathNumMap_.find(remoteRank) == rank2PathNumMap_.end() || rank2PathNumMap_.at(remoteRank) == 0) {
            HCCL_ERROR("No path to remoteRank[%d]", remoteRank);
            return HcclResult::HCCL_E_INTERNAL;
        }
        resReqIter->second = rank2PathNumMap_.at(remoteRank);
        maxPathNum = std::max(maxPathNum, rank2PathNumMap_.at(remoteRank)); // 每个rank的远端搬运最多需要pathNum条流
    }
    tempResReq.queNum = 1 + maxPathNum * std::min(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, tempRankSize_);
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    HCCL_DEBUG(
        "[InsTempAlltoAllMesh] Rank[%d], VtopoSize[%zu], requiredQue Num [%u].", myRank_, tempVTopo_[0].size(),
        tempResReq.queNum);
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAlltoAllMesh::SetA2ASendRecvInfo(const A2ASendRecvInfo& sendRecvInfo) { localSendRecvInfo_ = sendRecvInfo; }

HcclResult InsTempAlltoAllMesh::GetScratchBufferInfo(const u64& scratchBufferSize, DataType dataType)
{
    // 需要变更为CCU/AICPU均只加载scratchSize
    u32 concurrentSendRecvNum = (tempRankSize_ > ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE) ?
                                    ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE :
                                    tempRankSize_;
    // scratch 不需要分两份，userIn直接一步到对端 scratch buffer
    u64 maxTmpMemSize = scratchBufferSize;
    u32 typeSize = DataTypeSizeGet(dataType);
    CHK_PRT_RET(
        typeSize == 0,
        HCCL_ERROR("[InsTempAlltoAllMesh] Rank [%d], Invalid dataSizePerVolume [%u].", myRank_, typeSize),
        HcclResult::HCCL_E_INTERNAL);
    u64 scratchInputMemSize = maxTmpMemSize / (concurrentSendRecvNum * typeSize) * typeSize;
    CHK_RET(SetBuffBlockSize(scratchInputMemSize));
    CHK_RET(SetConcurrentSendRecvNum(concurrentSendRecvNum));

    HCCL_INFO(
        "[InsTempAlltoAllMesh][GetScratchBufferInfo] concurrentSendRecvNum[%u], scratchInputMemSize[%llu]",
        concurrentSendRecvNum, scratchInputMemSize);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::SetBuffBlockSize(const u64 buffBlockSize)
{
    CHK_PRT_RET(
        buffBlockSize == 0, HCCL_ERROR("[InsTempAlltoAllMesh][SetBuffBlockSize] buffBlockSize should not be zero"),
        HcclResult::HCCL_E_PARA);
    buffBlockSize_ = buffBlockSize;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::SetConcurrentSendRecvNum(const u32 concurrentSendRecvNum)
{
    CHK_PRT_RET(
        concurrentSendRecvNum == 0,
        HCCL_ERROR("[InsTempAlltoAllMesh][SetConcurrentSendRecvNum] concurrentSendRecvNum should not be zero"),
        HcclResult::HCCL_E_PARA);
    concurrentSendRecvNum_ = concurrentSendRecvNum;
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempAlltoAllMesh::CalcStepNum()
{
    u32 numSubStep = 0;

    for (u32 destRank = 0; destRank < tempRankSize_; destRank++) {
        if (destRank == static_cast<u32>(myRank_)) {
            continue;
        }
        u32 currRankSendSubStep = ((localSendRecvInfo_.sendLength[destRank] + buffBlockSize_ - 1) / buffBlockSize_);
        u32 currRankRecvSubStep = ((localSendRecvInfo_.recvLength[destRank] + buffBlockSize_ - 1) / buffBlockSize_);
        numSubStep = std::max(numSubStep, std::max(currRankSendSubStep, currRankRecvSubStep));
    }
    return numSubStep;
}

HcclResult InsTempAlltoAllMesh::CalcSendSliceInfo(u32 remoteRank, UsrData& sendSliceInfo)
{
    // 判断在打平的视角下，当前的remoteRank 是在本rank的左边还是右边
    u32 pairNum = concurrentSendRecvNum_ / 2; // 每轮次，会与concurrentSendRecvNum个对端通信，左右各pairNum个
    u32 gapRight = (tempRankSize_ + remoteRank - myRank_) % tempRankSize_;
    u32 gapLeft = (tempRankSize_ + myRank_ - remoteRank) % tempRankSize_;
    u32 sendSrcBuffIdx = 0;
    u32 sendDstBuffIdx = 0;
    if (gapLeft < gapRight) {
        // 离左边更近
        u32 gap = gapLeft;
        sendSrcBuffIdx = pairNum + ((gap - 1) % pairNum);
        sendDstBuffIdx = pairNum - 1 - ((gap - 1) % pairNum);
    } else if (gapLeft > gapRight) {
        // 离右边更近
        u32 gap = gapRight;
        sendSrcBuffIdx = pairNum - 1 - ((gap - 1) % pairNum);
        sendDstBuffIdx = pairNum + ((gap - 1) % pairNum);
    } else {
        sendSrcBuffIdx = 0;
        sendDstBuffIdx = 0;
    }
    u64 sendDataOffset = 0;
    u64 remainSendLen = localSendRecvInfo_.sendLength[remoteRank];
    u64 sendSrcBuffOffset = sendSrcBuffIdx * buffBlockSize_;
    u64 sendDstBuffOffset = sendDstBuffIdx * buffBlockSize_;
    while (remainSendLen > 0) {
        u64 currDataRemainLen = localSendRecvInfo_.sendLength[remoteRank] - sendDataOffset;
        u64 sendLen = std::min(buffBlockSize_, currDataRemainLen);
        u64 userInOffset = localSendRecvInfo_.sendOffset[remoteRank] + sendDataOffset;
        u64 userOutOffset = localSendRecvInfo_.recvOffset[myRank_] + sendDataOffset;

        DataSlice userInSlice = DataSlice(BufferType::INPUT, userInOffset, sendLen);
        DataSlice sendScratchSlice = DataSlice(buffInfo_.inBuffType, sendSrcBuffOffset, sendLen);
        DataSlice sendDstSlice = DataSlice(buffInfo_.outBuffType, sendDstBuffOffset, sendLen);
        DataSlice userOutSlice = DataSlice(BufferType::OUTPUT, userOutOffset, sendLen);
        sendSliceInfo.usrInSlices.emplace_back(userInSlice);
        sendSliceInfo.scratchInSlices.emplace_back(sendScratchSlice);
        sendSliceInfo.scratchOutSlices.emplace_back(sendDstSlice);
        sendSliceInfo.usrOutSlices.emplace_back(userOutSlice);
        sendDataOffset += sendLen;
        remainSendLen -= sendLen;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::CalcRecvSliceInfo(u32 remoteRank, UsrData& readSliceInfo)
{
    u32 pairNum = concurrentSendRecvNum_ / 2;
    u32 gapRight = (remoteRank - myRank_ + tempRankSize_) % tempRankSize_;
    u32 gapLeft = (myRank_ - remoteRank + tempRankSize_) % tempRankSize_;
    u32 recvSrcBuffIdx = 0;
    u32 recvDstBuffIdx = 0;
    if (gapLeft < gapRight) {
        u32 gap = gapLeft;
        recvSrcBuffIdx = pairNum - 1 - ((gap - 1) % pairNum);
        recvDstBuffIdx = pairNum + ((gap - 1) % pairNum);
    } else if (gapLeft > gapRight) {
        u32 gap = gapRight;
        recvSrcBuffIdx = pairNum + ((gap - 1) % pairNum);
        recvDstBuffIdx = pairNum - 1 - ((gap - 1) % pairNum);
    } else {
        recvSrcBuffIdx = 0;
        recvDstBuffIdx = 0;
    }
    u64 recvDataOffset = 0;
    u64 remainRecvLen = localSendRecvInfo_.recvLength[remoteRank];
    u64 recvSrcBuffOffset = recvSrcBuffIdx * buffBlockSize_;
    u64 recvDstBuffOffset = recvDstBuffIdx * buffBlockSize_;
    while (remainRecvLen > 0) {
        u64 currDataRemainLen = localSendRecvInfo_.recvLength[remoteRank] - recvDataOffset;
        u64 recvLen = std::min(buffBlockSize_, currDataRemainLen);
        u64 userOutOffset = localSendRecvInfo_.recvOffset[remoteRank] + recvDataOffset;
        u64 userInOffset = localSendRecvInfo_.sendOffset[myRank_] + recvDataOffset;

        DataSlice recvInSlice = DataSlice(BufferType::INPUT, userInOffset, recvLen);
        DataSlice recvScratchSlice = DataSlice(buffInfo_.inBuffType, recvSrcBuffOffset, recvLen);
        DataSlice recvDstSlice = DataSlice(buffInfo_.outBuffType, recvDstBuffOffset, recvLen);
        DataSlice userOutSlice = DataSlice(BufferType::OUTPUT, userOutOffset, recvLen);
        readSliceInfo.usrInSlices.emplace_back(recvInSlice);
        readSliceInfo.scratchInSlices.emplace_back(recvScratchSlice);
        readSliceInfo.scratchOutSlices.emplace_back(recvDstSlice);
        readSliceInfo.usrOutSlices.emplace_back(userOutSlice);
        recvDataOffset += recvLen;
        remainRecvLen -= recvLen;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::CalcSendRecvAllSliceInfo(
    std::unordered_map<u32, UsrData>& sendSliceInfoMap, std::unordered_map<u32, UsrData>& recvSliceInfoMap)
{
    for (u32 remoteRank = 0; remoteRank < tempRankSize_; remoteRank++) {
        if (remoteRank == static_cast<u32>(myRank_)) {
            continue;
        }
        UsrData readSliceInfo;
        UsrData sendSliceInfo;
        CalcSendSliceInfo(remoteRank, sendSliceInfo);
        CalcRecvSliceInfo(remoteRank, readSliceInfo);
        sendSliceInfoMap[remoteRank] = sendSliceInfo;
        recvSliceInfoMap[remoteRank] = readSliceInfo;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempAlltoAllMesh::CalcCommRankSetforOneLoop(u32 roundIdx, const u32 groupRankSize, std::vector<u32>& commRanks) const
{
    commRanks.clear();
    u32 pairNumPerRound = concurrentSendRecvNum_ / 2;
    u32 pairSize = (groupRankSize < concurrentSendRecvNum_) ? (groupRankSize + 1) / 2 : pairNumPerRound;
    for (u32 i = roundIdx * pairNumPerRound + 1; i < (roundIdx * pairNumPerRound + pairSize + 1); i++) {
        u32 leftRemoteRank = (myRank_ + tempRankSize_ - i) % tempRankSize_;
        u32 rightRemoteRank = (myRank_ + i) % tempRankSize_;
        if (leftRemoteRank == rightRemoteRank) {
            commRanks.push_back(leftRemoteRank);
            break;
        } else {
            commRanks.push_back(leftRemoteRank);
            commRanks.push_back(rightRemoteRank);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::CopySendDataToScratch(
    u32 step, const std::vector<u32>& commRanks, std::unordered_map<u32, UsrData>& sendSliceInfo,
    const ResLinks& tempLinks, std::vector<InsQuePtr>& queues) const
{
    HCCL_INFO("[InsTempAlltoAllMesh] CopySendDataToScratch");
    u32 queueId = 0;
    for (u32 i = 0; i < commRanks.size(); i++) {
        u32 remoteRank = commRanks[i];
        HCCL_INFO("remoteRank = %u", remoteRank);
        u32 linkNum = rank2PathNumMap_.at(remoteRank);
        std::vector<LinkData> links = tempLinks.at(remoteRank);
        std::vector<float> dataSplitRate(linkNum);
        CHK_RET(CalcDataSplitRateForLinks(links, dataSplitRate));
        UsrData& currSendSliceInfo = sendSliceInfo[remoteRank];
        for (u32 j = 0; j < linkNum; j++) {
            InsQuePtr queue = queues[queueId];
            queueId++;
            HCCL_INFO("queueId=%u", queueId);
            if (step < currSendSliceInfo.usrInSlices.size()) {
                DataSlice& userInSliceAllLinks = currSendSliceInfo.usrInSlices[step];
                DataSlice& sendScratchSliceAllLinks = currSendSliceInfo.scratchInSlices[step];
                DataSlice userInSlice = CalcDataSliceForLinks(userInSliceAllLinks, dataSplitRate, j, dataType_);
                DataSlice sendScratchSlice
                    = CalcDataSliceForLinks(sendScratchSliceAllLinks, dataSplitRate, j, dataType_);

                InsQuePtr queue = queues[i];
                CHK_RET(LocalCopy(queue, userInSlice, sendScratchSlice));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::SendRecvData(
    u32 step, const std::vector<u32>& commRanks, std::unordered_map<u32, UsrData>& sendSliceInfo,
    std::unordered_map<u32, UsrData>& readSliceInfo, const ResLinks& tempLinks, std::vector<InsQuePtr>& queues) const
{
    CHK_PRT_RET(
        queues.empty(), HCCL_ERROR("[InsTempAlltoAllMesh][SendRecvData] empty queues"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(queues[0]);

    if (commRanks.size() * maxPathNum > queues.size()) {
        HCCL_ERROR("[InsTempAlltoAllMesh][SendRecvData] commRanks.size() * maxPathNum > queues.size() is wrong");
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t queuesId = 0;
    for (u32 i = 0; i < commRanks.size(); i++) {
        s32 remoteRank = static_cast<s32>(commRanks[i]);
        u32 linkNum = rank2PathNumMap_.at(remoteRank);
        UsrData& currSendSliceInfo = sendSliceInfo[remoteRank];
        UsrData& currReadSliceInfo = readSliceInfo[remoteRank];
        std::vector<LinkData> links = tempLinks.at(remoteRank);
        std::vector<float> dataSplitRate(linkNum);
        CHK_RET(CalcDataSplitRateForLinks(links, dataSplitRate));
        std::vector<DataSlice>& currSendSrcSlices
            = (dmaMode_ == DmaMode::GET) ? currSendSliceInfo.scratchInSlices : currSendSliceInfo.usrInSlices;
        std::vector<DataSlice>& currSendDstSlices
            = (dmaMode_ == DmaMode::GET) ? currSendSliceInfo.usrOutSlices : currSendSliceInfo.scratchOutSlices;
        std::vector<DataSlice>& currReadSrcSlices
            = (dmaMode_ == DmaMode::GET) ? currReadSliceInfo.scratchInSlices : currReadSliceInfo.usrInSlices;
        std::vector<DataSlice>& currReadDstSlices
            = (dmaMode_ == DmaMode::GET) ? currReadSliceInfo.usrOutSlices : currReadSliceInfo.scratchOutSlices;
        for (u32 j = 0; j < linkNum; j++) {
            InsQuePtr queue = queues[queuesId];
            queuesId++;

            LinkData link = tempLinks.at(remoteRank)[j];
            if (step < currSendSrcSlices.size() && step < currReadSrcSlices.size()) {
                DataSlice& sendSrcSliceAllLinks = currSendSrcSlices[step];
                DataSlice& sendDstSliceAllLinks = currSendDstSlices[step];
                DataSlice sendSrcSlice = CalcDataSliceForLinks(sendSrcSliceAllLinks, dataSplitRate, j, dataType_);
                DataSlice sendDstSlice = CalcDataSliceForLinks(sendDstSliceAllLinks, dataSplitRate, j, dataType_);
                std::vector<DataSlice> sendSrcSliceVec = {sendSrcSlice};
                std::vector<DataSlice> sendDstSliceVec = {sendDstSlice};
                SlicesList sendDataSlice(sendSrcSliceVec, sendDstSliceVec);
                DataSlice& recvSrcSliceAllLinks = currReadSrcSlices[step];
                DataSlice& recvDstSliceAllLinks = currReadDstSlices[step];
                DataSlice recvSrcSlice = CalcDataSliceForLinks(recvSrcSliceAllLinks, dataSplitRate, j, dataType_);
                DataSlice recvDstSlice = CalcDataSliceForLinks(recvDstSliceAllLinks, dataSplitRate, j, dataType_);
                std::vector<DataSlice> recvSrcSliceVec = {recvSrcSlice};
                std::vector<DataSlice> recvDstSliceVec = {recvDstSlice};
                SlicesList recvDataSlice(recvSrcSliceVec, recvDstSliceVec);
                TxRxSlicesList sendRecvSlice(sendDataSlice, recvDataSlice);

                TxRxLinks sendRecvLinks(link, link);
                SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlice);
                CHK_RET(SendRecv(sendRecvInfo, queue, 0, true, dmaMode_));
                HCCL_DEBUG(
                    "[InsTempAlltoAllMesh][SendRecvData] step[%u], commRank[%u], remoteRank[%d] run send and recv.",
                    step, i, remoteRank);
            } else if (step < currSendSrcSlices.size()) {
                DataSlice& sendSrcSliceAllLinks = currSendSrcSlices[step];
                DataSlice& sendDstSliceAllLinks = currSendDstSlices[step];
                DataSlice sendSrcSlice = CalcDataSliceForLinks(sendSrcSliceAllLinks, dataSplitRate, j, dataType_);
                DataSlice sendDstSlice = CalcDataSliceForLinks(sendDstSliceAllLinks, dataSplitRate, j, dataType_);
                std::vector<DataSlice> sendSrcSliceVec = {sendSrcSlice};
                std::vector<DataSlice> sendDstSliceVec = {sendDstSlice};
                SlicesList sendDataSlice(sendSrcSliceVec, sendDstSliceVec);
                DataInfo sendDataInfo(link, sendDataSlice);
                CHK_RET(Send(sendDataInfo, queue));
                HCCL_DEBUG(
                    "[InsTempAlltoAllMesh][SendRecvData] step[%u], commRank[%u], remoteRank[%d] run send.", step, i,
                    remoteRank);
            } else if (step < currReadSrcSlices.size()) {
                DataSlice& recvSrcSliceAllLinks = currReadSrcSlices[step];
                DataSlice& recvDstSliceAllLinks = currReadDstSlices[step];
                DataSlice recvSrcSlice = CalcDataSliceForLinks(recvSrcSliceAllLinks, dataSplitRate, j, dataType_);
                DataSlice recvDstSlice = CalcDataSliceForLinks(recvDstSliceAllLinks, dataSplitRate, j, dataType_);
                std::vector<DataSlice> recvSrcSliceVec = {recvSrcSlice};
                std::vector<DataSlice> recvDstSliceVec = {recvDstSlice};
                SlicesList recvDataSlice(recvSrcSliceVec, recvDstSliceVec);
                DataInfo recvDataInfo(link, recvDataSlice);
                CHK_RET(Recv(recvDataInfo, queue));
                HCCL_DEBUG(
                    "[InsTempAlltoAllMesh][SendRecvData] step[%u], commRank[%u], remoteRank[%d] run recv.", step, i,
                    remoteRank);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::CopyRecvDataFromScratch(
    u32 step, const std::vector<u32>& commRanks, std::unordered_map<u32, UsrData>& readSliceInfo,
    const ResLinks& tempLinks, std::vector<InsQuePtr>& queues) const
{
    HCCL_INFO("[InsTempAlltoAllMesh] CopyRecvDataFromScratch");
    u32 queueId = 0;
    for (u32 i = 0; i < commRanks.size(); i++) {
        u32 remoteRank = commRanks[i];
        HCCL_INFO("remoteRank = %u", remoteRank);
        u32 linkNum = rank2PathNumMap_.at(remoteRank);
        std::vector<LinkData> links = tempLinks.at(remoteRank);
        std::vector<float> dataSplitRate(linkNum);
        CHK_RET(CalcDataSplitRateForLinks(links, dataSplitRate));
        UsrData& currReadSliceInfo = readSliceInfo[remoteRank];
        for (u32 j = 0; j < linkNum; j++) {
            InsQuePtr queue = queues[queueId];
            queueId++;
            HCCL_INFO("queueId=%u", queueId);
            if (step < currReadSliceInfo.scratchOutSlices.size()) {
                DataSlice& recvScratchSliceAllLinks = currReadSliceInfo.scratchOutSlices[step];
                DataSlice& userOutSliceAllLinks = currReadSliceInfo.usrOutSlices[step];
                DataSlice recvScratchSlice
                    = CalcDataSliceForLinks(recvScratchSliceAllLinks, dataSplitRate, j, dataType_);
                DataSlice userOutSlice = CalcDataSliceForLinks(userOutSliceAllLinks, dataSplitRate, j, dataType_);
                CHK_RET(LocalCopy(queue, recvScratchSlice, userOutSlice));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::RunSendRecvBufferLoop(
    u32 step, const std::vector<u32>& commRanks, std::unordered_map<u32, UsrData>& sendSliceInfoMap,
    std::unordered_map<u32, UsrData>& recvSliceInfoMap, const ResLinks& tempLinks, std::vector<InsQuePtr>& queues) const
{
    if (commRanks.size() > 1) {
        CHK_RET(PreSyncInterQueues(queues));
    }

    if (dmaMode_ == DmaMode::PUT) {
        CHK_RET(SendRecvData(step, commRanks, sendSliceInfoMap, recvSliceInfoMap, tempLinks, queues));
        CHK_RET(CopyRecvDataFromScratch(step, commRanks, recvSliceInfoMap, tempLinks, queues));
    } else if (dmaMode_ == DmaMode::GET) {
        CHK_RET(CopySendDataToScratch(step, commRanks, sendSliceInfoMap, tempLinks, queues));
        CHK_RET(SendRecvData(step, commRanks, sendSliceInfoMap, recvSliceInfoMap, tempLinks, queues));
    }

    if (commRanks.size() > 1) {
        CHK_RET(PostSyncInterQueues(queues));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::RunSendRecvForAllRanks(
    u32 step, std::unordered_map<u32, UsrData>& sendSliceInfoMap, std::unordered_map<u32, UsrData>& recvSliceInfoMap,
    const ResLinks& tempLinks, std::vector<InsQuePtr>& queues)
{
    u64 commLoops = (tempRankSize_ + concurrentSendRecvNum_ - 1) / concurrentSendRecvNum_;
    u32 leftRankSize = tempRankSize_ - 1; // leftRankSize中去掉本卡
    std::vector<u32> commRanks;
    for (u32 roundIdx = 0; roundIdx < commLoops && leftRankSize > 0; roundIdx++) {
        u32 groupRankSize = (leftRankSize > concurrentSendRecvNum_) ? concurrentSendRecvNum_ : leftRankSize;
        CHK_RET(CalcCommRankSetforOneLoop(roundIdx, groupRankSize, commRanks));
        CHK_RET(RunSendRecvBufferLoop(step, commRanks, sendSliceInfoMap, recvSliceInfoMap, tempLinks, queues));
        leftRankSize -= groupRankSize;
        HCCL_DEBUG(
            "[InsTempAlltoAllMesh][RunSendRecvForAllRanks] commRanksSize[%zu], roundIdx[%u] finish.", commRanks.size(),
            roundIdx);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::LocalDataCopy(InsQuePtr tempInsQue)
{
    u64 userInOffset = localSendRecvInfo_.sendOffset[myRank_];
    u64 sendSize = localSendRecvInfo_.sendLength[myRank_];
    u64 userOutOffset = localSendRecvInfo_.recvOffset[myRank_];
    u64 recvSize = localSendRecvInfo_.recvLength[myRank_];
    DataSlice userInSlice = DataSlice(BufferType::INPUT, userInOffset, sendSize);
    DataSlice userOutSlice = DataSlice(BufferType::OUTPUT, userOutOffset, recvSize);
    CHK_RET(LocalCopy(tempInsQue, userInSlice, userOutSlice));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempFuncs;
    (void)sliceInfoVec;
    if (tempInsQues.size() == 0) {
        HCCL_ERROR("[CcuTempAlltoAllMesh1D] tempInsQues.size() is zero.");
        return HcclResult::HCCL_E_PARA;
    }
    dmaMode_ = DmaMode::PUT;
    if (IsPcieLink(tempLinks)) {
        dmaMode_ = DmaMode::GET;
    }
    buffInfo_ = buffInfo;
    u32 totalStep = CalcStepNum();
    HCCL_INFO(
        "[InsTempAlltoAllMesh] AllToAll full mesh total step is [%u], tempInsQuesSize[%zu].", totalStep,
        tempInsQues.size());
    std::unordered_map<u32, UsrData> sendSliceInfoMap;
    std::unordered_map<u32, UsrData> recvSliceInfoMap;
    CHK_RET(CalcSendRecvAllSliceInfo(sendSliceInfoMap, recvSliceInfoMap));
    std::vector<InsQuePtr> localCopyQues;
    if (tempInsQues.size() > 1) {
        localCopyQues.push_back(tempInsQues[0]);
        localCopyQues.push_back(tempInsQues.back());
    }

    if (localCopyQues.size() > 1) {
        CHK_RET(PreSyncInterQueues(localCopyQues));
    }
    // 最后一条流做localCopy
    CHK_RET(LocalDataCopy(tempInsQues.back()));
    // 其余的流去做sendRecv
    std::vector<InsQuePtr> sendRecvQues(tempInsQues.begin(), tempInsQues.begin() + (tempInsQues.size() - 1));
    for (u32 step = 0; step < totalStep; step++) {
        CHK_RET(RunSendRecvForAllRanks(step, sendSliceInfoMap, recvSliceInfoMap, tempLinks, sendRecvQues));
        HCCL_DEBUG("[InsTempAlltoAllMesh] AllToAll full mesh step[%u] execute success", step);
    }
    if (localCopyQues.size() > 1) {
        CHK_RET(PostSyncInterQueues(localCopyQues));
    }
    HCCL_INFO("[InsTempAlltoAllMesh] AllToAll full mesh rank[%d] finish.", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
