/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoall_symmetric_memory.h"

namespace hccl {
AlltoAllFullMeshSymmetricMemory::AlltoAllFullMeshSymmetricMemory(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{}

AlltoAllFullMeshSymmetricMemory::~AlltoAllFullMeshSymmetricMemory() {}

HcclResult AlltoAllFullMeshSymmetricMemory::GenerateSubStreamInfo(
    const std::vector<Stream>& subStreams, const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain)
{
    u32 totalSubstreamSize = sdmaConcurrentNum_;
    if (subStreams.size() < totalSubstreamSize || meshSignalMainToSub.size() < totalSubstreamSize
        || meshSignalSubToMain.size() < totalSubstreamSize) {
        HCCL_ERROR(
            "[AlltoAllFullMeshSymmetricMemory][GenerateSubStreamInfo]subStreamsSize[%zu], meshSignalMainToSubSize[%zu]"
            "meshSignalSubToMainSize[%zu] is smaller than totalSubstreamSize[%u]",
            subStreams.size(), meshSignalMainToSub.size(), meshSignalSubToMain.size(), totalSubstreamSize);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(
        links_.size() < userRankSize_,
        HCCL_ERROR(
            "[AlltoAllFullMeshSymmetricMemory][GenerateSubStreamInfo]"
            "links_.size()[%zu] is smaller than userRankSize_[%u].",
            links_.size(), userRankSize_),
        HCCL_E_PARA);
    HCCL_DEBUG(
        "subStreams.size[%zu], meshSignalMainToSub.size[%zu], links_.size[%zu]", subStreams.size(),
        meshSignalMainToSub.size(), links_.size());
    for (u32 sdmaIndex = 0; sdmaIndex < sdmaConcurrentNum_; sdmaIndex++) {
        sdmaSubStream_.push_back(subStreams[sdmaIndex]);
        sdmaMeshSignalMainToSub_.push_back(meshSignalMainToSub[sdmaIndex]);
        sdmaMeshSignalSubToMain_.push_back(meshSignalSubToMain[sdmaIndex]);
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::Prepare(PrepareData& param)
{
    mainStream_ = param.stream;
    userRank_ = param.userRank;
    userRankSize_ = param.userRankSize;
    links_ = *param.linksPtr;
    sendRecvInfoPtr_ = param.sendRecvInfoPtr;
    devNumInlocalPod_ = param.devNumInlocalPod;
    rankIdxInPod_ = param.rankIdxInPod;
    opType_ = param.opType;
    algOpContext_ = param.algOpContext;

    podStartRank_ = userRank_ - rankIdxInPod_;
    podEndRank_ = podStartRank_ + devNumInlocalPod_ - 1;
    sdmaConcurrentNum_ = (devNumInlocalPod_ > ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) ?
                             (ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) :
                             (devNumInlocalPod_);

    HCCL_DEBUG(
        "[AlltoAllFullMeshSymmetricMemory]devNumInlocalPod_[%u], userRankSize_[%u] podStartRank_[%u]"
        "podEndRank_[%u], sdmaConcurrentNum_[%u]",
        devNumInlocalPod_, userRankSize_, podStartRank_, podEndRank_, sdmaConcurrentNum_);

    CHK_PRT_RET(
        userRankSize_ == 0, HCCL_ERROR("[AlltoAllFullMeshSymmetricMemory][Prepare]userRankSize_ is zero."),
        HCCL_E_PARA);

    userInput_ = param.inputMem;
    userOutput_ = param.outputMem;
    workMode_ = param.workMode;

    CHK_RET(GenerateSubStreamInfo(*param.subStreamsPtr, *param.signalPtr, *param.signalAuxPtr));
    return HCCL_SUCCESS;
}

std::string AlltoAllFullMeshSymmetricMemory::GetStreamIndexString()
{
    std::string res = "";
    for (auto& info : subStreamReadInfo_) {
        u32 destRank = info.first;
        u32 streamIndex = destRank % sdmaConcurrentNum_;
        res += std::to_string(streamIndex) + ", ";
    }
    return res;
}

void AlltoAllFullMeshSymmetricMemory::UpdateCurrRankRecvInfo(
    [[maybe_unused]] u32 roundIdx, [[maybe_unused]] u32 side, u32 destRank, ReadDataBlock& readInfo)
{
    const ZCopySendRecvInfo& sendRecvInfo = *sendRecvInfoPtr_;
    u64 recvLen = sendRecvInfo.localRecvLength[destRank];
    u64 userOutOffset = sendRecvInfo.localRecvOffset[destRank];
    u64 remoteUserInOffset = sendRecvInfo.remoteSendOffset[destRank];
    HCCL_DEBUG(
        "[AlltoAllFullMeshSymmetricMemory][UpdateCurrRankRecvInfo] usrRank[%u] recv from destRank [%u]"
        "recvLen[%lu] remoteUserInOffset[%llu] userOutOffset[%llu]",
        userRank_, destRank, recvLen, remoteUserInOffset, userOutOffset);
    readInfo = {recvLen, remoteUserInOffset, userOutOffset};
}

void AlltoAllFullMeshSymmetricMemory::UpdateSendRecvInfo(
    u32 roundIdx, std::unordered_map<u32, ReadDataBlock>& subStreamReadInfo,
    const std::vector<std::vector<std::pair<u32, u32>>>& partialCommRankSet)
{
    for (u32 side = 0; side < partialCommRankSet.size(); side++) {
        for (u32 j = 0; j < partialCommRankSet[side].size(); j++) {
            u32 readRemoteRank = partialCommRankSet[side][j].first;
            if (readRemoteRank == userRank_) {
                continue;
            }
            ReadDataBlock readInfo;
            UpdateCurrRankRecvInfo(roundIdx, side, readRemoteRank, readInfo);

            subStreamReadInfo[readRemoteRank] = readInfo;
        }
    }
}

void AlltoAllFullMeshSymmetricMemory::UpdateRemoteRankSet(u32 roundIdx, u32 groupRankSize)
{
    if (sdmaConcurrentNum_ == 1) {
        UpdatePartialCommunicationRankSetPairWise(roundIdx, groupRankSize);
    } else {
        UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
    }
}

void AlltoAllFullMeshSymmetricMemory::UpdatePartialCommunicationRankSetPairWise(u32 roundIdx, u32 groupRankSize)
{
    partialCommRankSet_.clear();
    partialCommRankSet_.resize(1);
    for (u32 i = roundIdx * sdmaConcurrentNum_; i < (roundIdx * sdmaConcurrentNum_ + groupRankSize); i++) {
        u32 readRemoteRank = podStartRank_ + (rankIdxInPod_ + devNumInlocalPod_ - i) % devNumInlocalPod_;
        u32 sendRemoteRank = podStartRank_ + (rankIdxInPod_ + i) % devNumInlocalPod_;
        partialCommRankSet_[0].push_back(std::make_pair(readRemoteRank, sendRemoteRank));
        HCCL_DEBUG(
            "[AlltoAllFullMeshSymmetricMemory][UpdatePartialCommunicationRankSetPairWise] userRank [%u] i[%u]"
            "readRemoteRank[%u] writeRemoteRank[%u]",
            userRank_, i, readRemoteRank, sendRemoteRank);
    }
    HCCL_DEBUG(
        "[AlltoAllFullMeshSymmetricMemory][UpdatePartialCommunicationRankSetPairWise] partialCommRankSet_ size[%zu]",
        partialCommRankSet_[0].size());
}

void AlltoAllFullMeshSymmetricMemory::UpdatePartialCommunicationRankSet(
    u32 roundIdx, u32 groupRankSize, std::vector<std::vector<std::pair<u32, u32>>>& partialCommRankSet)
{
    partialCommRankSet.clear();
    partialCommRankSet.resize(RANK_SET_COMPUTE_CONST + 1);
    u32 pairNumPerRound = sdmaConcurrentNum_ / RANK_SET_COMPUTE_CONST;
    u32 pairSize = (groupRankSize < sdmaConcurrentNum_) ?
                       (groupRankSize + RANK_SET_COMPUTE_CONST - 1) / RANK_SET_COMPUTE_CONST :
                       pairNumPerRound;
    for (u32 i = roundIdx * pairNumPerRound + 1; i < (roundIdx * pairNumPerRound + pairSize + 1); i++) {
        u32 leftRemoteRank = podStartRank_ + (rankIdxInPod_ + devNumInlocalPod_ - i) % devNumInlocalPod_;
        u32 rightRemoteRank = podStartRank_ + (rankIdxInPod_ + i) % devNumInlocalPod_;
        if (leftRemoteRank == rightRemoteRank) {
            partialCommRankSet[2].push_back(std::make_pair(leftRemoteRank, leftRemoteRank));
        } else {
            partialCommRankSet[0].push_back(std::make_pair(leftRemoteRank, leftRemoteRank));
            partialCommRankSet[1].push_back(std::make_pair(rightRemoteRank, rightRemoteRank));
        }
        HCCL_DEBUG(
            "[AlltoAllFullMeshSymmetricMemory][UpdatePartialCommunicationRankSet] round[%u] userRank [%u] i[%u]"
            "read/write leftRemoteRank[%u] rightRemoteRank[%u]",
            roundIdx, userRank_, i, leftRemoteRank, rightRemoteRank);
    }
    HCCL_DEBUG(
        "[AlltoAllFullMeshSymmetricMemory][UpdatePartialCommunicationRankSet] round[%u] partialCommRankSet_ total "
        "size[%zu]",
        roundIdx, partialCommRankSet[0].size() + partialCommRankSet[1].size() + partialCommRankSet[2].size());
}

// 主流只需要通知当前子步骤需要收发数据的 SDMA 流，减少同步开销
HcclResult AlltoAllFullMeshSymmetricMemory::NotifySubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(
            LocalNotify::Post(mainStream_, dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            sdmaSubStream_[streamIndex], dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[AlltoAllFullMeshSymmetricMemory][NotifySubStreamStart] userRank [%u] main stream notify sdma stream [%s]",
        userRank_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::WaitSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            sdmaSubStream_[streamIndex], dispatcher_, sdmaMeshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(
            LocalNotify::Wait(mainStream_, dispatcher_, sdmaMeshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[AlltoAllFullMeshSymmetricMemory][WaitSubStreamFinish] userRank [%u] main stream wait sdma stream [%s]",
        userRank_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::NotifyRemoteRankStart()
{
    u32 streamIndex = 0;
    for (auto& sendRecvSide : partialCommRankSet_) {
        for (auto& sendRecvPair : sendRecvSide) {
            u32 recvRank = sendRecvPair.first;
            u32 sendRank = sendRecvPair.second;
            if (sendRank == userRank_) {
                continue;
            }
            Stream& currStream = sdmaSubStream_[streamIndex];
            const LINK& readTransport = links_[recvRank];
            const LINK& sendTransport = links_[sendRank];

            CHK_RET(sendTransport->TxAck(currStream));
            CHK_RET(readTransport->RxAck(currStream));

            streamIndex++;
        }
    }
    HCCL_INFO("[AlltoAllFullMeshSymmetricMemory][NotifyRemoteRankStart] done");
    return HCCL_SUCCESS;
}

bool AlltoAllFullMeshSymmetricMemory::IsPostSyncEnable(u32 roundIdx)
{
    bool isPostSyncEnable = false;
    isPostSyncEnable = (roundIdx == lastRoundIdx_) && algOpContext_.opRetryHandler.retryEnable;
    return isPostSyncEnable;
}

HcclResult AlltoAllFullMeshSymmetricMemory::SdmaMainStreamWait(u32 roundIdx)
{
    // SDMA wait
    u32 streamIndex = 0;
    for (auto& sendRecvSide : partialCommRankSet_) {
        for (auto& sendRecvPair : sendRecvSide) {
            u32 recvRank = sendRecvPair.first;
            u32 sendRank = sendRecvPair.second;
            if (sendRank == userRank_) {
                continue;
            }
            HCCL_DEBUG(
                "[AlltoAllFullMeshSymmetricMemory][SdmaMainStreamWait] userRank [%u], recvRank[%u], "
                "sendRank[%u], sdma stream [%u], "
                "post sync info: roundIdx[%u], lastRoundIdx_[%u] main stream wait",
                userRank_, recvRank, sendRank, streamIndex, roundIdx, lastRoundIdx_);
            CHK_RET(LocalNotify::Wait(
                mainStream_, dispatcher_, sdmaMeshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));

            streamIndex++;
        }
    }
    HCCL_INFO("[AlltoAllFullMeshSymmetricMemory][SdmaMainStreamWait] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::SdmaMainStreamPost(u32 roundIdx)
{
    // SDMA post
    u32 streamIndex = 0;
    for (auto& sendRecvSide : partialCommRankSet_) {
        for (auto& sendRecvPair : sendRecvSide) {
            u32 recvRank = sendRecvPair.first;
            u32 sendRank = sendRecvPair.second;
            if (sendRank == userRank_) {
                continue;
            }
            HCCL_DEBUG(
                "[AlltoAllFullMeshSymmetricMemory][SdmaMainStreamPost] userRank [%u], recvRank[%u], "
                "sendRank[%u], sdma stream [%u], "
                "post sync info: roundIdx[%u], lastRoundIdx_[%u] main stream post",
                userRank_, recvRank, sendRank, streamIndex, roundIdx, lastRoundIdx_);
            CHK_RET(LocalNotify::Post(
                mainStream_, dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));

            streamIndex++;
        }
    }
    HCCL_INFO("[AlltoAllFullMeshSymmetricMemory][SdmaMainStreamPost] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::SetPostSyncTasks(u32 roundIdx)
{
    // SDMA wait
    CHK_RET(SdmaMainStreamWait(roundIdx));
    // SDMA post
    CHK_RET(SdmaMainStreamPost(roundIdx));
    HCCL_DEBUG("[AlltoAllFullMeshSymmetricMemory][SetPostSyncTasks] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::SDMAwithRemoteRankAndNotifyEnd(u32 roundIdx)
{
    bool isPostSyncEnable = IsPostSyncEnable(roundIdx);
    if (isPostSyncEnable) {
        // 下发主流上的后同步wait和post
        CHK_RET(SetPostSyncTasks(roundIdx));
    }
    u32 streamIndex = 0;
    for (auto& sendRecvSide : partialCommRankSet_) {
        for (auto& sendRecvPair : sendRecvSide) {
            u32 recvRank = sendRecvPair.first;
            u32 sendRank = sendRecvPair.second;
            if (sendRank == userRank_) {
                continue;
            }
            const ReadDataBlock& readInfo = subStreamReadInfo_[recvRank];
            Stream& currStream = sdmaSubStream_[streamIndex];
            const LINK& readTransport = links_[recvRank];
            const LINK& sendTransport = links_[sendRank];

            const LINK& intraNeighboorTransport = links_[recvRank];
            CHK_PTR_NULL(intraNeighboorTransport);
            void* remDMAMemPtr = nullptr;
            CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::INPUT_MEM, &remDMAMemPtr));
            DeviceMem remoteUserInMem = DeviceMem::create(static_cast<u8*>(remDMAMemPtr), userInput_.size());
            DeviceMem srcMem = remoteUserInMem.range(readInfo.remoteOffset, readInfo.recvLen);
            DeviceMem dstMem = userOutput_.range(readInfo.recvOffset, readInfo.recvLen);
            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, dstMem, srcMem, currStream, readTransport->GetRemoteRank(), readTransport->GetLinkType()));
            HCCL_DEBUG(
                "[AlltoAllFullMeshSymmetricMemory][SendRecvData] userRank [%u], recvRank[%u],"
                "sdma stream [%u] read data from remote offset [%llu] len [%llu] to local [%llu], "
                "post sync info: roundIdx[%u], lastRoundIdx_[%u]",
                userRank_, recvRank, streamIndex, readInfo.remoteOffset, readInfo.recvLen, readInfo.recvOffset,
                roundIdx, lastRoundIdx_);
            if (isPostSyncEnable) {
                HCCL_DEBUG("[AlltoAllFullMeshSymmetricMemory][SendRecvData] post sync begins");
                CHK_RET(LocalNotify::Post(
                    currStream, dispatcher_, sdmaMeshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
                CHK_RET(LocalNotify::Wait(
                    currStream, dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
            }
            CHK_RET(readTransport->TxDataSignal(currStream));
            CHK_RET(sendTransport->RxDataSignal(currStream));

            streamIndex++;
        }
    }
    HCCL_INFO("[AlltoAllFullMeshSymmetricMemory][SDMAwithRemoteRankAndNotifyEnd] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::SendRecvData(u32 roundIdx)
{
    HCCL_DEBUG(
        "[AlltoAllFullMeshSymmetricMemory][SendRecvData] userRank [%u] sdma stream [%s] wait main stream", userRank_,
        GetStreamIndexString().c_str());
    CHK_RET(NotifyRemoteRankStart());
    CHK_RET(WaitSubStreamFinish());
    CHK_RET(NotifySubStreamStart());
    CHK_RET(SDMAwithRemoteRankAndNotifyEnd(roundIdx));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::LocalCopy()
{
    const ZCopySendRecvInfo& sendRecvInfo = *sendRecvInfoPtr_;
    DeviceMem src = userInput_.range(sendRecvInfo.remoteSendOffset[userRank_], sendRecvInfo.localRecvLength[userRank_]);
    DeviceMem dst = userOutput_.range(sendRecvInfo.localRecvOffset[userRank_], sendRecvInfo.localRecvLength[userRank_]);
    HCCL_DEBUG(
        "[AlltoAllFullMeshSymmetricMemory][LocalCopy]userRank [%u] copy from userInput [%llu]"
        "to userOutput [%llu] dstLen[%llu]",
        userRank_, sendRecvInfo.remoteSendOffset[userRank_], sendRecvInfo.localRecvOffset,
        sendRecvInfo.localRecvLength[userRank_]);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::RunGroupFullMeshAlltoall(u32 roundIdx)
{
    subStreamReadInfo_.clear();
    UpdateSendRecvInfo(roundIdx, subStreamReadInfo_, partialCommRankSet_);
    CHK_RET(NotifySubStreamStart());
    CHK_RET(SendRecvData(roundIdx));
    if (!islocalCpyDone_) {
        CHK_RET(LocalCopy());
        islocalCpyDone_ = true;
    }
    CHK_RET(WaitSubStreamFinish());
    return HCCL_SUCCESS;
}

HcclResult
AlltoAllFullMeshSymmetricMemory::RunSDMATasks(u32 roundIdx, u32 groupRankSize, [[maybe_unused]] u32 leftRankSize)
{
    UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
    CHK_RET(RunGroupFullMeshAlltoall(roundIdx));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::RunSDMA(HcclOpMetaInfoDef& opMeta)
{
    // 计算每个rank分组fullmesh后需要通信的轮次，向上取整
    commRounds_ = (devNumInlocalPod_ + sdmaConcurrentNum_ - 1) / sdmaConcurrentNum_;
    u32 leftRankSize = devNumInlocalPod_ - 1; // leftRankSize中去掉本卡
    lastRoundIdx_
        = std::min((leftRankSize + sdmaConcurrentNum_ - 1) / sdmaConcurrentNum_, static_cast<u32>(commRounds_)) - 1;
    HCCL_DEBUG(
        "[AlltoAllFullMeshSymmetricMemory][RunSDMA] userRank [%u] communication rounds[%llu]"
        "post sync info: lastRoundIdx_[%u] devNumInlocalPod_[%u] sdmaConcurrentNum_[%u]",
        userRank_, commRounds_, lastRoundIdx_, devNumInlocalPod_, sdmaConcurrentNum_);

    u32 currentLeftRankSize = devNumInlocalPod_ - 1; // leftRankSize中去掉本卡
    for (u32 roundIdx = 0; roundIdx < commRounds_ && currentLeftRankSize > 0; roundIdx++) {
        CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
        u32 groupRankSize = (currentLeftRankSize > sdmaConcurrentNum_) ? sdmaConcurrentNum_ : currentLeftRankSize;
        CHK_RET(RunSDMATasks(roundIdx, groupRankSize, currentLeftRankSize));
        currentLeftRankSize -= groupRankSize;
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
    }

    HCCL_INFO("[AlltoAllFullMeshSymmetricMemory][RunSDMA] finished.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllFullMeshSymmetricMemory::RunAsync()
{
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, userInput_.size(), true);
    CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));

    if (userRankSize_ == 1) {
        HCCL_INFO("[AlltoAllFullMeshSymmetricMemory][RunAsync] do localcopy with 1 rank");
        CHK_RET(LocalCopy());
        return HCCL_SUCCESS;
    }

    if (devNumInlocalPod_ > 1) {
        CHK_RET(RunSDMA(opMeta));
    }

    HCCL_INFO("[AlltoAllFullMeshSymmetricMemory][RunAsync] finished.");
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_2_ALL_FULL_MESH_SYMMETRIC_MEMORY, AlltoAllFullMeshSymmetricMemory);
} // namespace hccl
