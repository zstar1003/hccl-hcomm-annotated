/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_direct_fullmesh.h"
#include "dispatcher_pub.h"

namespace hccl {
AlltoAllVDirectFullMesh::AlltoAllVDirectFullMesh(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AlltoAllVDirectFullMesh::~AlltoAllVDirectFullMesh() {}

HcclResult AlltoAllVDirectFullMesh::GenerateSubStreamInfo(
    const std::vector<Stream>& subStreams, const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain)
{
    u32 totalSubstreamSize
        = (totalRdmaRankNum_ > 0) ? (sdmaConcurrentNum_ + rdmaConcurrentNum_ + 1) : (sdmaConcurrentNum_);
    if (subStreams.size() < totalSubstreamSize || meshSignalMainToSub.size() < totalSubstreamSize
        || meshSignalSubToMain.size() < totalSubstreamSize) {
        HCCL_ERROR(
            "[AlltoAllVDirectFullMesh][GenerateSubStreamInfo]subStreamsSize[%zu], meshSignalMainToSubSize[%zu]"
            "meshSignalSubToMainSize[%zu] is smaller than totalSubstreamSize[%u]",
            subStreams.size(), meshSignalMainToSub.size(), meshSignalSubToMain.size(), totalSubstreamSize);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(
        links_.size() < userRankSize_,
        HCCL_ERROR(
            "[AlltoAllVDirectFullMesh][GenerateSubStreamInfo]"
            "links_.size()[%zu] is smaller than userRankSize_[%u].",
            links_.size(), userRankSize_),
        HCCL_E_PARA);
    HCCL_DEBUG(
        "subStreams.size[%zu], meshSignalMainToSub.size[%zu], links_.size[%zu]", subStreams.size(),
        meshSignalMainToSub.size(), links_.size());
    u32 index = 0;
    for (u32 sdmaIndex = 0; sdmaIndex < sdmaConcurrentNum_; sdmaIndex++) {
        sdmaSubStream_.push_back(subStreams[index]);
        sdmaMeshSignalMainToSub_.push_back(meshSignalMainToSub[index]);
        sdmaMeshSignalSubToMain_.push_back(meshSignalSubToMain[index]);
        index++;
    }
    for (u32 localIndex = 0; localIndex < sdmaConcurrentNum_; localIndex++) {
        localSubStream_.push_back(subStreams[index]);
        localSignalMainToSub_.push_back(meshSignalMainToSub[index]);
        localSignalSubToMain_.push_back(meshSignalSubToMain[index]);
        index++;
    }
    if (totalRdmaRankNum_ > 0) {
        rdmaSubStreams_.push_back(subStreams[index]);
        main2RdmaControlStreamNotify_ = meshSignalMainToSub[index];
        rdmaControl2MainStreamNotify_ = meshSignalSubToMain[index];
        index++;
        for (u32 rdmaIndex = 0; rdmaIndex < rdmaConcurrentNum_; rdmaIndex++) {
            rdmaSubStreams_.push_back(subStreams[index]);
            rdmaControl2SubNotifies_.push_back(meshSignalMainToSub[index]);
            rdmaSub2ControlNotifies_.push_back(meshSignalSubToMain[index]);
            index++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::Prepare(PrepareData& param)
{
    needAlltoallvCache_ = param.needAlltoallvCache;
    HCCL_INFO(
        "[AlltoAllVDirectFullMesh][Prepare] set needAlltoallvCache_[%u] for alltoallv aicpu cache",
        needAlltoallvCache_);

    mainStream_ = param.stream;
    userRank_ = param.userRank;
    userRankSize_ = param.userRankSize;
    links_ = *param.linksPtr;
    localSendRecvInfoPtr_ = param.localSendRecvInfoPtr;
    devNumInlocalPod_ = param.devNumInlocalPod;
    rankIdxInPod_ = param.rankIdxInPod;
    opType_ = param.opType;
    algOpContext_ = param.algOpContext;

    podStartRank_ = userRank_ - rankIdxInPod_;
    podEndRank_ = podStartRank_ + devNumInlocalPod_ - 1;
    sdmaConcurrentNum_ = (devNumInlocalPod_ > ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) ?
                             (ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) :
                             (devNumInlocalPod_);

    totalRdmaRankNum_ = userRankSize_ - devNumInlocalPod_;
    rdmaConcurrentNum_ = (totalRdmaRankNum_ > ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) ?
                             (ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) :
                             (totalRdmaRankNum_);
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh]devNumInlocalPod_[%u], userRankSize_[%u] podStartRank_[%u]"
        "podEndRank_[%u], totalRdmaRankNum_[%u], sdmaConcurrentNum_[%u], rdmaConcurrentNum_[%u]",
        devNumInlocalPod_, userRankSize_, podStartRank_, podEndRank_, totalRdmaRankNum_, sdmaConcurrentNum_,
        rdmaConcurrentNum_);

    CHK_PRT_RET(
        userRankSize_ == 0, HCCL_ERROR("[AlltoAllVDirectFullMesh][Prepare]userRankSize_ is zero."), HCCL_E_PARA);

    userInput_ = param.inputMem;
    userOutput_ = param.outputMem;
    cclInMem_ = param.cclInMem;
    cclOutMem_ = param.cclOutMem;
    workMode_ = param.workMode;
    isSuPodAsym_ = param.isSuPodAsym;

    // 注意: 如果isBigCount的计算逻辑发生变化, 需要同步修改IsBigCountForAlltoallv()中的代码
    u64 maxSendLen = CalcMaxSendLen();
    isBigCount_ = (maxSendLen > ALLTOALLV_DIRECT_FULLMESH_BIG_SIZE) ? true : false;
    CHK_RET(GenerateSubStreamInfo(*param.subStreamsPtr, *param.signalPtr, *param.signalAuxPtr));

    if (algOpContext_.mc2Handler.stepSize > 0) {
        sdmaConcurrentNum_ = (devNumInlocalPod_ > 1) ? 1 : (devNumInlocalPod_);
        // MC2细粒度不需要本地并发处理
        isBigCount_ = false;
    }

    /* 考虑当group0 的rank 跟 group 1的所有rank通信时，每次都要收发，所以取sdmaConcurrentNum_块；
    跟group 0内的rank通信有一块儿浪费 */
    // 注意: 如果sdmaDataBlockSize_的计算逻辑发生变化,
    // 需要同步修改framework下CalcMetadataForFirstAlltoallv()函数中的part 1
    u32 blockGroup
        = (isBigCount_ || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLV || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLVC) ?
              2 :
              1;
    sdmaDataBlockSize_ = (cclInMem_.size() / std::max(1u, sdmaConcurrentNum_ * blockGroup));
    // 向下对齐到16k Byte
    if (sdmaDataBlockSize_ > HCCL_MIN_SLICE_ALIGN_910B) {
        sdmaDataBlockSize_ = (sdmaDataBlockSize_ / HCCL_MIN_SLICE_ALIGN_910B) * HCCL_MIN_SLICE_ALIGN_910B;
    }
    CHK_PRT_RET(
        sdmaDataBlockSize_ == 0, HCCL_ERROR("[AlltoAllVDirectFullMesh][Prepare]sdmaDataBlockSize_is zero."),
        HCCL_E_INTERNAL);
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][Prepare] userRank [%u] total cclsize[%llu],"
        "sdmaDataBlockSize_[%llu], BigCountFlag[%d], stepSize[%u]",
        userRank_, cclInMem_.size(), sdmaDataBlockSize_, isBigCount_, algOpContext_.mc2Handler.stepSize);

    // 一半的CCLOut用来发送RDMA数据，另一半用来接收RDMA数据，因此需要除以2
    rdmaDataBlockSize_ = cclOutMem_.size() / std::max(1u, rdmaConcurrentNum_) / 2;

    return HCCL_SUCCESS;
}

std::string AlltoAllVDirectFullMesh::GetStreamIndexString()
{
    std::string res = "";
    for (auto& info : subStreamReadInfo_) {
        u32 destRank = info.first;
        u32 streamIndex = destRank % sdmaConcurrentNum_;
        res += std::to_string(streamIndex) + ", ";
    }
    return res;
}

u64 AlltoAllVDirectFullMesh::CalcMaxSendLen()
{
    u64 maxSendLen = 0;
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;

    for (u32 dstRank = 0; dstRank < localSendRecvInfo.sendLength.size(); dstRank++) {
        maxSendLen = std::max(maxSendLen, localSendRecvInfo.sendLength[dstRank]);
    }

    HCCL_DEBUG("[AlltoAllVDirectFullMesh][CalcMaxSendLen] maxSendLen[%llu]", maxSendLen);
    return maxSendLen;
}

HcclResult AlltoAllVDirectFullMesh::UpdateCurrRankRecvInfo(
    u32 step, u32 roundIdx, u32 side, u32 destRank, std::vector<ReadDataBlock>& readInfo,
    std::unordered_map<u32, ReadDataBlock>& subStreamZcopyReadInfo, u32 maxRecvStep)
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    u64 remainRecvLen = localSendRecvInfo.recvLength[destRank];
    u64 scratchOffset = 0;
    u32 bufferIdx = 0;
    u32 pairNum = sdmaConcurrentNum_ / RANK_SET_COMPUTE_CONST;
    if (sdmaConcurrentNum_ == 1) { // 保证和当前rank距离一样时，send/recv用的是同一块buff
        bufferIdx = 0;
    } else if (side == 0) { // 在curRank左边
        u32 gap = (userRank_ - destRank + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - (gap - roundIdx * pairNum);
    } else if (side == 1) { // 在curRank右边
        u32 gap = (destRank - userRank_ + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - 1 + (gap - roundIdx * pairNum);
    } else { // 最后一个中间位置的rank
        bufferIdx = 0;
    }

    if ((isBigCount_ || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLV || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLVC)
        && (roundIdx % RANK_SET_COMPUTE_CONST != 0)) { // 奇数轮，用下半Buffer
        bufferIdx += sdmaConcurrentNum_;
    }

    scratchOffset = bufferIdx * sdmaDataBlockSize_;

    u32 recvStepIdx = 0;
    u64 dataOffset = 0;
    HCCL_DEBUG(
        "step[%u] round[%u] usrRank[%u] total recv localSendRecvInfo.recvLength[%llu] from dstRank[%u] bufferIdx[%u]",
        step, roundIdx, userRank_, remainRecvLen, destRank, bufferIdx);

    // alltoallv类算子的零长拷贝, 需要调用MemcpyAsync保证aicpu cache使能时placeholder正确下发
    // (cache不使能时为空函数调用)
    if (needAlltoallvCache_ && remainRecvLen == 0) {
        // 获取local user output offset
        const u64 recvLen = 0;
        u64 userOutOffset = localSendRecvInfo.recvOffset[destRank];
        HCCL_DEBUG(
            "[AlltoAllVDirectFullMesh][UpdateCurrRankRecvInfo] usrRank[%u] recv from destRank [%u] "
            "recvStepIdx[%u] recvLen[%lu] userOutOffset[%llu] scratchOffset[%llu]",
            userRank_, destRank, recvStepIdx, recvLen, userOutOffset, scratchOffset);

        // 更新零长拷贝的read info
        ReadDataBlock readBlock = {recvLen, scratchOffset, userOutOffset};
        subStreamZcopyReadInfo[destRank] = readBlock;

        // sendCount为0, step和readInfo.size一定为0
        CHK_PRT_RET(
            maxRecvStep > 0,
            HCCL_ERROR(
                "[AlltoAllVDirectFullMesh][UpdateCurrRankRecvInfo] maxRecvStep[%u] != 0 for remainRecvLen[%llu]",
                maxRecvStep, remainRecvLen),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            readInfo.size() != 0,
            HCCL_ERROR("[AlltoAllVDirectFullMesh][UpdateCurrRankRecvInfo] invalid readInfo.size[%u]", readInfo.size()),
            HCCL_E_INTERNAL);
    } else {
        while (recvStepIdx < maxRecvStep && remainRecvLen > 0) {
            u64 currDataRemainLen = localSendRecvInfo.recvLength[destRank] - dataOffset;
            u64 recvLen = std::min(sdmaDataBlockSize_, currDataRemainLen);
            u64 userOutOffset = localSendRecvInfo.recvOffset[destRank] + dataOffset;
            HCCL_DEBUG(
                "[AlltoAllVDirectFullMesh][UpdateCurrRankRecvInfo] usrRank[%u] recv from destRank [%u] "
                "recvStepIdx[%u] recvLen[%lu] userOutOffset[%llu] scratchOffset[%llu]",
                userRank_, destRank, recvStepIdx, recvLen, userOutOffset, scratchOffset);
            readInfo.push_back({recvLen, scratchOffset, userOutOffset});
            dataOffset += recvLen;
            recvStepIdx++;
            remainRecvLen -= recvLen;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::UpdateCurrRankSendInfo(
    u32 step, u32 roundIdx, u32 side, u32 destRank, std::vector<SendDataBlock>& sendInfo,
    std::unordered_map<u32, SendDataBlock>& subStreamZcopySendInfo, u32 maxSendStep)
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    u64 remainSendLen = localSendRecvInfo.sendLength[destRank];

    u64 scratchOffset = 0;
    u32 bufferIdx = 0;
    u32 pairNum = sdmaConcurrentNum_ / RANK_SET_COMPUTE_CONST;
    if (sdmaConcurrentNum_ == 1) { // 保证和当前rank距离一样时，send/recv用的是同一块buff
        bufferIdx = 0;
    } else if (side == 0) { // 在curRank左边
        u32 gap = (userRank_ - destRank + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - 1 + (gap - roundIdx * pairNum);
    } else if (side == 1) { // 在curRank右边
        u32 gap = (destRank - userRank_ + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - (gap - roundIdx * pairNum);
    } else { // 最后一个中间位置的rank
        bufferIdx = 0;
    }

    if ((isBigCount_ || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLV || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLVC)
        && (roundIdx % RANK_SET_COMPUTE_CONST != 0)) { // 奇数轮，用下半Buffer
        bufferIdx += sdmaConcurrentNum_;
    }
    scratchOffset = bufferIdx * sdmaDataBlockSize_;

    // 更新hcclOffset到dstRank的映射, 用于alltoallv算子aicpu展开的SQE缓存
    if (needAlltoallvCache_) {
        // alltoallv cache只针对小数据量, 至多只有1个step
        CHK_PRT_RET(
            step != 0,
            HCCL_ERROR(
                "[AlltoAllVDirectFullMesh][UpdateCurrRankRecvInfo] needAlltoallvCache_[%u] step[%u]",
                needAlltoallvCache_, step),
            HCCL_E_INTERNAL);

        std::unordered_map<uint64_t, std::vector<uint32_t>>::iterator mapIter
            = hcclOffsetDstRanksMap_.find(scratchOffset);
        if (mapIter == hcclOffsetDstRanksMap_.end()) {
            constexpr uint32_t singleRankVecSize = 1;
            std::pair<std::unordered_map<uint64_t, std::vector<uint32_t>>::iterator, bool> emplaceResult
                = hcclOffsetDstRanksMap_.emplace(scratchOffset, std::vector<uint32_t>(singleRankVecSize, destRank));
            CHK_PRT_RET(
                !emplaceResult.second,
                HCCL_ERROR(
                    "[AlltoAllVDirectFullMesh][UpdateCurrRankSendInfo] fail to insert hcclOffset[%llu]-dstRank[%u] "
                    "pair",
                    scratchOffset, destRank),
                HCCL_E_INTERNAL);
            mapIter = emplaceResult.first;
        } else {
            // 虽然同一个dstRank不需要重复计算sendInfo, 但不同dstRanks在multi-round case下可能对应相同的hcclOffset
            CHK_PRT_RET(
                mapIter->second.size() == 0,
                HCCL_ERROR(
                    "[AlltoAllVDirectFullMesh][UpdateCurrRankSendInfo] empty dstRanks for hcclOffset[%llu] before add "
                    "destRank[%u]",
                    mapIter->second, destRank, scratchOffset),
                HCCL_E_INTERNAL);
            mapIter->second.push_back(destRank);
        }
        HCCL_DEBUG(
            "[AlltoAllVDirectFullMesh][UpdateCurrRankSendInfo] mapIter->first[%llu] mapIter->second.size[%u] "
            "destRank[%u]",
            mapIter->first, mapIter->second.size(), destRank);
    }

    u32 sendStepIdx = 0;
    u64 dataOffset = 0;
    HCCL_DEBUG(
        "step[%u] round[%u] usrRank[%u] total send localSendRecvInfo.sendLength[%llu] to dstRank[%u] bufferIdx[%u]",
        step, roundIdx, userRank_, remainSendLen, destRank, bufferIdx);

    if (needAlltoallvCache_ && remainSendLen == 0) { // alltoallv类算子的零长拷贝, 需要调用MemcpyAsync保证aicpu
                                                     // cache使能时placeholder正确下发 (cache不使能时为空函数调用)
        // 获取local user input offset
        const u64 sendLen = 0;
        u64 userInOffset = localSendRecvInfo.sendOffset[destRank];
        HCCL_DEBUG(
            "[AlltoAllVDirectFullMesh][UpdateCurrRankSendInfo] usrRank[%u] send to destRank [%u]"
            " sendStepIdx[%u] sendLen[%lu] userInOffset[%llu] scratchOffset[%llu]",
            userRank_, destRank, sendStepIdx, sendLen, userInOffset, scratchOffset);

        // 更新零长拷贝的send info
        SendDataBlock sendBlock = {sendLen, userInOffset, scratchOffset};
        subStreamZcopySendInfo[destRank] = sendBlock;

        // sendCount为0, step和sendInfo.size一定为0
        CHK_PRT_RET(
            maxSendStep > 0,
            HCCL_ERROR(
                "[AlltoAllVDirectFullMesh][UpdateCurrRankSendInfo] maxSendStep[%u] != 0 for remainSendLen[%llu]",
                maxSendStep, remainSendLen),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            sendInfo.size() != 0,
            HCCL_ERROR("[AlltoAllVDirectFullMesh][UpdateCurrRankSendInfo] invalid sendInfo.size[%u]", sendInfo.size()),
            HCCL_E_INTERNAL);
    } else {
        while (sendStepIdx < maxSendStep && remainSendLen > 0) {
            u64 currDataRemainLen = localSendRecvInfo.sendLength[destRank] - dataOffset;
            u64 sendLen = std::min(sdmaDataBlockSize_, currDataRemainLen);
            u64 userInOffset = localSendRecvInfo.sendOffset[destRank] + dataOffset;
            HCCL_DEBUG(
                "[AlltoAllVDirectFullMesh][UpdateCurrRankSendInfo] usrRank[%u] send to destRank [%u]"
                " sendStepIdx[%u] sendLen[%lu] userInOffset[%llu] scratchOffset[%llu]",
                userRank_, destRank, sendStepIdx, sendLen, userInOffset, scratchOffset);
            sendInfo.push_back({sendLen, userInOffset, scratchOffset});
            dataOffset += sendLen;
            sendStepIdx++;
            remainSendLen -= sendLen;
        }
    }

    return HCCL_SUCCESS;
}

void AlltoAllVDirectFullMesh::UpdateSendRecvInfo(
    u32 step, u32 roundIdx, std::unordered_map<u32, std::vector<ReadDataBlock>>& subStreamReadInfo,
    std::unordered_map<u32, std::vector<SendDataBlock>>& subStreamSendInfo,
    std::unordered_map<u32, ReadDataBlock>& subStreamZcopyReadInfo,
    std::unordered_map<u32, SendDataBlock>& subStreamZcopySendInfo,
    const std::vector<std::vector<std::pair<u32, u32>>>& partialCommRankSet)
{
    for (u32 side = 0; side < partialCommRankSet.size(); side++) {
        for (u32 j = 0; j < partialCommRankSet[side].size(); j++) {
            u32 readRemoteRank = partialCommRankSet[side][j].first;
            if (readRemoteRank == userRank_) {
                continue;
            }
            u32 currDestRecvStep = recvNumSubStep_[readRemoteRank];
            std::vector<ReadDataBlock> readInfo;
            UpdateCurrRankRecvInfo(
                step, roundIdx, side, readRemoteRank, readInfo, subStreamZcopyReadInfo, currDestRecvStep);

            subStreamReadInfo[readRemoteRank] = readInfo;
        }
    }

    for (u32 side = 0; side < partialCommRankSet.size(); side++) {
        for (u32 j = 0; j < partialCommRankSet[side].size(); j++) {
            u32 sendRemoteRank = partialCommRankSet[side][j].second;
            if (sendRemoteRank == userRank_) {
                continue;
            }
            u32 currDestSendStep = sendNumSubStep_[sendRemoteRank];
            std::vector<SendDataBlock> sendInfo;
            UpdateCurrRankSendInfo(
                step, roundIdx, side, sendRemoteRank, sendInfo, subStreamZcopySendInfo, currDestSendStep);

            subStreamSendInfo[sendRemoteRank] = sendInfo;
        }
    }
}

void AlltoAllVDirectFullMesh::UpdateOpBaseSubStreamInfo(u32 step, u32 roundIdx)
{
    if (roundIdx == 0 || !isBigCount_) {
        subStreamReadInfo_.clear();
        subStreamSendInfo_.clear();
        if (needAlltoallvCache_) {
            subStreamZcopyReadInfo_.clear();
            subStreamZcopySendInfo_.clear();
        }
        UpdateSendRecvInfo(
            step, roundIdx, subStreamReadInfo_, subStreamSendInfo_, subStreamZcopyReadInfo_, subStreamZcopySendInfo_,
            partialCommRankSet_);
    }
    if (isBigCount_ && (roundIdx < commRounds_ - 1)) {
        nextSubStreamReadInfo_.clear();
        nextSubStreamSendInfo_.clear();
        if (needAlltoallvCache_) {
            nextSubStreamZcopyReadInfo_.clear();
            nextSubStreamZcopySendInfo_.clear();
        }
        UpdateSendRecvInfo(
            step, roundIdx + 1, nextSubStreamReadInfo_, nextSubStreamSendInfo_, nextSubStreamZcopyReadInfo_,
            nextSubStreamZcopySendInfo_, nextPartialCommRankSet_);
    }
}

HcclResult AlltoAllVDirectFullMesh::PrepareIntraData(
    u32 step, std::unordered_map<u32, std::vector<SendDataBlock>>& subStreamSendInfo,
    std::unordered_map<u32, SendDataBlock>& subStreamZcopySendInfo)
{
    u32 sendDataIndex = 0;
    for (auto& sdmaInfo : subStreamSendInfo) {
        const std::vector<SendDataBlock>& sendInfo = sdmaInfo.second;

        // 对于alltoallv类算子, 零长拷贝需要调用MemcpyAsync保证aicpu cache使能时placeholder正确下发
        // 注意: alltoallv aicpu cache只考虑小数据量 (即max step为1), 所以只需要在step 0时下发一个placeholder SQE即可
        if (needAlltoallvCache_) {
            // alltoallv cache只针对小数据量, 至多只有1个step
            CHK_PRT_RET(
                step != 0,
                HCCL_ERROR(
                    "[AlltoAllVDirectFullMesh][UpdateCurrRankRecvInfo] needAlltoallvCache_[%u] step[%u]",
                    needAlltoallvCache_, step),
                HCCL_E_INTERNAL);

            const u32 sendRank = sdmaInfo.first;
            std::unordered_map<u32, SendDataBlock>::const_iterator mapIter = subStreamZcopySendInfo.find(sendRank);
            if (mapIter != subStreamZcopySendInfo.end()) { // sendRank的sendCount为0
                // 零长拷贝下, sendRank对应的step和sendInfo.size一定为0
                CHK_PRT_RET(
                    sendNumSubStep_[sdmaInfo.first] > 0,
                    HCCL_ERROR("invalid sendNumSubStep_[%u][%u] != 0", sdmaInfo.first, sendNumSubStep_[sdmaInfo.first]),
                    HCCL_E_INTERNAL);
                CHK_PRT_RET(
                    sendInfo.size() > 0,
                    HCCL_ERROR(
                        "[AlltoAllVDirectFullMesh][PrepareIntraData] invalid sendInfo.size[%u] != 0", sendInfo.size()),
                    HCCL_E_INTERNAL);

                // 获取零长拷贝的发送偏移
                const SendDataBlock& sendBlock = mapIter->second;
                CHK_PRT_RET(
                    sendBlock.sendLen != 0,
                    HCCL_ERROR(
                        "[AlltoAllVDirectFullMesh][PrepareIntraData] invalid sendBlock.sendLen[%llu] != 0",
                        sendBlock.sendLen),
                    HCCL_E_INTERNAL);

                // 强制调用HcclD2DMemcpyAsync下发cache-memcpy placeholder (aicpu cache使能时才会生效,
                // 未使能时会直接返回)
                DeviceMem src = userInput_.range(sendBlock.userInOffset, sendBlock.sendLen);
                DeviceMem dst = cclInMem_.range(sendBlock.scratchOffset, sendBlock.sendLen);
                HCCL_DEBUG(
                    "[AlltoAllVDirectFullMesh][PrepareIntraData]userRank [%u] copy from userInOffset[%llu]"
                    "len[%u] to scratchOffset [%llu]",
                    userRank_, sendBlock.userInOffset, sendBlock.sendLen, sendBlock.scratchOffset);
                reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(true);
                HCCL_INFO(
                    "[AlltoAllVDirectFullMesh][PrepareIntraData] generate cache-memcpy placeholder for sendRank[%u]",
                    sendRank);
                if (isBigCount_) {
                    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, localSubStream_[sendDataIndex]));
                } else {
                    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
                }
                reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(false);
            }
        }

        if (step < sendNumSubStep_[sdmaInfo.first]) {
            DeviceMem src = userInput_.range(sendInfo[step].userInOffset, sendInfo[step].sendLen);
            DeviceMem dst = cclInMem_.range(sendInfo[step].scratchOffset, sendInfo[step].sendLen);
            HCCL_DEBUG(
                "[AlltoAllVDirectFullMesh][PrepareIntraData]userRank [%u] copy from userInOffset[%llu]"
                "len[%u] to scratchOffset [%llu]",
                userRank_, sendInfo[step].userInOffset, sendInfo[step].sendLen, sendInfo[step].scratchOffset);
            if (isBigCount_) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, localSubStream_[sendDataIndex]));
            } else {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
            }
        }
        sendDataIndex++;
    }
    return HCCL_SUCCESS;
}

void AlltoAllVDirectFullMesh::UpdateRemoteRankSet(u32 roundIdx, u32 groupRankSize)
{
    if (sdmaConcurrentNum_ == 1) {
        UpdatePartialCommunicationRankSetPairWise(roundIdx, groupRankSize);
    } else {
        UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
    }
}

void AlltoAllVDirectFullMesh::UpdatePartialCommunicationRankSetPairWise(u32 roundIdx, u32 groupRankSize)
{
    partialCommRankSet_.clear();
    partialCommRankSet_.resize(1);
    for (u32 i = roundIdx * sdmaConcurrentNum_; i < (roundIdx * sdmaConcurrentNum_ + groupRankSize); i++) {
        u32 readRemoteRank = podStartRank_ + (rankIdxInPod_ + devNumInlocalPod_ - i) % devNumInlocalPod_;
        u32 sendRemoteRank = podStartRank_ + (rankIdxInPod_ + i) % devNumInlocalPod_;
        partialCommRankSet_[0].push_back(std::make_pair(readRemoteRank, sendRemoteRank));
        HCCL_DEBUG(
            "[AlltoAllVDirectFullMesh][UpdatePartialCommunicationRankSetPairWise] userRank [%u] i[%u]"
            "readRemoteRank[%u] writeRemoteRank[%u]",
            userRank_, i, readRemoteRank, sendRemoteRank);
    }
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][UpdatePartialCommunicationRankSetPairWise] partialCommRankSet_ size[%zu]",
        partialCommRankSet_[0].size());
}

void AlltoAllVDirectFullMesh::UpdatePartialCommunicationRankSet(
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
            "[AlltoAllVDirectFullMesh][UpdatePartialCommunicationRankSet] round[%u] userRank [%u] i[%u]"
            "read/write leftRemoteRank[%u] rightRemoteRank[%u]",
            roundIdx, userRank_, i, leftRemoteRank, rightRemoteRank);
    }
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][UpdatePartialCommunicationRankSet] round[%u] partialCommRankSet_ total size[%zu]",
        roundIdx, partialCommRankSet[0].size() + partialCommRankSet[1].size() + partialCommRankSet[2].size());
}

// 主流只需要通知当前子步骤需要收发数据的 SDMA 流，减少同步开销
HcclResult AlltoAllVDirectFullMesh::NotifySubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(
            LocalNotify::Post(mainStream_, dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            sdmaSubStream_[streamIndex], dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, sdmaSubStream_[streamIndex], dispatcher_));
    }
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][NotifySubStreamStart] userRank [%u] main stream notify sdma stream [%s]", userRank_,
        GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::WaitSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            sdmaSubStream_[streamIndex], dispatcher_, sdmaMeshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(
            LocalNotify::Wait(mainStream_, dispatcher_, sdmaMeshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][WaitSubStreamFinish] userRank [%u] main stream wait sdma stream [%s]", userRank_,
        GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::NotifyLocalSubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < subStreamSendInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, localSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            localSubStream_[streamIndex], dispatcher_, localSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::WaitLocalSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < subStreamSendInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            localSubStream_[streamIndex], dispatcher_, localSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, localSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

u32 AlltoAllVDirectFullMesh::CalcNumSubStep()
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;

    sendNumSubStep_.clear();
    recvNumSubStep_.clear();
    u32 numSubStep = 0;

    for (u32 destRank = podStartRank_; destRank < podStartRank_ + devNumInlocalPod_; destRank++) {
        if (destRank == userRank_) {
            continue;
        }

        u32 currRankSendSubStep
            = ((localSendRecvInfo.sendLength[destRank] + sdmaDataBlockSize_ - 1) / sdmaDataBlockSize_);
        sendNumSubStep_[destRank] = currRankSendSubStep;

        u32 currRankRecvSubStep
            = ((localSendRecvInfo.recvLength[destRank] + sdmaDataBlockSize_ - 1) / sdmaDataBlockSize_);
        recvNumSubStep_[destRank] = currRankRecvSubStep;
        HCCL_DEBUG(
            "[AlltoAllVDirectFullMesh][CalcNumSubStep] userRank [%u] currRankSendSubStep[%u]"
            "currRankRecvSubStep[%u]",
            userRank_, currRankSendSubStep, currRankRecvSubStep);
        numSubStep = std::max(numSubStep, std::max(currRankSendSubStep, currRankRecvSubStep));
    }
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][CalcNumSubStep] userRank [%u] max communication step[%u]", userRank_, numSubStep);
    return numSubStep;
}

HcclResult AlltoAllVDirectFullMesh::NotifyRemoteRankStart(u32 step)
{
    u32 streamIndex = 0;
    for (auto& sendRecvSide : partialCommRankSet_) {
        for (auto& sendRecvPair : sendRecvSide) {
            u32 recvRank = sendRecvPair.first;
            u32 sendRank = sendRecvPair.second;
            if (sendRank == userRank_) {
                continue;
            }
            const std::vector<ReadDataBlock>& readInfo = subStreamReadInfo_[recvRank];
            const std::vector<SendDataBlock>& sendInfo = subStreamSendInfo_[sendRank];
            Stream& currStream = sdmaSubStream_[streamIndex];
            const LINK& readTransport = links_[recvRank];
            const LINK& sendTransport = links_[sendRank];

            if (needAlltoallvCache_) {
                // alltoallv cache只针对小数据量, 至多只有1个step
                CHK_PRT_RET(
                    step != 0,
                    HCCL_ERROR(
                        "[AlltoAllVDirectFullMesh][NotifyRemoteRankStart] needAlltoallvCache_[%u] step[%u]",
                        needAlltoallvCache_, step),
                    HCCL_E_INTERNAL);

                std::unordered_map<u32, SendDataBlock>::const_iterator mapIter = subStreamZcopySendInfo_.find(sendRank);
                if (mapIter != subStreamZcopySendInfo_.end()) { // sendRank的sendCount为0
                    // 零长拷贝下, sendRank对应的sendInfo.size一定为0
                    CHK_PRT_RET(
                        sendInfo.size() > 0,
                        HCCL_ERROR(
                            "[AlltoAllVDirectFullMesh][NotifyRemoteRankStart] invalid sendInfo.size[%u] != 0",
                            sendInfo.size()),
                        HCCL_E_INTERNAL);

                    // 生成cache-write placeholder
                    reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(true);
                    HCCL_INFO(
                        "[AlltoAllVDirectFullMesh][NotifyRemoteRankStart] generate cache-write placeholder for "
                        "sendRank[%u]",
                        sendRank);
                    CHK_RET(sendTransport->TxAck(currStream));
                    reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(false);
                }
            }
            if (step < sendInfo.size()) {
                CHK_RET(sendTransport->TxAck(currStream));
            }

            if (needAlltoallvCache_) {
                std::unordered_map<u32, ReadDataBlock>::const_iterator mapIter = subStreamZcopyReadInfo_.find(recvRank);
                if (mapIter != subStreamZcopyReadInfo_.end()) { // recvRank的recvCount为0
                    // 零长拷贝下, recvRank对应的readInfo.size一定为0
                    CHK_PRT_RET(
                        readInfo.size() > 0,
                        HCCL_ERROR(
                            "[AlltoAllVDirectFullMesh][NotifyRemoteRankStart] invalid readInfo.size[%u] != 0",
                            readInfo.size()),
                        HCCL_E_INTERNAL);

                    // 生成cache-write placeholder
                    reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(true);
                    HCCL_INFO(
                        "[AlltoAllVDirectFullMesh][NotifyRemoteRankStart] generate cache-notify placeholder for "
                        "recvRank[%u]",
                        recvRank);
                    CHK_RET(readTransport->RxAck(currStream));
                    reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(false);
                }
            }
            if (step < readInfo.size()) {
                CHK_RET(readTransport->RxAck(currStream));
            }
            streamIndex++;
        }
    }
    HCCL_INFO("[AlltoAllVDirectFullMesh][NotifyRemoteRankStart] done");
    return HCCL_SUCCESS;
}

bool AlltoAllVDirectFullMesh::IsPostSyncEnable(u32 step, u32 roundIdx)
{
    bool isPostSyncEnable = false;
    isPostSyncEnable = (step == lastStep_) && (roundIdx == lastRoundIdx_) && algOpContext_.opRetryHandler.retryEnable;
    return isPostSyncEnable;
}

HcclResult AlltoAllVDirectFullMesh::SdmaMainStreamWait(u32 step, u32 roundIdx)
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
            const std::vector<ReadDataBlock>& readInfo = subStreamReadInfo_[recvRank];

            if (needAlltoallvCache_) {
                // alltoallv cache只针对小数据量, 至多只有1个step
                CHK_PRT_RET(
                    step != 0,
                    HCCL_ERROR(
                        "[AlltoAllVDirectFullMesh][SdmaMainStreamWait] needAlltoallvCache_[%u] step[%u]",
                        needAlltoallvCache_, step),
                    HCCL_E_INTERNAL);

                std::unordered_map<u32, ReadDataBlock>::const_iterator mapIter = subStreamZcopyReadInfo_.find(recvRank);
                if (mapIter != subStreamZcopyReadInfo_.end()) { // recvRank的recvCount为0
                    // 零长拷贝下, recvRank对应的readInfo.size一定为0
                    CHK_PRT_RET(
                        readInfo.size() > 0,
                        HCCL_ERROR(
                            "[AlltoAllVDirectFullMesh][SdmaMainStreamWait] invalid readInfo.size[%u] != 0",
                            readInfo.size()),
                        HCCL_E_INTERNAL);

                    // 正常下NotifyWait SQE (本地主从流同步, 由于从流不存在跨卡数据搬运, 主流wait后会立刻wake up)
                    HCCL_DEBUG(
                        "[AlltoAllVDirectFullMesh][SdmaMainStreamWait] userRank [%u], recvRank[%u], "
                        "sendRank[%u], sdma stream [%u], "
                        "post sync info: step[%u], roundIdx[%u], lastStep_[%u], lastRoundIdx_[%u] main stream wait",
                        userRank_, recvRank, sendRank, streamIndex, step, roundIdx, lastStep_, lastRoundIdx_);
                    CHK_RET(LocalNotify::Wait(
                        mainStream_, dispatcher_, sdmaMeshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
                }
            }

            if (step < readInfo.size()) {
                HCCL_DEBUG(
                    "[AlltoAllVDirectFullMesh][SdmaMainStreamWait] userRank [%u], recvRank[%u], "
                    "sendRank[%u], sdma stream [%u], "
                    "post sync info: step[%u], roundIdx[%u], lastStep_[%u], lastRoundIdx_[%u] main stream wait",
                    userRank_, recvRank, sendRank, streamIndex, step, roundIdx, lastStep_, lastRoundIdx_);
                CHK_RET(LocalNotify::Wait(
                    mainStream_, dispatcher_, sdmaMeshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
            }
            streamIndex++;
        }
    }
    HCCL_INFO("[AlltoAllVDirectFullMesh][SdmaMainStreamWait] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::SdmaMainStreamPost(u32 step, u32 roundIdx)
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
            const std::vector<ReadDataBlock>& readInfo = subStreamReadInfo_[recvRank];

            if (needAlltoallvCache_) {
                // alltoallv cache只针对小数据量, 至多只有1个step
                CHK_PRT_RET(
                    step != 0,
                    HCCL_ERROR(
                        "[AlltoAllVDirectFullMesh][SdmaMainStreamWait] needAlltoallvCache_[%u] step[%u]",
                        needAlltoallvCache_, step),
                    HCCL_E_INTERNAL);

                std::unordered_map<u32, ReadDataBlock>::const_iterator mapIter = subStreamZcopyReadInfo_.find(recvRank);
                if (mapIter != subStreamZcopyReadInfo_.end()) { // recvRank的recvCount为0
                    // 零长拷贝下, recvRank对应的readInfo.size一定为0
                    CHK_PRT_RET(
                        readInfo.size() > 0,
                        HCCL_ERROR(
                            "[AlltoAllVDirectFullMesh][SdmaMainStreamWait] invalid readInfo.size[%u] != 0",
                            readInfo.size()),
                        HCCL_E_INTERNAL);

                    // 正常下NotifyRecord SQE
                    HCCL_DEBUG(
                        "[AlltoAllVDirectFullMesh][SdmaMainStreamPost] userRank [%u], recvRank[%u], "
                        "sendRank[%u], sdma stream [%u], "
                        "post sync info: step[%u], roundIdx[%u], lastStep_[%u], lastRoundIdx_[%u] main stream post",
                        userRank_, recvRank, sendRank, streamIndex, step, roundIdx, lastStep_, lastRoundIdx_);
                    CHK_RET(LocalNotify::Post(
                        mainStream_, dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
                }
            }

            if (step < readInfo.size()) {
                HCCL_DEBUG(
                    "[AlltoAllVDirectFullMesh][SdmaMainStreamPost] userRank [%u], recvRank[%u], "
                    "sendRank[%u], sdma stream [%u], "
                    "post sync info: step[%u], roundIdx[%u], lastStep_[%u], lastRoundIdx_[%u] main stream post",
                    userRank_, recvRank, sendRank, streamIndex, step, roundIdx, lastStep_, lastRoundIdx_);
                CHK_RET(LocalNotify::Post(
                    mainStream_, dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
            }
            streamIndex++;
        }
    }
    HCCL_INFO("[AlltoAllVDirectFullMesh][SdmaMainStreamPost] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::SetPostSyncTasks(u32 step, u32 roundIdx)
{
    // SDMA wait
    CHK_RET(SdmaMainStreamWait(step, roundIdx));
    if (rdmaConcurrentNum_ > 0) {
        // RDMA wait
        HCCL_DEBUG("[AlltoAllVDirectFullMesh][SetPostSyncTasks] rdma post sync info: main stream wait");
        CHK_RET(RdmaControlNotifyMainFinish());
    }
    // SDMA post
    CHK_RET(SdmaMainStreamPost(step, roundIdx));
    if (rdmaConcurrentNum_ > 0) {
        // RDMA post
        HCCL_DEBUG("[AlltoAllVDirectFullMesh][SetPostSyncTasks] rdma post sync info: main stream post");
        CHK_RET(MainNotifyRdmaControlStart());
    }
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][SetPostSyncTasks] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::SDMAwithRemoteRankAndNotifyEnd(u32 step, u32 roundIdx)
{
    bool isPostSyncEnable = IsPostSyncEnable(step, roundIdx);
    if (isPostSyncEnable) {
        // 下发主流上的后同步wait和post
        CHK_RET(SetPostSyncTasks(step, roundIdx));
    }
    u32 streamIndex = 0;
    for (auto& sendRecvSide : partialCommRankSet_) {
        for (auto& sendRecvPair : sendRecvSide) {
            u32 recvRank = sendRecvPair.first;
            u32 sendRank = sendRecvPair.second;
            if (sendRank == userRank_) {
                continue;
            }
            const std::vector<ReadDataBlock>& readInfo = subStreamReadInfo_[recvRank];
            const std::vector<SendDataBlock>& sendInfo = subStreamSendInfo_[sendRank];
            Stream& currStream = sdmaSubStream_[streamIndex];
            const LINK& readTransport = links_[recvRank];
            const LINK& sendTransport = links_[sendRank];

            // 对于alltoallv类算子, 零长拷贝需要调用MemcpyAsync保证aicpu cache使能时placeholder正确下发
            // 注意: alltoallv aicpu cache只考虑小数据量 (即max step为1), 所以只需要在step 0时下发一个placeholder
            // SQE即可
            if (needAlltoallvCache_) {
                // alltoallv cache只针对小数据量, 至多只有1个step
                CHK_PRT_RET(
                    step != 0,
                    HCCL_ERROR(
                        "[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] needAlltoallvCache_[%u] step[%u]",
                        needAlltoallvCache_, step),
                    HCCL_E_INTERNAL);

                std::unordered_map<u32, ReadDataBlock>::const_iterator mapIter = subStreamZcopyReadInfo_.find(recvRank);
                if (mapIter != subStreamZcopyReadInfo_.end()) { // recvRank的recvCount为0
                    // 零长拷贝下, recvRank对应的readInfo.size一定为0
                    CHK_PRT_RET(
                        readInfo.size() > 0,
                        HCCL_ERROR(
                            "[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] invalid readInfo.size[%u] != 0",
                            readInfo.size()),
                        HCCL_E_INTERNAL);

                    // 获取零长拷贝的接收偏移
                    const ReadDataBlock& readBlock = mapIter->second;
                    CHK_PRT_RET(
                        readBlock.recvLen != 0,
                        HCCL_ERROR(
                            "[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] invalid readBlock.recvLen[%llu] "
                            "!= 0",
                            readBlock.recvLen),
                        HCCL_E_INTERNAL);

                    // 强制调用HcclD2DMemcpyAsync下发cache-memcpy/write placeholder (aicpu cache使能时才会下发,
                    // 未使能时会直接返回)
                    const LINK& intraNeighboorTransport = links_[recvRank];
                    CHK_PTR_NULL(intraNeighboorTransport);
                    void* remDMAMemPtr = nullptr;
                    CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::INPUT_MEM, &remDMAMemPtr));
                    DeviceMem remoteCCLInMem = DeviceMem::create(static_cast<u8*>(remDMAMemPtr), cclInMem_.size());
                    DeviceMem srcMem = remoteCCLInMem.range(readBlock.remoteOffset, readBlock.recvLen);
                    DeviceMem dstMem = userOutput_.range(readBlock.recvOffset, readBlock.recvLen);
                    reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(true);
                    HCCL_INFO(
                        "[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] generate cache-memcpy placeholder "
                        "for recvRank[%u]",
                        recvRank);
                    CHK_RET(HcclD2DMemcpyAsync(
                        dispatcher_, dstMem, srcMem, currStream, readTransport->GetRemoteRank(),
                        readTransport->GetLinkType()));
                    HCCL_INFO(
                        "[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] generate cache-write placeholder "
                        "for recvRank[%u]",
                        recvRank);
                    CHK_RET(readTransport->TxDataSignal(currStream));
                    reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(false);

                    // 正常下NotifyRecord/Wait SQE (本地主从流同步, 从流不存在跨卡数据拷贝,
                    // 下发placeholder后会立刻post主流并进入wait)
                    HCCL_DEBUG(
                        "[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] userRank [%u], recvRank[%u], "
                        "sendRank[%u],"
                        "sdma stream [%u] read data from remote offset [%llu] len [%llu] to local [%llu], "
                        "post sync info: step[%u], roundIdx[%u], lastStep_[%u], lastRoundIdx_[%u]",
                        userRank_, recvRank, sendRank, streamIndex, readBlock.remoteOffset, readBlock.recvLen,
                        readBlock.recvOffset, step, roundIdx, lastStep_, lastRoundIdx_);
                    if (isPostSyncEnable) {
                        HCCL_DEBUG("[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] post sync begins");
                        CHK_RET(LocalNotify::Post(
                            currStream, dispatcher_, sdmaMeshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
                        CHK_RET(LocalNotify::Wait(
                            currStream, dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
                    }
                }
            }

            if (step < readInfo.size()) {
                const LINK& intraNeighboorTransport = links_[recvRank];
                CHK_PTR_NULL(intraNeighboorTransport);
                void* remDMAMemPtr = nullptr;
                CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::INPUT_MEM, &remDMAMemPtr));
                DeviceMem remoteCCLInMem = DeviceMem::create(static_cast<u8*>(remDMAMemPtr), cclInMem_.size());
                DeviceMem srcMem = remoteCCLInMem.range(readInfo[step].remoteOffset, readInfo[step].recvLen);
                DeviceMem dstMem = userOutput_.range(readInfo[step].recvOffset, readInfo[step].recvLen);
                CHK_RET(HcclD2DMemcpyAsync(
                    dispatcher_, dstMem, srcMem, currStream, readTransport->GetRemoteRank(),
                    readTransport->GetLinkType()));
                HCCL_DEBUG(
                    "[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] userRank [%u], recvRank[%u], "
                    "sendRank[%u],"
                    "sdma stream [%u] read data from remote offset [%llu] len [%llu] to local [%llu], "
                    "post sync info: step[%u], roundIdx[%u], lastStep_[%u], lastRoundIdx_[%u]",
                    userRank_, recvRank, sendRank, streamIndex, readInfo[step].remoteOffset, readInfo[step].recvLen,
                    readInfo[step].recvOffset, step, roundIdx, lastStep_, lastRoundIdx_);
                if (isPostSyncEnable) {
                    HCCL_DEBUG("[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] post sync begins");
                    CHK_RET(LocalNotify::Post(
                        currStream, dispatcher_, sdmaMeshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
                    CHK_RET(LocalNotify::Wait(
                        currStream, dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
                }
                CHK_RET(readTransport->TxDataSignal(currStream));
            }

            if (needAlltoallvCache_) {
                std::unordered_map<u32, SendDataBlock>::const_iterator mapIter = subStreamZcopySendInfo_.find(sendRank);
                if (mapIter != subStreamZcopySendInfo_.end()) { // sendRank的sendCount为0
                    // 零长拷贝下, sendRank对应的sendInfo.size一定为0
                    CHK_PRT_RET(
                        sendInfo.size() > 0,
                        HCCL_ERROR(
                            "[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] invalid sendInfo.size[%u] != 0",
                            sendInfo.size()),
                        HCCL_E_INTERNAL);

                    // 生成cache-notify placeholder
                    reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(true);
                    HCCL_INFO(
                        "[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] generate cache-notify placeholder "
                        "for sendRank[%u]",
                        sendRank);
                    CHK_RET(sendTransport->RxDataSignal(currStream));
                    reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(false);
                }
            }

            if (step < sendInfo.size()) {
                CHK_RET(sendTransport->RxDataSignal(currStream));
            }
            streamIndex++;
        }
    }
    HCCL_INFO("[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::SendRecvData(u32 step, u32 roundIdx)
{
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][SendRecvData] userRank [%u] sdma stream [%s] wait main stream", userRank_,
        GetStreamIndexString().c_str());
    CHK_RET(NotifyRemoteRankStart(step));
    CHK_RET(WaitSubStreamFinish());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(NotifySubStreamStart());
    if (isBigCount_ && (roundIdx < commRounds_ - 1)) {
        CHK_RET(NotifyLocalSubStreamStart());
        CHK_RET(PrepareIntraData(step, nextSubStreamSendInfo_, nextSubStreamZcopySendInfo_));
    }
    CHK_RET(SDMAwithRemoteRankAndNotifyEnd(step, roundIdx));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::LocalCopy()
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    DeviceMem src = userInput_.range(localSendRecvInfo.sendOffset[userRank_], localSendRecvInfo.sendLength[userRank_]);
    DeviceMem dst = userOutput_.range(localSendRecvInfo.recvOffset[userRank_], localSendRecvInfo.recvLength[userRank_]);
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][LocalCopy]userRank [%u] copy from userInput [%llu] len [%llu]"
        "to userOutput [%llu] dstLen[%llu]",
        userRank_, localSendRecvInfo.sendOffset[userRank_], localSendRecvInfo.sendLength[userRank_],
        localSendRecvInfo.recvOffset[userRank_], localSendRecvInfo.recvLength[userRank_]);
    if (needAlltoallvCache_ && localSendRecvInfo.sendLength[userRank_] == 0) {
        reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(true);
    }
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
    if (needAlltoallvCache_ && localSendRecvInfo.sendLength[userRank_] == 0) {
        reinterpret_cast<DispatcherPub*>(dispatcher_)->SetPlaceholder(false);
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunGroupFullMeshAlltoall(u32 roundIdx, u32 step)
{
    UpdateOpBaseSubStreamInfo(step, roundIdx);
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    if (isBigCount_ && (roundIdx == 0)) {
        CHK_RET(NotifyLocalSubStreamStart());
        CHK_RET(PrepareIntraData(step, subStreamSendInfo_, subStreamZcopySendInfo_));
        CHK_RET(WaitLocalSubStreamFinish());
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    } else if (!isBigCount_) {
        CHK_RET(PrepareIntraData(step, subStreamSendInfo_, subStreamZcopySendInfo_));
    }
    CHK_RET(NotifySubStreamStart());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(SendRecvData(step, roundIdx));
    if (step == 0 && !islocalCpyDone_) {
        CHK_RET(LocalCopy());
        islocalCpyDone_ = true;
    }
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(WaitSubStreamFinish());
    if (isBigCount_ && (roundIdx < commRounds_ - 1)) {
        CHK_RET(WaitLocalSubStreamFinish());
    }
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    return HCCL_SUCCESS;
}

// 主流通知RDMA控制流启动
HcclResult AlltoAllVDirectFullMesh::MainNotifyRdmaControlStart()
{
    CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, rdmaControl2MainStreamNotify_, INVALID_VALUE_STAGE));
    CHK_RET(LocalNotify::Wait(rdmaSubStreams_[0], dispatcher_, rdmaControl2MainStreamNotify_, INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

// RDMA控制流通知主流任务完成
HcclResult AlltoAllVDirectFullMesh::RdmaControlNotifyMainFinish()
{
    CHK_RET(LocalNotify::Post(rdmaSubStreams_[0], dispatcher_, main2RdmaControlStreamNotify_, INVALID_VALUE_STAGE));
    CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, main2RdmaControlStreamNotify_, INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

// RDMA控制流通知从流启动任务
HcclResult AlltoAllVDirectFullMesh::RdmaControlNotifySubStart()
{
    for (u32 i = 1; i < rdmaSubStreams_.size(); i++) {
        CHK_RET(
            LocalNotify::Post(rdmaSubStreams_[0], dispatcher_, rdmaSub2ControlNotifies_[i - 1], INVALID_VALUE_STAGE));
        CHK_RET(
            LocalNotify::Wait(rdmaSubStreams_[i], dispatcher_, rdmaSub2ControlNotifies_[i - 1], INVALID_VALUE_STAGE));
    }

    return HCCL_SUCCESS;
}

// 从流通知RDMA控制流任务结束
HcclResult AlltoAllVDirectFullMesh::SubNotifyRdmaControlFinish()
{
    for (u32 i = 1; i < rdmaSubStreams_.size(); i++) {
        CHK_RET(
            LocalNotify::Post(rdmaSubStreams_[i], dispatcher_, rdmaControl2SubNotifies_[i - 1], INVALID_VALUE_STAGE));
        CHK_RET(
            LocalNotify::Wait(rdmaSubStreams_[0], dispatcher_, rdmaControl2SubNotifies_[i - 1], INVALID_VALUE_STAGE));
    }

    return HCCL_SUCCESS;
}

u32 AlltoAllVDirectFullMesh::GetNextDstRank(u32& curDstRank)
{
    if (curDstRank >= userRankSize_) {
        curDstRank = curDstRank % userRankSize_;
    }
    if (curDstRank == podStartRank_) {
        curDstRank += devNumInlocalPod_;
    }
    curDstRank = curDstRank % userRankSize_;
    return curDstRank++;
}

u32 AlltoAllVDirectFullMesh::GetPreSrcRank(u32& curDstRank)
{
    if (curDstRank == podStartRank_ + devNumInlocalPod_ - 1) {
        curDstRank = (curDstRank + userRankSize_ - devNumInlocalPod_) % userRankSize_;
    }

    if (curDstRank == 0) {
        curDstRank = userRankSize_ - 1;
        return 0;
    }
    return curDstRank--;
}

void AlltoAllVDirectFullMesh::GenRdmaSendInfo(u32 dstRank, std::vector<SendDataBlock>& sendInfo)
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    u64 sendOffset = localSendRecvInfo.sendOffset[dstRank];
    u64 sendLength = localSendRecvInfo.sendLength[dstRank];
    while (sendLength > 0) {
        u64 curSendLength = std::min(sendLength, rdmaDataBlockSize_);
        SendDataBlock sendData;
        sendData.userInOffset = sendOffset;
        sendData.sendLen = curSendLength;
        u32 index = dstRank % rdmaConcurrentNum_;
        sendData.scratchOffset = rdmaDataBlockSize_ * index;
        sendInfo.push_back(sendData);
        sendOffset += curSendLength;
        sendLength -= curSendLength;
        HCCL_DEBUG(
            "[GenRdmaSendInfo] userRank[%u], dstRank[%u], sendData.userInOffset[%llu],"
            "sendData.sendLen[%llu], sendData.scratchOffset[%llu]",
            userRank_, dstRank, sendData.userInOffset, sendData.sendLen, sendData.scratchOffset);
    }
    return;
}

void AlltoAllVDirectFullMesh::GenRdmaRecvInfo(u32 srcRank, std::vector<RecvDataBlock>& recvInfo)
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    u64 recvOffset = localSendRecvInfo.recvOffset[srcRank];
    u64 recvLength = localSendRecvInfo.recvLength[srcRank];
    while (recvLength > 0) {
        u64 curRecvLength = std::min(recvLength, rdmaDataBlockSize_);
        RecvDataBlock recvData;
        recvData.recvOffset = recvOffset;
        recvData.recvLen = curRecvLength;
        u32 index = srcRank % rdmaConcurrentNum_;
        recvData.scratchOffset = rdmaDataBlockSize_ * index + rdmaDataBlockSize_ * rdmaConcurrentNum_;
        recvInfo.push_back(recvData);
        recvOffset += curRecvLength;
        recvLength -= curRecvLength;
        HCCL_DEBUG(
            "[GenRdmaRecvInfo] userRank[%llu], srcRank[%u], recvData.recvOffset[%llu],"
            "recvData.recvLen[%llu], recvData.scratchOffset[%llu]",
            userRank_, srcRank, recvData.recvOffset, recvData.recvLen, recvData.scratchOffset);
    }
    return;
}

// 将数据从userIn拷贝到CCL out
HcclResult
AlltoAllVDirectFullMesh::CopyDataForSend(u32 dstRank, std::vector<SendDataBlock>& sendInfo, u32 curStep, Stream stream)
{
    if (curStep >= sendInfo.size()) {
        return HCCL_SUCCESS;
    }
    DeviceMem src = userInput_.range(sendInfo[curStep].userInOffset, sendInfo[curStep].sendLen);
    DeviceMem dst = cclOutMem_.range(sendInfo[curStep].scratchOffset, sendInfo[curStep].sendLen);
    HCCL_DEBUG(
        "[CopyDataForSend] userRank[%u], dstRank[%u], userInOffset[%llu], sendLen[%llu], scratchOffset[%llu]",
        userRank_, dstRank, sendInfo[curStep].userInOffset, sendInfo[curStep].sendLen, sendInfo[curStep].scratchOffset);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RdmaPostSync(Stream& stream)
{
    CHK_RET(LocalNotify::Post(stream, dispatcher_, rdmaControl2SubNotifies_[0], INVALID_VALUE_STAGE));
    CHK_RET(LocalNotify::Wait(rdmaSubStreams_[0], dispatcher_, rdmaControl2SubNotifies_[0], INVALID_VALUE_STAGE));

    CHK_RET(LocalNotify::Post(rdmaSubStreams_[0], dispatcher_, rdmaSub2ControlNotifies_[0], INVALID_VALUE_STAGE));
    CHK_RET(LocalNotify::Wait(stream, dispatcher_, rdmaSub2ControlNotifies_[0], INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

// 从流完成RDMA数据的收发
HcclResult AlltoAllVDirectFullMesh::SendRecvRdmaData(
    u32 dstRank, u32 srcRank, std::vector<SendDataBlock>& sendInfo, std::vector<RecvDataBlock>& recvInfo, u32 round,
    u32 index, u32 curStep, Stream stream)
{
    const LINK& sendTransport = links_[dstRank];
    const LINK& recvTransport = links_[srcRank];
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][SendRecvRdmaData] userRank[%u], dstRank[%u], srcRank[%u]", userRank_, dstRank,
        srcRank);
    u32 minStep = std::min(sendInfo.size(), recvInfo.size());
    CHK_PTR_NULL(sendTransport);
    CHK_PTR_NULL(recvTransport);
    if (curStep < minStep) {
        CHK_RET(recvTransport->TxAck(stream));
        CHK_RET(sendTransport->RxAck(stream));
        u64 sendSrcOffset = (dstRank % rdmaConcurrentNum_) * rdmaDataBlockSize_;
        void* srcPtr = static_cast<u8*>(cclOutMem_.ptr()) + sendSrcOffset;
        u32 dstIndex = userRank_ % rdmaConcurrentNum_;
        u64 sendDstOffset = (dstIndex + rdmaConcurrentNum_) * rdmaDataBlockSize_;
        CHK_RET(
            sendTransport->TxAsync(UserMemType::OUTPUT_MEM, sendDstOffset, srcPtr, sendInfo[curStep].sendLen, stream));

        u64 recvDstOffset = (srcRank % rdmaConcurrentNum_ + rdmaConcurrentNum_) * rdmaDataBlockSize_;
        void* dstPtr = static_cast<u8*>(cclOutMem_.ptr()) + recvDstOffset;
        u64 recvSrcOffset = (userRank_ % rdmaConcurrentNum_) * rdmaDataBlockSize_;
        CHK_RET(
            recvTransport->RxAsync(UserMemType::OUTPUT_MEM, recvSrcOffset, dstPtr, recvInfo[curStep].recvLen, stream));
        if ((round == lastRdmaRoundIdx_) && (index == lastRdmaDstRanksIdx_) && (curStep == lastRdmaStep_)
            && (sdmaConcurrentNum_ > 1) && algOpContext_.opRetryHandler.retryEnable) {
            HCCL_DEBUG("[AlltoAllVDirectFullMesh][SendRecvRdmaData] post sync begins");
            CHK_RET(RdmaPostSync(stream));
        }
        CHK_RET(recvTransport->PostFinAck(stream));
        CHK_RET(sendTransport->WaitFinAck(stream));
        HCCL_DEBUG(
            "[AlltoAllVDirectFullMesh][SendRecvRdmaData] sendSrcOffset[%llu], sendDstOffset[%llu],"
            "recvDstOffset[%llu], recvSrcOffset[%llu], srcPtr[%p], dstPtr[%p]",
            sendSrcOffset, sendDstOffset, recvDstOffset, recvSrcOffset, srcPtr, dstPtr);
    } else if (curStep < sendInfo.size()) {
        CHK_RET(sendTransport->RxAck(stream));
        u64 sendSrcOffset = (dstRank % rdmaConcurrentNum_) * rdmaDataBlockSize_;
        void* srcPtr = static_cast<u8*>(cclOutMem_.ptr()) + sendSrcOffset;
        u32 dstIndex = userRank_ % rdmaConcurrentNum_;
        u64 sendDstOffset = (dstIndex + rdmaConcurrentNum_) * rdmaDataBlockSize_;
        CHK_RET(
            sendTransport->TxAsync(UserMemType::OUTPUT_MEM, sendDstOffset, srcPtr, sendInfo[curStep].sendLen, stream));
        CHK_RET(sendTransport->WaitFinAck(stream));
    } else {
        CHK_RET(recvTransport->TxAck(stream));
        u64 recvDstOffset = (srcRank % rdmaConcurrentNum_ + rdmaConcurrentNum_) * rdmaDataBlockSize_;
        void* dstPtr = static_cast<u8*>(cclOutMem_.ptr()) + recvDstOffset;
        u64 recvSrcOffset = (userRank_ % rdmaConcurrentNum_) * rdmaDataBlockSize_;
        CHK_RET(
            recvTransport->RxAsync(UserMemType::OUTPUT_MEM, recvSrcOffset, dstPtr, recvInfo[curStep].recvLen, stream));
        if ((round == lastRdmaRoundIdx_) && (index == lastRdmaDstRanksIdx_) && (curStep == lastRdmaStep_)
            && (sdmaConcurrentNum_ > 1) && algOpContext_.opRetryHandler.retryEnable) {
            HCCL_DEBUG("[AlltoAllVDirectFullMesh][SendRecvRdmaData] post sync begins");
            CHK_RET(RdmaPostSync(stream));
        }
        CHK_RET(recvTransport->PostFinAck(stream));
    }
    return HCCL_SUCCESS;
}

// 从流将接收到的数据拷贝到输出
HcclResult AlltoAllVDirectFullMesh::CopyRecvDataToOutput(
    u32 srcRank, std::vector<RecvDataBlock>& recvInfo, u32 curStep, Stream stream)
{
    if (curStep >= recvInfo.size()) {
        return HCCL_SUCCESS;
    }
    u64 srcOffset = (srcRank % rdmaConcurrentNum_ + rdmaConcurrentNum_) * rdmaDataBlockSize_;
    DeviceMem src = cclOutMem_.range(srcOffset, recvInfo[curStep].recvLen);
    DeviceMem dst = userOutput_.range(recvInfo[curStep].recvOffset, recvInfo[curStep].recvLen);
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][CopyRecvDataToOutput] userRank[%u], srcRank[%u], srcOffset[%llu],"
        "recvInfo[curStep].recvOffset[%llu], recvLen[%llu]",
        userRank_, srcRank, srcOffset, recvInfo[curStep].recvOffset, recvInfo[curStep].recvLen);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream));
    return HCCL_SUCCESS;
}

HcclResult
AlltoAllVDirectFullMesh::ProcessSingleGroupRdmaData(std::vector<u32>& dstRanks, std::vector<u32>& srcRanks, u32 round)
{
    lastRdmaDstRanksIdx_ = dstRanks.size() - 1;
    for (u32 index = 0; index < dstRanks.size(); index++) {
        u32 dstRank = dstRanks[index];
        u32 srcRank = srcRanks[index];
        Stream stream = rdmaSubStreams_[index + 1];

        std::vector<SendDataBlock> sendInfo;
        std::vector<RecvDataBlock> recvInfo;
        GenRdmaSendInfo(dstRank, sendInfo);
        GenRdmaRecvInfo(srcRank, recvInfo);
        u32 totalStep = std::max(sendInfo.size(), recvInfo.size());
        lastRdmaStep_ = totalStep - 1;
        for (u32 curStep = 0; curStep < totalStep; curStep++) {
            CHK_RET(CopyDataForSend(dstRank, sendInfo, curStep, stream));
            CHK_RET(SendRecvRdmaData(dstRank, srcRank, sendInfo, recvInfo, round, index, curStep, stream));
            CHK_RET(CopyRecvDataToOutput(srcRank, recvInfo, curStep, stream));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::ProcessRdmaData()
{
    // RDMA通信轮次
    u32 rdmaRoundNum = (totalRdmaRankNum_ + rdmaConcurrentNum_ - 1) / rdmaConcurrentNum_;
    lastRdmaRoundIdx_ = rdmaRoundNum - 1;

    u32 leftRankNum = totalRdmaRankNum_;
    u32 curSrcRank = INVALID_VALUE_RANKID;
    u32 curDstRank = INVALID_VALUE_RANKID;
    if (isSuPodAsym_) {
        for (u32 i = 0; i < userRankSize_; i++) {
            if (i < podStartRank_ || i > podEndRank_) {
                curSrcRank = i;
                curDstRank = i;
                break;
            }
        }
    } else {
        curDstRank = (userRank_ + devNumInlocalPod_) % userRankSize_;
        curSrcRank = (userRank_ + userRankSize_ - devNumInlocalPod_) % userRankSize_;
    }

    for (u32 round = 0; round < rdmaRoundNum; round++) {
        u32 curProcessRankNum = leftRankNum >= rdmaConcurrentNum_ ? rdmaConcurrentNum_ : leftRankNum;
        leftRankNum -= curProcessRankNum;

        std::vector<u32> dstRanks;
        std::vector<u32> srcRanks;
        for (u32 i = 0; i < curProcessRankNum; i++) {
            dstRanks.push_back(GetNextDstRank(curDstRank));
            if (isSuPodAsym_) {
                srcRanks.push_back(dstRanks.back());
            } else {
                srcRanks.push_back(GetPreSrcRank(curSrcRank));
            }
        }
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, rdmaSubStreams_[0], dispatcher_));
        CHK_RET(RdmaControlNotifySubStart());
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, rdmaSubStreams_[0], dispatcher_));
        CHK_RET(ProcessSingleGroupRdmaData(dstRanks, srcRanks, round));
        CHK_RET(SubNotifyRdmaControlFinish());
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, rdmaSubStreams_[0], dispatcher_));
    }
    HCCL_INFO("[AlltoAllVDirectFullMesh][ProcessRdmaData] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunRDMA()
{
    // 先启动RDMA通信
    CHK_RET(MainNotifyRdmaControlStart());
    CHK_RET(ProcessRdmaData());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(LocalCopy());
    islocalCpyDone_ = true;
    HCCL_INFO("[AlltoAllVDirectFullMesh][RunRDMA] finished.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunSDMATasks(u32 roundIdx, u32 step, u32 groupRankSize, u32 leftRankSize)
{
    if (isBigCount_) {
        if (roundIdx == 0) {
            UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
        }
        if (roundIdx < commRounds_ - 1) {
            u32 nextgroupRankSize = (leftRankSize - groupRankSize > sdmaConcurrentNum_) ? sdmaConcurrentNum_ :
                                                                                          leftRankSize - groupRankSize;
            UpdatePartialCommunicationRankSet(roundIdx + 1, nextgroupRankSize, nextPartialCommRankSet_);
        }
        CHK_RET(RunGroupFullMeshAlltoall(roundIdx, step));

        if (roundIdx < commRounds_ - 1) {
            partialCommRankSet_ = nextPartialCommRankSet_;
            subStreamSendInfo_ = nextSubStreamSendInfo_;
            subStreamReadInfo_ = nextSubStreamReadInfo_;
            if (needAlltoallvCache_) {
                subStreamZcopySendInfo_ = nextSubStreamZcopySendInfo_;
                subStreamZcopyReadInfo_ = nextSubStreamZcopyReadInfo_;
            }
        }
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, localSubStream_));
    } else {
        UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
        CHK_RET(RunGroupFullMeshAlltoall(roundIdx, step));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunSDMAFineGrained(u32 totalStep, HcclOpMetaInfoDef& opMeta)
{
    if (totalStep > 1) {
        // 细粒度场景不支持切分
        HCCL_ERROR(
            "[AlltoAllVDirectFullMesh][RunSDMAFineGrained] AlltoAllV is not supported when totalStep[%u] > 1, "
            "HCCL buffer is insufficient. stepSize : %u ",
            totalStep, algOpContext_.mc2Handler.stepSize);
        return HCCL_E_NOT_SUPPORT;
    } else if (totalStep == 0) {
        // totalStep不需要通信，但是需要适配高阶API wait/write
        for (u32 roundIdx = 0; roundIdx < commRounds_; roundIdx++) {
            CHK_RET(mc2HandlerPub.Mc2WaitValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler), roundIdx));
            CHK_RET(mc2HandlerPub.Mc2WriteValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler)));
            HCCL_INFO("[AlltoAllVDirectFullMesh][RunSDMAFineGrained] step is 0 finished.");
        }
    } else {
        // totalStep == 1 细粒度修改
        u32 leftRankSize = devNumInlocalPod_; // leftRankSize中去掉本卡
        for (u32 roundIdx = 0; roundIdx < commRounds_ && leftRankSize > 0; roundIdx++) {
            CHK_RET(mc2HandlerPub.Mc2WaitValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler), roundIdx));
            CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
            u32 groupRankSize = (leftRankSize > sdmaConcurrentNum_) ? sdmaConcurrentNum_ : leftRankSize;
            UpdateRemoteRankSet(roundIdx, groupRankSize);
            CHK_RET(RunGroupFullMeshAlltoall(roundIdx, 0));
            leftRankSize -= groupRankSize;
            CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
            CHK_RET(mc2HandlerPub.Mc2WriteValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler)));
        }
        HCCL_INFO("[AlltoAllVDirectFullMesh][RunSDMAFineGrained] fine-grained finished.");
        return HCCL_SUCCESS;
    }

    if (totalStep == 0 && !islocalCpyDone_) {
        CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
        CHK_RET(LocalCopy());
        islocalCpyDone_ = true;
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[AlltoAllVDirectFullMesh][RunSDMAFineGrained] finished.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunSDMA(HcclOpMetaInfoDef& opMeta)
{
    u32 totalStep = CalcNumSubStep();
    lastStep_ = totalStep - 1;
    // 计算每个rank分组fullmesh后需要通信的轮次，向上取整
    commRounds_ = (devNumInlocalPod_ + sdmaConcurrentNum_ - 1) / sdmaConcurrentNum_;
    u32 leftRankSize = devNumInlocalPod_ - 1; // leftRankSize中去掉本卡
    lastRoundIdx_
        = std::min((leftRankSize + sdmaConcurrentNum_ - 1) / sdmaConcurrentNum_, static_cast<u32>(commRounds_)) - 1;
    HCCL_DEBUG(
        "[AlltoAllVDirectFullMesh][RunSDMA] userRank [%u] communication rounds[%llu] totalStep [%u] "
        "stepSize [%u], post sync info: lastStep_[%u] lastRoundIdx_[%u] devNumInlocalPod_[%u] sdmaConcurrentNum_[%u]",
        userRank_, commRounds_, totalStep, algOpContext_.mc2Handler.stepSize, lastStep_, lastRoundIdx_,
        devNumInlocalPod_, sdmaConcurrentNum_);

    if (UNLIKELY(algOpContext_.mc2Handler.stepSize > 0)) {
        CHK_RET(RunSDMAFineGrained(totalStep, opMeta));
    } else {
        if (totalStep == 0 && !islocalCpyDone_) {
            CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
            CHK_RET(LocalCopy());
            islocalCpyDone_ = true;
            CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
            return HCCL_SUCCESS;
        }

        for (u32 step = 0; step < totalStep; step++) {
            u32 currentLeftRankSize = devNumInlocalPod_ - 1; // leftRankSize中去掉本卡
            for (u32 roundIdx = 0; roundIdx < commRounds_ && currentLeftRankSize > 0; roundIdx++) {
                CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
                u32 groupRankSize
                    = (currentLeftRankSize > sdmaConcurrentNum_) ? sdmaConcurrentNum_ : currentLeftRankSize;
                CHK_RET(RunSDMATasks(roundIdx, step, groupRankSize, currentLeftRankSize));
                currentLeftRankSize -= groupRankSize;
                CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
            }
        }
    }

    HCCL_INFO("[AlltoAllVDirectFullMesh][RunSDMA] finished.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunAsync()
{
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, cclInMem_.size(), true);
    CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));

    if (algOpContext_.mc2Handler.stepSize > 0) {
        if (algOpContext_.mc2Handler.stepSize > userRankSize_
            || userRankSize_ % algOpContext_.mc2Handler.stepSize != 0) {
            HCCL_ERROR(
                "[AlltoAllVDirectFullMesh][RunAsync] Step size should be less than or equal to the rank size, "
                "and the rank size should be a multiple of the step size, but the step size is [%u] and the rank size "
                "is [%u].",
                algOpContext_.mc2Handler.stepSize, userRankSize_);
            return HCCL_E_PARA;
        }
        if (userRankSize_ == 1) {
            HCCL_INFO("[AlltoAllVDirectFullMesh][RunAsync] AlltoAllV do localcopy with 1 rank");
            CHK_RET(mc2HandlerPub.Mc2WaitValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler), 0));
            CHK_RET(LocalCopy());
            CHK_RET(mc2HandlerPub.Mc2WriteValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler)));
            return HCCL_SUCCESS;
        }
    }

    if (userRankSize_ == 1) {
        HCCL_INFO("[AlltoAllVDirectFullMesh][RunAsync] do localcopy with 1 rank");
        CHK_RET(LocalCopy());
        return HCCL_SUCCESS;
    }

    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    if (totalRdmaRankNum_ > 0) {
        CHK_RET(RunRDMA());
    }

    CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, rdmaSubStreams_));

    if (devNumInlocalPod_ > 1) {
        CHK_RET(RunSDMA(opMeta));
    }

    if (totalRdmaRankNum_ > 0) {
        // 等待RDMA通信结束
        CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
        CHK_RET(RdmaControlNotifyMainFinish());
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, rdmaSubStreams_));
    }

    HCCL_INFO("[AlltoAllVDirectFullMesh][RunAsync] finished.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::GetNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    (void)links;
    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }

    u32 devNumInlocalPod = nslbAdjInfo.dstRankNum;
    u32 totalRdmaRankNum = rankSize - devNumInlocalPod;

    u32 rdmaConcurrentNum = (totalRdmaRankNum > ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) ?
                                (ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) :
                                (totalRdmaRankNum);
    if (rdmaConcurrentNum == 0) {
        return HCCL_SUCCESS;
    }
    // RDMA通信轮次
    u32 rdmaRoundNum = (totalRdmaRankNum + rdmaConcurrentNum - 1) / rdmaConcurrentNum;
    if (rdmaRoundNum == 0) {
        return HCCL_SUCCESS;
    }
    u32 currStage = rank / devNumInlocalPod;

    for (u32 step = 0; step < rdmaRoundNum; step++) {
        u32 sendTo = (rank + devNumInlocalPod + step) % rankSize;
        u32 sendToStag = sendTo / devNumInlocalPod;
        if (currStage == sendToStag) {
            // 此时认为时同一个超节点内通讯
            sendTo = (sendTo + devNumInlocalPod) % rankSize;
        }
        NslbDpAdjInfo adjInfoStep = {};
        adjInfoStep.dstLocalRankId = sendTo;
        adjInfoStep.phaseId = step + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::GetHcclOffsetDstRanksMap(
    std::unordered_map<uint64_t, std::vector<uint32_t>>& hcclOffsetDstRanksMap) const
{
    hcclOffsetDstRanksMap.clear();
    hcclOffsetDstRanksMap = hcclOffsetDstRanksMap_; // Deep copy

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH, AlltoAllVDirectFullMesh);
} // namespace hccl
