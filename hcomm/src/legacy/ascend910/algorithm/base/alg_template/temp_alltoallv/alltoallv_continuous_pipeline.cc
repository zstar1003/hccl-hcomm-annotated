/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_continuous_pipeline.h"

#include <vector>
#include <algorithm>
#include "alg_template_register.h"

namespace hccl {
AlltoallvContinuousPipeline::AlltoallvContinuousPipeline(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher)
{}

AlltoallvContinuousPipeline::~AlltoallvContinuousPipeline() {}

HcclResult AlltoallvContinuousPipeline::PrepareSendRecvInfo(std::vector<SendRecvInfo>& sendRecvInfoList)
{
    if (sendRecvInfoList.size() == 1) {
        // 真实业务场景
        SendRecvInfo& localSendRecvInfo = sendRecvInfoList[0];
        localSendCounts_ = std::move(localSendRecvInfo.sendCounts);
        localSendDispls_ = std::move(localSendRecvInfo.sendDispls);
        localRecvCounts_ = std::move(localSendRecvInfo.recvCounts);
        localRecvDispls_ = std::move(localSendRecvInfo.recvDispls);
        needCollectInfo_ = true; // 需要收集信息
        std::copy(localRecvCounts_.begin(), localRecvCounts_.end(), intraRecvCounts_[intraRankId_].begin());
    } else {
        // 适配算法分析器，实际业务不会走这个分支
        SendRecvInfo& localSendRecvInfo = sendRecvInfoList[userRank_];

        std::copy(
            localSendRecvInfo.sendCounts.begin(), localSendRecvInfo.sendCounts.end(),
            std::back_inserter(localSendCounts_));
        std::copy(
            localSendRecvInfo.sendDispls.begin(), localSendRecvInfo.sendDispls.end(),
            std::back_inserter(localSendDispls_));
        std::copy(
            localSendRecvInfo.recvCounts.begin(), localSendRecvInfo.recvCounts.end(),
            std::back_inserter(localRecvCounts_));
        std::copy(
            localSendRecvInfo.recvDispls.begin(), localSendRecvInfo.recvDispls.end(),
            std::back_inserter(localRecvDispls_));

        for (u32 intraRankIdx = 0; intraRankIdx < intraRankSize_; ++intraRankIdx) {
            const u32 remoteRank = interRankId_ * intraRankSize_ + intraRankIdx;
            SendRecvInfo& sendRecvInfo = sendRecvInfoList[remoteRank];
            std::copy(
                sendRecvInfo.recvCounts.begin(), sendRecvInfo.recvCounts.end(), intraRecvCounts_[intraRankIdx].begin());
        }
        needCollectInfo_ = false;
    }

    intraLoopNum_ = GetLocalLoopNum();
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::PrepareTopoInfo(
    const u32 userRank, const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo)
{
    constexpr u32 MIN_RANKSIZE = 2;
    interRankSize_ = level1CommInfo.localRankSize;
    CHK_PRT_RET(
        interRankSize_ < MIN_RANKSIZE,
        HCCL_ERROR(
            "[AlltoallvContinuousPipeline][PrepareTopoInfo] Unexpected inter rank size[%u], which should >= 2.",
            interRankSize_),
        HCCL_E_PARA);

    intraRankSize_ = level0CommInfo.localRankSize;
    CHK_PRT_RET(
        intraRankSize_ < MIN_RANKSIZE,
        HCCL_ERROR(
            "[AlltoallvContinuousPipeline][PrepareTopoInfo] Unexpected intra rank size[%u], which should >= 2.",
            intraRankSize_),
        HCCL_E_PARA);

    userRankSize_ = intraRankSize_ * interRankSize_;
    userRank_ = userRank;
    interRankId_ = level1CommInfo.localRank;
    intraRankId_ = level0CommInfo.localRank;
    HCCL_INFO(
        "[AlltoallvContinuousPipeline][PrepareTopoInfo] userRank[%u], intraRankId[%u], intraRankSize[%u], "
        "interRankId[%u], interRankSize[%u]",
        userRank_, intraRankId_, intraRankSize_, interRankId_, interRankSize_);

    // 按照module将rank分组
    ranksPerModule_.resize(interRankSize_);
    for (u32 interRank = 0; interRank < interRankSize_; ++interRank) {
        ranksPerModule_[interRank].resize(intraRankSize_);
        for (u32 intraRank = 0; intraRank < intraRankSize_; ++intraRank) {
            ranksPerModule_[interRank][intraRank] = interRank * intraRankSize_ + intraRank;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::Prepare(
    const u32 userRank, const A2aPipelineMemory& a2aPipelineMemory, const SubCommInfo& level0CommInfo,
    const SubCommInfo& level1CommInfo, const Stream& mainStream, std::vector<Stream>& subStream,
    std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub,
    std::vector<SendRecvInfo>& sendRecvInfoList, const HcclDataType dataType, const HcclWorkflowMode workMode,
    u32 timeOut)
{
    // 运行模式：当前只支持单算子
    workMode_ = workMode;
    CHK_PRT_RET(
        workMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE,
        HCCL_ERROR("[AlltoallvContinuousPipeline] This template supports opbase mode only."), HCCL_E_INTERNAL);

    // 拓扑信息
    CHK_RET(PrepareTopoInfo(userRank, level0CommInfo, level1CommInfo));

    // 并发度暂定为1 - 不并发
    rdmaConcurrentNum_ = 1;

    // 数据类型
    dataType_ = dataType;
    unitSize_ = DataUnitSize(dataType_);

    // 内存
    inputMem_ = a2aPipelineMemory.userInput;
    outputMem_ = a2aPipelineMemory.userOutput;
    inBuffer_ = a2aPipelineMemory.cclInBuffer;
    outBuffer_ = a2aPipelineMemory.cclOutBuffer;

    flagAreaRefreshData_.resize(userRankSize_);

    // server内其他卡的recv counts
    intraRecvCounts_.resize(intraRankSize_);
    for (auto& countVec : intraRecvCounts_) {
        countVec.resize(userRankSize_);
    }

    // 流和notify
    mainStream_ = mainStream;
    CHK_RET(PartitionSubStreamsAndNotifies(subStream, notifyMain, notifySub));

    // 链路
    intraLinks_ = level0CommInfo.links;
    interLinks_ = level1CommInfo.links;
    HCCL_INFO(
        "[AlltoallvContinuousPipeline][Prepare] Link info: interLinksNum[%u], intraLinksNum[%u]", interLinks_.size(),
        intraLinks_.size());

    // pingpong模式: module间只有1步：双module
    enablePingPong_ = rdmaConcurrentNum_ >= interRankSize_ - 1;
    // 切分buffer
    CHK_RET(SplitBuffer(enablePingPong_));

    // 收发信息
    CHK_RET(PrepareSendRecvInfo(sendRecvInfoList));

    // 设置flag超时时间，多加60s
    constexpr u16 waitFlagTimeout = 60;
    waitFlagTimeoutSec_ = timeOut + waitFlagTimeout;

    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::SplitBuffer(const bool enablePingPong)
{
    // 单个counts数组的大小
    const u64 singleRankCountsInfoSize = sizeof(u64) * userRankSize_;
    // 全局counts的大小
    const u64 globalCountsInfoSize = singleRankCountsInfoSize * userRankSize_;

    const u64 bufferSize = inBuffer_.size();
    u32 blockNum = userRankSize_;

    if (enablePingPong) {
        // 乒乓模式两倍分块
        blockNum = userRankSize_ * PINGPONG_MEM_NUM;
        HCCL_INFO("[AlltoallvContinuousPipeline][SplitBuffer] Use ping-pong mode.");
    }

    CHK_PRT_RET(
        blockNum == 0, HCCL_ERROR("[AlltoallvContinuousPipeline][SplitBuffer]Unexpected blockNum[%u].", blockNum),
        HCCL_E_INTERNAL);

    // 初始化用于记录buffer中每个分块当作存放了多少数据的vector
    // 如果是pingpong模式，需要两倍的大小
    inBufferDataSize_.resize(blockNum);

    u64 alignSize = HCCL_MIN_SLICE_ALIGN;
    const u64 minBufferSize = globalCountsInfoSize + alignSize * blockNum;
    CHK_PRT_RET(
        bufferSize < minBufferSize,
        HCCL_ERROR(
            "[AlltoallvContinuousPipeline][SplitBuffer]Insufficient buffer size [%llu Byte]; it needs to be "
            "greater than [%llu Byte].",
            bufferSize, minBufferSize),
        HCCL_E_MEMORY);

    if (bufferSize > globalCountsInfoSize + HCCL_MIN_SLICE_ALIGN_910_93 * blockNum) {
        alignSize = HCCL_MIN_SLICE_ALIGN_910_93;
    }

    countsPerBlock_ = (((bufferSize - globalCountsInfoSize) / blockNum) / alignSize * alignSize)
                      / unitSize_; // 前面已经可以保证countsPerBlock_大于0，不再检查
    sizePerBlock_ = countsPerBlock_ * unitSize_;

    for (u32 rank = 0; rank < userRankSize_; ++rank) {
        infoOffsets_.emplace_back(sizePerBlock_ * blockNum + singleRankCountsInfoSize * rank);
    }

    for (u32 blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
        dataBlockOffsets_.emplace_back(sizePerBlock_ * blockIdx);
    }

    HCCL_INFO(
        "[AlltoallvContinuousPipeline][SplitBuffer] Split buffer done, alignSize[%llu], sizePerBlock[%llu], "
        "countsPerBlock[%llu], blockNum[%u]",
        alignSize, sizePerBlock_, countsPerBlock_, blockNum);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::PartitionSubStreamsAndNotifies(
    const std::vector<Stream>& subStreams, const std::vector<std::shared_ptr<LocalNotify>>& signalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>>& signalSubToMain)
{
    constexpr u32 DEVICE_EIGHT = 8;
    const u32 sdmaConcurrentNum = std::min(intraRankSize_ - 1, DEVICE_EIGHT);
    const u32 totalSubstreamSize = rdmaConcurrentNum_ + sdmaConcurrentNum;
    CHK_PRT_RET(
        subStreams.size() < totalSubstreamSize || signalMainToSub.size() < totalSubstreamSize
            || signalSubToMain.size() < totalSubstreamSize,
        HCCL_ERROR(
            "[AlltoallvContinuousPipeline] subStreams size [%u] or signalMainToSub size [%u] or signalSubToMain "
            "size [%u] is smaller than totalSubstreamSize [%u].",
            subStreams.size(), signalMainToSub.size(), signalSubToMain.size(), totalSubstreamSize),
        HCCL_E_PARA);

    u32 index = 0;

    // 用于SDMA通信的从流和主从同步notify
    for (u32 i = 0; i < sdmaConcurrentNum; ++i) {
        subStreams_.push_back(subStreams[index]);
        sdmaSubStreams_.push_back(subStreams[index]);
        streamNotifyMainToSdmaSub_.push_back(signalMainToSub[index]);
        streamNotifySdmaSubToMain_.push_back(signalSubToMain[index]);
        index++;
    }

    // 用于RDMA通信的从流和主从同步notify
    for (u32 i = 0; i < rdmaConcurrentNum_; ++i) {
        subStreams_.push_back(subStreams[index]);
        rdmaSubStreams_.push_back(subStreams[index]);
        streamNotifyMainToRdmaSub_.push_back(signalMainToSub[index]);
        streamNotifyRdmaSubToMain_.push_back(signalSubToMain[index]);
        index++;
    }

    HCCL_INFO(
        "[AlltoallvContinuousPipeline][PartitionSubStreamsAndNotifies] Done, sdma: #streams[%zu], "
        "#notifyMainSub[%zu], #notifySubToMain[%zu]; rdma: #streams[%zu], #notifyMainSub[%zu], "
        "#notifySubToMain[%zu].",
        sdmaSubStreams_.size(), streamNotifyMainToSdmaSub_.size(), streamNotifySdmaSubToMain_.size(),
        rdmaSubStreams_.size(), streamNotifyMainToRdmaSub_.size(), streamNotifyRdmaSubToMain_.size());

    return HCCL_SUCCESS;
}

inline u32 AlltoallvContinuousPipeline::GetSdmaSubStreamIdx(const u32 remoteRank) const
{
    return (remoteRank > intraRankId_ ? remoteRank - 1 : remoteRank) % sdmaSubStreams_.size();
}

inline u64 AlltoallvContinuousPipeline::GetLocalSendCountOfRank(const u32 targetRank) const
{
    return localSendCounts_[targetRank];
}

inline u64 AlltoallvContinuousPipeline::GetLocalSendDisplOfRank(const u32 targetRank) const
{
    return localSendDispls_[targetRank];
}

inline u64 AlltoallvContinuousPipeline::GetLocalRecvCountOfRank(const u32 sourceRank) const
{
    return localRecvCounts_[sourceRank];
}

inline u64 AlltoallvContinuousPipeline::GetLocalRecvDisplOfRank(const u32 sourceRank) const
{
    return localRecvDispls_[sourceRank];
}

inline u64 AlltoallvContinuousPipeline::GetDataBlockOffset(const u32 rank, const u32 bufferIdx) const
{
    if (enablePingPong_) {
        return dataBlockOffsets_[(bufferIdx % PINGPONG_MEM_NUM) * userRankSize_ + rank];
    }
    return dataBlockOffsets_[rank];
}

u32 AlltoallvContinuousPipeline::GetLocalLoopNum() const
{
    u64 maxCount = 0;
    for (u32 rank = 0; rank < userRankSize_; ++rank) {
        if (rank != userRank_) {
            const u64 sendCount = GetLocalSendCountOfRank(rank);
            maxCount = maxCount < sendCount ? sendCount : maxCount;
            const u64 recvCount = GetLocalRecvCountOfRank(rank);
            maxCount = maxCount < recvCount ? recvCount : maxCount;
        }
    }

    // 向上取整
    const u32 localLoopNum = static_cast<u32>((maxCount + countsPerBlock_ - 1) / countsPerBlock_);
    HCCL_INFO(
        "[AlltoallvContinuousPipeline][GetLocalLoopNum] maxCount[%llu], localLoopNum[%u]", maxCount, localLoopNum);
    return localLoopNum;
}

HcclResult AlltoallvContinuousPipeline::UpdateLocalSendInfo(const u32 targetRank, const u64 count)
{
    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][UpdateLocalSendInfo]userRank[%u], count[%llu], before "
        "update, send info of rank[%u] is [count:%llu, displ:%llu].",
        userRank_, count, targetRank, localSendCounts_[targetRank], localSendDispls_[targetRank]);

    const u64 maxCount = std::min(localSendCounts_[targetRank], count);
    localSendCounts_[targetRank] -= maxCount;
    localSendDispls_[targetRank] += maxCount;

    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][UpdateLocalSendInfo]userRank[%u], count[%llu], after "
        "update, send info of rank[%u] is [count:%llu, displ:%llu].",
        userRank_, count, targetRank, localSendCounts_[targetRank], localSendDispls_[targetRank]);

    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::UpdateLocalRecvInfo(const u32 sourceRank, const u64 count)
{
    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][UpdateLocalRecvInfo]userRank[%u], count[%llu], before "
        "update, receive info of rank[%u] is [count:%llu, displ:%llu].",
        userRank_, count, sourceRank, localRecvCounts_[sourceRank], localRecvDispls_[sourceRank]);

    const u64 maxCount = std::min(localRecvCounts_[sourceRank], count);
    localRecvCounts_[sourceRank] -= maxCount;
    localRecvDispls_[sourceRank] += maxCount;

    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][UpdateLocalRecvInfo]userRank[%u], count[%llu], after "
        "update, receive info of rank[%u] is [count:%llu, displ:%llu].",
        userRank_, count, sourceRank, localRecvCounts_[sourceRank], localRecvDispls_[sourceRank]);

    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::NotifySdmaSubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < sdmaSubStreams_.size(); ++streamIndex) {
        CHK_RET(
            LocalNotify::Post(mainStream_, dispatcher_, streamNotifySdmaSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            sdmaSubStreams_[streamIndex], dispatcher_, streamNotifySdmaSubToMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::WaitSdmaSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < sdmaSubStreams_.size(); ++streamIndex) {
        CHK_RET(LocalNotify::Post(
            sdmaSubStreams_[streamIndex], dispatcher_, streamNotifyMainToSdmaSub_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(
            LocalNotify::Wait(mainStream_, dispatcher_, streamNotifyMainToSdmaSub_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::NotifyRdmaSubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < rdmaSubStreams_.size(); ++streamIndex) {
        CHK_RET(
            LocalNotify::Post(mainStream_, dispatcher_, streamNotifyRdmaSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            rdmaSubStreams_[streamIndex], dispatcher_, streamNotifyRdmaSubToMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::WaitRdmaSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < rdmaSubStreams_.size(); ++streamIndex) {
        CHK_RET(LocalNotify::Post(
            rdmaSubStreams_[streamIndex], dispatcher_, streamNotifyMainToRdmaSub_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(
            LocalNotify::Wait(mainStream_, dispatcher_, streamNotifyMainToRdmaSub_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::InterSdmaRx(
    const LINK& linkLeft, const LINK& linkRight, std::vector<TxMemoryInfo>& sendMems,
    std::vector<RxMemoryInfo>& recvMems, Stream& stream) const
{
    const bool needRecvFromLinkLeft = !recvMems.empty();
    const bool needSendToLinkRight = !sendMems.empty();

    // 前同步，通知right我已准备好，可以从我这里读；等待left通知它已准备好，可以从它那里读
    if (needSendToLinkRight) {
        CHK_RET(linkRight->TxAck(stream));
    }
    if (needRecvFromLinkLeft) {
        CHK_RET(linkLeft->RxAck(stream));
    }

    // 从left读
    for (const auto& memInfo : recvMems) {
        void* srcMemPtr = nullptr;
        CHK_RET(linkLeft->GetRemoteMem(memInfo.srcMemType, &srcMemPtr));
        DeviceMem dstMem = DeviceMem::create(memInfo.dst, memInfo.len);
        DeviceMem srcMem(static_cast<s8*>(srcMemPtr) + memInfo.srcOffset, memInfo.len);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dstMem, srcMem, stream, linkLeft->GetRemoteRank(), linkLeft->GetLinkType()));
    }

    // 尾同步，通知left我已读完，等待right通知它已读完
    if (needRecvFromLinkLeft) {
        CHK_RET(linkLeft->TxDataSignal(stream));
    }
    if (needSendToLinkRight) {
        CHK_RET(linkRight->RxDataSignal(stream));
    }

    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][InterSdmaRx] Done. linkLeft.rank[%u], linkRight.rank[%u], "
        "recvMems.size[%zu]",
        linkLeft->GetRemoteRank(), linkRight->GetRemoteRank(), recvMems.size());
    return HCCL_SUCCESS;
}

// 跨module通信，通过RDMA从link left读或向link right写
HcclResult AlltoallvContinuousPipeline::InterRdmaTxRx(
    const LINK& linkLeft, const LINK& linkRight, std::vector<TxMemoryInfo>& sendMems,
    std::vector<RxMemoryInfo>& recvMems, Stream& stream) const
{
    const bool needRecvFromLinkLeft = !recvMems.empty();
    const bool needSendToLinkRight = !sendMems.empty();

    if (needRecvFromLinkLeft) {
        CHK_RET(linkLeft->TxAck(stream));
    }
    if (needSendToLinkRight) {
        CHK_RET(linkRight->RxAck(stream));
        CHK_RET(linkRight->TxAsync(sendMems, stream));
    }
    if (needRecvFromLinkLeft) {
        CHK_RET(linkLeft->RxAsync(recvMems, stream));
        CHK_RET(linkLeft->PostFinAck(stream));
    }
    if (needSendToLinkRight) {
        CHK_RET(linkRight->WaitFinAck(stream));
    }

    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][InterRdmaTxRx] Done. linkLeft.rank[%u], linkRight.rank[%u], "
        "sendMems.size[%zu], recvMems.size[%zu]",
        linkLeft->GetRemoteRank(), linkRight->GetRemoteRank(), sendMems.size(), recvMems.size());
    return HCCL_SUCCESS;
}

HcclResult
AlltoallvContinuousPipeline::LocalCopyFromInputToInBuffer(const u32 targetRank, Stream& stream, const u32 loopIdx)
{
    // 根据send displs来计算input的位置，取min(countsPerBlock_, count)个数
    const u64 copyCount = std::min(GetLocalSendCountOfRank(targetRank), countsPerBlock_);
    if (copyCount == 0) {
        return HCCL_SUCCESS;
    }

    const u64 copySize = copyCount * unitSize_;

    // 从input拷贝到in buffer对应的分块里
    const u64 srcOffset = GetLocalSendDisplOfRank(targetRank) * unitSize_;
    const u64 dstOffset = GetDataBlockOffset(targetRank, loopIdx);
    DeviceMem src = inputMem_.range(srcOffset, copySize);
    DeviceMem dst = inBuffer_.range(dstOffset, copySize);

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream));

    // 刷新send info
    CHK_RET(UpdateLocalSendInfo(targetRank, copyCount));

    // 记录in buffer中该分块存放了多少数据
    inBufferDataSize_[targetRank] = copySize;

    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][LocalCopy][FromInputToInBuffer] done, userRank[%u], targetRank[%u], "
        "srcOffset[%llu], dstOffset[%llu], copyCount[%llu], copySize[%llu], loopIdx[%u]",
        userRank_, targetRank, srcOffset, dstOffset, copyCount, copySize, loopIdx);
    return HCCL_SUCCESS;
}

HcclResult
AlltoallvContinuousPipeline::LocalCopyFromOutBufferToOutput(const u32 sourceRank, Stream& stream, const u32 loopIdx)
{
    const u64 copyCount = std::min(GetLocalRecvCountOfRank(sourceRank), countsPerBlock_);
    if (copyCount == 0) {
        return HCCL_SUCCESS;
    }

    // 从out buffer对应分块拷贝到output
    const u64 copySize = copyCount * unitSize_;
    const u64 srcOffset = GetDataBlockOffset(sourceRank, loopIdx);
    const u64 dstOffset = GetLocalRecvDisplOfRank(sourceRank) * unitSize_;
    DeviceMem src = outBuffer_.range(srcOffset, copySize);
    DeviceMem dst = outputMem_.range(dstOffset, copySize);

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream));

    // 刷新receive info
    CHK_RET(UpdateLocalRecvInfo(sourceRank, copyCount));

    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][LocalCopy][FromOutBufferToOutput] done, userRank[%u], sourceRank[%u], "
        "srcOffset[%llu], dstOffset[%llu], copyCount[%llu], copySize[%llu], loopIdx[%u]",
        userRank_, sourceRank, srcOffset, dstOffset, copyCount, copySize, loopIdx);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::LocalCopySelfDataFromInputToOutput(Stream& stream)
{
    const u64 copyCount = GetLocalSendCountOfRank(userRank_);
    if (copyCount == 0) {
        return HCCL_SUCCESS;
    }

    // 从input拷贝到output
    const u64 copySize = copyCount * unitSize_;
    const u64 srcOffset = GetLocalSendDisplOfRank(userRank_) * unitSize_;
    const u64 dstOffset = GetLocalRecvDisplOfRank(userRank_) * unitSize_;
    DeviceMem src = inputMem_.range(srcOffset, copySize);
    DeviceMem dst = outputMem_.range(dstOffset, copySize);

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream));

    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][LocalCopy][SelfDataFromInputToOutput] done, userRank[%u], "
        "srcOffset[%llu], dstOffset[%llu], copyCount[%llu], copySize[%llu]",
        userRank_, srcOffset, dstOffset, copyCount, copySize);
    return HCCL_SUCCESS;
}

HcclResult
AlltoallvContinuousPipeline::SdmaSendFromInputToRemoteOutBuffer(const u32 targetRank, Stream& stream, const u32 loopIdx)
{
    const u64 sendCount = std::min(GetLocalSendCountOfRank(targetRank), countsPerBlock_);
    if (sendCount == 0) {
        return HCCL_SUCCESS;
    }

    // 从input发送到remote out buffer，目的位置是第[本userRank_]个分块
    const u64 sendSize = sendCount * unitSize_;
    const u64 srcOffset = GetLocalSendDisplOfRank(targetRank) * unitSize_;
    const u64 dstOffset = GetDataBlockOffset(userRank_, loopIdx);
    DeviceMem src = inputMem_.range(srcOffset, sendSize);

    const LINK& link = intraLinks_[targetRank % intraRankSize_];
    void* remMemPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
    DeviceMem dst = DeviceMem::create(static_cast<u8*>(remMemPtr) + dstOffset, sendSize);

    // 前后同步在外层处理
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream, targetRank, link->GetLinkType()));

    // 刷新send info
    CHK_RET(UpdateLocalSendInfo(targetRank, sendCount));

    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][Sdma][SendFromInputToRemoteOutBuffer] done, userRank[%u], "
        "targetRank[%u], srcOffset[%llu], dstOffset[%llu], sendCount[%llu], sendSize[%llu], loopIdx[%u]",
        userRank_, targetRank, srcOffset, dstOffset, sendCount, sendSize, loopIdx);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::SdmaReadFromRemoteOutBufferToOutput(
    const u32 sourceRank, Stream& stream, const u32 loopIdx)
{
    // 需要recv counts信息
    CHK_PRT_RET(
        needCollectInfo_,
        HCCL_ERROR("[AlltoallvContinuousPipeline][SdmaReadFromRemoteOutBufferToOutput] No receive info."),
        HCCL_E_INTERNAL);

    const u64 readCount = std::min(GetLocalRecvCountOfRank(sourceRank), countsPerBlock_);
    if (readCount == 0) {
        return HCCL_SUCCESS;
    }

    // 从remote out buffer读取到output，源位置是第[sourceRank / intraRankSize_ * intraRankSize_ + intraRankId_]个分块
    const u64 readSize = readCount * unitSize_;
    const u64 srcBlockIdx = sourceRank / intraRankSize_ * intraRankSize_ + intraRankId_;
    const u64 srcOffset = GetDataBlockOffset(srcBlockIdx, loopIdx);
    const u64 dstOffset = GetLocalRecvDisplOfRank(sourceRank) * unitSize_;

    const LINK& link = intraLinks_[sourceRank % intraRankSize_];
    void* remMemPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
    DeviceMem src = DeviceMem::create(static_cast<u8*>(remMemPtr) + srcOffset, readSize);

    DeviceMem dst = outputMem_.range(dstOffset, readSize);

    // 前后同步在外层处理
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream, sourceRank, link->GetLinkType()));

    // 刷新receive info
    CHK_RET(UpdateLocalRecvInfo(sourceRank, readCount));

    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][Sdma][ReadFromRemoteOutBufferToOutput] done, userRank[%u], "
        "sourceRank[%u], srcOffset[%llu], dstOffset[%llu], readCount[%llu], readSize[%llu], loopIdx[%u]",
        userRank_, sourceRank, srcOffset, dstOffset, readCount, readSize, loopIdx);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::InterSendAndReceive(
    const u32 sendRank, const u32 recvRank, Stream& stream, const u32 loopIdx)
{
    // 计算出对端rank所在module的首rank，因为要向对端rank发送[首rank, 首rank+intraRankSize]rank的数据
    const u32 sendModuleFirstRank = sendRank / intraRankSize_ * intraRankSize_;
    const u32 recvModuleFirstRank = recvRank / intraRankSize_ * intraRankSize_;

    std::vector<TxMemoryInfo> sendMems;
    std::vector<RxMemoryInfo> recvMems;
    sendMems.reserve(intraRankSize_);
    recvMems.reserve(intraRankSize_);

    const LINK& sendLink = interLinks_[sendRank / intraRankSize_];
    const LINK& recvLink = interLinks_[recvRank / intraRankSize_];
    const bool isSDMALink = sendLink->IsSpInlineReduce() || recvLink->IsSpInlineReduce();

    for (u32 rankOffset = 0; rankOffset < intraRankSize_; ++rankOffset) {
        const u32 targetRank = sendModuleFirstRank + rankOffset;
        const u64 sendSrcOffset = GetDataBlockOffset(targetRank, loopIdx);
        const u64 sendDstOffset = GetDataBlockOffset(interRankId_ * intraRankSize_ + rankOffset, loopIdx);
        const u64 sendSize = inBufferDataSize_[targetRank];
        inBufferDataSize_[targetRank] = 0;
        if (sendSize > 0) {
            sendMems.emplace_back(TxMemoryInfo{
                UserMemType::OUTPUT_MEM, sendDstOffset, static_cast<s8*>(inBuffer_.ptr()) + sendSrcOffset, sendSize});
        }
        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][InterSendAndReceive]inter send userRank[%u], sendRank[%u], "
            "targetRank[%u], srcOffset[%llu], dstOffset[%llu], sendSize[%llu], loopIdx[%u]",
            userRank_, sendRank, targetRank, sendSrcOffset, sendDstOffset, sendSize, loopIdx);

        const u32 sourceRank = recvModuleFirstRank + intraRankId_;
        const u32 actualTargetRank = interRankId_ * intraRankSize_ + rankOffset;
        const u64 recvSrcOffset = GetDataBlockOffset(actualTargetRank, loopIdx);
        const u64 recvDstOffset = GetDataBlockOffset(recvModuleFirstRank + rankOffset, loopIdx);
        const u64 recvCount = std::min(countsPerBlock_, intraRecvCounts_[rankOffset][sourceRank]);
        const u64 recvSize = recvCount * unitSize_;
        if (recvCount > 0) {
            recvMems.emplace_back(RxMemoryInfo{
                UserMemType::INPUT_MEM, recvSrcOffset, static_cast<s8*>(outBuffer_.ptr()) + recvDstOffset, recvSize});
            intraRecvCounts_[rankOffset][sourceRank] -= recvCount;
        }
        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][InterSendAndReceive]inter recv userRank[%u], recvRank[%u], "
            "sourceRank[%u], targetRank[%u], srcOffset[%llu], dstOffset[%llu], readSize[%llu], loopIdx[%u]",
            userRank_, recvRank, sourceRank, actualTargetRank, recvSrcOffset, recvDstOffset, recvSize, loopIdx);
    }
    if (isSDMALink) {
        // SDMA读
        CHK_RET(InterSdmaRx(recvLink, sendLink, sendMems, recvMems, stream));
    } else {
        // RDMA
        CHK_RET(InterRdmaTxRx(recvLink, sendLink, sendMems, recvMems, stream));
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::DoSdmaSync(const SdmaSyncType syncType)
{
    for (u32 rank = 0; rank < intraRankSize_; ++rank) {
        if (rank == intraRankId_) {
            continue;
        }
        const u32 streamIndex = GetSdmaSubStreamIdx(rank);
        const LINK& link = intraLinks_[rank];
        Stream& subStream = sdmaSubStreams_[streamIndex];
        if (syncType == SdmaSyncType::PRE_SYNC) {
            // 前同步
            CHK_RET(link->TxAck(subStream));
            CHK_RET(link->RxAck(subStream));
        } else {
            // 尾同步
            CHK_RET(link->TxDataSignal(subStream));
            CHK_RET(link->RxDataSignal(subStream));
        }
    }
    HCCL_DEBUG("[AlltoallvContinuousPipeline][DoSdmaSync] Sync done, syncType[%d].", syncType);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::DoLocalCopy(const u32 beginStepNum, const u32 endStepNum, const u32 loopIdx)
{
    CHK_PRT_RET(
        beginStepNum == endStepNum,
        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][DoLocalCopy]beginStepNum[%u] == endStepNum[%u], return success.",
            beginStepNum, endStepNum),
        HCCL_SUCCESS);

    for (u32 step = beginStepNum + 1; step < endStepNum + 1; ++step) {
        const u32 sendModuleId = (interRankId_ + step) % interRankSize_;
        for (const auto remoteRank : ranksPerModule_[sendModuleId]) {
            CHK_RET(LocalCopyFromInputToInBuffer(remoteRank, mainStream_, loopIdx));
        }
    }
    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][DoLocalCopy] done. beginStepNum[%u], endStepNum[%u], loopIdx[%u]", beginStepNum,
        endStepNum, loopIdx);
    return HCCL_SUCCESS;
}

HcclResult
AlltoallvContinuousPipeline::DoIntraDistribution(const u32 beginStepNum, const u32 endStepNum, const u32 loopIdx)
{
    CHK_PRT_RET(
        beginStepNum == endStepNum,
        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][DoIntraDistribution]beginStepNum[%u] == endStepNum[%u], return "
            "success.",
            beginStepNum, endStepNum),
        HCCL_SUCCESS);

    for (u32 step = beginStepNum + 1; step < endStepNum + 1; ++step) {
        const u32 recvModuleId = (interRankId_ + interRankSize_ - step) % interRankSize_;
        HCCL_DEBUG("[AlltoallvContinuousPipeline][DoIntraDistribution] recvModuleId[%u]", recvModuleId);
        for (const auto remoteRank : ranksPerModule_[recvModuleId]) {
            const u32 remoteIntraRank = remoteRank % intraRankSize_;
            HCCL_DEBUG("[AlltoallvContinuousPipeline][DoIntraDistribution] remoteIntraRank[%u]", remoteIntraRank);
            if (intraRankId_ == remoteIntraRank) {
                // 如果是同号卡，直接从out buffer拷贝到output
                CHK_RET(LocalCopyFromOutBufferToOutput(remoteRank, mainStream_, loopIdx));
            } else {
                // 如果不是同号卡，从module内它对应的同号卡获取
                const u32 streamIndex = GetSdmaSubStreamIdx(remoteIntraRank);
                HCCL_DEBUG(
                    "[AlltoallvContinuousPipeline][DoIntraDistribution] streamIndex[%u], sdmaSubStreams_.size()[%zu]",
                    streamIndex, sdmaSubStreams_.size());
                Stream& subStream = sdmaSubStreams_[streamIndex];
                CHK_RET(SdmaReadFromRemoteOutBufferToOutput(remoteRank, subStream, loopIdx));
            }
        }
    }

    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][DoIntraDistribution] done. beginStepNum[%u], endStepNum[%u], loopIdx[%u]",
        beginStepNum, endStepNum, loopIdx);
    return HCCL_SUCCESS;
}

HcclResult
AlltoallvContinuousPipeline::DoInterSendReceive(const u32 beginStepNum, const u32 endStepNum, const u32 loopIdx)
{
    CHK_PRT_RET(
        beginStepNum == endStepNum,
        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][DoInterSendReceive]beginStepNum[%u] == endStepNum[%u], return "
            "success.",
            beginStepNum, endStepNum),
        HCCL_SUCCESS);

    u32 streamIdx = 0;
    for (u32 step = beginStepNum + 1; step < endStepNum + 1; ++step) {
        const u32 sendRank = (userRank_ + step * intraRankSize_) % userRankSize_;
        const u32 recvRank = (userRank_ + userRankSize_ - step * intraRankSize_) % userRankSize_;
        CHK_RET(InterSendAndReceive(sendRank, recvRank, rdmaSubStreams_[streamIdx++], loopIdx));
    }
    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][DoInterSendReceive] done. beginStepNum[%u], endStepNum[%u], loopIdx[%u]",
        beginStepNum, endStepNum, loopIdx);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::DoLevel0LocalCopy(const u32 loopIdx)
{
    for (const auto remoteRank : ranksPerModule_[interRankId_]) {
        if (remoteRank == userRank_) {
            continue;
        }
        CHK_RET(LocalCopyFromOutBufferToOutput(remoteRank, mainStream_, loopIdx));
    }
    HCCL_DEBUG("[AlltoallvContinuousPipeline][DoLevel0LocalCopy] done, loopIdx[%u].", loopIdx);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::DoLevel0SdmaSend(const u32 loopIdx)
{
    for (const auto remoteRank : ranksPerModule_[interRankId_]) {
        if (remoteRank == userRank_) {
            continue;
        }
        const u32 remoteIntraRank = remoteRank % intraRankSize_;
        const u32 streamIndex = GetSdmaSubStreamIdx(remoteIntraRank);
        Stream& subStream = sdmaSubStreams_[streamIndex];
        CHK_RET(SdmaSendFromInputToRemoteOutBuffer(remoteRank, subStream, loopIdx));
    }

    HCCL_DEBUG("[AlltoallvContinuousPipeline][DoLevel0SdmaSend] done, loopIdx[%u].", loopIdx);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::DoIntraInfoBroadcast()
{
    for (const auto remoteRank : ranksPerModule_[interRankId_]) {
        if (remoteRank == userRank_) {
            continue;
        }
        const u32 remoteIntraRank = remoteRank % intraRankSize_;
        const u32 streamIndex = GetSdmaSubStreamIdx(remoteIntraRank);
        Stream& subStream = sdmaSubStreams_[streamIndex];

        const LINK& link = intraLinks_[remoteRank % intraRankSize_];
        void* remInPtr = nullptr;
        void* remOutPtr = nullptr;
        CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &remInPtr));
        CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &remOutPtr));

        // 前后同步在外层处理，直接发送
        // 发送counts信息，从output发送到remote out buffer，目的位置是第[本userRank_]个info分块
        const u64 infoSize = userRankSize_ * sizeof(u64);
        const u64 infoOffset = infoOffsets_[userRank_];
        DeviceMem infoSrc = outBuffer_.range(infoOffset, infoSize);
        DeviceMem infoDst = DeviceMem::create(static_cast<u8*>(remOutPtr) + infoOffset, infoSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, infoDst, infoSrc, subStream, remoteRank, link->GetLinkType()));

        // 发送flag，flag值为[LocalLoopNum + 1]，从input发送到remote in buffer，目的位置是第[本userRank_]个u32
        const u64 flagSize = sizeof(u32);
        const u64 flagOffset = infoOffsets_[0] + userRank_ * flagSize;
        DeviceMem flagSrc = inBuffer_.range(flagOffset, flagSize);
        DeviceMem flagDst = DeviceMem::create(static_cast<u8*>(remInPtr) + flagOffset, flagSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, flagDst, flagSrc, subStream, remoteRank, link->GetLinkType()));

        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][DoIntraInfoBroadcast] userRank[%u], send info to remoteRank[%u], "
            "infoOffset[%llu], infoSize[%llu], flagOffset[%llu], flagSize[%llu]",
            userRank_, remoteRank, infoOffset, infoSize, flagOffset, flagSize);
    }

    HCCL_DEBUG("[AlltoallvContinuousPipeline][DoIntraInfoBroadcast] done.");
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::DoLocalWriteInfoAndFlagAndInterSync()
{
    HCCL_DEBUG("[AlltoallvContinuousPipeline][DoLocalWriteInfoAndFlagAndInterSync] start.");

    if (!needCollectInfo_) {
        return HCCL_SUCCESS;
    }

    // 将counts信息写到out buffer的info区域
    void* infoPtr = localRecvCounts_.data();
    const u64 infoSize = userRankSize_ * sizeof(u64);
    const u64 infoOffset = infoOffsets_[userRank_];
    DeviceMem infoSrc = DeviceMem::create(infoPtr, infoSize);
    DeviceMem infoDst = outBuffer_.range(infoOffset, infoSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, infoDst, infoSrc, mainStream_));

    // 将in buffer的flag区域刷0，第[userRank]个u32设为[LocalLoopNum + 1]
    const u64 flagAreaSize = userRankSize_ * sizeof(u32);
    flagAreaRefreshData_[userRank_] = intraLoopNum_ + 1; // +1是应对LocalLoopNum为0的情况
    DeviceMem flagSrc = DeviceMem::create(flagAreaRefreshData_.data(), flagAreaSize);
    DeviceMem flagDst = inBuffer_.range(infoOffsets_[0], flagAreaSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, flagDst, flagSrc, mainStream_));

    // 在主流上下一个搬1的任务，kernel可以通过轮询dst是否为1，确保flag区域已被刷值，避免flag区域还是随机值时就开始轮询。
    DeviceMem refreshFlagSrc = DeviceMem::create(&flagAreaRefreshValue_, sizeof(flagAreaRefreshValue_));
    DeviceMem refreshFlagDst = DeviceMem::create(&flagAreaRefreshFlag_, sizeof(flagAreaRefreshFlag_));
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, refreshFlagDst, refreshFlagSrc, mainStream_));

    CHK_RET(LaunchTask(dispatcher_, mainStream_));

    HCCL_INFO(
        "[AlltoallvContinuousPipeline][DoLocalWriteInfoAndFlagAndInterSync] write counts and flag. userRank[%u], "
        "infoPtr[%p], infoSize[%llu], infoOffset[%llu]",
        userRank_, infoPtr, infoSize, infoOffset);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::WaitValueOfRank(const u32 rank, const HcclUs& startTimeUs, u32& value)
{
    const auto* valuePtr = reinterpret_cast<u32*>(static_cast<u8*>(inBuffer_.ptr()) + infoOffsets_[0]) + rank;
    HcclUs lastUt = startTimeUs;
    s64 timeout = static_cast<s64>(waitFlagTimeoutSec_) * 1000 * 1000;
    constexpr s64 printStateInterval = 30 * 1000 * 1000; // 每隔30s打印一次状态
    HCCL_DEBUG(
        "[AlltoallvContinuousPipeline][WaitValueOfRank] start waiting value of rank[%u], valuePtr[%p], timeout[%llu]s.",
        rank, valuePtr, waitFlagTimeoutSec_);

    while (flagAreaRefreshFlag_ == 0 || *valuePtr == 0) {
        const HcclUs currentUt = TIME_NOW();
        // 等待value过程，每隔30秒打印一次状态
        if (DURATION_US(currentUt - lastUt).count() > printStateInterval) {
            lastUt = currentUt;
            if (flagAreaRefreshFlag_ == 0) {
                HCCL_RUN_INFO(
                    "[AlltoallvContinuousPipeline][WaitValueOfRank] The Previous task has not been completed."
                    " userRank[%u]",
                    userRank_);
            } else {
                HCCL_RUN_INFO("[AlltoallvContinuousPipeline][WaitValueOfRank] waiting value of rank[%u]", rank);
            }
        }

        CHK_PRT_RET(
            DURATION_US(currentUt - startTimeUs).count() > timeout,
            HCCL_ERROR(
                "[AlltoallvContinuousPipeline][WaitValueOfRank] Waiting for the value of rank[%u] timed out.", rank),
            HCCL_E_TIMEOUT);
    }
    value = *valuePtr;

    // 每次执行算子开头都有重置flag区域的task，所以此处不需要重置为0

    HCCL_DEBUG("[AlltoallvContinuousPipeline][WaitValueOfRank] Got value of rank[%u], value[%u].", rank, value);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::WaitAndCalReceiveInfo()
{
    // 计算自己以及module内其他卡的receive count
    HCCL_DEBUG("[AlltoallvContinuousPipeline][WaitAndCalReceiveInfo] start.");
    const HcclUs startUt = TIME_NOW();

    for (u32 intraRankIdx = 0; intraRankIdx < intraRankSize_; ++intraRankIdx) {
        if (intraRankIdx == intraRankId_) {
            continue;
        }
        const u32 remoteRank = interRankId_ * intraRankSize_ + intraRankIdx;
        u32 remoteValue = 0;
        CHK_RET(WaitValueOfRank(remoteRank, startUt, remoteValue));

        const auto* countsPtr = reinterpret_cast<u64*>(static_cast<u8*>(outBuffer_.ptr()) + infoOffsets_[remoteRank]);
        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][WaitAndCalReceiveInfo] remoteRank[%u], infoOffset[%llu], "
            "countsPtr[%p]",
            remoteRank, infoOffsets_[remoteRank], countsPtr);

        for (u32 i = 0; i < userRankSize_; ++i) {
            HCCL_DEBUG("[AlltoallvContinuousPipeline][WaitAndCalReceiveInfo] countsPtr[%u]=[%llu]", i, countsPtr[i]);
            intraRecvCounts_[intraRankIdx][i] = countsPtr[i];
        }

        const u32 remoteLoopNum = remoteValue - 1; // remoteValue一定大于0
        intraLoopNum_ = std::max(intraLoopNum_, remoteLoopNum);
        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][WaitAndCalReceiveInfo] remoteRank[%u], remoteLoopNum[%u], "
            "intraLoopNum_[%u]",
            remoteRank, remoteLoopNum, intraLoopNum_);
    }

    HCCL_DEBUG("[AlltoallvContinuousPipeline][WaitAndCalReceiveInfo] done. loopNum[%u]", intraLoopNum_);
    return HCCL_SUCCESS;
}

HcclResult AlltoallvContinuousPipeline::RunAsync()
{
    // 在开始前，先将counts信息拷贝到info区域，并且刷新一下flag区域
    CHK_RET(DoLocalWriteInfoAndFlagAndInterSync());

    // 按照机间pairwise的方式计算每轮的步数，等于level1的rank size - 1
    const u32 stepsPerLoop = interRankSize_ - 1;

    // 需要发给其他module的每块数据都会经历三步：本地拷贝至in buffer、经RDMA链路发送到同号卡、由同号卡用SDMA分发到接收卡
    TaskState localCopyState;
    TaskState interState;
    TaskState intraState;

    localCopyState.stepNumNext = std::min(rdmaConcurrentNum_, stepsPerLoop);

    // 外层loop，要重复多少轮，默认为0，在获取到全局counts信息后刷新
    u32 repeatLoopNum = 0;

    // 第一步，需要把counts信息广播给机内其他rank
    bool needDoIntraInfoBroadcast = needCollectInfo_;

    while (localCopyState.stepNum < stepsPerLoop || interState.stepNum < stepsPerLoop
           || intraState.stepNum < stepsPerLoop) {
        // 每一轮首次做跨module收发的同时，做level0的SDMA写，每张卡从input写到对端的out buffer
        const bool needDoLevel0SdmaWrite = (interState.stepNumNext != 0 && interState.stepNum == 0);
        // 每一轮首次做机内分发的同时，每张卡从out buffer将level0其他卡发来的数据拷至output
        const bool needDoLevel0LocalCopy = (intraState.stepNumNext != 0 && intraState.stepNum == 0);

        // intra的stepNum小于stepNumNext，说明本轮需要做intra分发(SDMA)
        const bool needDoIntraTasks = intraState.stepNum < intraState.stepNumNext;
        // inter的stepNum小于stepNumNext，说明本轮需要做inter收发(RDMA)
        const bool needDoInterTasks = interState.stepNum < interState.stepNumNext;

        const bool hasSdmaTask = needDoIntraTasks || needDoLevel0SdmaWrite || needDoIntraInfoBroadcast;
        const bool hasRdmaTask = needDoInterTasks;

        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][RunAsyncLoop][start] localCopy[step:%u, next:%u, loop:%u], "
            "inter[step:%u, next:%u, loop:%u], intra[step:%u, next:%u, loop:%u]",
            localCopyState.stepNum, localCopyState.stepNumNext, localCopyState.loopNum, interState.stepNum,
            interState.stepNumNext, interState.loopNum, intraState.stepNum, intraState.stepNumNext, intraState.loopNum);

        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][RunAsyncLoop] needDoLevel0SdmaWrite[%d], needDoLevel0LocalCopy[%d], "
            "hasSdmaTask[%d], hasRdmaTask[%d]",
            needDoLevel0SdmaWrite, needDoLevel0LocalCopy, hasSdmaTask, hasRdmaTask);

        if (hasSdmaTask) {
            // 本轮有SDMA任务。主流通知SDMA从流，SDMA从流等待主流，前同步
            CHK_RET(NotifySdmaSubStreamStart());
            CHK_RET(DoSdmaSync(SdmaSyncType::PRE_SYNC));

            // 下发一组主从流同步，拉齐SDMA任务，避免任务不同时拉起导致性能下降
            CHK_RET(WaitSdmaSubStreamFinish());
            CHK_RET(NotifySdmaSubStreamStart());
        }

        if (hasRdmaTask) {
            // 本轮有RDMA任务。主流通知RDMA从流，RDMA从流等待主流
            CHK_RET(NotifyRdmaSubStreamStart());
        }

        if (needDoLevel0SdmaWrite) {
            CHK_RET(DoLevel0SdmaSend(interState.loopNum));

            if (interState.loopNum == 0) {
                // 首轮，本卡input到output的拷贝也在这时做
                CHK_RET(LocalCopySelfDataFromInputToOutput(mainStream_));
            }
        }

        if (interState.loopNum == 0 && needDoInterTasks) {
            // 第一轮，在做inter分发前，等待、获取receive信息
            if (needCollectInfo_) {
                CHK_RET(WaitAndCalReceiveInfo()); // 阻塞函数
                needCollectInfo_ = false;
            }

            // 刷新重复轮数：总轮数-1
            repeatLoopNum = intraLoopNum_ > 0 ? intraLoopNum_ - 1 : 0;
            if (localCopyState.stepNum == stepsPerLoop && localCopyState.loopNum < repeatLoopNum) {
                // 如果需要做多轮，在此处立即刷新local copy stepNum，让第二轮的任务尽早开始
                ++localCopyState.loopNum;
                localCopyState.stepNum = 0;
                localCopyState.stepNumNext = std::min(rdmaConcurrentNum_, stepsPerLoop);
            }
        }

        // intraStepNum小于interStepNum，表示需要做机内分发
        CHK_RET(DoIntraDistribution(intraState.stepNum, intraState.stepNumNext, intraState.loopNum));
        // interStepNum小于localCopyStepNum，表示需要做机间收发
        CHK_RET(DoInterSendReceive(interState.stepNum, interState.stepNumNext, interState.loopNum));
        // localCopyStepNum小于stepsPerLoop，根据并发度拷贝需要的数据到in buffer
        CHK_RET(DoLocalCopy(localCopyState.stepNum, localCopyState.stepNumNext, localCopyState.loopNum));

        if (needDoIntraInfoBroadcast) {
            // 第一轮第一步，机内广播本卡的counts信息
            CHK_RET(DoIntraInfoBroadcast());
            needDoIntraInfoBroadcast = false;
        }

        if (needDoLevel0LocalCopy) {
            CHK_RET(DoLevel0LocalCopy(intraState.loopNum));
        }

        if (hasSdmaTask) {
            // SDMA尾同步，主流等待SDMA从流，SDMA从流通知主流
            CHK_RET(DoSdmaSync(SdmaSyncType::POST_SYNC));
            CHK_RET(WaitSdmaSubStreamFinish());
        }
        if (hasRdmaTask) {
            // 主流等待RDMA从流，RDMA从流通知主流
            CHK_RET(WaitRdmaSubStreamFinish());
        }

        // 下发task
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, subStreams_));

        // 更新每种任务的当前步数
        intraState.stepNum = intraState.stepNumNext;
        interState.stepNum = interState.stepNumNext;
        localCopyState.stepNum = localCopyState.stepNumNext;

        // 更新每种任务的下一步目标步数
        intraState.stepNumNext = interState.stepNumNext;
        interState.stepNumNext = localCopyState.stepNumNext;
        localCopyState.stepNumNext = std::min(localCopyState.stepNumNext + rdmaConcurrentNum_, stepsPerLoop);

        // 检查是否需要重复执行，若需要，将对应的step num刷回为0
        if (intraState.stepNum == stepsPerLoop && intraState.loopNum < interState.loopNum) {
            ++intraState.loopNum;
            intraState.stepNum = 0;
            intraState.stepNumNext = std::min(rdmaConcurrentNum_, stepsPerLoop);
        }
        if (interState.stepNum == stepsPerLoop && interState.loopNum < localCopyState.loopNum) {
            ++interState.loopNum;
            interState.stepNum = 0;
            interState.stepNumNext = std::min(rdmaConcurrentNum_, stepsPerLoop);
        }
        if (localCopyState.stepNum == stepsPerLoop && localCopyState.loopNum < repeatLoopNum) {
            ++localCopyState.loopNum;
            localCopyState.stepNum = 0;
            localCopyState.stepNumNext = std::min(rdmaConcurrentNum_, stepsPerLoop);
        }

        HCCL_DEBUG(
            "[AlltoallvContinuousPipeline][RunAsyncLoop][end] localCopy[step:%u, next:%u, loop:%u], "
            "inter[step:%u, next:%u, loop:%u], intra[step:%u, next:%u, loop:%u]",
            localCopyState.stepNum, localCopyState.stepNumNext, localCopyState.loopNum, interState.stepNum,
            interState.stepNumNext, interState.loopNum, intraState.stepNum, intraState.stepNumNext, intraState.loopNum);
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_2_ALL_V_CONTINUOUS_PIPELINE, AlltoallvContinuousPipeline);
} // namespace hccl
