/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allltoall_pipeline_mesh_pairwise_ping_pong.h"
#include <numeric>
#include "alg_template_register.h"

namespace hccl {

// 需要将 ccl 切成两份，ping-pong 时也根据收发次数取模2决定使用 ping mem 还是 pong mem
static const u32 PING_PONG_CONST_NUM = 2;
static const u32 INTRA_STREAM_INFO_SENDLEN_INDEX = 0;           // intraStreamInfo 中 sendLen 的下标
static const u32 INTRA_STREAM_INFO_RECVLEN_INDEX = 1;           // intraStreamInfo 中 recvLen 的下标
static const u32 INTRA_STREAM_INFO_RECV_LOCAL_OFFSET_INDEX = 2; // intraStreamInfo 中 recvRemoteOffset 的下标

AlltoallPipelineMeshPairwisePingPong::AlltoallPipelineMeshPairwisePingPong(const HcclDispatcher dispatcher)
    : AlltoallPipelineBase(dispatcher)
{}

AlltoallPipelineMeshPairwisePingPong::~AlltoallPipelineMeshPairwisePingPong() {}

u32 AlltoallPipelineMeshPairwisePingPong::CalcInterNumSteps() { return interRankSize_ - 1; }

// 适配新CollExecutor接口
HcclResult AlltoallPipelineMeshPairwisePingPong::Prepare(
    u32 userRank, A2aPipelineMemory A2aPipelineMemory, const SubCommInfo& level0CommInfo,
    const SubCommInfo& level1CommInfo, Stream& mainStream, std::vector<Stream>& subStream,
    std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub,
    std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, HcclWorkflowMode workMode)
{
    AlltoallPipelineBase::Prepare(
        userRank, A2aPipelineMemory, level0CommInfo, level1CommInfo, mainStream, subStream, notifyMain, notifySub,
        allMeshAggregationSendRecvInfo, workMode);
    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        pingPongMemSize_ = (cclIn_.size() / PING_PONG_CONST_NUM);
    } else {
        pingPongMemSize_ = (scratchMem_.size() / PING_PONG_CONST_NUM);
    }
    intraDataBlockSize_ = (pingPongMemSize_ / intraRankSize_);
    if (intraDataBlockSize_ > HCCL_MIN_SLICE_ALIGN_910B) {
        intraDataBlockSize_ = (intraDataBlockSize_ / HCCL_MIN_SLICE_ALIGN_910B) * HCCL_MIN_SLICE_ALIGN_910B;
    }
    memStatusInMesh_ = std::vector<bool>(intraRankSize_, false);
    CHK_RET(DeviceMemMapping());
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwisePingPong::DeviceMemMapping()
{
    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        interTransportSend_ = cclIn_;
        interTransportRecv_ = cclOut_;
        intraTransportSend_ = cclOut_;
        interSendPing_ = interTransportSend_.range(0, pingPongMemSize_);
        interSendPong_ = interTransportSend_.range(pingPongMemSize_, pingPongMemSize_);
        interRecvPing_ = interTransportRecv_.range(0, pingPongMemSize_);
        interRecvPong_ = interTransportRecv_.range(pingPongMemSize_, pingPongMemSize_);
    } else {
        interRecvPing_ = scratchMem_.range(0, pingPongMemSize_);
        interRecvPong_ = scratchMem_.range(pingPongMemSize_, pingPongMemSize_);
    }
    intraSendPing_ = interRecvPing_;
    intraSendPong_ = interRecvPong_;

    for (u32 intraRank = 0; intraRank < intraRankSize_; intraRank++) {
        if (intraRank == intraRankId_) {
            continue;
        }
        LINK& intraNeighboorTransport = intraLinks_[intraRank];
        void* remDMAMemPtr = nullptr;
        CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::INPUT_MEM, &remDMAMemPtr));
        DeviceMem remoteIntraSend = DeviceMem::create(
            static_cast<u8*>(remDMAMemPtr),
            workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ? cclIn_.size() : scratchMem_.size());
        DeviceMem remoteIntraSendPing = remoteIntraSend.range(0, pingPongMemSize_);
        DeviceMem remoteIntraSendPong = remoteIntraSend.range(pingPongMemSize_, pingPongMemSize_);
        intraNeighBoorMemory_[intraRank] = {remoteIntraSendPing, remoteIntraSendPong};
    }
    return HCCL_SUCCESS;
}

// 将需要发送给其他 mesh 的数据准备好，并计算好 TxMemoryInfo
HcclResult AlltoallPipelineMeshPairwisePingPong::PrepareInterSendData(u32 mainStep, u32 subStep)
{
    nextInterSendData_.clear();
    u32 interSendRankStart = ((interRankId_ + 1 + mainStep) % interRankSize_) * intraRankSize_;
    DeviceMem interSendMem = (interSendUsePingMem_ ? interSendPing_ : interSendPong_);
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwisePingPong][PrepareInterSendData] userRank %u, interRank %u, "
        "intraRank %u in main step %llu, sub step %llu send to remote %s",
        userRank_, interRankId_, intraRankId_, mainStep, subStep,
        sendToInterDstMemPing_ ? "interRecvPingMem" : "interRecvPongMem");
    u64 preStepMaxSend = intraDataBlockSize_ * subStep;
    for (u32 i = 0; i < intraRankSize_; i++) {
        u32 dataIndex = i + interSendRankStart;
        u64 totalSendLen = localSendRecvInfo_.sendLength[dataIndex];
        u64 sendLen = std::min(intraDataBlockSize_, std::max(totalSendLen, preStepMaxSend) - preStepMaxSend);
        if (sendLen == 0) {
            continue;
        }
        HCCL_DEBUG(
            "[AlltoallPipelineMeshPairwisePingPong][PrepareInterSendData] userRank %u, interRank %u, "
            "intraRank %u data index %llu move from userInput offset %llu length %llu to %s, total size %llu, "
            "send to remote %s",
            userRank_, interRankId_, intraRankId_, dataIndex, localSendRecvInfo_.sendOffset[dataIndex] + preStepMaxSend,
            sendLen, interSendUsePingMem_ ? "localInterSendPingMem" : "localInterSendPongMem", totalSendLen,
            sendToInterDstMemPing_ ? "interRecvPingMem" : "interRecvPongMem");
        DeviceMem src = inputMem_.range(localSendRecvInfo_.sendOffset[dataIndex] + preStepMaxSend, sendLen);
        DeviceMem dst = interSendMem.range(i * intraDataBlockSize_, sendLen);
        // 单算子模式需要搬到 CCL，图模式省去这一步
        if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
        }
        nextInterSendData_.emplace_back(TxMemoryInfo{
            UserMemType::OUTPUT_MEM, (sendToInterDstMemPing_ ? 0 : pingPongMemSize_) + i * intraDataBlockSize_,
            workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ? dst.ptr() : src.ptr(), sendLen});
    }
    return HCCL_SUCCESS;
}

// 将需要发送给其他 mesh 的数据准备好，并准备好 TxMemoryInfo
HcclResult AlltoallPipelineMeshPairwisePingPong::PrepareInterRecvData(u32 mainStep, u32 subStep)
{
    nextInterRecvData_.clear();
    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        // 单算子模式本次要接收的数据都放在 CCL，直接整块接收
        nextInterRecvData_.emplace_back(RxMemoryInfo{
            UserMemType::INPUT_MEM, recvFromInterSrcMemPing_ ? 0u : pingPongMemSize_,
            (interRecvUsePingMem_ ? interRecvPing_ : interRecvPong_).ptr(), pingPongMemSize_});
    } else {
        // 图模式需要计算数据放在对端 userInput 的位置
        u32 recvFromRank = (userRank_ + groupRankSize_ - (mainStep + 1) * intraRankSize_) % groupRankSize_;
        const std::vector<u64>& remoteSendLength = (*allMeshAggregationSendRecvInfo_)[recvFromRank].sendLength;
        const std::vector<u64>& remoteSendOffset = (*allMeshAggregationSendRecvInfo_)[recvFromRank].sendOffset;
        u64 dataStartOffset = subStep * intraDataBlockSize_;
        for (u32 i = 0; i < intraRankSize_; i++) {
            u64 totalRecvDataLen = remoteSendLength[meshRankStart_ + i];
            u64 recvLen = std::min(std::max(totalRecvDataLen, dataStartOffset) - dataStartOffset, intraDataBlockSize_);
            if (recvLen == 0) {
                continue;
            }
            u64 recvRemoteOffset = remoteSendOffset[meshRankStart_ + i] + dataStartOffset;
            nextInterRecvData_.emplace_back(RxMemoryInfo{
                UserMemType::INPUT_MEM, recvRemoteOffset,
                (interRecvUsePingMem_ ? interRecvPing_ : interRecvPong_).range(i * intraDataBlockSize_, recvLen).ptr(),
                recvLen});
            HCCL_DEBUG(
                "[AlltoallPipelineMeshPairwisePingPong][PrepareInterRecvData] userRank %u, interRank %u, "
                "intraRank %u recv from remote userInput offset %llu length %llu to %s offset %llu",
                userRank_, interRankId_, intraRankId_, recvRemoteOffset, recvLen,
                interRecvUsePingMem_ ? "localInterRecvPingMem" : "localInterRecvPongMem", i * intraDataBlockSize_);
        }
    }
    return HCCL_SUCCESS;
}

// 准备下一次mesh间需要收发的数据，单算子模式需要从 userInput 搬到 CCLBuffer，图模式则仅需要准备好 TxMemoryInfo
HcclResult AlltoallPipelineMeshPairwisePingPong::PrepareInterData(u32 mainStep, u32 subStep)
{
    CHK_RET(PrepareInterSendData(mainStep, subStep));
    CHK_RET(PrepareInterRecvData(mainStep, subStep));
    return HCCL_SUCCESS;
}

// 将原先在userInput，且需要发到本mesh内其他卡的数据搬到CCL
HcclResult AlltoallPipelineMeshPairwisePingPong::PrepareIntraData(u32 subStep)
{
    u64 dataStartOffset = subStep * intraDataBlockSize_;
    for (u32 i = 0; i < intraRankSize_; i++) {
        u32 dataIndex = i + meshRankStart_;
        u64 totalSendDataLen = localSendRecvInfo_.sendLength[dataIndex];
        u64 sendLen = std::min(std::max(totalSendDataLen, dataStartOffset) - dataStartOffset, intraDataBlockSize_);
        if (i == intraRankId_ || sendLen == 0) {
            continue;
        }
        DeviceMem src = inputMem_.range(localSendRecvInfo_.sendOffset[dataIndex] + dataStartOffset, sendLen);
        DeviceMem dst
            = (intraSendUsePingMem_ ? intraSendPing_ : intraSendPong_).range(i * intraDataBlockSize_, sendLen);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
        HCCL_DEBUG(
            "[AlltoallPipelineMeshPairwisePingPong][PrepareIntraData] userRank %u, interRank %u, intraRank %u"
            "data index %u move from userInput offset %llu length %llu to %s, total size %llu ",
            userRank_, interRankId_, intraRankId_, dataIndex,
            localSendRecvInfo_.sendOffset[dataIndex] + dataStartOffset, sendLen,
            intraSendUsePingMem_ ? "IntraPingMem" : "IntraPongMem", totalSendDataLen);
    }
    return HCCL_SUCCESS;
}

// 计算mesh内其他卡此时RDMA接收到的数据是在 cclIn 还是 cclOut
void AlltoallPipelineMeshPairwisePingPong::UpdateRemoteMemStatusIntra(u32 step)
{
    for (u32 intraRank = 0; intraRank < intraRankSize_; intraRank++) {
        if (intraRank == intraRankId_)
            continue;
        u32 intraRankHaveRecv = 0;
        for (u32 i = 1; i <= step; i++) {
            const std::vector<u64>& intraRankRecvFrom
                = (*allMeshAggregationSendRecvInfo_)
                      [(meshRankStart_ + groupRankSize_ + intraRank - i * intraRankSize_) % groupRankSize_]
                          .sendLength;
            u64 maxRecvLen = std::accumulate(
                intraRankRecvFrom.begin() + meshRankStart_, intraRankRecvFrom.begin() + meshRankStart_ + intraRankSize_,
                0ULL, [](u64 a, u64 b) {
                    return a > b ? a : b;
                });
            intraRankHaveRecv += ((maxRecvLen + intraDataBlockSize_ - 1) / intraDataBlockSize_);
        }
        memStatusInMesh_[intraRank] = ((intraRankHaveRecv % PING_PONG_CONST_NUM) == 0);
    }
}

// 计算本卡接收数据的那张卡和本卡将要发数据的那张卡在这个大步骤中的
// 第一个小步骤从哪块ccl收以及发到哪块ccl, 每次切换
void AlltoallPipelineMeshPairwisePingPong::UpdateRemoteMemStatusInter(u32 step)
{
    u32 recvGlobalRank = (userRank_ + groupRankSize_ - (step + 1) * intraRankSize_) % groupRankSize_;
    u32 recvInterRank = ((interRankId_ + interRankSize_ - (step + 1)) % interRankSize_);
    u32 numRecvRankHaveSend = 0;
    u32 numSendRankHaveRecv = 0;
    const std::vector<u64>& recvRankSendLen = (*allMeshAggregationSendRecvInfo_)[recvGlobalRank].sendLength;
    for (u32 i = 1; i <= step; i++) {
        u32 firstBlockIndex = (((recvInterRank + i) % interRankSize_) * intraRankSize_);
        u64 maxSendLen = std::accumulate(
            recvRankSendLen.begin() + firstBlockIndex, recvRankSendLen.begin() + firstBlockIndex + intraRankSize_, 0ULL,
            [](u64 a, u64 b) {
                return a > b ? a : b;
            });
        numRecvRankHaveSend += ((maxSendLen + intraDataBlockSize_ - 1) / intraDataBlockSize_);
        const std::vector<u64>& sendRankRecvFrom
            = (*allMeshAggregationSendRecvInfo_)[(userRank_ + i * intraRankSize_) % groupRankSize_].sendLength;
        u64 maxRecvLen = std::accumulate(
            sendRankRecvFrom.begin() + meshRankStart_, sendRankRecvFrom.begin() + meshRankStart_ + intraRankSize_, 0ULL,
            [](u64 a, u64 b) {
                return a > b ? a : b;
            });
        numSendRankHaveRecv += ((maxRecvLen + intraDataBlockSize_ - 1ULL) / intraDataBlockSize_);
    }
    // 首次默认都从对端pingMem读，本卡接收数据来源的那张卡每发一次数据切换一次
    recvFromInterSrcMemPing_ = ((numRecvRankHaveSend % PING_PONG_CONST_NUM) == 0);
    // 首次默认发到对端pingMem，本卡发送数据目的地的那张卡每接收一次数据切换一次
    sendToInterDstMemPing_ = ((numSendRankHaveRecv % PING_PONG_CONST_NUM) == 0);
}

// 收集本次 SDMA 子步骤每条流需要收发的长度，偏移地址，内存状态信息避免重复计算影响性能
void AlltoallPipelineMeshPairwisePingPong::UpdateIntraStreamInfo(u32 interRankDistance, u32 subStep)
{
    intraStreamInfo_.clear();
    u32 firstDataBlockIndex = (meshRankStart_ + groupRankSize_ - interRankDistance * intraRankSize_) % groupRankSize_;
    const std::vector<u64>& sendInfo
        = (*allMeshAggregationSendRecvInfo_)[firstDataBlockIndex + intraRankId_].sendLength;
    u64 dataStartOffset = subStep * intraDataBlockSize_;
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwisePingPong][UpdateSDMAStreamInfo] userRank %u, "
        "interRank %u, intraRank %u, interRankDistance %llu, sub step %llu",
        userRank_, interRankId_, intraRankId_, interRankDistance, subStep);
    for (u32 i = 0; i < intraRankSize_; i++) {
        u64 totalSendDataLen = sendInfo[meshRankStart_ + i];
        u64 totalRecvDataLen = localSendRecvInfo_.recvLength[i + firstDataBlockIndex];
        u64 sendLen = std::min(std::max(totalSendDataLen, dataStartOffset) - dataStartOffset, intraDataBlockSize_);
        u64 recvLen = std::min(std::max(totalRecvDataLen, dataStartOffset) - dataStartOffset, intraDataBlockSize_);
        u64 localOffset = localSendRecvInfo_.recvOffset[i + firstDataBlockIndex] + subStep * intraDataBlockSize_;
        if (i != intraRankId_) {
            intraStreamInfo_[i] = {sendLen, recvLen, localOffset};
            HCCL_DEBUG(
                "[AlltoallPipelineMeshPairwisePingPong][UpdateSDMAStreamInfo] userRank %u, interRank %u, "
                "intraRank %u, sdma stream %llu need send %llu and read length %llu to local offset %llu",
                userRank_, interRankId_, intraRankId_, i, sendLen, recvLen, localOffset);
        }
    }
}

HcclResult AlltoallPipelineMeshPairwisePingPong::SendRecvDataIntraMesh()
{
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwisePingPong][ReadDataInMesh] userRank %u, "
        "interRank %u, intraRank %u, sdma stream %s wait main stream",
        userRank_, interRankId_, intraRankId_, GetStreamIndexString().c_str());
    bool anySend = false;
    for (auto& sdmaInfo : intraStreamInfo_) {
        u32 streamIndex = sdmaInfo.first;
        u64 recvLen = sdmaInfo.second[INTRA_STREAM_INFO_RECVLEN_INDEX];
        u64 recvOffset = sdmaInfo.second[INTRA_STREAM_INFO_RECV_LOCAL_OFFSET_INDEX];
        Stream& currStream = subStream_[streamIndex];
        LINK& readTransport = intraLinks_[streamIndex];
        CHK_RET(readTransport->TxAck(currStream));
        CHK_RET(readTransport->RxAck(currStream));
        if (recvLen > 0) {
            DeviceMem src = intraNeighBoorMemory_[streamIndex][(memStatusInMesh_[streamIndex] ? 0 : 1)].range(
                intraRankId_ * intraDataBlockSize_, recvLen);
            DeviceMem dst = outputMem_.range(recvOffset, recvLen);
            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, dst, src, currStream, readTransport->GetRemoteRank(), readTransport->GetLinkType()));
        }
        CHK_RET(readTransport->TxDataSignal(currStream));
        HCCL_DEBUG(
            "[AlltoallPipelineMeshPairwisePingPong][ReadDataInMesh] userRank %u, interRank %u, "
            "intraRank %u, sdma stream %llu read data from remote %s offset %llu len %llu to local %llu",
            userRank_, interRankId_, intraRankId_, streamIndex,
            memStatusInMesh_[streamIndex] ? "IntraSendPingMem" : "IntraSendPongMem", intraRankId_ * intraDataBlockSize_,
            recvLen, recvOffset);
        memStatusInMesh_[streamIndex] = (!memStatusInMesh_[streamIndex]);
        CHK_RET(readTransport->RxDataSignal(currStream));
        anySend = true;
    }
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwisePingPong][ReadDataInMesh] userRank %u, "
        "interRank %u, intraRank %u, sdma stream %s notify main stream",
        userRank_, interRankId_, intraRankId_, GetStreamIndexString().c_str());
    intraSendUsePingMem_ ^= anySend;
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwisePingPong::SendRecvDataInterMesh(u32 step, bool doSend, bool doRecv)
{
    Stream& interStream = subStream_[intraRankId_];
    LINK& interRecvTransport = interLinks_[(interRankId_ + interRankSize_ - 1 - step) % interRankSize_];
    LINK& interSendTransport = interLinks_[(interRankId_ + 1 + step) % interRankSize_];
    if (doRecv) {
        CHK_RET(interRecvTransport->TxAck(interStream));
    }
    if (doSend) {
        CHK_RET(interSendTransport->RxAck(interStream));
        CHK_RET(interSendTransport->TxAsync(
            UserMemType::OUTPUT_MEM, (sendToInterDstMemPing_ ? 0u : pingPongMemSize_),
            (interSendUsePingMem_ ? interSendPing_ : interSendPong_).ptr(), pingPongMemSize_, interStream));
        interSendUsePingMem_ ^= true;
        sendToInterDstMemPing_ ^= true;
    }
    if (doRecv) {
        CHK_RET(interRecvTransport->RxAsync(
            UserMemType::INPUT_MEM, (recvFromInterSrcMemPing_ ? 0u : pingPongMemSize_),
            (interRecvUsePingMem_ ? interRecvPing_ : interRecvPong_).ptr(), pingPongMemSize_, interStream));
        CHK_RET(interRecvTransport->PostFinAck(interStream));
        interRecvUsePingMem_ ^= true;
        recvFromInterSrcMemPing_ ^= true;
    }
    if (doSend) {
        CHK_RET(interSendTransport->WaitFinAck(interStream));
    }
    CHK_RET(ExecuteBarrier(interRecvTransport, interSendTransport, interStream));
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwisePingPong::LocalCopyDataRecvFromInter(u32 mainStep, u32 subStep)
{
    u64 localRecvLen = localSendRecvInfo_
                           .recvLength[(userRank_ + groupRankSize_ - (mainStep + 1) * intraRankSize_) % groupRankSize_];
    u64 localRecvOff = localSendRecvInfo_
                           .recvOffset[(userRank_ + groupRankSize_ - (mainStep + 1) * intraRankSize_) % groupRankSize_];
    u64 currStepRecvLen = std::min(localRecvLen - subStep * intraDataBlockSize_, intraDataBlockSize_);
    DeviceMem src = (interRecvUsePingMem_ ? interRecvPong_ : interRecvPing_)
                        .range(intraRankId_ * intraDataBlockSize_, currStepRecvLen);
    DeviceMem dst = outputMem_.range(localRecvOff + subStep * intraDataBlockSize_, currStepRecvLen);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwisePingPong::PreProcess()
{
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwisePingPong][PreProcess] userRank %u, interRank %u, intraRank %u, "
        "main stream notify RDMA stream %llu start send",
        userRank_, interRankId_, intraRankId_, intraRankId_);
    // 搬下次要做 Server 间收发的数据到 ccl buffer
    CHK_RET(PrepareInterData(0u, 0u));
    // 主流notify RDMA流
    ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_);
    CHK_RET(NotifyInterStreamStart());

    // 先做一部分 mesh 内 SDMA 操作，剩下的数据待到整体 RDMA 做完之后再补
    CHK_RET(PrepareIntraData(0u));
    UpdateIntraStreamInfo(0u, 0u);
    ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_);
    CHK_RET(NotifyIntraStreamStart());
    CHK_RET(SendRecvDataIntraMesh());
    ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_);
    // 主流搬本地那块数据
    DeviceMem src = inputMem_.range(localSendRecvInfo_.sendOffset[userRank_], localSendRecvInfo_.sendLength[userRank_]);
    DeviceMem dst
        = outputMem_.range(localSendRecvInfo_.recvOffset[userRank_], localSendRecvInfo_.recvLength[userRank_]);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
    return HCCL_SUCCESS;
}

// 分别计算当前大步骤需要做几次mesh间收和发，和mesh内收和发（mesh间收的次数和mesh内发的次数相同）
void AlltoallPipelineMeshPairwisePingPong::GetNumSubStep(
    u32 step, u32& interSendSubStep, u32& interRecvSubStep, u32& intraSubStep)
{
    u32 sendRankStart = ((interRankId_ + 1 + step) % interRankSize_) * intraRankSize_;
    u32 recvRankStart = ((interRankId_ + interRankSize_ - 1 - step) % interRankSize_) * intraRankSize_;
    const std::vector<u64>& sendInfo = (*allMeshAggregationSendRecvInfo_)
                                           [(userRank_ + groupRankSize_ - intraRankSize_ * (step + 1)) % intraRankSize_]
                                               .sendLength;
    u64 maxInterSendLen = std::accumulate(
        localSendRecvInfo_.sendLength.begin() + sendRankStart,
        localSendRecvInfo_.sendLength.begin() + sendRankStart + intraRankSize_, 0ULL, [](u64 a, u64 b) {
            return a > b ? a : b;
        });
    u64 maxInterRecvLen = std::accumulate(
        sendInfo.begin() + meshRankStart_, sendInfo.begin() + meshRankStart_ + intraRankSize_, 0ULL, [](u64 a, u64 b) {
            return a > b ? a : b;
        });
    u64 maxIntraRecvLen = std::accumulate(
        localSendRecvInfo_.recvLength.begin() + recvRankStart,
        localSendRecvInfo_.recvLength.begin() + recvRankStart + intraRankSize_, 0ULL, [](u64 a, u64 b) {
            return a > b ? a : b;
        });
    interSendSubStep = (maxInterSendLen + intraDataBlockSize_ - 1) / intraDataBlockSize_;
    interRecvSubStep = (maxInterRecvLen + intraDataBlockSize_ - 1) / intraDataBlockSize_;
    // mesh 的收发步数取决于本卡从其它mesh收到的需要转发到mesh内其他卡的数据以及本卡需要做mesh内读的其他卡数据
    intraSubStep = (std::max(maxInterRecvLen, maxIntraRecvLen) + intraDataBlockSize_ - 1) / intraDataBlockSize_;
}

HcclResult AlltoallPipelineMeshPairwisePingPong::PipelineSend(u32 step, bool isLastStep)
{
    CHK_RET(ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_));
    u64 maxDataBlock = 0;
    for (const SendRecvInfo& info : (*allMeshAggregationSendRecvInfo_)) {
        for (u64 sendLen : info.sendLength) {
            maxDataBlock = std::max(maxDataBlock, sendLen);
        }
    }
    u32 totalSubStep = (maxDataBlock + intraDataBlockSize_ - 1) / intraDataBlockSize_;
    // 计算需要从源端哪块内存收数据和发到哪块目的内存
    u64 localRecvLen
        = localSendRecvInfo_.recvLength[(userRank_ + groupRankSize_ - (step + 1) * intraRankSize_) % groupRankSize_];
    for (u32 subStep = 0; subStep < totalSubStep; subStep++) {
        // RDMA 收发数据
        SendRecvDataInterMesh(step, true, true);
        if ((subStep + 1u) == totalSubStep) {
            CHK_RET(PrepareInterData(step + 1, 0u));
        } else {
            CHK_RET(PrepareInterData(step, subStep + 1u));
        }
        ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_);
        CHK_RET(WaitIntraStreamFinish());
        CHK_RET(WaitInterStreamFinish());
        ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_);
        UpdateIntraStreamInfo(step + 1u, subStep);
        CHK_RET(NotifyIntraStreamStart());
        CHK_RET(SendRecvDataIntraMesh());
        ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_);
        if ((!isLastStep && subStep < totalSubStep) || (isLastStep && subStep < (totalSubStep - 1))) {
            CHK_RET(NotifyInterStreamStart());
        }
        if (localRecvLen > subStep * intraDataBlockSize_) {
            CHK_RET(LocalCopyDataRecvFromInter(step, subStep));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwisePingPong::PostProcess()
{
    CHK_RET(ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_));
    CHK_RET(WaitIntraStreamFinish());
    ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_);
    u64 maxDataBlock = 0;
    for (const SendRecvInfo& info : (*allMeshAggregationSendRecvInfo_)) {
        for (u64 sendLen : info.sendLength) {
            maxDataBlock = std::max(maxDataBlock, sendLen);
        }
    }
    u64 stepLast = (maxDataBlock + intraDataBlockSize_ - 1) / intraDataBlockSize_;
    for (u64 i = 1; i < stepLast; i++) {
        UpdateIntraStreamInfo(0, i);
        CHK_RET(PrepareIntraData(i));
        ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_);
        CHK_RET(NotifyIntraStreamStart());
        ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_);
        CHK_RET(SendRecvDataIntraMesh());
        CHK_RET(WaitIntraStreamFinish());
        CHK_RET(ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(
    TemplateType::TEMPLATE_ALL_2_ALL_PIPELINE_MESH_PAIRWISE_PING_PONG, AlltoallPipelineMeshPairwisePingPong);
} // namespace hccl
