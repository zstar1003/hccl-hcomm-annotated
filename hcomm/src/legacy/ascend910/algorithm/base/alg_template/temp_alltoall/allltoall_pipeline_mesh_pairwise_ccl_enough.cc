/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allltoall_pipeline_mesh_pairwise_ccl_enough.h"
#include "alg_template_register.h"

namespace hccl {

static const u32 INTRA_STREAM_INFO_SENDLEN_INDEX = 0;            // intraStreamInfo 中 sendLen 的下标
static const u32 INTRA_STREAM_INFO_RECVLEN_INDEX = 1;            // intraStreamInfo 中 recvLen 的下标
static const u32 INTRA_STREAM_INFO_RECV_REMOTE_OFFSET_INDEX = 2; // intraStreamInfo 中 recvRemoteOffset 的下标
static const u32 INTRA_STREAM_INFO_RECV_LOCAL_OFFSET_INDEX = 3;  // intraStreamInfo 中 recvLocalOffset 的下标

AlltoallPipelineMeshPairwiseCCLEnough::AlltoallPipelineMeshPairwiseCCLEnough(const HcclDispatcher dispatcher)
    : AlltoallPipelineBase(dispatcher)
{}

AlltoallPipelineMeshPairwiseCCLEnough::~AlltoallPipelineMeshPairwiseCCLEnough() {}

u32 AlltoallPipelineMeshPairwiseCCLEnough::CalcInterNumSteps() { return interRankSize_ - 1; }

// 适配新CollExecutor接口
HcclResult AlltoallPipelineMeshPairwiseCCLEnough::Prepare(
    u32 userRank, A2aPipelineMemory A2aPipelineMemory, const SubCommInfo& level0CommInfo,
    const SubCommInfo& level1CommInfo, Stream& mainStream, std::vector<Stream>& subStream,
    std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub,
    std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, HcclWorkflowMode workMode)
{
    AlltoallPipelineBase::Prepare(
        userRank, A2aPipelineMemory, level0CommInfo, level1CommInfo, mainStream, subStream, notifyMain, notifySub,
        allMeshAggregationSendRecvInfo, workMode);
    GetIntraScratchOffset();
    CHK_RET(DeviceMemMapping());
    return HCCL_SUCCESS;
}

// 统一计算每步 mesh 内收发时从各卡 scratch 读取的 offset 和 length
HcclResult AlltoallPipelineMeshPairwiseCCLEnough::GetIntraScratchOffset()
{
    for (u32 i = 0; i < intraRankSize_; i++) {
        intraScratchOffsetMap_[i] = std::vector<u64>();
        intraScratchLengMap_[i] = std::vector<u64>();
        u64 startOffset = 0;
        for (u32 remoteRank = i; remoteRank < groupRankSize_; remoteRank += intraRankSize_) {
            if (remoteRank == userRank_) {
                localScratchOffset_ = startOffset;
            }
            const std::vector<u64>& remoteSendOffset = (*allMeshAggregationSendRecvInfo_)[remoteRank].sendOffset;
            const std::vector<u64>& remoteSendLength = (*allMeshAggregationSendRecvInfo_)[remoteRank].sendLength;
            intraScratchOffsetMap_[i].push_back(
                startOffset + (remoteSendOffset[userRank_] - remoteSendOffset[meshRankStart_]));
            startOffset
                += (remoteSendOffset[meshRankEnd_] + remoteSendLength[meshRankEnd_] - remoteSendOffset[meshRankStart_]);
            intraScratchLengMap_[i].push_back(remoteSendLength[userRank_]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwiseCCLEnough::DeviceMemMapping()
{
    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        interTransportSend_ = cclIn_;
        interTransportRecv_ = cclOut_;
        intraTransportSend_ = cclOut_;
    } else {
        interTransportSend_ = inputMem_;
        interTransportRecv_ = scratchMem_;
        intraTransportSend_ = scratchMem_;
    }
    for (u32 intraRank = 0; intraRank < intraRankSize_; intraRank++) {
        if (intraRank == intraRankId_) {
            continue;
        }
        LINK& intraNeighboorTransport = intraLinks_[intraRank];
        void* remDMAMemPtr = nullptr;
        CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::INPUT_MEM, &remDMAMemPtr));
        DeviceMem remoteAlltoallScratch = DeviceMem::create(
            static_cast<u8*>(remDMAMemPtr),
            workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ? cclIn_.size() : scratchMem_.size());
        intraNeighBoorMemory_[intraRank] = {remoteAlltoallScratch};
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwiseCCLEnough::PrepareInterData(u32 step)
{
    // 准备 mesh 间发送信息
    nextInterSendData_.clear();
    u32 interSendRankStart = ((interRankId_ + 1 + step) % interRankSize_) * intraRankSize_;
    u32 interSendRankEnd = interSendRankStart + intraRankSize_ - 1;
    u64 startMemOffset = localSendRecvInfo_.sendOffset[interSendRankStart];
    u64 meshSendLength = localSendRecvInfo_.sendOffset[interSendRankEnd]
                         + localSendRecvInfo_.sendLength[interSendRankEnd] - startMemOffset;
    u64 sendDestOffset = 0;
    for (u32 relatedRank = intraRankId_; relatedRank < userRank_; relatedRank += intraRankSize_) {
        const SendRecvInfo& info = (*allMeshAggregationSendRecvInfo_)[relatedRank];
        sendDestOffset
            += (info.sendOffset[interSendRankEnd] + info.sendLength[interSendRankEnd]
                - info.sendOffset[interSendRankStart]);
    }
    DeviceMem srcMem = inputMem_.range(startMemOffset, meshSendLength);
    DeviceMem dstMem = interTransportSend_.range(startMemOffset, meshSendLength);
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwiseCCLEnough][PrepareInterSendData] userRank %u, interRank %u, "
        "intraRank %u move from userInput offset %llu length %llu to interTransportSend, send to remote offset "
        "%llu",
        userRank_, interRankId_, intraRankId_, startMemOffset, meshSendLength, sendDestOffset);

    HCCL_DEBUG("user size %u, inter size %u, intra size %u", groupRankSize_, interRankSize_, intraRankSize_);
    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStream_));
    }
    nextInterSendData_.emplace_back(
        TxMemoryInfo{UserMemType::OUTPUT_MEM, sendDestOffset, dstMem.ptr(), meshSendLength});

    // 准备 mesh 间接收信息
    nextInterRecvData_.clear();
    u32 recvBlockStart = (((interRankId_ + interRankSize_ - 1u - step) % interRankSize_) * intraRankSize_);
    u64 recvLocalOffset = 0;
    for (u32 relatedRank = intraRankId_; relatedRank < recvBlockStart; relatedRank += intraRankSize_) {
        const SendRecvInfo& info = (*allMeshAggregationSendRecvInfo_)[relatedRank];
        recvLocalOffset
            += (info.sendOffset[meshRankEnd_] + info.sendLength[meshRankEnd_] - info.sendOffset[meshRankStart_]);
    }
    const SendRecvInfo& recvRankInfo = (*allMeshAggregationSendRecvInfo_)[recvBlockStart + intraRankId_];
    u64 recvRemoteOffset = recvRankInfo.sendOffset[meshRankStart_];
    u64 recvLength
        = (recvRankInfo.sendOffset[meshRankEnd_] + recvRankInfo.sendLength[meshRankEnd_]
           - recvRankInfo.sendOffset[meshRankStart_]);
    nextInterRecvData_.emplace_back(RxMemoryInfo{
        UserMemType::INPUT_MEM, recvRemoteOffset, interTransportRecv_.range(recvLocalOffset, recvLength).ptr(),
        recvLength});
    return HCCL_SUCCESS;
}

// 将原先在userInput，且需要发到本mesh内其他卡的数据搬到CCL
HcclResult AlltoallPipelineMeshPairwiseCCLEnough::PrepareIntraData()
{
    u64 startMemOffset = localSendRecvInfo_.sendOffset[meshRankStart_];
    u64 meshSendLength
        = localSendRecvInfo_.sendOffset[meshRankEnd_] + localSendRecvInfo_.sendLength[meshRankEnd_] - startMemOffset;
    u64 intraSendOffset = 0;
    for (u32 relatedRank = intraRankId_; relatedRank < meshRankStart_; relatedRank += intraRankSize_) {
        const SendRecvInfo& info = (*allMeshAggregationSendRecvInfo_)[relatedRank];
        intraSendOffset
            += (info.sendOffset[meshRankEnd_] + info.sendLength[meshRankEnd_] - info.sendOffset[meshRankStart_]);
    }
    DeviceMem srcMem = inputMem_.range(startMemOffset, meshSendLength);
    DeviceMem dstMem = intraTransportSend_.range(intraSendOffset, meshSendLength);
    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStream_));
    }
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwiseCCLEnough][PrepareIntraData] userRank %u, interRank %u, intraRank %u "
        "copy from userInput offset %llu length %llu to intraTransportSend offset %llu",
        userRank_, interRankId_, intraRankId_, startMemOffset, meshSendLength, intraSendOffset);
    return HCCL_SUCCESS;
}

void AlltoallPipelineMeshPairwiseCCLEnough::UpdateIntraStreamInfo(u32 step)
{
    intraStreamInfo_.clear();
    u32 localMeshIndex = (interRankId_ + interRankSize_ - step) % interRankSize_;
    u32 firstDataBlockIndex = (meshRankStart_ + groupRankSize_ - step * intraRankSize_) % groupRankSize_;
    const std::vector<u64>& sendLengths
        = (*allMeshAggregationSendRecvInfo_)[firstDataBlockIndex + intraRankId_].sendLength;
    const std::vector<u64>& recvLengths = localSendRecvInfo_.recvLength;
    const std::vector<u64>& recvOffsets = localSendRecvInfo_.recvOffset;
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwiseCCLEnough][UpdateIntraStreamInfo] userRank %u, "
        "interRank %u, intraRank %u, step %u",
        userRank_, interRankId_, intraRankId_, step);
    for (u32 intraRank = 0; intraRank < intraRankSize_; intraRank++) {
        u64 sendLen = sendLengths[meshRankStart_ + intraRank];
        u64 recvLen = recvLengths[firstDataBlockIndex + intraRank];
        u64 recvRemoteOffset = intraScratchOffsetMap_[intraRank][localMeshIndex];
        u64 recvLocalOffset = recvOffsets[firstDataBlockIndex + intraRank];
        if (intraRank != intraRankId_) {
            intraStreamInfo_[intraRank] = {sendLen, recvLen, recvRemoteOffset, recvLocalOffset};
            HCCL_DEBUG(
                "[AlltoallPipelineMeshPairwiseCCLEnough][UpdateIntraStreamInfo] userRank %u, interRank %u, "
                "intraRank %u, sdma stream %u need send %llu and read length %llu from remote offset %llu "
                "to local offset %llu",
                userRank_, interRankId_, intraRankId_, intraRank, sendLen, recvLen, recvRemoteOffset, recvLocalOffset);
        }
    }
}

HcclResult AlltoallPipelineMeshPairwiseCCLEnough::SendRecvDataIntraMesh()
{
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwiseCCLEnough][SendRecvDataIntraMesh] userRank %u, "
        "interRank %u, intraRank %u, sdma stream %s wait main stream",
        userRank_, interRankId_, intraRankId_, GetStreamIndexString().c_str());
    for (auto& intraInfo : intraStreamInfo_) {
        u32 streamIndex = intraInfo.first;
        u64 recvLen = intraInfo.second[INTRA_STREAM_INFO_RECVLEN_INDEX];
        Stream& currStream = subStream_[streamIndex];
        LINK& readTransport = intraLinks_[streamIndex];
        CHK_RET(readTransport->TxAck(currStream));
        CHK_RET(readTransport->RxAck(currStream));
        u64 recvRemoteOffset = intraInfo.second[INTRA_STREAM_INFO_RECV_REMOTE_OFFSET_INDEX];
        u64 recvLocalOffset = intraInfo.second[INTRA_STREAM_INFO_RECV_LOCAL_OFFSET_INDEX];
        DeviceMem src = intraNeighBoorMemory_[streamIndex][0].range(recvRemoteOffset, recvLen);
        DeviceMem dst = outputMem_.range(recvLocalOffset, recvLen);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, currStream, readTransport->GetRemoteRank(), readTransport->GetLinkType()));
        CHK_RET(readTransport->TxDataSignal(currStream));
        HCCL_DEBUG(
            "[AlltoallPipelineMeshPairwiseCCLEnough][SendRecvDataIntraMesh] userRank %u, interRank %u, "
            "intraRank %u, sdma stream %llu read data from remote offset %llu len %llu to local %llu",
            userRank_, interRankId_, intraRankId_, streamIndex, recvRemoteOffset, recvLen, recvLocalOffset);
        CHK_RET(readTransport->RxDataSignal(currStream));
    }
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwiseCCLEnough][SendRecvDataIntraMesh] userRank %u, interRank %u, "
        "intraRank %u, sdma stream %s notify main stream",
        userRank_, interRankId_, intraRankId_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwiseCCLEnough::SendRecvDataInterMesh(u32 step)
{
    Stream& interStream = subStream_[intraRankId_];
    LINK& interRecvTransport = interLinks_[(interRankId_ + interRankSize_ - 1 - step) % interRankSize_];
    LINK& interSendTransport = interLinks_[(interRankId_ + 1 + step) % interRankSize_];
    CHK_RET(interRecvTransport->TxAck(interStream));
    CHK_RET(interSendTransport->RxAck(interStream));
    CHK_RET(interSendTransport->TxAsync(nextInterSendData_, interStream));
    CHK_RET(interRecvTransport->RxAsync(nextInterRecvData_, interStream));
    CHK_RET(interRecvTransport->PostFinAck(interStream));
    CHK_RET(interSendTransport->WaitFinAck(interStream));
    CHK_RET(ExecuteBarrier(interRecvTransport, interSendTransport, interStream));
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwiseCCLEnough::LocalCopyDataRecvFromInter(u32 interRankDistance)
{
    u64 scratchOffset
        = intraScratchOffsetMap_[intraRankId_][(interRankId_ + interRankSize_ - interRankDistance) % interRankSize_];
    u64 recvLen = localSendRecvInfo_
                      .recvLength[(userRank_ + groupRankSize_ - interRankDistance * intraRankSize_) % groupRankSize_];
    u64 userOutOffset
        = localSendRecvInfo_
              .recvOffset[(userRank_ + groupRankSize_ - interRankDistance * intraRankSize_) % groupRankSize_];
    DeviceMem src = interTransportRecv_.range(scratchOffset, recvLen);
    DeviceMem dst = outputMem_.range(userOutOffset, recvLen);
    HCCL_DEBUG(
        "[AlltoallPipelineMeshPairwiseCCLEnough][LocalCopyDataRecvFromInter]local move from "
        "interTransportRecv_ offset %llu length %llu to outputMem_ %llu",
        scratchOffset, recvLen, userOutOffset);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwiseCCLEnough::PreProcess()
{
    // server 间收发时间较长，先搬 server 间收发所需数据然后马上让server间开始收发
    CHK_RET(PrepareInterData(0));
    CHK_RET(NotifyInterStreamStart());
    // 之后将 server 内所需数据准备好之后唤醒server内从流收发
    UpdateIntraStreamInfo(0);
    CHK_RET(PrepareIntraData());
    CHK_RET(NotifyIntraStreamStart());
    CHK_RET(SendRecvDataIntraMesh());
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwiseCCLEnough::PipelineSend(u32 step, bool isLastStep)
{
    CHK_RET(SendRecvDataInterMesh(step));
    CHK_RET(PrepareInterData(step + 1u));
    CHK_RET(WaitInterStreamFinish());
    CHK_RET(WaitIntraStreamFinish());
    CHK_RET(ExecEmptyTask(inputMem_, outputMem_, mainStream_, dispatcher_));
    if (!isLastStep) {
        CHK_RET(NotifyInterStreamStart());
    }
    UpdateIntraStreamInfo(step + 1u);
    CHK_RET(NotifyIntraStreamStart());
    CHK_RET(SendRecvDataIntraMesh());
    CHK_RET(LocalCopyDataRecvFromInter(step + 1u));
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineMeshPairwiseCCLEnough::PostProcess()
{
    // 最后的收尾工作
    CHK_RET(LocalCopyDataRecvFromInter(0));
    CHK_RET(WaitIntraStreamFinish());
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(
    TemplateType::TEMPLATE_ALL_2_ALL_PIPELINE_MESH_PAIRWISE_CCL_ENOUGH, AlltoallPipelineMeshPairwiseCCLEnough);
} // namespace hccl
