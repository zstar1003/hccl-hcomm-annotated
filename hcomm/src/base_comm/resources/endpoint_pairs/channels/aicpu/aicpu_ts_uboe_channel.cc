/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "aicpu_ts_uboe_channel.h"
#include "orion_adpt_utils.h"
#include "hcomm_res_mgr.h"
#include "env_config/env_config_v2.h"
#include "makebufs_helper.h"
#include "endpoint.h"

// Orion
#include "adapter_rts_common.h"
#include "topo_common_types.h"
#include "rdma_handle_manager.h"

namespace hcomm {

AicpuTsUboeChannel::AicpuTsUboeChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : AicpuTsUboeUbRtpChannelHelper(endpointHandle, channelDesc)
{
    type_ = static_cast<u32>(Hccl::TransportType::UBoE);
}

AicpuTsUboeChannel::~AicpuTsUboeChannel() = default;

HcclResult AicpuTsUboeChannel::Init()
{
    s32 devLogicId;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));
    CHK_RET(ParseInputParam());
    CHK_RET(BuildSocket());
    CHK_RET(BuildNotify());
    CHK_RET(BuildDrainResource());
    /*
        HccpRaGetDevBaseAttr
        获取urma read/write 单个wr的最大传输数据大小
        调用前,rdmaHandle_要在ParseInputParam中被赋值好,之后BuildConnection会使用获取的属性
        uboe的BuildConnection不再Init里面执行，Init之后会有单独流程建链
    */
    CHK_RET(HccpRaGetDevBaseAttr(rdmaHandle_, &devBaseAttr_));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::BuildConnection()
{
    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(localEp_, remoteEp_, channelDesc_, ctx));

    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true; // aicpu 为 true
    HCCL_INFO(
        "[AicpuTsUboeChannel::%s] LinkProtocol[%s], locIpv4Addr[%s], rmtIpv4Addr[%s]", __func__,
        ctx.protocol.Describe().c_str(), ctx.locAddr.Describe().c_str(), ctx.rmtAddr.Describe().c_str());
    HCCL_INFO(
        "[AicpuTsUboeChannel::%s] locAddr_[%s], rmtAddr_[%s]", __func__, locAddr_.Describe().c_str(),
        rmtAddr_.Describe().c_str());

    // UBOE 协议对应 HCOMM_TA_RTP_UBOE_TIMEOUT，由 base_comm 从环境变量获取后传入
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(taTimeOutValue));
    taTimeOut = static_cast<u8>(taTimeOutValue);
    std::unique_ptr<Hccl::DevUbConnection> ubConn = std::make_unique<Hccl::DevUbUboeConnection>(
        rdmaHandle_, locAddr_, rmtAddr_, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL, ctx.locAddr, ctx.rmtAddr,
        ctx.qosPre, taTimeOut);
    CHK_SMART_PTR_NULL(ubConn);

    if (devBaseAttr_.maxReadSize == 0 || devBaseAttr_.maxWriteSize == 0) {
        HCCL_ERROR(
            "[AicpuTsUboeChannel][%s] maxReadSize[%u] or maxWriteSize[%u] must not be zero", __func__,
            devBaseAttr_.maxReadSize, devBaseAttr_.maxWriteSize);
        return HCCL_E_PARA;
    }
    ubConn->SetMaxReadSize(devBaseAttr_.maxReadSize);
    ubConn->SetMaxWriteSize(devBaseAttr_.maxWriteSize);
    HCCL_INFO(
        "[AicpuTsUboeChannel][%s] maxReadSize[%u], maxWriteSize[%u]", __func__, devBaseAttr_.maxReadSize,
        devBaseAttr_.maxWriteSize);

    commonRes_.connVec.clear();
    commonRes_.connVec.emplace_back(ubConn.get());
    connections_.clear();
    connections_.push_back(std::move(ubConn));
    return HCCL_SUCCESS;
}

void AicpuTsUboeChannel::EidPack()
{
    Hccl::IpAddress locIpv4Addr;
    CommAddrToIpAddress(localEp_.commAddr, locIpv4Addr);
    Hccl::RdmaHandleManager::GetInstance().GetEidByIpv4Addr(locIpv4Addr, locAddr_);
    sendEidData_ = locAddr_.GetUniqueId();
    HCCL_INFO(
        "[AicpuTsUboeChannel::%s] locIpv4Addr[%s], locAddr_[%s], sendEidData_ size[%u]", __func__,
        locIpv4Addr.Describe().c_str(), locAddr_.Describe().c_str(), sendEidData_.size());
}

void AicpuTsUboeChannel::SendEidData()
{
    EidPack();
    socket_->SendAsync(sendEidData_.data(), sendEidData_.size());
    HCCL_INFO("[AicpuTsUboeChannel::%s] send eid data, size=%llu", __func__, sendEidData_.size());
}

void AicpuTsUboeChannel::RecvEidData()
{
    recvEidData_.resize(sendEidData_.size());
    socket_->RecvAsync(ReinterpretAs<u8*>(recvEidData_.data()), recvEidData_.size());
    HCCL_INFO("[AicpuTsUboeChannel::%s] recv eid data, size=%llu", __func__, recvEidData_.size());
}

void AicpuTsUboeChannel::RecvEidDataProcess() { RmtEidUnpackProc(rmtAddr_); }

void AicpuTsUboeChannel::RmtEidUnpackProc(Hccl::IpAddress& rmtAddr)
{
    Hccl::IpAddress rmtEidAddr(recvEidData_);
    rmtAddr = rmtEidAddr;
    HCCL_INFO("[AicpuTsUboeChannel::%s] rmtAddr[%s]", __func__, rmtAddr.Describe().c_str());
}

void AicpuTsUboeChannel::SendFinish()
{
    HCCL_INFO("start send Finish Msg [%s]", FINISH_MSG);
    sendFinishMsg_ = std::vector<char>(FINISH_MSG, FINISH_MSG + FINISH_MSG_SIZE);
    socket_->SendAsync(sendFinishMsg_.data(), FINISH_MSG_SIZE);
    HCCL_INFO("end send Finish Msg [%s]", FINISH_MSG);
}

void AicpuTsUboeChannel::RecvFinish()
{
    recvFinishMsg_.resize(FINISH_MSG_SIZE);
    HCCL_INFO("start recv Finish Msg [%s]", FINISH_MSG);
    socket_->RecvAsync(ReinterpretAs<u8*>(recvFinishMsg_.data()), FINISH_MSG_SIZE);
    HCCL_INFO("end recv Finish Msg [%s]", FINISH_MSG);
}

void AicpuTsUboeChannel::HandleProcessData()
{
    if (RecvDataProcess()) {
        uboeStatus = UboeStatus::SEND_FIN;
    } else {
        channelStatus = ChannelStatus::READY;
        uboeStatus = UboeStatus::READY;
    }
}

void AicpuTsUboeChannel::ProcessUboeState()
{
    auto SetState = [this](UboeStatus next, ChannelStatus ch) {
        this->uboeStatus = next;
        this->channelStatus = ch;
    };

    switch (uboeStatus) {
        case UboeStatus::INIT:
            SetState(UboeStatus::SEND_EID, ChannelStatus::SOCKET_OK);
            break;
        case UboeStatus::SEND_EID:
            SendEidData();
            SetState(UboeStatus::RECV_EID, channelStatus);
            break;
        case UboeStatus::RECV_EID:
            RecvEidData();
            SetState(UboeStatus::PROCESS_EID_DATA, channelStatus);
            break;
        case UboeStatus::PROCESS_EID_DATA:
            RecvEidDataProcess();
            SetState(UboeStatus::BUILD_CONN, channelStatus);
            break;
        case UboeStatus::BUILD_CONN:
            BuildConn();
            SetState(UboeStatus::SEND_SIZE, channelStatus);
            break;
        case UboeStatus::SEND_SIZE:
            if (IsResReady()) {
                SendDataSize();
                SetState(UboeStatus::RECV_SIZE, channelStatus);
            }
            break;
        case UboeStatus::RECV_SIZE:
            RecvDataSize();
            SetState(isRecvFirst_ ? UboeStatus::RECV_DATA : UboeStatus::SEND_DATA, channelStatus);
            break;
        case UboeStatus::SEND_DATA:
            SendExchangeData();
            SetState(isRecvFirst_ ? UboeStatus::PROCESS_DATA : UboeStatus::RECV_DATA, channelStatus);
            break;
        case UboeStatus::RECV_DATA:
            RecvExchangeData();
            SetState(isRecvFirst_ ? UboeStatus::SEND_DATA : UboeStatus::PROCESS_DATA, channelStatus);
            break;
        case UboeStatus::PROCESS_DATA:
            HandleProcessData();
            break;
        case UboeStatus::SEND_FIN:
            if (IsConnsReady()) {
                SendFinish();
                SetState(UboeStatus::RECV_FIN, channelStatus);
            }
            break;
        case UboeStatus::RECV_FIN:
            RecvFinish();
            SetState(UboeStatus::SET_READY, channelStatus);
            break;
        case UboeStatus::SET_READY:
            channelStatus = ChannelStatus::READY;
            SetState(UboeStatus::READY, ChannelStatus::READY);
            break;
        default:
            break;
    }
}

HcclResult AicpuTsUboeChannel::CheckSocketStatus(const std::string& socketOperator)
{
    CHK_PTR_NULL(socket_);
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    uint32_t retryCount = 0;
    while (true) {
        Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
        if (socketStatus == Hccl::SocketStatus::OK) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_INFO(
                "[AicpuTsUboeChannel][%s] socket operation[%s] success, elapsed[%lld]ms, retryCount[%u]", __func__,
                socketOperator.c_str(), elapsed, retryCount);
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout || socketStatus == Hccl::SocketStatus::TIMEOUT) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[AicpuTsUboeChannel][%s] socket operation[%s] timeout, socketStatus[%u], elapsed[%lld]ms, "
                "retryCount[%u]",
                __func__, socketOperator.c_str(), static_cast<uint32_t>(socketStatus), elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }
        retryCount++;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    std::vector<Hccl::LocalRmaBuffer*> bufferVecTemp;
    CHK_RET(MakeRmaBufferVecFromMemHandles(memHandles, memHandleNum, bufferVecTemp, "AicpuTsUboeChannel"));

    if (bufferVecTemp.size() == 0) {
        HCCL_WARNING("[AicpuTsUboeChannel][%s] bufferNum is 0.", __func__);
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(socket_);
    HCCL_INFO("[AicpuTsUboeChannel][%s] bufferNum[%zu]", __func__, bufferVecTemp.size());

    std::vector<char> localSendData;
    Hccl::BinaryStream sendStream;
    BufferVecPack(sendStream, bufferVecTemp);
    sendStream.Dump(localSendData);

    u32 sendSize = localSendData.size();
    socket_->SendAsync(&sendSize, sizeof(sendSize));
    HCCL_INFO("[AicpuTsUboeChannel][%s] Send size[%u] of data.", __func__, sendSize);
    CHK_RET(CheckSocketStatus("SendDataSize"));

    u32 recvSize = 0;
    socket_->RecvAsync(ReinterpretAs<u8*>(&recvSize), sizeof(recvSize));
    CHK_RET(CheckSocketStatus("RecvDataSize"));
    HCCL_INFO("[AicpuTsUboeChannel][%s] Recv size[%u] of data.", __func__, recvSize);

    socket_->SendAsync(localSendData.data(), localSendData.size());
    HCCL_INFO("[AicpuTsUboeChannel][%s] Send data, size[%zu].", __func__, localSendData.size());
    CHK_RET(CheckSocketStatus("SendExchangeData"));

    std::vector<char> localRecvData(recvSize);
    socket_->RecvAsync(ReinterpretAs<u8*>(localRecvData.data()), localRecvData.size());
    CHK_RET(CheckSocketStatus("RecvExchangeData"));
    HCCL_INFO("[AicpuTsUboeChannel][%s] Recv data success.", __func__);

    std::vector<std::unique_ptr<Hccl::RemoteUbRmaBuffer>> rmtBufferTemp{};
    Hccl::BinaryStream recvStream(localRecvData);
    RmtBufferVecUnpackProc(static_cast<u32>(bufferVecTemp.size()), recvStream, rmtBufferTemp, UboeRmtBufType::BUFFER);

    rmtBufferVec_.insert(
        rmtBufferVec_.end(), std::make_move_iterator(rmtBufferTemp.begin()),
        std::make_move_iterator(rmtBufferTemp.end()));
    commonRes_.bufferVec.insert(commonRes_.bufferVec.end(), bufferVecTemp.begin(), bufferVecTemp.end());
    cacheValid_ = false;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::Clean()
{
    memTransport_.reset();

    commonRes_.connVec.clear();
    connections_.clear();

    rmtNotifyVec_.clear();
    locBufferVec_.clear();

    recvData_.clear();
    recvFinishMsg_.clear();
    recvEidData_.clear();
    sendData_.clear();
    sendFinishMsg_.clear();
    sendEidData_.clear();

    bufferNum_ = 0;
    connNum_ = 0;
    recvDataSize_ = 0;
    locAddr_ = Hccl::IpAddress();
    rmtAddr_ = Hccl::IpAddress();

    {
        std::lock_guard<std::mutex> lock(remoteMemsMutex_);
        rmtBufferVec_.clear();
        cacheValid_ = false;
        remoteUserMems_.clear();
        memInfoCopies_.clear();
        memInfoPointers_.clear();
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeChannel::Resume()
{
    channelStatus = ChannelStatus::INIT;
    uboeStatus = UboeStatus::INIT;
    HCCL_INFO("[AicpuTsUboeChannel][%s] reset uboe state machine to rebuild connection.", __func__);
    return HCCL_SUCCESS;
}

ChannelStatus AicpuTsUboeChannel::GetStatus()
{
    if (channelStatus == ChannelStatus::READY) {
        return channelStatus;
    }
    if (channelStatus == ChannelStatus::INIT)
        uboeStatus = UboeStatus::INIT;

    if (!IsSocketReady())
        return channelStatus;

    ProcessUboeState();

    return channelStatus;
}

} // namespace hcomm
