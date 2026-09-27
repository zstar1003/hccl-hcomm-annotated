/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_cpu_roce_channel.h"
#include "endpoint.h"
#include "dpu_notify/dpu_notify_manager.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "exception_handler.h"
#include "cpu_roce_endpoint.h"
#include "adapter_error_manager_pub.h"

// Orion
#include "orion_adapter_hccp.h"
#include "orion_adpt_utils.h"
#include "exchange_rdma_buffer_dto.h"
#include "rdma_handle_manager.h"
#include "exchange_rdma_conn_dto.h"
#include "sal.h"
#include "adapter_hccp.h"
#include "binary_stream.h"
#include "notify_pool_impl.h"
#include "hccp_common.h"
#include "dlprof_function.h"

namespace hcomm_experimental {
constexpr u32 FENCE_TIMEOUT_MS = 30 * 1000; // 定义最大等待30秒
constexpr u32 MEM_BLOCK_SIZE = 128;
constexpr uint16_t DEFAULT_LISTENING_PORT = 60001;
constexpr u32 SEND_RQE_COUNT = 16;
constexpr u32 HYBRID_QP_STATUS_POLL_SLEEP_US = 1000;
constexpr u32 DEFAULT_NOTIFY_WAIT_TIMEOUT_S = 30;
constexpr uint32_t kHostResourceId = 0U;

static inline CommMemType HcclMemTypeToCommMemType(HcclMemType type)
{
    switch (type) {
        case HCCL_MEM_TYPE_DEVICE:
            return COMM_MEM_TYPE_DEVICE;
        case HCCL_MEM_TYPE_HOST:
            return COMM_MEM_TYPE_HOST;
        default:
            return COMM_MEM_TYPE_INVALID;
    }
}

HostCpuRoceChannel::HostCpuRoceChannel(EndpointHandle endpointHandle, HcommChannelDesc channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

HostCpuRoceChannel::~HostCpuRoceChannel()
{
    HcclResult ret;

    if (isHybridMode_ && connections_.size() != 0) {
        auto qpInfo = connections_[0]->GetQpInfo();
        struct MrInfoT mrInfo {};

        for (uint32_t i = 0; i < hccl::MEM_TYPE_RESERVED; i++) {
            if (localMemMsg_[i].addr == nullptr) {
                continue;
            }

            mrInfo.addr = localMemMsg_[i].addr;
            ret = HrtRaMrDereg(qpInfo.qpHandle, &mrInfo);
            if (ret != HCCL_SUCCESS) {
                HCCL_INFO(
                    "[~HostCpuRoceChannel] Dereg mem, ret=%d, type=%d, addr:%p, lkey:%d, size:%llu, access:%d", ret, i,
                    mrInfo.addr, mrInfo.lkey, mrInfo.size, mrInfo.access);
            }

            if (localMemMsg_[i].notifyId != INVALID_DPU_NOTIFY_ID) {
                delete[] (int8_t*)localMemMsg_[i].addr;
            }
            localMemMsg_[i].addr = nullptr;

            HCCL_INFO(
                "[~HostCpuRoceChannel] Dereg mem, type=%d, addr:%p, lkey:%d, size:%llu, access:%d", i, mrInfo.addr,
                mrInfo.lkey, mrInfo.size, mrInfo.access);
        }
    }

    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNum_, localDpuNotifyIds_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HostCpuRoceChannel::~HostCpuRoceChannel] exception occurred, HcclResult=[%d]", ret);
    }

    if (socket_ != nullptr) {
        hcomm::SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
        socket_ = nullptr;
    }
}

HcclResult HostCpuRoceChannel::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    CHK_PTR_NULL(endpointHandle_);
    HCCL_INFO(
        "[HostCpuRoceChannel][%s] Start. endpointHandle[0x%llx]", __func__,
        reinterpret_cast<uint64_t>(endpointHandle_));
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle_);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();
    CHK_PTR_NULL(rdmaHandle_);

    // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = reinterpret_cast<Hccl::Socket*>(channelDesc_.socket);
    // If HIXL, socket is nullptr for now, will be built later.
    notifyNum_ = channelDesc_.notifyNum;

    if (channelDesc_.exchangeAllMems) { // true for HIXL, false for HCCL
        // 3. Get memHandles from endpoint
        HCCL_INFO("[HostCpuRoceChannel][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalRdmaRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(localEpPtr->GetAllMemHandles(reinterpret_cast<void**>(&memHandles), &memHandleNum));
        HCCL_INFO("[HostCpuRoceChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalRdmaRmaBuffer>& localRdmaBuffer = memHandles[i];
            auto buf = localRdmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[HostCpuRoceChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memInfo[%s].", __func__, i,
                localRdmaBuffer->GetAddr(), localRdmaBuffer->GetSize(), buf->GetMemInfo().c_str());
            localRmaBuffers_.emplace_back(localRdmaBuffer.get());
        }
    } else {
        // 3. 从 channelDesc 的 memHandle，获得 bufs_
        HCCL_INFO("[HostCpuRoceChannel][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_PTR_NULL(channelDesc_.memHandles);
        for (uint32_t i = 0; i < channelDesc_.memHandleNum; ++i) {
            CHK_PTR_NULL(channelDesc_.memHandles[i]);
            auto* localRdmaBuffer = static_cast<Hccl::LocalRdmaRmaBuffer*>(channelDesc_.memHandles[i]);
            localRmaBuffers_.emplace_back(localRdmaBuffer);
        }
    }

    auto* localCpuRoceEpPtr = dynamic_cast<CpuRoceEndpoint*>(localEpPtr);
    if (localCpuRoceEpPtr == nullptr) {
        HCCL_ERROR("[HostCpuRoceChannel][%s] endpointHandle_ is not CpuRoceEndpoint.", __func__);
        return HCCL_E_INTERNAL;
    }
    CpuRoceEndpoint::Capabilities caps{};
    CHK_RET(localCpuRoceEpPtr->GetCapabilities(caps));
    maxMsgSize_ = caps.maxMsgSize;
    lbMax_ = caps.lbMax;
    constexpr uint64_t TWO_GB = 0x80000000ULL; // 2GB
    if (maxMsgSize_ > TWO_GB) {
        HCCL_RUN_WARNING(
            "[HostCpuRoceChannel][%s] maxMsgSize_[0x%llx] exceeds 2GB, value may be incorrect.", __func__, maxMsgSize_);
    }
    HCCL_INFO("[HostCpuRoceChannel][%s] maxMsgSize_[0x%llx].", __func__, maxMsgSize_);

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::StartListen()
{
    uint16_t port = channelDesc_.port;
    HCCL_INFO(
        "[HostCpuRoceChannel::%s] Start. EndpointHandle[0x%llx], port[%u]", __func__,
        reinterpret_cast<uint64_t>(endpointHandle_), port);
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[HostCpuRoceChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    CHK_RET(localEpPtr->ServerSocketListen(port));
    HCCL_INFO("[HostCpuRoceChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[HostCpuRoceChannel::%s] socket ptr is NULL, rebuild Socket", __func__);

    Hccl::LinkData linkData = hcomm::BuildDefaultLinkData();
    CHK_RET(hcomm::EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[HostCpuRoceChannel::%s] built linkData: %s", __func__, linkData.Describe().c_str());
    uint16_t port = channelDesc_.port;
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO(
            "[HostCpuRoceChannel::%s] channelDesc port is 0, use default port [%u] for BuildSocket", __func__, port);
    }
    std::string socketTag = "AUTOMATIC_SOCKET_TAG";
    Hccl::SocketConfig socketConfig
        = (channelDesc_.role != HCOMM_SOCKET_ROLE_RESERVED) ?
              Hccl::SocketConfig(linkData, port, socketTag, channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER) :
              Hccl::SocketConfig(linkData, port, socketTag);
    CHK_RET(hcomm::SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));
    HCCL_INFO("[HostCpuRoceChannel::%s] BuildSocket SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::BuildConnection()
{
    u32 loopTimes = 0;
    if (lbMax_ > 0) {
        if (channelDesc_.roceAttr.queueNum == 1) {
            loopTimes = lbMax_;
        } else {
            loopTimes = channelDesc_.roceAttr.queueNum;
        }
    } else {
        loopTimes = channelDesc_.roceAttr.queueNum;
    }

    for (u32 i = 0; i < loopTimes; i++) {
        std::unique_ptr<HostRdmaConnection> conn;
        EXCEPTION_CATCH(conn = std::make_unique<HostRdmaConnection>(socket_, rdmaHandle_), return HCCL_E_INTERNAL);
        CHK_PTR_NULL(conn);
        CHK_RET(conn->Init());
        Hccl::QpInfo& qpInfo = conn->GetQpInfo();
        if (lbMax_ > 0) {
            qpInfo.lbValue = i % lbMax_;
        }
        qpInfo.serviceLevel = channelDesc_.roceAttr.sl;
        qpInfo.trafficClass = channelDesc_.roceAttr.tc;
        qpInfo.retryCnt = channelDesc_.roceAttr.retryCnt;
        qpInfo.retryInterval = channelDesc_.roceAttr.retryInterval;
        HCCL_INFO(
            "[HostCpuRoceChannel::BuildConnection] QpInfo[%u]: lbValue[%u], serviceLevel[%u], trafficClass[%u], "
            "retryCnt[%u], retryInterval[%u].",
            i, qpInfo.lbValue, qpInfo.serviceLevel, qpInfo.trafficClass, qpInfo.retryCnt, qpInfo.retryInterval);
        connections_.emplace_back(std::move(conn));
    }
    connNum_ = connections_.size();
    wqeNums_.resize(connNum_, 0);
    HCCL_INFO("[HostCpuRoceChannel::BuildConnection] Success, Qp count = %u", connNum_);
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::BuildNotify()
{
    CHK_RET(DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum_, localDpuNotifyIds_));
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::BuildBuffer()
{
    bufferNum_ = localRmaBuffers_.size();
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::Init()
{
    devicePhyId_ = kHostResourceId;

    CHK_RET(ParseInputParam());
    if (channelDesc_.exchangeAllMems && channelDesc_.role != HCOMM_SOCKET_ROLE_CLIENT) {
        CHK_RET(StartListen());
    }
    CHK_RET(BuildSocket());
    CHK_RET(BuildConnection());
    CHK_RET(BuildNotify());
    CHK_RET(BuildBuffer());

    return HCCL_SUCCESS;
}

// 当前AICPU和框架没有改为返回错误码形式，所有暂时使用该方法转换
ChannelStatus HostCpuRoceChannel::GetStatus()
{
    ChannelStatus status;
    HcclResult ret = GetStatus(status);
    if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
        HCCL_ERROR("[HostCpuRoceChannel::GetStatus] get status exception occurred, HcclResult=[%d]", ret);
        return ChannelStatus::FAILED;
    }
    return status;
}

HcclResult HostCpuRoceChannel::ProcessStatus()
{
    switch (channelStatus_) {
        case ChannelStatus::READY:
            if (socket_ != nullptr) {
                hcomm::SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
            }
            return HCCL_SUCCESS;
        case ChannelStatus::SOCKET_TIMEOUT:
            HCCL_ERROR("[HostCpuRoceChannel::ProcessStatus] get socket timeout");
            return HCCL_E_ROCE_CONNECT;
        default:
            return HCCL_E_AGAIN;
    }
}

HcclResult HostCpuRoceChannel::GetStatus(ChannelStatus& status)
{
    switch (rdmaStatus_) {
        case RdmaStatus::INIT:
            // 检查socket状态
            CHK_RET(CheckSocketStatus());
            break;
        case RdmaStatus::SOCKET_OK:
            CHK_RET(ExchangeCapability());
            rdmaStatus_ = RdmaStatus::CAP_EXCHANGED;
            break;
        case RdmaStatus::CAP_EXCHANGED:
            // 准备资源
            CHK_RET(CreateQp());
            rdmaStatus_ = RdmaStatus::QP_CREATED;
            break;
        case RdmaStatus::QP_CREATED:
            // 发送交换数据
            if (isHybridMode_) {
                CHK_RET(ExchangeDataHybird());
            } else {
                CHK_RET(ExchangeData());
            }
            rdmaStatus_ = RdmaStatus::DATA_EXCHANGE;
            break;
        case RdmaStatus::DATA_EXCHANGE:
            if (isHybridMode_) {
                CHK_RET(ConnectSingleQpHybrid([]() -> bool {
                    return 0;
                }));
            } else {
                CHK_RET(ModifyQp());
            }
            rdmaStatus_ = RdmaStatus::QP_MODIFIED;
            // modify完就不需要再轮询状态了，直接向下走准备Rqe的流程。
            [[fallthrough]];
        case RdmaStatus::QP_MODIFIED:
            // Prepare Rqes
            if (!isHybridMode_) {
                for (uint32_t i = 0; i < SEND_RQE_COUNT; ++i) {
                    CHK_RET(IbvPostRecv());
                }
            }
            [[fallthrough]];
        default:
            rdmaStatus_ = RdmaStatus::CONN_OK;
            channelStatus_ = ChannelStatus::READY;
    }

    status = channelStatus_;
    return ProcessStatus();
}

HcclResult HostCpuRoceChannel::CheckSocketStatus()
{
    CHK_PTR_NULL(socket_);
    HCCL_DEBUG("[HostCpuRoceChannel::CheckSocketStatus] socket GetStatus start");
    Hccl::SocketStatus socketStatus = socket_->GetStatus(); // socket状态机
    HCCL_DEBUG("[HostCpuRoceChannel::CheckSocketStatus] socket status = %s", socketStatus.Describe().c_str());
    if (socketStatus == Hccl::SocketStatus::OK) {
        rdmaStatus_ = RdmaStatus::SOCKET_OK;
        channelStatus_ = ChannelStatus::SOCKET_OK;
    } else if (socketStatus == Hccl::SocketStatus::TIMEOUT) {
        channelStatus_ = ChannelStatus::SOCKET_TIMEOUT;
    }
    return HCCL_SUCCESS;
}

// 准备资源（创建QP）
HcclResult HostCpuRoceChannel::CreateQp()
{
    for (auto& conn : connections_) {
        Hccl::CHECK_NULLPTR(
            conn, Hccl::StringFormat("[HostCpuRoceChannel::%s] failed, connection pointer is nullptr", __func__));
        HcclResult ret = conn->CreateQp();
        if (ret == HCCL_E_AGAIN) {
            return HCCL_SUCCESS;
        }
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }
    HCCL_INFO("[HostCpuRoceChannel::IsResReady] all connections resources connected.");
    return HCCL_SUCCESS;
}

// 交换数据
HcclResult HostCpuRoceChannel::ExchangeData()
{
    HCCL_INFO(
        "[HostCpuRoceChannel::%s] Start to SendExchangeData, notifyNum=%u, bufferNum=%u, connNum=%u", __func__,
        notifyNum_, bufferNum_, connNum_);

    // 同步数据打包
    Hccl::BinaryStream binaryStream;
    NotifyVecPack(binaryStream);
    CHK_RET(BufferVecPack(binaryStream));
    CHK_RET(ConnVecPack(binaryStream));

    std::vector<char> sendData{};
    binaryStream.Dump(sendData);
    uint64_t sendSize = sendData.size();
    std::vector<char> recvData{};
    uint64_t recvSize = 0;

    EXCEPTION_HANDLE_BEGIN
    // 同步发送数据包尺寸
    CHK_PRT_RET(
        !socket_->Send(reinterpret_cast<void*>(&sendSize), sizeof(sendSize)),
        HCCL_ERROR("[HostCpuRoceChannel::%s] Send sendSize failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO(
        "[HostCpuRoceChannel::%s] Send size[%llu] of data success. [%llu] bytes sent.", __func__, sendSize,
        sizeof(sendSize));

    // 同步接收数据包尺寸
    CHK_PRT_RET(
        !socket_->Recv(reinterpret_cast<void*>(&recvSize), sizeof(recvSize)),
        HCCL_ERROR("[HostCpuRoceChannel::%s] Recv recvSize failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO(
        "[HostCpuRoceChannel::%s] Receive size[%llu] of data success. [%llu] bytes received.", __func__, recvSize,
        sizeof(recvSize));

    // 同步发送数据
    CHK_PRT_RET(
        !socket_->Send(reinterpret_cast<void*>(sendData.data()), sendSize),
        HCCL_ERROR("[HostCpuRoceChannel::%s] Send exchange data failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO("[HostCpuRoceChannel::%s] Send Exchange Data success. [%llu] bytes sent.", __func__, sendSize);

    // 同步接收数据
    HCCL_INFO("[HostCpuRoceChannel::%s] Start to Receive Exchange Data", __func__);
    recvData.resize(recvSize);
    CHK_PRT_RET(
        !socket_->Recv(reinterpret_cast<void*>(recvData.data()), recvSize),
        HCCL_ERROR("[HostCpuRoceChannel::%s] Recv exchange data failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO("[HostCpuRoceChannel::%s] Receive Exchange Data success. [%llu] bytes received.", __func__, recvSize);
    EXCEPTION_HANDLE_END

    // 同步数据解包
    Hccl::BinaryStream recvBinStream(recvData);
    // CHK_RET(HandshakeMsgUnpack(recvBinStream));
    CHK_RET(NotifyVecUnpack(recvBinStream));
    CHK_RET(RmtBufferVecUnpackProc(recvBinStream));
    CHK_RET(ConnVecUnpackProc(recvBinStream));

    HCCL_INFO("[HostCpuRoceChannel::%s] Unpack exchange Data success.", __func__);
    return HCCL_SUCCESS;
}

void HostCpuRoceChannel::NotifyVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << notifyNum_;
    HCCL_INFO("start pack DpuRoceChannel notifyVec");
    u32 pos = 0;
    for (auto& it : localDpuNotifyIds_) {
        binaryStream << it;
        HCCL_INFO("pack notify pos=%u, notifyId=%u", pos, it);
        pos++;
    }
}

HcclResult HostCpuRoceChannel::BufferVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << bufferNum_;
    HCCL_INFO("[HostCpuRoceChannel::%s] start to pack RmaBuffers", __func__);
    u32 pos = 0;
    for (auto& it : localRmaBuffers_) {
        binaryStream << pos;
        if (it != nullptr) { // 非空的buffer，从buffer中获取 dto
            std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
            if (dto == nullptr) {
                return HCCL_E_INTERNAL;
            }
            dto->Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u dto %s", pos, dto->Describe().c_str());
        } else { // 空的buffer，dto所有字段为0(size=0)
            Hccl::ExchangeRdmaBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u, dto is null %s", pos, exchangeDto.Describe().c_str());
        }
        pos++;
    }
    HCCL_INFO("[HostCpuRoceChannel::%s] pack RmaBuffers finish", __func__);
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::ConnVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << connNum_;
    HCCL_INFO("[HostCpuRoceChannel::%s] start to pack connections", __func__);
    u32 pos = 0;
    binaryStream << channelDesc_.roceAttr.queueNum;
    for (auto& it : connections_) {
        binaryStream << pos;

        binaryStream << channelDesc_.roceAttr.retryCnt;
        binaryStream << channelDesc_.roceAttr.retryInterval;
        binaryStream << channelDesc_.roceAttr.sl;
        binaryStream << channelDesc_.roceAttr.tc;

        std::unique_ptr<Hccl::Serializable> dto = nullptr;
        CHK_RET(it->GetExchangeDto(dto));
        dto->Serialize(binaryStream);
        HCCL_INFO("pack connection pos=%u, dto %s", pos, dto->Describe().c_str());
        pos++;
    }
    HCCL_INFO("[HostCpuRoceChannel::%s] pack connections finish", __func__);
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::RmtBufferVecUnpackProc(Hccl::BinaryStream& binaryStream)
{
    u32 rmtNum;
    binaryStream >> rmtNum;

    HCCL_INFO("[HostCpuRoceChannel::%s] bufferNum_=%u, rmtNum=%u", __func__, bufferNum_, rmtNum);

    rmtRmaBuffers_.resize(rmtNum);
    for (u32 i = 0; i < rmtNum; i++) {
        u32 pos;
        binaryStream >> pos;
        if (pos >= rmtNum) {
            HCCL_ERROR("[HostCpuRoceChannel::%s] pos=%u out of range (rmtNum=%u)", __func__, pos, rmtNum);
            return HCCL_E_INTERNAL;
        }
        Hccl::ExchangeRdmaBufferDto dto;
        dto.Deserialize(binaryStream);

        HCCL_INFO("[HostCpuRoceChannel::%s] pos=%u, dto %s", __func__, pos, dto.Describe().c_str());
        EXCEPTION_CATCH(
            rmtRmaBuffers_[pos] = std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle_, dto),
            HCCL_ERROR(
                "[HostCpuRoceChannel::%s] make_unique<Hccl::RemoteRdmaRmaBuffer> throws an exception!", __func__);
            return HCCL_E_INTERNAL);
        HCCL_INFO(
            "[HostCpuRoceChannel::%s] pos=%u, rmtRmaBuffer=%s", __func__, pos, rmtRmaBuffers_[pos]->Describe().c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::NotifyVecUnpack(Hccl::BinaryStream& binaryStream)
{
    uint32_t notifySize = 0;
    binaryStream >> notifySize;
    if (notifySize != notifyNum_) {
        HCCL_ERROR(
            "[HostCpuRoceChannel::NotifyVecUnpack] rmtNum=%u is not equal to localNum=%u", notifySize, notifyNum_);
        return HCCL_E_ROCE_CONNECT;
    }
    remoteDpuNotifyIds_.clear();
    u32 pos = 0;
    for (pos = 0; pos < notifySize; pos++) {
        uint32_t notifyId;
        binaryStream >> notifyId;
        remoteDpuNotifyIds_.push_back(notifyId);
    }
    HCCL_INFO("[HostCpuRoceChannel::NotifyVecUnpack] unpack dpuNotify");
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::ConnVecUnpackProc(Hccl::BinaryStream& binaryStream)
{
    u32 rmtConnNum;
    binaryStream >> rmtConnNum;
    HCCL_INFO("start unpack conn, connNum=%u, rmtConnNum=%u", connNum_, rmtConnNum);
    if (connNum_ != rmtConnNum) {
        HCCL_ERROR("connNum=%u is not equal to rmtConnNum=%u", connNum_, rmtConnNum);
        return HCCL_E_ROCE_CONNECT;
    }

    uint32_t localQpNum = channelDesc_.roceAttr.queueNum;
    binaryStream >> channelDesc_.roceAttr.queueNum;
    if (localQpNum != channelDesc_.roceAttr.queueNum) {
        HCCL_ERROR("localQpNum[%u] is not equal to remoteQpNum[%u]", localQpNum, channelDesc_.roceAttr.queueNum);
        return HCCL_E_ROCE_CONNECT;
    }

    rmtConnDtos_.resize(rmtConnNum);
    for (u32 i = 0; i < rmtConnNum; i++) {
        u32 pos;
        binaryStream >> pos;
        binaryStream >> channelDesc_.roceAttr.retryCnt;
        binaryStream >> channelDesc_.roceAttr.retryInterval;
        binaryStream >> channelDesc_.roceAttr.sl;
        binaryStream >> channelDesc_.roceAttr.tc;
        rmtConnDtos_[i].Deserialize(binaryStream);
    }

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::ModifyQp()
{
    for (uint32_t i = 0; i < connections_.size(); i++) {
        auto& conn = connections_[i];
        Hccl::CHECK_NULLPTR(
            conn, Hccl::StringFormat("[HostCpuRoceChannel::%s] failed, connection pointer is nullptr", __func__));
        CHK_RET(conn->ParseRmtExchangeDto(rmtConnDtos_[i]));
        Hccl::QpInfo& qpInfo = conn->GetQpInfo();
        qpInfo.serviceLevel = channelDesc_.roceAttr.sl;
        qpInfo.trafficClass = channelDesc_.roceAttr.tc;
        qpInfo.retryCnt = channelDesc_.roceAttr.retryCnt;
        qpInfo.retryInterval = channelDesc_.roceAttr.retryInterval;
        HCCL_INFO(
            "[HostCpuRoceChannel::ModifyQp] QpInfo: serviceLevel[%u], trafficClass[%u], retryCnt[%u], "
            "retryInterval[%u].",
            qpInfo.serviceLevel, qpInfo.trafficClass, qpInfo.retryCnt, qpInfo.retryInterval);
        HcclResult ret = conn->ModifyQp();
        if (ret == HCCL_E_AGAIN) {
            return HCCL_SUCCESS;
        }
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }
    HCCL_INFO("[HostCpuRoceChannel::IsResReady] all connections resources connected.");
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::GetRemoteMem(HcclMem** remoteMem, uint32_t* memNum, char** memTags)
{
    CHK_PRT_RET(remoteMem == nullptr, HCCL_ERROR("[GetRemoteMem] remoteMem is nullptr"), HCCL_E_PTR);
    CHK_PRT_RET(memNum == nullptr, HCCL_ERROR("[GetRemoteMem] memNum is nullptr"), HCCL_E_PTR);
    CHK_PRT_RET(memTags == nullptr, HCCL_ERROR("[GetRemoteMem] memTags is nullptr"), HCCL_E_PTR);

    *remoteMem = nullptr;
    *memNum = 0;

    std::lock_guard<std::mutex> lock(remoteMemsMutex_);

    uint32_t totalCount = rmtRmaBuffers_.size();
    if (totalCount == 0) {
        HCCL_INFO("[GetRemoteMem] No remote memory regions available");
        return HCCL_SUCCESS;
    }
    // 释放之前的内存
    remoteMemsPtr_.reset();
    remoteMemsPtr_ = std::make_unique<HcclMem[]>(totalCount);
    CHK_PTR_NULL(remoteMemsPtr_);

    for (uint32_t i = 0; i < totalCount; i++) {
        auto& rmtRmaBuffer = rmtRmaBuffers_[i];
        remoteMemsPtr_[i].type = rmtRmaBuffer->GetMemType();
        remoteMemsPtr_[i].addr = reinterpret_cast<void*>(rmtRmaBuffer->GetAddr());
        remoteMemsPtr_[i].size = rmtRmaBuffer->GetSize();
        memTags[i] = const_cast<char*>(rmtRmaBuffer->GetMemInfo().c_str());
        HCCL_INFO(
            "[%s] addr[%p] size[%zu] rmtRmaBuffer[%p]", __func__, reinterpret_cast<void*>(rmtRmaBuffer->GetAddr()),
            rmtRmaBuffer->GetSize(), rmtRmaBuffer.get());
    }

    *memNum = totalCount;
    *remoteMem = remoteMemsPtr_.get();
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::GetUserRemoteMem(CommMem** remoteMem, char*** memTag, uint32_t* memNum)
{
    CHK_PRT_RET(remoteMem == nullptr, HCCL_ERROR("[GetUserRemoteMem] remoteMem is nullptr"), HCCL_E_PTR);
    CHK_PRT_RET(memTag == nullptr, HCCL_ERROR("[GetUserRemoteMem] memTag is nullptr"), HCCL_E_PTR);
    CHK_PRT_RET(memNum == nullptr, HCCL_ERROR("[GetUserRemoteMem] memNum is nullptr"), HCCL_E_PTR);

    *remoteMem = nullptr;
    *memTag = nullptr;
    *memNum = 0;

    std::lock_guard<std::mutex> lock(remoteMemsMutex_);

    if (rmtRmaBuffers_.size() <= 1) {
        HCCL_INFO("[GetUserRemoteMem] No user remote memory found, rmtRmaBuffers_ size=%zu", rmtRmaBuffers_.size());
        return HCCL_SUCCESS;
    }

    uint32_t userMemCount = static_cast<uint32_t>(rmtRmaBuffers_.size() - 1);

    if (!userRemoteMemCacheValid_) {
        userRemoteMems_.resize(userMemCount);
        tagCopies_.clear();
        tagCopies_.reserve(userMemCount);
        tagPointers_.clear();
        tagPointers_.reserve(userMemCount);

        for (uint32_t i = 0; i < userMemCount; ++i) {
            auto& rmtBuffer = rmtRmaBuffers_[i + 1]; // skip cclBuffer at index 0
            if (rmtBuffer == nullptr) {
                userRemoteMems_[i].type = COMM_MEM_TYPE_INVALID;
                userRemoteMems_[i].addr = nullptr;
                userRemoteMems_[i].size = 0;
                tagCopies_.emplace_back();
                tagPointers_.push_back(const_cast<char*>(tagCopies_.back().c_str()));
                continue;
            }

            userRemoteMems_[i].type = HcclMemTypeToCommMemType(rmtBuffer->GetMemType());
            userRemoteMems_[i].addr = reinterpret_cast<void*>(rmtBuffer->GetAddr());
            userRemoteMems_[i].size = rmtBuffer->GetSize();

            tagCopies_.emplace_back(rmtBuffer->GetMemInfo());
            tagPointers_.push_back(const_cast<char*>(tagCopies_.back().c_str()));
        }

        userRemoteMemCacheValid_ = true;
    }

    *remoteMem = userRemoteMems_.data();
    *memTag = tagPointers_.data();
    *memNum = userMemCount;

    return HCCL_SUCCESS;
}

std::vector<Hccl::QpInfo> HostCpuRoceChannel::GetQpInfos() const
{
    std::vector<Hccl::QpInfo> qpInfos;
    for (auto& rdmaConn : connections_) {
        qpInfos.emplace_back(rdmaConn->GetQpInfo());
    }
    return qpInfos;
}

std::string HostCpuRoceChannel::Describe() const
{
    std::string msg = "HostCpuRoceChannel{";
    msg += Hccl::StringFormat("notifyNum:%u, dpuNotifyList:[-]", notifyNum_);
    msg += Hccl::StringFormat(", bufferNum:%u, localRmaBuffers: [", bufferNum_);
    for (auto& buf : localRmaBuffers_) {
        msg += buf->Describe();
        msg += ", ";
    }
    msg += Hccl::StringFormat("], connNum:%u, connections:[", connNum_);
    for (auto& conn : connections_) {
        msg += conn->Describe();
        msg += ", ";
    }
    msg += Hccl::StringFormat("], rdmaHandle: %p, %s, ", rdmaHandle_, channelStatus_.Describe().c_str());

    if (socket_ != nullptr) {
        msg += socket_->Describe();
    }

    msg += ", ";
    // msg += attr_.Describe();
    return msg;
}

HcclResult HostCpuRoceChannel::SetDfxCallback(std::function<HcclResult(const Hccl::TaskParam&, u64)> callback)
{
    dfxCallback_ = callback;
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::IbvPostRecv() const
{
    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
    CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);
    CHK_PRT_RET(
        localRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] localRmaBuffer is Empty", __func__),
        HCCL_E_ROCE_CONNECT);
    CHK_PRT_RET(
        rmtRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] rmtRmaBuffers is Empty", __func__),
        HCCL_E_ROCE_CONNECT);

    // 准备wr
    HCCL_INFO("[HostCpuRoceChannel::%s] call ibv_post_recv", __func__);
    for (uint32_t i = 0; i < qpInfo.size(); i++) {
        ibv_recv_wr recvWr{};
        ibv_recv_wr* recvbadWr = nullptr;
        ibv_sge recvsgList{};
        recvsgList.addr = localRmaBuffers_[0]->GetBufferInfo().first
                          + MEM_BLOCK_SIZE * i; // 本端起始地址，cclbuffer最小为1MB，足够使用
        recvsgList.length = MEM_BLOCK_SIZE;
        recvsgList.lkey = localRmaBuffers_[0]->GetLkey(); // 本端的访问秘钥
        recvWr.wr_id = i;
        recvWr.sg_list = &recvsgList;
        recvWr.next = nullptr;
        recvWr.num_sge = 1;

        HCCL_INFO("qp_state[%u] = [%u]", i, qpInfo[i].qp->state);
        int32_t ret = ibv_post_recv(qpInfo[i].qp, &recvWr, &recvbadWr);
        CHK_PRT_RET(
            ret == ENOMEM,
            HCCL_WARNING(
                "[HostCpuRoceChannel][%s] post recv wqe overflow. ret:%d, "
                "badWr->wr_id[%llu], badWr->sg_list->addr[%llu]",
                __func__, ret, recvbadWr->wr_id, recvbadWr->sg_list->addr),
            HCCL_E_AGAIN);

        CHK_PRT_RET(
            ret != 0,
            HCCL_ERROR(
                "[HostCpuRoceChannel][%s] ibv_post_recv failed. ret:%d, "
                "badWr->wr_id[%llu], badWr->sg_list->addr[%llu]",
                __func__, ret, recvbadWr->wr_id, recvbadWr->sg_list->addr),
            HCCL_E_NETWORK);
    }

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::PrepareNotifyWrResource(
    uint32_t qpIdx, const uint64_t len, const uint32_t remoteNotifyIdx, struct ibv_send_wr& notifyRecordWr,
    Hccl::TaskParam& taskParam) const
{
    taskParam.beginTime = hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    if (remoteNotifyIdx >= remoteDpuNotifyIds_.size()) {
        HCCL_ERROR(
            "[HostCpuRoceChannel::%s] remoteNotifyIdx[%u] out of the range of remoteDpuNotifyIds_[%zu].", __func__,
            remoteNotifyIdx, remoteDpuNotifyIds_.size());
        return HCCL_E_PARA;
    }
    uint32_t dpuNotifyId = remoteDpuNotifyIds_[remoteNotifyIdx];

    CHK_PRT_RET(
        localRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] localRmaBuffer is Empty", __func__),
        HCCL_E_ROCE_CONNECT);
    CHK_PRT_RET(
        rmtRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] rmtRmaBuffers is Empty", __func__),
        HCCL_E_ROCE_CONNECT);

    // 构造send_WR
    notifyRecordWr.sg_list->addr = localRmaBuffers_[0]->GetBufferInfo().first; // 本端起始地址
    notifyRecordWr.sg_list->length = 0;                                        // 取的本端长度
    notifyRecordWr.sg_list->lkey = localRmaBuffers_[0]->GetLkey();             // 本端的访问秘钥
    notifyRecordWr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    notifyRecordWr.send_flags = IBV_SEND_SIGNALED;
    notifyRecordWr.imm_data = dpuNotifyId;
    notifyRecordWr.next = nullptr;
    notifyRecordWr.num_sge = 1;
    notifyRecordWr.wr_id = qpIdx; // 用户定义工作请求id，建议：设为有意义的值
    notifyRecordWr.wr.rdma.rkey = rmtRmaBuffers_[0]->GetRkey();                               // 远端秘钥
    notifyRecordWr.wr.rdma.remote_addr = static_cast<uint64_t>(rmtRmaBuffers_[0]->GetAddr()); // 远端地址

    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_INLINE_WRITE;
    taskParam.taskPara.DMA.dst = reinterpret_cast<void*>(static_cast<uint64_t>(rmtRmaBuffers_[0]->GetAddr()));
    taskParam.taskPara.DMA.size = len;
    taskParam.taskPara.DMA.notifyID = dpuNotifyId;
    taskParam.taskPara.DMA.notifyValue = 1;
    taskParam.taskPara.DMA.linkType = Hccl::DfxLinkType::ROCE;
    taskParam.taskPara.DMA.dmaOp = Hccl::DmaOp::HCCL_DMA_WRITE;
    Hccl::IpAddress locIpAddr{};
    CHK_RET(hcomm::CommAddrToIpAddress(localEp_.commAddr, locIpAddr));
    std::string locAddrStr = locIpAddr.GetIpStr();
    CHK_SAFETY_FUNC_RET(memcpy_s(
        taskParam.taskPara.DMA.locAddr, sizeof(taskParam.taskPara.DMA.locAddr), locAddrStr.c_str(), locAddrStr.size()));
    Hccl::IpAddress rmtIpAddr{};
    CHK_RET(hcomm::CommAddrToIpAddress(remoteEp_.commAddr, rmtIpAddr));
    std::string rmtAddrStr = rmtIpAddr.GetIpStr();
    CHK_SAFETY_FUNC_RET(memcpy_s(
        taskParam.taskPara.DMA.rmtAddr, sizeof(taskParam.taskPara.DMA.rmtAddr), rmtAddrStr.c_str(), rmtAddrStr.size()));

    return HCCL_SUCCESS;
}

hccl::MemType HostCpuRoceChannel::NotifyIdToMemtypeHybird(uint32_t remoteNotifyIdx)
{
    if (remoteNotifyIdx == 0) {
        return hccl::MemType::ACK_NOTIFY_MEM;
    } else {
        return hccl::MemType::DATA_NOTIFY_MEM;
    }

    return hccl::MemType::DATA_NOTIFY_MEM;
}

HcclResult HostCpuRoceChannel::BuildNotifyWrHybird(const uint32_t remoteNotifyIdx, struct ibv_send_wr& notifRecordWr)
{
    hccl::MemType type = NotifyIdToMemtypeHybird(remoteNotifyIdx);

    notifRecordWr.sg_list->addr = reinterpret_cast<uint64_t>(localMemMsg_[hccl::NOTIFY_SRC_MEM].addr);
    notifRecordWr.sg_list->length = localMemMsg_[hccl::NOTIFY_SRC_MEM].len;
    notifRecordWr.sg_list->lkey = localMemMsg_[hccl::NOTIFY_SRC_MEM].lkey;
    notifRecordWr.opcode = IBV_WR_RDMA_WRITE;
    notifRecordWr.send_flags = IBV_SEND_SIGNALED;
    notifRecordWr.next = nullptr;
    notifRecordWr.num_sge = 1;
    notifRecordWr.wr_id = 0;
    notifRecordWr.wr.rdma.rkey = remoteMemMsg_[type].lkey;
    notifRecordWr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(remoteMemMsg_[type].addr);

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::NotifyRecord(const uint32_t remoteNotifyIdx)
{
    // 1.构造send_WR
    struct ibv_send_wr notifyRecordWr {};
    struct ibv_send_wr* sendbadWr = nullptr;
    struct ibv_sge sgList {};
    notifyRecordWr.sg_list = &sgList;
    Hccl::TaskParam taskParam{};

    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
    CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);

    // 3.调用ibv_post_send
    for (uint32_t i = 0; i < qpInfo.size(); i++) {
        HCCL_INFO("[HostCpuRoceChannel::%s] call ibv_post_send, qp_state[%u] = [%u]", __func__, i, qpInfo[i].qp->state);
        if (isHybridMode_) {
            BuildNotifyWrHybird(remoteNotifyIdx, notifyRecordWr);
        } else {
            CHK_RET(PrepareNotifyWrResource(i, MEM_BLOCK_SIZE, remoteNotifyIdx, notifyRecordWr, taskParam));
        }
        int32_t ret = ibv_post_send(qpInfo[i].qp, &notifyRecordWr, &sendbadWr);
        if (ret != 0 && sendbadWr == nullptr) {
            HCCL_ERROR("[HostCpuRoceChannel::%s] ibv_post_send failed while badWr is nullptr", __func__);
            return HCCL_E_INTERNAL;
        }
        CHK_PRT_RET(
            ret == ENOMEM,
            HCCL_WARNING(
                "[HostCpuRoceChannel][%s] post send wqe overflow. ret:%d, badWr->wr_id[%llu], "
                "badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], badWr->wr.ud.remote_qpn[%u]",
                __func__, ret, sendbadWr->wr_id, sendbadWr->sg_list->addr, sendbadWr->wr.rdma.remote_addr,
                sendbadWr->wr.ud.remote_qpn),
            HCCL_E_AGAIN);

        CHK_PRT_RET(
            ret != 0,
            HCCL_ERROR(
                "[HostCpuRoceChannel][%s] ibv_post_send failed. ret:%d, badWr->wr_id[%llu], "
                "badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], badWr->wr.ud.remote_qpn[%u]",
                __func__, ret, sendbadWr->wr_id, sendbadWr->sg_list->addr, sendbadWr->wr.rdma.remote_addr,
                sendbadWr->wr.ud.remote_qpn),
            HCCL_E_NETWORK);
        if (wqeNums_[i] == INT32_MAX) {
            HCCL_ERROR("[HostCpuRoceChannel::%s] wqeNums_[%u] has reached the maximum value of uint32_t.", __func__, i);
            return HCCL_E_INTERNAL;
        }
        wqeNums_[i]++;
        HCCL_INFO("[HostCpuRoceChannel::NotifyRecord] NotifyRecord end, wqeNums_[%u]=%d", i, wqeNums_[i]);
    }

    taskParam.endTime = hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    if (dfxCallback_ != nullptr) {
        return dfxCallback_(taskParam, reinterpret_cast<u64>(this));
    }
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout)
{
    Hccl::TaskParam taskParam{};
    taskParam.beginTime = hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();

    if (isHybridMode_) {
        return NotifyWaitHybrid(localNotifyIdx, timeout);
    }

    CHK_PRT_RET(
        localNotifyIdx >= localDpuNotifyIds_.size(),
        HCCL_ERROR(
            "[HostCpuRoceChannel::%s] localNotifyIdx[%u] out of the range of localDpuNotifyIds_[%zu].", __func__,
            localNotifyIdx, localDpuNotifyIds_.size()),
        HCCL_E_PARA);

    uint32_t dpuNotifyId = localDpuNotifyIds_[localNotifyIdx];

    // 1. 准备WR
    struct ibv_wc wc {};
    std::lock_guard<std::mutex> lock(cq_mutex);
    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
    CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);
    HCCL_INFO(
        "[HostCpuRoceChannel::NotifyWait] poll recvCq = %p, localNotifyIdx = %u, notifyId = %u.", qpInfo[0].recvCq,
        localNotifyIdx, dpuNotifyId);

    // 2.轮询rq_cq
    auto startTime = std::chrono::steady_clock::now();
    auto waitTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(timeout));
    for (uint32_t i = 0; i < qpInfo.size(); i++) {
        CHK_PRT_RET(
            qpInfo[i].recvCq == nullptr, HCCL_ERROR("[HostCpuRoceChannel::%s] recvCq[%u] is null", __func__, i),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            qpInfo[i].qp == nullptr, HCCL_ERROR("[HostCpuRoceChannel::%s] qp[%u] is null", __func__, i),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            qpInfo[i].recvCq->context == nullptr,
            HCCL_ERROR("[HostCpuRoceChannel::%s] recvCq[%u]->context is null", __func__, i), HCCL_E_INTERNAL);

        while (true) {
            auto actualNum = ibv_poll_cq(qpInfo[i].recvCq, 1, &wc);
            CHK_PRT_RET(
                actualNum < 0,
                HCCL_ERROR("[HostCpuRoceChannel::%s] ibv_poll_cq err. actualNum=%d", __func__, actualNum),
                HCCL_E_NETWORK);

            if (actualNum > 0 && wc.imm_data == dpuNotifyId) {
                if (wc.status != IBV_WC_SUCCESS) {
                    HCCL_ERROR("[HostCpuRoceChannel][%s] ibv_poll_cq return wc.status[%d].", __func__, wc.status);
                    return ReportWcStatusError(wc.status);
                }
                HCCL_INFO("[HostCpuRoceChannel::NotifyWait] poll cq success");
                break;
            } else if (actualNum > 0) {
                CHK_PRT_RET(
                    true,
                    HCCL_ERROR(
                        "[HostCpuRoceChannel::%s] polled cq unexpected. imm_data[%u] != dpuNotifyId[%u]", __func__,
                        wc.imm_data, dpuNotifyId),
                    HCCL_E_NETWORK);
            }

            if ((std::chrono::steady_clock::now() - startTime) >= waitTime) {
                CHK_PRT_RET(
                    true,
                    HCCL_ERROR("[HostCpuRoceChannel][%s] call ibv_poll_cq timeout. actualNum=%d", __func__, actualNum),
                    HCCL_E_TIMEOUT);
            }
        }
    }

    CHK_RET(IbvPostRecv());
    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_NOTIFY_WAIT;
    taskParam.taskPara.Notify.notifyID = dpuNotifyId;
    taskParam.taskPara.Notify.value = 1;
    taskParam.endTime = hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    if (dfxCallback_ != nullptr) {
        return dfxCallback_(taskParam, reinterpret_cast<u64>(this));
    }
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::ReportWcStatusError(enum ibv_wc_status status)
{
    Hccl::IpAddress localIp, remoteIp;
    (void)hcomm::CommAddrToIpAddress(localEp_.commAddr, localIp);
    (void)hcomm::CommAddrToIpAddress(remoteEp_.commAddr, remoteIp);
    HCCL_ERROR("[HostCpuRoceChannel::%s] wc_status[%d].", __func__, status);
    RPT_INPUT_ERR(
        true, "EI0013",
        std::vector<std::string>(
            {"localServerId", "localDeviceId", "localDeviceIp", "remoteServerId", "remoteDeviceId", "remoteDeviceIp"}),
        std::vector<std::string>(
            {std::to_string(localEp_.loc.device.serverIdx), std::to_string(localEp_.loc.device.devPhyId),
             localIp.GetIpStr(), std::to_string(remoteEp_.loc.device.serverIdx),
             std::to_string(remoteEp_.loc.device.devPhyId), remoteIp.GetIpStr()}));
    return HCCL_E_NETWORK;
}

HcclResult HostCpuRoceChannel::PrepareWriteWrResource(
    const void* dst, const void* src, const uint64_t len, const uint32_t remoteNotifyIdx,
    struct ibv_send_wr& writeWithNotifyWr, Hccl::TaskParam& taskParam) const
{
    taskParam.beginTime = hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    if (remoteNotifyIdx >= remoteDpuNotifyIds_.size()) {
        HCCL_ERROR(
            "[HostCpuRoceChannel::%s] remoteNotifyIdx[%u] out of the range of remoteDpuNotifyIds_[%zu].", __func__,
            remoteNotifyIdx, remoteDpuNotifyIds_.size());
        return HCCL_E_PARA;
    }
    uint32_t dpuNotifyId = remoteDpuNotifyIds_[remoteNotifyIdx];

    CHK_PRT_RET(
        localRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] localRmaBuffer is Empty", __func__),
        HCCL_E_ROCE_CONNECT);
    CHK_PRT_RET(
        rmtRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] rmtRmaBuffers is Empty", __func__),
        HCCL_E_ROCE_CONNECT);

    // 1. 构造WR
    CHK_PRT_RET(
        len > UINT32_MAX, HCCL_ERROR("[HostCpuRoceChannel][%s] the len[%llu] exceeds the size of u32.", __func__, len),
        HCCL_E_PARA);

    size_t localIdx = 0;
    CHK_RET(FindLocalBuffer(reinterpret_cast<uint64_t>(src), len, localIdx));
    size_t rmtIdx = 0;
    CHK_RET(FindRemoteBuffer(reinterpret_cast<uint64_t>(dst), len, rmtIdx));

    writeWithNotifyWr.sg_list->addr = reinterpret_cast<uint64_t>(src); // 本端起始地址
    writeWithNotifyWr.sg_list->length = static_cast<uint32_t>(len);
    writeWithNotifyWr.sg_list->lkey = localRmaBuffers_[localIdx]->GetLkey(); // 本端的访问秘钥

    writeWithNotifyWr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    writeWithNotifyWr.send_flags = IBV_SEND_SIGNALED;
    writeWithNotifyWr.next = nullptr;
    writeWithNotifyWr.num_sge = 1;
    writeWithNotifyWr.wr_id = 0;
    writeWithNotifyWr.imm_data = dpuNotifyId;
    writeWithNotifyWr.wr.rdma.rkey = rmtRmaBuffers_[rmtIdx]->GetRkey();
    writeWithNotifyWr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(dst);

    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY;
    taskParam.taskPara.DMA.src = src;
    taskParam.taskPara.DMA.dst = dst;
    taskParam.taskPara.DMA.size = len;
    taskParam.taskPara.DMA.notifyID = dpuNotifyId;
    taskParam.taskPara.DMA.notifyValue = 1;
    taskParam.taskPara.DMA.linkType = Hccl::DfxLinkType::ROCE;
    taskParam.taskPara.DMA.dmaOp = Hccl::DmaOp::HCCL_DMA_WRITE;
    Hccl::IpAddress locIpAddr{};
    CHK_RET(hcomm::CommAddrToIpAddress(localEp_.commAddr, locIpAddr));
    std::string locAddrStr = locIpAddr.GetIpStr();
    CHK_SAFETY_FUNC_RET(memcpy_s(
        taskParam.taskPara.DMA.locAddr, sizeof(taskParam.taskPara.DMA.locAddr), locAddrStr.c_str(), locAddrStr.size()));
    Hccl::IpAddress rmtIpAddr{};
    CHK_RET(hcomm::CommAddrToIpAddress(remoteEp_.commAddr, rmtIpAddr));
    std::string rmtAddrStr = rmtIpAddr.GetIpStr();
    CHK_SAFETY_FUNC_RET(memcpy_s(
        taskParam.taskPara.DMA.rmtAddr, sizeof(taskParam.taskPara.DMA.rmtAddr), rmtAddrStr.c_str(), rmtAddrStr.size()));

    return HCCL_SUCCESS;
}

HcclResult
HostCpuRoceChannel::WriteWithNotify(void* dst, const void* src, const uint64_t len, const uint32_t remoteNotifyIdx)
{
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    CHK_PRT_RET(
        maxMsgSize_ == 0, HCCL_ERROR("[HostCpuRoceChannel::%s] maxMsgSize_ is 0, channel not initialized", __func__),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        GetQpInfos().empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);
    HCCL_INFO(
        "[HostCpuRoceChannel::%s] START. dst[%p], src[%p], len[0x%llx], remoteNotifyIdx[%u].", __func__, dst, src, len,
        remoteNotifyIdx);
    std::vector<int> wqeNumBefore = wqeNums_;

    if (isHybridMode_) {
        return WriteWithNotifyHybrid(dst, src, len, remoteNotifyIdx);
    }

    // 前 N-1 块: 普通 RDMA_WRITE
    uint64_t offset = 0;
    while (len - offset > maxMsgSize_) {
        CHK_RET(PostRdmaOp(
            __func__, IBV_WR_RDMA_WRITE, static_cast<char*>(const_cast<void*>(src)) + offset,
            static_cast<const char*>(dst) + offset, maxMsgSize_));
        offset += maxMsgSize_;
    }

    // 尾块: RDMA_WRITE_WITH_IMM，携带 notify
    void* tailDst = static_cast<char*>(dst) + offset;
    const void* tailSrc = static_cast<const char*>(src) + offset;
    uint64_t tailLen = len - offset;

    // 计算每个qp需要发送的数据量
    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
    uint32_t useQpNum = qpInfo.size();
    uint32_t tileLen = tailLen / useQpNum;
    uint32_t tileLenTail = tailLen - (useQpNum - 1) * tileLen;
    if ((qpInfo.size() != 1) && (tileLen != 0) && (tileLen < channelDesc_.roceAttr.qpThreshold)) {
        useQpNum = (tailLen - 1) / channelDesc_.roceAttr.qpThreshold
                   + 1; // 自适应选择QP数发送数据，保证每个qp分担的数据量满足最小阈值
        if (useQpNum > qpInfo.size()) {
            useQpNum = qpInfo.size();
        }
        tileLen = tailLen / useQpNum;
        tileLenTail = tailLen - (useQpNum - 1) * tileLen;
        HCCL_INFO(
            "[HostCpuRoceChannel::%s] The data allocated to each Qp (%u) is below the Qp Threshold (%u). "
            "the count of Qp for data sending is adaptively adjusted to %u.",
            __func__, tileLen, channelDesc_.roceAttr.qpThreshold, useQpNum);
    }

    // 构造 WR
    Hccl::TaskParam taskParam{};
    uint32_t wrLen;
    for (uint32_t i = 0; i < qpInfo.size(); i++) {
        if (i < useQpNum - 1) {
            wrLen = tileLen;
        } else if (i == useQpNum - 1) {
            wrLen = tileLenTail;
        } else {
            wrLen = 0;
        }
        struct ibv_send_wr writeWithNotifyWr {};
        struct ibv_sge sgList {};
        writeWithNotifyWr.sg_list = &sgList;
        CHK_RET(PrepareWriteWrResource(
            static_cast<char*>(tailDst) + tileLen * i, static_cast<const char*>(tailSrc) + tileLen * i, wrLen,
            remoteNotifyIdx, writeWithNotifyWr, taskParam));
        CHK_RET(PostAndCheckSend(qpInfo[i].qp, i, __func__, writeWithNotifyWr));
        HCCL_INFO(
            "[HostCpuRoceChannel::%s] SUCCESS. qp[%u], wrlen[0x%llx], newWqe[%u], wqeNums_[%u].", __func__, i, wrLen,
            wqeNums_[i] - wqeNumBefore[i], wqeNums_[i]);
    }
    fenceFlag_ = false;
    taskParam.endTime = hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    if (dfxCallback_ != nullptr) {
        return dfxCallback_(taskParam, reinterpret_cast<u64>(this));
    }

    return HCCL_SUCCESS;
}

void HostCpuRoceChannel::BuildRdmaWr(
    const char* caller, ibv_wr_opcode opcode, void* localAddr, const void* remoteAddr, uint64_t len, size_t localIdx,
    size_t rmtIdx, struct ibv_send_wr& wr, struct ibv_sge& sg) const
{
    (void)caller;
    wr.sg_list = &sg;
    wr.sg_list->addr = reinterpret_cast<uint64_t>(localAddr);
    wr.sg_list->length = static_cast<uint32_t>(len);
    wr.sg_list->lkey = localRmaBuffers_[localIdx]->GetLkey();

    wr.opcode = opcode;
    wr.send_flags = (fenceFlag_ == true ? (IBV_SEND_SIGNALED | IBV_SEND_FENCE) : IBV_SEND_SIGNALED);
    wr.next = nullptr;
    wr.num_sge = 1;
    wr.wr_id = 0;
    wr.wr.rdma.rkey = rmtRmaBuffers_[rmtIdx]->GetRkey();
    wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(remoteAddr);
}

HcclResult
HostCpuRoceChannel::PostAndCheckSend(struct ibv_qp* qp, uint32_t qpIdx, const char* caller, struct ibv_send_wr& wr)
{
    struct ibv_send_wr* badWr = nullptr;
    s32 ret = ibv_post_send(qp, &wr, &badWr);
    if (ret != 0 && badWr == nullptr) {
        HCCL_ERROR("[HostCpuRoceChannel::%s] ibv_post_send failed while badWr is nullptr", caller);
        return HCCL_E_INTERNAL;
    }
    CHK_PRT_RET(
        ret == ENOMEM,
        HCCL_WARNING(
            "[HostCpuRoceChannel::%s] post send wqe overflow. ret:%d, "
            "badWr->wr_id[%llu], badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], "
            "badWr->wr.ud.remote_qpn[%u]",
            caller, ret, badWr->wr_id, badWr->sg_list->addr, badWr->wr.rdma.remote_addr, badWr->wr.ud.remote_qpn),
        HCCL_E_AGAIN);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HostCpuRoceChannel::%s] ibv_post_send failed. ret:%d, "
            "badWr->wr_id[%llu], badWr->sg_list->addr[%llu], badWr->wr.rdma.remote_addr[%llu], "
            "badWr->wr.ud.remote_qpn[%u]",
            caller, ret, badWr->wr_id, badWr->sg_list->addr, badWr->wr.rdma.remote_addr, badWr->wr.ud.remote_qpn),
        HCCL_E_NETWORK);
    CHK_PRT_RET(
        wqeNums_[qpIdx] == INT32_MAX,
        HCCL_ERROR("[HostCpuRoceChannel::%s] wqeNums_[%u] has reached the maximum value of uint32_t.", caller, qpIdx),
        HCCL_E_INTERNAL);
    wqeNums_[qpIdx]++;
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::PostRdmaOp(
    const char* caller, ibv_wr_opcode opcode, void* localAddr, const void* remoteAddr, const uint64_t len)
{
    HCCL_INFO(
        "[HostCpuRoceChannel::%s] Slice START. localAddr[%p], remoteAddr[%p], len[0x%llx].", caller, localAddr,
        remoteAddr, len);

    CHK_PRT_RET(
        GetQpInfos().empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", caller), HCCL_E_ROCE_CONNECT);
    CHK_PRT_RET(
        localRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] localRmaBuffer is Empty", caller),
        HCCL_E_ROCE_CONNECT);
    CHK_PRT_RET(
        rmtRmaBuffers_.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] rmtRmaBuffers is Empty", caller),
        HCCL_E_ROCE_CONNECT);
    if (len > maxMsgSize_) {
        HCCL_WARNING(
            "[HostCpuRoceChannel::%s] len[0x%llx] exceeds maxMsgSize_[0x%llx], caller should slice before posting.",
            caller, len, maxMsgSize_);
    }

    // 1. 查找 buffer 索引
    auto startTime = std::chrono::steady_clock::now();
    size_t localIdx = 0;
    CHK_RET(FindLocalBuffer(reinterpret_cast<uint64_t>(localAddr), len, localIdx));
    size_t rmtIdx = 0;
    CHK_RET(FindRemoteBuffer(reinterpret_cast<uint64_t>(remoteAddr), len, rmtIdx));
    auto endTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    HCCL_INFO("[HostCpuRoceChannel::%s] check buffer takes time [%lld]us", caller, elapsed);

    // 2. 构造 WR 并发送
    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
    uint32_t useQpNum = qpInfo.size();
    uint32_t tileLen = len / useQpNum;
    uint32_t tileLenTail = len - (useQpNum - 1) * tileLen;
    if ((tileLen != 0) && (tileLen < channelDesc_.roceAttr.qpThreshold)) {
        useQpNum = (len - 1) / channelDesc_.roceAttr.qpThreshold
                   + 1; // 自适应选择QP数发送数据，保证每个qp分担的数据量满足最小阈值
        if (useQpNum > qpInfo.size()) {
            useQpNum = qpInfo.size();
        }
        tileLen = len / useQpNum;
        tileLenTail = len - (useQpNum - 1) * tileLen;
        HCCL_INFO(
            "[HostCpuRoceChannel::%s] The data allocated to each Qp (%u) is below the Qp Threshold (%u). "
            "the count of Qp for data sending is adaptively adjusted to %u.",
            __func__, tileLen, channelDesc_.roceAttr.qpThreshold, useQpNum);
    }

    uint32_t wrLen;
    for (uint32_t i = 0; i < qpInfo.size(); i++) {
        if (i < useQpNum - 1) {
            wrLen = tileLen;
        } else if (i == useQpNum - 1) {
            wrLen = tileLenTail;
        } else {
            wrLen = 0;
        }
        struct ibv_send_wr wr {};
        struct ibv_sge sg;
        BuildRdmaWr(
            caller, opcode, static_cast<char*>(localAddr) + tileLen * i,
            static_cast<const char*>(remoteAddr) + tileLen * i, wrLen, localIdx, rmtIdx, wr, sg);
        CHK_RET(PostAndCheckSend(qpInfo[i].qp, i, caller, wr));
        HCCL_INFO(
            "[HostCpuRoceChannel::%s] Slice SUCCESS. qp[%u] wrLen[0x%llx], wqeNums_[%u]=%d", caller, i, wrLen, i,
            wqeNums_[i]);
    }
    fenceFlag_ = false;
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::Write(void* dst, const void* src, const uint64_t len)
{
    CHK_PRT_RET(
        maxMsgSize_ == 0, HCCL_ERROR("[HostCpuRoceChannel::%s] maxMsgSize_ is 0, channel not initialized", __func__),
        HCCL_E_INTERNAL);
    HCCL_INFO("[HostCpuRoceChannel::%s] START. dst[%p], src[%p], len[0x%llx].", __func__, dst, src, len);
    std::vector<int> wqeNumBefore = wqeNums_;
    uint64_t offset = 0;
    while (offset < len) {
        uint64_t chunkLen = std::min(len - offset, maxMsgSize_);
        CHK_RET(PostRdmaOp(
            __func__, IBV_WR_RDMA_WRITE, static_cast<char*>(const_cast<void*>(src)) + offset,
            static_cast<const char*>(dst) + offset, chunkLen));
        offset += chunkLen;
    }
    HCCL_INFO(
        "[HostCpuRoceChannel::%s] SUCCESS. len[0x%llx], newWqe[%d], wqeNums_[%d].", __func__, len,
        wqeNums_[0] - wqeNumBefore[0], wqeNums_[0]);
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::Read(void* dst, const void* src, const uint64_t len)
{
    CHK_PRT_RET(
        maxMsgSize_ == 0, HCCL_ERROR("[HostCpuRoceChannel::%s] maxMsgSize_ is 0, channel not initialized", __func__),
        HCCL_E_INTERNAL);
    HCCL_INFO("[HostCpuRoceChannel::%s] START. dst[%p], src[%p], len[0x%llx].", __func__, dst, src, len);
    std::vector<int> wqeNumBefore = wqeNums_;
    uint64_t offset = 0;
    while (offset < len) {
        uint64_t chunkLen = std::min(len - offset, maxMsgSize_);
        CHK_RET(PostRdmaOp(
            __func__, IBV_WR_RDMA_READ, static_cast<char*>(dst) + offset, static_cast<const char*>(src) + offset,
            chunkLen));
        offset += chunkLen;
    }
    HCCL_INFO(
        "[HostCpuRoceChannel::%s] SUCCESS. len[0x%llx], newWqe[%d], wqeNums_[%d].", __func__, len,
        wqeNums_[0] - wqeNumBefore[0], wqeNums_[0]);
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::FindLocalBuffer(const uint64_t addr, const uint64_t len, size_t& targetIdx) const
{
    uint64_t endAddr = addr + len;
    HCCL_INFO(
        "[HostCpuRoceChannel::%s] START. Finding buffer addr[0x%llx], len[0x%llx], addr+len[0x%llx].", __func__, addr,
        len, endAddr);
    for (size_t i = 0; i < localRmaBuffers_.size(); ++i) {
        CHK_PTR_NULL(localRmaBuffers_[i]);
        uint64_t bufAddr = localRmaBuffers_[i]->GetBufferInfo().first;
        uint64_t bufSize = localRmaBuffers_[i]->GetBufferInfo().second;
        uint64_t bufEndAddr = bufAddr + bufSize;
        HCCL_INFO(
            "[HostCpuRoceChannel::%s] Comparing with saved localRmaBuffer[%zu]: addr[0x%llx], len[0x%llx], "
            "addr+len[0x%llx].",
            __func__, i, bufAddr, bufSize, bufEndAddr);
        if (addr >= bufAddr && endAddr <= bufEndAddr) {
            targetIdx = i;
            HCCL_INFO("[HostCpuRoceChannel::%s] SUCCESS. targetIdx[%zu]", __func__, targetIdx);
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR(
        "[HostCpuRoceChannel::%s] FAIL. Can not find target buffer addr[0x%llx], len[0x%llx], addr+len[0x%llx].",
        __func__, addr, len, endAddr);
    return HCCL_E_NOT_FOUND;
}

HcclResult HostCpuRoceChannel::FindRemoteBuffer(const uint64_t addr, const uint64_t len, size_t& targetIdx) const
{
    uint64_t endAddr = addr + len;
    HCCL_INFO(
        "[HostCpuRoceChannel::%s] START. Finding buffer addr[0x%llx], len[0x%llx], addr+len[0x%llx].", __func__, addr,
        len, endAddr);
    for (size_t i = 0; i < rmtRmaBuffers_.size(); ++i) {
        CHK_PTR_NULL(rmtRmaBuffers_[i]);
        uint64_t bufAddr = static_cast<uint64_t>(rmtRmaBuffers_[i]->GetAddr());
        uint64_t bufSize = rmtRmaBuffers_[i]->GetSize();
        uint64_t bufEndAddr = bufAddr + bufSize;
        HCCL_INFO(
            "[HostCpuRoceChannel::%s] Comparing with saved rmtRmaBuffers[%zu]: addr[0x%llx], len[0x%llx], "
            "addr+len[0x%llx].",
            __func__, i, bufAddr, bufSize, bufEndAddr);
        if (addr >= bufAddr && endAddr <= bufEndAddr) {
            targetIdx = i;
            HCCL_INFO("[HostCpuRoceChannel::%s] SUCCESS. targetIdx[%zu]", __func__, targetIdx);
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR(
        "[HostCpuRoceChannel::%s] FAIL. Can not find target buffer addr[0x%llx], len[0x%llx], addr+len[0x%llx].",
        __func__, addr, len, endAddr);
    return HCCL_E_NOT_FOUND;
}

HcclResult HostCpuRoceChannel::WaitForFenceCompletion()
{
    const std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
    CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[HostCpuRoceChannel::%s] qpInfos is Empty", __func__), HCCL_E_ROCE_CONNECT);
    uint32_t fenceCount = 0;
    for (uint32_t i = 0; i < qpInfo.size(); i++) {
        if (wqeNums_[i] == 0) {
            fenceCount += 1;
        }
    }
    if (fenceCount == qpInfo.size()) {
        fenceFlag_ = true;
        HCCL_INFO("[HostCpuRoceChannel::%s] SUCCESS. elements in wqeNums_ are 0.", __func__);
        return HCCL_SUCCESS;
    }

    auto timeout = std::chrono::milliseconds(FENCE_TIMEOUT_MS);
    for (uint32_t i = 0; i < qpInfo.size(); i++) {
        std::vector<struct ibv_wc> wc(wqeNums_[i]);
        CHK_PRT_RET(
            qpInfo[i].sendCq == nullptr, HCCL_ERROR("[HostCpuRoceChannel::%s] qp[%u] sendCq is null", __func__, i),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            qpInfo[i].qp == nullptr, HCCL_ERROR("[HostCpuRoceChannel::%s] qp[%u] is null", __func__, i),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            qpInfo[i].sendCq->context == nullptr,
            HCCL_ERROR("[HostCpuRoceChannel::%s] qp[%u] sendCq->context is null", __func__, i), HCCL_E_INTERNAL);

        auto startTime = std::chrono::steady_clock::now();
        while (true) {
            int actualNum = IbvPollCq(qpInfo[i].sendCq, wqeNums_[i], wc.data());
            if (actualNum < 0) {
                HCCL_ERROR(
                    "[HostCpuRoceChannel::%s] qp[%u] ibv_poll_cq failed. actualNum: %d.", __func__, i, actualNum);
                return HCCL_E_NETWORK;
            }

            if (actualNum > wqeNums_[i]) {
                HCCL_ERROR(
                    "[HostCpuRoceChannel::%s] qp[%u] ibv_poll_cq polled more completions (%d) than expected (%d).",
                    __func__, i, actualNum, wqeNums_[i]);
                return HCCL_E_INTERNAL;
            } else if (actualNum > 0) {
                for (int j = 0; j < actualNum; j++) {
                    if (wc[j].status != IBV_WC_SUCCESS) {
                        HCCL_ERROR(
                            "[HostCpuRoceChannel::%s] ibv_poll_cq error. wc[%d] status: %d.", __func__, j,
                            wc[j].status);
                        return HCCL_E_NETWORK;
                    }
                }
                wqeNums_[i] -= actualNum; // 减去已经完成的数量，继续等待剩余的完成
                if (wqeNums_[i] == 0) {
                    break; // 所有的wqe都已经完成，退出循环
                }
                startTime = std::chrono::steady_clock::now(); // 有进展，重置超时计时
            }

            if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                HCCL_ERROR(
                    "[HostCpuRoceChannel][%s] qp[%u] call ibv_poll_cq timeout, remaining wqeNum[%u].", __func__, i,
                    wqeNums_[i]);
                return HCCL_E_TIMEOUT;
            }
        }
        wqeNums_[i] = 0; // 所有的wqe都已经完成，重置计数器
        HCCL_INFO("[HostCpuRoceChannel::%s] SUCCESS. wqeNums_[%u]=%d.", __func__, i, wqeNums_[i]);
    }
    fenceFlag_ = true;
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::ChannelFence()
{
    std::lock_guard<std::mutex> lock(sendCq_mutex);
    Hccl::TaskParam taskParam{};
    taskParam.beginTime = hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HCCL_INFO("[HostCpuRoceChannel::%s] ChannelFence start, wqeNums_[0]=%d", __func__, wqeNums_[0]);
    HcclResult ret = WaitForFenceCompletion();
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    taskParam.taskType = Hccl::TaskParamType::TASK_DPU_CHANNEL_FENCE;
    taskParam.taskPara.Notify.notifyID = INVALID_U64;
    taskParam.taskPara.Notify.value = 1;
    taskParam.endTime = hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    if (dfxCallback_ != nullptr) {
        return dfxCallback_(taskParam, reinterpret_cast<u64>(this));
    }
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    CHK_PTR_NULL(notifyNum);
    *notifyNum = notifyNum_;
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::GetHcclBuffer(void*& addr, uint64_t& size)
{
    if (rmtRmaBuffers_.empty()) {
        HCCL_ERROR(
            "[HostCpuRoceChannel::%s] remote buffer is empty, please check if channel complete exchange data",
            __func__);
        return HCCL_E_INTERNAL;
    }
    addr = reinterpret_cast<void*>(rmtRmaBuffers_[0]->GetAddr());
    size = static_cast<uint64_t>(rmtRmaBuffers_[0]->GetSize());
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::Clean() { return HCCL_SUCCESS; }

HcclResult HostCpuRoceChannel::Resume() { return HCCL_SUCCESS; }

HcclResult HostCpuRoceChannel::CreateNotifyHybird(hccl::MemType notifyType, uint32_t notifyId)
{
    localNotifyAccess_ = 7;
    localNotifySize_ = 4;

    int8_t* ptr = new (std::nothrow) int8_t[localNotifySize_];
    CHK_PTR_NULL(ptr);

    if (memset_s(ptr, localNotifySize_, 0, localNotifySize_) < 0) {
        HCCL_ERROR("[HostCpuRoceChannel::CreateNotifyHybird] memset_s failed");
        delete[] ptr;
        return HCCL_E_MEMORY;
    }

    struct MrInfoT mrInfo {};
    mrInfo.addr = ptr;
    mrInfo.size = localNotifySize_;
    mrInfo.access = localNotifyAccess_;
    auto qpInfo = connections_[0]->GetQpInfo();
    if (HrtRaMrReg(qpInfo.qpHandle, &mrInfo) != HCCL_SUCCESS) {
        HCCL_ERROR("[HostCpuRoceChannel::CreateNotifyHybird] MrReg failed");
        delete[] ptr;
        return HCCL_E_MEMORY;
    }

    localMemMsg_[notifyType].addr = ptr;
    localMemMsg_[notifyType].lkey = mrInfo.lkey;
    localMemMsg_[notifyType].memType = notifyType;
    localMemMsg_[notifyType].len = localNotifySize_;
    localMemMsg_[notifyType].notifyId = notifyId;

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::CreateNotifyValueBufferHybird()
{
    if (CreateNotifyHybird(hccl::MemType::NOTIFY_SRC_MEM, hccl::MemType::NOTIFY_SRC_MEM) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HostCpuRoceChannel::CreateNotifyValueBufferHybird]Create host notify fail, type=%d",
            hccl::MemType::NOTIFY_SRC_MEM);
        return HCCL_E_MEMORY;
    }
    *reinterpret_cast<uint32_t*>(localMemMsg_[hccl::MemType::NOTIFY_SRC_MEM].addr) = 1;
    return HCCL_SUCCESS;
}

HcclResult
HostCpuRoceChannel::CreateNotifyBufferHybird(hccl::MemType notifyType, uint32_t notifyId, u8*& data, u64& size)
{
    if (CreateNotifyHybird(notifyType, notifyId) != HCCL_SUCCESS) {
        HCCL_ERROR("[HostCpuRoceChannel::CreateNotifyBufferHybird]Create host notify buffer fail, type=%d", notifyType);
        return HCCL_E_MEMORY;
    }

    CHK_SAFETY_FUNC_RET(memcpy_s(data, size, reinterpret_cast<void*>(&localMemMsg_[notifyType]), sizeof(hccl::MemMsg)));

    data += sizeof(hccl::MemMsg);
    size -= sizeof(hccl::MemMsg);

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::ExchangeCapability()
{
    HCCL_INFO("[Hybrid][HostCpuRoceChannel] Starting capability exchange");

    // 1. 构造本地能力信息（使用公共头文件中的默认值）
    RoCECapability localCap;
    localCap.InitDefaults();
    localCap.nicDeploy = NICDeployment::NIC_DEPLOYMENT_HOST;
    localCap.commStack = CommStackType::COMM_STACK_HOST_CPU_ROCE;

    // 2. 发送本地能力（合并为单次发送：totalLength已包含结构体大小）
    CHK_PRT_RET(
        !socket_->Send(&localCap, sizeof(localCap)),
        HCCL_ERROR("[HostCpuRoceChannel::%s] Send exchange localCap failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO("[Hybrid][HostCpuRoceChannel] Sent capability, version=%u", localCap.version);

    // 3. 接收对端能力（单次接收）
    RoCECapability recvCap;
    CHK_PRT_RET(
        !socket_->Recv(&recvCap, sizeof(recvCap)), HCCL_ERROR("[HostCpuRoceChannel::%s] Recv recvCap failed", __func__),
        HCCL_E_NETWORK);
    HCCL_INFO("[Hybrid][HostCpuRoceChannel] recvCap success");

    // 4. 先检查魔数，如果不对可能是旧版本，需要回退
    if (!RoCECapability::CheckMagic(reinterpret_cast<uint8_t*>(&recvCap), sizeof(recvCap))) {
        HCCL_WARNING("[Hybrid][HostCpuRoceChannel] Magic mismatch, peer may be old version. "
                     "Falling back to native mode.");
        // 回退到原生模式
        isHybridMode_ = false;
        // 标记为"跳过混合模式协商"，后续流程继续使用原生模式
        remoteCap_.magic = 0; // 标记为无效
        return HCCL_SUCCESS;
    }

    // 5. 魔数正确，解析对端能力
    if (!remoteCap_.Deserialize(reinterpret_cast<uint8_t*>(&recvCap), sizeof(recvCap))) {
        HCCL_ERROR("[Hybrid][HostCpuRoceChannel] Failed to deserialize capability");
        return HCCL_E_PARA;
    }

    // 6. 校验字段有效性
    if (!remoteCap_.Validate()) {
        HCCL_ERROR("[Hybrid][HostCpuRoceChannel] Capability validation failed");
        return HCCL_E_INTERNAL;
    }

    // 7. 版本兼容性处理（高版本兼容低版本）
    if (remoteCap_.version > ROCE_CAPABILITY_VERSION) {
        // 对端版本更高，使用本地版本的功能集（最小公分母）
        HCCL_INFO(
            "[Hybrid][HostCpuRoceChannel] Remote version %u > local %u, using local version features",
            remoteCap_.version, ROCE_CAPABILITY_VERSION);
    } else if (remoteCap_.version < ROCE_CAPABILITY_VERSION) {
        // 对端版本更低，使用对端版本的功能集（向下兼容）
        HCCL_INFO(
            "[Hybrid][HostCpuRoceChannel] Remote version %u < local %u, using remote version features",
            remoteCap_.version, ROCE_CAPABILITY_VERSION);
    }

    isHybridMode_ = (remoteCap_.commStack == CommStackType::COMM_STACK_TRANSPORT_IBVERBS) ? true : false;

    HCCL_INFO(
        "[Hybrid][HostCpuRoceChannel] Capability exchange success, "
        "remote commStack=%u, version=%u, mode=%s",
        static_cast<uint8_t>(remoteCap_.commStack), remoteCap_.version, isHybridMode_ ? "hybrid" : "normal");
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::RegisterUserMemHybird()
{
    struct MrInfoT mrInfo {};
    mrInfo.addr = reinterpret_cast<void*>(localRmaBuffers_[0]->GetAddr());
    mrInfo.size = localRmaBuffers_[0]->GetSize();
    mrInfo.access = 7;
    auto qpInfo = connections_[0]->GetQpInfo();
    CHK_RET(HrtRaMrReg(qpInfo.qpHandle, &mrInfo));

    localMemMsg_[hccl::USER_OUTPUT_MEM].addr = reinterpret_cast<void*>(localRmaBuffers_[0]->GetAddr());
    localMemMsg_[hccl::USER_OUTPUT_MEM].lkey = mrInfo.lkey;
    localMemMsg_[hccl::USER_OUTPUT_MEM].memType = hccl::USER_OUTPUT_MEM;
    localMemMsg_[hccl::USER_OUTPUT_MEM].len = localRmaBuffers_[0]->GetSize();
    localMemMsg_[hccl::USER_OUTPUT_MEM].notifyId = INVALID_DPU_NOTIFY_ID;

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::BuildExchangeDataLengthHybird()
{
    exchangeDataTotalSize_ = 0;
    exchangeDataTotalSize_ += sizeof(u32);              // qp数量
    exchangeDataTotalSize_ += sizeof(hccl::MemMsg) * 2; // output、input buffer
    exchangeDataTotalSize_ += sizeof(hccl::MemMsg) * 3; // 3个Notify
    exchangeDataTotalSize_ += sizeof(u8);               // atomic value
    HCCL_INFO("[BuildExchangeDataLengthHybird]ExchangeDataSize:%ld", exchangeDataTotalSize_);
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::BuildExchangeDataHybird()
{
    CHK_RET(BuildExchangeDataLengthHybird());

    exchangeDataForSend_.resize(exchangeDataTotalSize_);

    u8* data = exchangeDataForSend_.data();
    u64 size = exchangeDataTotalSize_;

    u32 qpNum = 1;
    CHK_SAFETY_FUNC_RET(memcpy_s(data, size, reinterpret_cast<void*>(&qpNum), sizeof(u32)));
    data += sizeof(u32);
    size -= sizeof(u32);

    CHK_SAFETY_FUNC_RET(
        memcpy_s(data, size, reinterpret_cast<void*>(&localMemMsg_[hccl::USER_OUTPUT_MEM]), sizeof(hccl::MemMsg)));
    data += sizeof(hccl::MemMsg);
    size -= sizeof(hccl::MemMsg);
    CHK_SAFETY_FUNC_RET(
        memcpy_s(data, size, reinterpret_cast<void*>(&localMemMsg_[hccl::USER_OUTPUT_MEM]), sizeof(hccl::MemMsg)));
    data += sizeof(hccl::MemMsg);
    size -= sizeof(hccl::MemMsg);

    CHK_RET(CreateNotifyValueBufferHybird());
    CHK_RET(CreateNotifyBufferHybird(hccl::DATA_NOTIFY_MEM, 1, data, size));
    CHK_RET(CreateNotifyBufferHybird(hccl::ACK_NOTIFY_MEM, 0, data, size));
    CHK_RET(CreateNotifyBufferHybird(hccl::DATA_ACK_NOTIFY_MEM, 2, data, size));

    u8 atomicWrite = 1;
    CHK_SAFETY_FUNC_RET(memcpy_s(data, size, reinterpret_cast<void*>(&atomicWrite), sizeof(u8)));
    data += sizeof(u8);
    size -= sizeof(u8);

    if (size != 0) {
        HCCL_ERROR("HostCpuRoceChannel::BuildExchangeDataHybird, failed to construct exchange data, size=%llu", size);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::GetRemoteAddrHybird(hccl::MemType memType, u8*& data, u64& size)
{
    CHK_SAFETY_FUNC_RET(
        memcpy_s(&remoteMemMsg_[static_cast<u32>(memType)], sizeof(hccl::MemMsg), data, sizeof(hccl::MemMsg)));
    data += sizeof(hccl::MemMsg);
    size -= sizeof(hccl::MemMsg);
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::ParseRecvExchangeDataHybird()
{
    u8* data = exchangeDataForRecv_.data();
    u64 size = exchangeDataTotalSize_;

    u32 remoteQpNum = 0;
    CHK_SAFETY_FUNC_RET(memcpy_s(reinterpret_cast<void*>(&remoteQpNum), sizeof(u32), data, sizeof(u32)));
    data += sizeof(u32);
    size -= sizeof(u32);

    CHK_RET(GetRemoteAddrHybird(hccl::USER_OUTPUT_MEM, data, size));
    CHK_RET(GetRemoteAddrHybird(hccl::USER_INPUT_MEM, data, size));
    CHK_RET(GetRemoteAddrHybird(hccl::DATA_NOTIFY_MEM, data, size));
    CHK_RET(GetRemoteAddrHybird(hccl::ACK_NOTIFY_MEM, data, size));
    CHK_RET(GetRemoteAddrHybird(hccl::DATA_ACK_NOTIFY_MEM, data, size));

    data += sizeof(u8);
    size -= sizeof(u8);

    if (size != 0) {
        HCCL_ERROR("HostCpuRoceChannel::ParseRecvExchangeDataHybird: failed to parse exchange data, size=%lld", size);
        return HCCL_E_INTERNAL;
    }

    Hccl::ExchangeRdmaBufferDto dto(
        (u64)remoteMemMsg_[static_cast<u32>(hccl::USER_OUTPUT_MEM)].addr,
        remoteMemMsg_[static_cast<u32>(hccl::USER_OUTPUT_MEM)].len,
        remoteMemMsg_[static_cast<u32>(hccl::USER_OUTPUT_MEM)].lkey, "HcclBuffer");
    rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle_, dto));

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::ConnectSingleQpHybrid(std::function<bool()> needStop)
{
    auto qpInfo = connections_[0]->GetQpInfo();
    bool hasSocket = (socket_ != nullptr);
    if (!hasSocket) {
        CHK_RET(hcomm::SocketMgr::GetInstance(devicePhyId_).GetSocket(*socketConfig_, socket_));
    }
    CHK_RET(HrtRaQpConnectAsync(qpInfo.qpHandle, socket_->GetFdHandle(), needStop));

    // 查询QP建链是否成功
    s32 qpStatus = 0;
    s32 raRet = 0;
    constexpr uint32_t timeoutSec = 120;
    constexpr auto timeout = std::chrono::seconds(timeoutSec);
    auto startTime = std::chrono::steady_clock::now();
    HCCL_INFO("HostCpuRoceChannel: waiting for qp status ready...");
    while (true) {
        CHK_PRT_RET(needStop(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("[Connect][Qp]get qp status timeout_=[%lld s], qp_status=[%d]", timeout.count(), qpStatus);
            if (!hasSocket) {
                hcomm::SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
            }
            return HCCL_E_TIMEOUT;
        }
        raRet = hrtGetRaQpStatus(qpInfo.qpHandle, &qpStatus);
        if ((!raRet) && (qpStatus == 1)) { // 为1时，qp 建链成功
            HCCL_INFO("In link ibv, QP get status success.");
            break;
        } else {
            SaluSleep(HYBRID_QP_STATUS_POLL_SLEEP_US);
        }
    }
    if (!hasSocket) {
        hcomm::SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
    }
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::ExchangeDataHybird()
{
    HCCL_INFO("[Hybrid] Starting hybrid data exchange");

    CHK_RET(RegisterUserMemHybird());

    CHK_RET(BuildExchangeDataHybird());

    CHK_PRT_RET(
        !socket_->Send(exchangeDataForSend_.data(), exchangeDataTotalSize_),
        HCCL_ERROR("[Hybrid] Send exchange data failed"), HCCL_E_NETWORK);

    exchangeDataForRecv_.resize(exchangeDataTotalSize_);
    CHK_PRT_RET(
        !socket_->Recv(exchangeDataForRecv_.data(), exchangeDataTotalSize_),
        HCCL_ERROR("[Hybrid] Recv exchange data failed"), HCCL_E_NETWORK);

    CHK_RET(ParseRecvExchangeDataHybird());

    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::WriteWithNotifyHybrid(void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    HCCL_INFO("[Hybrid] WriteWithNotifyHybrid start, len=%lu", len);

    // 参数校验
    CHK_PRT_RET(localRmaBuffers_.empty(), HCCL_ERROR("[Hybrid] localRmaBuffer is Empty"), HCCL_E_ROCE_CONNECT);

    std::vector<Hccl::QpInfo> qpInfo = GetQpInfos();
    CHK_PRT_RET(qpInfo.empty(), HCCL_ERROR("[Hybrid] qpInfos is Empty"), HCCL_E_ROCE_CONNECT);

    // 获取本地 buffer 信息
    hccl::MemType type = NotifyIdToMemtypeHybird(remoteNotifyIdx);

    // 校验数据长度
    CHK_PRT_RET(
        len > localRmaBuffers_[0]->GetSize(),
        HCCL_ERROR("[Hybrid] Data length %lu exceeds buffer size %lu", len, localRmaBuffers_[0]->GetSize()),
        HCCL_E_PARA);

    // 构造发送 WR 链：数据 WR + Notify WR
    struct ibv_send_wr dataWr {};
    struct ibv_send_wr notifyWr {};
    struct ibv_send_wr* badWr = nullptr;
    struct ibv_sge dataSge {};
    struct ibv_sge notifySge {};

    // 1. 数据 WR（RDMA Write）
    dataSge.addr = reinterpret_cast<uint64_t>(src);
    dataSge.length = len;
    dataSge.lkey = localRmaBuffers_[0]->GetLkey();

    dataWr.wr_id = 0;
    dataWr.opcode = IBV_WR_RDMA_WRITE;
    dataWr.send_flags = IBV_SEND_SIGNALED; // 需要 CQE 确认完成
    dataWr.sg_list = &dataSge;
    dataWr.num_sge = 1;
    dataWr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(dst);
    dataWr.wr.rdma.rkey = rmtRmaBuffers_[0]->GetRkey();

    // 2. Notify WR（写入对端 TransportIbverbs 的 Notify 内存）
    notifySge.addr = reinterpret_cast<uint64_t>(localMemMsg_[hccl::NOTIFY_SRC_MEM].addr);
    notifySge.length = localMemMsg_[hccl::NOTIFY_SRC_MEM].len;
    notifySge.lkey = localMemMsg_[hccl::NOTIFY_SRC_MEM].lkey; // 使用本地 buffer 的 lkey

    notifyWr.wr_id = 1;
    notifyWr.opcode = IBV_WR_RDMA_WRITE;
    notifyWr.send_flags = IBV_SEND_SIGNALED;
    notifyWr.sg_list = &notifySge;
    notifyWr.num_sge = 1;
    // Notify 写入对端 hostNotifyAddr 的偏移位置
    notifyWr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(remoteMemMsg_[type].addr);
    notifyWr.wr.rdma.rkey = remoteMemMsg_[type].lkey;

    // 链接 WR 链：dataWr -> notifyWr
    dataWr.next = &notifyWr;
    notifyWr.next = nullptr;

    // 3. 下发 WR 链
    int32_t ret = ibv_post_send(qpInfo[0].qp, &dataWr, &badWr);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[Hybrid] ibv_post_send failed, ret=%d", ret), HCCL_E_NETWORK);

    wqeNums_[0] += 2;

    HCCL_INFO("[Hybrid] WriteWithNotifyHybrid success");
    return HCCL_SUCCESS;
}

HcclResult HostCpuRoceChannel::NotifyWaitHybrid(uint32_t localNotifyIdx, uint32_t timeout)
{
    HCCL_INFO("[Hybrid] NotifyWaitHybrid start, idx=%u", localNotifyIdx);

    hccl::MemType type = NotifyIdToMemtypeHybird(localNotifyIdx);

    // 使用配置的超时时间和轮询间隔（timeOut单位为秒，默认30s）
    uint32_t pollTimeout = (timeout == 0) ? DEFAULT_NOTIFY_WAIT_TIMEOUT_S : timeout;
    uint32_t pollInterval = 1;

    // 使用原子操作读取 Notify 内存
    std::atomic<uint32_t>* notifyAddr = reinterpret_cast<std::atomic<uint32_t>*>(localMemMsg_[type].addr);
    const uint64_t expectedValue = 1;

    auto startTime = std::chrono::steady_clock::now();
    auto waitTime = std::chrono::seconds(pollTimeout);

    while (true) {
        // 使用原子操作读取，确保内存可见性
        if (notifyAddr->load(std::memory_order_acquire) == expectedValue) {
            // 读取成功后清零，为下一次通知做准备
            notifyAddr->store(0, std::memory_order_release);
            HCCL_INFO("[Hybrid] NotifyWaitHybrid success");
            return HCCL_SUCCESS;
        }

        // 检查超时
        if ((std::chrono::steady_clock::now() - startTime) >= waitTime) {
            HCCL_ERROR("[Hybrid] NotifyWaitHybrid timeout, notify idx:%d", localNotifyIdx);
            return HCCL_E_TIMEOUT;
        }

        SaluSleep(pollInterval);
    }
}

} // namespace hcomm_experimental
