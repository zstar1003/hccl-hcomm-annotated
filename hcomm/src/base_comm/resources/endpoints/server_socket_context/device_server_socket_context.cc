/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_server_socket_context.h"
#include "log.h"
#include "socket/socket.h"
#include "orion_adpt_utils.h"
#include "server_socket_manager.h"

namespace hcomm {

DeviceServerSocketContext::DeviceServerSocketContext(
    Hccl::ConnectProtoType protoType, uint32_t devPhyId, EndpointLocType locType, const CommAddr& commAddr)
    : protoType_(protoType),
      devPhyId_(devPhyId),
      locType_(locType),
      commAddr_(commAddr)
{}

DeviceServerSocketContext::~DeviceServerSocketContext()
{
    // 在派生类析构期内虚表仍指向本类，显式调停止监听以释放 ServerSocketManager 资源。
    // 析构函数 noexcept，ServerSocketStopListenImpl 返回值忽略。
    std::lock_guard<std::mutex> lock(portMutex_);
    if (dynamicPort_ != HCCL_INVALID_PORT) {
        (void)ServerSocketStopListenImpl(dynamicPort_);
    }
    dynamicPort_ = HCCL_INVALID_PORT;
}

HcclResult DeviceServerSocketContext::ServerSocketListen(const uint32_t port)
{
    if (locType_ != ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_INFO("[DeviceServerSocketContext][%s] locType[%d] skip create ServerSocket", __func__, locType_);
        return HCCL_SUCCESS;
    }
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(commAddr_, ipAddr));

    Hccl::DevNetPortType type = Hccl::DevNetPortType(protoType_);
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(devPhyId_), type, 0, ipAddr);
    uint32_t requestPort = port;
    HCCL_INFO(
        "[DeviceServerSocketContext][%s] devPhyId[%u], localPort[%s], requestPort[%u]", __func__, devPhyId_,
        localPort.Describe().c_str(), requestPort);

    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, devPhyId_, &requestPort));
    return HCCL_SUCCESS;
}

HcclResult DeviceServerSocketContext::ServerSocketStopListenImpl(const uint32_t port)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(commAddr_, ipAddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(protoType_);
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(devPhyId_), type, 0, ipAddr);
    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, port));
    return HCCL_SUCCESS;
}

HcclResult DeviceServerSocketContext::ServerSocketStopListen(const uint32_t port)
{
    return ServerSocketStopListenImpl(port);
}

HcclResult DeviceServerSocketContext::ServerSocketGetListenPort(uint32_t* port)
{
    std::lock_guard<std::mutex> lock(portMutex_);
    if (locType_ != ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_INFO("[DeviceServerSocketContext][%s] locType[%d] skip create ServerSocket", __func__, locType_);
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(port);
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(commAddr_, ipAddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(protoType_);
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(devPhyId_), type, 0, ipAddr);
    HCCL_INFO(
        "[DeviceServerSocketContext][%s] devPhyId[%u], localPort[%s]", __func__, devPhyId_,
        localPort.Describe().c_str());

    if (dynamicPort_ != HCCL_INVALID_PORT) {
        *port = dynamicPort_;
        HCCL_INFO(
            "[DeviceServerSocketContext][%s] already listening, return existing port[%u]", __func__, dynamicPort_);
        return HCCL_SUCCESS;
    }
    uint32_t requestPort = 0;
    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, devPhyId_, &requestPort));
    if (requestPort == 0 || requestPort == HCCL_INVALID_PORT) {
        HCCL_ERROR("[DeviceServerSocketContext][%s] get listen port failed, port is invalid", __func__);
        return HCCL_E_NETWORK;
    }
    dynamicPort_ = requestPort;
    *port = dynamicPort_;
    return HCCL_SUCCESS;
}

} // namespace hcomm
