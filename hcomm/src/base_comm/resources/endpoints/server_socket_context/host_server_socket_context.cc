/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_server_socket_context.h"
#include "log.h"
#include "adapter_rts_common.h"
#include "orion_adpt_utils.h"
#include "server_socket_manager.h"
#include "externalinput_pub.h"

namespace hcomm {

HostServerSocketContext::HostServerSocketContext(Hccl::ConnectProtoType protoType, const CommAddr& commAddr)
    : protoType_(protoType),
      commAddr_(commAddr)
{}

HostServerSocketContext::~HostServerSocketContext()
{
    // 在派生类析构期内虚表仍指向本类，显式调停止监听以释放 ServerSocketManager 资源。
    // 析构函数 noexcept，ServerSocketStopListenImpl 返回值忽略。
    std::lock_guard<std::mutex> lock(portMutex_);
    if (dynamicPort_ != HCCL_INVALID_PORT) {
        (void)ServerSocketStopListenImpl(dynamicPort_);
    }
    dynamicPort_ = HCCL_INVALID_PORT;
}

HcclResult HostServerSocketContext::ServerSocketListen(const uint32_t port)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(commAddr_, ipAddr));
    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(protoType_);
    Hccl::PortData localPort = Hccl::PortData(devPhyId, type, 0, ipAddr);
    HCCL_INFO("[HostServerSocketContext::%s] devPhyId[%u] ipAddr[%s]", __func__, devPhyId, ipAddr.Describe().c_str());
    uint32_t requestPort = port;
    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::HOST_NIC_TYPE, devPhyId, &requestPort));
    return HCCL_SUCCESS;
}

HcclResult HostServerSocketContext::ServerSocketStopListenImpl(const uint32_t port)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(commAddr_, ipAddr));
    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(protoType_);
    Hccl::PortData localPort = Hccl::PortData(devPhyId, type, 0, ipAddr);
    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, port));
    return HCCL_SUCCESS;
}

HcclResult HostServerSocketContext::ServerSocketStopListen(const uint32_t port)
{
    return ServerSocketStopListenImpl(port);
}

HcclResult HostServerSocketContext::ServerSocketGetListenPort(uint32_t* port)
{
    std::lock_guard<std::mutex> lock(portMutex_);
    CHK_PTR_NULL(port);
    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(commAddr_, ipAddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(protoType_);
    Hccl::PortData localPort = Hccl::PortData(devPhyId, type, 0, ipAddr);
    HCCL_INFO("[HostServerSocketContext::%s] devPhyId[%u], ipAddr[%s]", __func__, devPhyId, ipAddr.Describe().c_str());

    if (dynamicPort_ != HCCL_INVALID_PORT) {
        *port = dynamicPort_;
        HCCL_INFO("[HostServerSocketContext::%s] already listening, return existing port[%u]", __func__, dynamicPort_);
        return HCCL_SUCCESS;
    }
    uint32_t requestPort = 0;
    CHK_RET(ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::HOST_NIC_TYPE, devPhyId, &requestPort));
    if (requestPort == 0 || requestPort == HCCL_INVALID_PORT) {
        HCCL_ERROR("[HostServerSocketContext::%s] get listen port failed, port is invalid", __func__);
        return HCCL_E_NETWORK;
    }
    dynamicPort_ = requestPort;
    *port = dynamicPort_;
    return HCCL_SUCCESS;
}

} // namespace hcomm
