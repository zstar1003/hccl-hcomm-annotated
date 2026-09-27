/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "server_socket_manager.h"

namespace hcomm {
HcclResult ServerSocketManager::ServerSocketStartListen(
    const Hccl::PortData& localPort, const Hccl::NicType nicType, const uint32_t devPhyId, uint32_t* port)
{
    if (nicType == Hccl::NicType::HOST_NIC_TYPE) {
        CHK_RET(HostSocketListen(localPort, devPhyId, port));
    } else if (nicType == Hccl::NicType::DEVICE_NIC_TYPE) {
        CHK_RET(DeviceSocketListen(localPort, devPhyId, port));
    } else {
        HCCL_ERROR("[ServerSocketManager][%s] illegal NicType[%s]", __func__, nicType.Describe().c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult
ServerSocketManager::HostSocketListen(const Hccl::PortData& localPort, const uint32_t devPhyId, uint32_t* port)
{
    std::lock_guard<std::mutex> lock(hostMutex_);
    uint32_t requestedPort = *port;
    if (hostServerSocketMap_.find(localPort) != hostServerSocketMap_.end()
        && hostServerSocketMap_[localPort].find(requestedPort) != hostServerSocketMap_[localPort].end()) {
        if (hostServerSocketMap_[localPort][requestedPort].second == UINT32_MAX) {
            HCCL_ERROR("[ServerSocketManager][%s]port listening count overflow UINT32_MAX", __func__);
            return HCCL_E_INTERNAL;
        }
        hostServerSocketMap_[localPort][requestedPort].second
            = hostServerSocketMap_[localPort][requestedPort].second + 1; // 计数+1
        HCCL_INFO("[ServerSocketManager][%s] reuse serverSocket", __func__);
        return HCCL_SUCCESS;
    }

    if (socketMgrCompat_ == nullptr) {
        EXCEPTION_CATCH(socketMgrCompat_ = std::make_unique<Hccl::SocketManager>(), return HCCL_E_INTERNAL);
    }
    bool isListen = socketMgrCompat_->CheckServerPortListening(localPort, requestedPort);

    Hccl::SocketHandle socketHandle{};
    EXCEPTION_CATCH(
        socketHandle = Hccl::HostSocketHandleManager::GetInstance().Create(devPhyId, localPort.GetAddr()),
        return HCCL_E_PARA);

    std::unique_ptr<Hccl::Socket> serverSocket{};
    EXCEPTION_CATCH(
        serverSocket = std::make_unique<Hccl::Socket>(
            socketHandle, localPort.GetAddr(), requestedPort, localPort.GetAddr(), "server", Hccl::SocketRole::SERVER,
            Hccl::NicType::HOST_NIC_TYPE),
        return HCCL_E_PARA); // 端口号可能冲突，需要SE做决定
    HCCL_INFO("[ServerSocketManager][%s] listen_socket_info[%s]", __func__, serverSocket->Describe().c_str());
    uint32_t actualPort = requestedPort;
    CHK_RET(ListenSocketIfNeeded(serverSocket.get(), isListen, actualPort, port));

    hostServerSocketMap_[localPort][actualPort] = std::make_pair(std::move(serverSocket), 1);

    return HCCL_SUCCESS;
}

HcclResult
ServerSocketManager::DeviceSocketListen(const Hccl::PortData& localPort, const uint32_t devPhyId, uint32_t* port)
{
    std::lock_guard<std::mutex> lock(deviceMutex_);
    uint32_t requestedPort = *port;
    if (deviceServerSocketMap_.find(localPort) != deviceServerSocketMap_.end()
        && deviceServerSocketMap_[localPort].find(requestedPort) != deviceServerSocketMap_[localPort].end()) {
        if (deviceServerSocketMap_[localPort][requestedPort].second == UINT32_MAX) {
            HCCL_ERROR("[ServerSocketManager][%s]port listening count overflow UINT32_MAX", __func__);
            return HCCL_E_INTERNAL;
        }
        deviceServerSocketMap_[localPort][requestedPort].second
            = deviceServerSocketMap_[localPort][requestedPort].second + 1; // 计数+1
        HCCL_INFO("[ServerSocketManager][%s] reuse serverSocket", __func__);
        return HCCL_SUCCESS;
    }

    if (socketMgrCompat_ == nullptr) {
        EXCEPTION_CATCH(socketMgrCompat_ = std::make_unique<Hccl::SocketManager>(), return HCCL_E_INTERNAL);
    }
    // 查询socketMgrCompat_，如果查询到已有serversocket，？new一个空壳，放进map里面，计数+1
    bool isListen = socketMgrCompat_->CheckServerPortListening(localPort, requestedPort);

    Hccl::SocketHandle socketHandle{};
    EXCEPTION_CATCH(
        socketHandle = Hccl::SocketHandleManager::GetInstance().Create(devPhyId, localPort), return HCCL_E_PARA);

    std::unique_ptr<Hccl::Socket> serverSocket;
    EXCEPTION_CATCH(
        serverSocket = std::make_unique<Hccl::Socket>(
            socketHandle, localPort.GetAddr(), requestedPort, localPort.GetAddr(), "server", Hccl::SocketRole::SERVER,
            Hccl::NicType::DEVICE_NIC_TYPE),
        return HCCL_E_PARA); // 端口号可能冲突，需要SE做决定
    HCCL_INFO("[ServerSocketManager][%s] listen_socket_info[%s]", __func__, serverSocket->Describe().c_str());
    uint32_t actualPort = requestedPort;
    CHK_RET(ListenSocketIfNeeded(serverSocket.get(), isListen, actualPort, port));
    deviceServerSocketMap_[localPort][actualPort] = std::make_pair(std::move(serverSocket), 1);

    return HCCL_SUCCESS;
}

HcclResult ServerSocketManager::ServerSocketStopListen(
    const Hccl::PortData& localPort, const Hccl::NicType nicType, const uint32_t port)
{
    if (nicType == Hccl::NicType::DEVICE_NIC_TYPE) {
        CHK_RET(DeviceSocketStopListen(localPort, port));
    } else if (nicType == Hccl::NicType::HOST_NIC_TYPE) {
        CHK_RET(HostSocketStopListen(localPort, port));
    } else {
        HCCL_ERROR("[ServerSocketManager][%s] illegal NicType[%s]", __func__, nicType.Describe().c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult ServerSocketManager::DeviceSocketStopListen(const Hccl::PortData& localPort, const uint32_t port)
{
    std::lock_guard<std::mutex> lock(deviceMutex_);
    if (deviceServerSocketMap_.find(localPort) != deviceServerSocketMap_.end()
        && deviceServerSocketMap_[localPort].find(port) != deviceServerSocketMap_[localPort].end()) {
        if (deviceServerSocketMap_[localPort][port].second == 0) {
            HCCL_ERROR("[ServerSocketManager][%s]port[%u] listening count already zero", __func__, port);
            return HCCL_E_INTERNAL;
        }
        deviceServerSocketMap_[localPort][port].second = deviceServerSocketMap_[localPort][port].second - 1; // 计数-1
        if (deviceServerSocketMap_[localPort][port].second == 0) {
            deviceServerSocketMap_[localPort].erase(port);
            CHK_RET(DeInitCompatServerIfListening(localPort, port));
        }
        if (deviceServerSocketMap_[localPort].empty()) {
            deviceServerSocketMap_.erase(localPort);
        }
        return HCCL_SUCCESS;
    }
    HCCL_ERROR(
        "[ServerSocketManager][%s] Can not stop listen cause {PortData[%s], port[%u]} is Not Listening", __func__,
        localPort.Describe().c_str(), port);
    return HCCL_E_NOT_FOUND;
}

HcclResult ServerSocketManager::HostSocketStopListen(const Hccl::PortData& localPort, const uint32_t port)
{
    std::lock_guard<std::mutex> lock(hostMutex_);
    if (hostServerSocketMap_.find(localPort) != hostServerSocketMap_.end()
        && hostServerSocketMap_[localPort].find(port) != hostServerSocketMap_[localPort].end()) {
        if (hostServerSocketMap_[localPort][port].second == 0) {
            HCCL_ERROR("[ServerSocketManager][%s]port[%u] listening count already zero", __func__, port);
            return HCCL_E_INTERNAL;
        }
        hostServerSocketMap_[localPort][port].second = hostServerSocketMap_[localPort][port].second - 1; // 计数-1
        if (hostServerSocketMap_[localPort][port].second == 0) {
            hostServerSocketMap_[localPort].erase(port);
            CHK_RET(DeInitCompatServerIfListening(localPort, port));
        }
        if (hostServerSocketMap_[localPort].empty()) {
            hostServerSocketMap_.erase(localPort);
        }
        return HCCL_SUCCESS;
    }
    HCCL_ERROR(
        "[ServerSocketManager][%s] Can not stop listen cause {PortData[%s], port[%u]} is Not Listening", __func__,
        localPort.Describe().c_str(), port);
    return HCCL_E_NOT_FOUND;
}

HcclResult ServerSocketManager::ListenSocketIfNeeded(
    Hccl::Socket* serverSocket, bool isListen, uint32_t& actualPort, uint32_t* outPort) const
{
    // port=0 表示自动分配端口，必须调 Listen(actualPort) 获取实际端口，不受 isListen 影响
    if (actualPort == 0) {
        EXCEPTION_CATCH(serverSocket->Listen(actualPort), return HCCL_E_INTERNAL);
        HCCL_INFO("[ServerSocketManager][%s] allocated port[%u]", __func__, actualPort);
        *outPort = actualPort;
        return HCCL_SUCCESS;
    }
    // 指定端口且已有监听则跳过，避免重复 Listen
    if (isListen) {
        HCCL_INFO("[ServerSocketManager][%s] port[%u] has been listen", __func__, actualPort);
        return HCCL_SUCCESS;
    }
    EXCEPTION_CATCH(serverSocket->Listen(), return HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ServerSocketManager::DeInitCompatServerIfListening(const Hccl::PortData& localPort, const uint32_t port)
{
    if (socketMgrCompat_ == nullptr) {
        EXCEPTION_CATCH(socketMgrCompat_ = std::make_unique<Hccl::SocketManager>(), return HCCL_E_INTERNAL);
    }
    bool isListen = socketMgrCompat_->CheckServerPortListening(localPort, port);
    if (isListen) {
        Hccl::PortData portDataCopy(
            localPort.GetRankId(), localPort.GetType(), localPort.GetProto(), localPort.GetId(), localPort.GetAddr());
        socketMgrCompat_->ServerDeInit(portDataCopy);
    }
    return HCCL_SUCCESS;
}

void ServerSocketManager::DeInitDeviceSockets(u32 devPhyId)
{
    std::lock_guard<std::mutex> lock(deviceMutex_);
    for (auto it = deviceServerSocketMap_.begin(); it != deviceServerSocketMap_.end();) {
        if (static_cast<uint32_t>(it->first.GetRankId()) == devPhyId) {
            for (auto& portEntry : it->second) {
                if (portEntry.second.first != nullptr) {
                    portEntry.second.first->Destroy();
                    portEntry.second.first.reset();
                }
            }
            it = deviceServerSocketMap_.erase(it);
        } else {
            ++it;
        }
    }
}

void ServerSocketManager::DeInitHostSockets(u32 devPhyId)
{
    std::lock_guard<std::mutex> lock(hostMutex_);
    for (auto it = hostServerSocketMap_.begin(); it != hostServerSocketMap_.end();) {
        if (static_cast<uint32_t>(it->first.GetRankId()) == devPhyId) {
            for (auto& portEntry : it->second) {
                if (portEntry.second.first != nullptr) {
                    portEntry.second.first->Destroy();
                    portEntry.second.first.reset();
                }
            }
            it = hostServerSocketMap_.erase(it);
        } else {
            ++it;
        }
    }
}

void ServerSocketManager::DeInit(u32 devPhyId)
{
    HCCL_INFO("[ServerSocketManager][%s] DeInit[%u]", __func__, devPhyId);
    DeInitDeviceSockets(devPhyId);
    DeInitHostSockets(devPhyId);
}

} // namespace hcomm
