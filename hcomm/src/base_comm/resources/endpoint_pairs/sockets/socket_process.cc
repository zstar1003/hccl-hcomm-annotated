/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "socket_process.h"
#include "socket_config.h"
#include "socket.h"
#include "ip_address.h"
#include "exception_handler.h"
#include "adapter_rts_common.h"
#include "endpoint.h"

using namespace std;

namespace hcomm {

SocketProcess& SocketProcess::GetInstance(s32 deviceLogicId)
{
    static SocketProcess socketProcess[MAX_MODULE_DEVICE_NUM];
    if (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[SocketProcess][%s] invalid deviceLogicId: %d", __func__, deviceLogicId);
        return socketProcess[0];
    }
    return socketProcess[deviceLogicId];
}

SocketProcess::~SocketProcess()
{
    unique_lock<std::mutex> lock(mutex_);
    isInit_ = false;
    for (auto& socketItem : serverSocketMap_) {
        if (socketItem.second != nullptr) {
            socketItem.second.get()->Destroy();
        }
    }
    serverSocketMap_.clear();

    for (auto& item : tag2socketMap_) {
        if (item.second.first != nullptr) {
            SocketMgr::GetInstance(devicePhyId_).DestroySocket(item.second.first);
        }
    }
    tag2socketMap_.clear();
    socket2TagMap_.clear();
}

HcclResult SocketProcess::DestroySocketHandle(SocketHandle socketHandle)
{
    Hccl::Socket* socket = static_cast<Hccl::Socket*>(socketHandle);
    if (socket == nullptr) {
        HCCL_WARNING("[SocketProcess][%s] socket[%p] is nullptr, please check", __func__, static_cast<void*>(socket));
        return HCCL_E_PARA;
    }

    unique_lock<std::mutex> lock(mutex_);
    auto socket2TagIter = socket2TagMap_.find(socket);
    if (socket2TagIter == socket2TagMap_.end()) {
        HCCL_WARNING("[SocketProcess][%s] socket[%p] not found, please check", __func__, static_cast<void*>(socket));
        return HCCL_E_NOT_FOUND;
    }

    string socketTag = socket2TagIter->second;
    auto tag2socketIter = tag2socketMap_.find(socketTag);
    if (tag2socketIter == tag2socketMap_.end()) {
        HCCL_WARNING("[SocketProcess][%s] socketTag[%s] not found, please check", __func__, socketTag.c_str());
        return HCCL_E_NOT_FOUND;
    }

    if (tag2socketIter->second.second > 0) {
        tag2socketIter->second.second--;
        HCCL_INFO(
            "[SocketProcess][%s] socket with tag[%s] refCnt: %u", __func__, socketTag.c_str(),
            tag2socketIter->second.second);
        return HCCL_SUCCESS;
    }

    HCCL_DEBUG("[SocketProcess][%s] destroy socket with tag[%s]", __func__, socketTag.c_str());
    Hccl::Socket* rawSocket = tag2socketIter->second.first;
    tag2socketMap_.erase(tag2socketIter);
    socket2TagMap_.erase(socket2TagIter);
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).DeleteWhiteList(rawSocket));
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).DestroySocket(rawSocket));

    return HCCL_SUCCESS;
}

HcclResult SocketProcess::GetSocket(SocketDesc* socketDesc, SocketHandle& socketHandle)
{
    CHK_PTR_NULL(socketDesc);
    CHK_RET(Init());
    HCCL_RUN_INFO(
        "[GetSocket][%s] initialized. devicePhyId: %u, this: %p", __func__, devicePhyId_, static_cast<void*>(this));

    if (!isInit_) {
        HCCL_ERROR("[SocketProcess][%s] SocketProcess not initialized, device may be destroyed", __func__);
        return HCCL_E_INTERNAL;
    }

    Hccl::IpAddress localIpaddr{};
    CHK_RET(CommAddrToIpAddress(socketDesc->localEndpoint.commAddr, localIpaddr));
    Hccl::IpAddress remoteIpaddr{};
    CHK_RET(CommAddrToIpAddress(socketDesc->remoteEndpoint.commAddr, remoteIpaddr));

    string socketTag
        = string(socketDesc->tag) + "_" + localIpaddr.GetIpStr().c_str() + "_" + remoteIpaddr.GetIpStr().c_str();
    HCCL_INFO("[SocketProcess][%s] socket with tag[%s].", __func__, socketTag.c_str());
    unique_lock<std::mutex> lock(mutex_);
    if (tag2socketMap_.find(socketTag) == tag2socketMap_.end()) {
        CHK_RET(BuildSocket(socketDesc, socketTag));
    } else {
        tag2socketMap_[socketTag].second++;
        HCCL_INFO(
            "[SocketProcess][%s] socket with tag[%s] already exists, num: %u.", __func__, socketTag.c_str(),
            tag2socketMap_[socketTag].second);
    }

    socketHandle = static_cast<SocketHandle>(tag2socketMap_[socketTag].first);
    HCCL_INFO("[SocketProcess][%s] socketHandle = %p", __func__, socketHandle);
    return HCCL_SUCCESS;
}

HcclResult SocketProcess::PutSocket(SocketHandle& socketHandle)
{
    CHK_PTR_NULL(socketHandle);
    Hccl::Socket* socket = static_cast<Hccl::Socket*>(socketHandle);
    SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket);
    return HCCL_SUCCESS;
}

HcclResult SocketProcess::GetStatus(SocketHandle socketHandle, SocketStates& socketStatus)
{
    Hccl::Socket* socket = static_cast<Hccl::Socket*>(socketHandle);
    unique_lock<std::mutex> lock(mutex_);
    if (socket == nullptr || socket2TagMap_.find(socket) == socket2TagMap_.end()) {
        HCCL_ERROR("[SocketProcess][%s] socket is nullptr or not found, please check", __func__);
        return HCCL_E_PARA;
    }
    lock.unlock();

    Hccl::SocketStatus status = socket->GetAsyncStatus();
    if (status == Hccl::SocketStatus::OK) {
        socketStatus = SocketStates::SOCKET_OK;
    } else if (status == Hccl::SocketStatus::TIMEOUT) {
        socketStatus = SocketStates::SOCKET_TIMEOUT;
    } else {
        socketStatus = SocketStates::SOCKET_CONNECTING;
    }

    return HCCL_SUCCESS;
}

HcclResult SocketProcess::SendNoBlock(SocketHandle socketHandle, void* sendbuffer, u64 sendSize, u64*& sentSize)
{
    Hccl::Socket* socket = static_cast<Hccl::Socket*>(socketHandle);
    unique_lock<std::mutex> lock(mutex_);
    if (socket == nullptr || socket2TagMap_.find(socket) == socket2TagMap_.end()) {
        HCCL_ERROR("[SocketProcess][%s] socket is nullptr or not found, please check", __func__);
        return HCCL_E_PARA;
    }
    lock.unlock();
    if (sentSize == nullptr || sendbuffer == nullptr) {
        HCCL_ERROR("[SocketProcess][%s] sentSize is nullptr or sendbuffer is nullptr, please check", __func__);
        return HCCL_E_PARA;
    }

    HcclResult ret = socket->ISendWithHeart(reinterpret_cast<u8*>(sendbuffer), sendSize, *sentSize);
    if (ret == HCCL_E_AGAIN) {
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[SocketProcess::%s] except send size[%llu]. actual [%zu] bytes sent.", __func__, sendSize, *sentSize);

    return ret;
}

HcclResult SocketProcess::RecvNoBlock(SocketHandle socketHandle, void* recvBuffer, u64 recvSize, u64*& recvedSize)
{
    Hccl::Socket* socket = static_cast<Hccl::Socket*>(socketHandle);
    unique_lock<std::mutex> lock(mutex_);
    if (socket == nullptr || socket2TagMap_.find(socket) == socket2TagMap_.end()) {
        HCCL_ERROR("[SocketProcess][%s] socket is nullptr or not found, please check", __func__);
        return HCCL_E_PARA;
    }
    lock.unlock();
    if (recvBuffer == nullptr || recvedSize == nullptr) {
        HCCL_ERROR("[SocketProcess][%s] recvBuffer is nullptr or recvedSize is nullptr, please check", __func__);
        return HCCL_E_PARA;
    }

    HcclResult ret = socket->IRecvWithHeart(reinterpret_cast<u8*>(recvBuffer), recvSize, *recvedSize);
    if (ret == HCCL_E_AGAIN) {
        return HCCL_SUCCESS; // 未收到数据，非错误
    }
    HCCL_DEBUG(
        "[SocketProcess::%s] except recv size[%llu]. actual [%zu] bytes received.", __func__, recvSize, *recvedSize);

    return ret;
}

HcclResult SocketProcess::Init()
{
    unique_lock<std::mutex> lock(mutex_);
    if (isInit_.load(std::memory_order_acquire)) {
        return HCCL_SUCCESS;
    }

    uint32_t deviceCount = 0;
    HcclResult ret = hrtGetDeviceCount(&deviceCount);
    if (ret != HCCL_SUCCESS || deviceCount == 0) {
        devicePhyId_ = 0;
        isInit_.store(true, std::memory_order_release);
        HCCL_RUN_INFO(
            "[SocketProcess][%s] host resource initialized. get device count ret[%d], count[%u], "
            "devicePhyId: %u, this: %p",
            __func__, ret, deviceCount, devicePhyId_, static_cast<void*>(this));
        return HCCL_SUCCESS;
    }

    s32 devLogicId = 0;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));

    isInit_.store(true, std::memory_order_release);
    HCCL_RUN_INFO(
        "[SocketProcess][%s] initialized successfully. deviceLogicId: %d, devicePhyId: %u, this: %p", __func__,
        devLogicId, devicePhyId_, static_cast<void*>(this));

    return HCCL_SUCCESS;
}

Hccl::SocketRole SocketProcess::ConvertToHcclSocketRole(HcommSocketRole& hcommRole) const
{
    switch (hcommRole) {
        case HCOMM_SOCKET_ROLE_CLIENT:
            return Hccl::SocketRole::CLIENT;
        case HCOMM_SOCKET_ROLE_SERVER:
            return Hccl::SocketRole::SERVER;
        case HCOMM_SOCKET_ROLE_RESERVED:
        default:
            HCCL_WARNING("[Convert] Invalid HcommSocketRole: %d, defaulting to CLIENT", hcommRole);
            return Hccl::SocketRole::CLIENT;
    }
}

HcclResult SocketProcess::BuildSocket(SocketDesc* socketDesc, const std::string& socketTag)
{
    if (tag2socketMap_.find(socketTag) != tag2socketMap_.end()) {
        return HCCL_SUCCESS;
    }

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(socketDesc->localEndpoint, socketDesc->remoteEndpoint, linkData));
    HCCL_INFO("[SocketProcess][%s] built linkData: %s", __func__, linkData.Describe().c_str());
    Hccl::SocketConfig socketConfig = Hccl::SocketConfig(
        linkData, string(socketDesc->tag), ConvertToHcclSocketRole(socketDesc->role), socketDesc->listenPort);
    auto localListenPair = std::make_pair(socketConfig.link.GetLocalPort(), socketConfig.listeningPort);

    Hccl::IpAddress ipaddr{};
    CHK_RET(CommAddrToIpAddress(socketDesc->localEndpoint.commAddr, ipaddr));
    if (socketDesc->role == HCOMM_SOCKET_ROLE_SERVER
        && serverSocketMap_.find(localListenPair) == serverSocketMap_.end()) {
        Hccl::SocketHandle serverSocketHandle
            = Hccl::SocketHandleManager::GetInstance().Get(devicePhyId_, localListenPair.first);
        if (serverSocketHandle == nullptr) {
            serverSocketHandle = Hccl::SocketHandleManager::GetInstance().Create(devicePhyId_, localListenPair.first);
        }
        EXCEPTION_CATCH(
            serverSocketMap_[localListenPair] = std::make_unique<Hccl::Socket>(
                serverSocketHandle, ipaddr, localListenPair.second, ipaddr, socketDesc->tag, Hccl::SocketRole::SERVER,
                Hccl::NicType::DEVICE_NIC_TYPE),
            return HCCL_E_PARA);
        HCCL_INFO("[%s] listen_socket_info[%s]", __func__, serverSocketMap_[localListenPair].get()->Describe().c_str());
        EXCEPTION_CATCH(serverSocketMap_[localListenPair].get()->Listen(), return HCCL_E_INTERNAL);
    }
    HCCL_INFO("[SocketProcess][%s] ip[%s] has been listening.", __func__, ipaddr.GetIpStr().c_str());

    Hccl::Socket* socket = nullptr;
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket));
    tag2socketMap_[socketTag].first = socket;
    tag2socketMap_[socketTag].second = 0;
    socket2TagMap_[socket] = socketTag;

    return HCCL_SUCCESS;
}

} // namespace hcomm
