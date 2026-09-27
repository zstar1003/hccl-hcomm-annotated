/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_roce_server_socket_context.h"
#include "log.h"

namespace hcomm {

// ---- AicpuTsRoceServerSocketContext（HcclSocket 自管理路径，从 AicpuTsRoceEndpoint 迁入）----

AicpuTsRoceServerSocketContext::AicpuTsRoceServerSocketContext(HcclNetDev netDev, uint32_t netDevRefPhyId)
    : netDev_(netDev),
      netDevRefPhyId_(netDevRefPhyId)
{}

AicpuTsRoceServerSocketContext::~AicpuTsRoceServerSocketContext() { ReleaseListenSocketRefs(); }

std::mutex& AicpuTsRoceServerSocketContext::ListenSocketMapMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<SocketMapKey, AicpuTsListenSocketSlot, SocketMapKeyHash>&
AicpuTsRoceServerSocketContext::GetServerSocketMap()
{
    static std::unordered_map<SocketMapKey, AicpuTsListenSocketSlot, SocketMapKeyHash> serverSocketMap;
    return serverSocketMap;
}

HcclResult AicpuTsRoceServerSocketContext::ServerSocketListen(const uint32_t port)
{
    const uint32_t listenPort = (port != 0U) ? port : kDefaultAicpuTsRocePort;
    const SocketMapKey key{netDevRefPhyId_, listenPort};
    std::lock_guard<std::mutex> lk(ListenSocketMapMutex());
    if (ReuseListenSocketIfExist(key, "reuse serverSocket")) {
        return HCCL_SUCCESS;
    }

    std::shared_ptr<hccl::HcclSocket> newServerSocket = nullptr;
    EXCEPTION_CATCH(
        newServerSocket = std::make_shared<hccl::HcclSocket>(static_cast<HcclNetDevCtx>(netDev_), listenPort),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(newServerSocket);

    HcclResult ret = newServerSocket->Init();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuTsRoceServerSocketContext][%s] HcclSocket Init failed, ret[%d]", __func__, ret);
        return ret;
    }

    ret = newServerSocket->Listen();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuTsRoceServerSocketContext][%s] HcclSocket Listen failed, ret[%d]", __func__, ret);
        return ret;
    }

    auto& serverSocketMap = GetServerSocketMap();
    serverSocketMap[key] = AicpuTsListenSocketSlot{newServerSocket, 1U};
    listenRefKeys_.push_back(key);
    hasListenSocketRef_ = true;
    HCCL_INFO(
        "[AicpuTsRoceServerSocketContext][%s] listen on key[dev=%u,port=%u] success", __func__, key.devicePhyId,
        key.port);
    return HCCL_SUCCESS;
}

bool AicpuTsRoceServerSocketContext::ReuseListenSocketIfExist(const SocketMapKey& key, const char* logPrefix)
{
    auto& serverSocketMap = GetServerSocketMap();
    auto it = serverSocketMap.find(key);
    if (it == serverSocketMap.end() || it->second.socket == nullptr) {
        return false;
    }
    it->second.refCount++;
    listenRefKeys_.push_back(key);
    hasListenSocketRef_ = true;
    HCCL_INFO(
        "[AicpuTsRoceServerSocketContext][%s] %s key[dev=%u,port=%u], ref[%u]", __func__, logPrefix, key.devicePhyId,
        key.port, it->second.refCount);
    return true;
}

void AicpuTsRoceServerSocketContext::ReleaseListenSocketRefs()
{
    std::lock_guard<std::mutex> lk(ListenSocketMapMutex());
    HCCL_INFO(
        "[AicpuTsRoceServerSocketContext][%s] netDevRefPhyId_[%u], listenRefKeys_.size[%zu]", __func__, netDevRefPhyId_,
        listenRefKeys_.size());

    std::vector<SocketMapKey> keys = std::move(listenRefKeys_);
    auto& sockMap = GetServerSocketMap();
    for (const auto& key : keys) {
        auto it = sockMap.find(key);
        if (it == sockMap.end()) {
            HCCL_INFO(
                "[AicpuTsRoceServerSocketContext][%s] key[dev=%u,port=%u] not found in sockMap", __func__,
                key.devicePhyId, key.port);
            continue;
        }
        HCCL_INFO(
            "[AicpuTsRoceServerSocketContext][%s] key[dev=%u,port=%u] refCount[%u] before decrement", __func__,
            key.devicePhyId, key.port, it->second.refCount);
        if (it->second.refCount > 0U) {
            it->second.refCount--;
        }
        HCCL_INFO(
            "[AicpuTsRoceServerSocketContext][%s] key[dev=%u,port=%u] refCount[%u] after decrement, socket shared_ptr "
            "use_count[%ld]",
            __func__, key.devicePhyId, key.port, it->second.refCount, it->second.socket.use_count());
        if (it->second.refCount == 0U) {
            HCCL_INFO(
                "[AicpuTsRoceServerSocketContext][%s] erasing key[dev=%u,port=%u] from sockMap", __func__,
                key.devicePhyId, key.port);
            (void)sockMap.erase(it);
        }
    }
}

HcclResult AicpuTsRoceServerSocketContext::AddListenSocketWhiteList(
    uint32_t port, const std::vector<SocketWlistInfo>& wlistInfos) const
{
    if (wlistInfos.empty()) {
        HCCL_ERROR("[AicpuTsRoceServerSocketContext][%s] empty whitelist", __func__);
        return HCCL_E_PARA;
    }
    std::lock_guard<std::mutex> lk(ListenSocketMapMutex());
    auto& sockMap = GetServerSocketMap();
    const uint32_t listenPort = (port != 0U) ? port : kDefaultAicpuTsRocePort;
    const SocketMapKey key{netDevRefPhyId_, listenPort};
    auto it = sockMap.find(key);
    if (it == sockMap.end() || it->second.socket == nullptr) {
        HCCL_ERROR(
            "[AicpuTsRoceServerSocketContext][%s] no listen socket for key[dev=%u,port=%u]", __func__, key.devicePhyId,
            key.port);
        return HCCL_E_NOT_FOUND;
    }
    std::vector<SocketWlistInfo> mutableCopy = wlistInfos;
    return it->second.socket->AddWhiteList(mutableCopy);
}

HcclResult AicpuTsRoceServerSocketContext::GetSocket(
    [[maybe_unused]] uint32_t port, const std::string& tag, std::shared_ptr<hccl::HcclSocket>& outConnected) const
{
    EXCEPTION_CATCH(
        (outConnected = std::make_shared<hccl::HcclSocket>(
             tag, static_cast<HcclNetDevCtx>(netDev_), hccl::HcclIpAddress(), 0,
             hccl::HcclSocketRole::SOCKET_ROLE_SERVER)),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(outConnected);
    CHK_RET(outConnected->Init());

    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceServerSocketContext::AcceptDataSocket(
    uint32_t port, const std::string& tag, std::shared_ptr<hccl::HcclSocket>& outConnected,
    uint32_t acceptTimeoutMs) const
{
    std::lock_guard<std::mutex> lk(ListenSocketMapMutex());
    auto& map = GetServerSocketMap();
    const uint32_t listenPort = (port != 0U) ? port : kDefaultAicpuTsRocePort;
    const SocketMapKey key{netDevRefPhyId_, listenPort};
    auto it = map.find(key);
    if (it == map.end() || it->second.socket == nullptr) {
        HCCL_ERROR(
            "[AicpuTsRoceServerSocketContext][%s] no listen socket for key[dev=%u,port=%u]", __func__, key.devicePhyId,
            key.port);
        return HCCL_E_NOT_FOUND;
    }
    return it->second.socket->Accept(tag, outConnected, acceptTimeoutMs);
}

} // namespace hcomm
