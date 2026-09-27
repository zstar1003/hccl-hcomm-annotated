/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "server_socket_mgr.h"
#include "hccl_common.h"
#include "exception_handler.h"
#include "orion_adpt_utils.h"
#include "socket_handle_manager.h"

namespace hcomm {

HcclResult ServerSocketMgr::ListenStart(const uint32_t devPhyId, const CommAddr& commAddr, const Hccl::NicType nicType)
{
    if (nicType != Hccl::NicType::DEVICE_NIC_TYPE && nicType != Hccl::NicType::HOST_NIC_TYPE) {
        HCCL_ERROR("[%s] nicType[%d] is not supported", __func__, static_cast<int>(nicType)); // 枚举用转换吗？
        return HCCL_E_PARA;
    }

    auto& socketMgr = ServerSocketMgr::GetInstance(devPhyId);

    CHK_RET(socketMgr.ListenStart_(commAddr, nicType));

    return HcclResult::HCCL_SUCCESS;
}

ServerSocketMgr& ServerSocketMgr::GetInstance(const uint32_t devicePhyId)
{
    static ServerSocketMgr socketMgr[MAX_MODULE_DEVICE_NUM + 1];

    uint32_t devPhyId = devicePhyId;
    if (devPhyId >= MAX_MODULE_DEVICE_NUM + 1) {
        HCCL_WARNING(
            "[%s] Invalid devicePhyId: %u, max allowed: %u, using default index: %u", __func__, devicePhyId,
            MAX_MODULE_DEVICE_NUM, MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM;
    }

    socketMgr[devPhyId].devPhyId_ = devPhyId;
    return socketMgr[devPhyId];
}

HcclResult ServerSocketMgr::ListenStart_(const CommAddr& commAddr, const Hccl::NicType nicType)
{
    std::lock_guard<std::mutex> lock(innerMutex_);
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(commAddr, ipAddr));

    if (nicType == Hccl::NicType::DEVICE_NIC_TYPE) {
        auto ipIter = deviceServerSocketMap_.find(ipAddr);
        if (ipIter != deviceServerSocketMap_.end()) {
            HCCL_INFO("[ServerSocketMgr][%s] device server socket already created.", __func__);
            return HcclResult::HCCL_SUCCESS;
        }
    } else {
        auto ipIter = hostServerSocketMap_.find(ipAddr);
        if (ipIter != hostServerSocketMap_.end()) {
            HCCL_INFO("[ServerSocketMgr][%s] host server socket already created.", __func__);
            return HcclResult::HCCL_SUCCESS;
        }
    }

    EXCEPTION_HANDLE_BEGIN
    const Hccl::DevNetPortType portType = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB); // 不能写死
    // todo: 暂时使用devPhyId构造rankId，id存疑？
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(devPhyId_), portType, 0, ipAddr);

    HCCL_INFO(
        "[ServerSocketMgr][%s] get socket handle, devPhyId[%u] locAddr[%s].", __func__, devPhyId_,
        ipAddr.Describe().c_str());

    Hccl::SocketHandle socketHandle = Hccl::SocketHandleManager::GetInstance().Create(devPhyId_, localPort);

    std::unique_ptr<Hccl::Socket> serverSocket = nullptr;
    constexpr uint32_t listenPort = 60001; // 端口号如何处理，可能冲突
    const std::string tag = "server";
    constexpr Hccl::SocketRole role = Hccl::SocketRole::SERVER;

    HCCL_INFO(
        "[ServerSocketMgr][%s] create server socket, "
        "locAddr[%s] rmtAddr[%s] tag[%s] role[%s].",
        __func__, ipAddr.Describe().c_str(), ipAddr.Describe().c_str(), tag.c_str(), role.Describe().c_str());

    serverSocket.reset(new (std::nothrow) Hccl::Socket(socketHandle, ipAddr, listenPort, ipAddr, tag, role, nicType));
    CHK_PTR_NULL(serverSocket);
    serverSocket->Listen();

    if (nicType == Hccl::NicType::DEVICE_NIC_TYPE) {
        deviceServerSocketMap_[ipAddr] = std::move(serverSocket); // IP校验？
    } else {
        hostServerSocketMap_[ipAddr] = std::move(serverSocket); // IP校验？
    }
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

void ServerSocketMgr::DeInit(u32 devPhyId)
{
    HCCL_INFO("[ServerSocketMgr][%s] DeInit[%u]", __func__, devPhyId);
    auto& inst = GetInstance(devPhyId);
    std::lock_guard<std::mutex> lock(inst.innerMutex_);
    for (auto& it : inst.deviceServerSocketMap_) {
        if (it.second != nullptr) {
            it.second->Destroy();
            it.second.reset();
        }
    }
    inst.deviceServerSocketMap_.clear();

    for (auto& it : inst.hostServerSocketMap_) {
        if (it.second != nullptr) {
            it.second->Destroy();
            it.second.reset();
        }
    }
    inst.hostServerSocketMap_.clear();
}

}; // namespace hcomm
