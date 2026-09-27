/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <algorithm>

#include "socket_mgr.h"
#include "hcomm_adapter_runtime.h"
#include "../channels/channel.h"
#include "orion_adpt_utils.h"
#include "host_socket_handle_manager.h"
#include "exception_handler.h"
#include "adapter_rts.h"
#include "env_config/env_config_v2.h"

namespace hcomm {

constexpr uint32_t TempServerListenPort = 60001; // 临时固定监听端口，用于功能验证
constexpr uint32_t kHostResourceId = 0U;

s32 g_linkTimeout = 0;
inline s32 EnvLinkTimeoutGet()
{
    g_linkTimeout
        = g_linkTimeout != 0 ? g_linkTimeout : Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut();
    return g_linkTimeout;
}

SocketMgr& SocketMgr::GetInstance(s32 phyId)
{
    static SocketMgr instances[MAX_MODULE_DEVICE_NUM]; // C++11 保证线程安全
    if (static_cast<u32>(phyId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[SocketMgr] devicePhyId >= MAX_MODULE_DEVICE_NUM, devicePhyId=%d, MAX_MODULE_DEVICE_NUM=%d", phyId,
            MAX_MODULE_DEVICE_NUM);
        return instances[0];
    }
    instances[phyId].devicePhyId_ = phyId;
    return instances[phyId];
}

HcclResult SocketMgr::Init()
{
    uint32_t runtimeDevicePhyId = 0;
    bool noDevice = false;
    CHK_RET(ResolveRuntimeDevicePhyId(runtimeDevicePhyId, noDevice));
    if (isLoaded_ && isHostOnlyInit_ == noDevice) {
        return HCCL_SUCCESS;
    }
    // 覆盖 GetInstance 和 EndpointPair 直接构造 SocketMgr 两种路径，保持 devicePhyId_ 与 runtime 当前设备一致。
    devicePhyId_ = noDevice ? kHostResourceId : runtimeDevicePhyId;
    isLoaded_ = true;
    isHostOnlyInit_ = noDevice;
    serverListenPort_ = TempServerListenPort;
    HCCL_INFO(
        "[SocketMgr][%s] init socket mgr, noDevice[%d], runtimeDevicePhyId[%u], devicePhyId[%u].", __func__, noDevice,
        runtimeDevicePhyId, devicePhyId_);
    return HCCL_SUCCESS;
}

HcclResult SocketMgr::AddWhiteList(const Hccl::SocketConfig& socketConfig, const Hccl::SocketHandle& socketHandle)
{
    EXCEPTION_HANDLE_BEGIN

    // 1. 创建 wlistInfo 对象
    Hccl::RaSocketWhitelist wlistInfo{};
    ;
    wlistInfo.connLimit = 1;
    wlistInfo.remoteIp = socketConfig.link.GetRemoteAddr();
    wlistInfo.tag = socketConfig.GetHccpTag();
    handle2WhiteListMap_[socketHandle].push_back(wlistInfo);

    std::vector<Hccl::RaSocketWhitelist> wlistInfoVec;
    wlistInfoVec.clear();
    wlistInfoVec.push_back(wlistInfo);

    // 2. 加入白名单
    Hccl::HrtRaSocketWhiteListAdd(socketHandle, wlistInfoVec);

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult SocketMgr::GetSocketHandle(const Hccl::SocketConfig& socketConfig, Hccl::SocketHandle& socketHandle) const
{
    EXCEPTION_HANDLE_BEGIN

    // 加异常捕获
    auto localPort = socketConfig.link.GetLocalPort();
    if (localPort.GetType() == Hccl::PortDeploymentType::DEV_NET) {
        socketHandle = Hccl::SocketHandleManager::GetInstance().Get(devicePhyId_, localPort);
        if (socketHandle == nullptr) {
            socketHandle = Hccl::SocketHandleManager::GetInstance().Create(devicePhyId_, localPort);
        }
    } else if (localPort.GetType() == Hccl::PortDeploymentType::HOST_NET) {
        socketHandle = Hccl::HostSocketHandleManager::GetInstance().Get(devicePhyId_, localPort.GetAddr());
        if (socketHandle == nullptr) {
            socketHandle = Hccl::HostSocketHandleManager::GetInstance().Create(devicePhyId_, localPort.GetAddr());
        }
    } else {
        HCCL_ERROR(
            "[SocketMgr] PortDeploymentType = %s, not support create socket.", localPort.GetType().Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    if (socketHandle == nullptr) {
        HCCL_ERROR(
            "[SocketMgr] socketHandle is nullptr, devicePhyId=%d, localPort[%s]", devicePhyId_,
            localPort.Describe().c_str());
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[SocketMgr][%s] socketHandle[%p] devicePhyId[%u] localPort[%s]", __func__, socketHandle, devicePhyId_,
        localPort.Describe().c_str());

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult SocketMgr::CreateSocket(const Hccl::SocketConfig& socketConfig, const Hccl::SocketHandle& socketHandle)
{
    EXCEPTION_HANDLE_BEGIN

    Hccl::IpAddress localIpAddress = socketConfig.link.GetLocalAddr();
    Hccl::IpAddress remoteIpAddress = socketConfig.link.GetRemoteAddr();
    Hccl::SocketRole socketRole = socketConfig.GetRole();
    std::string hccpSocketTag = socketConfig.GetHccpTag();
    serverListenPort_ = socketConfig.listeningPort; // serverListenPort_这个变量似乎没用

    std::unique_ptr<Hccl::Socket> tmpSocket = nullptr;
    if (socketConfig.link.GetType() == Hccl::PortDeploymentType::DEV_NET) {
        EXCEPTION_CATCH(
            tmpSocket = std::make_unique<Hccl::Socket>(
                socketHandle, localIpAddress, socketConfig.listeningPort, remoteIpAddress, hccpSocketTag, socketRole,
                Hccl::NicType::DEVICE_NIC_TYPE),
            return HCCL_E_PTR);
        HCCL_INFO("[SocketMgr][%s] client_socket_info[%s]", __func__, tmpSocket->Describe().c_str());
        tmpSocket->ConnectAsync();
    } else if (socketConfig.link.GetType() == Hccl::PortDeploymentType::HOST_NET) {
        EXCEPTION_CATCH(
            tmpSocket = std::make_unique<Hccl::Socket>(
                socketHandle, localIpAddress, socketConfig.listeningPort, remoteIpAddress, hccpSocketTag, socketRole,
                Hccl::NicType::HOST_NIC_TYPE),
            return HCCL_E_PTR);
        HCCL_INFO("[SocketMgr][%s] client_socket_info[%s]", __func__, tmpSocket->Describe().c_str());
        tmpSocket->Connect();
    } else {
        HCCL_ERROR(
            "[SocketMgr] PortDeploymentType = %s, not support create socket.",
            socketConfig.link.GetType().Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    socketMap_[socketConfig] = std::move(tmpSocket);
    socketInUseMap_[socketMap_[socketConfig].get()] = false;

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult SocketMgr::CreateSocketWithSocketHandle(const Hccl::SocketConfig& socketConfig)
{
    Hccl::SocketHandle socketHandle;
    CHK_RET(GetSocketHandle(socketConfig, socketHandle));
    CHK_RET(AddWhiteList(socketConfig, socketHandle));
    CHK_RET(CreateSocket(socketConfig, socketHandle));

    return HCCL_SUCCESS;
}

HcclResult SocketMgr::MakeSocketInUse(Hccl::Socket*& socket)
{
    if (socketInUseMap_.find(socket) != socketInUseMap_.end()) {
        socketInUseMap_[socket] = true;
    } else {
        HCCL_ERROR("[SocketMgr][%s] CreateSocket succeeded but socket not found in socketInUseMap", __func__);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult SocketMgr::GetNewSocket(const Hccl::SocketConfig& socketConfig, Hccl::Socket*& socket)
{
    CHK_RET(CreateSocketWithSocketHandle(socketConfig));

    // 再次查找
    std::unordered_map<Hccl::SocketConfig, std::unique_ptr<Hccl::Socket>>::iterator it = socketMap_.find(socketConfig);
    if (it == socketMap_.end()) {
        HCCL_ERROR("[SocketMgr][%s] CreateSocket succeeded but socket not found in socketMap", __func__);
        return HCCL_E_INTERNAL;
    }
    socket = it->second.get();
    return HCCL_SUCCESS;
}

HcclResult SocketMgr::GetSocket(const Hccl::SocketConfig& socketConfig, Hccl::Socket*& socket)
{
    std::unique_lock<std::mutex> lock(mutex_);
    CHK_RET(Init());
    // 1. 先查找
    std::unordered_map<Hccl::SocketConfig, std::unique_ptr<Hccl::Socket>>::iterator it = socketMap_.begin();

    for (; it != socketMap_.end(); ++it) {
        if (std::equal_to<Hccl::SocketConfig>{}(socketConfig, it->first)) {
            socket = it->second.get();
            break;
        }
    }
    if (it != socketMap_.end()) {
        if (socketConfig.hostNic2DeviceNicMode_ != 0) {
            HCCL_INFO(
                "[SocketMgr][%s] destroy a socket[%p] in hostNic2DeviceNicMode", __func__, static_cast<void*>(socket));
            socket->Destroy();
            socketMap_.erase(it);
            socketInUseMap_.erase(socket);
        } else {
            HCCL_INFO("[SocketMgr][%s] find a correct socket in map", __func__);
            auto timeoutPoint = std::chrono::steady_clock::now() + std::chrono::seconds(EnvLinkTimeoutGet())
                                - std::chrono::seconds(10);
            while (socketInUseMap_[socket] == true) {
                if (socketAvailableCv_.wait_until(lock, timeoutPoint) == std::cv_status::timeout) {
                    HCCL_ERROR("[SocketMgr][%s] Get Socket Time Out", __func__);
                    return HCCL_E_TIMEOUT;
                }
            }
            CHK_RET(MakeSocketInUse(socket));
            return HCCL_SUCCESS;
        }
    }

    // 2. 不存在则创建
    CHK_RET(GetNewSocket(socketConfig, socket));
    CHK_RET(MakeSocketInUse(socket));
    return HCCL_SUCCESS;
}

// 仅通信域管理层的host网卡使用，后续需归一到通信域管理层的socket管理模块
HcclResult SocketMgr::GetHostSocket(const Hccl::SocketConfig& socketConfig, Hccl::Socket*& socket)
{
    CHK_RET(Init());
    // 1. 先查找
    auto it = socketMap_.find(socketConfig);
    if (it != socketMap_.end()) {
        if (socketConfig.hostNic2DeviceNicMode_ != 0) {
            socket = it->second.get();
            HCCL_INFO(
                "[SocketMgr][%s] destroy a socket[%p] in hostNic2DeviceNicMode", __func__, static_cast<void*>(socket));
            socket->Destroy();
            socketMap_.erase(it);
            socketInUseMap_.erase(socket);
        } else {
            socket = it->second.get();
            return HCCL_SUCCESS;
        }
    }

    // 2. 不存在则创建
    CHK_RET(GetNewSocket(socketConfig, socket));
    return HCCL_SUCCESS;
}

HcclResult SocketMgr::PutSocket(const Hccl::SocketConfig*& socketConfig, Hccl::Socket*& socket)
{
    HCCL_INFO("[SocketMgr][%s] start to put a socket", __func__);
    CHK_PTR_NULL(socket);
    CHK_RET(UpdateSocketConfig(socketConfig, socket));
    for (auto it = socketMap_.begin(); it != socketMap_.end(); ++it) {
        if (it->second.get() == socket) {
            socketInUseMap_[it->second.get()] = false;
            socketAvailableCv_.notify_all();
            socket = nullptr;
            return HCCL_SUCCESS;
        }
    }
    HCCL_INFO("[SocketMgr][%s] socket not found in socketInUseMap", __func__);
    return HCCL_SUCCESS;
}

HcclResult SocketMgr::UpdateSocketConfig(const Hccl::SocketConfig*& socketConfig, Hccl::Socket*& socket)
{
    for (auto it = socketMap_.begin(); it != socketMap_.end(); ++it) {
        if (it->second.get() == socket) {
            socketConfig = &(it->first);
            return HCCL_SUCCESS;
        }
    }
    HCCL_INFO("[SocketMgr][%s] socket not found in socketMap", __func__);
    return HCCL_SUCCESS;
}

HcclResult SocketMgr::DeleteWhiteList(const Hccl::Socket* socket)
{
    std::unique_lock<std::mutex> lock(mutex_);
    CHK_PTR_NULL(socket);
    bool socketExist = false;
    for (auto it = socketMap_.begin(); it != socketMap_.end(); ++it) {
        if (it->second.get() == socket) {
            socketExist = true;
            break;
        }
    }
    if (!socketExist) {
        HCCL_WARNING(
            "[DeleteWhiteList] socket[%p] not found in socketMap_, nothing to delete.",
            static_cast<const void*>(socket));
        return HCCL_SUCCESS;
    }
    auto iter = handle2WhiteListMap_.find(socket->GetFdHandle());
    if (iter == handle2WhiteListMap_.end()) {
        HCCL_WARNING(
            "[DeleteWhiteList] socketHandle[%p] not found in handle2WhiteListMap_, nothing to delete.",
            socket->GetFdHandle());
        return HCCL_SUCCESS;
    }

    std::vector<Hccl::RaSocketWhitelist>& wlistInfoVec = iter->second;
    if (wlistInfoVec.empty()) {
        HCCL_WARNING(
            "[DeleteWhiteList] socketHandle[%p] has empty white list, nothing to delete.", socket->GetFdHandle());
        return HCCL_SUCCESS;
    }

    EXCEPTION_CATCH(Hccl::HrtRaSocketWhiteListDel(socket->GetFdHandle(), wlistInfoVec), return HCCL_E_INTERNAL);
    handle2WhiteListMap_.erase(iter);

    return HCCL_SUCCESS;
}

HcclResult SocketMgr::DestroySocket(Hccl::Socket* socket)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (socket == nullptr) {
        HCCL_WARNING("[DestroySocket] socket is nullptr, nothing to destroy.");
        return HCCL_SUCCESS;
    }
    bool socketExist = false;
    for (auto it = socketMap_.begin(); it != socketMap_.end(); ++it) {
        if (it->second.get() == socket) {
            socketExist = true;
            HCCL_INFO(
                "[DestroySocket] Erasing socket inuse info with tag[%s] from socketInUseMap.",
                it->first.GetHccpTag().c_str());
            socketInUseMap_.erase(socket);
            HCCL_INFO("[DestroySocket] Erasing socket with tag[%s] from socketMap.", it->first.GetHccpTag().c_str());
            socketMap_.erase(it);
            break;
        }
    }
    if (!socketExist) {
        HCCL_WARNING("[DestroySocket] socket is not exist in socketMap_, nothing to destroy.");
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}

void SocketMgr::DeInit(u32 devPhyId)
{
    HCCL_INFO("[SocketMgr][%s] DeInit devPhyId[%u]", __func__, devPhyId);
    auto& inst = GetInstance(static_cast<s32>(devPhyId));
    std::lock_guard<std::mutex> lock(inst.mutex_);
    for (auto& it : inst.socketMap_) {
        if (it.second != nullptr) {
            it.second->Destroy();
            it.second.reset();
        }
    }
    inst.socketMap_.clear();
    inst.socketInUseMap_.clear();
    inst.handle2WhiteListMap_.clear();
    inst.isLoaded_ = false;
}

} // namespace hcomm
