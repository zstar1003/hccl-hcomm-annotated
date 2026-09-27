/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_socket_handle_manager.h"
#include "exception_util.h"

namespace Hccl {

HostSocketHandleManager::HostSocketHandleManager() { hostSocketHandleMap.resize(MAX_DEVICE_NUM); }

HostSocketHandleManager::~HostSocketHandleManager() { DECTOR_TRY_CATCH("HostSocketHandleManager", DestroyAll()); }

HostSocketHandleManager& HostSocketHandleManager::GetInstance()
{
    static HostSocketHandleManager hostSocketHandleManager;
    return hostSocketHandleManager;
}

SocketHandle HostSocketHandleManager::Create(DevId devicePhyId, const IpAddress& hostIp)
{
    HCCL_INFO("[HostSocketHandleManager::%s] start", __func__);

    // 加锁
    std::lock_guard<std::mutex> lock(socketHandleLock);

    if (isDestroy) {
        HCCL_WARNING(
            "[HostSocketHandleManager::%s] devicePhyId[%u] HostSocketHandleManager has been destroyed", __func__,
            devicePhyId);
        return nullptr;
    }

    // 校验devicePhyId
    CHK_PRT_THROW(
        (devicePhyId > hostSocketHandleMap.size() - 1),
        HCCL_ERROR("[HostSocketHandleManager::%s] devicePhyId[%u] error", __func__, devicePhyId),
        InvalidParamsException, "devicePhyId error");

    // 若socketHandle已存在，引用计数+1
    auto& handles = hostSocketHandleMap[devicePhyId];
    auto ip = hostIp.GetIpStr();
    if (handles.count(ip) != 0) {
        handles[ip].second.Ref();
        HCCL_INFO(
            "[HostSocketHandleManager::%s] devicePhyId[%u] hostIp[%d] socket has initialized,"
            " ref[%u]",
            __func__, devicePhyId, ip.c_str(), handles[ip].second.Count());
        return handles[ip].first;
    }

    // 初始化socketHandle
    RaInterface intf{};
    intf.phyId = devicePhyId;
    intf.address = hostIp;

    SocketHandle socketHandle = HrtRaSocketInit(HrtNetworkMode::PEER, intf);
    handles[ip] = std::make_pair(socketHandle, Referenced());
    handles[ip].second.Ref();

    HCCL_INFO(
        "[HostSocketHandleManager::%s] devicePhyId[%u] hostIp[%s] create end.", __func__, devicePhyId,
        hostIp.GetIpStr().c_str());
    return socketHandle;
}

SocketHandle HostSocketHandleManager::Get(DevId devicePhyId, const IpAddress& hostIp)
{
    std::lock_guard<std::mutex> lock(socketHandleLock);

    if (isDestroy) {
        HCCL_WARNING(
            "[HostSocketHandleManager::%s] devicePhyId[%u] HostSocketHandleManager has been destroyed", __func__,
            devicePhyId);
        return nullptr;
    }

    if (devicePhyId > hostSocketHandleMap.size() - 1) {
        HCCL_WARNING("HostSocketHandleManager for devicePhyId=%u does not exist", devicePhyId);
        return nullptr;
    }

    auto tempiter = hostSocketHandleMap[devicePhyId].find(hostIp.GetIpStr());
    if (tempiter == hostSocketHandleMap[devicePhyId].end()) {
        HCCL_WARNING("HostSocketHandleManager for IpAddress=%s does not exist", hostIp.GetIpStr().c_str());
        return nullptr;
    }

    return tempiter->second.first;
}

void HostSocketHandleManager::DestroyAll()
{
    std::lock_guard<std::mutex> lock(socketHandleLock);

    isDestroy = true;

    for (u32 i = 0; i < hostSocketHandleMap.size(); ++i) {
        for (const auto& innerMap : hostSocketHandleMap[i]) {
            u32 count = innerMap.second.second.Count();
            CHK_PRT_CONT(
                count != 0, HCCL_WARNING(
                                "[HostSocketHandleManager::%s] release is not as expected, "
                                "devicePhyId[%u] hostIp[%s] ref[%u]",
                                __func__, i, innerMap.first.c_str(), count));
            DECTOR_TRY_CATCH("HrtRaSocketDeInit Exception", HrtRaSocketDeInit(innerMap.second.first));
        }
        hostSocketHandleMap[i].clear();
    }
}

void HostSocketHandleManager::Destroy(DevId devicePhyId, const IpAddress& hostIp)
{
    std::lock_guard<std::mutex> lock(socketHandleLock);

    if (isDestroy) {
        HCCL_WARNING(
            "[HostSocketHandleManager::%s] devicePhyId[%u] HostSocketHandleManager has been destroyed", __func__,
            devicePhyId);
        return;
    }

    // 校验devicePhyId
    CHK_PRT_THROW(
        (devicePhyId > hostSocketHandleMap.size() - 1),
        HCCL_ERROR("[HostSocketHandleManager::%s] devicePhyId[%u] invalid", __func__, devicePhyId),
        InvalidParamsException, "devicePhyId invalid");

    // 校验hostIp
    CHK_PRT_THROW(
        hostSocketHandleMap[devicePhyId].count(hostIp.GetIpStr()) == 0,
        HCCL_ERROR(
            "[HostSocketHandleManager::%s] devicePhyId[%u] hostIp[%s] does not exist", __func__, devicePhyId,
            hostIp.GetIpStr().c_str()),
        InvalidParamsException, "hostIp not exist");

    // 引用计数-1
    auto& socketHandleRef = hostSocketHandleMap[devicePhyId][hostIp.GetIpStr()];
    socketHandleRef.second.Unref();

    // 打印
    u32 count = socketHandleRef.second.Count();
    HCCL_INFO(
        "[HostSocketHandleManager::%s] devicePhyId[%u] hostIp[%s] release one, ref[%u].", __func__, devicePhyId,
        hostIp.GetIpStr().c_str(), count);

    // 若引用计数为0则deinit
    if (count == 0) {
        HrtRaSocketDeInit(socketHandleRef.first);
        hostSocketHandleMap[devicePhyId].erase(hostIp.GetIpStr());
        HCCL_INFO(
            "[HostSocketHandleManager::%s] devicePhyId[%u] hostIp[%s] deinit success.", __func__, devicePhyId,
            hostIp.GetIpStr().c_str());
    }
}

} // namespace Hccl
