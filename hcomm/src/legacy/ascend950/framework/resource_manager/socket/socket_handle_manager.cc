/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "socket_handle_manager.h"
#include "internal_exception.h"

namespace Hccl {

SocketHandleManager::SocketHandleManager()
{
    hccpSocketHandleMap.resize(MAX_DEVICE_NUM);
    for (u32 i = 0; i < hccpSocketHandleMap.size(); ++i) {
        hccpSocketHandleMap[i].resize(LINK_PROTO_TYPE_NUM);
    }
}

SocketHandleManager::~SocketHandleManager() { DECTOR_TRY_CATCH("SocketHandleManager", DestroyAll()); }

SocketHandleManager& SocketHandleManager::GetInstance()
{
    static SocketHandleManager socketHandleManager;
    return socketHandleManager;
}

SocketHandle SocketHandleManager::Create(DevId devicePhyId, const PortData& localPort)
{
    RaInterface intf{};
    intf.phyId = devicePhyId;
    if (localPort.GetType() == PortDeploymentType::P2P) {
        intf.address = localPort.GetAddr();
    } else if (localPort.GetType() == PortDeploymentType::DEV_NET) {
        intf.address = localPort.GetAddr();
    } else {
        string msg = StringFormat("Not support this type now: %s", localPort.GetType().Describe().c_str());
        THROW<NotSupportException>(msg);
    }

    std::lock_guard<std::mutex> lock(socketHandleLock);
    if (devicePhyId > hccpSocketHandleMap.size() - 1
        || static_cast<u32>(localPort.GetProto()) > hccpSocketHandleMap[devicePhyId].size() - 1) {
        string msg = StringFormat(
            "devicePhyId %u or prototype %u out of range", devicePhyId, static_cast<u32>(localPort.GetProto()));
        THROW<InternalException>(msg);
    }
    if (hccpSocketHandleMap[devicePhyId][static_cast<u32>(localPort.GetProto())].find(localPort.GetAddr())
        != hccpSocketHandleMap[devicePhyId][static_cast<u32>(localPort.GetProto())].end()) {
        return hccpSocketHandleMap[devicePhyId][static_cast<u32>(localPort.GetProto())][localPort.GetAddr()];
    }
    SocketHandle socketHandle = HrtRaSocketInit(HrtNetworkMode::HDC, intf);

    if ((u32)localPort.GetProto() > LINK_PROTO_TYPE_NUM - 1) {
        HrtRaSocketDeInit(socketHandle);
        HCCL_ERROR("Invalid LinkProtoType.");
        return nullptr;
    }
    hccpSocketHandleMap[devicePhyId][static_cast<u32>(localPort.GetProto())][localPort.GetAddr()] = socketHandle;
    return socketHandle;
}

SocketHandle SocketHandleManager::Get(u32 devicePhyId, const PortData& localPort)
{
    std::lock_guard<std::mutex> lock(socketHandleLock);
    if (devicePhyId > hccpSocketHandleMap.size() - 1
        || static_cast<u32>(localPort.GetProto()) > hccpSocketHandleMap[devicePhyId].size() - 1) {
        return nullptr;
    }
    if (hccpSocketHandleMap[devicePhyId][static_cast<u32>(localPort.GetProto())].find(localPort.GetAddr())
        == hccpSocketHandleMap[devicePhyId][static_cast<u32>(localPort.GetProto())].end()) {
        return nullptr;
    }
    return hccpSocketHandleMap[devicePhyId][static_cast<u32>(localPort.GetProto())][localPort.GetAddr()];
}

void SocketHandleManager::DestroyAll()
{
    if (destroyed.load()) {
        return;
    }
    destroyed.store(true);

    std::lock_guard<std::mutex> lock(socketHandleLock);
    for (u32 i = 0; i < hccpSocketHandleMap.size(); ++i) {
        for (u32 j = 0; j < hccpSocketHandleMap[i].size(); ++j) {
            for (auto& iterHandle : hccpSocketHandleMap[i][j]) {
                if (iterHandle.second != nullptr) {
                    DECTOR_TRY_CATCH("RaSocketDeinit", HrtRaSocketDeInit(iterHandle.second));
                    iterHandle.second = nullptr;
                }
            }
        }
    }
    hccpSocketHandleMap.clear();
}

void SocketHandleManager::DeInit(u32 devPhyId)
{
    if (destroyed.load()) {
        return;
    }
    HCCL_INFO("[SocketHandleManager][%s] DeInit[%u]", __func__, devPhyId);
    std::lock_guard<std::mutex> lock(socketHandleLock);
    if (devPhyId >= hccpSocketHandleMap.size()) {
        return;
    }
    for (u32 j = 0; j < hccpSocketHandleMap[devPhyId].size(); ++j) {
        for (auto& iterHandle : hccpSocketHandleMap[devPhyId][j]) {
            if (iterHandle.second != nullptr) {
                DECTOR_TRY_CATCH("RaSocketDeinit", HrtRaSocketDeInit(iterHandle.second));
                iterHandle.second = nullptr;
            }
        }
        hccpSocketHandleMap[devPhyId][j].clear();
    }
}

} // namespace Hccl
