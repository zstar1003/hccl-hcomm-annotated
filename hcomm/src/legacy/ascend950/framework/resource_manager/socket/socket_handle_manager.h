/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_SOCKET_HANDLE_MANAGER_H
#define HCCLV2_SOCKET_HANDLE_MANAGER_H

#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

#include "port.h"
#include "virtual_topo.h"
#include "ip_address.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

constexpr u32 MAX_DEVICE_NUM = 65;
constexpr u32 LINK_PROTO_TYPE_NUM = static_cast<u32>(LinkProtoType::__COUNT__);
constexpr u32 MAX_PORT_NUM = 16;

using SocketHandle = void*;
class SocketHandleManager {
public:
    static SocketHandleManager& GetInstance();

    SocketHandle Create(u32 devicePhyId, const PortData& localPort);

    SocketHandle Get(u32 devicePhyId, const PortData& localPort);

    void DeInit(u32 devPhyId);

    ~SocketHandleManager();

private:
    std::vector<std::vector<std::unordered_map<IpAddress, SocketHandle>>>
        hccpSocketHandleMap; // key: devicePhyId, ConnectProtoType, portAddr, 目前仅限于device侧

    SocketHandleManager();

    void DestroyAll();

    SocketHandleManager(const SocketHandleManager& socketHandleManager) = delete;

    SocketHandleManager& operator=(const SocketHandleManager& socketHandleManager) = delete;

    std::mutex socketHandleLock;
    std::atomic<bool> destroyed{false};
};

} // namespace Hccl

#endif // HCCLV2_SOCKET_HANDLE_MANAGER_H
