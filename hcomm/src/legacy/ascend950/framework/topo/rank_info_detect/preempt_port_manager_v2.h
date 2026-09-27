/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_PREEMPT_SOCKET_MANAGER_H
#define HCCL_PREEMPT_SOCKET_MANAGER_H

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "hccl/base.h"
#include "orion_adapter_hccp.h"
#include "env_func.h"
#include "socket.h"
#include "referenced.h"

namespace Hccl {

struct EnumClassHash {
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};

// 以IP和port的粒度管理已经抢占的port的计数器
using IpPortRef = std::unordered_map<std::string, std::pair<u32, Referenced>>;

class PreemptPortManager {
public:
    ~PreemptPortManager();

    static PreemptPortManager& GetInstance(s32 deviceLogicId);

    // 尝试在给定范围内抢占一个端口
    void ListenPreempt(
        const std::shared_ptr<Socket>& listenSocket, const std::vector<SocketPortRange>& portRange, u32& usePort);
    // 释放一个已抢占的端口
    void Release(const std::shared_ptr<Socket>& listenSocket);

private:
    explicit PreemptPortManager();

    void PreemptPortInRange(
        const std::shared_ptr<Socket>& listenSocket, HrtNetworkMode netMode,
        const std::vector<SocketPortRange>& portRange, u32& usePort);
    void ReleasePreempt(IpPortRef& portRef, const std::shared_ptr<Socket>& listenSocket, HrtNetworkMode netMode);

    bool IsAlreadyListening(const IpPortRef& ipPortRef, const std::string& ipAddr, const u32 port);
    std::string GetRangeStr(const std::vector<SocketPortRange>& portRangeVec);

    static bool initialized;
    s32 deviceLogicId_;
    u32 devicePhyId_;
    std::mutex preemptMutex_;

    // 不同类型网卡上抢占的Socket
    std::unordered_map<HrtNetworkMode, IpPortRef, EnumClassHash> preemptSockets_;
};
} // namespace Hccl
#endif // HCCL_PREEMPT_SOCKET_MANAGER_H
