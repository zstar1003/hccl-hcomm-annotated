/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_ROCE_SERVER_SOCKET_CONTEXT_H
#define AICPU_TS_ROCE_SERVER_SOCKET_CONTEXT_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "port.h"
#include "hccl_mem_defs.h"
#include "hccl_socket.h"
#include "externalinput_pub.h"
#include "server_socket_context.h"

namespace hcomm {

// AicpuTsRoce 默认监听端口（从 aicpu_ts_roce_endpoint.cc 匿名空间提升，供 context 与 endpoint 数据面共用）
constexpr uint32_t kDefaultAicpuTsRocePort = 16666;

// ---- HcclSocket 自管理路径（静态 serverSocketMap + refCount 复用，从 AicpuTsRoceEndpoint 迁入）----

// AicpuTsRoce 监听 socket 复用表 key/槽位（从 aicpu_ts_roce_endpoint.h 迁入，供 context 与 endpoint 数据面共用）
struct AicpuTsListenSocketSlot {
    std::shared_ptr<hccl::HcclSocket> socket{};
    uint32_t refCount{0U};
};

struct SocketMapKey {
    uint32_t devicePhyId;
    uint32_t port;

    bool operator==(const SocketMapKey& other) const { return devicePhyId == other.devicePhyId && port == other.port; }
};

struct SocketMapKeyHash {
    size_t operator()(const SocketMapKey& k) const
    {
        return std::hash<uint32_t>()(k.devicePhyId) ^ (std::hash<uint32_t>()(k.port) << 1);
    }
};

// AICPU_TS RoCE 专用：自建 HcclSocket + 静态 serverSocketMap（refCount 复用），不走 ServerSocketManager，无
// dynamicPort_
class AicpuTsRoceServerSocketContext : public ServerSocketContext {
public:
    AicpuTsRoceServerSocketContext(HcclNetDev netDev, uint32_t netDevRefPhyId);
    ~AicpuTsRoceServerSocketContext() override; // 析构释放本实例持有的监听引用（ReleaseListenSocketRefs）
    HcclResult
    ServerSocketListen(uint32_t port) override; // 默认端口 kDefaultAicpuTsRocePort；ReuseListenSocketIfExist 复用
    // ServerSocketStopListen/ServerSocketGetListenPort 迁移前（Endpoint 时期）未覆写，
    // 继承基类默认 HCCL_E_NOT_SUPPORT，保持迁移前对外行为

    // ---- 数据面方法（从 AicpuTsRoceEndpoint 原封迁入，调用方改经本 context 访问）----
    HcclResult GetSocket(uint32_t port, const std::string& tag, std::shared_ptr<hccl::HcclSocket>& outConnected) const;
    HcclResult AcceptDataSocket(
        uint32_t port, const std::string& tag, std::shared_ptr<hccl::HcclSocket>& outConnected,
        uint32_t acceptTimeoutMs = 0) const;

    HcclResult AddListenSocketWhiteList(uint32_t port, const std::vector<SocketWlistInfo>& wlistInfos) const;

    // 监听 socket 复用表（静态 map + 静态锁，从 AicpuTsRoceEndpoint 迁入；数据面方法 Accept/WhiteList 经此访问）
    static std::unordered_map<SocketMapKey, AicpuTsListenSocketSlot, SocketMapKeyHash>& GetServerSocketMap();
    static std::mutex& ListenSocketMapMutex();

    // 析构顺序需要：宿主 AicpuTsRoceEndpoint 析构体内需先释放监听引用再释放共享 netDev
    // （保持原 AicpuTsRoceEndpoint 析构顺序）
    void ReleaseListenSocketRefs();
    bool HasListenSocketRef() const { return hasListenSocketRef_; }
    uint32_t GetNetDevRefPhyId() const { return netDevRefPhyId_; }

private:
    bool ReuseListenSocketIfExist(const SocketMapKey& key, const char* logPrefix);

    HcclNetDev netDev_;
    uint32_t netDevRefPhyId_;
    std::vector<SocketMapKey> listenRefKeys_;
    bool hasListenSocketRef_{false};
};

} // namespace hcomm

#endif // AICPU_TS_ROCE_SERVER_SOCKET_CONTEXT_H
