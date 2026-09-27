/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_PAIR_H
#define ENDPOINT_PAIR_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include "channels/channel.h"
#include "socket_mgr.h"
#include "../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "../../../../legacy/ascend950/framework/resource_manager/socket/socket_manager.h"

using EndpointDescPair = std::pair<EndpointDesc, EndpointDesc>;

// 重载 == 操作符，用于 EndpointDesc 在 std::unordered_map 比较
inline bool operator==(const EndpointDesc& a, const EndpointDesc& b) noexcept
{
    return std::memcmp(&a, &b, sizeof(EndpointDesc)) == 0;
}

namespace std {

template <>
struct hash<EndpointDesc> {
    size_t operator()(const EndpointDesc& e) const noexcept
    {
        // FNV-1a（64位）对字节序列做hash
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&e);
        size_t h
            = sizeof(size_t) == 8 ? static_cast<size_t>(14695981039346656037ull) : static_cast<size_t>(2166136261u);

        const size_t prime
            = sizeof(size_t) == 8 ? static_cast<size_t>(1099511628211ull) : static_cast<size_t>(16777619u);

        for (size_t i = 0; i < sizeof(EndpointDesc); ++i) {
            h ^= static_cast<size_t>(p[i]);
            h *= prime;
        }
        return h;
    }
};

template <>
struct hash<EndpointDescPair> {
    size_t operator()(const EndpointDescPair& p) const noexcept
    {
        size_t h1 = std::hash<EndpointDesc>{}(p.first);
        size_t h2 = std::hash<EndpointDesc>{}(p.second);
        size_t h = h1;
        h ^= h2 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

} // namespace std

namespace hcomm {
/**
 * @note 职责：通信设备Endpoint对（EndpointPair，或连接）的C++类声明，
 * 管理该Endpoint对上的本地和远端注册内存、多个Channel、以及socket等。两个Endpoint的通信协议一致。
 */
class EndpointPair {
public:
    EndpointPair(
        EndpointDesc localEndpointDesc, EndpointDesc remoteEndpointDesc, const Hccl::RankIpPortMapPtr& rankIpPortMap)
        : localEndpointDesc_(localEndpointDesc),
          remoteEndpointDesc_(remoteEndpointDesc),
          rankIpPortMap_(rankIpPortMap)
    {}
    ~EndpointPair();

    HcclResult Init();

    // 临时方案：新增临时接口用于支持混跑
    HcclResult GetSocket(
        const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, u32 reuseIdx,
        const uint32_t listenPort, Hccl::Socket*& socket, uint32_t devicePhyId, uint32_t remoteDevicePhyId);
    HcclResult GetHostSocketWithRank(
        const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, const uint32_t listenPort,
        u32 reuseIdx, Hccl::Socket*& socket);

    HcclResult ServerInit(
        const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, u32 reuseIdx, uint32_t devicePhyId,
        uint32_t remoteDevicePhyId);
    HcclResult GetConnectedSocket(
        const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, u32 reuseIdx,
        const uint32_t listenPort, Hccl::Socket*& socket, uint32_t devicePhyId, uint32_t remoteDevicePhyId);

    HcclResult CreateChannel(
        EndpointHandle endpointHandle, CommEngine engine, u32 reuseIdx, HcommChannelDesc* channelDescs,
        ChannelHandle* channels);

    HcclResult DestroyChannel(CommEngine engine, u32 reuseIdx);

    bool IsChannelNotExist(CommEngine engine, u32 reuseIdx);

    // 持锁返回 channelHandles_ 副本，避免外部持有引用时与 CreateChannel/DestroyChannel 并发修改产生数据竞争
    std::unordered_map<CommEngine, std::vector<ChannelHandle>> GetChannelHandles() const;

    // 持锁读取指定引擎指定槽位的句柄，避免外部引用内部向量造成并发读写
    bool GetChannelHandle(CommEngine engine, u32 reuseIdx, ChannelHandle& handle) const;

    // 反查 handle -> (engine, 真实槽位)
    bool FindChannelLoc(ChannelHandle handle, CommEngine& engine, u32& reuseIdx) const;

private:
    HcclResult EnsureSocketMgrCompat(const uint32_t myRank, const std::string& socketTag);
    Hccl::SocketConfig BuildSocketConfig(const Hccl::LinkData& linkData, const std::string& socketTag) const;
    HcclResult HandleHostSocketOrBuildLinkData(
        const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, u32 reuseIdx,
        const uint32_t listenPort, Hccl::Socket*& socket, uint32_t devicePhyId, uint32_t remoteDevicePhyId,
        Hccl::LinkData& linkData, bool& isHost);
    HcclResult GetSocketInternal(
        const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, u32 reuseIdx,
        const uint32_t listenPort, Hccl::Socket*& socket, uint32_t devicePhyId, uint32_t remoteDevicePhyId,
        bool connectMode);

    EndpointDesc localEndpointDesc_{};
    EndpointDesc remoteEndpointDesc_{};
    std::unique_ptr<Hccl::SocketManager> socketMgrCompat_;
    std::mutex socketMgrMtx_;
    std::unordered_map<CommEngine, std::vector<ChannelHandle>> channelHandles_{};
    // handle -> (engine, 真实槽位) 反查索引
    std::unordered_map<ChannelHandle, std::pair<CommEngine, u32>> handleToLoc_{};
    // 保护 channelHandles_ 与 handleToLoc_ 的并发访问
    mutable std::mutex channelMtx_{};
    Hccl::RankIpPortMapPtr rankIpPortMap_;
    uint32_t devicePhyId_{};
    std::unique_ptr<SocketMgr> socketMgr_;
};

} // namespace hcomm

#endif // ENDPOINT_PAIR_H
