/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_SOCKET_MANAGER_H
#define HCCLV2_SOCKET_MANAGER_H

#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <memory>
#include <atomic>

#include "../../unified_platform/resource/socket/socket.h"
#include "virtual_topo.h"
#include "socket_config.h"
#include "env_func.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

using RankIpPortMap = std::unordered_map<u32, std::unordered_map<IpAddress, u32>>;
using RankIpPortMapPtr = std::shared_ptr<RankIpPortMap>;

class CommunicatorImpl;
class SocketManager {
public:
    SocketManager() = default;
    SocketManager(
        const CommunicatorImpl& communicator, u32 localRank, u32 devicePhyId, u32 deviceLogicId,
        std::function<shared_ptr<Socket>(
            IpAddress& localIpAddress, IpAddress& remoteIpAddress, u32 listenPort, SocketHandle socketHandle,
            const std::string& tag, SocketRole socketRole, NicType nicType)>
            socketProducer
        = nullptr);

    SocketManager(u32 localRank, u32 devicePhyId, u32 deviceLogicId, const std::string& socketTag);

    HcclResult SetDeviceServerListenPortMap(const RankIpPortMapPtr& rankListenPortMap);
    HcclResult GetSubCommDeviceServerListenPortMap(const std::vector<u32>& rankIds, RankIpPortMapPtr& subMap) const;

    u32 GetDeviceListenPort(const u32& rankId, const IpAddress& ipAddress);

    void BatchCreateSockets(const vector<LinkData>& links);
    void ServerListen(const SocketConfig& socketConfig);
    void ConnectSockets(const SocketConfig& socketConfig);

    void BatchCreateSockets(const SocketConfig& socketConfig);

    void ServerInit(PortData& localPort);

    static void ServerInitAll(NewRankInfo& rankInfo);

    void ServerDeInit(PortData& localPort) const;

    Socket* CreateConnectedSocket(const SocketConfig& socketConfig);

    Socket* GetConnectedSocket(const SocketConfig& socketConfig) const;

    bool CheckServerPortListening(const PortData& portData, const uint32_t port) const;

    // 供通信域初始化阶段登记已抢占的 host 监听 socket，供算子下发阶段复用，避免跨阶段端口竞争
    bool RegisterHostListenSocket(const PortData& portData, std::shared_ptr<Socket> listenSocket) const;

    void DestroyAll();

    void AddWhiteList(PortData& localPort, vector<RaSocketWhitelist>& wlistInfoVec) const;

    bool DelWhiteList(PortData& localPort, vector<RaSocketWhitelist>& wlistInfoVec) const;

    ~SocketManager();

    SocketManager(const SocketManager& socketManager) = delete;

    SocketManager& operator=(const SocketManager& socketManager) = delete;

private:
    void PrepareLinkAndServerInit(const SocketConfig& socketConfig);
    void BatchServerInit(const vector<LinkData>& links);
    void BatchAddWhiteList(const vector<LinkData>& links);
    void BatchCreateConnectedSockets(const vector<LinkData>& links);
    void AddWhiteList(const SocketConfig& socketConfig);
    const CommunicatorImpl* comm;
    static std::unordered_map<PortData, shared_ptr<Socket>>& GetServerSocketMap();
    u32 localRank;
    u32 devicePhyId;
    u32 deviceLogicId_;
    RankIpPortMapPtr rankListenPortMap_;
    std::atomic<uint32_t> defaultListenPort_{0};
    std::function<shared_ptr<Socket>(
        IpAddress& localIpAddress, IpAddress& remoteIpAddress, u32 listenPort, SocketHandle socketHandle,
        const std::string& tag, SocketRole socketRole, NicType nicType)>
        socketProducer
        = [](IpAddress& localIpAddress, IpAddress& remoteIpAddress, u32 listenPort, SocketHandle socketHandle,
             const std::string& tag, SocketRole socketRole, NicType nicType) -> shared_ptr<Socket> {
        auto tmpSocket = std::make_shared<Socket>(
            socketHandle, localIpAddress, listenPort, remoteIpAddress, tag, socketRole, nicType);
        HCCL_INFO("create socket with role %u", static_cast<u32>(socketRole));
        return tmpSocket;
    };

    std::unordered_map<SocketConfig, shared_ptr<Socket>> connectedSocketMap;
    std::unordered_map<PortData, vector<RaSocketWhitelist>> socketWlistMap{};

    Socket* GetServerListenSocket(const PortData& localPort) const;
    std::set<LinkData> availableLinks;

    std::string socketTag_{};
    static std::mutex socketLock;
};

} // namespace Hccl

#endif // HCCLV2_SOCKET_MANAGER_H
