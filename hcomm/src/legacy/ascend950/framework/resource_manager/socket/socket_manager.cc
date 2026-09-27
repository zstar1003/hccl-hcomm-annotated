/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <set>
#include <algorithm>
#include "socket_manager.h"
#include "socket_handle_manager.h"
#include "communicator_impl.h"
#include "null_ptr_exception.h"
#include "exception_util.h"
#include "stl_util.h"
#include "preempt_port_manager_v2.h"
#include "timeout_exception.h"
#include "p2p_enable_manager.h"
#include "phy_topo_builder.h"

namespace Hccl {
std::mutex SocketManager::socketLock;

void SocketManager::PrepareLinkAndServerInit(const SocketConfig& socketConfig)
{
    LinkData link = socketConfig.link;

    if (!Contain(availableLinks, link)) {
        if (link.GetLinkProtocol() == LinkProtocol::PCIE) {
            std::vector<uint32_t> remoteDevices;
            remoteDevices.push_back(link.GetRemoteDeviceId());
            auto ret = P2PEnableManager::GetInstance().WaitP2PEnabled(remoteDevices);
            if (ret != HCCL_SUCCESS) {
                THROW<TimeoutException>(
                    StringFormat("WaitP2PEnabled failed, devicePhyId=%d", link.GetRemoteDeviceId()));
            }
        }
        availableLinks.insert({link});
    }

    if (GetConnectedSocket(socketConfig) == nullptr) {
        auto portData = link.GetLocalPort();
        SocketRole role = link.GetLocalRankId() < link.GetRemoteRankId() ? SocketRole::SERVER : SocketRole::CLIENT;
        if (role == SocketRole::SERVER) {
            ServerInit(portData);
        }
    }
}

void SocketManager::ServerListen(const SocketConfig& socketConfig) { PrepareLinkAndServerInit(socketConfig); }

void SocketManager::ConnectSockets(const SocketConfig& socketConfig)
{
    if (GetConnectedSocket(socketConfig) == nullptr) {
        AddWhiteList(socketConfig);
        CreateConnectedSocket(socketConfig);
    }
}

void SocketManager::BatchCreateSockets(const vector<LinkData>& links)
{
    vector<LinkData> pendingLinks;
    for (auto& link : links) {
        if (Contain(availableLinks, link)) {
            continue;
        }
        pendingLinks.emplace_back(link);
    }

    if (pendingLinks.empty()) {
        return;
    }

    for (auto& link : pendingLinks) {
        if (link.GetLinkProtocol() == LinkProtocol::PCIE) {
            std::vector<uint32_t> remoteDevices;
            remoteDevices.push_back(link.GetRemoteDeviceId());
            auto ret = P2PEnableManager::GetInstance().WaitP2PEnabled(remoteDevices);
            if (ret != HCCL_SUCCESS) {
                THROW<TimeoutException>(
                    StringFormat("WaitP2PEnabled failed, devicePhyId=%d", link.GetRemoteDeviceId()));
            }
        }
    }
    BatchServerInit(pendingLinks);
    BatchAddWhiteList(pendingLinks);
    BatchCreateConnectedSockets(pendingLinks);

    availableLinks.insert(pendingLinks.begin(), pendingLinks.end());
}

void SocketManager::BatchCreateSockets(const SocketConfig& socketConfig)
{
    PrepareLinkAndServerInit(socketConfig);
    if (GetConnectedSocket(socketConfig) == nullptr) {
        AddWhiteList(socketConfig);
        CreateConnectedSocket(socketConfig);
    }
}

void SocketManager::AddWhiteList(const SocketConfig& socketConfig)
{
    unordered_map<PortData, vector<RaSocketWhitelist>> wlistMap{};
    LinkData link = socketConfig.link;

    // 通过虚拟拓扑获取Peer可能为空，如果为空，需要抛异，NullPtrException
    // 这里检查rankGraph完整性的逻辑是什么？
    SocketRole role = link.GetLocalRankId() < link.GetRemoteRankId() ? SocketRole::SERVER : SocketRole::CLIENT;
    if (role == SocketRole::SERVER) {
        if (comm) {
            auto peer = comm->GetRankGraph()->GetPeer(link.GetRemoteRankId());
            if (peer == nullptr) {
                auto msg = StringFormat("Fail to get peer of rank %d!", link.GetRemoteRankId());
                THROW<NullPtrException>(msg);
            }
        }

        RaSocketWhitelist wlistInfo{};
        wlistInfo.connLimit = 1;
        wlistInfo.remoteIp = link.GetRemoteAddr();
        wlistInfo.tag = socketConfig.GetHccpTag();

        auto port = link.GetLocalPort();
        vector<RaSocketWhitelist> wlistInfoVec{wlistInfo};
        AddWhiteList(port, wlistInfoVec);
        socketWlistMap[port] = wlistInfoVec;
    }
}

void SocketManager::BatchServerInit(const vector<LinkData>& links)
{
    for (auto& link : links) {
        SocketRole role = link.GetLocalRankId() < link.GetRemoteRankId() ? SocketRole::SERVER : SocketRole::CLIENT;
        if (role == SocketRole::SERVER) {
            auto portData = link.GetLocalPort();
            ServerInit(portData);
        }
    }
}

void SocketManager::BatchAddWhiteList(const vector<LinkData>& links)
{
    unordered_map<PortData, vector<RaSocketWhitelist>> wlistMap{};

    for (const auto& link : links) {
        // 通过虚拟拓扑获取Peer可能为空，如果为空，需要抛异，NullPtrException
        SocketRole role = link.GetLocalRankId() < link.GetRemoteRankId() ? SocketRole::SERVER : SocketRole::CLIENT;
        if (role == SocketRole::SERVER) {
            if (comm) {
                auto peer = comm->GetRankGraph()->GetPeer(link.GetRemoteRankId());
                if (peer == nullptr) {
                    auto msg = StringFormat("Fail to get peer of rank %d!", link.GetRemoteRankId());
                    THROW<NullPtrException>(msg);
                }
            }

            RaSocketWhitelist wlistInfo{};
            ;
            wlistInfo.connLimit = 1;
            wlistInfo.remoteIp = link.GetRemoteAddr();

            std::string linkTag = socketTag_;
            // 获取到reuseIdx不为0时，tag需要拼接_reuseIdx；为0时不拼接，不影响原socket公用
            if (link.GetReuseIdx() != "0") {
                linkTag += ("_" + link.GetReuseIdx());
            }
            SocketConfig socketConfig(link.GetRemoteRankId(), link, linkTag);
            string hccpSocketTag = socketConfig.GetHccpTag();

            wlistInfo.tag = hccpSocketTag;
            wlistMap[link.GetLocalPort()].push_back(wlistInfo);
        }
    }

    for (auto& i : wlistMap) {
        auto port = i.first;
        AddWhiteList(port, i.second);
        socketWlistMap[port] = i.second;
    }
}

void SocketManager::BatchCreateConnectedSockets(const vector<LinkData>& links)
{
    for (auto& link : links) {
        auto remoteRank = link.GetRemoteRankId();
        std::string socketTag = socketTag_;
        if (link.GetReuseIdx() != "0") {
            socketTag += ("_" + link.GetReuseIdx());
        }
        SocketConfig socketConfig(remoteRank, link, socketTag);
        CreateConnectedSocket(socketConfig);
    }
}

void SocketManager::ServerInit(PortData& localPort)
{
    std::lock_guard<std::mutex> lock(socketLock);
    IpAddress ipAddress = localPort.GetAddr();
    u32 serverListenPort = localPort.GetType() == PortDeploymentType::P2P ?
                               GetDeviceListenPort(localPort.GetRankId(), DEVICE_PORT_KEY_IPADDRESS) :
                               GetDeviceListenPort(localPort.GetRankId(), ipAddress);

    auto& serverSocketMap = SocketManager::GetServerSocketMap();
    auto serverSocketInMap = serverSocketMap.find(localPort);
    if (serverSocketInMap != serverSocketMap.end()) {
        auto oldServerSocket = serverSocketMap.at(localPort);
        u32 oldServerListenPort = oldServerSocket->GetListenPort();
        if (oldServerListenPort != serverListenPort) {
            // 自定义算子的时候，会持有一个不关联通信域的SocketManager,
            // 从而获取到的是默认端口，在单卡多进程的时候需要重新导向合适的端口。 通信域算子又可以切换回来。
            bool success = oldServerSocket->Listen(serverListenPort);
            HCCL_INFO(
                "[SocketManager::%s] %s change listen port %u to %u, ret[%u]", __func__, localPort.Describe().c_str(),
                oldServerListenPort, serverListenPort, success);
        }
        HCCL_INFO("[%s] find localPort in serverSocketMap, localPort [%s]", __func__, localPort.Describe().c_str());
        return;
    }

    SocketHandle hccpSocketHandle = SocketHandleManager::GetInstance().Create(devicePhyId, localPort);
    NicType nicType
        = localPort.GetType() == PortDeploymentType::P2P ? NicType::DEVICE_VNIC_TYPE : NicType::DEVICE_NIC_TYPE;
    auto serverSocket = socketProducer(
        ipAddress, ipAddress, serverListenPort, hccpSocketHandle, "server", SocketRole::SERVER, nicType);
    bool success = serverSocket->Listen(serverListenPort);
    if (success) {
        HCCL_RUN_INFO(
            "[SocketManager::%s] Local %s listen the port %u success", __func__, localPort.Describe().c_str(),
            serverListenPort);
    } else {
        string msg = StringFormat(
            "[SocketManager::%s] Local %s listen the port %u failed, maybe other process be listen it", __func__,
            localPort.Describe().c_str(), serverListenPort);
        MACRO_THROW(InvalidParamsException, msg);
    }
    serverSocketMap[localPort] = std::move(serverSocket);
}

void SocketManager::ServerInitAll(NewRankInfo& rankInfo)
{
    vector<SocketPortRange> listenPortRanges = EnvConfig::GetInstance().GetHostNicConfig().GetDeviceSocketPortRange();
    if (listenPortRanges.empty()) {
        HCCL_RUN_INFO("[SocketManager::%s] socket port range not configured.", __func__);
        return;
    }

    const std::string& topoPath = CommunicatorImpl::GetTopoFilePath();
    PhyTopoBuilder::GetInstance().Build(topoPath);

    std::lock_guard<std::mutex> lock(socketLock);
    auto devLogicId = HrtGetDevice();
    auto& serverSocketMap = SocketManager::GetServerSocketMap();
    u32 rankId = rankInfo.rankId;
    u32 localId = rankInfo.localId;
    u32 devicePhyId = rankInfo.deviceId;
    for (auto& rankLevelInfo : rankInfo.rankLevelInfos) {
        shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> graph
            = PhyTopo::GetInstance()->GetTopoGraph(rankLevelInfo.netLayer);
        if (graph == nullptr) {
            HCCL_DEBUG("[SocketManager::%s]Can't find the layout %u Graph!", __func__, rankLevelInfo.netLayer);
            continue;
        }
        std::vector<std::shared_ptr<PhyTopo::Link>> links = graph->GetEdges(localId);
        for (auto& link : links) {
            if (link->GetSourceIFace()->GetPos() == AddrPosition::HOST) {
                continue;
            }
            HCCL_DEBUG("[SocketManager::%s] find the device link %s", __func__, link->Describe().c_str());
            const std::set<LinkProtocol>& protocols = link->GetLinkProtocols();
            for (auto& protocol : protocols) {
                PortDeploymentType deployType = AddrPos2PortDeploymentType(link->GetSourceIFace()->GetPos(), protocol);
                LinkProtoType protoType = LinkProtocol2LinkProtoType(protocol);
                const std::set<std::string>& ports = link->GetSourceIFace()->GetPorts();
                for (auto& rankAddr : rankLevelInfo.rankAddrs) {
                    // topo查得网口使用则打开建链
                    std::set<std::string> intersectSet;
                    std::set_intersection(
                        ports.begin(), ports.end(), rankAddr.ports.begin(), rankAddr.ports.end(),
                        std::inserter(intersectSet, intersectSet.begin()));
                    if (intersectSet.empty()) {
                        continue;
                    }
                    PortData localPort{static_cast<RankId>(rankId), deployType, protoType, 0, rankAddr.addr};
                    u32 listenPort = DEFAULT_VALUE_TCPPORT;
                    if (serverSocketMap.find(localPort) != serverSocketMap.end()) {
                        // 单进程多通信域，找到老端口直接返回老端口
                        listenPort = serverSocketMap[localPort]->GetListenPort();
                        HCCL_INFO(
                            "[SocketManager::%s] Device %s use the old device port %u in same process.", __func__,
                            localPort.Describe().c_str(), listenPort);
                    } else {
                        // 首次执行启用新端口
                        SocketHandle hccpSocketHandle
                            = SocketHandleManager::GetInstance().Create(devicePhyId, localPort);
                        IpAddress ipAddress = localPort.GetAddr();
                        NicType nicType = localPort.GetType() == PortDeploymentType::P2P ? NicType::DEVICE_VNIC_TYPE :
                                                                                           NicType::DEVICE_NIC_TYPE;
                        auto serverSocket = std::make_shared<Socket>(
                            hccpSocketHandle, ipAddress, listenPort, ipAddress, "server", SocketRole::SERVER, nicType);
                        PreemptPortManager::GetInstance(devLogicId)
                            .ListenPreempt(serverSocket, listenPortRanges, listenPort);
                        serverSocketMap[localPort] = std::move(serverSocket);
                        HCCL_RUN_INFO(
                            "[SocketManager::%s] Device %s listen the preempt port %u", __func__,
                            localPort.Describe().c_str(), listenPort);
                    }
                    rankAddr.socketPort_ = listenPort;
                    rankInfo.devicePort = listenPort;
                }
            }
        }
    }
}

void SocketManager::ServerDeInit(PortData& localPort) const
{
    std::lock_guard<std::mutex> lock(socketLock);
    auto& serverSocketMap = SocketManager::GetServerSocketMap();
    auto res = GetServerListenSocket(localPort);
    // 待修改 stop listen maybe needed
    if (res != nullptr) {
        serverSocketMap.erase(localPort);
    }
}

Socket* SocketManager::CreateConnectedSocket(const SocketConfig& socketConfig)
{
    auto res = GetConnectedSocket(socketConfig);
    if (res != nullptr) {
        return res;
    }

    HCCL_INFO("[SocketManager::%s] Create connected socket for tag %s.", __func__, socketConfig.tag.c_str());

    const PortData& localPort = socketConfig.link.GetLocalPort();
    const PortData& remotePort = socketConfig.link.GetRemotePort();

    auto socketHandle = SocketHandleManager::GetInstance().Get(devicePhyId, localPort);
    if (socketHandle == nullptr) {
        socketHandle = SocketHandleManager::GetInstance().Create(devicePhyId, socketConfig.link.GetLocalPort());
    }

    if (socketHandle == nullptr) {
        THROW<NullPtrException>(StringFormat(
            "socketHandle of is nullptr, devicePhyId=%d, port=%s", devicePhyId, localPort.Describe().c_str()));
    }
    IpAddress localIpAddress = socketConfig.link.GetLocalAddr();
    IpAddress remoteIpAddress = socketConfig.link.GetRemoteAddr();
    SocketRole socketRole = socketConfig.GetRole();
    string hccpSocketTag = socketConfig.GetHccpTag();

    u32 serverListenPort = localPort.GetType() == PortDeploymentType::P2P ?
                               GetDeviceListenPort(remotePort.GetRankId(), DEVICE_PORT_KEY_IPADDRESS) :
                               GetDeviceListenPort(remotePort.GetRankId(), remoteIpAddress);
    NicType nicType
        = localPort.GetType() == PortDeploymentType::P2P ? NicType::DEVICE_VNIC_TYPE : NicType::DEVICE_NIC_TYPE;
    auto tmpSocket = socketProducer(
        localIpAddress, remoteIpAddress, serverListenPort, socketHandle, hccpSocketTag, socketRole, nicType);
    HCCL_INFO(
        "[SocketManager::%s] Connect async the remote %s port %u.", __func__, remotePort.Describe().c_str(),
        serverListenPort);
    tmpSocket->ConnectAsync();
    connectedSocketMap[socketConfig] = std::move(tmpSocket);
    return connectedSocketMap[socketConfig].get();
}

Socket* SocketManager::GetConnectedSocket(const SocketConfig& socketConfig) const
{
    HCCL_INFO("[SocketManager::%s] Get connected socket for tag %s.", __func__, socketConfig.tag.c_str());
    auto res = connectedSocketMap.find(socketConfig);
    if (res != connectedSocketMap.end()) {
        return res->second.get();
    }

    return nullptr;
}

void SocketManager::DestroyAll()
{
    for (auto& i : socketWlistMap) {
        auto port = i.first;
        DelWhiteList(port, i.second);
    }
    socketWlistMap.clear();

    for (auto& socket : connectedSocketMap) {
        if (socket.second != nullptr) {
            socket.second->Destroy();
        }
    }
    connectedSocketMap.clear();
    availableLinks.clear();
}

Socket* SocketManager::GetServerListenSocket(const PortData& localPort) const
{
    auto& serverSocketMap = SocketManager::GetServerSocketMap();
    auto res = serverSocketMap.find(localPort);
    if (res != serverSocketMap.end()) {
        return (res->second).get();
    }

    return nullptr;
}

SocketManager::SocketManager(
    const CommunicatorImpl& communicator, u32 localRank, u32 devicePhyId, u32 deviceLogicId,
    std::function<shared_ptr<Socket>(
        IpAddress& localIpAddress, IpAddress& remoteIpAddress, u32 listenPort, SocketHandle socketHandle,
        const std::string& tag, SocketRole socketRole, NicType nicType)>
        socketProducer)
    : comm(&communicator),
      localRank(localRank),
      devicePhyId(devicePhyId),
      deviceLogicId_(deviceLogicId)
{
    if (socketProducer != nullptr) {
        this->socketProducer = socketProducer;
    }

    if (comm != nullptr) {
        socketTag_ = comm->GetEstablishLinkSocketTag();
    }
}

SocketManager::SocketManager(u32 localRank, u32 devicePhyId, u32 deviceLogicId, const std::string& socketTag)
    : comm(nullptr),
      localRank(localRank),
      devicePhyId(devicePhyId),
      deviceLogicId_(deviceLogicId)
{
    socketTag_ = socketTag;
}

void SocketManager::AddWhiteList(PortData& localPort, vector<RaSocketWhitelist>& wlistInfoVec) const
{
    auto socketHandle = SocketHandleManager::GetInstance().Get(devicePhyId, localPort);
    if (socketHandle == nullptr) {
        THROW<NullPtrException>(StringFormat(
            "socketHandle of is nullptr, devicePhyId=%d, port=%s", devicePhyId, localPort.Describe().c_str()));
    }
    HrtRaSocketWhiteListAdd(socketHandle, wlistInfoVec);
}

bool SocketManager::DelWhiteList(PortData& localPort, vector<RaSocketWhitelist>& wlistInfoVec) const
{
    auto socketHandle = SocketHandleManager::GetInstance().Get(devicePhyId, localPort);
    if (socketHandle == nullptr) {
        return false;
    }
    HrtRaSocketWhiteListDel(socketHandle, wlistInfoVec);
    return true;
}

HcclResult SocketManager::SetDeviceServerListenPortMap(const RankIpPortMapPtr& rankListenPortMap)
{
    CHK_PTR_NULL(rankListenPortMap);
    std::lock_guard<std::mutex> lock(socketLock);
    rankListenPortMap_ = rankListenPortMap;
    return HCCL_SUCCESS;
}

HcclResult
SocketManager::GetSubCommDeviceServerListenPortMap(const std::vector<u32>& rankIds, RankIpPortMapPtr& subMap) const
{
    std::lock_guard<std::mutex> lock(socketLock);
    subMap = std::make_shared<RankIpPortMap>();
    if (rankListenPortMap_ == nullptr) {
        HCCL_WARNING("[SocketManager::%s]rankListenPortMap_ is null, subMap is empty.", __func__);
        return HCCL_SUCCESS;
    }
    for (u32 subRankId = 0; subRankId < rankIds.size(); ++subRankId) {
        u32 rankId = rankIds[subRankId];
        if (rankListenPortMap_->find(rankId) == rankListenPortMap_->end()) {
            HCCL_WARNING("[SocketManager::%s]Can't find listen port for rank %u to sub comm.", __func__, rankId);
        } else {
            subMap->insert(std::make_pair(subRankId, rankListenPortMap_->at(rankId)));
        }
    }
    return HCCL_SUCCESS;
}

u32 SocketManager::GetDeviceListenPort(const u32& rankId, const IpAddress& ipAddress)
{
    if (rankListenPortMap_ != nullptr) {
        auto rankIter = rankListenPortMap_->find(rankId);
        if (rankIter != rankListenPortMap_->end()) {
            auto ipIter = rankIter->second.find(ipAddress);
            if (ipIter != rankIter->second.end()) {
                return ipIter->second;
            }
        }
    }
    u32 cachedPort = defaultListenPort_.load(std::memory_order_relaxed);
    if (cachedPort != 0) {
        return cachedPort;
    }
    auto portRanges = EnvConfig::GetInstance().GetHostNicConfig().GetDeviceSocketPortRange();
    if (!portRanges.empty()) {
        defaultListenPort_.store(portRanges[0].min, std::memory_order_relaxed);
        HCCL_INFO(
            "[SocketManager::%s] Can't find rankId[%u], addr[%s] listen port, use port[%u] from "
            "HCCL_NPU_SOCKET_PORT_RANGE and cache it",
            __func__, rankId, ipAddress.Describe().c_str(), defaultListenPort_.load(std::memory_order_relaxed));
    } else {
        defaultListenPort_.store(DEFAULT_VALUE_TCPPORT, std::memory_order_relaxed);
        HCCL_WARNING(
            "[SocketManager::%s] Can't find rankId[%u], addr[%s] listen port, use default port[%u] and cache it",
            __func__, rankId, ipAddress.Describe().c_str(), defaultListenPort_.load(std::memory_order_relaxed));
    }
    return defaultListenPort_.load(std::memory_order_relaxed);
}

SocketManager::~SocketManager() { DECTOR_TRY_CATCH("SocketManager", DestroyAll()); }

std::unordered_map<PortData, shared_ptr<Socket>>& SocketManager::GetServerSocketMap()
{
    static std::unordered_map<PortData, shared_ptr<Socket>> serverSocketMap;
    return serverSocketMap;
}

bool SocketManager::CheckServerPortListening(const PortData& portData, const uint32_t port) const
{
    std::lock_guard<std::mutex> lock(socketLock);
    auto& serverSocketMap = SocketManager::GetServerSocketMap();
    auto iterSocket = serverSocketMap.find(portData);
    if (iterSocket == serverSocketMap.end()) {
        return false;
    }
    if (iterSocket->second->GetListenPort() != port) {
        return false;
    }
    return true;
}

bool SocketManager::RegisterHostListenSocket(const PortData& portData, std::shared_ptr<Socket> listenSocket) const
{
    std::lock_guard<std::mutex> lock(socketLock);
    auto& serverSocketMap = SocketManager::GetServerSocketMap();
    auto iter = serverSocketMap.find(portData);
    if (iter != serverSocketMap.end()) {
        HCCL_WARNING(
            "[SocketManager::%s] portData[%s] already registered, skip.", __func__, portData.Describe().c_str());
        return false;
    }
    serverSocketMap[portData] = listenSocket;
    HCCL_INFO(
        "[SocketManager::%s] register host listen socket, portData[%s], listenPort[%u].", __func__,
        portData.Describe().c_str(), listenSocket->GetListenPort());
    return true;
}

} // namespace Hccl
