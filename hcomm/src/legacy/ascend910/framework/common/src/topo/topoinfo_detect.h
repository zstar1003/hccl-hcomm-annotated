/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_DETECT_H
#define TOPOINFO_DETECT_H

#include <thread>
#include <vector>
#include "comm.h"
#include "hccl_comm_pub.h"
#include "topoinfo_exchange_server.h"
#include "topoinfo_exchange_agent.h"
#include "env_config.h"
#include "hccl_socket.h"
#include "hccl_network_pub.h"
#include "universal_concurrent_map.h"

namespace hccl {
class TopoInfoDetect {
public:
    explicit TopoInfoDetect();
    virtual ~TopoInfoDetect();
    HcclResult SetupGroupMember(u32 rankSize, u32 myrank, const HcclRootHandle& rootInfo); // Group内成员rank
    HcclResult SetupAgent(
        u32 rankSize, u32 myrank, const HcclRootHandle& rootInfo, const HcclRankHandle& rankHandle,
        const CommConfig& commConfig, bool isScalable = false); // 分层建链时使用的SetupAgent，isScalable压制分层分支
    HcclResult CheckHostNicLinkUp(const HcclIpAddress& hostIP) const;
    HcclResult PrepareHandle(HcclRankHandle& rankHandle, std::vector<HcclIpAddress>& whitelist); // 准备要发送的agent
    HcclResult SetupRank(std::shared_ptr<HcclSocket>& agentConnRoot); // 分层建链时获得每个GroupLeader的监听端口
    HcclResult SetupAgentByMasterInfo(HcclIpAddress& localHostIp, const HcclRootHandle& rootInfo);
    HcclResult SetupServer(HcclRootHandle& rootInfo);
    HcclResult GroupLeaderAccept(
        HcclRankHandle& grpLeaderInfo, std::vector<HcclIpAddress> whitelist,
        std::shared_ptr<HcclSocket> grpLeaderToRoot);
    HcclResult GroupLeaderListen(HcclRankHandle& rankHandle, std::vector<HcclIpAddress>& whitelist);
    HcclResult SetupServerByMasterInfo(const HcclIpAddress& masterIP, u32 masterPort, const HcclRootHandle& rootInfo);
    HcclResult Teardown();
    HcclResult WaitComplete(const HcclRootHandle& rootInfo);
    HcclResult GetClusterInfo(RankTable_t& clusterInfo);
    HcclResult GetLocalRankInfo(HcclBasicRankInfo& rankInfo);
    HcclResult GetRankId(u32& rankId);
    HcclResult TransformRankTableStr(const RankTable_t& clusterInfo, std::string& ranktableStr);
    HcclResult GetServerConnections(std::map<u32, std::shared_ptr<HcclSocket>>& connectSockets);
    HcclResult GetAgentConnection(std::shared_ptr<HcclSocket>& connectSocket);
    HcclResult SendGroupLeaderPort(std::shared_ptr<HcclSocket>& connectSocket, HcclRankHandle& rankHandle);
    HcclResult GetAgentListenSocket(HcclSocketPortConfig& commPortConfig);
    HcclResult
    GenerateRootInfo(const HcclIpAddress& hostIP, u32 hostPort, u32 devicePhysicID, HcclRootHandle& rootInfo) const;
    HcclResult GetGroupLeader(HcclRankHandle& rankHandle);
    HcclResult SetIsInterSuperPodRetryEnable(bool isRetry);

protected:
    HcclResult StartListenNetwork(
        std::shared_ptr<HcclSocket>& listenSocket, HcclNetDevCtx netDevCtx, const HcclIpAddress& hostIP, u32& usePort,
        const std::vector<HcclSocketPortRange>& portRanges);
    HcclResult SetupRootServerNetwork(
        HcclIpAddress& hostIP, u32& hostPort, std::vector<HcclSocketPortRange>& portRanges,
        std::vector<HcclIpAddress>& whitelist);
    s32 deviceLogicID_;
    HcclIpAddress bootstrapHostIP_{};
    HcclNetDevCtx serverPortCtx_{nullptr};
    u32 devicePhysicID_{INVALID_UINT};
    std::shared_ptr<HcclSocket> listenSocket_{nullptr};
    HcclRootHandle rootInfo_;
    std::shared_ptr<TopoInfoExchangeServer> pTopoExchangeServer_{nullptr};
    std::unique_ptr<std::thread> exchangeServerThreadPtr_{nullptr};
    static UniversalConcurrentMap<u32, volatile u32> g_topoExchangeServerStatus_;

private:
    HcclResult CheckHostNicFlags(const std::string& ifName, u32 flags) const;
    HcclResult TeardownAgent();
    HcclResult TeardownServer();
    HcclResult Struct2JsonRankTable(const RankTable_t& clusterInfo, nlohmann::json& ClusterJson);
    HcclResult GetRootHostIP(const std::vector<HcclIpAddress>& whitelist, HcclIpAddress& ip, u32 devPhyId);
    HcclResult StartNetwork(HcclIpAddress& hostIP, bool bInitDevNic);
    HcclResult StopNetwork(HcclIpAddress& hostIP, bool bInitDevNic);
    HcclResult
    StartRootNetwork(const HcclIpAddress& hostIP, u32& usePort, const std::vector<HcclSocketPortRange>& portRanges);
    HcclResult StartGroupLeaderNetwork(
        const std::vector<HcclIpAddress>& whitelist, const HcclIpAddress& hostIP, u32& bindPort,
        const std::vector<HcclSocketPortRange>& portRanges);
    HcclResult AddSocketWhiteList(u32 port, const std::vector<HcclIpAddress>& whitelist) const;
    HcclResult GenerateLocalRankInfo(u32 rankSize, u32 rankID, HcclBasicRankInfo& localRankInfo);
    HcclResult GetSuperPodInfo(s32 deviceLogicId, std::string& superPodId, u32& superDeviceId);
    HcclResult ReadHostSocketWhitelist(std::vector<HcclIpAddress>& whitelist) const;
    HcclResult GetAllHostIfInfos(std::vector<std::pair<std::string, HcclIpAddress>>& ifInfos, u32 devPhyId) const;
    HcclResult GetAllValidHostIfInfos(
        const std::vector<HcclIpAddress>& whitelist, std::vector<std::pair<std::string, HcclIpAddress>>& ifInfos,
        u32 devPhyId) const;
    HcclResult TransformDeviceList(
        const RankTable_t& clusterInfo, std::vector<RankInfo_t>& tmpRankList, nlohmann::json& perServerJson,
        u32 serverIndex) const;
    HcclResult TransformSuperPodList(const std::vector<RankInfo_t>& rankInfo, nlohmann::json& superPodListJson) const;
    void SetupTopoExchangeServer(
        s32 devicePhysicID, s32 deviceLogicID, HcclIpAddress hostIP, u32 hostPort, std::vector<HcclIpAddress> whitelist,
        HcclNetDevCtx netDevCtx, std::shared_ptr<HcclSocket> listenSocket, bool isMasterInfo = false);
    void SetupTopoGroupLeader(
        s32 devicePhysicID, s32 deviceLogicID, HcclIpAddress hostIP, u32 hostPort, std::vector<HcclIpAddress> whitelist,
        HcclNetDevCtx netDevCtx, std::shared_ptr<HcclSocket> listenSocket, std::shared_ptr<HcclSocket> grpLeaderToRoot,
        bool isMasterInfo = false);
    HcclResult WaitTopoExchangeServerCompelte(u32 idx) const;
    void SetBootstrapHostIP(HcclIpAddress& ip);
    HcclIpAddress GetBootstrapHostIP() const;
    HcclResult
    FilterDevIPs(std::vector<HcclIpAddress>& sourceDeviceIPs, std::vector<HcclIpAddress>& targetDeviceIPs) const;
    HcclResult CalcGroupSizeAndRank(const u32 nRanks, const u32 rank, u32& groupSize, u32& groupRank);
    HcclResult
    PreemptDeviceNicPort(const u32 devPhyId, const s32 devLogicId, const HcclIpAddress& deviceIp, u32& usePort);
    HcclResult PreemptBackupDeviceNicPort(
        const u32 devPhyId, const s32 devLogicId, const HcclIpAddress& deviceIp, const HcclIpAddress& backupDeviceIp,
        u32& usePort);
    HcclResult PreemptDeviceVnicPort(HcclBasicRankInfo& localRankInfo);
    HcclResult GetDeviceBackupNicInfo(HcclBasicRankInfo& localRankInfo);
    HcclBasicRankInfo localRankInfo_;
    RankTable_t clusterTopoInfo_;
    u32 identifierNum_;
    HcclNetDevCtx agentPortCtx_{nullptr};
    HcclNetDevCtx devNicCtx_{nullptr};
    HcclSocketPortConfig commPortConfig_;
    std::shared_ptr<hccl::TopoInfoExchangeAgent> pTopoExchangeAgent_{nullptr};
    HcclRankHandle grpLeader_;
    bool isInterSuperPodRetryEnable_;
    CommConfig commConfig_;
};
} // namespace hccl
#endif /* TOPOINFO_DETECT_H */
