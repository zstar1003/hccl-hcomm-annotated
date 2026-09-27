/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_EXCHANGE_AGENT_H
#define TOPOINFO_EXCHANGE_AGENT_H

#include <vector>
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "topoinfo_struct.h"
#include "topoinfo_exchange_base.h"
#include "comm.h"
#include "hccl_socket.h"
#include "hccl_network_pub.h"

namespace hccl {
using HcclBasicRankInfo = struct HcclBasicRankInfoDef {
    HcclIpAddress hostIP;
    u32 hostPort{HCCL_INVALID_PORT};
    u32 rank{0};
    u32 rankSize{0};
    NICDeployment nicDeploy{NICDeployment::NIC_DEPLOYMENT_DEVICE};
    DevType deviceType{DevType::DEV_TYPE_910};
    s32 deviceLogicID{0};
    u32 devicePhysicID{0};
    std::vector<HcclIpAddress> deviceIP;
    std::vector<HcclIpAddress> backupDeviceIP;
    u32 deviceNicPort{HCCL_INVALID_PORT};
    u32 deviceVnicPort{HCCL_INVALID_PORT};
    u32 backupDevicePort{HCCL_INVALID_PORT};
    u32 superDeviceId{INVALID_UINT}; // 超节点内device id，超节点内唯一
    std::string superPodId{""};      // 超节点标识
    TlsStatus tlsStatus{TlsStatus::UNKNOWN};
};

class TopoInfoExchangeAgent : public TopoInfoExchangeBase {
public:
    explicit TopoInfoExchangeAgent(
        HcclIpAddress& serverIp, u32 serverPort, std::string identifier, HcclNetDevCtx netDevCtx,
        HcclBasicRankInfo localRankInfo);
    explicit TopoInfoExchangeAgent(
        HcclIpAddress& serverIp, u32 serverPort, std::string identifier, HcclNetDevCtx netDevCtx,
        HcclBasicRankInfo localRankInfo, u32 connSize, u32 connRank);
    explicit TopoInfoExchangeAgent(
        HcclIpAddress& serverIp, u32 serverPort, std::string identifier, HcclNetDevCtx netDevCtx,
        HcclBasicRankInfo localRankInfo, HcclRankHandle rankInfo);
    ~TopoInfoExchangeAgent() override;
    HcclResult SetupMember();
    HcclResult Setup();
    HcclResult SetupRank(std::shared_ptr<HcclSocket> socket);
    HcclResult SetupByMasterInfo();
    HcclResult Teardown();
    HcclResult GetClusterTopoInfo(RankTable_t& clusterInfo);
    HcclResult GetIdentifier(u32& identify);
    HcclResult GetConnection(std::shared_ptr<HcclSocket>& socket);
    HcclResult GetGroupLeader(HcclRankHandle& rankHandle);
    HcclResult SendGroupLeaderPortInfo(std::shared_ptr<HcclSocket> socket, HcclRankHandle& rankHandle);
    HcclResult SetIsInterSuperPodRetryEnable(bool isInterSuperPodRetryEnable);
    HcclResult SetIsScalable(bool isScalable);

private:
    HcclResult DetectClusterTopoInfo(std::shared_ptr<HcclSocket> socket, RankTable_t& clusterTopoInfo);
    HcclResult Connect(HcclIpAddress& serverIp, u32 port, std::shared_ptr<HcclSocket>& socket);
    HcclResult ConnectWithRetry(HcclIpAddress& serverIp, u32 port, std::shared_ptr<HcclSocket>& socket);
    HcclResult TryRecvFromServer(std::shared_ptr<HcclSocket>& socket, u32 retryTime);
    HcclResult GetConnection(HcclIpAddress& serverIp, u32 port, std::shared_ptr<HcclSocket>& socket);
    HcclResult Disconnect(std::shared_ptr<HcclSocket>& socket);
    HcclResult RecvGrpLeaderInfo(std::shared_ptr<HcclSocket> socket, GroupLeader_t& leaderInfo);
    HcclResult SetServerIdx(RankTable_t& clusterInfo) const;
    HcclResult GroupSuperPodsByRankContinuity(RankTable_t& clusterInfo) const;
    HcclResult SetSuperPodIdx(RankTable_t& clusterInfo) const;
    HcclResult GenerateLocalRankInfo(u32 rankSize, u32 rankID);
    void GenerateAgentID(HcclBasicRankInfo& localRankInfo, std::string& agentID);
    HcclResult ConstructRankTableMsg(RankTable_t& clusterInfo);
    void ConstructRankTableServerId(std::string& serverId);
    HcclResult SetTransportInfo(RankTable_t& clusterInfo);
    HcclResult VerifyClusterInfo(RankTable_t& clusterInfo);
    HcclResult VerifyClusterDeviceIP(const RankTable_t& clusterInfo);
    HcclResult VerifyClusterBackupDeviceIP(RankTable_t& clusterInfo);
    HcclResult VerifyClusterRankID(const RankTable_t& clusterInfo) const;
    HcclResult VerifyServerDevicePhysicID(const std::vector<RankInfo_t>& serverInfo) const;
    HcclResult VerifyClusterSuperPodInfo(const std::vector<RankInfo_t>& rankInfo) const;
    HcclResult VerifyClusterTlsConsistency(const RankTable_t& clusterInfo);
    void AddRankInfoToTlsStatusMap(
        const RankInfo_t& rankInfo, std::unordered_map<std::string, std::vector<u32>>& tlsStatusRankMap) const;
    void GenerateTlsStatusStr(
        std::string& tlsStatusStr, const std::unordered_map<std::string, std::vector<u32>>& tlsStatusRankMap) const;
    void ReportTlsConfigurationError(
        const std::string& tlsInconsistentTlsType, const std::string& tlsInconsistentEnableStr,
        const std::string& tlsInconsistentDisableStr, const std::string& tlsUnknownRankStr) const;
    void PrintSocketTimeoutReasons(HcclIpAddress& serverIp, u32 port, std::shared_ptr<HcclSocket>& socket) const;

    bool HasRepeatedIP(const std::vector<HcclIpAddress>& deviceAIP, const std::vector<HcclIpAddress>& deviceBIP) const;
    HcclResult DetectTransportType(
        const RankInfo_t& localRankInfo, const RankInfo_t& remoteRankInfo, TransportType& transportType) const;
    std::string Dec2Hex(s32 i, u32 width);
    HcclIpAddress serverIP_;
    u32 serverPort_;
    std::string identifier_;
    SocketHandle socketHandle_;
    HcclBasicRankInfo localRankInfo_;
    HcclRankHandle localRankHandle_;
    RankTable_t clusterTopoInfo_;
    GroupLeader_t grpLeaderInfo_;
    HcclRankHandle grpLeader_;
    HcclNetDevCtx netDevCtx_{nullptr};
    std::shared_ptr<HcclSocket> socket_;
    u32 connSize_;
    u32 connRank_;
    bool isRetry_;
    bool isScalable_{false};
};
} // namespace hccl

#endif /* TOPOINFO_EXCHANGE_AGENT_H */
