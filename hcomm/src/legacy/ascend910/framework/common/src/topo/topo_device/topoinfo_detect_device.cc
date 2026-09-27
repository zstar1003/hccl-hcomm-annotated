/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_detect.h"
#include <string>
#include "adapter_rts_common.h"
#include "hccl_whitelist.h"
#include "hccl_socket.h"
#include "sal_pub.h"
#include "device_capacity.h"
#include "preempt_port_manager.h"

using namespace std;
namespace hccl {
UniversalConcurrentMap<u32, volatile u32> TopoInfoDetect::g_topoExchangeServerStatus_;

TopoInfoDetect::TopoInfoDetect()
    : deviceLogicID_(INVALID_INT),
      localRankInfo_(),
      clusterTopoInfo_(),
      isInterSuperPodRetryEnable_(GetExternalInputInterSuperPodRetryEnable())
{}

TopoInfoDetect::~TopoInfoDetect() {}

HcclResult
TopoInfoDetect::GetServerConnections([[maybe_unused]] std::map<u32, std::shared_ptr<HcclSocket>>& connectSockets)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetAgentListenSocket([[maybe_unused]] HcclSocketPortConfig& commPortConfig)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetAgentConnection([[maybe_unused]] std::shared_ptr<HcclSocket>& connectSocket)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::SendGroupLeaderPort(
    [[maybe_unused]] std::shared_ptr<HcclSocket>& connectSocket, [[maybe_unused]] HcclRankHandle& rankHandle)
{
    return HCCL_E_NOT_SUPPORT;
}

void TopoInfoDetect::SetupTopoGroupLeader(
    [[maybe_unused]] s32 devicePhysicID, [[maybe_unused]] s32 deviceLogicID, [[maybe_unused]] HcclIpAddress hostIP,
    [[maybe_unused]] u32 hostPort, [[maybe_unused]] vector<HcclIpAddress> whitelist,
    [[maybe_unused]] HcclNetDevCtx netDevCtx, [[maybe_unused]] std::shared_ptr<HcclSocket> listenSocket,
    [[maybe_unused]] std::shared_ptr<HcclSocket> grpLeaderToRoot, [[maybe_unused]] bool isMasterInfo)
{
    return;
}

void TopoInfoDetect::SetupTopoExchangeServer(
    [[maybe_unused]] s32 devicePhysicID, [[maybe_unused]] s32 deviceLogicID, [[maybe_unused]] HcclIpAddress hostIP,
    [[maybe_unused]] u32 hostPort, [[maybe_unused]] vector<HcclIpAddress> whitelist,
    [[maybe_unused]] HcclNetDevCtx netDevCtx, [[maybe_unused]] std::shared_ptr<HcclSocket> listenSocket,
    [[maybe_unused]] bool isMasterInfo)
{
    return;
}
HcclResult TopoInfoDetect::SetupServerByMasterInfo(
    [[maybe_unused]] const HcclIpAddress& masterIP, [[maybe_unused]] u32 masterPort,
    [[maybe_unused]] const HcclRootHandle& rootInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::SetupServer([[maybe_unused]] HcclRootHandle& rootInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::GroupLeaderListen(
    [[maybe_unused]] HcclRankHandle& rankHandle, [[maybe_unused]] vector<HcclIpAddress>& whitelist)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GroupLeaderAccept(
    [[maybe_unused]] HcclRankHandle& grpLeaderInfo, [[maybe_unused]] vector<HcclIpAddress> whitelist,
    [[maybe_unused]] std::shared_ptr<HcclSocket> grpLeaderToRoot)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GenerateRootInfo(
    [[maybe_unused]] const HcclIpAddress& hostIP, [[maybe_unused]] u32 hostPort, [[maybe_unused]] u32 devicePhysicID,
    [[maybe_unused]] HcclRootHandle& rootInfo) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::CalcGroupSizeAndRank(
    [[maybe_unused]] const u32 nRanks, [[maybe_unused]] const u32 rank, [[maybe_unused]] u32& groupSize,
    [[maybe_unused]] u32& groupRank)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::SetupGroupMember(
    [[maybe_unused]] u32 rankSize, [[maybe_unused]] u32 myrank, [[maybe_unused]] const HcclRootHandle& rootInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::TeardownServer() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::WaitTopoExchangeServerCompelte([[maybe_unused]] u32 idx) const { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::PrepareHandle(
    [[maybe_unused]] HcclRankHandle& rankHandle, [[maybe_unused]] std::vector<HcclIpAddress>& whitelist)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::SetupAgent(
    [[maybe_unused]] u32 rankSize, [[maybe_unused]] u32 myrank, [[maybe_unused]] const HcclRootHandle& rootInfo,
    [[maybe_unused]] const HcclRankHandle& rankHandle, [[maybe_unused]] const CommConfig& commConfig,
    [[maybe_unused]] bool isScalable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::CheckHostNicLinkUp([[maybe_unused]] const HcclIpAddress& hostIP) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::SetupRank([[maybe_unused]] std::shared_ptr<HcclSocket>& agentConnRoot)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::TeardownAgent() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::SetupAgentByMasterInfo(
    [[maybe_unused]] HcclIpAddress& localHostIp, [[maybe_unused]] const HcclRootHandle& rootInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::WaitComplete([[maybe_unused]] const HcclRootHandle& rootInfo) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::Teardown() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::ReadHostSocketWhitelist([[maybe_unused]] vector<HcclIpAddress>& whitelist) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetAllHostIfInfos(
    [[maybe_unused]] vector<pair<string, HcclIpAddress>>& ifInfos, [[maybe_unused]] u32 devPhyId) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetAllValidHostIfInfos(
    [[maybe_unused]] const vector<HcclIpAddress>& whitelist,
    [[maybe_unused]] vector<pair<string, HcclIpAddress>>& ifInfos, [[maybe_unused]] u32 devPhyId) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetRootHostIP(
    [[maybe_unused]] const vector<HcclIpAddress>& whitelist, [[maybe_unused]] HcclIpAddress& ip,
    [[maybe_unused]] u32 devPhyId)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetGroupLeader([[maybe_unused]] HcclRankHandle& rankHandle) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::SetIsInterSuperPodRetryEnable([[maybe_unused]] bool isRetry) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoDetect::StartRootNetwork(
    [[maybe_unused]] const HcclIpAddress& hostIP, [[maybe_unused]] u32& usePort,
    [[maybe_unused]] const std::vector<HcclSocketPortRange>& portRanges)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::StartListenNetwork(
    [[maybe_unused]] std::shared_ptr<HcclSocket>& listenSocket, [[maybe_unused]] HcclNetDevCtx netDevCtx,
    [[maybe_unused]] const HcclIpAddress& hostIP, [[maybe_unused]] u32& usePort,
    [[maybe_unused]] const std::vector<HcclSocketPortRange>& portRanges)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::SetupRootServerNetwork(
    [[maybe_unused]] HcclIpAddress& hostIP, [[maybe_unused]] u32& hostPort,
    [[maybe_unused]] std::vector<HcclSocketPortRange>& portRanges,
    [[maybe_unused]] std::vector<HcclIpAddress>& whitelist)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::StartGroupLeaderNetwork(
    [[maybe_unused]] const vector<HcclIpAddress>& whitelist, [[maybe_unused]] const HcclIpAddress& hostIP,
    [[maybe_unused]] u32& bindPort, [[maybe_unused]] const std::vector<HcclSocketPortRange>& portRanges)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::AddSocketWhiteList(
    [[maybe_unused]] u32 port, [[maybe_unused]] const vector<HcclIpAddress>& whitelist) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::StartNetwork([[maybe_unused]] HcclIpAddress& hostIP, [[maybe_unused]] bool bInitDevNic)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::StopNetwork([[maybe_unused]] HcclIpAddress& hostIP, [[maybe_unused]] bool bInitDevNic)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::FilterDevIPs(
    [[maybe_unused]] std::vector<HcclIpAddress>& sourceDeviceIPs,
    [[maybe_unused]] std::vector<HcclIpAddress>& targetDeviceIPs) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::PreemptDeviceNicPort(
    [[maybe_unused]] const u32 devPhyId, [[maybe_unused]] const s32 devLogicId,
    [[maybe_unused]] const HcclIpAddress& deviceIp, [[maybe_unused]] u32& usePort)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::PreemptDeviceVnicPort([[maybe_unused]] HcclBasicRankInfo& localRankInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::PreemptBackupDeviceNicPort(
    [[maybe_unused]] const u32 devPhyId, [[maybe_unused]] const s32 devLogicId,
    [[maybe_unused]] const HcclIpAddress& deviceIp, [[maybe_unused]] const HcclIpAddress& backupDeviceIp,
    [[maybe_unused]] u32& usePort)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetDeviceBackupNicInfo([[maybe_unused]] HcclBasicRankInfo& localRankInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GenerateLocalRankInfo(
    [[maybe_unused]] u32 rankSize, [[maybe_unused]] u32 rankID, [[maybe_unused]] HcclBasicRankInfo& localRankInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetSuperPodInfo(
    [[maybe_unused]] s32 deviceLogicId, [[maybe_unused]] std::string& superPodId, [[maybe_unused]] u32& superDeviceId)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::GetClusterInfo([[maybe_unused]] RankTable_t& clusterInfo) { return HCCL_E_NOT_SUPPORT; }
HcclResult TopoInfoDetect::GetRankId(u32& rankId)
{
    rankId = identifierNum_;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetLocalRankInfo([[maybe_unused]] HcclBasicRankInfo& rankInfo) { return HCCL_E_NOT_SUPPORT; }

void TopoInfoDetect::SetBootstrapHostIP(HcclIpAddress& ip) { bootstrapHostIP_ = ip; }

HcclIpAddress TopoInfoDetect::GetBootstrapHostIP() const { return bootstrapHostIP_; }
HcclResult TopoInfoDetect::TransformRankTableStr(
    [[maybe_unused]] const RankTable_t& clusterInfo, [[maybe_unused]] string& ranktableStr)
{
    return HCCL_E_NOT_SUPPORT;
}
HcclResult TopoInfoDetect::TransformDeviceList(
    [[maybe_unused]] const RankTable_t& clusterInfo, [[maybe_unused]] vector<RankInfo_t>& tmpRankList,
    [[maybe_unused]] nlohmann::json& perServerJson, [[maybe_unused]] u32 serverIndex) const
{
    return HCCL_E_NOT_SUPPORT;
}
HcclResult TopoInfoDetect::Struct2JsonRankTable(
    [[maybe_unused]] const RankTable_t& clusterInfo, [[maybe_unused]] nlohmann::json& ClusterJson)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoDetect::TransformSuperPodList(
    [[maybe_unused]] const std::vector<RankInfo_t>& rankInfo, [[maybe_unused]] nlohmann::json& superPodListJson) const
{
    return HCCL_E_NOT_SUPPORT;
}
} // namespace hccl
