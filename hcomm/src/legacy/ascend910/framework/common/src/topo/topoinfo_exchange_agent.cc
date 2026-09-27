/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_exchange_agent.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include "externalinput_pub.h"
#include "adapter_error_manager_pub.h"
#include "config.h"
#include "sal_pub.h"
#include "device_capacity.h"

namespace hccl {
constexpr s32 DEVICE_LOGIC_ID_LENGTH = 4;
constexpr u32 AGENT_MAX_RETRY_TIME = 3;

TopoInfoExchangeAgent::TopoInfoExchangeAgent(
    HcclIpAddress& serverIp, u32 serverPort, std::string identifier, HcclNetDevCtx netDevCtx,
    HcclBasicRankInfo localRankInfo)
    : serverIP_(serverIp),
      serverPort_(serverPort),
      identifier_(identifier),
      localRankInfo_(localRankInfo),
      clusterTopoInfo_(),
      netDevCtx_(netDevCtx),
      isRetry_(GetExternalInputInterSuperPodRetryEnable())
{}

TopoInfoExchangeAgent::TopoInfoExchangeAgent(
    HcclIpAddress& serverIp, u32 serverPort, std::string identifier, HcclNetDevCtx netDevCtx,
    HcclBasicRankInfo localRankInfo, u32 connSize, u32 connRank)
    : serverIP_(serverIp),
      serverPort_(serverPort),
      identifier_(identifier),
      localRankInfo_(localRankInfo),
      clusterTopoInfo_(),
      netDevCtx_(netDevCtx),
      connSize_(connSize),
      connRank_(connRank),
      isRetry_(GetExternalInputInterSuperPodRetryEnable())
{}

TopoInfoExchangeAgent::TopoInfoExchangeAgent(
    HcclIpAddress& serverIp, u32 serverPort, std::string identifier, HcclNetDevCtx netDevCtx,
    HcclBasicRankInfo localRankInfo, HcclRankHandle rankInfo)
    : serverIP_(serverIp),
      serverPort_(serverPort),
      identifier_(identifier),
      localRankInfo_(localRankInfo),
      localRankHandle_(rankInfo),
      clusterTopoInfo_(),
      netDevCtx_(netDevCtx),
      isRetry_(GetExternalInputInterSuperPodRetryEnable())
{}

TopoInfoExchangeAgent::~TopoInfoExchangeAgent() { Teardown(); }

HcclResult TopoInfoExchangeAgent::SetIsInterSuperPodRetryEnable(bool isInterSuperPodRetryEnable)
{
    isRetry_ = isInterSuperPodRetryEnable;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::SetIsScalable(bool isScalable)
{
    isScalable_ = isScalable;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::Setup()
{
    connSize_ = localRankInfo_.rankSize;
    connRank_ = localRankInfo_.rank;
    // 填充要发送的localRankHandle的值
    localRankHandle_.rankId = localRankInfo_.rank;
    HcclResult ret = ConnectWithRetry(serverIP_, serverPort_, socket_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TopoInfoExchangeAgent][Setup]TopoExchangeAgent: "
            "connect server[%s : %u] failed",
            serverIP_.GetReadableAddress(), serverPort_),
        ret);
    HCCL_INFO(
        "TopoExchangeAgent: client connect with server ip[%s] port[%u] success.", serverIP_.GetReadableAddress(),
        serverPort_);

    if (!isByMasterInfo_ && !isScalable_ && localRankInfo_.rankSize > TOPO_HIERARCHICAL_ENABLE_THRESHOLD) {
        ret = socket_->Send(&localRankHandle_, sizeof(localRankHandle_));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SendRankHandle]errNo[0x%016llx] rankID[%s] send localRankHandle to remote by"
                "client fdHandle failed, ret[%u]",
                HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), localRankInfo_.rank, ret),
            ret);

        CHK_RET(RecvGrpLeaderInfo(socket_, grpLeaderInfo_));
        u32 grpIndex = localRankInfo_.rank / TOPO_MAX_GROUP_SIZE;
        grpLeader_ = grpLeaderInfo_.GroupLeaderList[grpIndex];
    } else {
        CHK_RET(DetectClusterTopoInfo(socket_, clusterTopoInfo_));
        ret = VerifyClusterInfo(clusterTopoInfo_);
        if (ret != HCCL_SUCCESS) {
            auto current = g_broadcastStage.load(std::memory_order_acquire);
            if (current == BroadcastStage::Started) {
                std::unique_lock<std::mutex> lock(g_broadcast_stage_mutex);
                std::chrono::seconds timeout(MAX_WAIT_BROADCAST_SECONDS);
                g_broadcast_stage_cv.wait_for(lock, timeout, [] {
                    return g_broadcastStage.load(std::memory_order_relaxed) == BroadcastStage::Completed;
                });
            }
            HCCL_ERROR(
                "[TopoInfoExchangeAgent][Setup]VerifyClusterInfo failed, g_broadcastStage[%d]",
                g_broadcastStage.load());
        }

        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::SetupRank(std::shared_ptr<HcclSocket> socket)
{
    CHK_RET(RecvGrpLeaderInfo(socket, grpLeaderInfo_));
    u32 grpIndex = localRankInfo_.rank / TOPO_MAX_GROUP_SIZE;
    grpLeader_ = grpLeaderInfo_.GroupLeaderList[grpIndex];
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::SetupMember()
{
    HcclResult ret = Connect(serverIP_, serverPort_, socket_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TopoInfoExchangeAgent][Setup]SetupGroupMember: "
            "connect server[%s : %u] failed",
            serverIP_.GetReadableAddress(), serverPort_),
        ret);
    HCCL_INFO(
        "SetupGroupMember: client connect with server ip[%s] port[%u] success.", serverIP_.GetReadableAddress(),
        serverPort_);

    CHK_RET(DetectClusterTopoInfo(socket_, clusterTopoInfo_));

    CHK_RET(VerifyClusterInfo(clusterTopoInfo_));

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::Teardown()
{
    CHK_RET(Disconnect(socket_));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::GetConnection(std::shared_ptr<HcclSocket>& socket)
{
    socket = socket_;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::GetGroupLeader(HcclRankHandle& rankHandle)
{
    rankHandle = grpLeader_;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::SetupByMasterInfo()
{
    isByMasterInfo_ = true;
    CHK_RET(Setup());
    return HCCL_SUCCESS;
}

HcclResult
TopoInfoExchangeAgent::DetectClusterTopoInfo(std::shared_ptr<HcclSocket> socket, RankTable_t& clusterTopoInfo)
{
    RankTable_t localBasicInfo;
    CHK_RET(ConstructRankTableMsg(localBasicInfo));
    CHK_RET(SendClusterInfo(socket, localBasicInfo));
    HCCL_INFO("topo exchange client send rank basic info success.");

    CHK_RET(RecvClusterInfo(socket, clusterTopoInfo));
    HCCL_INFO("topo exchange client get rank basic info success.");

    // 按照rankId排序
    std::vector<RankInfo_t>& rankList = clusterTopoInfo_.rankList;
    sort(rankList.begin(), rankList.end(), [](const RankInfo_t& a, const RankInfo_t& b) {
        return a.rankId < b.rankId;
    });

    CHK_RET(SetServerIdx(clusterTopoInfo));
    CHK_RET(GroupSuperPodsByRankContinuity(clusterTopoInfo));
    CHK_RET(SetSuperPodIdx(clusterTopoInfo));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::GroupSuperPodsByRankContinuity(RankTable_t& clusterInfo) const
{
    // 按照superPodId将节点分组，相同superPodId在一个组
    // clusterInfo已经按照rankId排好序，按顺序插入到新的subRankTable中，不需要再排序
    std::map<std::string, std::vector<RankInfo_t*>> podGroupClusters;
    for (auto& rankInfo : clusterInfo.rankList) {
        rankInfo.originalSuperPodId = rankInfo.superPodId; // 把用户配置的原始superPodId先保存下来
        podGroupClusters[rankInfo.superPodId].emplace_back(&rankInfo);
    }
    std::set<std::string> superPodIdSet;
    std::map<std::string, std::pair<u32, u32>> superPodIdRanges; // 记录每个逻辑超节点的rank id范围
    for (auto& subCluster : podGroupClusters) {
        auto& subClusterInfo = subCluster.second;
        if (subClusterInfo.size() <= 1) {
            continue;
        }
        u32 groupId = 0;
        superPodIdSet.insert(subCluster.first);
        RankInfo_t preRank = *(subClusterInfo[0]);
        superPodIdRanges[preRank.superPodId] = {preRank.rankId, preRank.rankId}; // 初始化范围
        for (u32 i = 1; i < subClusterInfo.size(); ++i) {
            RankInfo_t& curRank = *(subClusterInfo[i]);
            // 当前的curRank和上一个preRank的rankId不连续，分配新的逻辑超节点ID
            if (curRank.rankId != preRank.rankId + 1) {
                std::string newSuperPodId = curRank.originalSuperPodId + "_HCCLSPLIT_" + std::to_string(groupId);
                curRank.superPodId = newSuperPodId;
                groupId++;
                superPodIdRanges[curRank.superPodId] = {curRank.rankId, curRank.rankId}; // 初始化新的范围
            } else {
                // 同一个sub通信域两个rank原始逻辑超节点是一致的
                // rankId连续 上一个rank的superPodId可能已经重新分配，需要更新当前superPodId为上一个rank的
                curRank.superPodId = preRank.superPodId;
                superPodIdRanges[curRank.superPodId].second = curRank.rankId; // 更新最大rank id
            }
            superPodIdSet.insert(curRank.superPodId);
            preRank = curRank;
        }
    }
    // 打印每个逻辑超节点的rank id范围，只打印包含_HCCLSPLIT_的逻辑超节点
    for (const auto& entry : superPodIdRanges) {
        auto superPodId = entry.first;
        if (superPodId.find("_HCCLSPLIT_") != std::string::npos) {
            auto range = entry.second;
            HCCL_RUN_INFO(
                "[TopoInfoExchangeAgent][%s]Split superPod, ID[%s], rank range[%u, %u]", __func__, superPodId.c_str(),
                range.first, range.second);
        }
    }
    clusterInfo.superPodNum = superPodIdSet.size();
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::SetServerIdx(RankTable_t& clusterInfo) const
{
    struct ServerSortInfo {
        u32 serverPosition;
        u32 selectedRankId;
    };
    std::vector<ServerSortInfo> serverSortInfoVec;
    for (u32 i = 0; i < clusterInfo.serverList.size(); i++) {
        for (u32 j = 0; j < clusterInfo.rankList.size(); j++) {
            if (clusterInfo.rankList[j].serverId == clusterInfo.serverList[i].serverId) {
                // 每个server的rankid都是连续的，只需要取每个server里任意一个rankid进行排序
                ServerSortInfo serverSortInfo;
                serverSortInfo.serverPosition = i;
                serverSortInfo.selectedRankId = clusterInfo.rankList[j].rankId;
                serverSortInfoVec.push_back(serverSortInfo);
                break;
            }
        }
    }
    sort(serverSortInfoVec.begin(), serverSortInfoVec.end(), [](const ServerSortInfo& a, const ServerSortInfo& b) {
        return a.selectedRankId < b.selectedRankId;
    });
    // 遍历ranklist，根据serverid获取serveridx
    for (u32 serverIdx = 0; serverIdx < serverSortInfoVec.size(); serverIdx++) {
        for (u32 j = 0; j < clusterInfo.rankList.size(); j++) {
            if (clusterInfo.rankList[j].serverId
                == clusterInfo.serverList[serverSortInfoVec[serverIdx].serverPosition].serverId) {
                clusterInfo.rankList[j].serverIdx = serverIdx;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::SetSuperPodIdx(RankTable_t& clusterInfo) const
{
    std::map<std::string, u32> spodIdToIdx;
    bool isDiffDeviceType = false;
    DevType standardDevType = DevType::DEV_TYPE_NOSOC;
    if (clusterInfo.rankList.size() > 0) {
        standardDevType = clusterInfo.rankList[0].deviceInfo.deviceType;
    }
    for (u32 i = 0; i < clusterInfo.rankList.size(); ++i) {
        RankInfo_t& rankInfo = clusterInfo.rankList[i];
        if (rankInfo.deviceInfo.deviceType != standardDevType) {
            isDiffDeviceType = true;
        }

        if (isDiffDeviceType) {
            rankInfo.superPodIdx = spodIdToIdx.size();
        } else if (spodIdToIdx.find(rankInfo.superPodId) == spodIdToIdx.end()) {
            rankInfo.superPodIdx = spodIdToIdx.size();
            spodIdToIdx.insert({rankInfo.superPodId, rankInfo.superPodIdx});
        } else if (spodIdToIdx[rankInfo.superPodId] + 1 == spodIdToIdx.size()) {
            rankInfo.superPodIdx = spodIdToIdx[rankInfo.superPodId];
        } else {
            u32 preIndex = (i > 0) ? i - 1 : i;
            RankInfo_t& preRankInfo = clusterInfo.rankList[preIndex];
            u32 index = 0;
            for (; index < preIndex; index++) {
                RankInfo_t& tmpRankInfo = clusterInfo.rankList[index];
                if (tmpRankInfo.superPodId == rankInfo.superPodId) {
                    break;
                }
            }
            // 超节点内rank id不连续
            HCCL_RUN_WARNING(
                "rank in superPodId is not continuous, pre: rank[%u] superPodId[%s], "
                "cur: rank[%u] superPodId[%s], ",
                preRankInfo.rankId, preRankInfo.superPodId.c_str(), rankInfo.rankId, rankInfo.superPodId.c_str());
            rankInfo.superPodIdx = spodIdToIdx[rankInfo.superPodId];
        }
        HCCL_INFO(
            "SetSuperPodIdx rankList[%u]: rankId[%u], superPodId[%s], superPodIdx[%u], sdid[%u]", i, rankInfo.rankId,
            rankInfo.superPodId.c_str(), rankInfo.superPodIdx, rankInfo.superDeviceId);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::GetClusterTopoInfo(RankTable_t& clusterInfo)
{
    clusterInfo.nicDeploy = clusterTopoInfo_.nicDeploy;
    clusterInfo.deviceNum = clusterTopoInfo_.deviceNum;
    clusterInfo.serverNum = clusterTopoInfo_.serverNum;
    clusterInfo.superPodNum = clusterTopoInfo_.superPodNum;
    clusterInfo.rankNum = clusterTopoInfo_.rankNum;
    clusterInfo.rankList = clusterTopoInfo_.rankList;
    clusterInfo.serverList = clusterTopoInfo_.serverList;

    return HCCL_SUCCESS;
}
HcclResult TopoInfoExchangeAgent::GetIdentifier(u32& identify)
{
    identify = identifierNum_;
    return HCCL_SUCCESS;
}
HcclResult TopoInfoExchangeAgent::Connect(HcclIpAddress& serverIp, u32 port, std::shared_ptr<HcclSocket>& socket)
{
    std::string tag = TOPO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(port);
    EXCEPTION_CATCH(
        (socket = std::make_shared<HcclSocket>(tag, netDevCtx_, serverIp, port, HcclSocketRole::SOCKET_ROLE_CLIENT)),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(socket);
    CHK_RET(socket->Init());
    CHK_RET(socket->Connect());

    return GetConnection(serverIp, port, socket);
}

HcclResult
TopoInfoExchangeAgent::ConnectWithRetry(HcclIpAddress& serverIp, u32 port, std::shared_ptr<HcclSocket>& socket)
{
    u32 retryTime = 1;
    HcclResult ret = HCCL_SUCCESS;
    while (retryTime <= AGENT_MAX_RETRY_TIME) {
        std::string tag = TOPO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(port);
        EXCEPTION_CATCH(
            (socket
             = std::make_shared<HcclSocket>(tag, netDevCtx_, serverIp, port, HcclSocketRole::SOCKET_ROLE_CLIENT)),
            return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(socket);
        CHK_RET(socket->Init());
        CHK_RET(socket->Connect());

        CHK_RET(GetConnection(serverIp, port, socket));

        ret = TryRecvFromServer(socket, retryTime);
        if (ret == HCCL_SUCCESS) {
            break;
        } else {
            retryTime++;
        }
    }
    return ret;
}

HcclResult TopoInfoExchangeAgent::TryRecvFromServer(std::shared_ptr<HcclSocket>& socket, u32 retryTime)
{
    // client端获取socket之后尝试从server接收数据，若在一定时间内没有接收到，则重新发起建链请求
    u32 timeout = GetExternalInputHcclLinkTimeOut() / AGENT_MAX_RETRY_TIME;
    char recvMsgBuf[sizeof(TOPO_EXCHANGE_CHECK_MESSAGE)] = {0};
    auto ret = HCCL_SUCCESS;
    if (retryTime == AGENT_MAX_RETRY_TIME) {
        ret = socket->Recv(recvMsgBuf, sizeof(TOPO_EXCHANGE_CHECK_MESSAGE), timeout);
    } else {
        // 重试时打印RUN_WARN日志
        SetErrToWarnSwitch(true);
        ret = socket->Recv(recvMsgBuf, sizeof(TOPO_EXCHANGE_CHECK_MESSAGE), timeout);
        SetErrToWarnSwitch(false);
    }

    if (ret == HCCL_SUCCESS) {
        HCCL_RUN_INFO("[%s]recvMes %s", __func__, recvMsgBuf);
        // 校验收到的是否正确，server端使用的是固定消息
        if (strncmp(recvMsgBuf, TOPO_EXCHANGE_CHECK_MESSAGE, sizeof(TOPO_EXCHANGE_CHECK_MESSAGE)) != 0) {
            HCCL_ERROR(
                "[%s]recv message check failed, expect [%s], but recv [%s]", __func__, TOPO_EXCHANGE_CHECK_MESSAGE,
                recvMsgBuf);
            return HCCL_E_INTERNAL;
        }
    } else if (retryTime < AGENT_MAX_RETRY_TIME) {
        HCCL_RUN_WARNING("[%s]client recv from server failed, will try to connect with server again.", __func__);
    } else {
        HCCL_ERROR("[%s]failed to recv messages from server with %u times", __func__, AGENT_MAX_RETRY_TIME);
    }

    return ret;
}

void TopoInfoExchangeAgent::PrintSocketTimeoutReasons(
    HcclIpAddress& serverIp, u32 port, std::shared_ptr<HcclSocket>& socket) const
{
    HCCL_ERROR("current rank connect to server timeout, maybe due to following reasons:");
    HCCL_ERROR(
        "1. local host ip is [%s], server host ip and port is [%s:%u], Please check the network connectivity. "
        "If it is not connected, modify the network configuration or use HCCL_SOCKET_IFNAME and HCCL_IF_BASE_PORT to "
        "specify ifname and server port.",
        socket->GetLocalIp().GetReadableIP(), serverIp.GetReadableIP(), port);
    HCCL_ERROR(
        "2. Check whether any other exceptions have occurred on server[%s] or "
        "whether the time difference between the execution of hcom on ranks exceeds the timeout threshold.",
        serverIp.GetReadableIP());
}

HcclResult TopoInfoExchangeAgent::GetConnection(HcclIpAddress& serverIp, u32 port, std::shared_ptr<HcclSocket>& socket)
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (true) {
        std::string errormessage = "1. The current node " + std::string(serverIp.GetReadableIP())
                                   + " is disconnected from the host of the root node "
                                   + std::string(localRankHandle_.ip)
                                   + ". "
                                     "2. the timeout set by the HCCL_CONNECT_TIMEOUT environment variable is too short";
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            RPT_INPUT_ERR(
                true, "EI0015", std::vector<std::string>({"error_reason"}), std::vector<std::string>({errormessage}));
            HCCL_ERROR(
                "[%s][%s] topo exchange agent get socket timeout! timeout[%lld s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), timeout);
            PrintSocketTimeoutReasons(serverIp, port, socket);
            sleep(WAIT_ERROR_BROADCAST_TIME);
            return HCCL_E_TIMEOUT;
        }
        HcclSocketStatus status = socket->GetStatus();
        if (status == HcclSocketStatus::SOCKET_CONNECTING) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else if (status != HcclSocketStatus::SOCKET_OK) {
            HCCL_ERROR("[Get][Connection]server: get socket failed ret[%d]", status);
            return HCCL_E_TCP_CONNECT;
        } else {
            HCCL_INFO("TopoInfoExchangeAgent get socket success.");
            std::string agentID;
            if (isByMasterInfo_) {
                agentID = localRankInfo_.superPodId + "/";
                GenerateAgentID(localRankInfo_, agentID);
            } else {
                std::string rankID = std::to_string(connRank_);
                agentID = std::string(16 - rankID.length(), '0') + rankID; // agent id为rank id，16位，左对齐补零
            }
            char agentBuf[MAX_AGENT_BUF_SIZE] = {0};
            s32 sRet = memcpy_s(agentBuf, sizeof(agentBuf), agentID.c_str(), agentID.size());
            CHK_PRT_RET(sRet != EOK, HCCL_ERROR("memcpy_s failed, errorno[%d]", sRet), HCCL_E_MEMORY);
            HcclResult ret = socket->Send(&agentBuf, sizeof(agentBuf));
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Get][Connection]errNo[0x%016llx] agentID[%s] send local rank id to remote "
                    "by client fdHandle failed, ret[%u]",
                    HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), agentBuf, ret),
                ret);
            ret = socket->Send(&connSize_, sizeof(connSize_));
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Get][Connection]errNo[0x%016llx] rank[%u] send local rank num[%u] to "
                    "remote by client fdHandle failed, ret[%u]",
                    HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), localRankInfo_.rank, localRankInfo_.rankSize, ret),
                ret);
            HCCL_INFO(
                "local rank[%u] get socket connection with server[%s] port[%u] success.", localRankInfo_.rank,
                serverIp.GetReadableAddress(), port);
            break;
        }
    }
    return HCCL_SUCCESS;
}

std::string TopoInfoExchangeAgent::Dec2Hex(s32 i, u32 width)
{
    std::string temp;
    std::stringstream ss;
    ss << std::hex << i;
    ss >> temp;
    if (width > temp.size()) {
        return std::string((width - temp.size()), '0') + temp;
    } else {
        HCCL_WARNING("Dec2Hex: length[%u] is over width[%u]", temp.size(), width);
    }
    return temp;
}

void TopoInfoExchangeAgent::GenerateAgentID(HcclBasicRankInfo& localRankInfo, std::string& agentID)
{
    struct in_addr addr = localRankInfo.hostIP.GetBinaryAddress().addr;
    struct in6_addr addr6 = localRankInfo.hostIP.GetBinaryAddress().addr6;
    if (localRankInfo.hostIP.IsIPv6()) {
        for (size_t i = 0; i < sizeof(addr6.s6_addr); i++) {
            agentID += Dec2Hex(addr6.s6_addr[i], 2); // 转换为2位十六进制数据，左对齐补零
        }
    } else {
        for (size_t i = 0; i < sizeof(addr.s_addr) / sizeof(u8); i++) {
            agentID += Dec2Hex(*(reinterpret_cast<u8*>(&addr.s_addr) + i), 2); // 转换为2位十六进制数据，左对齐补零
        }
    }
    agentID.append("/");
    std::string devID = std::to_string(localRankInfo.deviceLogicID);
    CHK_PRT_RET(devID.size() > DEVICE_LOGIC_ID_LENGTH, HCCL_ERROR("deviceLogicID[%s] is invalid", devID.c_str()), );
    // device id转换为4位十进制数字，左对齐补零
    agentID.append(std::string((DEVICE_LOGIC_ID_LENGTH - devID.size()), '0') + devID);
    HCCL_INFO("GenerateAgentID agentID[%s]", agentID.c_str());
    return;
}

HcclResult TopoInfoExchangeAgent::Disconnect(std::shared_ptr<HcclSocket>& socket)
{
    CHK_RET(DisconnectSocket(socket));
    socket = nullptr;

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::RecvGrpLeaderInfo(std::shared_ptr<HcclSocket> socket, GroupLeader_t& leaderInfo)
{
    // 每次获取之前先清空 保证填充之后的数据是最新的
    leaderInfo.grpLeaderNum = 0;
    leaderInfo.GroupLeaderList.clear();
    CHK_RET(RecvGrpLeaderInfoMsg(socket, leaderInfo));
    return HCCL_SUCCESS;
}

HcclResult
TopoInfoExchangeAgent::SendGroupLeaderPortInfo(std::shared_ptr<HcclSocket> socket, HcclRankHandle& rankHandle)
{
    CHK_RET(GetConnection(socket));
    HcclResult ret = socket->Send(&rankHandle, sizeof(rankHandle));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TopoInfoExchangeAgent][SendGroupLeaderPortInfo]errNo[0x%016llx] "
            "send grpleader port info fail",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::ConstructRankTableMsg(RankTable_t& clusterInfo)
{
    RankInfo_t myRankInfo;
    myRankInfo.rankId = localRankInfo_.rank;
    myRankInfo.hostIp = localRankInfo_.hostIP;
    myRankInfo.hostPort = localRankInfo_.hostPort;
    myRankInfo.deviceInfo.devicePhyId = localRankInfo_.devicePhysicID;
    myRankInfo.deviceInfo.deviceIp = localRankInfo_.deviceIP;
    myRankInfo.deviceInfo.deviceType = localRankInfo_.deviceType;
    myRankInfo.deviceInfo.backupDeviceIp = localRankInfo_.backupDeviceIP;
    myRankInfo.deviceInfo.port = localRankInfo_.deviceNicPort;
    myRankInfo.deviceInfo.vnicPort = localRankInfo_.deviceVnicPort;
    myRankInfo.deviceInfo.backupPort = localRankInfo_.backupDevicePort;
    myRankInfo.superPodId = localRankInfo_.superPodId;
    myRankInfo.superDeviceId = localRankInfo_.superDeviceId;
    myRankInfo.tlsStatus = localRankInfo_.tlsStatus;
    ConstructRankTableServerId(myRankInfo.serverId);

    ServerInfo_t myServerInfo;
    myServerInfo.serverId = myRankInfo.serverId;

    clusterInfo.nicDeploy = localRankInfo_.nicDeploy;
    clusterInfo.rankList.push_back(myRankInfo);
    clusterInfo.serverList.push_back(myServerInfo);
    return HCCL_SUCCESS;
}

void TopoInfoExchangeAgent::ConstructRankTableServerId(std::string& serverId)
{
    serverId = localRankInfo_.hostIP.GetReadableIP();
    // 配置逻辑超节点时, serverId要根据逻辑超节点划分
    if (localRankInfo_.deviceType == DevType::DEV_TYPE_910_93 && GetExternalInputLogicSuperPodId().empty() == false) {
        serverId += "_" + GetExternalInputLogicSuperPodId();
    }
    HCCL_INFO("ConstructRankTableServerId serverId %s", serverId.c_str());
}

HcclResult TopoInfoExchangeAgent::SetTransportInfo(RankTable_t& clusterInfo)
{
    CHK_PRT_RET(
        clusterInfo.rankList.size() <= localRankInfo_.rank,
        HCCL_ERROR(
            "[Set][TransportInfo]rank list is "
            "invalid. size[%zu] should be greater than myRank[%u].",
            clusterInfo.rankList.size(), localRankInfo_.rank),
        HCCL_E_INTERNAL);
    RankInfo_t& myRankInfo = clusterInfo.rankList[localRankInfo_.rank];
    TransportInfo_t transportInfo = {};

    for (u32 index = 0; index < clusterInfo.rankList.size(); index++) {
        transportInfo.dstRankId = clusterInfo.rankList[index].rankId;
        HcclResult ret = DetectTransportType(myRankInfo, clusterInfo.rankList[index], transportInfo.transportType);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Set][TransportInfo]rank[%u] detect transport type failed, ret[%u]. "
                "remote[%u]",
                localRankInfo_.rank, ret, transportInfo.dstRankId),
            ret);
        myRankInfo.transportInfo.push_back(transportInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::DetectTransportType(
    const RankInfo_t& localRankInfo, const RankInfo_t& remoteRankInfo, TransportType& transportType) const
{
    if (remoteRankInfo.serverId == localRankInfo.serverId) {
        transportType = TransportType::TRANS_TYPE_P2P;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::VerifyClusterInfo(RankTable_t& clusterInfo)
{
    std::string errormessage;

    if (clusterInfo.rankList.size() != localRankInfo_.rankSize) {
        errormessage = "The number of ranks[" + std::to_string(localRankInfo_.rankSize)
                       + "]passed by the communicator initialization interface does not match the number of ranks["
                       + std::to_string(clusterInfo.rankList.size())
                       + "] obtained during cluster information negotiation.";
        HCCL_ERROR(
            "[%s][%s]%s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), errormessage.c_str());
        return HCCL_E_PARA;
    }

    if (clusterInfo.rankNum != localRankInfo_.rankSize) {
        errormessage = "The number of ranks[" + std::to_string(localRankInfo_.rankSize)
                       + "]passed by the communicator initialization interface does not match the number of ranks["
                       + std::to_string(clusterInfo.rankNum) + "] obtained during cluster information negotiation.";
        HCCL_ERROR(
            "[%s][%s]%s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), errormessage.c_str());
        return HCCL_E_PARA;
    }

    if (clusterInfo.serverNum != clusterInfo.serverList.size()) {
        errormessage = "server num[" + std::to_string(clusterInfo.serverNum) + "] is different with server list size["
                       + std::to_string(clusterInfo.serverList.size()) + "] in total topo rank info";
        RPT_INPUT_ERR(
            true, "EI0015", std::vector<std::string>({"error_reason"}), std::vector<std::string>({errormessage}));
        HCCL_ERROR(
            "[%s][%s]%s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), errormessage.c_str());
        return HCCL_E_PARA;
    }

    if (clusterInfo.nicDeploy != localRankInfo_.nicDeploy) {
        errormessage = "nicDeploy[" + std::to_string(static_cast<int>(localRankInfo_.nicDeploy))
                       + "] is different with nicDeploy[" + std::to_string(static_cast<int>(clusterInfo.nicDeploy))
                       + "] in total topo rank info";
        RPT_INPUT_ERR(
            true, "EI0015", std::vector<std::string>({"error_reason"}), std::vector<std::string>({errormessage}));
        HCCL_ERROR(
            "[%s][%s]%s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), errormessage.c_str());
        return HCCL_E_PARA;
    }

    CHK_RET(VerifyClusterRankID(clusterInfo));
    if (localRankInfo_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        CHK_RET(VerifyClusterDeviceIP(clusterInfo));
        CHK_RET(VerifyClusterBackupDeviceIP(clusterInfo));
    }
    std::map<std::string, std::vector<RankInfo_t>> serverMap;
    for (uint32_t i = 0; i < clusterInfo.rankList.size(); i++) {
        auto iter = serverMap.find(clusterInfo.rankList[i].serverId);
        if (iter == serverMap.end()) {
            std::vector<RankInfo_t> vec;
            vec.push_back(clusterInfo.rankList[i]);
            serverMap.insert({clusterInfo.rankList[i].serverId, vec});
        } else {
            serverMap[clusterInfo.rankList[i].serverId].push_back(clusterInfo.rankList[i]);
        }
    }

    if (clusterInfo.serverNum != serverMap.size()) {
        errormessage = "server num[" + std::to_string(clusterInfo.serverNum) + "] is different with server num["
                       + std::to_string(serverMap.size()) + "] in total topo rank info";
        RPT_INPUT_ERR(
            true, "EI0015", std::vector<std::string>({"error_reason"}), std::vector<std::string>({errormessage}));
        HCCL_ERROR(
            "[%s][%s]%s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), errormessage.c_str());
        return HCCL_E_PARA;
    }

    uint32_t deviceNumInServer = 0;
    for (auto& server : serverMap) {
        CHK_PRT_RET(
            (server.second.size() == 0),
            HCCL_ERROR(
                "[%s][%s]server ip[%s] has %u device.", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), server.first.c_str(), server.second.size()),
            HCCL_E_PARA);

        if (deviceNumInServer != 0) {
            HCCL_WARNING(
                "[%s][%s]server ip[%s] has %u devices, other server has %u.", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), server.first.c_str(), server.second.size(), deviceNumInServer);
        }
        deviceNumInServer = server.second.size();
        HcclResult ret = VerifyServerDevicePhysicID(server.second);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s][%s]server id[%s] verify device physic id failed.", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), server.first.c_str()),
            HCCL_E_PARA);
    }

    bool useSuperPodMode = false;
    CHK_RET(IsSuperPodMode(useSuperPodMode));
    bool isSinglePodInterHccs
        = clusterInfo.superPodNum == 1 && GetExternalInputInterHccsDisable() == false && useSuperPodMode;
    // 单超节点，并且节点间走HCCS场景，不校验ip family
    if (clusterInfo.serverNum > 1 && !isSinglePodInterHccs) {
        CHK_RET(CheckRankIpFamily(clusterInfo.rankList));
    }

    // 超节点校验
    CHK_RET(VerifyClusterSuperPodInfo(clusterInfo.rankList));

    // TLS开关一致性校验
    CHK_RET(VerifyClusterTlsConsistency(clusterInfo));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::VerifyClusterDeviceIP(const RankTable_t& clusterInfo)
{
    if (clusterInfo.rankList.size() == 1) {
        return HCCL_SUCCESS;
    }
    if (clusterInfo.serverList.size() == 1) {
        // 单机场景对 device ip不做要求
        return HCCL_SUCCESS;
    }
    bool useSuperPodMode = false;
    CHK_RET(IsSuperPodMode(useSuperPodMode));
    if (clusterInfo.superPodNum == 1 && GetExternalInputInterHccsDisable() == false && useSuperPodMode) {
        // 单超节点，并且节点间走HCCS场景，device ip不做要求
        return HCCL_SUCCESS;
    }
    for (u32 i = 0; i < (clusterInfo.rankList.size() - 1); i++) {
        for (u32 j = (i + 1); j < clusterInfo.rankList.size(); j++) {
            bool isErr = HasRepeatedIP(
                clusterInfo.rankList[i].deviceInfo.deviceIp, clusterInfo.rankList[j].deviceInfo.deviceIp);
            if (isErr) {
                std::string errormessage = "The device IP address "
                                           + std::string(clusterInfo.rankList[i].deviceInfo.deviceIp[0].GetReadableIP())
                                           + " of rank " + std::to_string(clusterInfo.rankList[i].rankId) + " on node "
                                           + clusterInfo.rankList[i].serverId + " is the same as the device IP address "
                                           + std::string(clusterInfo.rankList[j].deviceInfo.deviceIp[0].GetReadableIP())
                                           + " of rank " + std::to_string(clusterInfo.rankList[j].rankId) + " on node "
                                           + clusterInfo.rankList[j].serverId;
                RPT_INPUT_ERR(
                    true, "EI0015", std::vector<std::string>({"error_reason"}),
                    std::vector<std::string>({errormessage}));

                HCCL_ERROR(
                    "[%s][%s]%s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                    errormessage.c_str());
                return HCCL_E_PARA;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::VerifyClusterBackupDeviceIP(RankTable_t& clusterInfo)
{
    if (localRankInfo_.deviceType != DevType::DEV_TYPE_910_93 || !isRetry_) {
        // 未开启重执行，则无需 backup device ip
        return HCCL_SUCCESS;
    }
    bool useSuperPodMode = false;
    CHK_RET(IsSuperPodMode(useSuperPodMode));
    if (!useSuperPodMode || clusterInfo.superPodNum == 1) {
        // 非多超节点场景，backup device ip 不做要求
        return HCCL_SUCCESS;
    }
    if (clusterInfo.rankList.size() == 1 || clusterInfo.serverList.size() == 1) {
        // 单卡或单机场景对 device ip 不做要求
        return HCCL_SUCCESS;
    }

    std::unordered_map<std::string, s32> devIp2PhyId;
    for (auto& rankInfo : clusterInfo.rankList) {
        for (auto& devIp : rankInfo.deviceInfo.deviceIp) {
            devIp2PhyId.emplace(devIp.GetReadableIP(), rankInfo.deviceInfo.devicePhyId);
        }
    }

    for (auto& rankInfo : clusterInfo.rankList) {
        for (auto& backupDevIp : rankInfo.deviceInfo.backupDeviceIp) {
            if (backupDevIp.IsInvalid()) {
                continue;
            }
            std::string backupIpStr = std::string(backupDevIp.GetReadableIP());
            if (devIp2PhyId.find(backupIpStr) == devIp2PhyId.end()) {
                HCCL_RUN_WARNING(
                    "[Verify][ClusterBackupDeviceIP]"
                    "backup devIp[%s] for devicePhyId[%d] is not in this comm. "
                    "The validation of this backup ip could not be verified! "
                    "Please notice it might be an invalid backup ip!",
                    backupIpStr.c_str(), rankInfo.deviceInfo.devicePhyId);
                continue;
            }

            s32 backupDevPhyId = devIp2PhyId[backupIpStr];
            std::string errormessage;
            if (backupDevPhyId == rankInfo.deviceInfo.devicePhyId) {
                errormessage = "PhyId[" + std::to_string(backupDevPhyId) + "] for backup devIp[" + backupIpStr
                               + "] is the same with self devicephyId["
                               + std::to_string(rankInfo.deviceInfo.devicePhyId)
                               + "]. Please do not use self ip as backup ip";
                RPT_INPUT_ERR(
                    true, "EI0015", std::vector<std::string>({"error_reason"}),
                    std::vector<std::string>({errormessage}));
                HCCL_ERROR(
                    "[%s][%s]errNo[0x%016llx], %s", LOG_KEYWORDS_INIT_GROUP.c_str(),
                    LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA), errormessage.c_str());
                return HCCL_E_PARA;
            }

            LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
            CHK_RET(hrtGetPairDeviceLinkType(rankInfo.deviceInfo.devicePhyId, backupDevPhyId, linkType));
            if (linkType != LinkTypeInServer::SIO_TYPE) {
                RPT_INPUT_ERR(
                    true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                    std::vector<std::string>(
                        {std::to_string(backupDevPhyId),
                         " \"backup_device_ip of "
                         "rank "
                             + std::to_string(rankInfo.rankId) + "\" ",
                         " \"is device_ip another Die under the same NPU\" "}));
                errormessage = "Value " + std::to_string(backupDevPhyId)
                               + " for rankTable variable \"backup_device_ip of "
                                 "rank "
                               + std::to_string(rankInfo.rankId)
                               + "\" is invalid, expected value \"is device_ip another Die under the same NPU\".";

                HCCL_ERROR(
                    "[%s][%s]errNo[0x%016llx], %s", LOG_KEYWORDS_INIT_GROUP.c_str(),
                    LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA), errormessage.c_str());
                return HCCL_E_PARA;
            }
        }
    }
    return HCCL_SUCCESS;
}

bool TopoInfoExchangeAgent::HasRepeatedIP(
    const std::vector<HcclIpAddress>& deviceAIP, const std::vector<HcclIpAddress>& deviceBIP) const
{
    for (u32 i = 0; i < deviceAIP.size(); i++) {
        for (u32 j = 0; j < deviceBIP.size(); j++) {
            if (deviceAIP[i] == deviceBIP[j]) {
                HCCL_WARNING("device ip[%s] is repeated.", deviceAIP[i].GetReadableAddress());
                return true;
            }
        }
    }
    return false;
}

HcclResult TopoInfoExchangeAgent::VerifyClusterRankID(const RankTable_t& clusterInfo) const
{
    if (clusterInfo.rankList.size() == 1) {
        return HCCL_SUCCESS;
    }
    for (u32 i = 0; i < (clusterInfo.rankList.size() - 1); i++) {
        for (u32 j = (i + 1); j < clusterInfo.rankList.size(); j++) {
            bool isErr = (clusterInfo.rankList[i].rankId == clusterInfo.rankList[j].rankId);
            if (isErr) {
                std::string errormessage
                    = "Rank ID " + std::to_string(clusterInfo.rankList[i].rankId) + " of device ID "
                      + std::to_string(clusterInfo.rankList[i].deviceInfo.devicePhyId) + " on node "
                      + clusterInfo.rankList[i].serverId + " is the same as that of device ID "
                      + std::to_string(clusterInfo.rankList[j].deviceInfo.devicePhyId) + " on node "
                      + clusterInfo.rankList[j].serverId;
                RPT_INPUT_ERR(
                    true, "EI0015", std::vector<std::string>({"error_reason"}),
                    std::vector<std::string>({errormessage}));
                HCCL_ERROR(
                    "[%s][%s]%s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                    errormessage.c_str());
                return HCCL_E_PARA;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::VerifyServerDevicePhysicID(const std::vector<RankInfo_t>& serverInfo) const
{
    if (serverInfo.size() == 1) {
        return HCCL_SUCCESS;
    }
    for (u32 i = 0; i < (serverInfo.size() - 1); i++) {
        for (u32 j = (i + 1); j < serverInfo.size(); j++) {
            bool isErr = (serverInfo[i].deviceInfo.devicePhyId == serverInfo[j].deviceInfo.devicePhyId);
            if (isErr) {
                std::string errormessage = "Rank " + std::to_string(serverInfo[i].rankId) + " of node "
                                           + serverInfo[i].serverId + " has the same physical device ID "
                                           + std::to_string(serverInfo[i].deviceInfo.devicePhyId) + " as the rank "
                                           + std::to_string(serverInfo[j].rankId);
                RPT_INPUT_ERR(
                    true, "EI0015", std::vector<std::string>({"error_reason"}),
                    std::vector<std::string>({errormessage}));
                HCCL_ERROR(
                    "[%s][%s]%s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                    errormessage.c_str());
                return HCCL_E_PARA;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::VerifyClusterSuperPodInfo(const std::vector<RankInfo_t>& rankInfo) const
{
    DevType curDevType = rankInfo.begin()->deviceInfo.deviceType;
    for (auto curRankInfo : rankInfo) {
        if (curDevType != curRankInfo.deviceInfo.deviceType) {
            HCCL_DEBUG("[Verify][SuperPodInfo] mix device type, does not need verify superPod info");
            return HCCL_SUCCESS;
        }
    }

    bool useSuperPodMode = false;
    CHK_RET(IsSuperPodMode(useSuperPodMode));
    CHK_PRT_RET(
        useSuperPodMode == false, HCCL_DEBUG("[Verify][SuperPodInfo] does not need verify superPod info"),
        HCCL_SUCCESS);

    std::string errormessage = "";
    // 获取每个超节点内的serverId
    std::map<std::string, std::set<std::string>> superPodSrvIdMap;       // super_pod_id -> serverId
    std::map<std::string, std::unordered_map<u32, u32>> superPodSdidMap; // super_pod_id -> superDeviceId
    for (u32 i = 0; i < rankInfo.size(); i++) {
        // 超节点模式下, 校验superPodId和sdid值有效
        if ((rankInfo[i].superPodId.empty() || rankInfo[i].superDeviceId == INVALID_UINT)
            && rankInfo[i].deviceInfo.deviceType == DevType::DEV_TYPE_910_93) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>(
                    {std::to_string(rankInfo[i].superDeviceId), "super_device_id",
                     "is less than the communication size " + std::to_string(rankInfo.size())
                         + " and must be unique"}));
            errormessage = "Value " + std::to_string(rankInfo[i].superDeviceId)
                           + " for rankTable variable superDeviceId is invalid, "
                             "expected value is less than the communication size "
                           + std::to_string(rankInfo.size()) + " and must be unique.";

            HCCL_ERROR(
                "[%s][%s]%s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                errormessage.c_str());
            return HCCL_E_PARA;
        }

        auto iter = superPodSrvIdMap.find(rankInfo[i].superPodId);
        if (iter == superPodSrvIdMap.end()) {
            std::set<std::string> serverIdSet;
            serverIdSet.insert(rankInfo[i].serverId);
            superPodSrvIdMap.insert({rankInfo[i].superPodId, serverIdSet});
        } else if (iter->second.find(rankInfo[i].serverId) == iter->second.end()) {
            iter->second.insert(rankInfo[i].serverId);
        }

        auto it = superPodSdidMap.find(rankInfo[i].superPodId);
        if (it == superPodSdidMap.end()) {
            std::unordered_map<u32, u32> superDeviceIdSet;
            superDeviceIdSet.insert({rankInfo[i].superDeviceId, rankInfo[i].rankId});
            superPodSdidMap.insert({rankInfo[i].superPodId, superDeviceIdSet});
        } else if (it->second.find(rankInfo[i].superDeviceId) == it->second.end()) {
            it->second.insert({rankInfo[i].superDeviceId, rankInfo[i].rankId});
        } else {
            // 超节点内superDeviceId在超节点内唯一
            if (it->second.find(rankInfo[i].superDeviceId) != it->second.end()) {
                RPT_INPUT_ERR(
                    true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                    std::vector<std::string>(
                        {std::to_string(rankInfo[i].superDeviceId),
                         " \"Device Id of server Id " + rankInfo[i].serverId + "\" ", "is unique"}));
                errormessage = "Value " + std::to_string(rankInfo[i].superDeviceId)
                               + " for rankTable "
                                 "variable \"Device Id of server Id "
                               + rankInfo[i].serverId + "\" is invalid, expected value is unique.";
                HCCL_ERROR(
                    "[%s][%s]%s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                    errormessage.c_str());
                return HCCL_E_PARA;
            }
        }
    }

    // 校验每个超节点内的server数量一致
    u32 serverNumPerPod = 0;
    for (auto iter = superPodSrvIdMap.begin(); iter != superPodSrvIdMap.end(); ++iter) {
        if (iter == superPodSrvIdMap.begin()) {
            serverNumPerPod = superPodSrvIdMap.begin()->second.size();
        }
        u32 serverNumCurPod = iter->second.size();
        if (serverNumPerPod != serverNumCurPod) {
            HCCL_DEBUG(
                "[Verify][SuperPodInfo]serverNum[%u] in superPod[%s] and serverNum[%u] in superPod[%s] "
                "are different.",
                serverNumPerPod, superPodSrvIdMap.begin()->first.c_str(), serverNumCurPod, iter->first.c_str());
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::VerifyClusterTlsConsistency(const RankTable_t& clusterInfo)
{
    bool isSupportCheckTlsStatus = true; // 用于标识是否存在不支持查询Tls开关状态的情况
    bool isTlsConsistent = true;         // 用于标识TLS开关状态是否一致
    std::unordered_map<std::string, std::vector<u32>> tlsEnableRank;
    std::unordered_map<std::string, std::vector<u32>> tlsDisableRank;
    std::unordered_map<std::string, std::vector<u32>> tlsUnknownRank;
    for (auto& rankInfo : clusterInfo.rankList) {
        if (rankInfo.tlsStatus == TlsStatus::ENABLE) {
            AddRankInfoToTlsStatusMap(rankInfo, tlsEnableRank);
        } else if (rankInfo.tlsStatus == TlsStatus::DISABLE) {
            AddRankInfoToTlsStatusMap(rankInfo, tlsDisableRank);
        } else {
            isSupportCheckTlsStatus = false;
            AddRankInfoToTlsStatusMap(rankInfo, tlsUnknownRank);
        }
    }
    // 将不一致的卡信息汇总成一个string
    std::string tlsInconsistentEnableStr = "";
    std::string tlsInconsistentDisableStr = "";
    std::string tlsInconsistentTlsType = "";
    if (!tlsEnableRank.empty() && !tlsDisableRank.empty()) {
        isTlsConsistent = false;
        tlsInconsistentTlsType = (tlsEnableRank.size() >= tlsDisableRank.size()) ? "Disable" : "Enable";
        GenerateTlsStatusStr(tlsInconsistentEnableStr, tlsEnableRank);
        GenerateTlsStatusStr(tlsInconsistentDisableStr, tlsDisableRank);
    }
    // 将不支持查询的卡的信息汇总成一个string
    std::string tlsUnknownRankStr = "";
    if (!isSupportCheckTlsStatus) {
        GenerateTlsStatusStr(tlsUnknownRankStr, tlsUnknownRank);
    }
    tlsUnknownRankStr = tlsUnknownRankStr.empty() ? "N/A" : tlsUnknownRankStr;
    // 四种不同情况
    if (isTlsConsistent && isSupportCheckTlsStatus) {
        // 1.通信域所有卡都支持查询TLS开关状态，并且TLS开关状态都是一致的。
        HCCL_INFO("[Verify][TlsConsistency] All ranks tlsStatus are consistent");
    } else if (!isTlsConsistent && isSupportCheckTlsStatus) {
        // 2.通信域所有卡都支持查询TLS开关状态，但是TLS开关状态存在不一致，报错。
        ReportTlsConfigurationError(tlsInconsistentTlsType, tlsInconsistentEnableStr, tlsInconsistentDisableStr, "N/A");
        return HCCL_E_PARA;
    } else if (isTlsConsistent && !isSupportCheckTlsStatus) {
        // 3.通信域内的部分卡不支持查询TLS开关状态，目前能查询到的卡的TLS开关状态是一致的，打印warning提醒
        HCCL_RUN_WARNING(
            "[Verify][TlsConsistency] Some ranks do not support to check tlsStatus, "
            "not support serverId/rankId: %s",
            tlsUnknownRankStr.c_str());
    } else {
        // 4.通信域内的部分卡不支持查询TLS开关状态，但是目前能查询到的卡的TLS开关状态已经不一致，报错
        ReportTlsConfigurationError(
            tlsInconsistentTlsType, tlsInconsistentEnableStr, tlsInconsistentDisableStr, tlsUnknownRankStr);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

void TopoInfoExchangeAgent::AddRankInfoToTlsStatusMap(
    const RankInfo_t& rankInfo, std::unordered_map<std::string, std::vector<u32>>& tlsStatusRankMap) const
{
    auto iter = tlsStatusRankMap.find(rankInfo.serverId);
    if (iter == tlsStatusRankMap.end()) {
        std::vector<u32> tlsStatusRankList;
        tlsStatusRankList.push_back(rankInfo.rankId);
        tlsStatusRankMap.insert({rankInfo.serverId, tlsStatusRankList});
    } else {
        iter->second.push_back(rankInfo.rankId);
    }
    return;
}

void TopoInfoExchangeAgent::GenerateTlsStatusStr(
    std::string& tlsStatusStr, const std::unordered_map<std::string, std::vector<u32>>& tlsStatusRankMap) const
{
    for (const auto& rankIt : tlsStatusRankMap) {
        tlsStatusStr += ("[" + rankIt.first + "/");
        for (const auto& rank : rankIt.second) {
            tlsStatusStr += std::to_string(rank) + ",";
        }
        if (!tlsStatusStr.empty() && tlsStatusStr.back() == ',') {
            tlsStatusStr = tlsStatusStr.substr(0, tlsStatusStr.size() - 1); // 删除逗号
        }
        tlsStatusStr += "];";
    }
    return;
}

void TopoInfoExchangeAgent::ReportTlsConfigurationError(
    const std::string& tlsInconsistentTlsType, const std::string& tlsInconsistentEnableStr,
    const std::string& tlsInconsistentDisableStr, const std::string& tlsUnknownRankStr) const
{
    std::string errormessage
        = "Value " + tlsInconsistentTlsType
          + " for config \"tls\" is invalid. Expected: \"All ranks are consistent. Current status: "
            "rankList for enabled tls: "
          + tlsInconsistentEnableStr + " rankList for disabled tls:" + tlsInconsistentDisableStr
          + " rankList for query failure tls:" + tlsUnknownRankStr + ".\"";
    RPT_INPUT_ERR(
        true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
        std::vector<std::string>(
            {tlsInconsistentTlsType, " \"tls\" ",
             " \"All ranks are consistent. Current status: rankList for enabled tls:" + tlsInconsistentEnableStr
                 + "; "
                   "rankList for disabled tls:"
                 + tlsInconsistentDisableStr + " rankList for query failure tls:" + tlsUnknownRankStr + ".\" "}));
    HCCL_ERROR(
        "[%s][%s] %s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), errormessage.c_str());
}
} // namespace hccl
