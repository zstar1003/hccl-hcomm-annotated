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
#include <cerrno>
#include <string>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include <unistd.h>
#include "adapter_hccp.h"
#include "adapter_rts_common.h"
#include "hccl_whitelist.h"
#include "hccl_socket.h"
#include "sal_pub.h"
#include "device_capacity.h"
#include "preempt_port_manager.h"

using namespace std;
namespace hccl {
constexpr u32 HOST_CONTROL_PORT_RANGE_SIZE = 31;
UniversalConcurrentMap<u32, volatile u32> TopoInfoDetect::g_topoExchangeServerStatus_;

TopoInfoDetect::TopoInfoDetect()
    : deviceLogicID_(INVALID_INT),
      localRankInfo_(),
      clusterTopoInfo_(),
      isInterSuperPodRetryEnable_(GetExternalInputInterSuperPodRetryEnable())
{}

TopoInfoDetect::~TopoInfoDetect()
{
    if (exchangeServerThreadPtr_ && exchangeServerThreadPtr_->joinable()) {
        exchangeServerThreadPtr_->join();
    }
    exchangeServerThreadPtr_ = nullptr;
    pTopoExchangeServer_ = nullptr;
    (void)Teardown();
    return;
}

HcclResult TopoInfoDetect::GetServerConnections(std::map<u32, std::shared_ptr<HcclSocket>>& connectSockets)
{
    if (pTopoExchangeServer_) {
        return pTopoExchangeServer_->GetConnections(connectSockets);
    } else {
        return HCCL_SUCCESS;
    }
}

HcclResult TopoInfoDetect::GetAgentListenSocket(HcclSocketPortConfig& commPortConfig)
{
    // 将抢占的端口传入comm connection参数中
    commPortConfig = commPortConfig_;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetAgentConnection(std::shared_ptr<HcclSocket>& connectSocket)
{
    CHK_SMART_PTR_NULL(pTopoExchangeAgent_);
    return pTopoExchangeAgent_->GetConnection(connectSocket);
}

HcclResult TopoInfoDetect::SendGroupLeaderPort(std::shared_ptr<HcclSocket>& connectSocket, HcclRankHandle& rankHandle)
{
    CHK_SMART_PTR_NULL(pTopoExchangeAgent_);
    HcclResult ret = pTopoExchangeAgent_->SendGroupLeaderPortInfo(connectSocket, rankHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Setup][SendGroupLeaderPortInfo]SendGroupLeaderPortInfo to root failed, ret[%d]", ret), ret);
    return HCCL_SUCCESS;
}

void TopoInfoDetect::SetupTopoGroupLeader(
    [[maybe_unused]] s32 devicePhysicID, s32 deviceLogicID, HcclIpAddress hostIP, u32 hostPort,
    vector<HcclIpAddress> whitelist, HcclNetDevCtx netDevCtx, std::shared_ptr<HcclSocket> listenSocket,
    std::shared_ptr<HcclSocket> grpLeaderToRoot, bool isMasterInfo)
{
    // 给当前线程添加名字
    SetThreadName("Hccl_TopoDetect_GroupLeader");

    HcclResult ret = hrtSetDevice(deviceLogicID);
    if (ret != HCCL_SUCCESS) {
        g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
            status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[Setup][TopoExchangeServer]set device[%d] failed, ret[%u]", deviceLogicID, ret);
        return;
    }

    pTopoExchangeServer_.reset(new (nothrow) TopoInfoExchangeServer(
        hostIP, hostPort, whitelist, netDevCtx, listenSocket, grpLeaderToRoot, rootInfo_.identifier));
    if (!pTopoExchangeServer_) {
        g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
            status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[Setup][TopoExchangeServer]build topoExchangeServer failed. ");
    } else {
        ret = isMasterInfo ? pTopoExchangeServer_->SetupByMasterInfo() : pTopoExchangeServer_->SetupGroupLeader();
        if (ret != HCCL_SUCCESS) {
            g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
                status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
            });
            HCCL_ERROR("[Setup][TopoExchangeServer]setup topoExchangeServer failed, ret[%u]", ret);
        }
    }

    ret = hrtResetDevice(deviceLogicID);
    if (ret != HCCL_SUCCESS) {
        g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
            status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[Setup][TopoExchangeServer]reset device[%d] failed, ret[%u]", deviceLogicID, ret);
        return;
    }
    g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
        status = TOPO_EXCHANGE_SERVER_STATUS_IDLE;
    });
}

void TopoInfoDetect::SetupTopoExchangeServer(
    [[maybe_unused]] s32 devicePhysicID, s32 deviceLogicID, HcclIpAddress hostIP, u32 hostPort,
    vector<HcclIpAddress> whitelist, HcclNetDevCtx netDevCtx, std::shared_ptr<HcclSocket> listenSocket,
    bool isMasterInfo)
{
    // 给当前线程添加名字
    SetThreadName("Hccl_TopoDetect");

    HcclResult ret = hrtSetDevice(deviceLogicID);
    if (ret != HCCL_SUCCESS) {
        g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
            status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[Setup][TopoExchangeServer]set device[%d] failed, ret[%u]", deviceLogicID, ret);
        return;
    }

    pTopoExchangeServer_.reset(new (nothrow) TopoInfoExchangeServer(
        hostIP, hostPort, whitelist, netDevCtx, listenSocket, rootInfo_.identifier));
    if (!pTopoExchangeServer_) {
        g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
            status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[Setup][TopoExchangeServer]build topoExchangeServer failed. ");
    } else {
        ret = isMasterInfo ? pTopoExchangeServer_->SetupByMasterInfo() : pTopoExchangeServer_->Setup();
        if (ret != HCCL_SUCCESS) {
            g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
                status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
            });
            HCCL_ERROR("[Setup][TopoExchangeServer]setup topoExchangeServer failed, ret[%u]", ret);
        }
    }

    ret = hrtResetDevice(deviceLogicID);
    if (ret != HCCL_SUCCESS) {
        g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
            status = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[Setup][TopoExchangeServer]reset device[%d] failed, ret[%u]", deviceLogicID, ret);
        return;
    }
    g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
        status = TOPO_EXCHANGE_SERVER_STATUS_IDLE;
    });
}
HcclResult
TopoInfoDetect::SetupServerByMasterInfo(const HcclIpAddress& masterIP, u32 masterPort, const HcclRootHandle& rootInfo)
{
    CHK_RET(hrtGetDevice(&deviceLogicID_));
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_));
    vector<HcclIpAddress> whitelist;
    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(ReadHostSocketWhitelist(whitelist));
    }
    rootInfo_ = rootInfo;
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_, true));
    HcclResult ret = CheckHostNicLinkUp(masterIP);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Setup][Server]host NIC status check failed, deviceLogicID[%d], devicePhysicID[%u], hostIP[%s], "
            "ret[%d].",
            deviceLogicID_, devicePhysicID_, masterIP.GetReadableAddress(), ret),
        ret);
    ret = StartRootNetwork(masterIP, masterPort, GetExternalInputHostSocketPortRange());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]%s failed, masterIP[%s] and masterPort[%u] ret[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), __func__, masterIP.GetReadableAddress(), masterPort, ret),
        ret);

    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(AddSocketWhiteList(masterPort, whitelist));
    }

    u32 hostPort = GetExternalInputMasterInfo().port;
    g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
        status = TOPO_EXCHANGE_SERVER_STATUS_RUNING;
    });

    thread threadHandle(
        &TopoInfoDetect::SetupTopoExchangeServer, this, devicePhysicID_, deviceLogicID_, masterIP,
        GetExternalInputMasterInfo().port, whitelist, serverPortCtx_, listenSocket_, true);
    threadHandle.detach();

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::SetupServer(HcclRootHandle& rootInfo)
{
    HcclIpAddress hostIP;
    u32 hostPort = HCCL_INVALID_PORT;
    std::vector<HcclSocketPortRange> portRanges;
    std::vector<HcclIpAddress> whitelist;
    CHK_RET(SetupRootServerNetwork(hostIP, hostPort, portRanges, whitelist));

    g_topoExchangeServerStatus_.EmplaceAndUpdate(hostPort, [](volatile u32& status) {
        status = TOPO_EXCHANGE_SERVER_STATUS_RUNING;
    });
    exchangeServerThreadPtr_.reset(new (nothrow) thread(
        &TopoInfoDetect::SetupTopoExchangeServer, this, devicePhysicID_, deviceLogicID_, hostIP, hostPort, whitelist,
        serverPortCtx_, listenSocket_, false));
    CHK_SMART_PTR_NULL(exchangeServerThreadPtr_);

    rootInfo = rootInfo_;
    HCCL_INFO("setup topo exchange server complete, identifier[%s]", rootInfo.identifier);
    return HCCL_SUCCESS;
}

// root侧网络初始化：
// 设备/网卡初始化、监听端口选择（固定端口或端口range抢占）、启动agent监听、生成rootInfo、下发白名单。
HcclResult TopoInfoDetect::SetupRootServerNetwork(
    HcclIpAddress& hostIP, u32& hostPort, std::vector<HcclSocketPortRange>& portRanges,
    std::vector<HcclIpAddress>& whitelist)
{
    CHK_RET(hrtGetDevice(&deviceLogicID_));

    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(ReadHostSocketWhitelist(whitelist));
    }
    hostIP = GetBootstrapHostIP();
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_, true));
    HCCL_INFO("[Setup][hcclIfBasePort]deviceLogicID_[%u], devicePhysicID_[%u]", deviceLogicID_, devicePhysicID_);

    // true代表感知白名单disable配置
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_, true));
    CHK_RET(GetRootHostIP(whitelist, hostIP, devicePhysicID_));
    SetBootstrapHostIP(hostIP);

    HcclResult ret = CheckHostNicLinkUp(hostIP);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[Setup][Server]host NIC status check failed, deviceLogicID[%d], devicePhysicID[%u], hostIP[%s], "
            "ret[%d].",
            deviceLogicID_, devicePhysicID_, hostIP.GetReadableAddress(), ret);
        return ret;
    }

    HCCL_INFO(
        "[Setup][Server]host NIC status check success, deviceLogicID[%d], devicePhysicID[%u], hostIP[%s].",
        deviceLogicID_, devicePhysicID_, hostIP.GetReadableAddress());

    u32 deviceNum = 0;
    CHK_RET(hrtGetDeviceCount(&deviceNum));
    CHK_PRT_RET(
        (static_cast<u32>(deviceLogicID_) >= deviceNum),
        HCCL_ERROR("[Setup][Server]deviceLogicID[%d] is invalid,deviceNum[%d].", deviceLogicID_, deviceNum),
        HCCL_E_PARA);

    hostPort = HCCL_INVALID_PORT;
    if (!GetExternalInputHostPortSwitch()) {
        if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
            // 若没有设置HCCL_HOST_SOCKET_PORT_RANGE和HCCL_IF_BASE_PORT 使用自动调整监听端口range[60000,60031]
            HCCL_RUN_INFO(
                "[Setup][Server] user not set base port and port range, use default port range[%u, %u]",
                HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_RANGE_SIZE);
            portRanges.push_back({HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_RANGE_SIZE});
        } else {
            hostPort = devicePhysicID_ + GetExternalInputHcclIfBasePort();
        }
    } else {
        portRanges = GetExternalInputHostSocketPortRange();
    }
    CHK_RET(StartRootNetwork(hostIP, hostPort, portRanges));
    CHK_RET(GenerateRootInfo(hostIP, hostPort, devicePhysicID_, rootInfo_));
    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(AddSocketWhiteList(hostPort, whitelist));
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GroupLeaderListen(HcclRankHandle& rankHandle, vector<HcclIpAddress>& whitelist)
{
    CHK_RET(hrtGetDevice(&deviceLogicID_));

    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(ReadHostSocketWhitelist(whitelist));
    }
    HcclIpAddress hostIP = GetBootstrapHostIP();
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_, true));
    HCCL_INFO("[Setup][hcclIfBasePort]deviceLogicID_[%u], devicePhysicID_[%u]", deviceLogicID_, devicePhysicID_);

    // true代表感知白名单disable配置
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_, true));

    CHK_RET(GetRootHostIP(whitelist, hostIP, devicePhysicID_));
    SetBootstrapHostIP(hostIP);

    u32 deviceNum = INVALID_INT;
    CHK_RET(hrtGetDeviceCount(&deviceNum));
    CHK_PRT_RET(
        (static_cast<u32>(deviceLogicID_) >= deviceNum),
        HCCL_ERROR("[Setup][GroupLeader]deviceLogicID[%d] is invalid,deviceNum[%d].", deviceLogicID_, deviceNum),
        HCCL_E_PARA);

    u32 hostPort = HCCL_INVALID_PORT;
    std::vector<HcclSocketPortRange> portRanges;
    if (!GetExternalInputHostPortSwitch()) {
        // 不开启host侧端口范围配置, 则使用默认端口
        if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
            // 若没有设置HCCL_HOST_SOCKET_PORT_RANGE和HCCL_IF_BASE_PORT 使用自动调整监听端口range[60000,60031]
            HCCL_RUN_INFO(
                "[Setup][GroupLeader] user not set base port and port range, use default port range[%u, %u]",
                HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_RANGE_SIZE);
            portRanges.push_back({HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_RANGE_SIZE});
        } else {
            hostPort = devicePhysicID_ + GetExternalInputHcclIfBasePort() + TOPO_GROUPLEADER_PORT_OFFSET;
        }
    } else {
        portRanges = GetExternalInputHostSocketPortRange();
    }

    CHK_RET(StartGroupLeaderNetwork(whitelist, hostIP, hostPort, portRanges));
    CHK_RET(GenerateRootInfo(hostIP, hostPort, devicePhysicID_, rankHandle));

    HCCL_INFO("rank bind port complete, port[%u]", hostPort);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GroupLeaderAccept(
    HcclRankHandle& grpLeaderInfo, vector<HcclIpAddress> whitelist, std::shared_ptr<HcclSocket> grpLeaderToRoot)
{
    rootInfo_ = grpLeaderInfo;
    exchangeServerThreadPtr_.reset(new (nothrow) thread(
        &TopoInfoDetect::SetupTopoGroupLeader, this, devicePhysicID_, deviceLogicID_, bootstrapHostIP_, rootInfo_.port,
        whitelist, serverPortCtx_, listenSocket_, grpLeaderToRoot, false));
    CHK_SMART_PTR_NULL(exchangeServerThreadPtr_);

    HCCL_INFO("setup group leader server complete, identifier[%s]", grpLeaderInfo.identifier);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GenerateRootInfo(
    const HcclIpAddress& hostIP, u32 hostPort, u32 devicePhysicID, HcclRootHandle& rootInfo) const
{
    u64 timestamp = 0;
    CHK_RET(SalGetCurrentTimestamp(timestamp));

    string identifier = hostIP.GetReadableAddress();
    identifier.append("_");
    identifier.append(to_string(hostPort));
    identifier.append("_");
    identifier.append(to_string(devicePhysicID));
    identifier.append("_");
    identifier.append(to_string(timestamp));
    CHK_PRT_RET(
        (identifier.length() >= ROOTINFO_INDENTIFIER_MAX_LENGTH),
        HCCL_ERROR("[Setup][Server]rootinfo identifier len[%u] is invalid.", identifier.length()), HCCL_E_INTERNAL);
    s32 sret
        = memcpy_s(&rootInfo.identifier[0], sizeof(rootInfo.identifier), identifier.c_str(), (identifier.length() + 1));
    CHK_PRT_RET(
        sret != EOK,
        HCCL_ERROR(
            "[Setup][Server]errNo[0x%016llx] memcpy failed. ret[%d], params:"
            "destMaxSize[%zu],count[%zu]",
            HCOM_ERROR_CODE(HCCL_E_MEMORY), sret, sizeof(rootInfo.identifier), (identifier.length() + 1)),
        HCCL_E_MEMORY);
    s32 sRet = strncpy_s(rootInfo.ip, sizeof(rootInfo.ip), hostIP.GetReadableIP(), strlen(hostIP.GetReadableIP()));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Setup][Server]str copy fail. return[%d]", sRet), HCCL_E_INTERNAL);
    rootInfo.port = hostPort;
    rootInfo.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    HCCL_INFO("rootInfo: ip[%s] port[%u] identifier[%s]", rootInfo.ip, rootInfo.port, rootInfo.identifier);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::CalcGroupSizeAndRank(const u32 nRanks, const u32 rank, u32& groupSize, u32& groupRank)
{
    u32 groupIndex = rank / TOPO_MAX_GROUP_SIZE;
    u32 groupNum = nRanks / TOPO_MAX_GROUP_SIZE;
    groupSize = groupIndex < groupNum ? TOPO_MAX_GROUP_SIZE : (nRanks - (TOPO_MAX_GROUP_SIZE * groupNum));
    groupRank = rank == 0 ? 0 : rank % TOPO_MAX_GROUP_SIZE;

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::SetupGroupMember(u32 rankSize, u32 myrank, const HcclRootHandle& rootInfo)
{
    CHK_RET(hrtGetDevice(&deviceLogicID_));

    HcclIpAddress rootIP(rootInfo.ip);
    CHK_PRT_RET(rootIP.IsInvalid(), HCCL_ERROR("string[%s] is invalid ip", rootInfo.ip), HCCL_E_PARA);
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_, true));

    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_, true));

    HcclIpAddress hostIP = GetBootstrapHostIP();
    CHK_RET(GetLocalHostIP(hostIP, devicePhysicID_));

    SetBootstrapHostIP(hostIP);

    bool bInitDevNic = rankSize != 1 ? true : false;
    HcclResult ret = StartNetwork(hostIP, bInitDevNic);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Setup][GroupMember]topo detect GroupMember start network failed! rank[%u]", myrank), ret);

    u32 groupSize = 0;
    u32 groupRank = 0;
    ret = CalcGroupSizeAndRank(rankSize, myrank, groupSize, groupRank);

    ret = GenerateLocalRankInfo(rankSize, myrank, localRankInfo_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Setup][GroupMember]topo detect generate local rank info failed! rank[%u]", myrank), ret);

    /* 首节点日志，建链失败属常见问题，在建链前记录相关信息 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]SetupGroupMember rankNum[%u], rank[%u], rootInfo identifier[%s], server[%s], "
        "deviceType[%d], logicDevId[%d], phydevId[%d], deviceIp[%s]",
        rankSize, myrank, rootInfo.identifier, localRankInfo_.hostIP.GetReadableAddress(), localRankInfo_.deviceType,
        localRankInfo_.deviceLogicID, localRankInfo_.devicePhysicID, localRankInfo_.deviceIP[0].GetReadableIP());

    pTopoExchangeAgent_.reset(new (nothrow) TopoInfoExchangeAgent(
        rootIP, rootInfo.port, rootInfo.identifier, agentPortCtx_, localRankInfo_, groupSize, groupRank));
    CHK_SMART_PTR_NULL(pTopoExchangeAgent_);
    CHK_RET(pTopoExchangeAgent_->SetIsInterSuperPodRetryEnable(isInterSuperPodRetryEnable_));
    CHK_RET(pTopoExchangeAgent_->SetupMember());
    CHK_RET(pTopoExchangeAgent_->GetClusterTopoInfo(clusterTopoInfo_));

    rootInfo_ = rootInfo;

    HCCL_INFO(
        "topo detect completed. myrank[%u], totalranks[%u], myhost[%s], totalservers[%u].", myrank, rankSize,
        localRankInfo_.hostIP.GetReadableAddress(), clusterTopoInfo_.serverNum);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::TeardownServer()
{
    if (pTopoExchangeServer_) {
        CHK_RET(pTopoExchangeServer_->Teardown());
    }

    if (serverPortCtx_) {
        HcclNetCloseDev(serverPortCtx_);
        serverPortCtx_ = nullptr;
        CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_));
    }
    HCCL_INFO("TopoInfoDetect TeardownServer ok, identifier[%s].", rootInfo_.identifier);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::WaitTopoExchangeServerCompelte(u32 idx) const
{
    const auto start = chrono::steady_clock::now();
    const auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    auto iter = g_topoExchangeServerStatus_.Find(idx);
    if (!iter.second) {
        return HCCL_SUCCESS;
    }
    u32 status = TOPO_EXCHANGE_SERVER_STATUS_RUNING;
    while (true) {
        auto it = g_topoExchangeServerStatus_.Find(idx);
        if (it.second) {
            status = it.first->second;
        }
        if (status == TOPO_EXCHANGE_SERVER_STATUS_ERROR) {
            HCCL_ERROR(
                "[Wait][TopoExchangeServerCompelte]topo detect failed. topoExchangeServer port[%u] failed.", idx);
            return HCCL_E_INTERNAL;
        } else if (status == TOPO_EXCHANGE_SERVER_STATUS_IDLE) {
            HCCL_INFO("topoExchangeServer[%u] completed.", idx);
            return HCCL_SUCCESS;
        } else {
            const auto elapsed = chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start);
            if (elapsed > timeout) {
                HCCL_ERROR(
                    "[Wait][TopoExchangeServerCompelte]wait topoExchangeServer[%u] complete timeout[%lld s]", idx,
                    elapsed);
                return HCCL_E_TIMEOUT;
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            continue;
        }
    };
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::PrepareHandle(HcclRankHandle& rankHandle, std::vector<HcclIpAddress>& whitelist)
{
    CHK_RET(hrtGetDevice(&deviceLogicID_));

    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(ReadHostSocketWhitelist(whitelist));
    }
    HcclIpAddress hostIP = GetBootstrapHostIP();
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_, true));
    HCCL_INFO("[Setup][hcclIfBasePort]deviceLogicID_[%u], devicePhysicID_[%u]", deviceLogicID_, devicePhysicID_);

    // true代表感知白名单disable配置
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_, true));

    CHK_RET(GetRootHostIP(whitelist, hostIP, devicePhysicID_));
    SetBootstrapHostIP(hostIP);

    u32 deviceNum = 0;
    CHK_RET(hrtGetDeviceCount(&deviceNum));
    CHK_PRT_RET(
        (static_cast<u32>(deviceLogicID_) >= deviceNum),
        HCCL_ERROR("[Setup][GroupLeader]deviceLogicID[%d] is invalid,deviceNum[%d].", deviceLogicID_, deviceNum),
        HCCL_E_PARA);

    u32 hostPort = HCCL_INVALID_PORT;
    CHK_RET(GenerateRootInfo(hostIP, hostPort, devicePhysicID_, rankHandle));

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::CheckHostNicLinkUp(const HcclIpAddress& hostIP) const
{
    const s32 family = hostIP.GetFamily();
    CHK_PRT_RET(
        (family != AF_INET) && (family != AF_INET6),
        HCCL_ERROR(
            "[Check][HostNicLink]address family[%d] is invalid, hostIP[%s].", family, hostIP.GetReadableAddress()),
        HCCL_E_PARA);

    const string ifName = hostIP.GetIfName();
    CHK_PRT_RET(
        ifName.empty() || (ifName.length() >= static_cast<size_t>(IFNAMSIZ)),
        HCCL_ERROR("[Check][HostNicLink]interface name[%s] is invalid, length[%zu].", ifName.c_str(), ifName.length()),
        HCCL_E_PARA);

    struct ifreq ifr = {};
    CHK_SAFETY_FUNC_RET(strncpy_s(ifr.ifr_name, sizeof(ifr.ifr_name), ifName.c_str(), ifName.length()));

    const int fd = socket(family, SOCK_DGRAM, 0);
    CHK_PRT_RET(
        fd < 0,
        HCCL_ERROR(
            "[Check][HostNicLink]create socket failed, family[%d], interface[%s], errno[%d].", family, ifName.c_str(),
            errno),
        HCCL_E_SYSCALL);

    if (ioctl(fd, SIOCGIFFLAGS, &ifr) < 0) {
        const s32 errorNo = errno;
        (void)close(fd);
        HCCL_ERROR(
            "[Check][HostNicLink]get interface flags failed, family[%d], interface[%s], errno[%d].", family,
            ifName.c_str(), errorNo);
        return HCCL_E_SYSCALL;
    }
    if (close(fd) != 0) {
        HCCL_ERROR(
            "[Check][HostNicLink]close socket failed, family[%d], interface[%s], errno[%d].", family, ifName.c_str(),
            errno);
        return HCCL_E_SYSCALL;
    }
    return CheckHostNicFlags(ifName, static_cast<u32>(static_cast<u16>(ifr.ifr_flags)));
}

HcclResult TopoInfoDetect::CheckHostNicFlags(const string& ifName, u32 flags) const
{
    const bool isUp = (flags & static_cast<u32>(IFF_UP)) != 0;
    const bool isRunning = (flags & static_cast<u32>(IFF_RUNNING)) != 0;
    if (!isUp || !isRunning) {
        HCCL_ERROR(
            "[Check][HostNicLink]host NIC link is down, interface[%s], ifrFlags[0x%x], isUp[%d], isRunning[%d].",
            ifName.c_str(), flags, static_cast<s32>(isUp), static_cast<s32>(isRunning));
        return HCCL_E_NETWORK;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::SetupAgent(
    u32 rankSize, u32 myrank, const HcclRootHandle& rootInfo, const HcclRankHandle& rankHandle,
    const CommConfig& commConfig, bool isScalable)
{
    commConfig_ = commConfig;
    CHK_PRT_RET(
        (rootInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST),
        HCCL_ERROR(
            "[Setup][Agent]hcclDeviceNicDisable is [%u] when nicDeploy from root is NIC_DEPLOYMENT_HOST",
            rootInfo.nicDeploy),
        HCCL_E_PARA);
    CHK_RET(hrtGetDevice(&deviceLogicID_));

    HcclIpAddress rootIP(rootInfo.ip);
    CHK_PRT_RET(rootIP.IsInvalid(), HCCL_ERROR("string[%s] is invalid ip", rootInfo.ip), HCCL_E_PARA);
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_, true));

    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_, true));

    HcclIpAddress hostIP = GetBootstrapHostIP();
    CHK_RET(GetLocalHostIP(hostIP, devicePhysicID_));

    SetBootstrapHostIP(hostIP);
    HcclResult ret = CheckHostNicLinkUp(hostIP);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[Setup][Agent]host NIC status check failed, deviceLogicID[%d], devicePhysicID[%u], hostIP[%s], "
            "ret[%d].",
            deviceLogicID_, devicePhysicID_, hostIP.GetReadableAddress(), ret);
        return ret;
    }

    HCCL_INFO(
        "[Setup][Agent]host NIC status check success, deviceLogicID[%d], devicePhysicID[%u], hostIP[%s].",
        deviceLogicID_, devicePhysicID_, hostIP.GetReadableAddress());

    bool bInitDevNic = rankSize != 1 ? true : false;
    ret = StartNetwork(hostIP, bInitDevNic);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]topo detect agent start network failed! rank[%u]", myrank), ret);

    ret = GenerateLocalRankInfo(rankSize, myrank, localRankInfo_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]topo detect generate local rank info failed! rank[%u]", myrank),
        ret);

    if (rankSize > TOPO_HIERARCHICAL_ENABLE_THRESHOLD && !isScalable) {
        /* 首节点日志，建链失败属常见问题，在建链前记录相关信息 */
        HCCL_RUN_INFO(
            "[HCCL_TRACE][Hierarchical]SetupAgent rankNum[%u], rank[%u], rootInfo identifier[%s], server[%s], "
            "serverPort[%u]"
            "deviceType[%d], logicDevId[%d], phydevId[%d], deviceIp[%s]",
            rankSize, myrank, rootInfo.identifier, localRankInfo_.hostIP.GetReadableAddress(), rootInfo.port,
            localRankInfo_.deviceType, localRankInfo_.deviceLogicID, localRankInfo_.devicePhysicID,
            localRankInfo_.deviceIP[0].GetReadableIP());

        pTopoExchangeAgent_.reset(new (nothrow) TopoInfoExchangeAgent(
            rootIP, rootInfo.port, rootInfo.identifier, agentPortCtx_, localRankInfo_, rankHandle));
        CHK_SMART_PTR_NULL(pTopoExchangeAgent_);
        CHK_RET(pTopoExchangeAgent_->SetIsInterSuperPodRetryEnable(isInterSuperPodRetryEnable_));
        CHK_RET(pTopoExchangeAgent_->Setup());
        CHK_RET(pTopoExchangeAgent_->GetGroupLeader(grpLeader_));
    } else {
        /* 首节点日志，建链失败属常见问题，在建链前记录相关信息 */
        HCCL_RUN_INFO(
            "[HCCL_TRACE][Flat]SetupAgent rankNum[%u], rank[%u], rootInfo identifier[%s], server[%s], serverPort[%u]"
            "deviceType[%d], logicDevId[%d], phydevId[%d], deviceIp[%s]",
            rankSize, myrank, rootInfo.identifier, localRankInfo_.hostIP.GetReadableAddress(), rootInfo.port,
            localRankInfo_.deviceType, localRankInfo_.deviceLogicID, localRankInfo_.devicePhysicID,
            localRankInfo_.deviceIP[0].GetReadableIP());

        pTopoExchangeAgent_.reset(new (nothrow) TopoInfoExchangeAgent(
            rootIP, rootInfo.port, rootInfo.identifier, agentPortCtx_, localRankInfo_));
        CHK_SMART_PTR_NULL(pTopoExchangeAgent_);
        CHK_RET(pTopoExchangeAgent_->SetIsInterSuperPodRetryEnable(isInterSuperPodRetryEnable_));
        CHK_RET(pTopoExchangeAgent_->SetIsScalable(isScalable));
        CHK_RET(pTopoExchangeAgent_->Setup());
        CHK_RET(pTopoExchangeAgent_->GetClusterTopoInfo(clusterTopoInfo_));
    }

    rootInfo_ = rootInfo;

    HCCL_INFO(
        "topo detect completed. myrank[%u], totalranks[%u], myhost[%s], totalservers[%u].", myrank, rankSize,
        localRankInfo_.hostIP.GetReadableAddress(), clusterTopoInfo_.serverNum);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::SetupRank(std::shared_ptr<HcclSocket>& agentConnRoot)
{
    CHK_RET(pTopoExchangeAgent_->SetupRank(agentConnRoot));
    CHK_RET(pTopoExchangeAgent_->GetGroupLeader(grpLeader_));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::TeardownAgent()
{
    bool bInitDevNic = clusterTopoInfo_.rankNum != 1 ? true : false;
    HcclIpAddress hostIP = GetBootstrapHostIP();

    if (!pTopoExchangeAgent_) { // 异常处理：如果没有创建agent，对标SetupAgent函数中的网络操作，则直接StopNetwork
        auto ret = StopNetwork(hostIP, bInitDevNic);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]topo detect agent stop network failed!"), ret);
        return HCCL_SUCCESS;
    }
    CHK_RET(pTopoExchangeAgent_->Teardown());

    auto ret = StopNetwork(hostIP, bInitDevNic);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]topo detect agent stop network failed!"), ret);
    HCCL_INFO("TopoInfoDetect TeardownAgent ok, identifier[%s].", rootInfo_.identifier);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::SetupAgentByMasterInfo(HcclIpAddress& localHostIp, const HcclRootHandle& rootInfo)
{
    CHK_RET(hrtGetDevice(&deviceLogicID_));
    SetBootstrapHostIP(localHostIp);
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_));
    rootInfo_ = rootInfo;
    HcclResult ret = CheckHostNicLinkUp(localHostIp);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Setup][Agent]host NIC status check failed, deviceLogicID[%d], devicePhysicID[%u], hostIP[%s], "
            "ret[%d].",
            deviceLogicID_, devicePhysicID_, localHostIp.GetReadableAddress(), ret),
        ret);
    bool bInitDevNic = GetExternalInputMasterInfo().rankSize != 1 ? true : false;
    ret = StartNetwork(localHostIp, bInitDevNic);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]topo detect agent start network failed!"), ret);

    bool errorFlag = false;
    do {
        HcclIpAddress rootIP(rootInfo.ip);
        CHK_PRT_BREAK(
            rootIP.IsInvalid(), HCCL_ERROR("[Setup][Agent]string[%s] is invalid ip", rootInfo.ip), errorFlag = true);
        ret = GenerateLocalRankInfo(GetExternalInputMasterInfo().rankSize, INVALID_VALUE_RANKID, localRankInfo_);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]topo detect generate local rank info failed"),
            errorFlag = true);

        pTopoExchangeAgent_.reset(new (nothrow) TopoInfoExchangeAgent(
            rootIP, rootInfo.port, rootInfo.identifier, agentPortCtx_, localRankInfo_));
        if (pTopoExchangeAgent_ == nullptr) {
            HCCL_ERROR("[Setup][Agent]pTopoExchangeAgent is nullptr");
            errorFlag = true;
            ret = HCCL_E_PTR;
            break;
        }

        CHK_RET(pTopoExchangeAgent_->SetIsInterSuperPodRetryEnable(isInterSuperPodRetryEnable_));
        ret = pTopoExchangeAgent_->SetupByMasterInfo();
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]setup by masterInfo failed"), errorFlag = true);
    } while (0);

    if (errorFlag) {
        // 如果StartNetwork后执行有报错，则先StopNetwork，再返回
        HcclResult result = StopNetwork(localHostIp, bInitDevNic);
        CHK_PRT_RET(result != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]topo detect agent stop network failed!"), result);

        HCCL_ERROR("[Setup][Agent]topo detect agent failed, return[%d]", ret);
        return ret;
    }

    CHK_RET(pTopoExchangeAgent_->GetClusterTopoInfo(clusterTopoInfo_));
    CHK_RET(pTopoExchangeAgent_->GetIdentifier(identifierNum_));

    HCCL_INFO(
        "topo detect completed. deviceLogicID[%u] totalranks[%u], myhost[%s], totalservers[%u].", deviceLogicID_,
        GetExternalInputMasterInfo().rankSize, localRankInfo_.hostIP.GetReadableAddress(), clusterTopoInfo_.serverNum);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::WaitComplete(const HcclRootHandle& rootInfo)
{
    return WaitTopoExchangeServerCompelte(rootInfo.port);
}

HcclResult TopoInfoDetect::Teardown()
{
    CHK_RET(TeardownAgent());
    CHK_RET(TeardownServer());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::ReadHostSocketWhitelist(vector<HcclIpAddress>& whitelist) const
{
    RPT_ENV_ERR(
        (GetExternalInputHcclWhiteListFile().length() == 0), "EI0001",
        std::vector<std::string>({"value", "env", "expect"}),
        vector<string>({"", "HCCL_WHITELIST_FILE", "a valid file path"}));

    CHK_PRT_RET(
        (GetExternalInputHcclWhiteListFile().length() == 0),
        HCCL_ERROR(
            "[%s][%s]environmental variable HCCL_WHITELIST_DISABLE is [0], "
            "but HCCL_WHITELIST_FILE is not set or not exist",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str()),
        HCCL_E_PARA);

    // 文件路径在处理外部输入时已经做过合法性判断, 无需再次校验
    HcclResult ret = HcclWhitelist::GetInstance().LoadConfigFile(GetExternalInputHcclWhiteListFile());

    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {GetExternalInputHcclWhiteListFile(), "HCCL_WHITELIST_FILE", "a valid whitelist file format"}));

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]hccl whitelist load config file[%s] failed. ret[%u].", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_ENV_CONFIG.c_str(), GetExternalInputHcclWhiteListFile().c_str(), ret),
        ret);
    CHK_RET(HcclWhitelist::GetInstance().GetHostWhiteList(whitelist));

    CHK_PRT_RET(
        whitelist.empty(),
        HCCL_ERROR(
            "[%s][%s]whitelist file[%s] have no valid host ip.", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_ENV_CONFIG.c_str(), GetExternalInputHcclWhiteListFile().c_str()),
        HCCL_E_UNAVAIL);
    HCCL_INFO("get host socket whitelist success. there are %zu host ip in the whitelist.", whitelist.size());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetAllHostIfInfos(vector<pair<string, HcclIpAddress>>& ifInfos, u32 devPhyId) const
{
    CHK_RET(hrtGetHostIf(ifInfos, devPhyId));

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetAllValidHostIfInfos(
    const vector<HcclIpAddress>& whitelist, vector<pair<string, HcclIpAddress>>& ifInfos, u32 devPhyId) const
{
    vector<pair<string, HcclIpAddress>> orginIfInfos;
    CHK_RET(GetAllHostIfInfos(orginIfInfos, devPhyId));

    for (auto& ifInfo : orginIfInfos) {
        auto iter = find(whitelist.begin(), whitelist.end(), ifInfo.second);
        if (iter != whitelist.end()) {
            ifInfos.push_back({ifInfo.first, ifInfo.second});
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetRootHostIP(const vector<HcclIpAddress>& whitelist, HcclIpAddress& ip, u32 devPhyId)
{
    if (!ip.IsInvalid()) {
        return HCCL_SUCCESS;
    }
    vector<pair<string, HcclIpAddress>> ifInfos;

    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(GetAllValidHostIfInfos(whitelist, ifInfos, devPhyId));
        CHK_PRT_RET(
            ifInfos.empty(), HCCL_ERROR("[Get][RootHostIP]there is no valid host if in whitelist."), HCCL_E_NOT_FOUND);
    } else {
        CHK_RET(GetAllHostIfInfos(ifInfos, devPhyId));
        CHK_PRT_RET(ifInfos.empty(), HCCL_ERROR("[Get][RootHostIP]there is no host if."), HCCL_E_NOT_FOUND);
    }

    CHK_RET(FindLocalHostIP(ifInfos, ip));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetGroupLeader(HcclRankHandle& rankHandle)
{
    rankHandle = grpLeader_;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::SetIsInterSuperPodRetryEnable(bool isRetry)
{
    isInterSuperPodRetryEnable_ = isRetry;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::StartRootNetwork(
    const HcclIpAddress& hostIP, u32& usePort, const std::vector<HcclSocketPortRange>& portRanges)
{
    CHK_RET(HcclNetOpenDev(&serverPortCtx_, NicType::HOST_NIC_TYPE, devicePhysicID_, deviceLogicID_, hostIP));
    CHK_PTR_NULL(serverPortCtx_);
    return StartListenNetwork(listenSocket_, serverPortCtx_, hostIP, usePort, portRanges);
}

// 监听socket启动：未指定端口时通过抢占获得监听端口，否则使用固定端口监听。
// Root节点agent监听（StartRootNetwork）与多root（scalable）的mesh全互联监听共用。
HcclResult TopoInfoDetect::StartListenNetwork(
    std::shared_ptr<HcclSocket>& listenSocket, HcclNetDevCtx netDevCtx, const HcclIpAddress& hostIP, u32& usePort,
    const std::vector<HcclSocketPortRange>& portRanges)
{
    if (usePort == HCCL_INVALID_PORT) {
        // 通过抢占的方式获得监听端口
        listenSocket.reset(new (nothrow) HcclSocket(netDevCtx));
        CHK_SMART_PTR_NULL(listenSocket);
        CHK_RET(listenSocket->Init());
        HcclResult ret
            = PreemptPortManager::GetInstance(deviceLogicID_).ListenPreempt(listenSocket, portRanges, usePort);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TopoInfoDetect][StartListenNetwork] devPhyId[%u], devLogicId[%u], host ip[%s], "
                "try to preempt port on host nic fail.",
                devicePhysicID_, deviceLogicID_, hostIP.GetReadableAddress()),
            ret);
    } else {
        // 1. 使用MasterInfo初始化时，不支持抢占master节点的监听端口
        // 2. 未配置port range时，不支持抢占监听端口
        listenSocket.reset(new (nothrow) HcclSocket(netDevCtx, usePort));
        CHK_SMART_PTR_NULL(listenSocket);
        CHK_RET(listenSocket->Init());
        HcclResult ret = listenSocket->Listen();
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s][%s]%s failed, ret[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                __func__, ret),
            ret);
    }

    HCCL_INFO("listen socket start with host ip[%s] and port[%u]", hostIP.GetReadableAddress(), usePort);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::StartGroupLeaderNetwork(
    const vector<HcclIpAddress>& whitelist, const HcclIpAddress& hostIP, u32& bindPort,
    const std::vector<HcclSocketPortRange>& portRanges)
{
    CHK_RET(HcclNetOpenDev(&serverPortCtx_, NicType::HOST_NIC_TYPE, devicePhysicID_, deviceLogicID_, hostIP));
    CHK_PTR_NULL(serverPortCtx_);
    if (bindPort == HCCL_INVALID_PORT) {
        // 通过抢占的方式获得GroupLeader节点监听的host端口
        listenSocket_.reset(new (nothrow) HcclSocket(serverPortCtx_));
        CHK_SMART_PTR_NULL(listenSocket_);
        CHK_RET(listenSocket_->Init());
        HcclResult ret
            = PreemptPortManager::GetInstance(deviceLogicID_).ListenPreempt(listenSocket_, portRanges, bindPort);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TopoInfoDetect][StartGroupLeaderNetwork] devPhyId[%u], devLogicId[%u], host ip[%s], "
                "try to preempt port on host nic fail.",
                devicePhysicID_, deviceLogicID_, hostIP.GetReadableAddress()),
            ret);
    } else {
        // 1. 使用MasterInfo初始化时，不支持抢占master节点的监听端口
        // 2. 未配置port range时，不支持抢占监听端口
        listenSocket_.reset(new (nothrow) HcclSocket(serverPortCtx_, bindPort));
        CHK_SMART_PTR_NULL(listenSocket_);
        CHK_RET(listenSocket_->Init());
        HcclResult ret = listenSocket_->Listen();
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s][%s]%s failed, ret[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                __func__, ret),
            ret);
    }

    HCCL_INFO("group leader start with host ip[%s] and port[%u]", hostIP.GetReadableAddress(), bindPort);

    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(AddSocketWhiteList(bindPort, whitelist));
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::AddSocketWhiteList(u32 port, const vector<HcclIpAddress>& whitelist) const
{
    vector<SocketWlistInfo> wlistInfosVec;
    for (auto ip : whitelist) {
        SocketWlistInfo wlistInfo;
        wlistInfo.connLimit = HOST_SOCKET_CONN_LIMIT;
        wlistInfo.remoteIp.addr = ip.GetBinaryAddress().addr;
        wlistInfo.remoteIp.addr6 = ip.GetBinaryAddress().addr6;
        string tag = TOPO_DETECT_TAG + "_" + rootInfo_.identifier + "_" + to_string(port);
        s32 sRet = memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), tag.c_str(), tag.size() + 1);
        if (sRet != EOK) {
            HCCL_ERROR("[Add][SocketWhiteList]memory copy failed. errorno[%d]", sRet);
            return HCCL_E_MEMORY;
        }
        wlistInfosVec.push_back(wlistInfo);
    }

    CHK_RET(listenSocket_->AddWhiteList(wlistInfosVec));

    HCCL_INFO("add socket white list success. total: %zu", whitelist.size());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::StartNetwork(HcclIpAddress& hostIP, bool bInitDevNic)
{
    CHK_RET(HcclNetOpenDev(&agentPortCtx_, NicType::HOST_NIC_TYPE, devicePhysicID_, deviceLogicID_, hostIP));
    CHK_PTR_NULL(agentPortCtx_);

    if (bInitDevNic) {
        CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhysicID_, deviceLogicID_, false));
        CHK_RET(
            HcclNetOpenDev(&devNicCtx_, NicType::DEVICE_NIC_TYPE, devicePhysicID_, deviceLogicID_, HcclIpAddress(0)));
        CHK_PTR_NULL(devNicCtx_);
    }

    HCCL_INFO("NetworkManager start host net success! ip[%s]", hostIP.GetReadableAddress());

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::StopNetwork(HcclIpAddress& hostIP, bool bInitDevNic)
{
    if (agentPortCtx_) {
        HcclNetCloseDev(agentPortCtx_);
        agentPortCtx_ = nullptr;
        CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_));
    }

    if (bInitDevNic) {
        if (devNicCtx_) {
            HcclNetCloseDev(devNicCtx_);
            devNicCtx_ = nullptr;
            CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhysicID_, deviceLogicID_));
        }
    }

    HCCL_INFO("NetworkManager stop host net success! ip[%s] ", hostIP.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::FilterDevIPs(
    std::vector<HcclIpAddress>& sourceDeviceIPs, std::vector<HcclIpAddress>& targetDeviceIPs) const
{
    std::vector<HcclIpAddress> deviceIPv4;
    std::vector<HcclIpAddress> deviceIPv6;
    for (auto& iter : sourceDeviceIPs) {
        if (iter.IsIPv6()) {
            deviceIPv6.push_back(iter);
        } else {
            deviceIPv4.push_back(iter);
        }
    }
    // 同时存在ipv4/ipv6时，除非指定socket family，否则ipv4优先
    // 只存在ipv4/ipv6单栈时，不受用户指定的socket family约束
    if ((((GetExternalInputHcclSocketFamily() == -1) || (GetExternalInputHcclSocketFamily() == AF_INET))
         && (!deviceIPv4.empty()))
        || deviceIPv6.empty()) {
        targetDeviceIPs = deviceIPv4;
        HCCL_RUN_INFO("select AF_INET family as device socket family.");
    } else if (!deviceIPv6.empty()) {
        std::sort(deviceIPv6.begin(), deviceIPv6.end());
        targetDeviceIPs.push_back(deviceIPv6[0]);
        HCCL_RUN_INFO("select AF_INET6 family as device socket family.");
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::PreemptDeviceNicPort(
    const u32 devPhyId, const s32 devLogicId, const HcclIpAddress& deviceIp, u32& usePort)
{
    HcclIpAddress devIp(std::string(deviceIp.GetReadableIP()));
    HcclNetDevCtx netCtx{nullptr};
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devPhyId, devLogicId, false, false));
    CHK_RET(HcclNetOpenDev(&netCtx, NicType::DEVICE_NIC_TYPE, devPhyId, devLogicId, devIp));
    CHK_PTR_NULL(netCtx);
    commPortConfig_.devNicListen = std::make_pair(nullptr, netCtx);

    commPortConfig_.devNicListen.first.reset(new (std::nothrow) HcclSocket(netCtx));
    CHK_SMART_PTR_NULL(commPortConfig_.devNicListen.first);
    CHK_RET(commPortConfig_.devNicListen.first->Init());

    HcclResult ret
        = PreemptPortManager::GetInstance(devLogicId)
              .ListenPreempt(commPortConfig_.devNicListen.first, GetExternalInputNpuSocketPortRange(), usePort);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TopoInfoDetect][PreemptDeviceNicPort] devPhyId[%u], devLogicId[%u], device ip[%s], "
            "try to preempt port on device nic fail.",
            devPhyId, devLogicId, devIp.GetReadableAddress()),
        ret);

    HCCL_INFO(
        "[TopoInfoDetect][PreemptDeviceNicPort]devPhyId[%u], devLogicId[%d], "
        "preempt port[%u] on ip[%s] success.",
        devPhyId, devLogicId, usePort, devIp.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::PreemptDeviceVnicPort(HcclBasicRankInfo& localRankInfo)
{
    u32 devPhyId = localRankInfo.devicePhysicID;
    s32 devLogicId = localRankInfo.deviceLogicID;
    HcclIpAddress vnicIp(devPhyId);
    bool useSuperPodMode = false;
    if (localRankInfo.superDeviceId != INVALID_UINT) {
        CHK_RET(IsSuperPodMode(useSuperPodMode));
    }
    if (useSuperPodMode) {
        CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
            devPhyId, DeviceIdType::DEVICE_ID_TYPE_SDID, localRankInfo.superDeviceId, vnicIp));
    } else {
        CHK_RET(hrtRaGetSingleSocketVnicIpInfo(devPhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, devPhyId, vnicIp));
    }

    HCCL_INFO("[TopoInfoDetect][PreemptDeviceVnicPort] vnicIp is [%s]", vnicIp.GetReadableAddress());

    HcclNetDevCtx netCtx{nullptr};
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devPhyId, devLogicId, false, false));
    CHK_RET(HcclNetOpenDev(&netCtx, NicType::VNIC_TYPE, devPhyId, devLogicId, vnicIp));
    CHK_PTR_NULL(netCtx);
    commPortConfig_.devVnicListen = std::make_pair(nullptr, netCtx);

    commPortConfig_.devVnicListen.first.reset(new (std::nothrow) HcclSocket(netCtx));
    CHK_SMART_PTR_NULL(commPortConfig_.devVnicListen.first);
    CHK_RET(commPortConfig_.devVnicListen.first->Init());

    HcclResult ret = PreemptPortManager::GetInstance(devLogicId)
                         .ListenPreempt(
                             commPortConfig_.devVnicListen.first, GetExternalInputNpuSocketPortRange(),
                             localRankInfo.deviceVnicPort);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TopoInfoDetect][PreemptDeviceVnicPort] devPhyId[%u], devLogicId[%u], vnicIp[%s], "
            "try to preempt port on vnic fail.",
            devPhyId, devLogicId, vnicIp.GetReadableAddress()),
        ret);

    HCCL_INFO(
        "[TopoInfoDetect][PreemptDeviceVnicPort] devPhyId[%u], devLogicId[%d], "
        "preempt vnic on ip[%s], port[%u] success.",
        devPhyId, devLogicId, vnicIp.GetReadableAddress(), localRankInfo.deviceVnicPort);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::PreemptBackupDeviceNicPort(
    const u32 devPhyId, const s32 devLogicId, const HcclIpAddress& deviceIp, const HcclIpAddress& backupDeviceIp,
    u32& usePort)
{
    HcclIpAddress devIp(std::string(deviceIp.GetReadableIP()));
    HcclIpAddress backupDevIp(std::string(backupDeviceIp.GetReadableIP()));
    HcclNetDevCtx netCtx{nullptr};
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devPhyId, devLogicId, false, true));
    CHK_RET(HcclNetOpenDev(&netCtx, NicType::DEVICE_NIC_TYPE, devPhyId, devLogicId, backupDevIp, devIp));
    CHK_PTR_NULL(netCtx);
    commPortConfig_.backupDevNicListen = std::make_pair(nullptr, netCtx);

    commPortConfig_.backupDevNicListen.first.reset(new (std::nothrow) HcclSocket(netCtx));
    CHK_SMART_PTR_NULL(commPortConfig_.backupDevNicListen.first);
    CHK_RET(commPortConfig_.backupDevNicListen.first->Init());

    HcclResult ret
        = PreemptPortManager::GetInstance(devLogicId)
              .ListenPreempt(commPortConfig_.backupDevNicListen.first, GetExternalInputNpuSocketPortRange(), usePort);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TopoInfoDetect][PreemptBackupDeviceNicPort] devPhyId[%u], devLogicId[%u], device ip[%s], "
            "backup device ip[%s], try to preempt port on device nic fail.",
            devPhyId, devLogicId, devIp.GetReadableAddress(), backupDevIp.GetReadableAddress()),
        ret);

    HCCL_INFO(
        "[TopoInfoDetect][PreemptBackupDeviceNicPort]devPhyId[%u], devLogicId[%d], local ip[%s]"
        "preempt port[%u] on backup device ip[%s] success[%u].",
        devPhyId, devLogicId, devIp.GetReadableAddress(), usePort, backupDevIp.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetDeviceBackupNicInfo(HcclBasicRankInfo& localRankInfo)
{
    std::vector<std::vector<HcclIpAddress>> chipDeviceIPs;
    CHK_RET(hrtRaGetDeviceAllNicIP(chipDeviceIPs));
    if (chipDeviceIPs.size() != 2U) {
        // 910A3场景一个chip上有两组deviceIP
        HCCL_RUN_WARNING("[TopoInfoDetect][GetDeviceBackupNicInfo]Fail to load backup device ip!"
                         "Please check the driver version!");
    } else {
        // 取到一组backup ip，按照devPhyId排序，ipv4在前，ipv6在后，每个网卡的ip顺序一一对应
        // 取其中对端网卡的ip作为备用网卡ip
        u32 ipIdex = 1U - (localRankInfo.devicePhysicID % 2U);
        CHK_RET(FilterDevIPs(chipDeviceIPs[ipIdex], localRankInfo.backupDeviceIP));
        HCCL_INFO(
            "[TopoInfoDetect][GetDeviceBackupNicInfo]devicePhysicID[%u], backupDeviceIP[0]:[%s]",
            localRankInfo.devicePhysicID, localRankInfo.backupDeviceIP[0].GetReadableAddress());
        // 开启device侧端口配置，并且存在备用网卡时，抢占一个备用网卡上的端口
        if (GetExternalInputNpuPortSwitch() && localRankInfo.backupDeviceIP.size() > 0) {
            u32 backupDevPhyId = INVALID_INT;
            u32 backupDevLogicId = INVALID_INT;
            CHK_RET(hrtGetPairDevicePhyId(localRankInfo.devicePhysicID, backupDevPhyId));
            CHK_RET(hrtGetDeviceIndexByPhyId(backupDevPhyId, backupDevLogicId));
            CHK_RET(PreemptBackupDeviceNicPort(
                backupDevPhyId, backupDevLogicId, localRankInfo.deviceIP[0], localRankInfo.backupDeviceIP[0],
                localRankInfo.backupDevicePort));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GenerateLocalRankInfo(u32 rankSize, u32 rankID, HcclBasicRankInfo& localRankInfo)
{
    localRankInfo.hostIP = GetBootstrapHostIP();
    localRankInfo.rank = rankID;
    localRankInfo.rankSize = rankSize;
    localRankInfo.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    if (devNicCtx_ != nullptr) {
        HcclResult ret = HcclNetDevGetTlsStatus(devNicCtx_, &localRankInfo.tlsStatus);
        if (ret != HCCL_SUCCESS && ret != HCCL_E_NOT_SUPPORT) {
            HCCL_RUN_WARNING("[GenerateLocalRankInfo] HcclNetDevGetTlsStatus failed ret[%u]", ret);
        }
    }

    CHK_RET(hrtGetDeviceType(localRankInfo.deviceType));
    CHK_RET(hrtGetDevice(reinterpret_cast<s32*>(&localRankInfo.deviceLogicID)));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(localRankInfo.deviceLogicID), localRankInfo.devicePhysicID));

    if (localRankInfo.deviceType == DevType::DEV_TYPE_910_93) {
        CHK_RET(GetSuperPodInfo(localRankInfo.deviceLogicID, localRankInfo.superPodId, localRankInfo.superDeviceId));
    }

    localRankInfo.deviceIP.clear();
    if (localRankInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && rankSize != 1) {
        std::vector<HcclIpAddress> deviceIPs;
        CHK_RET(hrtRaGetDeviceIP(localRankInfo.devicePhysicID, deviceIPs));
        CHK_RET(FilterDevIPs(deviceIPs, localRankInfo.deviceIP));
        // 开启device侧端口配置时，需要抢占监听端口
        if (GetExternalInputNpuPortSwitch()) {
            // 如果有device nic，则抢占device nic的port
            if (localRankInfo.deviceIP.size() > 0) {
                CHK_RET(PreemptDeviceNicPort(
                    localRankInfo.devicePhysicID, localRankInfo.deviceLogicID, localRankInfo.deviceIP[0],
                    localRankInfo.deviceNicPort));
            }
            // 使用device网卡时，必定抢占vnic上的port
            CHK_RET(PreemptDeviceVnicPort(localRankInfo));
            commPortConfig_.devPortSwitchOn = true;
        }

        // 此处不知道拓扑形态，无法判断是否需要backupIp，只能从硬件类型和重执行开关判断一下
        bool useSuperPodMode = false;
        CHK_RET(IsSuperPodMode(useSuperPodMode));
        if (useSuperPodMode && commConfig_.GetConfigAicpuUnfold() && isInterSuperPodRetryEnable_) {
            CHK_RET(GetDeviceBackupNicInfo(localRankInfo));
        }
    }

    if (localRankInfo.deviceIP.size() == 0) {
        // 和 rank table 保持一致，如果没有device网卡时，默认填充 0。
        HcclIpAddress invalidAddr;
        localRankInfo.deviceIP.push_back(invalidAddr);
        HCCL_RUN_INFO("no device ip: use 0 as device ip.");
    }
    if (localRankInfo.backupDeviceIP.size() == 0) {
        // 如果没有 backup device ip 时，默认填充 0。
        HcclIpAddress invalidAddr;
        localRankInfo.backupDeviceIP.push_back(invalidAddr);
        HCCL_RUN_INFO("no backup device ip: use 0 as device ip.");
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetSuperPodInfo(s32 deviceLogicId, std::string& superPodId, u32& superDeviceId)
{
    // 解析super_pod_id
    superPodId = GetExternalInputLogicSuperPodId(); // 逻辑super pod id
    if (superPodId.empty()) {
        s64 val = 0;
        CHK_RET(hrtGetDeviceInfo(
            deviceLogicId, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
            HcclRtDeviceInfoType::HCCL_INFO_TYPE_SUPER_POD_ID, val));
        superPodId = std::to_string(val); // 真实super pod id
    }

    // 解析sdid
    s64 sdid = 0;
    CHK_RET(hrtGetDeviceInfo(
        deviceLogicId, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM, HcclRtDeviceInfoType::HCCL_INFO_TYPE_SDID,
        sdid));
    superDeviceId = static_cast<u32>(sdid);
    HCCL_INFO(
        "[Get][SuperPodInfo]deviceLogicID[%d], superPodId[%s], superDeviceId[%u]", deviceLogicId, superPodId.c_str(),
        superDeviceId);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetClusterInfo(RankTable_t& clusterInfo)
{
    CHK_PRT_RET(
        (clusterTopoInfo_.rankList.size() == 0),
        HCCL_ERROR("[Get][ClusterInfo]GetClusterInfo failed, topo detect has not started."), HCCL_E_INTERNAL);
    clusterInfo = clusterTopoInfo_;
    return HCCL_SUCCESS;
}
HcclResult TopoInfoDetect::GetRankId(u32& rankId)
{
    rankId = identifierNum_;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetLocalRankInfo(HcclBasicRankInfo& rankInfo)
{
    CHK_PRT_RET(
        (localRankInfo_.rankSize == 0),
        HCCL_ERROR("[Get][LocalRankInfo]GetLocalRankInfo failed, topo "
                   "detect has not started."),
        HCCL_E_INTERNAL);
    rankInfo = localRankInfo_;
    return HCCL_SUCCESS;
}

void TopoInfoDetect::SetBootstrapHostIP(HcclIpAddress& ip) { bootstrapHostIP_ = ip; }

HcclIpAddress TopoInfoDetect::GetBootstrapHostIP() const { return bootstrapHostIP_; }
HcclResult TopoInfoDetect::TransformRankTableStr(const RankTable_t& clusterInfo, string& ranktableStr)
{
    nlohmann::json basicJson;
    HcclResult ret = Struct2JsonRankTable(clusterInfo, basicJson);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_WARNING("cluster info to json failed ,ret[%d]", ret), HCCL_E_INTERNAL);
    ranktableStr = basicJson.dump(2); // dump参数为2
    return HCCL_SUCCESS;
}
HcclResult TopoInfoDetect::TransformDeviceList(
    const RankTable_t& clusterInfo, vector<RankInfo_t>& tmpRankList, nlohmann::json& perServerJson,
    u32 serverIndex) const
{
    for (auto it = tmpRankList.begin(); it != tmpRankList.end();) {
        if (it->serverId == clusterInfo.serverList[serverIndex].serverId) {
            nlohmann::json perDeviceJson;
            perDeviceJson[PROP_DEV_ID] = to_string(it->deviceInfo.devicePhyId);
            perDeviceJson[PROP_RANK_ID] = to_string(it->rankId);
            perDeviceJson[PROP_SUPER_DEVICE_ID] = to_string(it->superDeviceId);
            if (it->deviceInfo.port != HCCL_INVALID_PORT) {
                perDeviceJson[PROP_DEV_NIC_PORT] = to_string(it->deviceInfo.port);
            }
            if (it->deviceInfo.vnicPort != HCCL_INVALID_PORT) {
                perDeviceJson[PROP_DEV_VNIC_PORT] = to_string(it->deviceInfo.vnicPort);
            }
            if (it->deviceInfo.backupPort != HCCL_INVALID_PORT) {
                perDeviceJson[PROP_BACKUP_DEV_PORT] = to_string(it->deviceInfo.backupPort);
            }
            if (clusterInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && it->deviceInfo.deviceIp.size() != 0
                && !it->deviceInfo.deviceIp[0].IsInvalid()) {
                perDeviceJson[PROP_DEV_IP] = std::string(it->deviceInfo.deviceIp[0].GetReadableIP());
            }
            if (clusterInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE
                && it->deviceInfo.backupDeviceIp.size() != 0 && !it->deviceInfo.backupDeviceIp[0].IsInvalid()) {
                perDeviceJson[PROP_BACKUP_DEV_IP] = std::string(it->deviceInfo.backupDeviceIp[0].GetReadableIP());
            }
            if (!it->hostIp.IsInvalid()) {
                perServerJson[PROP_HOST_IP] = std::string(it->hostIp.GetReadableIP());
            }
            perServerJson[PROP_DEVICE].push_back(perDeviceJson);
            it = tmpRankList.erase(it);
        } else {
            it++;
        }
    }
    return HCCL_SUCCESS;
}
HcclResult TopoInfoDetect::Struct2JsonRankTable(const RankTable_t& clusterInfo, nlohmann::json& ClusterJson)
{
    nlohmann::json serverListJson;
    ClusterJson[PROP_SERVER_COUNT] = to_string(clusterInfo.serverNum);
    vector<RankInfo_t> tmpRankList = clusterInfo.rankList;
    ClusterJson[PROP_SERVER_LIST] = serverListJson;
    for (u32 i = 0; i < clusterInfo.serverNum; i++) {
        nlohmann::json perServerJson;
        perServerJson[PROP_SERVER_ID] = clusterInfo.serverList[i].serverId;
        nlohmann::json deviceList;
        perServerJson[PROP_DEVICE] = deviceList;
        CHK_RET(TransformDeviceList(clusterInfo, tmpRankList, perServerJson, i));
        ClusterJson[PROP_SERVER_LIST].push_back(perServerJson);
    }

    nlohmann::json superPodListJson;
    CHK_RET(TransformSuperPodList(clusterInfo.rankList, superPodListJson));
    ClusterJson[PROP_SUPER_POD_LIST] = superPodListJson;

    ClusterJson[PROP_STATUS] = "completed";
    ClusterJson[PROP_VERSION] = (localRankInfo_.deviceType == DevType::DEV_TYPE_910_93) ? "1.2" : "1.0";
    return HCCL_SUCCESS;
}

HcclResult
TopoInfoDetect::TransformSuperPodList(const std::vector<RankInfo_t>& rankInfo, nlohmann::json& superPodListJson) const
{
    // 按照 <super_pod_id, <server_id>> 格式从RankInfo_t中解析super pod信息
    std::map<std::string, std::set<std::string>> superPodMap;
    for (u32 i = 0; i < rankInfo.size(); i++) {
        auto iter = superPodMap.find(rankInfo[i].superPodId);
        if (iter == superPodMap.end()) {
            std::set<std::string> perSuperPod;
            perSuperPod.insert(rankInfo[i].serverId);
            superPodMap.insert(std::pair<std::string, std::set<string>>(rankInfo[i].superPodId, perSuperPod));
        } else {
            // superDeviceId在VerifyClusterSuperPodInfo中已经查重校验过
            iter->second.insert(rankInfo[i].serverId);
        }
    }

    for (auto it = superPodMap.begin(); it != superPodMap.end(); ++it) {
        nlohmann::json superPodIdJson;
        superPodIdJson[PROP_SUPER_POD_ID] = it->first;
        nlohmann::json serverListJson;
        for (auto perServer = it->second.begin(); perServer != it->second.end(); ++perServer) {
            nlohmann::json perServerJson;
            perServerJson[PROP_SERVER_ID] = *perServer;
            serverListJson.push_back(perServerJson);
        }
        superPodIdJson[PROP_SERVER_LIST] = serverListJson;
        superPodListJson.push_back(superPodIdJson);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
