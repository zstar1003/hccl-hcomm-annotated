/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_rt.h"
#include "rank_info_detect.h"
#include <thread>
#include <stdio.h>
#include "sal.h"
#include "rank_info_detect_service.h"
#include "hccp_peer_manager.h"
#include "hccp_hdc_manager.h"
#include "internal_exception.h"
#include "network_api_exception.h"
#include "null_ptr_exception.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "whitelist.h"
#include "socket.h"
#include "host_socket_handle_manager.h"
#include "env_config/env_config_v2.h"
#include "root_handle_v2.h"
#include "bootstrap_ip.h"
#include "preempt_port_manager_v2.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

constexpr u32 HOST_CONTROL_BASE_PORT = 60000; // 控制面起始port
constexpr u32 HOST_CONTROL_PORT_COUNT = 15;
constexpr u32 HCCL_WHITELIST_ON = 1;
constexpr u32 HOST_SOCKET_CONN_LIMIT = 8; // HCCL_AISERVER_DEVICE_NUM (8)

UniversalConcurrentMap<u32, u32> RankInfoDetect::g_detectServerStatus_;

RankInfoDetect::RankInfoDetect()
{
    userDevId_ = HrtGetDevice();
    aclError aclRet = aclrtGetLogicDevIdByUserDevId(userDevId_, &devLogicId_); // userDevId 转 logicDevId
    CHK_PRT_THROW(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[RankInfoDetect::%s] aclrtGetLogicDevIdByUserDevId failed, userDevId[%d], ret[%d]", __func__, userDevId_,
            aclRet),
        InternalException, "get logicDevId fail");
    s32 deviceNum = HrtGetDeviceCount();
    CHK_PRT_THROW(
        userDevId_ >= deviceNum,
        HCCL_ERROR("[RankInfoDetect::%s] userDevId[%d] is invalid, deviceNum[%d].", __func__, userDevId_, deviceNum),
        InternalException, "get hostIp fail");
    // 临时处理，当前拓扑探测阶段本质上都是使用的userDevId_，当前获取到devLogicId_是为了后续整改
    // 如果使用devLogicId_会在多进程各自设置了RT_VISIBLE_DEVICES环境变量时，可能会因为当前进行无法访问某些devLogicId_代表的rank而报错
    devLogicId_ = userDevId_;

    s32 phyDevId = 0;
    aclRet = aclrtGetPhyDevIdByUserDevId(userDevId_, &phyDevId); // userDevId 转 phyDevId
    CHK_PRT_THROW(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[RankInfoDetect::%s] aclrtGetPhyDevIdByUserDevId failed, userDevId[%d], ret[%d]", __func__, userDevId_,
            aclRet),
        InternalException, "get phyDevId fail");
    devPhyId_ = static_cast<u32>(phyDevId);

    HCCL_INFO(
        "[RankInfoDetect::%s] end, deviceNum[%d], userDevId_[%d], devLogicId_[%d], devPhyId_[%u].", __func__, deviceNum,
        userDevId_, devLogicId_, devPhyId_);
}

void RankInfoDetect::SetupServer(HcclRootHandleV2& rootHandle)
{
    HCCL_DEBUG("[RankInfoDetect::%s] setup server start.", __func__);

    // host网卡使能
    HccpPeerManager::GetInstance().Init(devLogicId_);

    // 获取LocalHostIP
    hostIp_ = GetBootstrapIp(devPhyId_);
    CHK_PRT_THROW(
        hostIp_.IsInvalid(), HCCL_ERROR("[RankInfoDetect::%s] get hostIp fail.", __func__), InternalException,
        "get hostIp fail");

    // 获取端口号port
    hostPort_ = GetHostListenPort();

    // 1. 创建serverSocket，启动监听并获取实际端口
    shared_ptr<Socket> serverSocket = ServerInit();

    // 2. 构建rootHandle
    GetRootHandle(rootHandle);

    // 3. 实际端口和identifier就绪后，使用最终tag下发白名单
    HcclResult ret = GetHandleAndAddHostSocketWhitelist();
    if (ret == HCCL_E_PTR) {
        THROW<NullPtrException>("[RankInfoDetect::%s] get host socket handle failed, ret[%d].", __func__, ret);
    }
    if (ret == HCCL_E_NETWORK) {
        THROW<NetworkApiException>("[RankInfoDetect::%s] add host socket whitelist failed, ret[%d].", __func__, ret);
    }
    if (ret != HCCL_SUCCESS) {
        THROW<InternalException>(
            "[RankInfoDetect::%s] configure host socket whitelist failed, ret[%d].", __func__, ret);
    }

    // 4. 拉起线程，调用RankInfoDetectService.Run()，注意新线程中需要HrtSetDevice
    std::thread threadHandle(
        &RankInfoDetect::SetupRankInfoDetectService, this, hrtErrMGetErrorContextPub(), serverSocket, devLogicId_,
        devPhyId_, identifier_, wlistInfo_);
    threadHandle.detach();

    HCCL_DEBUG("[RankInfoDetect::%s] setup server end.", __func__);
}

SocketHandle RankInfoDetect::GetHostSocketHandle()
{
    HCCL_DEBUG("[RankInfoDetect::%s] server get host socket handle start.", __func__);

    // 获取socket句柄
    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Create(devPhyId_, hostIp_);

    // 白名单tag依赖实际监听端口和identifier，在GetRootHandle后再构造并下发
    hostSocketWlist_.clear();
    if (!EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable()) {
        Whitelist::GetInstance().GetHostWhiteList(hostSocketWlist_);
        CHK_PRT_THROW(
            hostSocketWlist_.empty(), HCCL_ERROR("[%s] whitelist file have no valid host ip.", __func__),
            InternalException, "get host ip error");
        u32 whiteListEnable = 1;
        HrtRaSocketSetWhiteListStatus(whiteListEnable);
    }

    HCCL_DEBUG("[RankInfoDetect::%s] get host socket handle success, socketHandle[%p].", __func__, hostSocketHandle);
    return hostSocketHandle;
}

shared_ptr<Socket> RankInfoDetect::ServerInit()
{
    HCCL_DEBUG("[RankInfoDetect::%s] server init start.", __func__);

    SocketHandle hccpHostSocketHandle = GetHostSocketHandle();
    std::shared_ptr<Socket> serverSocket = std::make_shared<Socket>(
        hccpHostSocketHandle, hostIp_, hostPort_, hostIp_, "server", SocketRole::SERVER, NicType::HOST_NIC_TYPE);
    if (hostPort_ == HCCL_INVALID_PORT) {
        auto portRange = EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange();
        if (portRange.empty()) {
            SocketPortRange defaultRange = {HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_COUNT};
            portRange.push_back(defaultRange);
        }
        PreemptPortManager::GetInstance(devLogicId_).ListenPreempt(serverSocket, portRange, hostPort_);
    } else {
        serverSocket->Listen();
    }

    HCCL_INFO("[RankInfoDetect::%s] serverSocket[%s] listen success.", __func__, serverSocket->Describe().c_str());
    return serverSocket;
}

HcclResult RankInfoDetect::GetHandleAndAddHostSocketWhitelist()
{
    // 同一对象重试SetupServer时，先清理上一次生成的条目，避免旧tag被重复下发
    wlistInfo_.clear();
    if (hostSocketWlist_.empty()) {
        return HCCL_SUCCESS;
    }

    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Get(devPhyId_, hostIp_);
    CHK_PRT_RET(
        hostSocketHandle == nullptr,
        HCCL_ERROR(
            "[RankInfoDetect::%s] get host socket handle failed before adding whitelist, "
            "devPhyId[%u], hostIp[%s], identifier[%s], listenPort[%u].",
            __func__, devPhyId_, hostIp_.GetIpStr().c_str(), identifier_.c_str(), hostPort_),
        HCCL_E_PTR);
    CHK_RET(AddHostSocketWhitelist(hostSocketHandle, hostSocketWlist_));
    return HCCL_SUCCESS;
}

HcclResult RankInfoDetect::AddHostSocketWhitelist(SocketHandle& socketHandle, const vector<IpAddress>& hostSocketWlist)
{
    HCCL_DEBUG("[RankInfoDetect::%s] start, hostSocketWlist size[%zu].", __func__, hostSocketWlist.size());

    for (auto& ipAddress : hostSocketWlist) {
        RaSocketWhitelist info{};
        info.remoteIp = ipAddress;
        info.connLimit = HOST_SOCKET_CONN_LIMIT;
        info.tag = RANK_INFO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(hostPort_);
        wlistInfo_.push_back(info);
    }

    TRY_CATCH_RETURN(HrtRaSocketWhiteListAdd(socketHandle, wlistInfo_));

    HCCL_DEBUG("[RankInfoDetect::%s] end, add wlistInfo size[%zu] success.", __func__, wlistInfo_.size());
    return HCCL_SUCCESS;
}

std::shared_ptr<Socket> RankInfoDetect::ClientInit(const HcclRootHandleV2& rootHandle)
{
    HCCL_DEBUG("[RankInfoDetect::%s] client init start devPhyId_[%u].", __func__, devPhyId_);

    // 获取socket句柄
    SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Create(devPhyId_, hostIp_);

    // 获取server端ip和port
    IpAddress serverIp = IpAddress(std::string(rootHandle.ip));
    u32 serverPort = rootHandle.listenPort;

    // 创建clientSocket
    std::string tag = RANK_INFO_DETECT_TAG + "_" + rootHandle.identifier + "_" + std::to_string(serverPort);
    std::shared_ptr<Socket> clientSocket = std::make_shared<Socket>(
        hostSocketHandle, hostIp_, serverPort, serverIp, tag, SocketRole::CLIENT, NicType::HOST_NIC_TYPE);

    HCCL_INFO("[RankInfoDetect::%s] clientSocket[%s] init end.", __func__, clientSocket->Describe().c_str());
    return clientSocket;
}

void RankInfoDetect::SetupAgent(u32 rankSize, u32 rankId, const HcclRootHandleV2& rootHandle)
{
    HCCL_DEBUG("[RankInfoDetect::%s] setup agent start.", __func__);

    // 网卡使能
    HccpPeerManager::GetInstance().Init(devLogicId_);
    HccpHdcManager::GetInstance().Init(devLogicId_);

    // 获取LocalHostIP
    hostIp_ = GetBootstrapIp(devPhyId_);
    CHK_PRT_THROW(
        hostIp_.IsInvalid(), HCCL_ERROR("[RankInfoDetect::%s] get hostIp fail.", __func__), InternalException,
        "get hostIp fail");

    // 创建clientSocket
    std::shared_ptr<Socket> clientSocket = ClientInit(rootHandle);

    // 1. 创建RankInfoDetectClient对象
    rankInfoDetectClient = std::make_shared<RankInfoDetectClient>(devPhyId_, rankSize, rankId, clientSocket);

    // 2. 调用RankInfoDetectClient.Setup, 获取rankTable
    rankInfoDetectClient->Setup(rankTable_);

    HCCL_DEBUG("[RankInfoDetect::%s] setup agent end.", __func__);
}

void RankInfoDetect::SetupRankInfoDetectService(
    ErrContextPub errorContext, shared_ptr<Socket> serverSocket, s32 devLogicId, u32 devPhyId, std::string identifier,
    vector<RaSocketWhitelist> wlistInfo)
{
    hrtErrMSetErrorContextPub(errorContext);
    HCCL_INFO(
        "[RankInfoDetect::%s] start, devLogicId[%d], devPhyId[%u], identifier[%s].", __func__, devLogicId, devPhyId,
        identifier.c_str());

    // 拓扑探测server开始状态
    u32 hostPort = serverSocket->GetListenPort();
    HCCL_INFO("[RankInfoDetect::%s] listen port[%u].", __func__, hostPort);

    g_detectServerStatus_.EmplaceAndUpdate(hostPort, [](u32& status) {
        status = RANKINFO_DETECT_SERVER_STATUS_RUNING;
    });

    HrtSetDevice(devLogicId);
    std::shared_ptr<RankInfoDetectService> rankInfoDetectService
        = make_shared<RankInfoDetectService>(devPhyId, serverSocket, identifier, wlistInfo);

    bool hasException = false;
    EXCEPTION_CATCH(rankInfoDetectService->Setup(), hasException = true);

    // 若有异常则设置error状态退出
    if (hasException == true) {
        g_detectServerStatus_.EmplaceAndUpdate(hostPort, [](u32& status) {
            status = RANKINFO_DETECT_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[RankInfoDetect::%s] end, status error.", __func__);
        return;
    }

    // 正常结束则设置为idle状态
    g_detectServerStatus_.EmplaceAndUpdate(hostPort, [](u32& status) {
        status = RANKINFO_DETECT_SERVER_STATUS_IDLE;
    });

    HCCL_DEBUG("[RankInfoDetect::%s] end, status idle.", __func__);

    // 确保root info流程先销毁server socket 再返回
    // 可能失败，需要将错误状态带出
    EXCEPTION_CATCH(serverSocket->Destroy(), hasException = true);
    HrtResetDevice(devLogicId);

    // 若有异常则设置error状态退出
    if (hasException == true) {
        g_detectServerStatus_.EmplaceAndUpdate(hostPort, [](u32& status) {
            status = RANKINFO_DETECT_SERVER_STATUS_ERROR;
        });
        HCCL_ERROR("[RankInfoDetect::%s] Destroy end, status error.", __func__);
        return;
    }

    HCCL_DEBUG("[RankInfoDetect::%s] end.", __func__);
}

u32 RankInfoDetect::GetHostListenPort()
{
    // 端口监听范围配置
    u32 listenPort = HCCL_INVALID_PORT;
    auto portRange = EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange();
    if (portRange.size() > 0) {
        HCCL_INFO("[RankInfoDetect::%s] SocketPortRange is configured.", __func__);
        return listenPort;
    }

    // Host网卡起始端口号
    u32 basePort = EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort();
    if (basePort != HCCL_INVALID_PORT) {
        listenPort = basePort + devPhyId_;
        HCCL_INFO("[RankInfoDetect::%s] BasePort is configured, listenPort[%u].", __func__, listenPort);
        return listenPort;
    }

    // 无环境变量设置，返回HCCL_INVALID_PORT触发PreemptPortManager轮询查找端口[60000, 60015]
    listenPort = HCCL_INVALID_PORT;
    HCCL_INFO(
        "[RankInfoDetect::%s] No port configuration, using default port range[%u, %u]", __func__,
        HOST_CONTROL_BASE_PORT, HOST_CONTROL_BASE_PORT + HOST_CONTROL_PORT_COUNT);
    return listenPort;
}

void RankInfoDetect::GetRootHandle(HcclRootHandleV2& rootHandle)
{
    u64 timestamp = 0;
    HcclResult ret = SalGetCurrentTimestamp(timestamp);
    CHK_PRT_THROW(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[RankInfoDetect::%s] get timestamp failed, ret[%d].", __func__, static_cast<int>(ret)),
        InternalException, "get timestamp failed");
    identifier_ = hostIp_.GetIpStr();
    identifier_.append("_");
    identifier_.append(to_string(hostPort_));
    identifier_.append("_");
    identifier_.append(to_string(devPhyId_));
    identifier_.append("_");
    identifier_.append(to_string(timestamp));
    CHK_PRT_THROW(
        (identifier_.length() >= ROOTINFO_IDENTIFIER_MAX_LENGTH),
        HCCL_ERROR("[RankInfoDetect::%s] rootInfo identifier len[%u] is invalid.", __func__, identifier_.length()),
        InternalException, "identifier error");

    s32 sRet = memcpy_s(
        &rootHandle.identifier[0], sizeof(rootHandle.identifier), identifier_.c_str(), (identifier_.length() + 1));
    CHK_PRT_THROW(
        sRet != EOK,
        HCCL_ERROR(
            "[RankInfoDetect::%s] memcpy failed. ret[%d], params: destMaxSize[%zu], count[%zu]", __func__, sRet,
            sizeof(rootHandle.identifier), (identifier_.length() + 1)),
        InternalException, "memcpy failed");

    sRet = strncpy_s(
        rootHandle.ip, sizeof(rootHandle.ip), hostIp_.GetIpStr().c_str(), strlen(hostIp_.GetIpStr().c_str()));
    CHK_PRT_THROW(
        sRet != EOK, HCCL_ERROR("[RankInfoDetect::%s] strncpy failed [%d]", __func__, sRet), InternalException,
        "strncpy failed");

    rootHandle.listenPort = hostPort_;
    rootHandle.netMode = HrtNetworkMode::HDC;

    HCCL_INFO(
        "[RankInfoDetect::%s] rootInfo: ip[%s] port[%u] identifier[%s]", __func__, rootHandle.ip, rootHandle.listenPort,
        identifier_.c_str());
}

void RankInfoDetect::GetRankTable(RankTableInfo& ranktable) const { ranktable = rankTable_; }

void RankInfoDetect::WaitComplete(u32 listenPort, u32 listenStatus) const
{
    // 若server拓扑探测已正常结束则退出
    auto iter = g_detectServerStatus_.Find(listenPort);
    HCCL_INFO("[RankInfoDetect::%s] detect server listenPort[%u] status[%u].", __func__, listenPort, iter.second);
    CHK_PRT_RET(
        !iter.second,
        HCCL_INFO("[RankInfoDetect::%s] detect server listenPort[%u] status idle.", __func__, listenPort), );

    const auto start = chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());

    u32 status = RANKINFO_DETECT_SERVER_STATUS_RUNING;
    while (true) {
        auto it = g_detectServerStatus_.Find(listenPort);
        if (it.second) {
            status = it.first->second;
        }
        if (status == RANKINFO_DETECT_SERVER_STATUS_ERROR) {
            THROW<InternalException>(
                StringFormat("[RankInfoDetect::%s] topo detect failed, port[%u].", __func__, listenPort));
        } else if (status == listenStatus) {
            HCCL_INFO("[RankInfoDetect::%s] topoExchangeServer port[%u] completed.", __func__, listenPort);
            return;
        } else {
            const auto elapsed = chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start);
            if (elapsed > timeout) {
                RPT_INPUT_ERR(
                    true, "EI0015", std::vector<std::string>({"error_reason"}),
                    std::vector<std::string>({StringFormat(
                        "Receiving message from the root node timed out "
                        "after %lld seconds. Timeout was set to %lld seconds. Check whether node %s reports an error.",
                        static_cast<long long>(elapsed.count()), static_cast<long long>(timeout.count()),
                        identifier_.c_str())}));
                HCCL_ERROR(
                    "[%s][%s] errNo[0x%016llx] topo exchange server get socket timeout, listenPort[%u], "
                    "elapsed[%lld s] > timeout[%lld s], identifier[%s]",
                    LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                    HCOM_ERROR_CODE(HcclResult::HCCL_E_TIMEOUT), listenPort, static_cast<long long>(elapsed.count()),
                    static_cast<long long>(timeout.count()), identifier_.c_str());
                THROW<TimeoutException>(StringFormat(
                    "[RankInfoDetect::%s] wait port[%u] complete timeout[%lld s]", __func__, listenPort, elapsed));
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            continue;
        }
    };
}

} // namespace Hccl
