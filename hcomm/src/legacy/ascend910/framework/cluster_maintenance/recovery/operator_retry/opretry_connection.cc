/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opretry_connection.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include "sal_pub.h"
#include "hccl_network_pub.h"
#include "externalinput_pub.h"
#include "adapter_rts_common.h"
#include "opretry_connection_pub.h"

namespace hccl {
UniversalConcurrentMap<std::string, OpRetryConnection::OpRetryConnectionPtr>* OpRetryConnection::instance_ = nullptr;
std::mutex OpRetryConnection::lock_;
bool OpRetryConnection::enable_ = true;

/** OpRetryConnectionPub 封装一层OpRetryConnection接口 */
void OpRetryConnectionPub::SetOpRetryConnEnable(bool enable) { OpRetryConnection::SetOpRetryConnEnable(enable); }

bool OpRetryConnectionPub::IsOpRetryConnEnable() { return OpRetryConnection::IsOpRetryConnEnable(); }

HcclResult OpRetryConnectionPub::Init(
    const std::string& group, u32 rankSize, const OpRetryServerInfo& serverInfo, const OpRetryAgentInfo& agentInfo,
    u32 rootRank)
{
    return OpRetryConnection::Init(group, rankSize, serverInfo, agentInfo, rootRank);
}

void OpRetryConnectionPub::DeInit(const std::string& group) { OpRetryConnection::DelInstance(group); }

HcclResult OpRetryConnectionPub::GetConns(
    const std::string& group, bool& isRoot, std::shared_ptr<HcclSocket>& agent,
    std::map<u32, std::shared_ptr<HcclSocket>>& server)
{
    if (!IsOpRetryConnEnable()) {
        HCCL_INFO("[OpRetryConnection][Init] op retry is disable, so don't need get conns");
        return HCCL_SUCCESS;
    }

    OpRetryConnection::OpRetryConnectionPtr conn;
    CHK_RET(OpRetryConnection::GetInstance(group, conn));

    isRoot = conn->IsRoot();
    CHK_RET(conn->GetAgentSocket(agent));
    if (isRoot) {
        CHK_RET(conn->GetServerSockets(server));
    }

    return HCCL_SUCCESS;
}
/** OpRetryConnectionPub 封装层结束 */

OpRetryConnection::OpRetryConnection() {}

OpRetryConnection::~OpRetryConnection() { DeInit(); }

void OpRetryConnection::SetOpRetryConnEnable(bool enable) { enable_ = enable; }

bool OpRetryConnection::IsOpRetryConnEnable() { return enable_; }

HcclResult OpRetryConnection::Init(
    const std::string& group, u32 rankSize, const OpRetryServerInfo& serverInfo, const OpRetryAgentInfo& agentInfo,
    u32 rootRank)
{
    if (!IsOpRetryConnEnable()) {
        HCCL_INFO("[OpRetryConnection][Init] op retry is disable");
        return HCCL_SUCCESS;
    }
    u32 rankId = agentInfo.userRank;
    HcclIpAddress serverIp = serverInfo.hostIP;
    u32 serverPort
        = serverInfo.hostPort == HCCL_INVALID_PORT ? GetServerPort() + serverInfo.devId : serverInfo.hostPort;
    HcclIpAddress localIp = agentInfo.hostIP;
    if (serverIp.IsInvalid() || localIp.IsInvalid()) {
        HCCL_ERROR(
            "[OpRetryConnection][Init] serverIp [%s] or localIp [%s] is invalid, "
            "check whether the value of host_ip in ranktable is correct.",
            serverIp.GetReadableIP(), localIp.GetReadableIP());
        return HCCL_E_PARA;
    }
    if (rankId >= rankSize || rankSize == 0 || rootRank >= rankSize) {
        HCCL_ERROR(
            "[OpRetryConnection][Init] opRetryConnection input params invalid,"
            "rankId [%u] rankSize [%u] serverIp [%s] localIp [%s] rootRank [%u]",
            rankId, rankSize, serverIp.GetReadableIP(), localIp.GetReadableIP(), rootRank);
        return HCCL_E_PARA;
    }

    HCCL_INFO(
        "[OpRetryConnection][Init] group[%s] rankId [%u] rankSize [%u] serverIp [%s] localIp [%s] rootRank [%u]",
        group.c_str(), rankId, rankSize, serverIp.GetReadableIP(), localIp.GetReadableIP(), rootRank);

    OpRetryConnectionPtr conn;
    CHK_RET(GetInstance(group, conn, true));
    conn->SetGroup(group);
    if (conn->Init(rankId, rankSize, serverIp, serverPort, serverInfo.devId, localIp, rootRank) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[OpRetryConnection][Init] group[%s] rankId [%u] rankSize [%u] serverIp [%s] localIp [%s] rootRank [%u] "
            "failed",
            group.c_str(), rankId, rankSize, serverIp.GetReadableIP(), localIp.GetReadableIP(), rootRank);
        HCCL_ERROR("There maybe some reasons to cause this error:");
        HCCL_ERROR(
            "1. The port may have been used so we will bind error. OpRetry used port range [%u-%u]", serverPort,
            serverPort + OP_RETRY_CONN_PORT_MAX_RANGE);
        HCCL_ERROR("2. Somebody may have already listen on those ports, so we connect to wrong server "
                   "and we will meet 'Recv unmatched ack' error");
        HCCL_ERROR("You can set system reserved port to avoid this error by:");
        HCCL_ERROR(
            "sysctl -w net.ipv4.ip_local_reserved_ports=%u-%u", serverPort, serverPort + OP_RETRY_CONN_PORT_MAX_RANGE);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::GetInstance(const std::string& group, OpRetryConnectionPtr& conn, bool forceNew)
{
    // instance_本身是个指针，需要lock_锁来保护
    std::lock_guard<std::mutex> lockGaurd(lock_);
    if (instance_ == nullptr) {
        instance_ = new (std::nothrow) UniversalConcurrentMap<std::string, OpRetryConnectionPtr>();
        CHK_PTR_NULL(instance_);
    }

    std::lock_guard<std::shared_timed_mutex> guard(instance_->GetMtx());
    if (forceNew || instance_->FindLockFree(group) == instance_->EndLockFree()) {
        OpRetryConnectionPtr tmpConn;
        EXCEPTION_CATCH(tmpConn = std::make_shared<OpRetryConnection>(), return HCCL_E_PTR);
        instance_->EmplaceLockFree(group, tmpConn);
    }

    auto it = instance_->FindLockFree(group);
    CHK_PRT_RET(
        it == instance_->EndLockFree(),
        HCCL_ERROR("[OpRetryConnection][GetInstance] create connection failed in group [%s]", group.c_str()),
        HCCL_E_MEMORY);
    conn = it->second;
    CHK_SMART_PTR_NULL(conn);

    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::DelInstance(const std::string& group)
{
    // instance_本身是个指针，需要lock_锁来保护
    std::lock_guard<std::mutex> lockGuard(lock_);
    if (instance_) {
        instance_->Erase(group);
        if (instance_->Size() == 0) {
            delete instance_;
            instance_ = nullptr;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::Init(
    u32 rankId, u32 rankSize, const HcclIpAddress& serverIp, u32 serverPort, s32 serverDevId,
    const HcclIpAddress& localIp, u32 rootRank)
{
    if (rankId >= rankSize || rankSize == 0 || serverIp.IsInvalid() || rootRank >= rankSize || localIp.IsInvalid()) {
        HCCL_ERROR(
            "[OpRetryConnection][Init] Invalid params, rankId [%u] rankSize [%u] serverIp [%s] localIp [%s] rootRank "
            "[%u]",
            rankId, rankSize, serverIp.GetReadableIP(), localIp.GetReadableIP(), rootRank);
        return HCCL_E_PARA;
    }

    rankId_ = rankId;
    rankSize_ = rankSize;
    serverIp_ = serverIp;
    serverPort_ = serverPort;
    localIp_ = localIp;
    rootRank_ = rootRank;

    HCCL_INFO(
        "[OpRetryConnection][Init] rankId[%u] rankSize[%d] rootRank[%u] serverIp[%s:%u] serverDevId[%d]", rankId_,
        rankSize_, rootRank_, serverIp.GetReadableIP(), serverPort_, serverDevId);

    CHK_RET(InitHcclNet());

    if (IsRoot()) {
        CHK_PRT_RET(
            StartListen() != HCCL_SUCCESS,
            HCCL_ERROR(
                "[OpRetryConnection][Init] Start listen failed, serverIp_[%s] serverPort_[%u]",
                serverIp_.GetReadableIP(), serverPort_),
            HCCL_E_TCP_CONNECT);
    }

    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    u32 retryConnectTimes = 0;
    do {
        CHK_PRT_RET(
            (std::chrono::steady_clock::now() - startTime) > timeout,
            HCCL_ERROR(
                "[OpRetryConnection][Init] Connect to server timeout [%ld s], serverIp_[%s] serverPort_[%u]", timeout,
                serverIp_.GetReadableIP(), serverPort_),
            HCCL_E_TCP_CONNECT);

        auto ret = Connect();
        if (ret == HCCL_SUCCESS) {
            break;
        } else if (ret == HCCL_E_AGAIN) {
            HCCL_ERROR(
                "[OpRetryConnection][Init] Connect to server failed, serverIp_[%s] serverPort_[%u], we try again",
                serverIp_.GetReadableIP(), serverPort_);
            retryConnectTimes++;
            SaluSleep(TEN_MILLISECOND_OF_USLEEP);
            continue;
        } else {
            HCCL_ERROR(
                "[OpRetryConnection][Init] Connect to server failed, serverIp_[%s] serverPort_[%u]",
                serverIp_.GetReadableIP(), serverPort_);
            return ret;
        }
    } while (true);

    if (retryConnectTimes > 0) {
        HCCL_ERROR(
            "[OpRetryConnection][Init] Client reconnect %u times to success, "
            "so the above or this error log can be ignored",
            retryConnectTimes);
    }

    if (IsRoot()) {
        CHK_RET(WaitAcceptFinish());
    }

    HCCL_INFO(
        "[OpRetryConnection][Init] success rankId[%u] rankSize[%d] rootRank[%u] serverIp[%s:%u]", rankId_, rankSize_,
        rootRank_, serverIp.GetReadableIP(), serverPort_);

    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::DeInit()
{
    HCCL_INFO("[OpRetryConnection][Deinit] tag[%s] ready to deinit", tag_.c_str());
    /* 等待后台线程结束 */
    backgroudThreadStop_ = true;
    if (backgroudThread_.joinable()) {
        backgroudThread_.join();
    }

    /* 关闭所有连接 */
    for (auto& socket : connectionSockets_) {
        socket.second->Close();
    }
    connectionSockets_.clear();
    StopListen();

    if (socket_) {
        HCCL_INFO("[OpRetryConnection] close socket");
        socket_->Close();
        socket_ = nullptr;
    }

    /* 释放NetCtx资源 */
    if (serverNetCtx_) {
        HcclNetCloseDev(serverNetCtx_);
        serverNetCtx_ = nullptr;
    }

    if (clientNetCtx_) {
        HcclNetCloseDev(clientNetCtx_);
        clientNetCtx_ = nullptr;
    }

    if (hcclNetInit_) {
        HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicalID_);
        hcclNetInit_ = false;
    }

    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::GetAgentSocket(std::shared_ptr<HcclSocket>& sock)
{
    if (socket_) {
        sock = socket_;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[OpRetryConnection][GetAgentSocket] socket is nullptr");
    return HCCL_E_UNAVAIL;
}

HcclResult OpRetryConnection::GetServerSockets(std::map<u32, std::shared_ptr<HcclSocket>>& socks)
{
    if (!IsRoot()) {
        HCCL_ERROR(
            "[OpRetryConnection][GetServerSockets] rank[%u] is not root rank [%u], so no server sockets", rankId_,
            rootRank_);
        return HCCL_E_UNAVAIL;
    }

    if (!connectionSockets_.empty() && connectionSockets_.size() == rankSize_) {
        socks = connectionSockets_;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR(
        "[OpRetryConnection][GetServerSockets] connection sockets count [%u] rankSize [%u]", connectionSockets_.size(),
        rankSize_);
    return HCCL_E_UNAVAIL;
}

HcclResult OpRetryConnection::InitHcclNet()
{
    CHK_RET(hrtGetDevice(&deviceLogicalID_));
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicalID_, devicePhysicID_));
    HCCL_INFO(
        "[OpRetryConnection][InitHcclNet] deviceLogicalID_[%d] devicePhysicID_[%u]", deviceLogicalID_, devicePhysicID_);

    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicalID_, true));
    hcclNetInit_ = true;

    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::LoadHostWhiteList(const std::string& whiteListFile)
{
    if (!whitelist_.empty()) {
        return HCCL_SUCCESS;
    }

    nlohmann::json fileContent;
    std::ifstream infile(whiteListFile.c_str(), std::ifstream::in);
    CHK_PRT_RET(
        !infile, HCCL_ERROR("[OpRetryConnection][LoadHostWhiteList]open file %s failed", whiteListFile.c_str()),
        HCCL_E_PARA);

    try {
        infile >> fileContent; // 将文件内容读取到json对象内
    } catch (...) {
        HCCL_ERROR(
            "[OpRetryConnection][LoadHostWhiteList]load file[%s] to json fail. please check json file format.",
            whiteListFile.c_str());
        infile.close();
        return HCCL_E_INTERNAL;
    }
    infile.close();

    CHK_PRT_RET(
        fileContent.find("host_ip") == fileContent.end(),
        HCCL_ERROR("[OpRetryConnection][LoadHostWhiteList] whitelist don't have host_ip"), HCCL_E_INTERNAL);
    nlohmann::json hostWhitelist = fileContent["host_ip"];
    for (auto& ipJson : hostWhitelist) {
        std::string ipStr;
        try {
            ipStr = ipJson.get<std::string>();
        } catch (...) {
            HCCL_ERROR(
                "[OpRetryConnection][LoadHostWhiteList]get ipStr from ipJson failed, please check host white list");
            return HCCL_E_PARA;
        }
        HcclIpAddress ip(ipStr);
        CHK_PRT_RET(
            ip.IsInvalid(), HCCL_ERROR("[OpRetryConnection][LoadHostWhiteList]string[%s] is invalid ip", ipStr.c_str()),
            HCCL_E_PARA);
        whitelist_.push_back(ip);
    }

    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::StartListen()
{
    CHK_RET(HcclNetOpenDev(&serverNetCtx_, NicType::HOST_NIC_TYPE, devicePhysicID_, deviceLogicalID_, localIp_));
    CHK_PTR_NULL(serverNetCtx_);

    auto enableWhiteList_ = GetExternalInputHcclEnableWhitelist();
    if (enableWhiteList_ != 0) {
        CHK_RET(GetHostSocketWhiteList());
    }

    EXCEPTION_CATCH((listenSocket_ = std::make_shared<HcclSocket>(serverNetCtx_, serverPort_)), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(listenSocket_);
    CHK_RET(listenSocket_->Init());
    CHK_RET(listenSocket_->Listen());

    if (enableWhiteList_ != 0) {
        CHK_RET(AddListenSocketWhiteList());
    }

    HCCL_INFO(
        "[OpRetryConnection] Server start with host ip[%s] and port[%u]", serverIp_.GetReadableAddress(), serverPort_);

    /* 拉起后台线程，在线程中进行异步接收，这样不会阻塞当前主线程，可以进行后续
     * 用户需要主动调用WaitAcceptFinished()去等待建链结束
     */
    acceptFinished_ = false;
    backgroudThreadStop_ = false;
    std::thread thread(&OpRetryConnection::RunAccept, this);
    backgroudThread_ = std::move(thread);

    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::StopListen()
{
    if (listenSocket_) {
        HCCL_INFO("[OpRetryConnection] Server stop listen socket");
        if (enableWhitelist_ && !wlistInfosVec_.empty()) {
            listenSocket_->DelWhiteList(wlistInfosVec_);
            enableWhitelist_ = false;
            wlistInfosVec_.clear();
        }

        listenSocket_->DeInit();
        listenSocket_ = nullptr;
    }

    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::Accept()
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    auto tag = GetTag();
    std::shared_ptr<HcclSocket> socket = nullptr;

    /* 这里我们按照通信域中rank数目依次accept所有连接 */
    u32 expectAcceptNum = rankSize_;
    while (expectAcceptNum > 0) {
        if (backgroudThreadStop_) {
            HCCL_ERROR(
                "[OpRetryConnection][Accept] OpRetryConnection in accepting but stop, rankSize_[%u] accept link[%u]",
                rankSize_, rankSize_ - expectAcceptNum);
            return HCCL_E_INTERNAL;
        }

        if ((std::chrono::steady_clock::now() - startTime) > timeout) {
            HCCL_ERROR(
                "[OpRetryConnection][Accept] OpRetryConnection accept timeout! timeout[%d s]",
                GetExternalInputHcclLinkTimeOut());
            return HCCL_E_TIMEOUT;
        }

        if (listenSocket_->Accept(tag, socket) == HCCL_SUCCESS) {
            HCCL_INFO("[OpRetryConnection][Accept] accept peer ip[%s]", socket->GetRemoteIp().GetReadableIP());
            // 因为socket是全双工的，所以Server与Client侧均可以先发送，再接收
            CHK_RET(SendAckInfo(socket));
            CHK_RET(RecvMetaInfo(socket));
            expectAcceptNum--;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::WaitAcceptFinish()
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    while (true) {
        if (backgroudThreadStop_) {
            HCCL_ERROR("[OpRetryConnection][WaitAcceptFinish] background thread stopped! may some error happened");
            return HCCL_E_INTERNAL;
        }

        if (acceptFinished_) {
            HCCL_INFO("[OpRetryConnection][WaitAcceptFinish] wait background thread accept finished");
            return HCCL_SUCCESS;
        }

        if ((std::chrono::steady_clock::now() - startTime) > timeout) {
            HCCL_ERROR(
                "[OpRetryConnection][WaitAcceptFinish] wait accept timeout! timeout[%d s]",
                GetExternalInputHcclLinkTimeOut());
            return HCCL_E_TIMEOUT;
        }
    }

    HCCL_ERROR("[OpRetryConnection][WaitAcceptFinish] code should not execute in there");
    return HCCL_E_INTERNAL;
}

HcclResult OpRetryConnection::RecvMetaInfo(std::shared_ptr<HcclSocket>& peerSocket)
{
    u32 peerRankId = INVALID_UINT;
    auto ret = peerSocket->Recv(&peerRankId, sizeof(peerRankId));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[OpRetryConnection][RecvMetaInfo] Recv peer meta info failed. ret [%d]", ret),
        ret);

    CHK_PRT_RET(
        peerRankId >= rankSize_,
        HCCL_ERROR(
            "[OpRetryConnection][RecvMetaInfo] Recv peer meta info invalid peerRankId [%u]. rankSize_[%u]", peerRankId,
            rankSize_),
        HCCL_E_INTERNAL);

    if (connectionSockets_.find(peerRankId) != connectionSockets_.end()) {
        HCCL_ERROR("[OpRetryConnection][RecvMetaInfo] Recv same rankId [%u]", peerRankId);
        return HCCL_E_INTERNAL;
    }

    connectionSockets_.insert({peerRankId, peerSocket});
    HCCL_INFO("[OpRetryConnection][RecvMetaInfo] Recv peer rankId [%u]. rankSize_[%u] success", peerRankId, rankSize_);
    return HCCL_SUCCESS;
}

/* 这里我们使用Server与Client侧约定好的rankSize信息作为Server侧的ACK报文 */
HcclResult OpRetryConnection::SendAckInfo(std::shared_ptr<HcclSocket>& peerSocket)
{
    auto ret = peerSocket->Send(&rankSize_, sizeof(rankSize_));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[OpRetryConnection][SendAckInfo] Send peer meta ack failed, ret %d", ret),
        ret);

    HCCL_INFO("[OpRetryConnection][SendAckInfo] rank [%u] send ack [%u] success", rankId_, rankSize_);
    return HCCL_SUCCESS;
}

void OpRetryConnection::RunAccept()
{
    SetThreadName("Hccl_RetryConn");

    /* 这里我们跟所有Common进程建链，包括自己 */
    if (Accept() != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[OpRetryConnection][Run] Accept failed, serverIp_[%s] serverPort_[%u]", serverIp_.GetReadableIP(),
            serverPort_);
        backgroudThreadStop_ = true;
        return;
    }

    /* 这里停止listen，是因为后续不再接收新的建链，同时也能够释放该端口，让后续通信域使用 */
    if (StopListen() != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[OpRetryConnection][Run] Stop listen failed, serverIp_[%s] serverPort_[%u]", serverIp_.GetReadableIP(),
            serverPort_);
        backgroudThreadStop_ = true;
        return;
    }

    acceptFinished_ = true;
    HCCL_INFO("[OpRetryConnection][Run] Accept all client success");
}

HcclResult OpRetryConnection::Connect()
{
    // 因为Connect可能会被多次调用，因此clientNetCtx不需要多次创建
    if (clientNetCtx_ == nullptr) {
        CHK_RET(HcclNetOpenDev(&clientNetCtx_, NicType::HOST_NIC_TYPE, devicePhysicID_, deviceLogicalID_, localIp_));
        CHK_PTR_NULL(clientNetCtx_);
    }

    // 因为Connect可能会被多次调用，因此如果socket已经存在则直接关闭，并创建新的
    if (socket_ != nullptr) {
        socket_->Close();
        socket_ = nullptr;
    }

    auto tag = GetTag();
    EXCEPTION_CATCH(
        (socket_ = std::make_shared<HcclSocket>(
             tag, clientNetCtx_, serverIp_, serverPort_, HcclSocketRole::SOCKET_ROLE_CLIENT)),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(socket_);
    CHK_RET(socket_->Init());
    CHK_RET(socket_->Connect());

    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (true) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("[OpRetryConnection][Connect] Get socket timeout! timeout [%ld s]", timeout);
            return HCCL_E_TIMEOUT;
        }

        auto status = socket_->GetStatus();
        if (status == HcclSocketStatus::SOCKET_CONNECTING) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            continue;
        } else if (status != HcclSocketStatus::SOCKET_OK) {
            HCCL_ERROR("[OpRetryConnection][Connect] Get socket failed, ret [%d]", status);
            return HCCL_E_TCP_CONNECT;
        } else {
            HCCL_INFO(
                "[OpRetryConnection][Connect] Get socket success with server [%s] port [%u]", serverIp_.GetReadableIP(),
                serverPort_);
            break;
        }
    }

    CHK_RET(SendMetaInfo());
    CHK_RET(RecvAckInfo());
    return HCCL_SUCCESS;
}

/* 目前meta信息主要是rankId */
HcclResult OpRetryConnection::SendMetaInfo()
{
    auto ret = socket_->Send(&rankId_, sizeof(rankId_));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_WARNING("[OpRetryConnection][SendMetaInfo] Send peer meta info failed, ret %d", ret),
        HCCL_E_AGAIN);

    HCCL_INFO("[OpRetryConnection][SendMetaInfo] rank [%u] send meta info success", rankId_);
    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::RecvAckInfo()
{
    u32 ackRankSize = 0;
    auto ret = socket_->Recv(&ackRankSize, sizeof(ackRankSize));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_WARNING("[OpRetryConnection][RecvAckTag] Recv peer ack info failed. ret [%d]", ret),
        HCCL_E_AGAIN);

    if (ackRankSize != rankSize_) {
        HCCL_ERROR("[OpRetryConnection][RecvAckTag] Recv unmatched ack [%u] expect [%u]", ackRankSize, rankSize_);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[OpRetryConnection][RecvAckTag] Recv ack [%u] success", ackRankSize);
    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::GetHostSocketWhiteList()
{
    auto whiteListFile = GetExternalInputHcclWhiteListFile();
    CHK_PRT_RET(
        (whiteListFile.length() == 0),
        HCCL_ERROR("[OpRetryConnection][GetHostSocketWhitelist]environment variable HCCL_WHITELIST_FILE is not set or "
                   "not exist"),
        HCCL_E_PARA);

    HcclResult ret = LoadHostWhiteList(whiteListFile);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[OpRetryConnection][GetHostSocketWhitelist]hccl whitelist load config file[%s] failed. ret[%u].",
            whiteListFile.c_str(), ret),
        ret);

    CHK_PRT_RET(
        whitelist_.empty(),
        HCCL_ERROR(
            "[OpRetryConnection][GetHostSocketWhitelist]whitelist file[%s] have no valid host ip.",
            whiteListFile.c_str()),
        HCCL_E_UNAVAIL);

    HCCL_INFO(
        "[OpRetry][GetHostSocketWhiteList]Get host socket whitelist success. there are %zu host ip in the whitelist.",
        whitelist_.size());
    return HCCL_SUCCESS;
}

HcclResult OpRetryConnection::AddListenSocketWhiteList()
{
    wlistInfosVec_.clear();
    for (auto ip : whitelist_) {
        SocketWlistInfo wlistInfo;
        wlistInfo.connLimit = HOST_SOCKET_CONN_LIMIT;
        wlistInfo.remoteIp.addr = ip.GetBinaryAddress().addr;
        wlistInfo.remoteIp.addr6 = ip.GetBinaryAddress().addr6;
        std::string tag = GetTag();
        s32 sRet = memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), tag.c_str(), tag.size() + 1);
        if (sRet != EOK) {
            HCCL_ERROR("[OpRetryConnection][AddListenSocketWhiteList]memory copy failed. errorno[%d]", sRet);
            return HCCL_E_MEMORY;
        }
        wlistInfosVec_.push_back(wlistInfo);
        HCCL_INFO("[OpRetryConnection][AddListenSocketWhiteList] add white ip %s", ip.GetReadableIP());
    }

    CHK_RET(listenSocket_->AddWhiteList(wlistInfosVec_));

    HCCL_INFO(
        "[OpRetryConnection][AddListenSocketWhiteList] add socket white list success. total: %zu", whitelist_.size());
    return HCCL_SUCCESS;
}

u32 OpRetryConnection::GetServerPort()
{
    u32 serverPort = HCCL_INVALID_PORT;
    if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
        serverPort = HOST_CONTROL_BASE_PORT;
    } else {
        serverPort = GetExternalInputHcclIfBasePort();
    }

    return serverPort;
}
} // namespace hccl
