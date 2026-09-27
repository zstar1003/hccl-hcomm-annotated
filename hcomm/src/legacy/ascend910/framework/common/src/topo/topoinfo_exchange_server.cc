/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_exchange_server.h"
#include <thread>
#include <fstream>
#include <iostream>
#include "adapter_rts_common.h"
#include "externalinput_pub.h"
#include "config.h"
#include "hccl_socket.h"
#include "sal_pub.h"
#include "topoinfo_exchange_dispatcher.h"
#include "preempt_port_manager.h"

namespace hccl {
const u32 DISPLAY_RANKNUM_PERLINE = 8;
const u32 SOCKET_ACCEPT_TIMEOUT = 60; // Server调用Accept等待的最大超时时间 60s
const u32 SOCKET_PRINT_COUNT = 3;     // 未建链打印的数量
using namespace std;
TopoInfoExchangeServer::TopoInfoExchangeServer(
    HcclIpAddress& hostIP, u32 hostPort, const std::vector<HcclIpAddress> whitelist, HcclNetDevCtx netDevCtx,
    std::shared_ptr<HcclSocket> listenSocket, const std::string& identifier)
    : hostIP_(hostIP),
      hostPort_(hostPort),
      whitelist_(whitelist),
      netDevCtx_(netDevCtx),
      listenSocket_(listenSocket),
      identifier_(identifier)
{}

TopoInfoExchangeServer::TopoInfoExchangeServer(
    HcclIpAddress& hostIP, u32 hostPort, const std::vector<HcclIpAddress> whitelist, HcclNetDevCtx netDevCtx,
    std::shared_ptr<HcclSocket> listenSocket, std::shared_ptr<HcclSocket> grpLeaderToRoot,
    const std::string& identifier)
    : hostIP_(hostIP),
      hostPort_(hostPort),
      whitelist_(whitelist),
      netDevCtx_(netDevCtx),
      listenSocket_(listenSocket),
      grpLeaderToRoot_(grpLeaderToRoot),
      identifier_(identifier)
{}

TopoInfoExchangeServer::~TopoInfoExchangeServer() {}

HcclResult TopoInfoExchangeServer::FailedConnectionAgentIdString(u32 rankSize, std::string& failedAgentIdList)
{
    HcclResult result = HCCL_E_NOT_FOUND;
    const u32 oriLength = failedAgentIdList.length();
    std::vector<bool> connectedRank(rankSize, false);
    for (auto it : connectSocketsWithRankID_) {
        if (it.first >= rankSize) {
            HCCL_ERROR(
                "[TopoInfoExchangeServer][FailedConnectionAgentIdString] invalid rank id[%u] from agent.", it.first);
            return HCCL_E_INTERNAL;
        }
        connectedRank[it.first] = true;
    }

    for (u32 i = 0; i < rankSize; i++) {
        if (!connectedRank[i]) {
            failedAgentIdList += std::to_string(i) + ',';
        }
    }

    return failedAgentIdList.length() > oriLength ? HCCL_SUCCESS : result;
}

HcclResult TopoInfoExchangeServer::Setup()
{
    HcclResult ret;
    HcclResult error = HCCL_SUCCESS;

    do {
        u32 expectRankSize = 0;
        std::string failedAgentIdList;
        HcclResult connectRet = Connect(connectSockets_, expectRankSize);
        if (connectRet != HCCL_SUCCESS) {
            HcclResult result = FailedConnectionAgentIdString(expectRankSize, failedAgentIdList);
            CHK_PRT_CONT(
                result == HCCL_SUCCESS,
                HCCL_ERROR("[TopoInfoExchangeServer]failed to connect rankList:[%s]", failedAgentIdList.c_str()));
        }
        u32 rankSize = connectSockets_.size();
        if (!isByMasterInfo_ && rankSize > TOPO_HIERARCHICAL_ENABLE_THRESHOLD) {
            ret = HierarchicalSendRecv();
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][Setup]HierarchicalSendRecv ranktable failed"),
                error = ret);
            HCCL_INFO("cluster topo exchange server HierarchicalSendRecv ranktable success.");
        } else {
            RankTable_t rankTable;
            ret = GetRanksBasicInfo(connectSockets_, rankTable);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][Setup]GetRanksBasicInfo failed"),
                error = ret);
            HCCL_INFO("cluster topo exchange server get rank basic info from all agent success.");

            g_broadcastStage.store(BroadcastStage::Started, std::memory_order_release);
            TopoInfoExchangeDispatcher dispatcher(this);
            ret = dispatcher.BroadcastRankTable(connectSockets_, rankTable, failedAgentIdList);
            {
                g_broadcastStage.store(BroadcastStage::Completed, std::memory_order_release);
                std::lock_guard<std::mutex> lock(g_broadcast_stage_mutex);
                g_broadcast_stage_cv.notify_all();
            }
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[TopoInfoExchangeServer][Setup]Broadcast Rank Basic Infos failed, connectFailedAgentIdList[%s]",
                    failedAgentIdList.c_str()),
                error = ret);
            HCCL_INFO("cluster topo exchange server send rank basic info to all agent success.");
            CHK_PRT_BREAK(
                connectRet != HCCL_SUCCESS,
                HCCL_ERROR("[TopoInfoExchangeServer][Setup]cluster topo exchange server connect client failed"),
                error = connectRet);
            HCCL_INFO("cluster topo exchange server connect with all agent success.");
        }
        ret = StopSocketListen(whitelist_, hostPort_);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TopoInfoExchangeServer][Setup]topo exchange server stop socket listen port[%u] failed.", hostPort_),
            error = ret);
    } while (0);
    if (error != HCCL_SUCCESS) {
        CHK_RET(Disconnect(connectSockets_));
        CHK_RET(StopNetwork(whitelist_, hostPort_));
    }

    HCCL_INFO("cluster topo exchange server completed, exit[%u].", error);
    return error;
}

HcclResult TopoInfoExchangeServer::SetScalableInfo(const ScalableServerInfo& info)
{
    nRoot_ = info.nRoot;
    rootIndex_ = info.rootIndex;
    groupSize_ = info.groupSize;
    deviceLogicId_ = info.deviceLogicId;
    meshInfos_ = info.meshInfos;
    meshListenSocket_ = info.meshListenSocket;
    HCCL_INFO(
        "[TopoInfoExchangeServer][SetScalableInfo] nRoot[%u], rootIndex[%u], groupSize[%u], deviceLogicId[%d]", nRoot_,
        rootIndex_, groupSize_, deviceLogicId_);
    return HCCL_SUCCESS;
}

// 多root建链：本root作为server，接受本组groupSize个rank上报子ranktable，
// 收齐后通过root间全互联广播合并出全局ranktable，再广播给本组rank。
HcclResult TopoInfoExchangeServer::SetupScalable()
{
    HcclResult error = SetupScalableCore();
    if (error != HCCL_SUCCESS) {
        CHK_RET(Disconnect(connectSockets_));
        CHK_RET(StopNetwork(whitelist_, hostPort_));
    }

    HCCL_INFO("cluster topo exchange server(scalable) completed, exit[%u].", error);
    return error;
}

// 通过dispatcher把ranktable广播给各rank，并更新/唤醒广播阶段状态
HcclResult TopoInfoExchangeServer::BroadcastRankTableAndStatus(
    const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, const RankTable_t& rankTable,
    const std::string& failedAgentIdList)
{
    g_broadcastStage.store(BroadcastStage::Started, std::memory_order_release);
    TopoInfoExchangeDispatcher dispatcher(this);
    HcclResult ret = dispatcher.BroadcastRankTable(connectSockets, rankTable, failedAgentIdList);
    {
        g_broadcastStage.store(BroadcastStage::Completed, std::memory_order_release);
        std::lock_guard<std::mutex> lock(g_broadcast_stage_mutex);
        g_broadcast_stage_cv.notify_all();
    }
    return ret;
}

// 多root建链的流程主体：连接本组rank → 收集组内子表 → root间合并 → 广播全局表 → 停止监听
HcclResult TopoInfoExchangeServer::SetupScalableCore()
{
    HcclResult ret;
    HcclResult error = HCCL_SUCCESS;

    do {
        u32 expectRankSize = 0;
        std::string failedAgentIdList;
        HcclResult connectRet = ScalableConnect(connectSockets_, expectRankSize);
        if (connectRet != HCCL_SUCCESS) {
            HcclResult result = FailedConnectionAgentIdString(expectRankSize, failedAgentIdList);
            CHK_PRT_CONT(
                result == HCCL_SUCCESS,
                HCCL_ERROR("[TopoInfoExchangeServer]failed to connect rankList:[%s]", failedAgentIdList.c_str()));
        }

        // 收集本组所有rank上报的子ranktable
        RankTable_t groupRankTable;
        ret = GetRanksBasicInfo(connectSockets_, groupRankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][SetupScalable]GetRanksBasicInfo failed"),
            error = ret);
        HCCL_INFO("topo exchange server(scalable) get group rank basic info success, groupSize[%u].", groupSize_);

        // root间全互联广播：收齐所有root的组内子表并合并出全局ranktable
        ret = RootMeshAllGatherAndMerge(groupRankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][SetupScalable]RootMeshAllGatherAndMerge failed"),
            error = ret);
        HCCL_INFO("topo exchange server(scalable) root mesh allgather and merge success.");

        // 广播全局ranktable给本组rank
        ret = BroadcastRankTableAndStatus(connectSockets_, rankTable_, failedAgentIdList);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TopoInfoExchangeServer][SetupScalable]Broadcast Rank Basic Infos failed, "
                "connectFailedAgentIdList[%s]",
                failedAgentIdList.c_str()),
            error = ret);
        HCCL_INFO("topo exchange server(scalable) send rank basic info to all group agent success.");
        CHK_PRT_BREAK(
            connectRet != HCCL_SUCCESS,
            HCCL_ERROR("[TopoInfoExchangeServer][SetupScalable]topo exchange server connect client failed"),
            error = connectRet);

        ret = StopSocketListen(whitelist_, hostPort_);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TopoInfoExchangeServer][SetupScalable]topo exchange server stop socket listen port[%u] failed.",
                hostPort_),
            error = ret);
    } while (0);

    return error;
}

HcclResult TopoInfoExchangeServer::ScalableConnect(
    std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, u32& rankSize)
{
    // 多root场景下，仅accept本组的groupSize个rank（含root自身），不依赖agent上报的rankNum
    return AcceptLoop(connectSockets, rankSize, groupSize_, true);
}

// accept循环每轮的状态：剩余时间不足1s则跳过本轮，整体超时返回ACCEPT_TIMEOUT
TopoInfoExchangeServer::SocketAcceptStatus TopoInfoExchangeServer::GetSocketAcceptStatus(
    const std::chrono::steady_clock::time_point& startTime, const std::chrono::seconds& timeout, u32& waitTime)
{
    auto topoExUsedTime = std::chrono::steady_clock::now() - startTime;
    if (topoExUsedTime >= timeout) {
        return SocketAcceptStatus::ACCEPT_TIMEOUT;
    }
    auto topoExResTime = timeout - topoExUsedTime;
    u32 topoExRes_i = std::chrono::duration_cast<std::chrono::seconds>(topoExResTime).count();
    if (topoExRes_i == 0) {
        return SocketAcceptStatus::ACCEPT_SKIP;
    }
    waitTime = topoExRes_i > SOCKET_ACCEPT_TIMEOUT ? SOCKET_ACCEPT_TIMEOUT : topoExRes_i;
    return SocketAcceptStatus::ACCEPT_WAIT;
}

void TopoInfoExchangeServer::LogAcceptTimeout(bool isScalable) const
{
    HCCL_ERROR(
        "[%s][%s]topo exchange server%s get socket timeout! timeout[%d s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
        LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), isScalable ? "(scalable)" : "", GetExternalInputHcclLinkTimeOut());
}

// 每轮计算剩余等待时间后accept一个连接，收满expectSocketNumInit个连接即完成。
HcclResult TopoInfoExchangeServer::AcceptLoop(
    std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, u32& rankSize, u32 expectSocketNumInit,
    bool isScalable)
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    u32 expectSocketNum = expectSocketNumInit;
    u32 previousRankNum = 0;
    bool isFirstAcceptTimeOut = false;

    while (expectSocketNum > 0) {
        u32 waitTime = SOCKET_ACCEPT_TIMEOUT;
        SocketAcceptStatus status = GetSocketAcceptStatus(startTime, timeout, waitTime);
        if (status == SocketAcceptStatus::ACCEPT_TIMEOUT) {
            LogAcceptTimeout(isScalable);
            DisplayConnectedRank(connectSockets, rankSize);
            return HCCL_E_TIMEOUT;
        }
        if (status == SocketAcceptStatus::ACCEPT_SKIP) {
            continue;
        }
        CHK_RET(AcceptSocket(
            connectSockets, rankSize, waitTime, expectSocketNum, previousRankNum, isFirstAcceptTimeOut, isScalable));
    }
    return HCCL_SUCCESS;
}

// 单次accept并处理建链结果：成功则记录rankNum并校验一致性，超时/连接错误则相应处理
HcclResult TopoInfoExchangeServer::AcceptSocket(
    std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, u32& rankSize, u32 waitTime,
    u32& expectSocketNum, u32& previousRankNum, bool& isFirstAcceptTimeOut, bool isScalable)
{
    std::shared_ptr<HcclSocket> socket;
    std::string tag = TOPO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(hostPort_);
    HcclResult ret = listenSocket_->Accept(tag, socket, waitTime);
    if (ret == HCCL_SUCCESS) {
        HCCL_INFO("listenSocket_->Accept completed.");
        // server获取socket之后进行一次数据收发用于判断是否都成功获取到了socket
        CHK_RET(socket->Send(TOPO_EXCHANGE_CHECK_MESSAGE, sizeof(TOPO_EXCHANGE_CHECK_MESSAGE)));
        u32 rankNum = 0;
        CHK_RET(GetRemoteFdAndRankSize(socket, connectSockets, rankNum));
        rankSize = rankNum;
        if (!isScalable) {
            expectSocketNum = (previousRankNum == 0) ? rankNum : expectSocketNum;
        }
        // 仍校验所有agent上报的rankNum一致（scalable场景为全局nRanks），但不作为accept数量
        CHK_RET(VerifyRemoteRankNum(previousRankNum, rankNum));
        expectSocketNum -= 1;
        isFirstAcceptTimeOut = false;
    } else if (ret == HCCL_E_TIMEOUT) {
        HCCL_INFO("listenSocket_->Accept TimeOut[%lld s]", waitTime);
        if (isFirstAcceptTimeOut) {
            return HCCL_SUCCESS;
        }
        isFirstAcceptTimeOut = true;
        DisplayConnectingStatus(previousRankNum, expectSocketNum, connectSockets);
    } else if (ret == HCCL_E_TCP_CONNECT) {
        HCCL_INFO("listenSocket_->Accept E_TCP_CONNECT");
        DisplayConnectedRank(connectSockets, rankSize);
        return HCCL_E_TCP_CONNECT;
    }
    return HCCL_SUCCESS;
}

// root间全互联广播：
//  1. 每个root收齐本组子表后，将子表直接发送给其余所有root（小index主动connect大index，大index
//  accept，避免双向connect死锁）
//  2. 每个root按rootIndex落槽维护 partials_，收齐 nRoot-1 个对端子表后合并出全局ranktable
// 由于每条partial按源rootIndex独立落槽，且每对root只互发一条，不存在跨连接乱序问题。
HcclResult TopoInfoExchangeServer::RootMeshAllGatherAndMerge(const RankTable_t& groupRankTable)
{
    if (nRoot_ <= 1) {
        // 单root退化为本组即全局
        partials_.clear();
        partials_.push_back(groupRankTable);
        return MergeRankTables(rankTable_);
    }

    partials_.clear();
    partials_.resize(nRoot_);
    partials_[rootIndex_] = groupRankTable;
    meshRecvCount_.store(0);

    HcclResult acceptRet = HCCL_SUCCESS;
    std::thread acceptThread;
    if (rootIndex_ > 0) {
        acceptThread = std::thread([this, &acceptRet]() {
            acceptRet = MeshAcceptWorker();
        });
    }

    HcclResult connectRet = MeshConnectToLargerRoots();

    if (acceptThread.joinable()) {
        acceptThread.join();
    }

    if (connectRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[TopoInfoExchangeServer][RootMeshAllGatherAndMerge]mesh connect to larger roots failed, ret[%u]",
            connectRet);
        return connectRet;
    }
    if (acceptRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[TopoInfoExchangeServer][RootMeshAllGatherAndMerge]mesh accept smaller roots failed, ret[%u]", acceptRet);
        return acceptRet;
    }
    if (meshRecvCount_.load() != nRoot_ - 1) {
        HCCL_ERROR(
            "[TopoInfoExchangeServer][RootMeshAllGatherAndMerge]recv partial count[%u] mismatch expect[%u]",
            meshRecvCount_.load(), nRoot_ - 1);
        return HCCL_E_INTERNAL;
    }
    return MergeRankTables(rankTable_);
}

// HcclSocket::Connect() 仅发起异步连接，fdHandle_ 在连接建立（GetStatus==SOCKET_OK）后才填充，
// 因此 mesh 客户端在 Send/Recv 前必须轮询等待连接建立，与 agent 侧 GetConnection 逻辑一致
HcclResult TopoInfoExchangeServer::WaitMeshConnectionEstablished(const std::shared_ptr<HcclSocket>& socket) const
{
    const auto startTime = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    while (true) {
        HcclSocketStatus status = socket->GetStatus();
        if (status == HcclSocketStatus::SOCKET_OK) {
            return HCCL_SUCCESS;
        }
        if (status == HcclSocketStatus::SOCKET_CONNECTING) {
            if (std::chrono::steady_clock::now() - startTime >= timeout) {
                HCCL_ERROR(
                    "[TopoInfoExchangeServer][WaitMeshConnectionEstablished]wait mesh connect timeout[%lld s]",
                    GetExternalInputHcclLinkTimeOut());
                return HCCL_E_TIMEOUT;
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            continue;
        }
        HCCL_ERROR(
            "[TopoInfoExchangeServer][WaitMeshConnectionEstablished]mesh socket establish failed, status[%d]",
            static_cast<int>(status));
        return HCCL_E_TCP_CONNECT;
    }
}

// 构建root间mesh连接tag：各root的identifier互不相同不能用于tag，
// 改用有序对<较小root, 较大root> + 较大root的meshPort（双方均可从meshInfos_算出一致的值），
// 保证连接/接受双方tag一致且每对root唯一
std::string TopoInfoExchangeServer::BuildMeshTag(u32 smallerRoot, u32 largerRoot) const
{
    // 复用公共构造，与 mesh 白名单下发/删除的 tag 严格一致
    return TopoInfoExchangeBase::BuildMeshTag(smallerRoot, largerRoot, meshInfos_[largerRoot].meshPort);
}

// 主动connect所有 index > 本root 的root（连接方先发后收，配合对方先收后发，单条消息不互锁）
HcclResult TopoInfoExchangeServer::MeshConnectToLargerRoots()
{
    for (u32 peer = rootIndex_ + 1; peer < nRoot_; ++peer) {
        HcclIpAddress peerIp(meshInfos_[peer].ip);
        CHK_PRT_RET(
            peerIp.IsInvalid() || meshInfos_[peer].meshPort == HCCL_INVALID_PORT,
            HCCL_ERROR(
                "[TopoInfoExchangeServer][MeshConnectToLargerRoots]invalid peer[%u] ip[%s] port[%u]", peer,
                meshInfos_[peer].ip, meshInfos_[peer].meshPort),
            HCCL_E_PARA);
        // 连接方只连接更大的root，tag 与 MeshAcceptWorker 的 accept tag 保持一致，RA socket才能配对
        std::string tag = BuildMeshTag(rootIndex_, peer);
        std::shared_ptr<HcclSocket> socket;
        EXCEPTION_CATCH(
            (socket = std::make_shared<HcclSocket>(
                 tag, netDevCtx_, peerIp, meshInfos_[peer].meshPort, HcclSocketRole::SOCKET_ROLE_CLIENT)),
            return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(socket);
        CHK_RET(socket->Init());
        CHK_RET(socket->Connect()); // 发起异步连接，等待对端root accept并建立连接
        // 连接建立（fdHandle可用）后才能收发
        CHK_RET(WaitMeshConnectionEstablished(socket));
        HCCL_INFO(
            "[TopoInfoExchangeServer][MeshConnectToLargerRoots]connect root[%u] ip[%s] port[%u] success.", peer,
            meshInfos_[peer].ip, meshInfos_[peer].meshPort);

        // 连接方：先发自己的rootIndex和组内子表，再收对端子表
        u32 srcIndex = rootIndex_;
        CHK_RET(socket->Send(&srcIndex, sizeof(srcIndex)));
        CHK_RET(MeshSendPartial(socket, partials_[rootIndex_]));
        CHK_RET(RecvClusterInfoMsg(socket, partials_[peer]));
        meshRecvCount_++;
        CHK_RET(DisconnectSocket(socket));
        HCCL_INFO("[TopoInfoExchangeServer][MeshConnectToLargerRoots]exchange partial with root[%u] success.", peer);
    }
    return HCCL_SUCCESS;
}

// 被动accept所有 index < 本root 的root（接受方先收后发，避免与连接方同时Connect导致死锁）。
// MeshAcceptWorker 运行在 RootMeshAllGatherAndMerge 新建的独立线程上，该线程未设置device上下文；
// 而 RecvClusterInfoMsg 内部 HostMem::alloc（hrtMallocHost）依赖当前线程的device，必须先设置，
// 否则内存分配失败返回HCCL_E_PTR。
HcclResult TopoInfoExchangeServer::MeshAcceptWorker()
{
    CHK_RET(hrtSetDevice(deviceLogicId_));
    HcclResult ret = MeshAcceptWorkerCore();
    (void)hrtResetDevice(deviceLogicId_);
    return ret;
}

HcclResult TopoInfoExchangeServer::MeshAcceptWorkerCore()
{
    for (u32 peer = 0; peer < rootIndex_; ++peer) {
        std::shared_ptr<HcclSocket> socket;
        // 接受方只接受更小的root，对端peer即该连接的较小端；
        // tag 用有序对<peer较小, rootIndex_较大>，与MeshConnectToLargerRoots的connect tag保持一致，RA socket才能配对
        std::string tag = BuildMeshTag(peer, rootIndex_);
        HcclResult ret = meshListenSocket_->Accept(tag, socket, SOCKET_ACCEPT_TIMEOUT);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[TopoInfoExchangeServer][MeshAcceptWorker]mesh accept failed, ret[%u]", ret), ret);

        // 先收对端的rootIndex与组内子表
        u32 srcIndex = INVALID_UINT;
        CHK_RET(socket->Recv(&srcIndex, sizeof(srcIndex)));
        CHK_PRT_RET(
            srcIndex >= nRoot_ || srcIndex >= rootIndex_,
            HCCL_ERROR(
                "[TopoInfoExchangeServer][MeshAcceptWorker]invalid src rootIndex[%u], nRoot[%u]", srcIndex, nRoot_),
            HCCL_E_INTERNAL);
        CHK_RET(RecvClusterInfoMsg(socket, partials_[srcIndex]));
        meshRecvCount_++;

        // 再把自己的组内子表发回给对端
        CHK_RET(MeshSendPartial(socket, partials_[rootIndex_]));
        CHK_RET(DisconnectSocket(socket));
        HCCL_INFO("[TopoInfoExchangeServer][MeshAcceptWorker]exchange partial with root[%u] success.", srcIndex);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::MeshSendPartial(std::shared_ptr<HcclSocket> socket, const RankTable_t& partial)
{
    nlohmann::json basicJson;
    CHK_RET(Struct2Json(partial, basicJson));
    basicJson[PROP_STEP] = currentStep_; // 与其他root保持相同step校验
    std::string buffer = basicJson.dump();
    u32 msgLen = buffer.length();
    CHK_RET(SendClusterInfoMsg(socket, partial, buffer, msgLen));
    return HCCL_SUCCESS;
}

// 按rootIndex顺序合并所有partial为全局ranktable（rankId全局有序，无需再排序）
HcclResult TopoInfoExchangeServer::MergeRankTables(RankTable_t& mergedTable)
{
    RankTable_t result;
    bool isFirst = true;
    for (u32 i = 0; i < nRoot_; ++i) {
        if (partials_[i].rankList.empty()) {
            HCCL_ERROR("[TopoInfoExchangeServer][MergeRankTables]partial of root[%u] is empty, nRoot[%u].", i, nRoot_);
            return HCCL_E_INTERNAL;
        }
        if (isFirst) {
            result.nicDeploy = partials_[i].nicDeploy;
            isFirst = false;
        } else if (result.nicDeploy != partials_[i].nicDeploy) {
            HCCL_ERROR(
                "[TopoInfoExchangeServer][MergeRankTables]nicDeploy mismatch, root[%u] nicDeploy[%u], expect[%u].", i,
                partials_[i].nicDeploy, result.nicDeploy);
            return HCCL_E_INTERNAL;
        }
        for (auto& rank : partials_[i].rankList) {
            result.rankList.push_back(rank);
        }
        for (auto& server : partials_[i].serverList) {
            if (!DoServerIdExist(result, server.serverId)) {
                result.serverList.push_back(server);
            }
        }
    }
    // 重新统计serverNum/rankNum/deviceNum/superPodNum，并校验nicDeploy。
    // 此处以 partials_[0] 为 nicDeploy 基准：循环内已保证其非空且与其余 partial 一致（isFirst 分支校验）
    CHK_RET(GetCommonTopoInfo(result, partials_[0]));
    CHK_RET(SortRankList(result));
    mergedTable = result;
    HCCL_INFO(
        "[TopoInfoExchangeServer][MergeRankTables]merge success, rankNum[%u], serverNum[%u], nRoot[%u].",
        result.rankNum, result.serverNum, nRoot_);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::HierarchicalSendRecv()
{
    TopoInfoExchangeDispatcher dispatcherGrpLeader(this);
    TopoInfoExchangeDispatcher dispatcherGrpLeaderPortInfo(this);
    TopoInfoExchangeDispatcher dispatcherRankTable(this);

    // get Group Leader info
    GroupLeader_t groupLeader;
    HcclResult ret = RecvGroupLeaderInfo(connectSockets_, groupLeader);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][Setup]RecvGroupLeaderInfo failed"), ret);

    HCCL_INFO("cluster topo exchange server get group leader info.");
    // BroadCast GroupLeader info
    ret = dispatcherGrpLeader.BroadcastGroupLeaderInfo(connectSockets_, groupLeader);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[TopoInfoExchangeServer][Setup]Broadcast Group Leader Infos No PortInfo failed"), ret);
    HCCL_INFO("cluster topo exchange server send groupleader info to all agent success.");

    // root接收每个GroupLeader传上来的port
    ret = RecvGroupLeaderPortInfo(grpLeaderSockets_, groupLeader);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][Setup]RecvGroupLeaderPortInfo failed"), ret);

    // BroadCast GroupLeader Port Info
    ret = dispatcherGrpLeaderPortInfo.BroadcastGroupLeaderInfo(connectSockets_, groupLeader);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[TopoInfoExchangeServer][Setup]Broadcast Group Leader Infos with PortInfo failed"), ret);
    HCCL_INFO("cluster topo exchange server send groupleader info to all agent success.");
    // root接收GroupLeader上传的ranktable
    RankTable_t rankTable;

    ret = GetRanksBasicInfo(grpLeaderSockets_, rankTable);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][Setup]RecvGroupClusterInfo failed"), ret);
    HCCL_INFO("cluster topo exchange server get rank basic info from all group leader success.");

    // root向GroupLeader广播全局ranktable
    ret = dispatcherRankTable.BroadcastRankTable(grpLeaderSockets_, rankTable, "");
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][Setup]Broadcast Rank Basic Infos failed"), ret);
    HCCL_INFO("cluster topo exchange server send rank basic info to all group leader success.");

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::RecvGroupLeaderInfo(
    const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, GroupLeader_t& groupLeader)
{
    u32 socketNumPerGrp = 0;
    u32 socketIndex = 0; // socket已经经过rankid（or superPodId + serverip + deviceid排序）
    bool isGroupLeader = true;
    std::map<u32, HcclRootHandle> GroupLeaders;

    for (auto& handle : connectSockets) {
        HcclRankHandle rankHandle;
        HcclResult ret = handle.second->Recv(&rankHandle, sizeof(HcclRankHandle));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Get][RecvGroupLeaderInfo]RecvGroupLeaderInfo from agentId[%s] failed, ret[%d]", handle.first.c_str(),
                ret),
            ret);
        if (isGroupLeader) {
            u32 GroupIndex = socketIndex / TOPO_MAX_GROUP_SIZE;
            GroupLeaders.insert(pair<u32, HcclRootHandle>(GroupIndex, rankHandle));
            grpLeaderSockets_.insert(handle);
            isGroupLeader = false;
        }

        socketNumPerGrp++;
        socketIndex++;
        if (socketNumPerGrp == TOPO_MAX_GROUP_SIZE) {
            isGroupLeader = true;
            socketNumPerGrp = 0;
        }
    }
    // 把GroupLeader信息存放到GroupLeaderList中 方便广播
    for (auto iter : GroupLeaders) {
        groupLeader.grpLeaderNum++;
        groupLeader.GroupLeaderList.emplace_back(iter.second);
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::RecvGroupLeaderPortInfo(
    const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, GroupLeader_t& groupLeader)
{
    HcclResult ret;
    groupLeader.GroupLeaderList.clear();
    for (auto& handle : connectSockets) {
        HcclRankHandle grpLeaderPortInfo;
        ret = handle.second->Recv(&grpLeaderPortInfo, sizeof(HcclRankHandle));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Get][RecvGroupLeaderPortInfo]RecvGroupLeaderPortInfo from grpLeader[%s] failed, ret[%d]",
                handle.first.c_str(), ret),
            ret);
        groupLeader.GroupLeaderList.emplace_back(grpLeaderPortInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::SetupGroupLeader()
{
    HcclResult ret;
    HcclResult error = HCCL_SUCCESS;

    do {
        TopoInfoExchangeDispatcher dispatcher(this);

        ret = GroupLeaderConnect(connectSockets_);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[TopoInfoExchangeServer][Setup]cluster topo exchange server connect client failed"),
            error = ret);
        HCCL_INFO("cluster topo exchange server connect with all agent success.");

        RankTable_t rankTable;
        // GroupLeader接收Group内rank上报的ranktable
        ret = GetRanksBasicInfo(connectSockets_, rankTable);
        currentStep_--;
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][Setup]RecvGroupClusterInfo failed"), error = ret);
        HCCL_INFO("cluster topo exchange server get rank basic info from all agent success.");

        HCCL_INFO("topo exchange client send rank basic info success.");
        CHK_RET(SendClusterInfo(grpLeaderToRoot_, rankTable));

        CHK_RET(RecvClusterInfo(grpLeaderToRoot_, rankTable_));
        currentStep_--;
        HCCL_INFO("topo exchange client get rank basic info success.");

        ret = dispatcher.BroadcastRankTable(connectSockets_, rankTable_, "");
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][Setup]Broadcast Rank Basic Infos failed"),
            error = ret);
        HCCL_INFO("cluster topo exchange server send rank basic info to all agent success.");

        ret = StopSocketListen(whitelist_, hostPort_);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[TopoInfoExchangeServer][Setup]topo exchange server stop socket listen failed."), error = ret);
    } while (0);

    if (error != HCCL_SUCCESS) {
        CHK_RET(Disconnect(connectSockets_));
        CHK_RET(StopNetwork(whitelist_, hostPort_));
    }

    HCCL_INFO("cluster topo exchange server completed, exit[%u].", error);

    return error;
}

HcclResult TopoInfoExchangeServer::Teardown()
{
    CHK_RET(Disconnect(connectSockets_));
    CHK_RET(StopNetwork(whitelist_, hostPort_));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::GetConnections(std::map<u32, std::shared_ptr<HcclSocket>>& connectSockets)
{
    connectSockets = connectSocketsWithRankID_;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::SetupByMasterInfo()
{
    isByMasterInfo_ = true;
    CHK_RET(Setup());
    return HCCL_SUCCESS;
}

HcclResult
TopoInfoExchangeServer::Connect(std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, u32& rankSize)
{
    return AcceptLoop(connectSockets, rankSize, 1, false);
}

HcclResult
TopoInfoExchangeServer::GroupLeaderConnect(std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets)
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    u32 groupMaxRankNum = TOPO_MAX_GROUP_SIZE;
    bool isFirstAcceptTimeOut = false;

    while (expectSocketNum_ > 0 && groupMaxRankNum > 0) {
        auto topoExUsedTime = std::chrono::steady_clock::now() - startTime;
        if (topoExUsedTime >= timeout) {
            HCCL_ERROR(
                "[%s][%s]topo exchange server get socket timeout! timeout[%d s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), GetExternalInputHcclLinkTimeOut());
            DisplayConnectedRank(connectSockets);
            return HCCL_E_TIMEOUT;
        }
        auto topoExResTime = timeout - topoExUsedTime;
        u32 topoExRes_i = std::chrono::duration_cast<std::chrono::seconds>(topoExResTime).count();
        u32 socketWaitTime = SOCKET_ACCEPT_TIMEOUT;
        if (topoExRes_i != 0) {
            socketWaitTime = topoExRes_i > SOCKET_ACCEPT_TIMEOUT ? SOCKET_ACCEPT_TIMEOUT : topoExRes_i;
        } else {
            continue;
        }
        std::shared_ptr<HcclSocket> socket;
        std::string tag = TOPO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(hostPort_);

        HcclResult ret = listenSocket_->Accept(tag, socket, socketWaitTime);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("listenSocket_->Accept completed.");
            u32 rankNum = 0;
            CHK_RET(GetRemoteFdAndRankSize(socket, connectSockets, rankNum));
            expectSocketNum_ = (previousRankNum_ == 0) ? rankNum : expectSocketNum_;
            groupMaxRankNum = (rankNum > TOPO_HIERARCHICAL_ENABLE_THRESHOLD) ? groupMaxRankNum : expectSocketNum_;
            CHK_RET(VerifyRemoteRankNum(previousRankNum_, rankNum));

            expectSocketNum_ -= 1;
            groupMaxRankNum -= 1;
            isFirstAcceptTimeOut = false;
        } else if (ret == HCCL_E_TIMEOUT) {
            HCCL_ERROR("listenSocket_->Accept TimeOut[%lld s]", socketWaitTime);
            if (isFirstAcceptTimeOut) {
                continue;
            }
            isFirstAcceptTimeOut = true;

            DisplayConnectingStatus(previousRankNum_, expectSocketNum_, connectSockets);
        } else if (ret == HCCL_E_TCP_CONNECT) {
            HCCL_INFO("listenSocket_->Accept E_TCP_CONNECT");
            DisplayConnectedRank(connectSockets);
            return HCCL_E_TCP_CONNECT;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::DisplayConnectingStatus(
    u32 totalSockets, u32 waitSockets, const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets) const
{
    if (totalSockets == 0 && waitSockets == 1) {
        return HCCL_SUCCESS;
    }

    // 单算子模式阶段性打印内容
    if (!isByMasterInfo_) {
        std::vector<bool> rankinfos(totalSockets, false);
        for (auto it : connectSockets) { // 建立映射
            u32 rankid = 0;
            CHK_RET(SalStrToULong(it.first, HCCL_BASE_DECIMAL, rankid));
            rankinfos.at(rankid) = true;
        }

        u32 unRankCount = 0; // 只打印前三条未建链的rank
        std::vector<string> unsocketinfos;
        for (u32 rankid = 0; rankid < totalSockets; rankid++) {
            if (unRankCount >= SOCKET_PRINT_COUNT) {
                break;
            }
            if (!rankinfos[rankid]) {
                unRankCount++;
                std::string rankID = std::to_string(rankid);
                std::string agentID = std::string(16 - rankID.length(), '0') + rankID;
                unsocketinfos.push_back(agentID);
            }
        }

        std::string infoStr = "succ sockets are [" + std::to_string((totalSockets - waitSockets))
                              + "], waiting sockets are [" + std::to_string(waitSockets) + "], wait sockets rankid: ";
        for (u32 index = 0; index < unsocketinfos.size(); index++) {
            if (index == (unsocketinfos.size() - 1)) {
                infoStr += "[" + unsocketinfos[index] + "]";
            } else {
                infoStr += "[" + unsocketinfos[index] + "],";
            }
        }

        HCCL_RUN_INFO("[HCCL_TRACE] %s", infoStr.c_str());
    } else {
        std::string infoStr = "succ sockets are [" + std::to_string(totalSockets - waitSockets)
                              + "], waiting sockets are [" + std::to_string(waitSockets) + "]";
        HCCL_RUN_INFO("[HCCL_TRACE] %s , isByMasterInfo[%d]", infoStr.c_str(), isByMasterInfo_);
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::GetRemoteFdAndRankSize(
    std::shared_ptr<HcclSocket>& socket, std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets,
    u32& rankSize)
{
    std::string agentID;
    CHK_RET(RecvRemoteAgentID(socket, agentID));
    auto iter = connectSockets.find(agentID);
    CHK_PRT_RET(
        iter != connectSockets.end(),
        HCCL_ERROR("[Get][Connection]GetConnection failed. agent[%s] has been connected.", agentID.c_str()),
        HCCL_E_INTERNAL);
    connectSockets.insert({agentID, socket});

    CHK_RET(RecvRemoteRankNum(socket, rankSize));

    u32 rankID = 0;
    if (!isByMasterInfo_) {
        CHK_RET(SalStrToULong(agentID, HCCL_BASE_DECIMAL, rankID));
        connectSocketsWithRankID_.insert({rankID, socket});
    }

    bool isRankIdUnAvailable = isByMasterInfo_ ? (false) : (rankID >= rankSize);
    CHK_PRT_RET(
        isRankIdUnAvailable,
        HCCL_ERROR(
            "[Get][Connection]rank"
            " num[%u] from remote[%s] invalid.",
            rankSize, agentID.c_str()),
        HCCL_E_INTERNAL);
    HCCL_INFO("get remote rank[%s / %u] success.", agentID.c_str(), rankSize);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::DisplayConnectedRank(
    const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, u32 rankNum)
{
    vector<string> ranksInfo;
    for (auto it : connectSockets) {
        ranksInfo.push_back(it.first);
    }
    u64 ranksLen = ranksInfo.size();
    u64 lineNum = (ranksInfo.size() % DISPLAY_RANKNUM_PERLINE == 0) ? (ranksInfo.size() / DISPLAY_RANKNUM_PERLINE) :
                                                                      (ranksInfo.size() / DISPLAY_RANKNUM_PERLINE + 1);
    HCCL_ERROR(
        "[%s][%s]total connected num is [%llu],line num is [%llu]", LOG_KEYWORDS_INIT_GROUP.c_str(), __func__, ranksLen,
        lineNum);
    if (rankNum != 0) {
        HCCL_ERROR("[%s][%s]need connect rankNum is [%u]", LOG_KEYWORDS_INIT_GROUP.c_str(), __func__, rankNum);
    }
    for (u64 i = 0; i < lineNum; i++) {
        string tmpRankList;
        for (u32 j = 0; j < DISPLAY_RANKNUM_PERLINE; j++) {
            u32 ranksInfoIndex = i * DISPLAY_RANKNUM_PERLINE + j;
            if (ranksInfoIndex < ranksInfo.size()) {
                tmpRankList += "[" + ranksInfo[ranksInfoIndex] + "]";
            } else {
                break;
            }
            tmpRankList += ((j == DISPLAY_RANKNUM_PERLINE - 1 || ranksInfoIndex == ranksInfo.size() - 1) ? ";" : ",");
        }
        HCCL_ERROR(
            "[%s][%s]connected rankinfo[LINE %llu]: %s", LOG_KEYWORDS_INIT_GROUP.c_str(), __func__, i,
            tmpRankList.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::Disconnect(std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets)
{
    std::unique_lock<std::mutex> lock(lock_);
    for (auto& socket : connectSockets) {
        CHK_RET(DisconnectSocket(socket.second));
    }
    connectSockets.clear();
    connectSocketsWithRankID_.clear();
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::DeleteSocketWhiteList(u32 port, const std::vector<HcclIpAddress>& whitelist)
{
    std::vector<SocketWlistInfo> wlistInfosVec;
    for (auto ip : whitelist) {
        SocketWlistInfo wlistInfo = {};
        wlistInfo.connLimit = HOST_SOCKET_CONN_LIMIT;
        wlistInfo.remoteIp.addr = ip.GetBinaryAddress().addr;
        wlistInfo.remoteIp.addr6 = ip.GetBinaryAddress().addr6;
        std::string tag = TOPO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(port);
        s32 sRet = memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), tag.c_str(), tag.size() + 1);
        if (sRet != EOK) {
            HCCL_ERROR("[Delete][SocketWhiteList]memory copy failed. errorno[%d]", sRet);
            return HCCL_E_MEMORY;
        }
        wlistInfosVec.push_back(wlistInfo);
    }

    listenSocket_->DelWhiteList(wlistInfosVec);

    HCCL_INFO("delete socket white list success. total: %zu", whitelist.size());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::StopSocketListen(const std::vector<HcclIpAddress>& whitelist, u32 hostPort)
{
    if (listenSocket_) {
        if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
            CHK_RET(DeleteSocketWhiteList(hostPort, whitelist));
        }
        if (isByMasterInfo_ || !GetExternalInputHostPortSwitch()) {
            CHK_RET(listenSocket_->DeInit());
        } else {
            s32 deviceLogicId = INVALID_INT;
            CHK_RET(hrtGetDevice(&deviceLogicId));
            CHK_RET(PreemptPortManager::GetInstance(deviceLogicId).Release(listenSocket_));
        }
        listenSocket_ = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::StopNetwork(const std::vector<HcclIpAddress>& whitelist, u32 hostPort)
{
    std::unique_lock<std::mutex> lock(lock_);
    CHK_RET(StopSocketListen(whitelist, hostPort));

    netDevCtx_ = nullptr;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::RecvRemoteAgentID(std::shared_ptr<HcclSocket> socket, std::string& agentID)
{
    char agentBuf[MAX_AGENT_BUF_SIZE] = {0};
    HcclResult ret = socket->Recv(agentBuf, sizeof(agentBuf));
    agentBuf[MAX_AGENT_BUF_SIZE - 1] = '\0';
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Recv][RemoteRankID]GetRemoteRankID receive rank id failed. ret[%d] ", ret),
        ret);
    agentID = agentBuf;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::RecvRemoteRankNum(std::shared_ptr<HcclSocket> socket, u32& remoteRankNum)
{
    HcclResult ret = socket->Recv(reinterpret_cast<char*>(&remoteRankNum), sizeof(remoteRankNum));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Recv][RemoteRankNum]GetRemoteRankID receive rank num failed. ret[%d]", ret),
        ret);
    CHK_PRT_RET(
        (remoteRankNum == 0),
        HCCL_ERROR("[Recv][RemoteRankNum]GetRemoteRankNum receive rank num "
                   "failed. rank num is zero."),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::VerifyRemoteRankNum(u32& previousRankNum, u32 remoteRankNum) const
{
    if (previousRankNum == 0) {
        previousRankNum = remoteRankNum;
    } else {
        CHK_PRT_RET(
            (remoteRankNum != previousRankNum),
            HCCL_ERROR(
                "[Verify][RemoteRankNum]VerifyRemoteRankNum failed. remoteRankNum[%u] is different "
                "from others[%u].",
                remoteRankNum, previousRankNum),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::GetRanksBasicInfo(
    const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, RankTable_t& rankTable)
{
    HcclResult ret;
    u32 socketIndex = 0; // socket已经经过rankid（or superPodId + serverip + deviceid排序）
    for (auto& handle : connectSockets) {
        ret = GetRankBasicInfo(handle.second, rankTable);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Get][RanksBasicInfo]GetRankBasicInfo from agentId[%s] failed, ret[%d]", handle.first.c_str(), ret),
            ret);
        if (isByMasterInfo_ && rankTable.rankList.size() > 0) { // masterInfo场景下无法获取rankid
            rankTable.rankList.back().rankId = socketIndex;
            connectSocketsWithRankID_.insert({socketIndex, handle.second});
        }

        HCCL_INFO(
            "GetRankBasicInfo from agentId[%s] rankId[%u] success.", handle.first.c_str(),
            rankTable.rankList.back().rankId);
        socketIndex++;
    }
    CHK_RET(SortRankList(rankTable));
    currentStep_++;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::GetRanksTransInfo(
    const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, RankTable_t& rankTable)
{
    HcclResult ret;
    u32 socketIndex = 0;
    for (auto& handle : connectSockets) {
        RankTable_t tmpRankTable;
        ret = RecvClusterInfoMsg(handle.second, tmpRankTable);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Get][RanksTransInfo]RecvClusterInfoMsg from rank[%s] failed, ret[%u]", handle.first.c_str(), ret),
            ret);
        CHK_PRT_RET(
            tmpRankTable.rankList.size() == 0,
            HCCL_ERROR("[Get][RanksTransInfo]received rank list "
                       "is empty."),
            HCCL_E_INTERNAL);
        for (u32 i = 0; i < tmpRankTable.rankList.size(); i++) {
            u32 currRank = isByMasterInfo_ ? socketIndex : tmpRankTable.rankList[i].rankId;
            if ((tmpRankTable.rankList[i].transportInfo.size()) != 0) {
                if (rankTable.rankList[currRank].transportInfo.size() == 0) {
                    rankTable.rankList[currRank] = tmpRankTable.rankList[i];
                } else {
                    HCCL_ERROR("[Get][RanksTransInfo]GetRanksTransInfo: rank[%u] transportInfo has existed.", currRank);
                    return HCCL_E_INTERNAL;
                }
            }
        }
        socketIndex++;
        HCCL_INFO("RecvClusterInfoMsg from rank[%s] success.", handle.first.c_str());
    }
    currentStep_++;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::SendIdentify(std::shared_ptr<HcclSocket> socket, u32 identify) const
{
    HcclResult ret = socket->Send(&identify, sizeof(identify));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][ClusterInfoMsg]errNo[0x%016llx] ra send identify failed! "
            "ret[%u]",
            HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), ret),
        ret);

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::GetRankBasicInfo(std::shared_ptr<HcclSocket> socket, RankTable_t& rankTable)
{
    RankTable_t tmpRankTable;
    CHK_RET(RecvClusterInfoMsg(socket, tmpRankTable));

    CHK_PRT_RET(
        tmpRankTable.rankList.size() == 0,
        HCCL_ERROR("[Get][RankBasicInfo]received rank list is "
                   "empty."),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        tmpRankTable.serverList.size() == 0,
        HCCL_ERROR("[Get][RankBasicInfo]received server list "
                   "is empty."),
        HCCL_E_INTERNAL);

    for (u32 i = 0; i < tmpRankTable.rankList.size(); i++) {
        rankTable.rankList.push_back(tmpRankTable.rankList[i]);
    }

    if (rankTable.serverList.size() == 0) {
        rankTable.serverList = tmpRankTable.serverList;
    } else {
        for (u32 i = 0; i < tmpRankTable.serverList.size(); i++) {
            if (!DoServerIdExist(rankTable, tmpRankTable.serverList[i].serverId)) {
                rankTable.serverList.push_back(tmpRankTable.serverList[i]);
            }
        }
    }

    CHK_RET(GetCommonTopoInfo(rankTable, tmpRankTable));

    return HCCL_SUCCESS;
}

bool TopoInfoExchangeServer::DoServerIdExist(const RankTable_t& rankTable, const std::string& serverId) const
{
    for (u32 i = 0; i < rankTable.serverList.size(); i++) {
        if (rankTable.serverList[i].serverId == serverId) {
            return true;
        }
    }
    return false;
}

HcclResult TopoInfoExchangeServer::GetCommonTopoInfo(RankTable_t& rankTable, const RankTable_t& orginRankTable) const
{
    if (rankTable.rankNum == 0) {
        rankTable.nicDeploy = orginRankTable.nicDeploy;
        HCCL_INFO("get rank basicInfo nicDeploy[%u]", rankTable.nicDeploy);
    } else {
        CHK_PRT_RET(
            rankTable.nicDeploy != orginRankTable.nicDeploy,
            HCCL_ERROR(
                "[Get][CommonTopoInfo]compare nicDeploy failed. curr[%u], recv[%u]", rankTable.nicDeploy,
                orginRankTable.nicDeploy),
            HCCL_E_INTERNAL);
    }

    rankTable.serverNum = rankTable.serverList.size();
    rankTable.rankNum = rankTable.rankList.size();
    CHK_RET(GetDevNum(rankTable.rankList, rankTable.deviceNum));
    CHK_RET(GetSuperPodNum(rankTable.rankList, rankTable.superPodNum));
    HCCL_INFO(
        "get rank basicInfo serverNum[%u] rankNum[%u] deviceNum[%u] superPodNum[%u], nicDeploy[%u].",
        rankTable.serverNum, rankTable.rankNum, rankTable.deviceNum, rankTable.superPodNum, rankTable.nicDeploy);
    return HCCL_SUCCESS;
}

bool RankIdCompare(const RankInfo_t& i, const RankInfo_t& j) { return (i.rankId > j.rankId); }

HcclResult TopoInfoExchangeServer::SortRankList(RankTable_t& rankTable) const
{
    std::sort(rankTable.rankList.begin(), rankTable.rankList.end(), RankIdCompare);
    return HCCL_SUCCESS;
}
} // namespace hccl
