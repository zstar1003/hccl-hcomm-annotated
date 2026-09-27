/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cluster_monitor.h"
#include "hccl_types.h"
#include "hccl_comm_pub.h"
#include "env_config/env_config_v2.h"
#include "log.h"

#include "hcclCommTaskException.h"
#include "ccuTaskException.h"
#include "coll_comm_mgr.h"
#include "heartbeat.h"
#include "comm_addr_logger.h"

constexpr u32 ONE_SECOND_OF_SLEEP = 1; // 1s
namespace hcomm {

ClusterMonitor::~ClusterMonitor() { DeInit(); }

ClusterUIDType ClusterMonitor::FormatUID(ClusterUIDCxt cxt) const
{
    ClusterUIDType uid{};
    // 构造唯一的uid: netInstanceId + local_id
    (void)snprintf_s(
        uid.id, sizeof(uid.id), sizeof(uid.id) - 1, "%s/%s", cxt.netInstId.c_str(),
        std::to_string(cxt.localId).c_str());

    return uid;
}

std::string ClusterMonitor::GetUID(const ClusterUIDType& uid) const { return uid.id; }

void ClusterMonitor::GetRemEndpointDescsPerLayer(
    uint32_t netLayer, HcclComm comm, const Hccl::RankGraph* rankGraph, const hccl::CollComm* collComm,
    std::map<uint32_t, std::vector<UIDContext>>& uidCtxs, std::set<uint32_t>& rankIdsSet)
{
    uint32_t* ranksPerLayer = nullptr;
    uint32_t rankNum = 0;
    auto myRankId = collComm->GetMyRankId();
    HcclRankGraphGetRanksByLayer(comm, netLayer, &ranksPerLayer, &rankNum); // 获取每层netLayer的所有rank
    for (uint32_t rankIdx = 0; rankIdx < rankNum; rankIdx++) {
        uint32_t rankId = ranksPerLayer[rankIdx];
        if (rankIdsSet.find(rankId) != rankIdsSet.end()) {
            continue; // rankSet维护了所有的ranks，如果已经加到Set说明该rank已经在更低的netLayer层级加入
        }
        rankIdsSet.insert(rankId);
        auto* netInstance = rankGraph->GetNetInstanceByRankId(0, rankId); // 查询对应rankId在netLayer=0的netInsId
        if (netInstance == nullptr) {
            continue; // 如果没有查询到netInstance，不报错，不把该rank加入needConnectRank，直接跳过该rank
        }
        auto netInstanceId = netInstance->GetNetInstId();
        auto localId = rankGraph->GetLocalId(rankId); // 根据rank查localId
        ClusterUIDCxt uidcxt(netInstanceId, localId);
        ClusterUIDType uid = FormatUID(uidcxt);
        if (myRankId == rankId) {
            myRankUID_ = uid;
            myRankLocalId_ = localId;
            myRankNetInstId_ = netInstanceId;
        }
        uid2FrameStatusMap_.insert(uid, FrameStatus());
        commIdMap_[collComm->GetCommId()].insert(std::make_pair(uid, false)); // 初始状态均为未连接，包含自己
        if (uidCtxs.find(netLayer) == uidCtxs.end()) {
            uidCtxs.insert(std::make_pair(netLayer, std::vector<UIDContext>()));
        }
        UIDContext uidCtx(uid, netLayer, rankId, localId, netInstanceId);
        uidCtxs[netLayer].emplace_back(uidCtx);
        HCCL_INFO("commId[%s] insert remoteUID[%s]", collComm->GetCommId().c_str(), GetUID(uid).c_str());
    }
}

HcclResult ClusterMonitor::GetRemEndpointDescs(
    HcclComm comm, std::map<uint32_t, std::vector<UIDContext>>& uidCtxs, std::vector<uint32_t>& netLayersVector)
{
    // 将所有远端的rank都加入到状态维护map中
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    Hccl::HcclCommunicator* commV2 = static_cast<Hccl::HcclCommunicator*>(collComm->GetCommunicatorV2());
    CHK_PTR_NULL(commV2); // 获取到legacy communicator，说明v2通信域
    void* rankGraphPtr = nullptr;
    CHK_RET(commV2->GetRankGraphV2(rankGraphPtr));
    CHK_PTR_NULL(rankGraphPtr);
    Hccl::RankGraph* rankGraph = static_cast<Hccl::RankGraph*>(rankGraphPtr);

    // 获取netLayer信息存入到netLayersVector中
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if (netLayerNum == 0) {
        HCCL_WARNING("[%s] no netLayer in RankGraph", __func__);
        return HCCL_SUCCESS;
    }
    netLayersVector.assign(netLayers, netLayers + netLayerNum);
    std::sort(netLayersVector.begin(), netLayersVector.end());
    std::set<uint32_t>
        rankIdsSet; // 存放通信域的唯一标识ranks，防止在netLayer>=1的时候，查到了netLayer=0已经存放的ranks
    for (auto netLayer : netLayersVector) {
        GetRemEndpointDescsPerLayer(netLayer, comm, rankGraph, collComm, uidCtxs, rankIdsSet);
    }

    return HCCL_SUCCESS;
}

std::string ClusterMonitor::FormatConnTag(HcommSocketRole role, std::pair<ClusterUIDType, ClusterUIDType> uidPair) const
{
    std::string tag;
    if (role == HcommSocketRole::HCOMM_SOCKET_ROLE_CLIENT) {
        tag = "HeartBeat_" + GetUID(uidPair.first) + "_to_" + GetUID(uidPair.second);
    } else {
        tag = "HeartBeat_" + GetUID(uidPair.second) + "_to_" + GetUID(uidPair.first);
    }

    return tag;
}

HcclResult ClusterMonitor::GetSocketDescFromRankInfo(
    HcclComm comm, uint32_t remoteRank, uint32_t netLayer, const ClusterUIDType& remoteUID, SocketDesc& socketDesc)
{
    uint32_t rmtPort = 0;
    uint32_t listenPort = 0;
    hccl::CollComm* collComm = static_cast<hccl::hcclComm*>(comm)->GetCollComm();
    auto rankGraph = collComm->GetRankGraph();
    auto myRankId = collComm->GetMyRankId();
    CHK_PTR_NULL(rankGraph);
    CHK_RET(rankGraph->GetDevicePort(remoteRank, &rmtPort));
    if (rmtPort > Hccl::MAX_VALUE_TCPPORT) {
        HCCL_ERROR(
            "[%s] Invalid port[%u] of Rank[%u], max valid port is %u", __func__, rmtPort, remoteRank,
            Hccl::MAX_VALUE_TCPPORT);
        return HCCL_E_PARA;
    }
    CommLink* links = nullptr;
    uint32_t linkNum = 0;
    HcclResult result = HcclRankGraphGetLinks(comm, netLayer, myRankId, remoteRank, &links, &linkNum);
    if (result != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[%s] Get links between myRank[%u] and remoteRank[%u] failed, ret:%d", __func__, myRankId, remoteRank,
            result);
        return HCCL_E_NOT_FOUND;
    }
    // 如果没有查询到任何链接，不报错，不把该link加入needConnectRank，直接返回成功
    if (linkNum == 0 || links[0].srcEndpointDesc.loc.locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST
        || links[0].dstEndpointDesc.loc.locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
        HCCL_INFO("[%s] no link between myRank[%u] and remoteRank[%u]", __func__, myRankId, remoteRank);
        return HCCL_E_NOT_FOUND;
    }
    // 查询该socket链接的server端监听的端口（监听方的选择策略需要跟SocketConfig中保持一致）
    Hccl::IpAddress localIpAddr{};
    Hccl::IpAddress remoteIpAddr{};
    CHK_RET(CommAddrToIpAddress(links[0].srcEndpointDesc.commAddr, localIpAddr));
    CHK_RET(CommAddrToIpAddress(links[0].dstEndpointDesc.commAddr, remoteIpAddr));
    if (localIpAddr < remoteIpAddr) { // local地址比remote地址小时，local作为server监听端
        // 查询localRankId对应的devPort
        CHK_RET(rankGraph->GetDevicePort(myRankId, &listenPort));
        socketDesc.role = HcommSocketRole::HCOMM_SOCKET_ROLE_SERVER;
        if (listenPort > Hccl::MAX_VALUE_TCPPORT) {
            HCCL_ERROR(
                "[%s] Invalid port[%u] of Rank[%u], max valid port is %u", __func__, listenPort, myRankId,
                Hccl::MAX_VALUE_TCPPORT);
            return HCCL_E_PARA;
        }
        socketDesc.listenPort = static_cast<uint16_t>(listenPort); // socketDesc.port中填监听端口号
    } else {
        socketDesc.role = HcommSocketRole::HCOMM_SOCKET_ROLE_CLIENT;
        socketDesc.listenPort
            = static_cast<uint16_t>(rmtPort); // socketDesc.port中填对端端口号(此场景下对端端口号也就是监听端口号)
    }
    // socket建链需要心跳专用的tag，用来区分业务的socket以及心跳的socket
    std::string tag = FormatConnTag(socketDesc.role, std::make_pair(myRankUID_, remoteUID));
    errno_t ret = memcpy_s(socketDesc.tag, sizeof(socketDesc.tag), tag.c_str(), tag.size() + 1);
    CHK_PRT_RET(
        (ret != EOK),
        HCCL_ERROR("[%s] memcpy_s failed, ret:%d, errno:%d, error:%s", __func__, ret, errno, strerror(errno)),
        HCCL_E_SYSCALL);
    socketDesc.localEndpoint = links[0].srcEndpointDesc;
    socketDesc.remoteEndpoint = links[0].dstEndpointDesc;
    return HCCL_SUCCESS;
}

HcclResult ClusterMonitor::InsertClusterMonitorCtx(
    HcclComm comm, UIDContext remoteCtx, std::map<ClusterUIDType, ClusterMonitorSocketCtx>& needConnectRank)
{
    bool newConn = true;
    SocketDesc socketDesc{};
    auto remoteUID = remoteCtx.uid;
    auto remoteRank = remoteCtx.rankId;
    auto netLayer = remoteCtx.netLayer;

    std::unique_lock<std::mutex> lock(threadLock_);
    if (monitorLinkStatusMap_.find(remoteUID) == monitorLinkStatusMap_.end()) {
        monitorLinkStatusMap_[remoteUID] = MonitorLinkStatus::MONITOR_LINK_NOT_START;
    } else if (
        monitorLinkStatusMap_[remoteUID] == MonitorLinkStatus::MONITOR_LINK_BUILDING
        || monitorLinkStatusMap_[remoteUID] == MonitorLinkStatus::MONITOR_LINK_COMPLETED) {
        newConn = false; // 说明之前已经有remoteUID在建链
    }

    // 获取端口号用来建链
    HcclResult ret = GetSocketDescFromRankInfo(comm, remoteRank, netLayer, remoteUID, socketDesc);
    if (ret == HCCL_E_NOT_FOUND) {
        return HCCL_SUCCESS; // 本rank无有效链接，不加入needConnectRank
    }
    ClusterMonitorSocketCtx ctx(socketDesc, newConn);
    needConnectRank.insert(std::make_pair(remoteUID, ctx));
    HCCL_INFO(
        "[%s] InsertClusterMonitorCtx for myRankUID_[%s], remoteUID[%s], role[%s], localEndpoint[commAddr:%s], "
        "remoteEndpoint[commAddr:%s], tag[%s], listenPort [%u], newConn[%d]",
        __func__, GetUID(myRankUID_).c_str(), GetUID(remoteUID).c_str(),
        (socketDesc.role == HcommSocketRole::HCOMM_SOCKET_ROLE_SERVER) ? "SERVER" : "CLIENT",
        hcomm::logger::CommAddrLogger::ToString(socketDesc.localEndpoint.commAddr).c_str(),
        hcomm::logger::CommAddrLogger::ToString(socketDesc.remoteEndpoint.commAddr).c_str(), socketDesc.tag,
        socketDesc.listenPort, newConn);
    return HCCL_SUCCESS;
}

HcclResult ClusterMonitor::GetSamePlaneRank(
    HcclComm comm, std::vector<UIDContext> singlePlaneCtx,
    std::map<ClusterUIDType, ClusterMonitorSocketCtx>& needConnectRank)
{
    uint32_t index = 0;
    for (; index < singlePlaneCtx.size(); index++) {
        if (singlePlaneCtx[index].uid == this->myRankUID_) { // 找出myRank在vector中的下标
            break;
        }
    }

    uint32_t singlePlaneSize = singlePlaneCtx.size(); // 包含myRank自己，一个平面所有的节点
    if (singlePlaneSize <= 1) {                       // 待连接的节点个数为0或1，无需连接
        HCCL_INFO("[%s] no need to connect", __func__);
        return HCCL_SUCCESS;
    } else if (singlePlaneSize == 2) { // 待连接的节点个数为2，不需要双ring环，一条边就够了
        uint32_t nextIndex = (index + 1) % singlePlaneSize; // 算出与本Rank相连，对端的节点
        HCCL_INFO("[%s] singlePlaneSize is 2, only connect nextIndex[%u]", __func__, nextIndex);
        CHK_RET(InsertClusterMonitorCtx(comm, singlePlaneCtx[nextIndex], needConnectRank));
    } else {
        uint32_t nextIndex = (index + 1) % singlePlaneSize;                  // 算出与本Rank相连，右手的节点
        uint32_t preIndex = (index + singlePlaneSize - 1) % singlePlaneSize; // 算出与本Rank相连，左手或回绕环的节点
        HCCL_INFO(
            "[%s] singlePlaneSize is %u, connect nextIndex[%u], preIndex[%u]", __func__, singlePlaneSize, nextIndex,
            preIndex);
        CHK_RET(InsertClusterMonitorCtx(comm, singlePlaneCtx[nextIndex], needConnectRank)); // 以本rank为起点，环的右手
        CHK_RET(InsertClusterMonitorCtx(comm, singlePlaneCtx[preIndex], needConnectRank)); // 以本rank为起点，环的左手
    }

    return HCCL_SUCCESS;
}

HcclResult ClusterMonitor::GetConnectRank(
    HcclComm comm, std::map<ClusterUIDType, ClusterMonitorSocketCtx>& needConnectRank,
    std::map<uint32_t, std::vector<UIDContext>> uidCtxs, std::vector<uint32_t>& netLayersVector)
{
    if (netLayersVector.empty() || uidCtxs.empty()) {
        HCCL_INFO("[%s] netLayersVector is empty, no netLayer in RankGraph", __func__);
        return HCCL_SUCCESS;
    }

    std::vector<UIDContext> layer0CommLinks; // 需要存入UIDContext，待后续查出对应的port/remoteUID
    // 先处理netLayer=0，按照netLayer=0全局唯一的localId升序排列，在level0不需要考虑host网卡的场景，host网卡只会在level1及以上的层级
    std::sort(uidCtxs[0].begin(), uidCtxs[0].end(), [&](const UIDContext& a, const UIDContext& b) {
        return a.localId < b.localId;
    });
    for (auto it = uidCtxs[0].begin(); it != uidCtxs[0].end(); ++it) {
        layer0CommLinks.push_back(*it); // netLayer为0
    }

    // 从layer=1开始，将commLinks存入vector中，找到所有与当前localId相同的节点
    std::vector<UIDContext> highLayerCommLinks;
    for (uint32_t netLayer : netLayersVector) {
        for (auto it = uidCtxs[netLayer].begin(); it != uidCtxs[netLayer].end(); ++it) {
            if (it->localId == this->myRankLocalId_) {
                // 在跨server、跨pod、跨超节点的场景，统一拿到local，打平处理为同一个平面，类似layer=0的情况
                // 由于A5上的devPhyId在64卡的场景下8个[0,7]，所以使用localId
                highLayerCommLinks.push_back(*it);
            }
        }
    }
    std::sort(highLayerCommLinks.begin(), highLayerCommLinks.end(), [&](const UIDContext& a, const UIDContext& b) {
        return a.netInstId < b.netInstId;
    });

    // 每个平面都分别成环
    CHK_RET(GetSamePlaneRank(comm, layer0CommLinks, needConnectRank));
    CHK_RET(GetSamePlaneRank(comm, highLayerCommLinks, needConnectRank));
    return HCCL_SUCCESS;
}

void ClusterMonitor::CreateHBLinksAsync()
{
    std::unique_lock<std::mutex> linksLock(clusertMonitorLinkMtx_);
    if (clusterLinkContext_.empty()) {
        return;
    }
    linkThreadRunning_ = true;
    std::queue<std::tuple<std::string, ClusterUIDType, ClusterMonitorSocketCtx>> connInfoQueue;
    for (auto& pair : clusterLinkContext_) {
        const std::string& commId = pair.first;
        auto& commIdConnInfoQueue = pair.second;
        while (!commIdConnInfoQueue.empty()) {
            connInfoQueue.push(
                std::make_tuple(commId, commIdConnInfoQueue.front().first, commIdConnInfoQueue.front().second));
            commIdConnInfoQueue.pop();
        }
    }
    linksLock.unlock();

    while (!connInfoQueue.empty()) {
        const std::string commId = std::get<0>(connInfoQueue.front());
        const ClusterUIDType& remUID = std::get<1>(connInfoQueue.front());
        ClusterMonitorSocketCtx& connInfo = std::get<2>(connInfoQueue.front());
        connInfo.PrintSocketDesc("CreateHBLinksAsync");
        auto it = linkThreadMap_.find(remUID);
        if (it != linkThreadMap_.end() && it->second->joinable()) {
            it->second->join();
            HCCL_INFO(
                "[CreateMonitorLinksAsync] monitor link thread has been joined. commId[%s], remote uid[%s].",
                commId.c_str(), GetUID(remUID).c_str());
        }
        linkThreadMap_[remUID].reset(
            new (std::nothrow) std::thread(&ClusterMonitor::CreateLinkWithRemotePonit, this, commId, remUID, connInfo));
        if (linkThreadMap_[remUID] == nullptr) {
            HCCL_RUN_WARNING(
                "commId[%s] establish rank[%s] to rank[%s] heartbeat connection failed. Reason: "
                "create thread failed.",
                commId.c_str(), GetUID(myRankUID_).c_str(), GetUID(remUID).c_str());
        }
        connInfoQueue.pop();
    }
    return;
}

HcclResult ClusterMonitor::CreateTransportHandle(ClusterMonitorSocketCtx& info) const
{
    info.PrintSocketDesc("CreateTransportHandle");
    if (info.socketHandler == nullptr) {
        return SocketCreate(&info.socketDesc, &info.socketHandler);
    }

    HCCL_WARNING("[CreateTransportHandle] socketHandler has been created, skip.");
    return HCCL_SUCCESS;
}

void ClusterMonitor::CreateLinkWithRemotePonit(
    std::string commId, ClusterUIDType rem, ClusterMonitorSocketCtx needConnectRank)
{
    // 给当前线程添加名字
    const std::string threadName = "hb" + GetUID(rem);
    SetThreadName(threadName);
    hrtSetDevice(deviceLogicId_);

    HcclResult ret = CreateTransportHandle(needConnectRank);
    if (ret != HCCL_SUCCESS) {
        HCCL_RUN_WARNING(
            "[CreateLinkWithRemote] CreateTransportHandle ret[%d], commId[%s], remote uid[%s].", ret, commId.c_str(),
            GetUID(rem).c_str());
        hrtResetDevice(deviceLogicId_);
        return;
    }

    auto createLinkTimeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    while (linkThreadRunning_.load()) {
        if ((std::chrono::steady_clock::now() - startTime) >= createLinkTimeout) {
            HCCL_RUN_WARNING(
                "establish rank[%s] to rank[%s] connection failed. Reason: link timeout,"
                "timeout[%llds], the HCCL_CONNECT_TIMEOUT may be insufficient. commId[%s].",
                GetUID(myRankUID_).c_str(), GetUID(rem).c_str(), createLinkTimeout.count(), commId.c_str());
            break;
        }

        SocketStates status;
        HcclResult ret = SocketGetStatus(needConnectRank.socketHandler, &status);
        if (ret != HCCL_SUCCESS) {
            HCCL_RUN_WARNING(
                "establish rank[%s] to rank[%s] connection failed. Reason: get socket status[%d] failed, commId[%s]",
                GetUID(myRankUID_).c_str(), GetUID(rem).c_str(), status, commId.c_str());
            SocketDestroy(needConnectRank.socketHandler);
            break;
        }

        if (status == SocketStates::SOCKET_TIMEOUT) {
            HCCL_RUN_WARNING(
                "establish rank[%s] to rank[%s] connection failed. Reason: get socket status timeout, commId[%s]",
                GetUID(myRankUID_).c_str(), GetUID(rem).c_str(), commId.c_str());
            SocketDestroy(needConnectRank.socketHandler);
            break;
        } else if (status == SocketStates::SOCKET_CONNECTING) {
            SalSleep(ONE_SECOND_OF_SLEEP);
            continue;
        }

        ret = OnConnectionEstablished(commId, rem, needConnectRank);
        if (ret != HCCL_SUCCESS) {
            HCCL_RUN_WARNING("OnConnectionEstablished not success, ret[%d]", ret);
        }
        break;
    }
    hrtResetDevice(deviceLogicId_);

    HCCL_INFO("[%s] Thread [%s] end...", __func__, threadName.c_str());
    return;
}

HcclResult ClusterMonitor::OnConnectionEstablished(
    const std::string& commId, const ClusterUIDType& rem, ClusterMonitorSocketCtx& needConnectRank)
{
    std::unique_lock<std::mutex> lock(threadLock_);
    if (commIdMap_.find(commId) == commIdMap_.end()) {
        HCCL_RUN_WARNING(
            "establish rank[%s] to rank[%s] connection failed. Reason: commId[%s] has been Unregistered.",
            GetUID(myRankUID_).c_str(), GetUID(rem).c_str(), commId.c_str());
        SocketDestroy(needConnectRank.socketHandler);
        lock.unlock();
        return HCCL_E_INTERNAL;
    }
    needConnectRank.newConn = false;
    uid2SocketRefMap_.insert(rem, needConnectRank);
    // 心跳socket建链完成后，需要立即及激活其心跳收发能力
    auto frameSize = sizeof(ClusterMonitorFrame);
    if (uid2SocketRefMap_[rem].recvBuffer.Init(hccl::BASE_NUMBER * frameSize)
        != HCCL_SUCCESS) { // 2倍帧长，确保不会溢出
        HCCL_RUN_WARNING(
            "establish rank[%s] to rank[%s] connection failed. Reason: socket recv buffer init failed. commId[%s].",
            GetUID(myRankUID_).c_str(), GetUID(rem).c_str(), commId.c_str());
        SocketDestroy(needConnectRank.socketHandler);
        uid2SocketRefMap_.erase(rem);
        lock.unlock();
        return HCCL_E_INTERNAL;
    }
    monitorLinkStatusMap_[rem] = MonitorLinkStatus::MONITOR_LINK_COMPLETED;
    commIdMap_[commId][rem] = true; // 更新状态为已连接
    lock.unlock();
    HCCL_RUN_INFO(
        "commId:[%s], establish rank[%s] to rank[%s] heartbeat connection success.", commId.c_str(),
        GetUID(myRankUID_).c_str(), GetUID(rem).c_str());
    return HCCL_SUCCESS;
}

HcclResult ClusterMonitor::SendFrameFromBuffer(ClusterUIDType& dst, ClusterMonitorFrame& cmFrame)
{
    if (cmFrame.status != ClusterMonitorStatus::CLUSTER_MONITOR_OK
        && uid2SocketRefMap_[dst].sendBuffer.size() < hccl::MAX_SENDBUFF_SIZE) {
        uid2SocketRefMap_[dst].sendBuffer.push(cmFrame);
    }
    while (uid2SocketRefMap_[dst].sendBuffer.size() > 0) {
        ClusterMonitorFrame hbf = uid2SocketRefMap_[dst].sendBuffer.front();
        u64 sendDis = sizeof(ClusterMonitorFrame) - uid2SocketRefMap_[dst].restSize;
        uint64_t compSize = 0;
        void* sendPtr = static_cast<char*>(static_cast<void*>(&hbf)) + sendDis;
        HcclResult ret
            = SocketSendNb(uid2SocketRefMap_[dst].socketHandler, sendPtr, uid2SocketRefMap_[dst].restSize, &compSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[CreateTransportHandle] SocketSendNb failed, ret[%d]", ret);
            return ret;
        }
        if (uid2SocketRefMap_[dst].restSize == compSize) {
            uid2SocketRefMap_[dst].sendBuffer.pop();
            uid2SocketRefMap_[dst].restSize = sizeof(ClusterMonitorFrame);
            HCCL_DEBUG(
                "[Heartbeat][SendFrame] Send Success, from [%s] to [%s] about [%s] by [%s] status[%d]",
                GetUID(myRankUID_).c_str(), GetUID(dst).c_str(), GetUID(cmFrame.crimer).c_str(),
                GetUID(cmFrame.informer).c_str(), cmFrame.status);
        } else {
            uid2SocketRefMap_[dst].restSize = uid2SocketRefMap_[dst].restSize - compSize;
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ClusterMonitor::SendFrame(
    ClusterUIDType& dst, ClusterUIDType& crimer, ClusterUIDType& informer, ClusterMonitorStatus status)
{
    ClusterMonitorFrame cmFrame(myRankUID_, dst, crimer, informer, status);
    if (uid2SocketRefMap_[dst].sendBuffer.size() > 0) {
        HcclResult ret = SendFrameFromBuffer(dst, cmFrame);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_WARNING("[SendFrameFromBuffer] failed, ret[%d]", ret), ret);
    } else {
        uint64_t compSize = 0;
        uint64_t expectSize = sizeof(ClusterMonitorFrame);
        HcclResult ret = SocketSendNb(uid2SocketRefMap_[dst].socketHandler, &cmFrame, expectSize, &compSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[CreateTransportHandle] SocketSendNb failed, ret[%d]", ret);
            return ret;
        }
        if (compSize == expectSize) {
            HCCL_DEBUG(
                "[Heartbeat][SendFrame] Send Success, from [%s] to [%s] about [%s] by [%s] status[%d]",
                GetUID(myRankUID_).c_str(), GetUID(dst).c_str(), GetUID(crimer).c_str(), GetUID(informer).c_str(),
                status);
        } else {
            HCCL_DEBUG(
                "[Heartbeat][SendFrame] Send Not Complete, from [%s] to [%s] about [%s] by [%s] status[%d], "
                "expectSize[%llu], compSize[%llu]",
                GetUID(myRankUID_).c_str(), GetUID(dst).c_str(), GetUID(crimer).c_str(), GetUID(informer).c_str(),
                status, expectSize, compSize);
            uid2SocketRefMap_[dst].restSize = expectSize - compSize;
            uid2SocketRefMap_[dst].sendBuffer.push(cmFrame);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ClusterMonitor::RecvFrame(ClusterUIDType rem)
{
    ClusterMonitorFrame cmFrame;
    u64 compSize = 0;
    u64 expectSize = sizeof(ClusterMonitorFrame);
    // 此处while循环用于最大限度的从socket中读取数据，直到没有数据可读或者发生错误。
    // 因为心跳帧较小，理论上一次recv就能读完。但为了兼容可能存在的粘包情况，增加循环读取的逻辑。
    while (true) {
        compSize = 0;
        HcclResult ret = SocketRecvNb(
            uid2SocketRefMap_[rem].socketHandler, &cmFrame, expectSize, (reinterpret_cast<uint64_t*>(&compSize)));
        if (ret == HCCL_SUCCESS && compSize > 0) {
            uid2SocketRefMap_[rem].recvBuffer.PushSeg(reinterpret_cast<u8*>(&cmFrame), compSize);
            if (uid2SocketRefMap_[rem].recvBuffer.Size() >= expectSize) {
                uid2SocketRefMap_[rem].recvBuffer.GetSeg(reinterpret_cast<u8*>(&cmFrame), expectSize);
                uid2SocketRefMap_[rem].recvBuffer.PopSeg(expectSize);
                CHK_RET(ParseFrame(cmFrame, rem));
            }
        } else if (ret == HCCL_E_INTERNAL) {
            HCCL_WARNING("SocketRecvNb recv rem[%s] fail", GetUID(rem).c_str());
            return ret;
        } else {
            // 当没有数据可读时，SocketRecvNb会返回成功但compSize为0，此时退出循环，继续进行后续的心跳发送和异常处理等逻辑
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ClusterMonitor::ParseFrame(ClusterMonitorFrame& cmFrame, ClusterUIDType& src)
{
    if (cmFrame.src != src || cmFrame.dst != myRankUID_) {
        HCCL_WARNING("rank[%s] recv wrong frame", GetUID(myRankUID_).c_str());
        return HCCL_E_INTERNAL;
    }

    HCCL_DEBUG(
        "[ClusterMonitor][ParseMonitorFrame] Recv Success, from [%s] to [%s] about [%s] by [%s] state[%d]",
        GetUID(cmFrame.src).c_str(), GetUID(cmFrame.dst).c_str(), GetUID(cmFrame.crimer).c_str(),
        GetUID(cmFrame.informer).c_str(), cmFrame.status);

    // 能够收到进程卡住表示心跳是正常的
    if (cmFrame.status == ClusterMonitorStatus::CLUSTER_MONITOR_OK) {
        uid2SocketRefMap_[src].lostNum = 0;
    }

    // 只有心跳非正常时才需要打印TRACE
    if (cmFrame.status != ClusterMonitorStatus::CLUSTER_MONITOR_OK) {
        SetStatus(cmFrame.crimer, cmFrame.informer, cmFrame.status); // 设置异常状态
    }

    return HCCL_SUCCESS;
}

void ClusterMonitor::DelErrorSocket()
{
    for (auto rem : errorSocket_) {
        HCCL_RUN_INFO(
            "rank[%s] Try to Send/recv HeartBeat to rank[%s]", GetUID(myRankUID_).c_str(), GetUID(rem).c_str());
        uid2FrameStatusMap_.erase(rem);
        if (uid2SocketRefMap_.has(rem)) {
            SocketDestroy(uid2SocketRefMap_[rem].socketHandler);
            while (uid2SocketRefMap_.erase(rem) != 0) {
            };
        }
    }
    errorSocket_.clear();
}

void ClusterMonitor::SetStatus(
    ClusterUIDType& crimer, ClusterUIDType& informer, ClusterMonitorStatus status, bool needBroadcast)
{
    if (uid2FrameStatusMap_[crimer].status != status) {
        uid2FrameStatusMap_[crimer].informer = informer;
        uid2FrameStatusMap_[crimer].status = status;
        uid2FrameStatusMap_[crimer].needBroadcast = needBroadcast;
        if (needBroadcast) {
            errRankQueue_.push(crimer);
        }

        errStatusQueue_.push(
            ClusterMonitorFrame(crimer, informer, status, TIME_NOW(), std::chrono::system_clock::now()));
        if (errStatusQueue_.size() > hccl::EVENT_MAX_CNT) {
            errStatusQueue_.pop();
        }
        HCCL_RUN_INFO(
            "[%s][%s]local rank [%s]: crimer rank [%s] status[%s] by informer rank [%s]",
            LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_HEARTBEAT_EVENT.c_str(), GetUID(myRankUID_).c_str(),
            GetUID(crimer).c_str(), GetClusterMonitorStatusStr(status).c_str(), GetUID(informer).c_str());
    }
}

HcclResult ClusterMonitor::ProcessConnectRanks(
    const std::string& commId, std::map<ClusterUIDType, ClusterMonitorSocketCtx>& needConnectRank)
{
    // 将双ring环的pair放入clusterLinkContext_管理多个通信域
    std::unique_lock<std::mutex> linkCtxlock(clusertMonitorLinkMtx_);
    for (auto& item : needConnectRank) {
        if (item.second.newConn == true) {
            // 一旦放入clusterLinkContext_中，就会被后台的异步建链线程推动建链
            clusterLinkContext_[commId].push(std::move(item));
        }
    }
    linkCtxlock.unlock();

    std::unique_lock<std::mutex> lock(threadLock_);
    for (auto& item : needConnectRank) {
        if (item.second.newConn == true) {
            // 由于newConn==true的item已经入队，后台推动异步建链，所以状态迁移为建链中
            monitorLinkStatusMap_[item.first] = MonitorLinkStatus::MONITOR_LINK_BUILDING;
        } else if (
            commIdMap_[commId].find(item.first) == commIdMap_[commId].end()
            || (commIdMap_[commId].count(item.first) && !commIdMap_[commId][item.first])) {
            // 若newConn=false，说明不是新增的连接
            // 1. 通信域找不到，2.通信域内能找到但还没有连接，计数++
            uid2SocketRefMap_.ref(item.first);
            HCCL_RUN_INFO(
                "commId:[%s], establish rank[%s] to rank[%s] heartbeat connection success.", commId.c_str(),
                GetUID(myRankUID_).c_str(), GetUID(item.first).c_str());
            commIdMap_[commId][item.first] = true; // 认为通信域中对应的连接已经建立
        }
    }
    lock.unlock();

    return HCCL_SUCCESS;
}

void ClusterMonitor::MonitorThread()
{
    // 给当前线程添加名字
    SetThreadName("Hccl_HeartBeat");

    hrtSetDevice(deviceLogicId_);
    HcclResult ret = HCCL_SUCCESS;
    uint32_t count = 0;
    while (clusterMonitorThreadFlag_) {
        CreateHBLinksAsync(); // 内部起线程对所有的socket进行异步建链
        threadLock_.lock();
        count++;
        if (count >= hccl::HEARTBEAT_COUNT) {
            count = 0;
            for (auto iter = uid2SocketRefMap_.begin(); iter != uid2SocketRefMap_.end(); iter++) {
                ClusterUIDType rem = iter->first;
                uid2SocketRefMap_[rem].lostNum++;
                // 先发送心跳帧，触发对端回复，才能准确地判断链路状态
                ret = SendFrame(rem, myRankUID_, myRankUID_, ClusterMonitorStatus::CLUSTER_MONITOR_OK);
                ret == HCCL_E_INTERNAL ? errorSocket_.push_back(rem) : void(0);
            }
            DelErrorSocket(); // 处理socket错误的句柄
        }

        for (auto iter = uid2SocketRefMap_.begin(); iter != uid2SocketRefMap_.end(); iter++) {
            ClusterUIDType rem = iter->first;
            ret = RecvFrame(rem);
            if (ret == HCCL_E_INTERNAL) {
                errorSocket_.push_back(rem);
            } else if (uid2SocketRefMap_[rem].lostNum >= lostThreshold_) {
                SetStatus(rem, myRankUID_, ClusterMonitorStatus::CLUSTER_MONITOR_LOST);
            }
        }
        DelErrorSocket();        // 处理socket错误的句柄
        ProcessExceptionEvent(); // 处理error cqe
        threadLock_.unlock();

        std::this_thread::sleep_for(std::chrono::milliseconds(hccl::BROADCAST_INTERVAL));
    }

    linkThreadRunning_ = false;
    // 在心跳进程结束之前join所有的建链线程
    for (auto& pair : linkThreadMap_) {
        if (pair.second != nullptr && pair.second->joinable()) {
            pair.second->join();
            HCCL_INFO("[%s] thread has joined. Remote uid is [%s]", __func__, GetUID(pair.first).c_str());
        }
    }

    hrtResetDevice(deviceLogicId_);
}

HcclResult ClusterMonitor::RunMonitorThread()
{
    HCCL_INFO("[%s] Start ClusterMonitorThread.", __func__);
    clusterMonitorThreadFlag_ = true;
    clusterMonitorThread_.reset(new (std::nothrow) std::thread(&ClusterMonitor::MonitorThread, this));
    CHK_SMART_PTR_NULL(clusterMonitorThread_);
    lostThreshold_ = hccl::HCCL_LOST_THRESHOLD; // 心跳丢失阈值为30s
    initialized_ = true;
    isDeInit_ = false;
    return HCCL_SUCCESS;
}

HcclResult ClusterMonitor::RegisterToClusterMonitor(HcclComm comm)
{
    HCCL_INFO("[%s] RegisterToClusterMonitor begin.", __func__);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    deviceLogicId_ = collComm->GetDeviceLogicId();

    // 单rank无对端，不支持心跳检测
    const std::string& commId = collComm->GetCommId();
    uint32_t rankSize = collComm->GetRankSize();
    CHK_PRT_RET(
        rankSize == 1,
        HCCL_WARNING(
            "[%s] commId[%s] rankSize[%u] no need to register to ClusterMonitor", __func__, commId.c_str(), rankSize),
        HCCL_SUCCESS);

    // 判断该通信域是否曾经添加到commIdMap_中
    std::unique_lock<std::mutex> lock(threadLock_);
    auto iter = commIdMap_.find(commId);
    if (iter != commIdMap_.end()) {
        HCCL_INFO("commId[%s] has Registered, skip.", commId.c_str());
        return HCCL_SUCCESS;
    }

    if (!initialized_) {
        // 开始起监控线程
        CHK_RET(RunMonitorThread());
    }
    lock.unlock();

    // 存放所有节点的上下文
    std::map<uint32_t, std::vector<UIDContext>> uidCtxs;
    std::vector<uint32_t> netLayersVector;

    // 获取从myRank出发，所有的对端，并维护commIdMap_及uid2FrameStatusMap_
    lock.lock();
    CHK_RET(GetRemEndpointDescs(comm, uidCtxs, netLayersVector));
    lock.unlock();

    // 解析heartbeat环境变量，如果配置为off则不去注册对应的rank
    auto clusterHeartBeatEnable = Hccl::EnvConfig::GetInstance().GetLogConfig().GetDfsConfig().clusterHeartBeatEnable;
    if (!clusterHeartBeatEnable) {
        HCCL_RUN_INFO(
            "[%s] HCCL_DFS_CONFIG cluster_heartbeat is off. It's unnecessary to "
            "register Ranks. commId[%s]",
            __func__, commId.c_str());
        return HCCL_SUCCESS;
    }

    // 从所有连接中，选择双ring环，存放到needConnectRank
    std::map<ClusterUIDType, ClusterMonitorSocketCtx> needConnectRank;
    CHK_RET(GetConnectRank(comm, needConnectRank, uidCtxs, netLayersVector));

    // 处理双ring环的连接（入队、更新状态、更新引用计数等）
    CHK_RET(ProcessConnectRanks(commId, needConnectRank));

    HCCL_INFO("[%s] commId[%s] RegisterRanks Completed", __func__, commId.c_str());
    return HCCL_SUCCESS;
}

HcclResult ClusterMonitor::DeInit()
{
    bool expected = false;
    if (!isDeInit_.compare_exchange_strong(expected, true)) {
        HCCL_INFO("[%s] already deinit, skip.", __func__);
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[%s] heartbeat deinit begin.", __func__);
    clusterMonitorThreadFlag_ = false;
    linkThreadRunning_ = false;

    if (clusterMonitorThread_) {
        if (clusterMonitorThread_->joinable()) {
            clusterMonitorThread_->join();
        }
    }
    {
        std::unique_lock<std::mutex> lock(threadLock_);
        for (SocketHandle handler : pendingDestroySockets_) {
            if (handler == nullptr) {
                continue;
            }
            HcclResult ret = SocketDestroy(handler);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("[DeInit] pending SocketDestroy failed, ret[%d]", ret);
            }
        }
        pendingDestroySockets_.clear();

        for (auto iter = uid2SocketRefMap_.begin(); iter != uid2SocketRefMap_.end(); iter++) {
            HcclResult ret = SocketDestroy(iter->second.socketHandler);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("[DeInit] SocketDestroy failed, ret[%d]", ret);
            }
        }
        uid2SocketRefMap_.clear();
        uid2FrameStatusMap_.clear();
    }
    std::queue<ClusterMonitorFrame> empty;
    std::swap(errStatusQueue_, empty);

    initialized_ = false;
    HCCL_INFO("[%s] heartbeat deinit end.", __func__);
    return HCCL_SUCCESS;
}

void ClusterMonitor::ClearClusterLinkContext(const std::string& commId, std::set<ClusterUIDType>& remInQueue)
{
    std::unique_lock<std::mutex> linkCtxlock(clusertMonitorLinkMtx_);
    auto ctxIter = clusterLinkContext_.find(commId);
    if (ctxIter != clusterLinkContext_.end()) {
        while (!ctxIter->second.empty()) {
            remInQueue.insert(ctxIter->second.front().first); // uid出队存入set中
            ctxIter->second.pop();
        }
    }
    clusterLinkContext_.erase(commId);
}

bool ClusterMonitor::UnregisterCommIdFromMaps(const std::string& commId, const std::set<ClusterUIDType>& remInQueue)
{
    std::unique_lock<std::mutex> lock(threadLock_);

    for (const auto& rem : remInQueue) {
        if (monitorLinkStatusMap_[rem] == MonitorLinkStatus::MONITOR_LINK_BUILDING) {
            monitorLinkStatusMap_[rem] = MonitorLinkStatus::MONITOR_LINK_NOT_START;
            HCCL_INFO(
                "[%s] commId[%s] rem[%s] is in clusterLinkContext_ deque. Status change to not start", __func__,
                commId.c_str(), GetUID(rem).c_str());
        }
    }
    auto iter = commIdMap_.find(commId);
    if (iter == commIdMap_.end()) {
        HCCL_INFO("commId[%s] hasn't Registered, skip", commId.c_str());
        return false;
    }

    for (const auto& remRank : commIdMap_[commId]) {
        ClusterUIDType rem = remRank.first;
        uid2FrameStatusMap_.erase(rem);
        if (remRank.second) {
            if (uid2SocketRefMap_.count(rem) == 1) {
                // 不在此处 SocketDestroy；摘入 pending，等 DeInit join 后再销毁
                SocketHandle handler = uid2SocketRefMap_[rem].socketHandler;
                if (handler != nullptr) {
                    pendingDestroySockets_.push_back(handler);
                }
                monitorLinkStatusMap_[rem] = MonitorLinkStatus::MONITOR_LINK_NOT_START;
            }
            HCCL_INFO("[%s]commId[%s] socket erase remote:%s", __func__, commId.c_str(), GetUID(rem).c_str());
            uid2SocketRefMap_.erase(rem);
        }
        HCCL_INFO("[%s]commId[%s] status erase remote:%s", __func__, commId.c_str(), GetUID(rem).c_str());
    }
    commIdMap_.erase(iter);
    HCCL_INFO("[%s]commId[%s] UnregisterRanks Completed.", __func__, commId.c_str());
    return true;
}

HcclResult ClusterMonitor::UnRegisterToClusterMonitor(const hccl::CollComm* collComm)
{
    CHK_PRT_RET(initialized_ == false, HCCL_WARNING("Heartbeat has been destroyed, or not initialized"), HCCL_SUCCESS);
    const std::string& commId = collComm->GetCommId();
    std::set<ClusterUIDType> remInQueue;
    ClearClusterLinkContext(commId, remInQueue);
    if (!UnregisterCommIdFromMaps(commId, remInQueue)) {
        return HCCL_SUCCESS;
    }
    if (commIdMap_.size() == 0) {
        HCCL_RUN_INFO("[%s]Entry HeartBeat DeInit.", __func__);
        CHK_RET(DeInit());
    }
    return HCCL_SUCCESS;
}

void ClusterMonitor::ProcessExceptionEvent()
{
    while (errRankQueue_.size() > 0) {
        ClusterUIDType cur = errRankQueue_.front();
        uid2FrameStatusMap_[cur].needBroadcast = false;
        for (auto iterRem = uid2SocketRefMap_.begin(); iterRem != uid2SocketRefMap_.end(); iterRem++) {
            ClusterUIDType rem = iterRem->first;
            if (rem != uid2FrameStatusMap_[cur].informer
                && uid2FrameStatusMap_[rem].status == ClusterMonitorStatus::CLUSTER_MONITOR_OK) {
                (void)SendFrame(rem, cur, uid2FrameStatusMap_[cur].informer, uid2FrameStatusMap_[cur].status);
            }
        }
        errRankQueue_.pop();
    }
    return;
}

constexpr u32 BASE_YEAR = 1900;
void GetCqeErrInfoFromTaskException(
    unsigned int remoteLocalId, unsigned int locDeviceId, unsigned short int status, std::string localEid,
    std::string remoteEid, std::string remoteInsId)
{
    if (!Hccl::EnvConfig::GetInstance().GetLogConfig().GetDfsConfig().clusterHeartBeatEnable) {
        HCCL_RUN_INFO(
            "[%s] HCCL_DFS_CONFIG cluster_heartbeat is off. It's unnecessary to "
            "get cqe error info.",
            __func__);
        return;
    }
    return hccl::CollCommMgr::GetInstance()
        .GetClusterMonitor(locDeviceId)
        .GetCqeErrInfoFromTaskException(remoteLocalId, status, localEid, remoteEid, remoteInsId);
}

void ClusterMonitor::GetCqeErrInfoFromTaskException(
    u32 remoteLocalId, uint16_t status, std::string localEid, std::string remoteEid, std::string remoteInsId)
{
    cqeErrInfo_.cqeRemoteLocalId = remoteLocalId;
    cqeErrInfo_.cqeStatus = status;
    cqeErrInfo_.cqeLocalEid = localEid;
    cqeErrInfo_.cqeRemoteEid = remoteEid;
    cqeErrInfo_.cqeRemoteInsId = remoteInsId;
    ClusterUIDCxt remoteUIDcxt(remoteInsId, remoteLocalId);
    ClusterUIDType localUID = myRankUID_;
    ClusterUIDType remoteUID = FormatUID(remoteUIDcxt);
    SetStatus(localUID, remoteUID, ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR, true);
    time_t tmpt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    auto duration_us
        = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
    //  提取总微秒数
    auto total_us = duration_us.count();
    // 分离秒和微秒部分
    auto microseconds = total_us % 1000000;
    struct tm* now = localtime(&tmpt);
    char errorLinkLogBuffer[LOG_TMPBUF_SIZE];

    s32 stringRet = snprintf_s(
        errorLinkLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "localInfo{local instanceId[%s], LocalId[%u], localEid[%s]}, remoteInfo{remote instanceId[%s], "
        "remoteLocalId[%u], remoteEid[%s]}",
        myRankNetInstId_.c_str(), myRankLocalId_, cqeErrInfo_.cqeLocalEid.c_str(), cqeErrInfo_.cqeRemoteInsId.c_str(),
        cqeErrInfo_.cqeRemoteLocalId, cqeErrInfo_.cqeRemoteEid.c_str());
    CHK_PRT_CONT(
        stringRet < 0,
        HCCL_ERROR("[ClusterMonitor][GetCqeErrInfoFromTaskException]snprintf error when log cqe error info"));

    if (now == nullptr) {
        HCCL_ERROR(
            "[%s][%s][%s]localtime fail, cqe error status[%u], %s", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_HEARTBEAT_EVENT.c_str(), LOG_KEYWORDS_CQE_ERROR.c_str(), cqeErrInfo_.cqeStatus,
            errorLinkLogBuffer);
    } else {
        HCCL_ERROR(
            "[%s][%s][%s]cqe error status[%u], time:[%04d-%02d-%02d %02d:%02d:%02d.%06lld], %s",
            LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_HEARTBEAT_EVENT.c_str(), LOG_KEYWORDS_CQE_ERROR.c_str(),
            cqeErrInfo_.cqeStatus, now->tm_year + BASE_YEAR, now->tm_mon + 1, now->tm_mday, now->tm_hour, now->tm_min,
            now->tm_sec, microseconds, errorLinkLogBuffer);
    }
    return;
}

void ClusterMonitor::MakeErrMsg(
    std::queue<ClusterMonitorFrame>& keyEvents, std::vector<std::string>& errStatusVec) const
{
    while (keyEvents.size() > 0) {
        auto& tmp = keyEvents.front();
        std::string crimerStr = GetUID(tmp.crimer);
        std::string informerStr = GetUID(tmp.informer);

        std::string headStr = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + LOG_KEYWORDS_HEARTBEAT_EVENT + "]"
                              + "Cluster Exception Location[IP/ID]:[";

        time_t tm = std::chrono::system_clock::to_time_t(tmp.TOASystem);
        std::string timeStr(ctime(&tm));
        if (!timeStr.empty()) { // ctime()函数自带换行符，需要去掉
            timeStr.pop_back();
        }
        timeStr = ", Arrival Time:[" + timeStr + "]";

        std::string errStr = ", ExceptionType:";
        std::string reasonStr = ", Possible Reason:";
        switch (tmp.status) {
            case ClusterMonitorStatus::CLUSTER_MONITOR_LOST:
                errStr = errStr + "[Heartbeat Lost Occurred]";
                reasonStr = reasonStr + "1. Process has exited, 2. Network Disconnected";
                errStr
                    = headStr + crimerStr + "]" + timeStr + ", Discoverer:[" + informerStr + "]" + errStr + reasonStr;
                break;
            case ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR:
                errStr = errStr + "[Error cqe Occurred]";
                reasonStr = reasonStr + "1.Network Disconnected, 2.Remote Rank Coredown";
                errStr = headStr + crimerStr + "]" + timeStr + errStr + reasonStr;
                break;
            default:
                errStr = " Unknown";
        }
        errStatusVec.emplace_back(errStr);
        keyEvents.pop();
    }
}

std::vector<std::string>
ClusterMonitor::PrintEvents(std::map<ClusterMonitorStatus, std::queue<ClusterMonitorFrame>>& keyEvents) const
{
    std::vector<std::string> errStatusVec;
    // 打印优先级 opretry not support > error cqe > stuck > lost
    MakeErrMsg(keyEvents[ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR], errStatusVec);
    MakeErrMsg(keyEvents[ClusterMonitorStatus::CLUSTER_MONITOR_LOST], errStatusVec);
    return errStatusVec;
}

std::vector<std::string> ClusterMonitor::GetErrStatusVecFromCluserMonitor()
{
    std::unique_lock<std::mutex> lock(threadLock_);
    std::map<ClusterMonitorStatus, std::queue<ClusterMonitorFrame>> keyEvents;
    while (errStatusQueue_.size() > 0) {
        auto& tmp = errStatusQueue_.front();
        keyEvents[tmp.status].push(tmp);
        errStatusQueue_.pop();
    }
    return PrintEvents(keyEvents);
}

std::vector<std::string> GetErrStatusVecFromCluserMonitor(s32 deviceLogicID)
{
    return hccl::CollCommMgr::GetInstance().GetClusterMonitor(deviceLogicID).GetErrStatusVecFromCluserMonitor();
}

__attribute__((constructor)) void ClusterMonitorCallBackInit()
{
    hcomm::RegisterGetAicpuCqeErrInfoCallBackHcomm(GetCqeErrInfoFromTaskException);
    hcomm::RegisterGetCcuCqeErrInfoCallBackHcomm(GetCqeErrInfoFromTaskException);
    hcomm::RegisterAicpuGetErrStatusVecCallBack(GetErrStatusVecFromCluserMonitor);
    hcomm::RegisterCcuGetErrStatusVecCallBack(GetErrStatusVecFromCluserMonitor);
}

} // namespace hcomm
