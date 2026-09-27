/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_EXCHANGE_SERVER_H
#define TOPOINFO_EXCHANGE_SERVER_H

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include "topoinfo_struct.h"
#include "topoinfo_exchange_base.h"
#include "comm.h"
#include "hccl_socket.h"
#include "hccl_network_pub.h"

namespace hccl {
// 多root（scalable）建链时，root间全互联mesh所需的建链信息。
// 由 TopoInfoDetectScalable 组装，在启动 server 线程前设置到 TopoInfoExchangeServer。
struct ScalableServerInfo {
    u32 nRoot{0};                                 // root总个数
    u32 rootIndex{INVALID_UINT};                  // 本root的index
    u32 groupSize{0};                             // 本组rank个数（含root自身）
    s32 deviceLogicId{INVALID_INT};               // 本root所在device，mesh accept线程建device上下文用
    std::vector<RootMeshInfo> meshInfos;          // 所有root的ip+meshPort，下标=rootIndex
    std::shared_ptr<HcclSocket> meshListenSocket; // root间mesh全互联监听socket
};

class TopoInfoExchangeServer : public TopoInfoExchangeBase {
public:
    explicit TopoInfoExchangeServer(
        HcclIpAddress& hostIP, u32 hostPort, const std::vector<HcclIpAddress> whitelist, HcclNetDevCtx netDevCtx,
        std::shared_ptr<HcclSocket> listenSocket, const std::string& identifier);
    explicit TopoInfoExchangeServer(
        HcclIpAddress& hostIP, u32 hostPort, const std::vector<HcclIpAddress> whitelist, HcclNetDevCtx netDevCtx,
        std::shared_ptr<HcclSocket> listenSocket, std::shared_ptr<HcclSocket> grpLeaderToRoot,
        const std::string& identifier);
    ~TopoInfoExchangeServer() override;
    HcclResult Setup();
    HcclResult SetupScalable();
    HcclResult SetupGroupLeader();
    HcclResult SetupByMasterInfo();
    HcclResult Teardown();
    HcclResult GetConnections(std::map<u32, std::shared_ptr<HcclSocket>>& connectSockets);
    HcclResult SetScalableInfo(const ScalableServerInfo& info);

private:
    HcclResult SetupScalableCore();
    // 通过dispatcher把ranktable广播给各rank，并更新/唤醒广播阶段状态
    HcclResult BroadcastRankTableAndStatus(
        const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, const RankTable_t& rankTable,
        const std::string& failedAgentIdList);
    // accept循环每轮的状态：可accept / 剩余不足1s跳过本轮 / 整体超时
    enum class SocketAcceptStatus {
        ACCEPT_WAIT,
        ACCEPT_SKIP,
        ACCEPT_TIMEOUT,
    };
    HcclResult Connect(std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, u32& rankSize);
    HcclResult ScalableConnect(std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, u32& rankSize);
    HcclResult AcceptLoop(
        std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, u32& rankSize, u32 expectSocketNumInit,
        bool isScalable);
    HcclResult AcceptSocket(
        std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, u32& rankSize, u32 waitTime,
        u32& expectSocketNum, u32& previousRankNum, bool& isFirstAcceptTimeOut, bool isScalable);
    SocketAcceptStatus GetSocketAcceptStatus(
        const std::chrono::steady_clock::time_point& startTime, const std::chrono::seconds& timeout, u32& waitTime);
    void LogAcceptTimeout(bool isScalable) const;
    // root间全互联：收集所有root的组内子ranktable并合并出全局ranktable
    HcclResult RootMeshAllGatherAndMerge(const RankTable_t& groupRankTable);
    HcclResult MeshConnectToLargerRoots();
    HcclResult MeshAcceptWorker();
    HcclResult MeshAcceptWorkerCore();
    HcclResult MeshSendPartial(std::shared_ptr<HcclSocket> socket, const RankTable_t& partial);
    // Connect()仅发起异步连接，fdHandle在连接建立后才可用；发送前需轮询等待连接建立
    HcclResult WaitMeshConnectionEstablished(const std::shared_ptr<HcclSocket>& socket) const;
    // 构建root间mesh连接tag：各root的identifier互不相同不能用于tag，
    // 改用有序对<较小root, 较大root> +
    // 较大root的meshPort（双方均可从meshInfos_算出一致的值），保证连接/接受双方tag一致且pair唯一
    std::string BuildMeshTag(u32 smallerRoot, u32 largerRoot) const;
    HcclResult MergeRankTables(RankTable_t& mergedTable);
    HcclResult GroupLeaderConnect(std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets);
    HcclResult GetConnection(std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets);
    HcclResult Disconnect(std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets);
    HcclResult DeleteSocketWhiteList(u32 port, const std::vector<HcclIpAddress>& whitelist);
    HcclResult StopNetwork(const std::vector<HcclIpAddress>& whitelist, u32 hostPort);
    HcclResult StopSocketListen(const std::vector<HcclIpAddress>& whitelist, u32 hostPort);
    HcclResult RecvGroupLeaderInfo(
        const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, GroupLeader_t& groupLeader);
    HcclResult RecvGroupLeaderPortInfo(
        const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, GroupLeader_t& groupLeader);
    HcclResult
    GetRanksBasicInfo(const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, RankTable_t& rankTable);
    HcclResult
    GetRanksTransInfo(const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, RankTable_t& rankTable);
    HcclResult GetRankBasicInfo(std::shared_ptr<HcclSocket> socket, RankTable_t& rankTable);
    HcclResult GetCommonTopoInfo(RankTable_t& rankTable, const RankTable_t& orginRankTable) const;
    HcclResult SortRankList(RankTable_t& rankTable) const;
    HcclResult RecvRemoteAgentID(std::shared_ptr<HcclSocket> socket, std::string& agentID);
    HcclResult RecvRemoteRankNum(std::shared_ptr<HcclSocket> socket, u32& remoteRankNum);
    HcclResult HierarchicalSendRecv();
    HcclResult VerifyRemoteRankNum(u32& previousRankNum, u32 remoteRankNum) const;
    HcclResult SendIdentify(std::shared_ptr<HcclSocket> socket, u32 identify) const;
    HcclResult
    DisplayConnectedRank(const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets, u32 rankNum = 0);
    HcclResult DisplayConnectingStatus(
        u32 totalSockets, u32 waitSockets,
        const std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets) const;
    bool DoServerIdExist(const RankTable_t& rankTable, const std::string& serverId) const;
    HcclResult GetRemoteFdAndRankSize(
        std::shared_ptr<HcclSocket>& socket, std::map<std::string, std::shared_ptr<HcclSocket>>& connectSockets,
        u32& rankSize);
    HcclResult FailedConnectionAgentIdString(u32 rankSize, std::string& failedAgentIdList);
    HcclIpAddress hostIP_;
    u32 hostPort_{HCCL_INVALID_PORT};
    SocketHandle socketHandle_;
    std::vector<HcclIpAddress> whitelist_;
    HcclNetDevCtx netDevCtx_{nullptr};
    std::shared_ptr<HcclSocket> listenSocket_;
    std::shared_ptr<HcclSocket> grpLeaderToRoot_;
    friend class TopoInfoExchangeDispatcher;
    std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets_;
    std::map<std::string, std::shared_ptr<HcclSocket>> grpLeaderSockets_;
    std::map<u32, std::shared_ptr<HcclSocket>> connectSocketsWithRankID_;
    std::mutex lock_;
    std::string identifier_;
    RankTable_t rankTable_;
    u32 expectSocketNum_ = 1;
    u32 previousRankNum_ = 0;
    // 多root（scalable）mesh全互联相关
    u32 nRoot_{0};
    u32 rootIndex_{INVALID_UINT};
    u32 groupSize_{0};
    s32 deviceLogicId_{INVALID_INT}; // 本root所在device，mesh accept线程建device上下文用
    std::vector<RootMeshInfo> meshInfos_;
    std::shared_ptr<HcclSocket> meshListenSocket_{nullptr};
    std::vector<RankTable_t> partials_; // partials_[rootIndex]=组内子表，其余为其他root的子表
    std::atomic<u32> meshRecvCount_{0}; // 已收到的其他root的partial个数
};
} // namespace hccl

#endif /* TOPOINFO_EXCHANGE_SERVER_H */
