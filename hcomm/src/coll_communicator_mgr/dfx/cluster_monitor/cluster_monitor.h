/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CLUSTER_MONITOR_H
#define CLUSTER_MONITOR_H
#include <atomic>
#include <thread>
#include <map>
#include <deque>
#include <mutex>
#include "hcclCommDfx.h"
#include "ring_buffer.h"
#include "coll_comm.h"
#include "reference_map.h"
#include "log.h"
#include "hccl/hccl_types.h"
#include "hccl_common.h"
#include "hccl_comm_socket_c_adpt.h"
#include "hccl_communicator.h"
#include "../../common/loggers/comm_addr_logger.h"

namespace hcomm {
using ClusterUIDType = struct HcclClusterMonitorUID {
    char id[2048] = {0}; // netInstanceId + localId 最大不超过2048字节
    bool operator==(const HcclClusterMonitorUID& that) const { return std::string(this->id) == std::string(that.id); }
    bool operator!=(const HcclClusterMonitorUID& that) const { return std::string(this->id) != std::string(that.id); }
    bool operator<(const HcclClusterMonitorUID& that) const { return std::string(this->id) < std::string(that.id); }
};
} // namespace hcomm

namespace std {
template <>
class hash<hcomm::HcclClusterMonitorUID> {
public:
    size_t operator()(const hcomm::HcclClusterMonitorUID& uid) const { return hash<string>()(string(uid.id)); }
};
} // namespace std

namespace hcomm {
enum class ClusterMonitorStatus {
    CLUSTER_MONITOR_OK,
    CLUSTER_MONITOR_LOST,
    CLUSTER_MONITOR_NOTIFY,
    CLUSTER_MONITOR_CQE_ERR,
    CLUSTER_MONITOR_OPRETRY_NOT_SUPPORT,
    CLUSTER_MONITOR_STUCK,
    CLUSTER_MONITOR_INCONSISTENT
};

struct ErrorCqeInfo {
    u32 cqeLocalId = 0;
    u32 cqeRemoteLocalId = 0;
    uint16_t cqeStatus = 0;
    std::string cqeLocalEid = "";
    std::string cqeRemoteEid = "";
    std::string cqeRemoteInsId = "";
    std::string cqeLocalInsId = "";
};
const std::map<ClusterMonitorStatus, std::string> CLUSTER_MONITOR_STATUS_STR_MAP{
    {ClusterMonitorStatus::CLUSTER_MONITOR_OK, "OK"},
    {ClusterMonitorStatus::CLUSTER_MONITOR_LOST, "LOST"},
    {ClusterMonitorStatus::CLUSTER_MONITOR_NOTIFY, "NOTIFY"},
    {ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR, "CQE ERROR"}};

inline std::string GetClusterMonitorStatusStr(ClusterMonitorStatus status)
{
    auto iter = CLUSTER_MONITOR_STATUS_STR_MAP.find(status);
    if (iter == CLUSTER_MONITOR_STATUS_STR_MAP.end()) {
        return "Unknown";
    } else {
        return iter->second;
    }
}

struct ClusterMonitorFrame {
    ClusterUIDType src{};      // 心跳建链的本端
    ClusterUIDType dst{};      // 心跳建链的远端
    ClusterUIDType crimer{};   // 异常的节点
    ClusterUIDType informer{}; // 把异常传输给自己的节点
    ClusterMonitorStatus status = ClusterMonitorStatus::CLUSTER_MONITOR_OK;
    HcclUs TOARelative{};       // time of arrival (Relative)
    HcclSystemTime TOASystem{}; // time of arrival (System)
    char reserved[256] = {0};   // 预留256个字段，后续扩展可存储其他信息
    ClusterMonitorFrame() {}
    ClusterMonitorFrame(
        ClusterUIDType& crimer, ClusterUIDType& informer, ClusterMonitorStatus status, HcclUs TOARelativeIn,
        HcclSystemTime TOASystemIn)
        : crimer(crimer),
          informer(informer),
          status(status),
          TOARelative(TOARelativeIn),
          TOASystem(TOASystemIn)
    {}
    ClusterMonitorFrame(
        ClusterUIDType& src, ClusterUIDType& dst, ClusterUIDType& crimer, ClusterUIDType& informer,
        ClusterMonitorStatus status)
        : src(src),
          dst(dst),
          crimer(crimer),
          informer(informer),
          status(status)
    {}
};

struct ClusterMonitorSocketCtx {                // 原ConnInfo
    SocketDesc socketDesc;                      // 与对端连接的描述符
    SocketHandle socketHandler;                 // 引用头文件定义
    std::queue<ClusterMonitorFrame> sendBuffer; // 用来发送的帧队列
    u32 restSize = 0;                           // 剩余待发送的帧长度
    hccl::RingBuffer recvBuffer;                // 用来接收的环形帧队列
    u32 lostNum = 0;                            // 丢失的心跳个数
    bool newConn = false;                       // 是否是新增的连接
    ClusterMonitorSocketCtx() {}
    ClusterMonitorSocketCtx(SocketDesc& socketDesc, bool newConn)
        : socketDesc(socketDesc),
          socketHandler(nullptr),
          newConn(newConn)
    {}

    void PrintSocketDesc(std::string tag) const
    {
        std::string localAddr = hcomm::logger::CommAddrLogger::ToString(socketDesc.localEndpoint.commAddr);
        std::string remoteAddr = hcomm::logger::CommAddrLogger::ToString(socketDesc.remoteEndpoint.commAddr);
        HCCL_DEBUG(
            "[%s] socketDesc: localEndpoint: {commAddr: %s, EndpointLocType: %d}, "
            "remoteEndpoint: {commAddr: %s, EndpointLocType: %d}, tag: %s, role: %d, listenPort: %u",
            tag.c_str(), localAddr.c_str(), socketDesc.localEndpoint.loc.locType, remoteAddr.c_str(),
            socketDesc.remoteEndpoint.loc.locType, socketDesc.tag, socketDesc.role, socketDesc.listenPort);
    }
};

struct UIDContext {
    ClusterUIDType uid;
    uint32_t netLayer{0};
    uint32_t rankId{0};
    uint32_t localId{0};     // 用来netLayer=0的时候排序使用
    std::string netInstId{}; // 用来netLayer>1的时候排序使用
    UIDContext() {}
    UIDContext(ClusterUIDType& uid, uint32_t netLayer, uint32_t rankId, uint32_t localId, std::string netInstId)
        : uid(uid),
          netLayer(netLayer),
          rankId(rankId),
          localId(localId),
          netInstId(netInstId)
    {}
};

struct ClusterUIDCxt {
    std::string netInstId;
    uint32_t localId;
    ClusterUIDCxt() {}
    ClusterUIDCxt(std::string& netInstId, uint32_t localId) : netInstId(netInstId), localId(localId) {}
};

class ClusterMonitor {
public:
    HcclResult RegisterToClusterMonitor(HcclComm comm);
    HcclResult UnRegisterToClusterMonitor(const hccl::CollComm* collComm);
    ClusterUIDType FormatUID(ClusterUIDCxt cxt) const;
    std::string GetUID(const ClusterUIDType& uid) const;
    std::string FormatConnTag(HcommSocketRole role, std::pair<ClusterUIDType, ClusterUIDType> uidPair) const;
    HcclResult InsertClusterMonitorCtx(
        HcclComm comm, UIDContext remoteCtx, std::map<ClusterUIDType, ClusterMonitorSocketCtx>& needConnectRank);
    HcclResult GetSocketDescFromRankInfo(
        HcclComm comm, uint32_t remoteRank, uint32_t netLayer, const ClusterUIDType& remoteUID, SocketDesc& socketDesc);
    HcclResult GetSamePlaneRank(
        HcclComm comm, std::vector<UIDContext> singlePlaneCtx,
        std::map<ClusterUIDType, ClusterMonitorSocketCtx>& needConnectRank);
    HcclResult GetConnectRank(
        HcclComm comm, std::map<ClusterUIDType, ClusterMonitorSocketCtx>& needConnectRank,
        std::map<uint32_t, std::vector<UIDContext>> uidCtxs, std::vector<uint32_t>& netLayersVector);
    void CreateHBLinksAsync();
    void
    SetStatus(ClusterUIDType& crimer, ClusterUIDType& informer, ClusterMonitorStatus status, bool needBroadcast = true);
    void MonitorThread();
    HcclResult RunMonitorThread();
    HcclResult
    SendFrame(ClusterUIDType& dst, ClusterUIDType& crimer, ClusterUIDType& informer, ClusterMonitorStatus status);
    void DelErrorSocket();
    void ProcessExceptionEvent();
    HcclResult RecvFrame(ClusterUIDType rem);
    HcclResult ParseFrame(ClusterMonitorFrame& cmFrame, ClusterUIDType& src);
    HcclResult DeInit();
    void GetCqeErrInfoFromTaskException(
        u32 remoteLocalId, uint16_t status, std::string localEid, std::string remoteEid, std::string remoteInsId);
    std::vector<std::string> GetErrStatusVecFromCluserMonitor();
    std::vector<std::string>
    PrintEvents(std::map<ClusterMonitorStatus, std::queue<ClusterMonitorFrame>>& keyEvents) const;
    void MakeErrMsg(std::queue<ClusterMonitorFrame>& keyEvents, std::vector<std::string>& errStatusVec) const;
    ClusterMonitor() = default;
    ~ClusterMonitor();

private:
    HcclResult GetRemEndpointDescs(
        HcclComm comm, std::map<uint32_t, std::vector<UIDContext>>& uidCtxs, std::vector<uint32_t>& netLayersVector);
    void GetRemEndpointDescsPerLayer(
        uint32_t netLayer, HcclComm comm, const Hccl::RankGraph* rankGraph, const hccl::CollComm* collComm,
        std::map<uint32_t, std::vector<UIDContext>>& uidCtxs, std::set<uint32_t>& rankIdsSet);

    HcclResult
    ProcessConnectRanks(const std::string& commId, std::map<ClusterUIDType, ClusterMonitorSocketCtx>& needConnectRank);
    void ClearClusterLinkContext(const std::string& commId, std::set<ClusterUIDType>& remInQueue);
    bool UnregisterCommIdFromMaps(const std::string& commId, const std::set<ClusterUIDType>& remInQueue);
    HcclResult CreateTransportHandle(ClusterMonitorSocketCtx& info) const;
    HcclResult OnConnectionEstablished(
        const std::string& commId, const ClusterUIDType& rem, ClusterMonitorSocketCtx& needConnectRank);
    HcclResult SendFrameFromBuffer(ClusterUIDType& dst, ClusterMonitorFrame& cmFrame);

    void CreateLinkWithRemotePonit(std::string commId, ClusterUIDType rem, ClusterMonitorSocketCtx needConnectRank);

    struct FrameStatus { // 专门用来给frame设置对应的状态
        ClusterMonitorStatus status = ClusterMonitorStatus::CLUSTER_MONITOR_OK;
        ClusterUIDType informer;
        bool needBroadcast = false;
        FrameStatus() {}
    };

    enum class MonitorLinkStatus {
        MONITOR_LINK_NOT_START,
        MONITOR_LINK_BUILDING,
        MONITOR_LINK_COMPLETED,
    };

    uint32_t myRankLocalId_;
    std::string myRankNetInstId_;
    ClusterUIDType myRankUID_;
    s32 deviceLogicId_{0};

    bool clusterMonitorThreadFlag_ = false;
    std::unique_ptr<std::thread> clusterMonitorThread_;
    // 防止重复初始化
    bool initialized_ = false;
    uint32_t lostThreshold_ = 0;
    std::atomic<bool> isDeInit_{false};
    std::atomic<bool> linkThreadRunning_{false};

    // 防止多线程同时初始化的线程锁
    std::mutex threadLock_;
    std::vector<ClusterUIDType> errorSocket_;
    std::queue<ClusterMonitorFrame> errStatusQueue_;

    // 通信域名称为key，ClusterUIDType表示1个节点,
    // bool表示0或1，是否连接上，用来指示是否有过这个通信域以及对应通信域里待连接的节点是否以及连接上，原groupMap_
    std::map<std::string, std::map<ClusterUIDType, bool>> commIdMap_;

    // 通信域名称为key, 一个通信域有多个待连接心跳的connInfo，存储到该结构体，待monitor线程轮询拿到, 原hbLinkConnInfo_
    std::map<std::string, std::queue<std::pair<ClusterUIDType, ClusterMonitorSocketCtx>>> clusterLinkContext_{};
    std::mutex clusertMonitorLinkMtx_; // 用来锁住clusterLinkContext_，原clusterLinkContext_

    // 存储UID与监控连接状态的map, NOT_START/BUILDING/COMPLETED，原rankId2LinkStatusMap_
    std::map<ClusterUIDType, MonitorLinkStatus> monitorLinkStatusMap_;

    // 存储UID与连接上下文的计数map，由于多个通信域都有可能使用同一个context去连接远端，需要计数处理，解注册时计数--，原rankId2SocketMap_
    hccl::ReferenceMap<ClusterUIDType, ClusterMonitorSocketCtx> uid2SocketRefMap_;

    // 用来做帧的统计计数，设置对应帧的状态，原rankId2StatusMap_
    hccl::ReferenceMap<ClusterUIDType, FrameStatus> uid2FrameStatusMap_;

    // uid与thread的维护关系，不同的remote起不同的异步建链线程，原linkThreadMap_
    std::map<ClusterUIDType, std::unique_ptr<std::thread>> linkThreadMap_{};

    // UnRegister 摘下、延后到 DeInit(join 之后) 再 SocketDestroy 的句柄
    std::vector<SocketHandle> pendingDestroySockets_;

    // 保存错误的节点
    std::queue<ClusterUIDType> errRankQueue_;

    ErrorCqeInfo cqeErrInfo_;
};
} // namespace hcomm
#endif // CLUSTER_MONITOR_H
