/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_HEARTBEAT_H
#define HCCL_HEARTBEAT_H

#include <thread>
#include <map>
#include <deque>
#include <mutex>

#include "hccl/hccl_types.h"
#include "log.h"
#include "reference_map.h"
#include "ring_buffer.h"
#include "common.h"
#include "sal_pub.h"
#include "hccl_socket_manager.h"
#include "transport_pub.h"
#include "topoinfo_struct.h"
#include "comm_config_pub.h"
namespace hccl {
using RankId = u32;
constexpr u32 BROADCAST_INTERVAL = 50;            // 背景线程执行周期为50 ms
constexpr u32 BROADCAST_INTERVAL_WITH_CHECK = 25; // 背景线程执行周期为25 ms
constexpr u32 STUCK_INTERVAL = 300000;            // 5min监控一次,默认 300000 ms
constexpr u32 STUCK_COUNT = STUCK_INTERVAL / BROADCAST_INTERVAL;
constexpr u32 OPINFO_SEND_NUM_BY_TAG = 500;                              // 一次心跳帧发送的算子信息个数
constexpr u32 OPINFO_TAG_QUEUE_NUM = 10;                                 // 一次心跳帧发送的算子信息个数
constexpr u32 HEARTBEAT_INTERVAL = 1000;                                 // 心跳帧发送周期为1000 ms
constexpr u32 HEARTBEAT_COUNT = HEARTBEAT_INTERVAL / BROADCAST_INTERVAL; // 心跳帧发送间隔数
constexpr u32 BASE_NUMBER = 2;
constexpr u32 RETRY_CQE_ARRAY_SIZE = 128; // 重执行时获取的CQE数组的最大数量，最大128
constexpr u32 JITTER_TIME = 300; // 关键事件允许的误差事件范围±300s。误差来源：EVENT和NOTIFY差异、传播耗时、计时误差
constexpr u32 EVENT_MAX_CNT = 5000;           // 防止内存泄漏，同时不能太短，防止正常事件被冲掉
constexpr u32 THROUND_MILS = 1000;            // 1000ms
constexpr u32 OPINFO_QUEUE_MAX_SIZE = 131072; // 算子下发校验队列最大算子个数，防止内存占用
constexpr u32 MAX_SENDBUFF_SIZE = 3072;       // SendBuff[dst] 最大数量
constexpr u32 SR_TAG_MAP_MAX_NUM = 65536;
constexpr u32 HBFRAME_SEND_LOOP_MAX_NUM = 120;
constexpr s32 HCCL_STUCK_DETECT_TIME_MIN = 60; // 卡住检测最短时间
constexpr s32 HCCL_STUCK_DETECT_TIME_BASE = 3; // 卡住检测时间为execTime/3
constexpr s32 HCCL_LOST_THRESHOLD = 30;        // 心跳丢失阈值为30s

using UIDType = struct HcclHeartBeatUid {
    char id[512] = {0}; // ip[IP_ADDRESS_BUFFER_LEN] + ifname[MAX_INTERFACE_NAME_LEN] + devid 最大不超过512字节
    bool operator==(const HcclHeartBeatUid& that) const { return std::string(this->id) == std::string(that.id); }
    bool operator!=(const HcclHeartBeatUid& that) const { return std::string(this->id) != std::string(that.id); }
    bool operator<(const HcclHeartBeatUid& that) const { return std::string(this->id) < std::string(that.id); }
};
} // namespace hccl

namespace std {
template <>
class hash<hccl::HcclHeartBeatUid> {
public:
    size_t operator()(const hccl::HcclHeartBeatUid& uid) const { return hash<string>()(string(uid.id)); }
};
} // namespace std

namespace hccl {
constexpr u8 HAS_CONN = 1;
constexpr u8 NO_CONN = 0;
constexpr u32 TIME_FROM_1900 = 1900;

enum class HeartBeatStatus {
    HEARTBEAT_OK,
    HEARTBEAT_LOST,
    HEARTBEAT_NOTIFY,
    HEARTBEAT_CQE_ERR,
    HEARTBEAT_OPRETRY_NOT_SUPPORT,
    HEARTBEAT_STUCK,
    HEARTBEAT_INCONSISTENT
};
const std::map<HeartBeatStatus, std::string> HEARTBEAT_STATUS_STR_MAP{
    {HeartBeatStatus::HEARTBEAT_OK, "OK"},
    {HeartBeatStatus::HEARTBEAT_LOST, "LOST"},
    {HeartBeatStatus::HEARTBEAT_NOTIFY, "NOTIFY"},
    {HeartBeatStatus::HEARTBEAT_CQE_ERR, "ERROR CQE"},
    {HeartBeatStatus::HEARTBEAT_OPRETRY_NOT_SUPPORT, "OPRETRY NOT SUPPORT"},
    {HeartBeatStatus::HEARTBEAT_STUCK, "STUCK"},
    {HeartBeatStatus::HEARTBEAT_INCONSISTENT, "INCONSISTENT"}};
inline std::string GetHeartBeatStatusStr(HeartBeatStatus status)
{
    auto iter = HEARTBEAT_STATUS_STR_MAP.find(status);
    if (iter == HEARTBEAT_STATUS_STR_MAP.end()) {
        return "Unknown";
    } else {
        return iter->second;
    }
}

struct CounterStat {
    std::pair<int32_t, int32_t> oldCounter{0, 0};
    std::pair<int32_t, int32_t> newCounter{0, 0};
    std::uint64_t issueCnt = 0;
    bool isNeedDetect = false;
    bool isFirst = true;
    std::uint64_t couterPrintInter = STUCK_COUNT;
    CounterStat() {};
};

enum class InconsistentType {
    NO_INCONSISTENT = 0,
    OPTYPE_INCONSISTENT = 1,
    DATATYPE_INCONSISTENT = 2,
    REDUCETYPE_INCONSISTENT = 3,
    ROOT_INCONSISTENT = 4,
    COUNT_INCONSISTENT = 5
};

const std::map<InconsistentType, std::string> OP_INCONSISTENT_STR_MAP{
    {InconsistentType::NO_INCONSISTENT, "no exist op inconsistent"},
    {InconsistentType::OPTYPE_INCONSISTENT, "op type inconsistent"},
    {InconsistentType::DATATYPE_INCONSISTENT, "op data type inconsistent"},
    {InconsistentType::REDUCETYPE_INCONSISTENT, "op reduce type inconsistent"},
    {InconsistentType::ROOT_INCONSISTENT, "op root inconsistent"},
    {InconsistentType::COUNT_INCONSISTENT, "op count inconsistent"}};

inline std::string GetInconsistentTypeStr(InconsistentType status)
{
    auto iter = OP_INCONSISTENT_STR_MAP.find(status);
    if (iter == OP_INCONSISTENT_STR_MAP.end()) {
        return "Unknown";
    } else {
        return iter->second;
    }
};

struct OpInfoDesc {
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_RESERVED;
    uint32_t root = 0;
    uint64_t count = 0;
    uint64_t index = 0;
    bool isValid = false;
};

struct OpInconsistentInfo {
    InconsistentType inconsistentType;
    std::string localInfo;
    std::string remoteInfo;
    OpInfoDesc opInfoDesc;
    OpInconsistentInfo(
        InconsistentType inconsistentType, const std::string& localInfo, const std::string& remoteInfo,
        const OpInfoDesc& opInfoDesc)
        : inconsistentType(inconsistentType),
          localInfo(localInfo),
          remoteInfo(remoteInfo),
          opInfoDesc(opInfoDesc)
    {}
};

struct OpInfoTagQueue {
    OpInfoDesc opInfoList[OPINFO_SEND_NUM_BY_TAG];
    char identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    u32 opInfoNum = 0;
};

struct OpInfoTagQueueFrame {
    OpInfoTagQueue opInfoTagQueue[OPINFO_TAG_QUEUE_NUM];
};

struct HeartBeatFrame {
    UIDType src;
    UIDType dst;
    UIDType crimer;
    UIDType informer;
    HeartBeatStatus status = HeartBeatStatus::HEARTBEAT_OK;
    HcclUs TOARelative;       // time of arrival (Relative)
    HcclSystemTime TOASystem; // time of arrival (System)
    HeartBeatFrame() {}
    HeartBeatFrame(
        UIDType& crimer, UIDType& informer, HeartBeatStatus status, HcclUs TOARelativeIn, HcclSystemTime TOASystemIn)
        : crimer(crimer),
          informer(informer),
          status(status),
          TOARelative(TOARelativeIn),
          TOASystem(TOASystemIn)
    {}
    HeartBeatFrame(UIDType& src, UIDType& dst, UIDType& crimer, UIDType& informer, HeartBeatStatus status)
        : src(src),
          dst(dst),
          crimer(crimer),
          informer(informer),
          status(status)
    {}
};

struct HeartBeatFrameWithOpCheck {
    UIDType src;
    UIDType dst;
    UIDType crimer;
    UIDType informer;
    HeartBeatStatus status = HeartBeatStatus::HEARTBEAT_OK;
    HcclUs TOARelative;       // time of arrival (Relative)
    HcclSystemTime TOASystem; // time of arrival (System)
    OpInfoTagQueueFrame opInfoTagQueueFrame;
    HeartBeatFrameWithOpCheck() {}
    HeartBeatFrameWithOpCheck(
        UIDType& crimer, UIDType& informer, HeartBeatStatus status, HcclUs TOARelativeIn, HcclSystemTime TOASystemIn)
        : crimer(crimer),
          informer(informer),
          status(status),
          TOARelative(TOARelativeIn),
          TOASystem(TOASystemIn)
    {}
    HeartBeatFrameWithOpCheck(UIDType& src, UIDType& dst, UIDType& crimer, UIDType& informer, HeartBeatStatus status)
        : src(src),
          dst(dst),
          crimer(crimer),
          informer(informer),
          status(status)
    {}
};

struct ConnInfo {
    std::shared_ptr<HcclSocket> socket = nullptr;
    std::queue<HeartBeatFrame> sendBuffer;
    std::queue<HeartBeatFrameWithOpCheck> sendBufferWithOpCheck;
    u32 restSize = 0;
    RingBuffer recvBuffer;
    u32 lostNum = 0;
    u32 lostReportCnt = 0;
    bool newConn = false;
    std::vector<SocketWlistInfo> wlistInfosVec;
    ConnInfo() {}
    ConnInfo(bool newConn, std::shared_ptr<HcclSocket>& socket) : socket(socket), newConn(newConn) {}
};

struct LinkInfo {
    std::string identifier;
    RankId localRank;
    std::string localServerId;
    s32 localDevicePhyId;
    RankId remoteRank;
    std::string remoteServerId;
    s32 remoteDevicePhyId;
    LinkInfo() {}
    LinkInfo(
        std::string& identifier, RankId localRank, std::string& localServerId, s32 localDevicePhyId, RankId remoteRank,
        std::string& remoteServerId, s32 remoteDevicePhyId)
        : identifier(identifier),
          localRank(localRank),
          localServerId(localServerId),
          localDevicePhyId(localDevicePhyId),
          remoteRank(remoteRank),
          remoteServerId(remoteServerId),
          remoteDevicePhyId(remoteDevicePhyId)
    {}
};

using ErrCqeInfo = struct TagErrCqeInfo {
    CqeInfo cqeInfo;
    LinkInfo linkInfo;
    u32 qpn;
    TagErrCqeInfo() {}
    TagErrCqeInfo(CqeInfo& cqeInfo, LinkInfo& linkInfo, u32 qpn) : cqeInfo(cqeInfo), linkInfo(linkInfo), qpn(qpn) {}
    bool operator<(const TagErrCqeInfo& other) const { return qpn < other.qpn; }
};

class Heartbeat {
public:
    static Heartbeat& GetInstance(s32 deviceLogicID);
    HcclResult RegisterToHeartBeat(
        u32 userRank, DevType devType, std::vector<RankInfo>& rankInfoList, const u32 port, const bool isNeedNic,
        const std::string& commIdentifier, bool useSuperPodMode, bool isUsedRdmaLevel0, bool retryEnable = false,
        bool backupEnable = false);
    HcclResult RegisterToHeartBeat(
        u32 userRank, DevType devType, std::vector<RankInfo>& rankInfoList, const u32 port, const bool isNeedNic,
        u32 peerRankId, const std::string& commIdentifier, const std::string& tag, bool useSuperPodMode,
        bool isUsedRdmaLevel0, bool retryEnable = false, bool backupEnable = false);
    HcclResult AddOpInfoToHeartBeat(const std::string& identifier, const OpInfoDesc& opInfo, const std::string& newTag);
    HcclResult DeleteOpInfoToHeartBeat(const std::string& identifier, const std::string& newTag);
    HcclResult UnRegisterRanks(const std::string& group = HCCL_WORLD_GROUP);
    // 集合通信，解开注册
    void UnRegisterToHeartBeat(DevType devType, const std::string& commIdentifier);
    // 非点对点通信，解开注册
    void UnRegisterToHeartBeat(DevType devType, const std::string& commIdentifier, const std::string& tag);
    HcclResult CheckErrorCqe(const std::string& identifier, HcclResult& result);
    HcclResult CheckOpInconsistentError(const std::string& identifier, HcclResult& result);
    HcclResult SetRankPortInfo(
        bool isUseRankPort, std::vector<u32>& nicRanksPorts, std::vector<u32>& vnicRanksPorts, bool devPortSwitchOn);
    std::vector<std::string> GetErrStatusVec(const std::string& group = HCCL_WORLD_GROUP);
    HcclResult GetQpnErr(const std::string& identifier, std::set<std::tuple<u32, u32, u32>>& qpErrSet);
    HcclResult BroadcastCqeErr(const std::string& identifier);
    HcclResult ClearAllCqeErr(const std::string& identifier);
    HcclResult ClearCqeErr(const std::string& identifier, u32 remoteRank, u32 qpn = 0);
    void SetOpretryErr();
    void GetIpQueue();
    bool IsPaused() const;
    bool IsResumed() const;

private:
    Heartbeat() = default;
    ~Heartbeat();
    HcclResult Init(
        const RankInfo& locRank, const bool useSuperPodMode, const bool isNeedNic, const u32 port,
        const std::string& group = HCCL_WORLD_GROUP);
    HcclResult DeInit();
    HcclResult RegisterRanks(
        DevType devType, const RankInfo& locRank, std::vector<RankInfo>& rankInfos, const u32 port,
        const bool isNeedNic, const std::string& group = HCCL_WORLD_GROUP, bool useSuperPodMode = false,
        bool isUsedRdma = false);
    std::string GetConnTag(HcclSocketRole role, UIDType& rem);
    HcclResult GetConnInfo(
        RankInfo& remRank, bool useSuperPodMode, HcclSocketRole role, HcclSocketType type,
        std::map<UIDType, ConnInfo>& needConnectRank);
    template <typename T>
    HcclResult GetSamePlaneConnInfo(
        HcclSocketType type, std::vector<std::pair<T, u32>>& connVec, T& locId, std::vector<RankInfo>& rankInfos,
        std::map<UIDType, ConnInfo>& needConnectRank, bool useSuperPodMode, u32 worldRank);
    HcclResult GetConnectRank(
        const RankInfo& locRank, std::vector<RankInfo>& rankInfos, std::map<UIDType, ConnInfo>& needConnectRank,
        bool useSuperPodMode, bool isUsedRdma = false);
    UIDType GetUId(const RankInfo& rankInfo) const;
    std::string FormatUId(const UIDType& uid) const;
    HcclResult SendFrame(UIDType& dst, UIDType& crimer, UIDType& informer, HeartBeatStatus status);
    HcclResult SendFrameWithOpCheck(
        UIDType& dst, UIDType& crimer, UIDType& informer, HeartBeatStatus status,
        const OpInfoTagQueueFrame& opInfoTagQueueFrame);
    HcclResult RecvFrame(UIDType& src);
    HcclResult RecvFrameWithOpCheck(UIDType& src);
    HcclResult ParseFrame(HeartBeatFrame& bf, UIDType& src);
    HcclResult ParseFrameWithOpCheck(HeartBeatFrameWithOpCheck& bf, UIDType& src);
    void SetStatus(UIDType& crimer, UIDType& informer, HeartBeatStatus status, bool needBroadcast = true);
    void HeartbeatStatusMonitor();
    void ProcessExceptionEvent();
    void ProcessCqeErrInfo();
    void DelErrorSocket();
    bool IsKeyEvent(HeartBeatFrame& event, HcclUs curTime, const std::string& group = HCCL_WORLD_GROUP);
    void MakeErrMsg(std::queue<HeartBeatFrame>& keyEvents, std::vector<std::string>& errStatusVec);
    std::vector<std::string> PrintEvents(std::map<HeartBeatStatus, std::queue<HeartBeatFrame>>& keyEvents);
    void StuckDetection(uint64_t& cnt, CounterStat& counterStat);
    void InitStuckDetection(CounterStat& counterStat);
    void PrintAndBroadCastErrorCqe(const ErrCqeInfo& info);
    void SaveQpnForOpRetry(const ErrCqeInfo& info);
    void OpRetryCQEHandle(const HcclNetDevCtx netDevCtx);
    bool GetRetryEnable(const ErrCqeInfo& info);
    HcclResult ClearRetryEnableMapItem(const std::string& identifier);
    void ProcessCqeErrInfoByNetDevCtx(const HcclIpAddress& nicIp);
    bool IsEnableBackupLink();
    void RegisterRetryInfo(const std::string& commIdentifier, bool retryEnable, bool backupEnable);
    HcclResult InitNic(
        const NicType nicType, const s32 devicePhyId, const s32 deviceLogicId, const hccl::HcclIpAddress ip,
        const u32 port, const bool isBackUp = false);
    HcclResult InitDeviceNic(const RankInfo& locRank, bool isNeedNic, u32 port);
    HcclResult InitHostNic(const RankInfo& locRank, bool isNeedNic, u32 port);
    u32 GetPort(HcclSocketType type, u32 remoteUserRank, u32 remoteDeviceId);
    u32 GetHostPort(s32 devicePhyId);
    HcclResult PrepareConnect(ConnInfo& info);
    void CreateLinkWithRemote(std::string group, UIDType rem, ConnInfo needConnectRank);
    void CreateHBLinksAsync();
    void AddOpInfo(const std::string& identifier, const OpInfoDesc& opInfo, const std::string& paramTag);
    void GetOneOpInfo(std::string& tag, OpInfoDesc& opInfo);
    void GetSendOpInfoList(OpInfoTagQueueFrame& opInfoTagQueueFrame);
    void SaveOpInfo(const OpInfoTagQueueFrame& opInfoTagQueueFrame, UIDType& src);
    HcclResult CheckIsSameOp(const OpInfoDesc& localOpInfo, const OpInfoDesc& remoteOpInfo, InconsistentType& status);
    void CheckRecvOpInfoList();
    void RegisterSROpIdentifier(const std::string& identifier, const std::string& paramTag);
    void AddInconsistentOpRecord(
        const std::string& identifier, const OpInfoDesc& localOpInfo, InconsistentType status,
        const std::string& localInfo, const std::string& remoteInfo);
    void CheckSnapshotStatus();
    struct Status {
        HeartBeatStatus status = HeartBeatStatus::HEARTBEAT_OK;
        UIDType informer;
        bool needBroadcast = false;
        Status() {}
    };

    enum class HBLinkStatus { HEARTBEAT_LINK_NOT_START, HEARTBEAT_LINK_BUILDING, HEARTBEAT_LINK_COMPLETED };

    HcclIpAddress vnicIp_;
    HcclIpAddress nicIp_;
    HcclIpAddress backupNicIp_;
    u32 devicePhyId_;
    u32 deviceBackUpPhyId_;
    u32 superDeviceId_;
    NICDeployment nicDeploy_;
    UIDType uid_;
    bool initialized_ = false;
    u32 lostThreshold_ = 0;
    bool isDeInit_ = false;
    bool startSendRecvTask_ = false;
    std::map<std::string, std::queue<std::pair<UIDType, ConnInfo>>> hbLinkConnInfo_{};
    std::mutex hbLinkConnInfoMtx_;
    std::map<std::string, std::map<UIDType, u8>> groupMap_;
    ReferenceMap<UIDType, ConnInfo> rankId2SocketMap_;
    ReferenceMap<UIDType, Status> rankId2StatusMap_;
    std::map<UIDType, HBLinkStatus> rankId2LinkStatusMap_;
    std::map<UIDType, std::unique_ptr<std::thread>> linkThreadMap_{};
    std::atomic<bool> linkThreadRunning_{false};
    std::atomic<u32> linkThreadCount_{0};
    std::unique_ptr<std::thread> sendRecvThread_;
    std::queue<HeartBeatFrame> errStatusQueue_;
    std::queue<UIDType> errRankQueue_;
    std::mutex ProcessLock_;
    u32 deviceLogicId_;
    u32 deviceBackupLogicId_;
    std::map<std::string, std::set<ErrCqeInfo>> remoteIpMap;
    std::set<u32> qpnDissociativeSet;
    std::mutex remoteIpMutex_;
    bool isUseRankPort_{false};
    bool devPortSwitchOn_{false};
    std::vector<u32> nicRanksPorts_;
    std::vector<u32> vnicRanksPorts_;
    std::vector<UIDType> errorSocket_;
    std::map<std::string, std::map<u32, std::set<ErrCqeInfo>>> rankMapForRetryAgent;
    std::mutex qpnMapMutexForRetry_;
    std::map<std::string, bool> retryEnableTable_;
    std::mutex retryEnableMutex_;
    std::set<std::string> backupEnableTable_;
    std::mutex backupEnableMutex_;
    std::mutex ctxMapMutex_;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap_;
    std::map<HcclIpAddress, std::shared_ptr<HcclSocket>> listenSocketMap_;
    s32 stuckDetectTime_;
    std::mutex opInfoQueueMutex_;
    std::deque<std::pair<std::string, OpInfoDesc>> opInfoQueue_;
    std::deque<std::pair<std::string, OpInfoDesc>> opInfoQueueForSend_;
    std::unordered_map<std::string, u64> opInfoIndexMap_;
    std::mutex opInfoMapMutex_;
    std::unordered_map<std::string, std::map<u64, OpInfoDesc>> opInfoMap_;
    std::list<std::tuple<OpInfoDesc, std::string, UIDType>> recvOpInfoList_;
    std::mutex inconsistentOpMutex_;
    std::map<std::string, OpInconsistentInfo> inconsistentOpMap_;
    std::mutex srTagMutex_;
    std::map<std::string, std::string> srTagMap_; // SR算子tag->identifier映射
    bool isPaused_{false};                        // heartbeat need to be paused when snapshot
};
} // namespace hccl

#endif // HCCL_HEARTBEAT_H
