/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DETECT_CONNECT_ANOMALIES_H
#define HCCL_DETECT_CONNECT_ANOMALIES_H
#include <queue>
#include <map>
#include <unordered_set>
#include <mutex>
#include <thread>
#include "hccl_socket.h"
#include "hccl_socket_manager.h"
#include "hccl_ip_address.h"
#include "common.h"

namespace hccl {
// todo 本端和对端的都得保存，并打印
constexpr size_t DEST_MAX_LEN = 128;
constexpr u32 MAX_WHITE_LIST_ENTRY = 16;
constexpr u32 ACCEPT_TIME_OF_USLEEP = 100000;
constexpr u32 IRECV_TIME_OF_USLEEP = 500000;
constexpr u32 CLIENT_TIME_OF_USLEEP = 1500000;
struct DetectInfo {
    s32 localDeviceId = 0XFFFFFFFF;
    s32 remoteDeviceId = 0XFFFFFFFF;

    char localDeviceIp[DEST_MAX_LEN]{};  // 用来查重
    char remoteDeviceIp[DEST_MAX_LEN]{}; // 用来查重

    char localServerId[DEST_MAX_LEN]{};
    char remoteServerId[DEST_MAX_LEN]{};
};

struct SendInfo {
    bool isSendNic = false;
    bool isSendVnic = false;
};

struct ErrInfo {
    RankInfo localRankInfo;
    RankInfo remoteRankInfo;
    NicType nicType;
    s32 deviceLogicId;
};

// 统一处理 IP 插入逻辑
template <typename ListType>
HcclResult AddWlistEntry(
    const HcclIpAddress& ipAddr, const std::string& tag, ListType& whiteList, std::vector<SocketWlistInfo>& wlistVec)
{
    // 查找是否已存在
    if (std::find(whiteList.begin(), whiteList.end(), ipAddr) != whiteList.end()) {
        return HCCL_SUCCESS;
    }

    // 构造白名单信息，按照最大限制16下发
    SocketWlistInfo wlistInfo = {};
    wlistInfo.connLimit = MAX_WHITE_LIST_ENTRY;
    wlistInfo.remoteIp.addr = ipAddr.GetBinaryAddress().addr;
    wlistInfo.remoteIp.addr6 = ipAddr.GetBinaryAddress().addr6;
    CHK_SAFETY_FUNC_RET(memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), tag.c_str(), tag.size() + 1));

    // 记录白名单信息
    HCCL_INFO("[AddWlistEntry] ip[%s] tag[%s]", ipAddr.GetReadableIP(), wlistInfo.tag);
    wlistVec.push_back(wlistInfo);
    whiteList.insert(ipAddr);
    return HCCL_SUCCESS;
}

class DetectConnectionAnomalies {
public:
    static DetectConnectionAnomalies& GetInstance(s32 deviceLogicID);
    void Init(std::vector<RankInfo>& rankInfos, bool isNeedNic);
    void AddIpQueue(RankInfo& localRankInfo, RankInfo& remoteRankInfo, NicType nicType, s32 deviceLogicId);
    HcclResult Detect();
    void Deinit();

private:
    void DetectMonitor();
    HcclResult GetIpQueue();
    HcclResult CreateServers(struct ErrInfo errInfo);
    std::string GetTag(HcclIpAddress& Ip, int i = 0);
    HcclResult AddWhiteList(std::shared_ptr<HcclSocket> socket, NicType nicType, std::string& tag);
    HcclResult DelWhiteList(
        HcclIpAddress& localIpAddr, std::vector<struct SocketWlistInfo> whiteListInfos,
        std::shared_ptr<HcclSocket> socket);
    HcclResult GetStatus(struct ErrInfo errInfo, std::shared_ptr<HcclSocket>& clientSocket);
    HcclResult Connect(struct ErrInfo errInfo, std::shared_ptr<HcclSocket>& clientSocket);
    HcclResult CreateDetectVnicLinks(struct ErrInfo errInfo);
    HcclResult CreateDetectNicLinks(struct ErrInfo errInfo);
    HcclResult CreateClients(struct ErrInfo errInfo, std::vector<std::unique_ptr<std::thread>>& linkClientThreads);
    HcclResult
    ConstructErrorInfo(std::shared_ptr<HcclSocket>& clientSocket, RankInfo& localRankInfo, RankInfo& remoteRankInfo);
    HcclResult CreateClient(struct ErrInfo errInfo);
    HcclResult processWhiteList(
        const HcclIpAddress& ipAddr, HcclIpAddress& localIpAddr, std::shared_ptr<HcclSocket> socket, NicType nicType);
    HcclResult WaitForDectect();
    HcclResult ProcessDetectionResults();
    std::string
    FormatDetectMessage(const std::string& localServerId, s32 localDeviceId, const DetectInfo& detectInfo) const;
    std::string BuildGroupedDetectMessage();
    void ThreadDestroy();
    ~DetectConnectionAnomalies() = default;
    DetectConnectionAnomalies() = default;
    int broadCastTime = 10; // 故障广播时间
    std::set<HcclIpAddress> uniqueIps_;
    bool threadExit_ = true;
    bool isNeedNic_ = false;
    bool isInitThread_ = false;
    std::mutex ipNictypeQueueMutex_;
    std::mutex ipConstuctMutex_;
    std::mutex whiteListMutex_;       // 删除白名单需要加锁
    std::mutex clientThreadMutex_;    // 删除clients需要加锁
    std::mutex clientResourcesMutex_; // 保护clientNicCtxs_和clientSockets_的并发访问
    std::mutex printDetectInfoMutex_; // 打印锁
    std::mutex detectThreadMutex_;
    std::set<HcclIpAddress> whiteVnicSet_; // 保存vnic的白名单 whiteVnicSet_
    std::set<HcclIpAddress> whiteNicSet_;  // 保存nic的白名单
    std::shared_ptr<HcclSocket> vnicSocket_ = nullptr;
    std::shared_ptr<HcclSocket> nicSocket_ = nullptr;
    std::vector<std::shared_ptr<HcclSocket>> clientSockets_; // 保存client端的socket
    std::map<HcclIpAddress, HcclIpAddress> ipMap_;
    std::map<HcclIpAddress, std::shared_ptr<HcclSocket>> socketMap_;
    std::map<HcclIpAddress, HcclNetDevCtx> nicNetDevCtxMap_;
    std::vector<std::shared_ptr<HcclSocket>> listenNicVec_;
    std::vector<std::shared_ptr<HcclSocket>> listenVnicVec_;
    std::queue<ErrInfo> ipNictypeQueue_;
    HcclNetDevCtx nicCtx_;
    HcclNetDevCtx vnicCtx_;
    std::vector<HcclNetDevCtx> clientNicCtxs_;
    std::unique_ptr<std::thread> getIpNictypeQueue_ = nullptr;
    std::unique_ptr<std::thread> detectVnicThread_ = nullptr;
    std::unique_ptr<std::thread> detectNicThread_ = nullptr;
    std::vector<struct SocketWlistInfo> vnicWhiteListInfosVec_; // 保存vnic白名单单信息，方便删除的时候使用
    std::vector<struct SocketWlistInfo> nicWhiteListInfosVec_; // 保存nic白名单单信息，方便删除的时候使用

    // 发送完成后添加，发送前查重
    std::unordered_map<std::string, SendInfo> sendErrorInfoMap_;
    // 接收到添加，接收前查重
    std::unordered_map<std::string, DetectInfo> recvErrorInfoMap_;
    std::mutex readRecvErrtInfo_;

    bool isCreateLink_ = false;
    bool isCreateNicLink_ = false;
    std::atomic<bool> isPrint_{false};
    std::atomic<int> errorCount_{0};
    std::vector<std::unique_ptr<std::thread>> linkClientThreads_; // 保存client拉起的线程
    Referenced initRef_;
    std::chrono::steady_clock::time_point startTime;
    std::mutex time_mutex;
    std::mutex print_mutex;
};
} // namespace hccl

#endif // HCCL_DETECT_CONNECT_ANOMALIES_H
