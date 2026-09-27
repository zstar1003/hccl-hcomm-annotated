/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SOCKET_MANAGER_H
#define HCCL_SOCKET_MANAGER_H

#include <map>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>
#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "hccl_ip_address.h"
#include "hccl_socket.h"
#include "common.h"

namespace hccl {
class PortInfo {
public:
    PortInfo(const HcclIpAddress& ip, u32 listenPort) : ip(ip), listenPort(listenPort) {}
    ~PortInfo() {}

    bool operator==(const PortInfo& portInfo) const { return listenPort == portInfo.listenPort && ip == portInfo.ip; }

    bool operator!=(const PortInfo& portInfo) const { return !(portInfo == *this); }

    bool operator<(const PortInfo& portInfo) const
    {
        if (ip < portInfo.ip) {
            return true;
        }
        if (portInfo.ip < ip) {
            return false;
        }
        return listenPort < portInfo.listenPort;
    }

    bool operator>(const PortInfo& portInfo) const { return portInfo < *this; }

    bool operator<=(const PortInfo& portInfo) const { return !(portInfo < *this); }

    bool operator>=(const PortInfo& portInfo) const { return !(*this < portInfo); }

    HcclIpAddress ip;
    u32 listenPort;
};
using NicHandleInfo = struct NicHandleInfoDef {
    HcclIpAddress ip;
    SocketHandle nicSocketHandle;
    NicType socketType;

    NicHandleInfoDef() : ip(), nicSocketHandle(nullptr), socketType(NicType::DEVICE_NIC_TYPE) {}
};

class HcclSocketManager {
public:
    explicit HcclSocketManager(NICDeployment nicDeployment, s32 deviceLogicId, u32 devicePhyId, u32 userRank);
    virtual ~HcclSocketManager();

    HcclResult AddWhiteList(const std::string& commTag, const HcclNetDevCtx netDevCtx, HcclRankLinkInfo remoteRankInfo);
    void DestroySockets(const std::string& commTag);
    void DestroySockets(const std::string& commTag, u32 rank);
    HcclResult CreateSockets(
        const std::string& commTag, bool isInterLink, const HcclNetDevCtx netDevCtx,
        const std::map<u32, HcclRankLinkInfo>& dstServerMap, const std::map<u32, HcclRankLinkInfo>& dstClientMap,
        std::map<u32, std::vector<std::shared_ptr<HcclSocket>>>& serverSocketsMap,
        std::map<u32, std::vector<std::shared_ptr<HcclSocket>>>& clientSocketsMap, bool isSupportReuse = false,
        bool isWaitEstablished = true);
    HcclResult
    GetListenPortByIp(const NICDeployment nicDeployment, const HcclIpAddress& ipAddr, std::set<u32>& listenedPort);

    void GetSocketsByRankIP(
        const std::string& commTag, u32 remoteRank, const HcclIpAddress& remoteIp, u32 socketsPerLink,
        std::vector<std::shared_ptr<HcclSocket>>& ipSockets, u32& gotLinkNum) const;
    void GetSocketsByRankIP(
        const HcclIpAddress& remoteIp, u32 socketsPerLink, const std::vector<std::shared_ptr<HcclSocket>>& rankSockets,
        std::vector<std::shared_ptr<HcclSocket>>& ipSockets, u32& gotLinkNum) const;

    HcclResult ServerInit(const HcclNetDevCtx netDevCtx, u32 port);
    HcclResult ServerDeInit(const HcclNetDevCtx netDevCtx, u32 port);

    HcclResult CreateSingleLinkSocket(
        const std::string& commTag, const HcclNetDevCtx netDevCtx, HcclRankLinkInfo rmtRank,
        std::vector<std::shared_ptr<HcclSocket>>& connectSockets, bool isWaitEstablished = true,
        bool isSupportReuse = false, s32 timeout = 0, uint32_t connectMode = 0);

    HcclResult WaitLinksEstablishCompleted(
        HcclSocketRole localRole, std::map<u32, std::vector<std::shared_ptr<HcclSocket>>>& socketsMap,
        std::map<u32, u32>& dstRankToUserRank, const RankInfo& loaclRankInfo, const RankInfo& remoteRankInfo,
        const HcclNetDevCtx& netDevCtx);
    void DestroySockets();

    void AbortAndDeleteSocket(
        const std::string& commTag, HcclSocketRole role,
        const std::map<u32, std::vector<std::shared_ptr<HcclSocket>>>& socketsMap);

    HcclResult SetStopFlag(bool value);
    bool GetStopFlag();
    HcclResult WaitLinkEstablish(
        std::shared_ptr<HcclSocket> socket,
        std::function<bool()> needStop =
            []() {
                return false;
            },
        s32 timeout = 0) const;
    HcclResult ServerDeInit(const HcclIpAddress& localIp, u32 port);

private:
    HcclResult AddWhiteList(
        const std::string& commTag, bool isInterLink, NicType socketType, const HcclIpAddress& localIp,
        const std::map<u32, HcclRankLinkInfo>& whiteListMap, uint32_t connectMode = 0);
    HcclResult DelWhiteList(const std::string& commTag);
    HcclResult CreateSockets(
        const std::string& commTag, bool isInterLink, const HcclNetDevCtx netDevCtx, NicType socketType,
        HcclSocketRole localRole, const HcclIpAddress& localIp, const HcclRankLinkInfo& remoteLinkInfo,
        std::vector<std::shared_ptr<HcclSocket>>& ipSockets, bool isSupportReuse, uint32_t connectMode = 0);
    HcclResult CreateSockets(
        const std::string& commTag, bool isInterLink, const HcclNetDevCtx netDevCtx, NicType socketType,
        HcclSocketRole localRole, const HcclIpAddress& localIp, const std::map<u32, HcclRankLinkInfo>& remoteInfos,
        std::map<u32, std::vector<std::shared_ptr<HcclSocket>>>& socketsMap, std::map<u32, u32>& dstRankToUserRank,
        bool isSupportReuse, uint32_t connectMode = 0);
    void DestroySockets(std::vector<std::shared_ptr<HcclSocket>> rankSockets);
    void TransformSocketStatus(HcclSocketStatus status, std::string& stringStatus) const;
    void PrintSocketsInfo(
        const std::string& localRole, u32 rank, std::vector<std::shared_ptr<HcclSocket>> ipSockets,
        std::string& sTlsStatus) const;
    void PrintErrorConnectionInfo(
        HcclSocketRole localRole, std::map<u32, std::vector<std::shared_ptr<HcclSocket>>>& rankSocketsMap,
        std::map<u32, u32>& dstRankToUserRank, TlsStatus& tlsStatus) const;
    void PrintErrorConnection(
        HcclSocketRole localRole, std::map<u32, std::vector<std::shared_ptr<HcclSocket>>>& rankSocketsMap,
        std::map<u32, u32>& dstRankToUserRank, TlsStatus& tlsStatus) const;
    u32 GetConnLimit(NicType socketType);
    std::string MakeUniqueConnTag(const std::string& commTag, bool isInterLink, u32 userRank, u32 indexForLink) const;
    HcclResult ConstructWhiteList(
        const std::string& commTag, bool isInterLink, NicType socketType, const HcclRankLinkInfo& dstRankLinkInfo,
        std::vector<SocketWlistInfo>& wlistInfosVec, uint32_t connectMode = 0);
    void SaveWhiteListInfo(
        const std::string& commTag, std::shared_ptr<HcclSocket>& socket, const std::vector<SocketWlistInfo> wlistInfos);
    HcclResult ConstructSockets(
        const std::string& commTag, bool isInterLink, const HcclNetDevCtx netDevCtx, u32 socketsPerLink,
        NicType socketType, u32 remoteUserRank, const HcclIpAddress& remoteIp, u32 remotePort,
        const HcclIpAddress& localIp, HcclSocketRole localRole, std::vector<std::shared_ptr<HcclSocket>>& socketList,
        uint32_t connectMode = 0);
    void SaveSockets(
        const std::string& commTag, u32 remoteRank, const HcclIpAddress& remoteIp,
        std::vector<std::shared_ptr<HcclSocket>>& ipSockets);

    HcclResult WaitLinksEstablishCompleted(
        HcclSocketRole localRole, std::map<u32, std::vector<std::shared_ptr<HcclSocket>>>& rankSocketsMap,
        s32 timeout = 0);

    void AddIpQueue(RankInfo& localRankInfo, RankInfo& remoteRankInfo, NicType nicType, s32 deviceLogicId) const;
    NICDeployment nicDeployment_;
    s32 deviceLogicId_;
    u32 devicePhyId_;
    u32 userRank_;

    // 后继这个放在HcclSocket中管理
    std::map<std::string, std::map<std::shared_ptr<HcclSocket>, std::vector<SocketWlistInfo>>> wlistInfosMap_;
    std::map<std::string, std::map<u32, std::vector<std::shared_ptr<HcclSocket>>>> commSocketsMap_;
    std::mutex wlistMapMutex_;
    std::mutex socketsMapMutex_;

    static std::mutex serverMapMutex_;
    static std::map<PortInfo, std::shared_ptr<HcclSocket>> serverSocketMap_;
    static std::map<PortInfo, Referenced> serverSocketRefMap_;

    std::atomic<bool> stopFlag_{false};
};

using IntraExchanger = struct IntraExchangerDef {
    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> socketsMap;
    std::shared_ptr<HcclSocketManager> socketManager;
    IntraExchangerDef() : socketsMap(), socketManager() {}
};

using RegisterDetectCallBack
    = void (*)(RankInfo& localRankInfo, RankInfo& remoteRankInfo, NicType nicType, s32 deviceLogicId);
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
void DetectCallBack(RegisterDetectCallBack p1);
#ifdef __cplusplus
}
#endif // __cplusplus

} // namespace hccl
#endif /* * HCCL_SOCKET_MANAGER_H */
