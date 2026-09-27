/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_REMOTE_ACCESS_H
#define COMM_REMOTE_ACCESS_H

#include <vector>
#include <memory>
#include <map>

#include "hccl/base.h"
#include "transport_remote_access.h"
#include "network_manager_pub.h"
#include "network/hccp.h"
#include "network/hccp_common.h"
#include "dispatcher_pub.h"
#include "dispatcher_pub.h"
#include "adapter_error_manager.h"
#include "workflow_pub.h"

namespace hccl {

constexpr s32 SOCKET_CONNECT_NO_CONNECTION = 0;
constexpr s32 SOCKET_CONNECT_OK = 1;
constexpr s32 SOCKET_CONNECT_TIMEOUT = 2;
constexpr s32 SOCKET_CONNECT_CONNECTING = 3;
struct LinkStatus_t {
    u32 userRank{INVALID_VALUE_RANKID};
    s32 status{0};
    bool isLinked{false};
    HcclIpAddress remoteIp;
    HcclIpAddress localIp;
    std::string tag{""};
};

class CommRemoteAccess {
public:
    explicit CommRemoteAccess(
        u32 rank, u32 devicePhyId, const std::map<u32, std::vector<HcclIpAddress>>& rankInfo,
        const std::vector<MemRegisterAddr>& addrInfos);
    ~CommRemoteAccess();
    HcclResult Init();
    std::shared_ptr<TransportRemoteAccess>& GetTransportByRank(const u32 dstRank);

private:
    HcclResult RescoucePrepare();
    HcclResult CommRemoteInitRa();
    HcclResult CommRemoteDeInitRa();
    HcclResult CalcRemoteLink();
    HcclResult PrepareSocket();
    HcclResult AddSocketWhiteList();
    HcclResult GetRaSocket(const u32 role, const struct SocketInfoT conn[], const u32 num);
    HcclResult CreateLinks();
    HcclResult CalcLinksRelation();
    HcclResult CalcLinksNum();
    HcclResult CreateInterClientLinks();
    HcclResult CreateInterServerLinks();
    HcclResult DealSuccRasocket(s32 sockRet, const u32 role, const struct SocketInfoT tmpConn[], const u32 num);
    void PrintErrorConnection(const u32 role, const u32 num);
    HcclResult PrintErrorConnectionInfo(const std::map<HcclIpAddress, LinkStatus_t>& linkStatusMap, u32 role);
    HcclResult GetDstRank(std::map<u32, std::vector<HcclIpAddress>>& dstMap, const HcclIpAddress& dstIp, u32& dstRank);
    HcclResult CreateInterThread(const u32 role, const SocketInfoT& socketInfo);
    HcclResult InitDestTransport(
        const ErrContext& error_context, u32 role, const HcclIpAddress& nicIp, const u32 dstRank,
        const std::string& threadStr, FdHandle socketFdHandle, u32* getThreadStatus);
    HcclResult GetNicByHandle(const SocketHandle socketHandle, HcclIpAddress& nicIp);
    HcclResult SetAccessPara(
        u32 role, const HcclIpAddress& nicIp, u32 dstRank, FdHandle socketFdhandle, RemoteAccessPara& accessPara);
    HcclResult DeleteSocketWhiteList();

    std::shared_ptr<TransportRemoteAccess> transportDummy_;
    std::multimap<u32, std::shared_ptr<TransportRemoteAccess>> remoteTransportMap_;

    u32 rank_; // 当前通信域的rank
    s32 deviceLogicId_;
    u32 devicePhyId_;
    u32 rankSize_;                // 当前通信域的ranksize
    NICDeployment nicDeployment_; // 网卡部署位置 0:host 1:device

    std::map<u32, std::vector<HcclIpAddress>> rankInfo_; // rank 与 ip 地址的 map
    std::vector<MemRegisterAddr> addrInfos_;
    std::vector<struct SocketWlistInfoT> wlistInfosVec_;
    std::map<u32, std::vector<HcclIpAddress>> dstInterServerMap_;
    std::map<u32, std::vector<HcclIpAddress>> dstInterClientMap_;
    std::map<u32, struct SocketInfoT> serverConnsMap_;
    std::map<u32, struct SocketInfoT> clientConnsMap_;
    RaResourceInfo raResourceInfo_;
    std::vector<SocketHandle> nicSocketHandle_;
    std::string tag_;
    std::map<HcclIpAddress, LinkStatus_t> clientLinkStatus_;
    std::map<HcclIpAddress, LinkStatus_t> serverLinkStatus_;

    std::vector<SocketInfoT> raSockets_;                    // 保存建链成功的socket, 用于创建transport实例
    std::vector<std::unique_ptr<std::thread>> linkThreads_; // 建链所需线程
    u32 threadsApplyNum_;                                   // 线程使用计数器
    HcclDispatcher dispatcher_;
    std::unique_ptr<NotifyPool> notifyPool_;
    std::vector<u32> threadsStatus_;
    std::mutex remoteTransportMapLock_;
    HcclWorkflowMode workflowMode_{HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE};
};
} // namespace hccl

#endif // COMM_REMOTE_ACCESS_H
