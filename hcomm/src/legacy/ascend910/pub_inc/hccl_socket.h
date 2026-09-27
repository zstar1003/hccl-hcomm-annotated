/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_HCCL_SOCKET_H
#define HCCL_HCCL_SOCKET_H

#include <vector>
#include <memory>
#include <atomic>
#include "hccl/hccl_types.h"
#include "hccl_common.h"
#include "hccl_ip_address.h"
#include "hccl_network_pub.h"
#include "adapter_hccp_common.h"
#include "externalinput_pub.h"

namespace hccl {
constexpr u32 SERVER_ROLE_SOCKET = 0;
constexpr u32 CLIENT_ROLE_SOCKET = 1;

constexpr u32 NIC_SOCKET_CONN_LIMIT = 1;
constexpr u32 VNIC_SOCKET_CONN_LIMIT = 1;
constexpr u32 SOCKET_BATCH_GET_LIMIT = 16; // 每次最多建立16个socket连接
constexpr u32 HOST_SOCKET_CONN_LIMIT = 16; // 用于host socket建链白名单，限制每个server的socket连接数

enum class HcclSocketType {
    SOCKET_NIC,
    SOCKET_HOST_NIC,
    SOCKET_VNIC,
};

enum class HcclSocketRole {
    SOCKET_ROLE_SERVER = 0, /* server 角色 */
    SOCKET_ROLE_CLIENT = 1, /* client 角色 */
    SOCKET_ROLE_RESERVED    /* 作为Listen Socket，或标识无需创建socket连接 */
};

enum class HcclSocketStatus {
    SOCKET_INIT = 0,
    SOCKET_CONNECTING = 1,
    SOCKET_OK = 2,
    SOCKET_TIMEOUT = 3,
    SOCKET_ERROR = 4,
};

// 如果一个Rank有多个IP里, 使用 std::vector<HcclRankLinkInfo> 描述
using HcclRankLinkInfo = struct HcclRankLinkInfoDef {
    u32 userRank;
    u32 devicePhyId;
    HcclIpAddress ip;
    u32 port;
    u32 socketsPerLink;

    HcclRankLinkInfoDef() : userRank(), devicePhyId(), ip(), port(), socketsPerLink() {}
};

class HcclSocket {
public:
    explicit HcclSocket(
        const std::string& tag, HcclNetDevCtx netDevCtx, const HcclIpAddress& remoteIp, u32 remotePort,
        HcclSocketRole localRole);
    explicit HcclSocket(HcclNetDevCtx netDevCtx, u32 localPort = HCCL_INVALID_PORT);

    ~HcclSocket();
    HcclResult Init();
    HcclResult DeInit();
    HcclResult Listen();
    HcclResult Listen(u32 port);
    HcclResult Connect();
    void Close();
    void SetStatus(HcclSocketStatus status);
    HcclSocketStatus GetStatus();
    HcclResult Accept(const std::string& tag, std::shared_ptr<HcclSocket>& socket, u32 acceptTimeOut = 0);
    HcclResult Send(const void* data, u64 size);
    HcclResult Recv(void* recvBuf, u32 recvBufLen, u32 timeout = 0);
    HcclResult Send(const std::string& sendMsg);
    HcclResult Recv(std::string& recvMsg, u32 timeout = 0);
    HcclResult ISend(void* data, u64 size, u64& compSize);
    HcclResult IRecv(void* recvBuf, u32 recvBufLen, u64& compSize);

    static bool IsSupportAsync();
    HcclResult SendAsync(const void* data, u64 size, u64* sentSize, void** reqHandle);
    HcclResult RecvAsync(void* recvBuf, u64 recvBufLen, u64* receivedSize, void** reqHandle);
    HcclResult GetAsyncReqResult(void* reqHandle, HcclResult& reqResult);

    HcclResult AddWhiteList(std::vector<SocketWlistInfo>& wlistInfoVec);
    HcclResult DelWhiteList(std::vector<SocketWlistInfo>& wlistInfoVec);

    std::string GetTag() const;
    NicType GetSocketType() const;
    HcclIpAddress GetRemoteIp() const;
    u32 GetRemotePort() const;
    HcclIpAddress GetLocalIp() const;
    u32 GetLocalPort() const;
    HcclSocketRole GetLocalRole() const;
    FdHandle GetFdHandle() const;
    void SetForceClose(bool forceClose);
    HcclResult SetStopFlag(bool value);
    bool GetStopFlag();

private:
    HcclSocketStatus ConvertRaSocketStatus(int raStatus);
    HcclResult GetNicSocketHandle();
    HcclResult GetNicSocketHandle(
        std::map<HcclIpAddress, IpSocket>& socketMap, const HcclIpAddress& ip, SocketHandle& nicSocketHandle);
    HcclResult ListenHostNet(HcclResult& ret, std::string& errormessage, u32& port);

    std::string tag_;
    HcclNetDevCtx netDevCtx_;
    NICDeployment nicDeployment_;
    NicType socketType_;
    s32 localDeviceLogicId_;
    SocketHandle nicSocketHandle_{nullptr};
    s32 localDevicePhyId_;
    HcclIpAddress remoteIp_;
    u32 remotePort_;
    HcclIpAddress localIp_;
    HcclIpAddress backupIp_;
    u32 localPort_;
    HcclSocketRole localRole_;
    HcclSocketStatus status_;
    FdHandle fdHandle_;
    bool isHostUseDevNic_{false};
    bool listened_{false};
    bool forceClose_{false};
    std::atomic<bool> stopFlag_{false};
    s32 sendStatus_{0};
    s32 recvStatus_{0};
};

} // namespace hccl
#endif // HCCL_HCCL_SOCKET_H
