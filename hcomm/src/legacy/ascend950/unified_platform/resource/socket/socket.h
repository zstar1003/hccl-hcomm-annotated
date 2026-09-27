/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_SOCKET_BASE_H
#define HCCLV2_SOCKET_BASE_H

#include <chrono>

#include "ip_address.h"
#include "socket_config.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

MAKE_ENUM(SocketStatus, INIT, LISTEN_STARTING, LISTENING, CONNECT_STARTING, CONNECTING, SENDING, RECVING, OK, TIMEOUT)

MAKE_ENUM(NicType, DEVICE_NIC_TYPE, HOST_NIC_TYPE, DEVICE_VNIC_TYPE)

using FdHandle = void*;
class Socket {
public:
    Socket(
        SocketHandle socketHandle, IpAddress localIp, u32 listenPort, IpAddress remoteIp, const std::string& tag,
        SocketRole role, NicType nicType)
        : socketHandle(socketHandle),
          localIp(localIp),
          listenPort(listenPort),
          remoteIp(remoteIp),
          tag(tag),
          role(role),
          nicType(nicType)
    {}

    virtual ~Socket();

    virtual void Listen();
    virtual void Connect();
    SocketStatus GetStatus(u32 timeout = 0);

    virtual SocketRole GetRole() const { return role; }
    virtual IpAddress GetRemoteIp() { return remoteIp; }
    void Destroy();
    void Close();
    void StopListen();

    bool Send(const void* sendBuf, u32 size) const;
    bool Recv(void* recvBuf, u32 size) const;

    bool Listen(u32& port);
    bool ISend(void* data, u64 size, u64& compSize) const;

    HcclResult ISendWithHeart(void* data, u64 size, u64& compSize) const;
    HcclResult IRecvWithHeart(void* data, u64 size, u64& compSize) const;

    bool IsListen() const { return isListening; }

    SocketStatus GetAsyncStatus();

    void ListenAsync();
    void ConnectAsync();
    void SendAsync(const void* sendBuf, u32 size);
    void RecvAsync(u8* recvBuf, u32 size);

    FdHandle GetFdHandle() const // will be used in hccp QP connecting
    {
        return fdHandle;
    }

    NicType GetNicType() const { return nicType; }

    IpAddress GetLocalIp() { return localIp; }

    u32 GetListenPort() const { return listenPort; }

    string Describe()
    {
        return StringFormat(
            "Socket[role=%s, localIp=%s, listenPort=%u, remoteIp=%s, tag=%s, nicType=%s]", role.Describe().c_str(),
            localIp.Describe().c_str(), listenPort, remoteIp.Describe().c_str(), tag.c_str(),
            nicType.Describe().c_str());
    }

private:
    SocketHandle socketHandle{nullptr}; // vnic/nic创建的handle，HCCP初始化返回的handle_
    IpAddress localIp;
    u32 listenPort{0};
    IpAddress remoteIp;
    const std::string tag;
    SocketRole role{SocketRole::CLIENT};
    FdHandle fdHandle{nullptr};
    SocketStatus socketStatus{SocketStatus::INIT};
    NicType nicType{NicType::INVALID};
    bool isConnected{false};
    bool isListening{false};
    bool isDestroyed{false};
    std::unique_ptr<SocketListenInfoT> listenInfo_{nullptr};

    std::chrono::steady_clock::time_point lastLogTime{}; // 抑制日志刷屏时间戳，刷新时可置空

    RequestHandle reqHandle{0};

    const void* sendDataBuff{nullptr}; // 发送缓冲区的起始地址，需要调用方保证内存生命周期
    unsigned long long sendSize{0};    // 调用Send接口入参，返回接口调用后实际发送的数据量
    u32 sendLeftSize{0};               // 发送缓冲区剩余待发送数据量
    u32 totalSendSize{0};              // 发送缓冲区已发送总数据量

    void* recvDataBuff{nullptr};    // 接受缓冲区的起始地址，需要调用方保证内存生命周期
    unsigned long long recvSize{0}; // 调用Recv接口入参，返回接口调用后实际接受的数据量
    u32 recvLeftSize{0};            // 接受缓冲区剩余待接受数据量
    u32 totalRecvSize{0};           // 接受缓冲区已接受总数据量

    void GetOneSocket();

    bool CheckStartRequestResult();
    bool CheckSendRequestResult();
    bool CheckRecvRequestResult();
    void PrintErrorSocketInfo();
};

} // namespace Hccl

#endif // HCCLV2_SOCKET_BASE_H
