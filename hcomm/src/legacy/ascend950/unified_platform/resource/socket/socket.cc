/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "socket.h"

#include "sal.h"
#include "socket_exception.h"
#include "network_api_exception.h"
#include "config_plf_log_v2.h"

namespace Hccl {

constexpr u32 MAX_TRANSFER_SIZE = 20 * 1024 * 1024;
constexpr u32 MAX_LOG_TIMEOUT_MS = 30000;
constexpr u32 ONE_MILLISECOND_OF_USLEEP = 1000;
constexpr u32 AUTO_LISTEN_PORT = 0;

Socket::~Socket()
{
    if (!isDestroyed) {
        DECTOR_TRY_CATCH("Socket", this->Destroy());
    }
}

void Socket::Listen()
{
    HCCL_INFO("[Socket::%s] listen start, listenPort[%u]", __func__, listenPort);
    HrtNetworkMode netMode = nicType == NicType::HOST_NIC_TYPE ? HrtNetworkMode::PEER : HrtNetworkMode::HDC;
    RaSocketListenParam param(socketHandle, listenPort, localIp);
    HrtRaSocketListenOneStart(param, netMode);
    isListening = true;
    socketStatus = SocketStatus::LISTENING;
}

bool Socket::Listen(u32& port)
{
    HCCL_INFO("[Socket::%s] trying to listen on port[%u]", __func__, port);
    HrtNetworkMode netMode = nicType == NicType::HOST_NIC_TYPE ? HrtNetworkMode::PEER : HrtNetworkMode::HDC;
    RaSocketListenParam param(socketHandle, port, localIp);
    bool ret = HrtRaSocketTryListenOneStart(param, netMode);
    CHK_PRT_RET(
        !ret,
        HCCL_WARNING(
            "[Socket::%s] socket[%s] listen unsuccessful, port[%u] is in use", __func__, Describe().c_str(), port),
        ret);

    port = port == AUTO_LISTEN_PORT ? param.port : port;
    listenPort = port;
    isListening = true;
    socketStatus = SocketStatus::LISTENING;

    PLF_CONFIG_INFO(PLF_CHANNEL, "[Socket] listen success, [%s].", Describe().c_str());
    return true;
}

void Socket::Connect()
{
    HCCL_INFO("socket role_ is %u, %s", static_cast<u32>(role), role.Describe().c_str());
    if (role == SocketRole::SERVER || socketStatus == SocketStatus::OK) {
        return;
    }

    RaSocketConnectParam param(socketHandle, remoteIp, listenPort, tag);
    HrtRaSocketConnectOne(param);
    HCCL_INFO("conn.tag %s", tag.c_str());

    socketStatus = SocketStatus::CONNECTING;
    PLF_CONFIG_INFO(PLF_CHANNEL, "[Socket] connect to remote, [%s].", this->Describe().c_str());
}

void Socket::PrintErrorSocketInfo()
{
    HCCL_ERROR("Socket::GetStatus failed.");
    HCCL_ERROR("Please check if env HCCL_SOCKET_IFNAME is set correctly, "
               "which can be verified by checking localIp and remoteIp in the socket info as follows.");
    HCCL_ERROR("%s", Describe().c_str());
}

SocketStatus Socket::GetStatus(u32 timeout)
{
    if (socketStatus == SocketStatus::OK) {
        HCCL_INFO("socketinfo.tag=%s status is OK, role=%s", tag.c_str(), role.Describe().c_str());
        return SocketStatus::OK;
    }

    RaSocketGetParam param(socketHandle, remoteIp, tag, fdHandle);
    RaSocketFdHandleParam result(nullptr, 0);
    TRY_CATCH_PROCESS_THROW(
        NetworkApiException, result = HrtRaBlockGetOneSocket(static_cast<u32>(role), param, timeout),
        "Socket::GetStatus failed", PrintErrorSocketInfo());

    fdHandle = result.fdHandle;

    // socket status:0 not connected 1:connected 2:connect timeout 3:connecting
    if (result.status == SOCKET_CONNECTED) {
        socketStatus = SocketStatus::OK;
        isConnected = true;
        PLF_CONFIG_INFO(PLF_CHANNEL, "[Socket] link established, [%s].", this->Describe().c_str());
    } else if (result.status == SOCKET_CONNECT_TIMEOUT) {
        socketStatus = SocketStatus::TIMEOUT;
    } else if (result.status == SOCKET_CONNECTING) {
        socketStatus = SocketStatus::CONNECTING;
    } else {
        socketStatus = SocketStatus::INIT;
    }
    HCCL_INFO(
        "socketinfo.tag=%s status=%s, role=%s", tag.c_str(), socketStatus.Describe().c_str(), role.Describe().c_str());
    return socketStatus;
}

bool Socket::Send(const void* sendBuf, u32 size) const
{
    HrtRaSocketBlockSend(fdHandle, sendBuf, size);
    return true;
}

bool Socket::Recv(void* recvBuf, u32 size) const
{
    HrtRaSocketBlockRecv(fdHandle, recvBuf, size);
    return true;
}

bool Socket::ISend(void* data, u64 size, u64& compSize) const
{
    return HrtRaSocketNonBlockSend(fdHandle, data, size, &compSize);
}

HcclResult Socket::ISendWithHeart(void* data, u64 size, u64& compSize) const
{
    return HrtRaSocketNonBlockSendHeart(fdHandle, data, size, &compSize);
}

HcclResult Socket::IRecvWithHeart(void* data, u64 size, u64& compSize) const
{
    return HrtRaSocketNonBlockRecvHeart(fdHandle, data, size, &compSize);
}

void Socket::Destroy()
{
    isDestroyed = true;
    EXCEPTION_CATCH(StopListen(), return);
    EXCEPTION_CATCH(Close(), return);
}

void Socket::Close()
{
    if (isConnected) {
        RaSocketCloseParam param(socketHandle, fdHandle);
        HrtRaSocketCloseOne(param);
        isConnected = false;
        PLF_CONFIG_INFO(PLF_CHANNEL, "[Socket] close, [%s].", this->Describe().c_str());
    }
}

void Socket::StopListen()
{
    if (isListening) {
        RaSocketListenParam param(socketHandle, listenPort, localIp);
        HrtRaSocketListenOneStop(param);
        isListening = false;
        PLF_CONFIG_INFO(PLF_CHANNEL, "[Socket] stop listen, [%s].", this->Describe().c_str());
    }
}

// 抑制日志刷屏，同一类型日志超时前只打印一次
inline bool CheckLogTime(std::chrono::steady_clock::time_point& lastTime)
{
    auto nowTime = std::chrono::steady_clock::now();
    if (nowTime - lastTime <= std::chrono::milliseconds(MAX_LOG_TIMEOUT_MS)) {
        return false;
    }

    lastTime = nowTime;
    return true;
}

inline void HandleSocketEAgain(RequestHandle lastReqHandle, std::chrono::steady_clock::time_point lastLogTime)
{
    if (CheckLogTime(lastLogTime)) {
        HCCL_WARNING(
            "[Socket][%s] reqhandle[%llu] get request result [SOCK_EAGAIN], sleep 1ms and retry.", __func__,
            lastReqHandle);
    }

    SaluSleep(ONE_MILLISECOND_OF_USLEEP);
}

bool Socket::CheckStartRequestResult()
{
    if (reqHandle == 0) {
        return true;
    }

    RequestHandle lastReqHandle = reqHandle;
    ReqHandleResult result = HrtRaGetAsyncReqResult(reqHandle);
    if (result == ReqHandleResult::NOT_COMPLETED) {
        if (CheckLogTime(lastLogTime)) {
            HCCL_INFO(
                "[Socket][%s] connect is not completed, reqHandle[%llu], [%s].", __func__, lastReqHandle,
                this->Describe().c_str());
        }

        return false;
    }

    // COMPLETED 表示调用接口成功，可以更新数据信息
    // SOCK_E_AGAIN 表示接口调用失败，需要重新调用接口
    // 其余结果为异常场景，抛出异常
    if (result == ReqHandleResult::COMPLETED) {
        lastLogTime = {};
        return true;
    } else if (result == ReqHandleResult::SOCK_E_AGAIN) {
        HandleSocketEAgain(lastReqHandle, lastLogTime);
    } else {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed, request handle[%llu] result[%s] is unexpected, [%s].", __func__, lastReqHandle,
            result.Describe().c_str(), this->Describe().c_str()));
    }

    if (socketStatus == SocketStatus::CONNECT_STARTING) {
        ConnectAsync();
    } else if (socketStatus == SocketStatus::LISTEN_STARTING) {
        ListenAsync();
    } else {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed, socket status[%s] is not expected, [%s].", __func__, socketStatus.Describe().c_str(),
            this->Describe().c_str()));
    }

    return false;
}

bool Socket::CheckSendRequestResult()
{
    if (reqHandle == 0) {
        return true;
    }

    RequestHandle lastReqHandle = reqHandle;
    ReqHandleResult result = HrtRaGetAsyncReqResult(reqHandle);
    if (result == ReqHandleResult::NOT_COMPLETED) {
        if (CheckLogTime(lastLogTime)) {
            HCCL_INFO(
                "[Socket][%s] reqHandle[%llu] send is not completed, [%s].", __func__, lastReqHandle,
                this->Describe().c_str());
        }

        return false;
    }

    if (sendSize > sendLeftSize) {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] prev send request handle[%llu] failed, "
            "send size[%u] is greater than expected[%u], [%s].",
            __func__, lastReqHandle, sendSize, sendLeftSize, this->Describe().c_str()));
    }

    // COMPLETED 表示调用接口成功，可以更新数据信息
    // SOCK_E_AGAIN 表示接口调用失败，需要重新调用接口
    // 其余结果为异常场景，抛出异常
    if (result == ReqHandleResult::COMPLETED) {
        totalSendSize += sendSize;
        sendLeftSize -= sendSize;
    } else if (result == ReqHandleResult::SOCK_E_AGAIN) {
        HandleSocketEAgain(lastReqHandle, lastLogTime);
    } else {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed, request handle[%llu] result[%s] is unexpected, [%s].", __func__, lastReqHandle,
            result.Describe().c_str(), this->Describe().c_str()));
    }

    // 如果仍有数据未处理，则继续调用接口
    if (sendLeftSize != 0) {
        sendSize = 0;
        reqHandle = HrtRaSocketSendAsync(
            fdHandle, ReinterpretAs<void*>(ReinterpretAs<uintptr_t>(sendDataBuff) + totalSendSize), sendLeftSize,
            sendSize);

        if (CheckLogTime(lastLogTime)) {
            HCCL_INFO(
                "[Socket][%s] reqHandle[%llu] need to retry send, start to send left size[%u], [%s].", __func__,
                reqHandle, sendLeftSize, this->Describe().c_str());
        }

        return false;
    }

    lastLogTime = {};
    HCCL_INFO(
        "[Socket][%s] pre send request[%llu] is completed, total send size[%u], [%s]", __func__, lastReqHandle,
        totalSendSize, this->Describe().c_str());
    return true;
}

bool Socket::CheckRecvRequestResult()
{
    if (reqHandle == 0) {
        return true;
    }

    RequestHandle lastReqHandle = reqHandle;
    ReqHandleResult result = HrtRaGetAsyncReqResult(reqHandle);
    if (result == ReqHandleResult::NOT_COMPLETED) {
        if (CheckLogTime(lastLogTime)) {
            HCCL_INFO(
                "[Socket][%s] reqHandle[%llu] recv is not completed, [%s].", __func__, lastReqHandle,
                this->Describe().c_str());
        }

        return false;
    }

    if (recvSize > recvLeftSize) {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] prev recv request handle[%llu] failed, "
            "recv size[%llu] is greater than expected[%u], [%s].",
            __func__, lastReqHandle, recvSize, recvLeftSize, this->Describe().c_str()));
    }

    // COMPLETED 表示调用接口成功，可以更新数据信息
    // SOCK_E_AGAIN 表示接口调用失败，需要重新调用接口
    // 其余结果为异常场景，抛出异常
    if (result == ReqHandleResult::COMPLETED) {
        PLF_CONFIG_INFO(
            PLF_CHANNEL, "[Socket] recv request[%llu] completed, real recvSize[%llu], totalRecvSize[%llu], [%s].",
            lastReqHandle, recvSize, static_cast<unsigned long long>(totalRecvSize) + recvSize,
            this->Describe().c_str());
        totalRecvSize += recvSize;
        recvLeftSize -= recvSize;
    } else if (result == ReqHandleResult::SOCK_E_AGAIN) {
        HandleSocketEAgain(lastReqHandle, lastLogTime);
    } else {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed, request handle[%llu] result[%s] is unexpected, [%s].", __func__, lastReqHandle,
            result.Describe().c_str(), this->Describe().c_str()));
    }

    // 如果仍有数据未处理，则继续调用接口
    if (recvLeftSize != 0) {
        recvSize = 0;
        reqHandle = HrtRaSocketRecvAsync(
            fdHandle, ReinterpretAs<void*>(ReinterpretAs<uintptr_t>(recvDataBuff) + totalRecvSize), recvLeftSize,
            recvSize);

        if (CheckLogTime(lastLogTime)) {
            HCCL_INFO(
                "[Socket][%s] reqHandle[%llu] need to retry recv, start to recv left size[%u], [%s].", __func__,
                reqHandle, recvLeftSize, this->Describe().c_str());
        }

        return false;
    }

    lastLogTime = {};
    HCCL_INFO(
        "[Socket][%s] pre recv request[%llu] is completed, total recv size[%u], [%s].", __func__, lastReqHandle,
        totalRecvSize, this->Describe().c_str());
    return true;
}

SocketStatus Socket::GetAsyncStatus()
{
    switch (socketStatus) {
        case SocketStatus::OK:
        case SocketStatus::TIMEOUT:
        case SocketStatus::LISTENING:
            break;
        case SocketStatus::LISTEN_STARTING: {
            if (CheckStartRequestResult()) {
                isListening = true;
                socketStatus = SocketStatus::LISTENING;
            }
            break;
        }
        case SocketStatus::SENDING: {
            if (CheckSendRequestResult()) {
                socketStatus = SocketStatus::OK;
            }
            break;
        }
        case SocketStatus::RECVING: {
            if (CheckRecvRequestResult()) {
                socketStatus = SocketStatus::OK;
            }

            break;
        }
        case SocketStatus::CONNECT_STARTING: {
            if (CheckStartRequestResult()) {
                GetOneSocket();
            }
            break;
        }
        case SocketStatus::INIT:
        case SocketStatus::CONNECTING:
        default:
            GetOneSocket();
    }

    return socketStatus;
}

void Socket::GetOneSocket()
{
    RaSocketGetParam param(socketHandle, remoteIp, tag, fdHandle);

    RaSocketFdHandleParam fdHandleParam(nullptr, 0);
    EXCEPTION_CATCH(fdHandleParam = RaGetOneSocket(static_cast<u32>(role), param), return);
    // socket status:0 not connected 1:connected 2:connect timeout 3:connecting
    if (fdHandleParam.status == SOCKET_CONNECTED) {
        // sockete 准备好时，可以读取信息
        fdHandle = fdHandleParam.fdHandle;
        socketStatus = SocketStatus::OK;
        isConnected = true;
        lastLogTime = {};
    } else if (fdHandleParam.status == SOCKET_CONNECT_TIMEOUT) {
        socketStatus = SocketStatus::TIMEOUT;
    } else if (fdHandleParam.status == SOCKET_CONNECTING) {
        if (CheckLogTime(lastLogTime)) {
            HCCL_INFO("[Socket][%s] socket is connecting, [%s]", __func__, this->Describe().c_str());
        }

        socketStatus = SocketStatus::CONNECTING;
    } else {
        socketStatus = SocketStatus::INIT;
    }
}

void Socket::ListenAsync()
{
    listenInfo_ = std::make_unique<SocketListenInfoT>();
    listenInfo_->socketHandle = socketHandle;
    listenInfo_->port = listenPort;
    reqHandle = RaSocketListenOneStartAsync(listenInfo_.get());
    socketStatus = SocketStatus::LISTEN_STARTING;
}

void Socket::ConnectAsync()
{
    if (role == SocketRole::SERVER || socketStatus == SocketStatus::OK) {
        return;
    }

    RaSocketConnectParam param(socketHandle, remoteIp, listenPort, tag);
    reqHandle = RaSocketConnectOneAsync(param);
    HCCL_INFO("[Socket][%s] conn.tag %s", __func__, tag.c_str());

    socketStatus = SocketStatus::CONNECT_STARTING;
}

void Socket::SendAsync(const void* sendBuf, u32 size)
{
    if (!sendBuf) {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed to send, "
            "sendBuf is nullptr, [%s].",
            __func__, this->Describe().c_str()));
    }

    if (size == 0) {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed to send, "
            "size is 0, [%s].",
            __func__, this->Describe().c_str()));
    }

    if (size > MAX_TRANSFER_SIZE) {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed to send, "
            "size[%u] is greater than max size[%u], [%s]",
            __func__, size, MAX_TRANSFER_SIZE, this->Describe().c_str()));
    }

    if (socketStatus != SocketStatus::OK) {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed to send, "
            "status[%s] is not ok, [%s].",
            __func__, socketStatus.Describe().c_str(), this->Describe().c_str()));
    }

    sendSize = 0;
    totalSendSize = 0;
    sendLeftSize = size;
    sendDataBuff = sendBuf;

    reqHandle = HrtRaSocketSendAsync(fdHandle, sendBuf, sendLeftSize, sendSize);
    HCCL_INFO(
        "[Socket][%s] reqHandle[%llu] start to send size[%u], [%s].", __func__, reqHandle, sendLeftSize,
        this->Describe().c_str());

    lastLogTime = {};
    socketStatus = SocketStatus::SENDING;
}

void Socket::RecvAsync(u8* recvBuf, u32 size)
{
    if (!recvBuf) {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed to recv, "
            "recvBuf is nullptr, [%s].",
            __func__, this->Describe().c_str()));
    }

    if (size == 0) {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed to recv, "
            "size is 0, [%s].",
            __func__, this->Describe().c_str()));
    }

    if (size > MAX_TRANSFER_SIZE) {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed to recv, "
            "size[%u] is greater than max size[%u], [%s].",
            __func__, size, MAX_TRANSFER_SIZE, this->Describe().c_str()));
    }

    if (socketStatus != SocketStatus::OK) {
        THROW<SocketException>(StringFormat(
            "[Socket][%s] failed to recv, "
            "status[%s] is not ok, [%s].",
            __func__, socketStatus.Describe().c_str(), this->Describe().c_str()));
    }

    recvSize = 0;
    totalRecvSize = 0;
    recvLeftSize = size;
    recvDataBuff = static_cast<void*>(recvBuf);

    reqHandle = HrtRaSocketRecvAsync(fdHandle, recvDataBuff, recvLeftSize, recvSize);
    HCCL_INFO(
        "[Socket][%s] reqHandle[%llu] start to recv size[%u], [%s].", __func__, reqHandle, recvLeftSize,
        this->Describe().c_str());

    lastLogTime = {};
    socketStatus = SocketStatus::RECVING;
}

} // namespace Hccl
