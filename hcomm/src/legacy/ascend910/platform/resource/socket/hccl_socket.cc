/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlhal_function.h"
#include "externalinput_pub.h"
#include "adapter_hccp.h"
#include "network_manager_pub.h"
#include "sal_pub.h"
#include "hccl_net_dev_defs.h"
#include "hccl_network.h"
#include "network/hccp_common.h"
#include "adapter_error_manager_pub.h"
#include "hccl_socket.h"

namespace hccl {
constexpr u32 MAX_MSG_STR_LEN = 2 * 1024;

HcclSocket::HcclSocket(
    const std::string& tag, HcclNetDevCtx netDevCtx, const HcclIpAddress& remoteIp, u32 remotePort,
    HcclSocketRole localRole)
    : tag_(tag),
      netDevCtx_(netDevCtx),
      remoteIp_(remoteIp),
      remotePort_(remotePort),
      localRole_(localRole),
      status_(HcclSocketStatus::SOCKET_INIT),
      fdHandle_(nullptr)
{}

HcclSocket::HcclSocket(HcclNetDevCtx netDevCtx, u32 localPort)
    : netDevCtx_(netDevCtx),
      localPort_(localPort),
      localRole_(HcclSocketRole::SOCKET_ROLE_RESERVED),
      status_(HcclSocketStatus::SOCKET_INIT),
      fdHandle_(nullptr)
{}

HcclSocket::~HcclSocket() { DeInit(); }

HcclResult HcclSocket::Init()
{
    CHK_PTR_NULL(netDevCtx_);
    socketType_ = (static_cast<hccl::NetDevContext*>(netDevCtx_))->GetNicType();
    localDevicePhyId_ = (static_cast<hccl::NetDevContext*>(netDevCtx_))->GetPhyId();
    localDeviceLogicId_ = (static_cast<hccl::NetDevContext*>(netDevCtx_))->GetLogicId();
    localIp_ = (static_cast<hccl::NetDevContext*>(netDevCtx_))->GetLocalIp();
    backupIp_ = (static_cast<hccl::NetDevContext*>(netDevCtx_))->GetBackupIp();

    // 默认场景下，只有VNIC使用强制断链，其它场景还是走优雅断链
    forceClose_ = (socketType_ == NicType::VNIC_TYPE);

    return HCCL_SUCCESS;
}

HcclResult HcclSocket::DeInit()
{
    Close();
    if (listened_) {
        if (socketType_ == NicType::VNIC_TYPE) {
            CHK_RET(NetworkManager::GetInstance(localDeviceLogicId_).StopVnic(localIp_, localPort_));
        } else if (socketType_ == NicType::DEVICE_NIC_TYPE) {
            CHK_RET(NetworkManager::GetInstance(localDeviceLogicId_).StopNic(localIp_, localPort_));
        } else {
            CHK_RET(NetworkManager::GetInstance(localDeviceLogicId_)
                        .StopHostNetAndListen(nicSocketHandle_, localIp_, localPort_));
        }

        listened_ = false;
        HCCL_INFO(
            "[HcclSocket][DeInit] device[%d] stops listen on ip[%s], port[%u] success, socketType[%u].",
            localDeviceLogicId_, localIp_.GetReadableAddress(), localPort_, socketType_);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclSocket::ListenHostNet(HcclResult& ret, std::string& errormessage, u32& port)
{
    bool rdmaFlag = !GetExternalInputHcclIsTcpMode();
    u32 proto = 0;
    SocketHandle hostSocketHandle;
    HcclNetDevGetProtoType(netDevCtx_, proto);
    if (rdmaFlag && proto == HCCL_PROTO_TYPE_ROCE) {
        rdmaFlag = true;
    } else {
        rdmaFlag = false;
    }
    ret = NetworkManager::GetInstance(localDeviceLogicId_)
              .StartHostNetAndListen(localIp_, hostSocketHandle, port, rdmaFlag);
    errormessage = "The IP address " + std::string(localIp_.GetReadableIP()) + " and port " + std::to_string(port)
                   + " have already been bound.";
    RPT_INPUT_ERR(
        ret == HCCL_E_UNAVAIL, "EI0019", std::vector<std::string>({"reason"}),
        std::vector<std::string>({errormessage}));
    return HCCL_SUCCESS;
}

HcclResult HcclSocket::Listen()
{
    CHK_PRT_RET(
        localPort_ == HCCL_INVALID_PORT,
        HCCL_ERROR("[HcclSocket][Listen]No port is set, please listen with a valid port."), HCCL_E_INTERNAL);
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());

    HcclResult ret = HCCL_E_RESERVED;
    std::string errormessage = "";
    if (socketType_ == NicType::VNIC_TYPE) {
        ret = NetworkManager::GetInstance(localDeviceLogicId_).StartVnic(localIp_, localPort_);
        errormessage = "The IP address " + std::string(localIp_.GetReadableIP()) + " and port "
                       + std::to_string(localPort_) + " have already been bound.";
        RPT_INPUT_ERR(
            ret == HCCL_E_UNAVAIL, "EI0020", std::vector<std::string>({"reason"}),
            std::vector<std::string>({errormessage}));
    } else if (socketType_ == NicType::DEVICE_NIC_TYPE) {
        bool rdmaFlag = !GetExternalInputHcclIsTcpMode();
        HCCL_DEBUG(
            "[%s]StartNic localDeviceLogicId_[%d], localIp_[%s], localPort_[%u], rdmaFlag[%d], "
            "socketType_[%d], backupIp_[%s]",
            __func__, localDeviceLogicId_, localIp_.GetReadableIP(), localPort_, rdmaFlag, socketType_,
            backupIp_.GetReadableIP());
        // 如果是backup，传入额外的rdev信息
        ret = NetworkManager::GetInstance(localDeviceLogicId_).StartNic(localIp_, localPort_, rdmaFlag, backupIp_);
        errormessage = "The IP address " + std::string(localIp_.GetReadableIP()) + " and port "
                       + std::to_string(localPort_) + " have already been bound.";
        RPT_INPUT_ERR(
            ret == HCCL_E_UNAVAIL, "EI0020", std::vector<std::string>({"reason"}),
            std::vector<std::string>({errormessage}));
    } else {
        CHK_RET(ListenHostNet(ret, errormessage, localPort_));
    }
    std::stringstream tmpMsgstream;
    tmpMsgstream
        << ((socketType_ == NicType::HOST_NIC_TYPE) ? ("[" + LOG_KEYWORDS_INIT_CHANNEL + "]") :
                                                      ("[" + LOG_KEYWORDS_INIT_GROUP + "]"))
        << "[" << LOG_KEYWORDS_RANKTABLE_DETECT << "]";
    std::string errmsg = tmpMsgstream.str();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "%s socket type[%u], listen on ip[%s] and specific port[%u] fail. "
            "Please check the port status and whether the port is being used by other process.",
            errmsg.c_str(), socketType_, localIp_.GetReadableAddress(), localPort_),
        ret);

    CHK_RET(GetNicSocketHandle());

    listened_ = true;
    HCCL_INFO(
        "[HcclSocket][Listen] device[%d] listens on ip[%s] port[%u] success, socketType[%u].", localDeviceLogicId_,
        localIp_.GetReadableAddress(), localPort_, socketType_);

    return HCCL_SUCCESS;
}

HcclResult HcclSocket::Listen(u32 port)
{
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());

    HcclResult ret = HCCL_E_RESERVED;
    std::string errormessage = "";
    HCCL_INFO("[HcclSocket][Listen] device[%d] trying to listen on port[%u]", localDeviceLogicId_, port);
    if (socketType_ == NicType::VNIC_TYPE) {
        ret = NetworkManager::GetInstance(localDeviceLogicId_).StartVnic(localIp_, port);
        errormessage = "The IP address " + std::string(localIp_.GetReadableIP()) + " and port " + std::to_string(port)
                       + " have already been bound.";
        RPT_INPUT_ERR(
            ret == HCCL_E_UNAVAIL, "EI0020", std::vector<std::string>({"reason"}),
            std::vector<std::string>({errormessage}));
    } else if (socketType_ == NicType::DEVICE_NIC_TYPE) {
        bool rdmaFlag = false;
        HCCL_DEBUG(
            "[%s]StartNic localDeviceLogicId_[%d], localIp_[%s], localPort_[%u], rdmaFlag[%d], "
            "socketType_[%d], backupIp_[%s]",
            __func__, localDeviceLogicId_, localIp_.GetReadableIP(), port, rdmaFlag, socketType_,
            backupIp_.GetReadableIP());
        // 如果是backup，传入额外的rdev信息
        ret = NetworkManager::GetInstance(localDeviceLogicId_).StartNic(localIp_, port, rdmaFlag, backupIp_);
        errormessage = "The IP address " + std::string(localIp_.GetReadableIP()) + " and port " + std::to_string(port)
                       + " have already been bound.";
        RPT_INPUT_ERR(
            ret == HCCL_E_UNAVAIL, "EI0020", std::vector<std::string>({"reason"}),
            std::vector<std::string>({errormessage}));
    } else {
        CHK_RET(ListenHostNet(ret, errormessage, port));
    }
    std::stringstream tmpMsgstream;
    tmpMsgstream
        << ((socketType_ == NicType::HOST_NIC_TYPE) ? ("[" + LOG_KEYWORDS_INIT_CHANNEL + "]") :
                                                      ("[" + LOG_KEYWORDS_INIT_GROUP + "]"))
        << "[" << LOG_KEYWORDS_RANKTABLE_DETECT << "]";
    std::string errmsg = tmpMsgstream.str();
    CHK_PRT_RET(
        ret == HCCL_E_UNAVAIL,
        HCCL_INFO(
            "%s socket type[%u], Could not listen on IP [%s] and port [%u], port already in use.", errmsg.c_str(),
            socketType_, localIp_.GetReadableAddress(), port),
        ret);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "%s socket type[%u], listen on ip[%s] and port[%u] fail,.", errmsg.c_str(), socketType_,
            localIp_.GetReadableAddress(), port),
        ret);

    CHK_RET(GetNicSocketHandle());

    localPort_ = port;
    listened_ = true;
    HCCL_INFO(
        "[HcclSocket][Listen] device[%d] listens on ip[%s] port[%u] success, socketType[%u].", localDeviceLogicId_,
        localIp_.GetReadableAddress(), localPort_, socketType_);

    return HCCL_SUCCESS;
}
HcclResult HcclSocket::AddWhiteList(std::vector<SocketWlistInfo>& wlistInfoVec)
{
    if (listened_ == false) {
        return HCCL_E_NOT_FOUND;
    }

    std::vector<struct SocketWlistInfoT> wlistInfosVec;
    for (auto remote : wlistInfoVec) {
        struct SocketWlistInfoT wlistInfo = {};
        wlistInfo.connLimit = remote.connLimit;
        wlistInfo.remoteIp.addr = remote.remoteIp.addr;
        wlistInfo.remoteIp.addr6 = remote.remoteIp.addr6;
        s32 sRet = memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), remote.tag, sizeof(remote.tag));
        if (sRet != EOK) {
            HCCL_ERROR("[Add][SocketWhiteList]memory copy failed. errorno[%d]", sRet);
            return HCCL_E_MEMORY;
        }
        wlistInfosVec.push_back(wlistInfo);
    }

    CHK_RET(hrtRaSocketWhiteListAdd(nicSocketHandle_, wlistInfosVec.data(), wlistInfosVec.size()));

    return HCCL_SUCCESS;
}

HcclResult HcclSocket::DelWhiteList(std::vector<SocketWlistInfo>& wlistInfoVec)
{
    if (listened_ == false) {
        return HCCL_E_NOT_FOUND;
    }

    std::vector<struct SocketWlistInfoT> wlistInfosVec;
    for (auto remote : wlistInfoVec) {
        struct SocketWlistInfoT wlistInfo = {};
        wlistInfo.connLimit = remote.connLimit;
        wlistInfo.remoteIp.addr = remote.remoteIp.addr;
        wlistInfo.remoteIp.addr6 = remote.remoteIp.addr6;
        s32 sRet = memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), remote.tag, sizeof(remote.tag));
        if (sRet != EOK) {
            HCCL_ERROR("[Delete][SocketWhiteList]memory copy failed. errorno[%d]", sRet);
            return HCCL_E_MEMORY;
        }
        wlistInfosVec.push_back(wlistInfo);
    }

    CHK_RET(hrtRaSocketWhiteListDel(nicSocketHandle_, wlistInfosVec.data(), wlistInfosVec.size()));

    return HCCL_SUCCESS;
}

HcclResult HcclSocket::Connect()
{
    if (status_ != HcclSocketStatus::SOCKET_INIT) {
        HCCL_ERROR("[Connect]socket status[%d] is not SOCKET_INIT, can not connect", status_);
        return HCCL_E_TCP_CONNECT;
    }

    CHK_RET(GetNicSocketHandle());

    // 作为客户端时, 向远端发起 Connect 请求; 作为服务端时, 暂什么也不做
    if (localRole_ == HcclSocketRole::SOCKET_ROLE_CLIENT) {
        SocketConnectInfoT connectInfo{};
        connectInfo.remoteIp.addr = remoteIp_.GetBinaryAddress().addr;
        connectInfo.remoteIp.addr6 = remoteIp_.GetBinaryAddress().addr6;
        connectInfo.socketHandle = nicSocketHandle_;
        connectInfo.port = remotePort_;
        CHK_SAFETY_FUNC_RET(strcpy_s(connectInfo.tag, SOCK_CONN_TAG_SIZE, tag_.c_str()));

        HCCL_INFO(
            "[Connect] localIp[%s], remoteIp[%s], socketHandle[%lu], tag[%s], port[%u]", localIp_.GetReadableAddress(),
            remoteIp_.GetReadableAddress(), nicSocketHandle_, connectInfo.tag, remotePort_);

        HcclResult ret = hrtRaSocketBatchConnect(&connectInfo, 1, MAX_VALUE_U32, [this]() -> bool {
            return this->GetStopFlag();
        });
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Connect] call ra socket connect failed. errorno[%d]", ret), ret);

        status_ = HcclSocketStatus::SOCKET_CONNECTING;
    }

    return HCCL_SUCCESS;
}

void HcclSocket::Close()
{
    HCCL_INFO(
        "[Close] localIp[%s], remoteIp[%s], socketHandle[%lu], tag[%s], port[%u] status[%d]",
        localIp_.GetReadableAddress(), remoteIp_.GetReadableAddress(), nicSocketHandle_, tag_.c_str(), remotePort_,
        status_);
    // 若socket处于超时状态，调用abort接口终止连接请求
    if (status_ == HcclSocketStatus::SOCKET_TIMEOUT) {
        // 刷新status，防止重复调用abort接口
        status_ = HcclSocketStatus::SOCKET_ERROR;
        bool isSupportRaSocketAbort = false;
        (void)IsSupportRaSocketAbort(isSupportRaSocketAbort);
        // 作为客户端时, 终止向远端发起的 connect 请求; 作为服务端时, 暂什么也不做
        if (isSupportRaSocketAbort && localRole_ == HcclSocketRole::SOCKET_ROLE_CLIENT && !remoteIp_.IsInvalid()
            && nicSocketHandle_) {
            SocketConnectInfoT connectInfo{};
            connectInfo.remoteIp.addr = remoteIp_.GetBinaryAddress().addr;
            connectInfo.remoteIp.addr6 = remoteIp_.GetBinaryAddress().addr6;
            connectInfo.socketHandle = nicSocketHandle_;
            connectInfo.port = remotePort_;
            auto cpyRet = strcpy_s(connectInfo.tag, SOCK_CONN_TAG_SIZE, tag_.c_str());
            if (cpyRet != EOK) {
                HCCL_ERROR("[Abort] strcpy_s failed, ret[%d], tag[%s]", cpyRet, tag_.c_str());
                connectInfo.tag[0] = '\0';
            }

            HcclResult ret = hrtRaSocketNonBlockBatchAbort(&connectInfo, 1);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[Abort] call ra socket abort failed. ret[%d]", ret);
            } else {
                HCCL_RUN_INFO("[Abort] call ra socket abort success. tag[%s]", connectInfo.tag);
            }
        }
    } else {
        // socket处于其他状态时调用close接口关闭socket
        if (fdHandle_ == nullptr) {
            HCCL_WARNING("[Close] socket's fdHandle is null, do not need close.");
            return;
        }
        SocketCloseInfoT closeInfo = {};
        closeInfo.socketHandle = nicSocketHandle_;
        closeInfo.fdHandle = fdHandle_;
        closeInfo.disuseLinger = static_cast<s32>(forceClose_);
        HCCL_DEBUG(
            "[HcclSocket][Close] socketType[%d] nicHandle[%p] fdHandle[%p] disuseLinger[%d]", socketType_,
            nicSocketHandle_, fdHandle_, closeInfo.disuseLinger);

        HcclResult sRet = hrtRaSocketBatchClose(&closeInfo, 1, 1);
        if (sRet != HCCL_SUCCESS) {
            HCCL_WARNING("[Close] errNo[0x%016llx] server socket batch close ret[%d] failed. not fatal", sRet);
        }

        fdHandle_ = nullptr;
    }

    return;
}

// 此接口用于DFX, 仅在 HcclSocketManager 中判断为连接异常时调用
void HcclSocket::SetStatus(HcclSocketStatus status)
{
    if (status != HcclSocketStatus::SOCKET_TIMEOUT && status != HcclSocketStatus::SOCKET_ERROR) {
        HCCL_WARNING("[Set]]Status] Only support set SOCKET_TIMEOUT or SOCKET_ERROR, status[%d]", status);
        return;
    }
    status_ = status;
}

HcclSocketStatus HcclSocket::ConvertRaSocketStatus(int raStatus)
{
    HcclSocketStatus status = HcclSocketStatus::SOCKET_INIT;
    /**< socket status:0 not connected 1:connected 2:connect timeout 3:connecting */
    switch (raStatus) {
        case 0: // 0 not connected
            status = HcclSocketStatus::SOCKET_INIT;
            break;
        case 1: // 1:connected
            status = HcclSocketStatus::SOCKET_OK;
            break;
        case 2: // 2:connect timeout
            status = HcclSocketStatus::SOCKET_TIMEOUT;
            break;
        case 3: // 3:connecting
            status = HcclSocketStatus::SOCKET_CONNECTING;
            break;
        default:
            status = HcclSocketStatus::SOCKET_ERROR;
            break;
    }
    return status;
}

HcclSocketStatus HcclSocket::GetStatus()
{
    if (status_ == HcclSocketStatus::SOCKET_OK || status_ == HcclSocketStatus::SOCKET_TIMEOUT
        || status_ == HcclSocketStatus::SOCKET_ERROR) {
        HCCL_DEBUG("[Get][Status]socket status is [%d].", status_);
        return status_;
    }

    if (GetNicSocketHandle()) {
        return HcclSocketStatus::SOCKET_INIT;
    }

    // 疑问: Listen Socket 会是什么状态？

    SocketInfoT socketInfo{};
    socketInfo.remoteIp.addr = remoteIp_.GetBinaryAddress().addr;
    socketInfo.remoteIp.addr6 = remoteIp_.GetBinaryAddress().addr6;
    socketInfo.socketHandle = nicSocketHandle_;
    s32 ret = strcpy_s(socketInfo.tag, SOCK_CONN_TAG_SIZE, tag_.c_str());
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[Get][Status]strcpy_s failed. ret[%u]", ret), HcclSocketStatus::SOCKET_ERROR);

    u32 connectedNum = 0;
    s32 sockRet = hrtRaGetSockets(static_cast<u32>(localRole_), &socketInfo, 1, &connectedNum);
    if ((connectedNum == 0 && sockRet == 0) || (sockRet == SOCK_EAGAIN)) {
        return HcclSocketStatus::SOCKET_CONNECTING;
    } else if (sockRet != 0) {
        HCCL_ERROR("[Get][Status]get rasocket error. role[%u] sockRet[%d] ", localRole_, sockRet);
        return HcclSocketStatus::SOCKET_ERROR;
    } else {
        if (connectedNum == 1) {
            status_ = ConvertRaSocketStatus(socketInfo.status);
            fdHandle_ = socketInfo.fdHandle;
            HCCL_INFO("[Get][Status]status_[%u] ", status_);
            return status_;
        } else {
            HCCL_ERROR("[Get][Status]total Sockets[%u], more than needed num[1]!", connectedNum);
            return HcclSocketStatus::SOCKET_ERROR;
        }
    }
}

HcclResult HcclSocket::Accept(const std::string& tag, std::shared_ptr<HcclSocket>& socket, u32 acceptTimeOut)
{
    if (listened_ == false) {
        HCCL_ERROR("[Accept]socket no listen, can not accepted.");
        return HCCL_E_PARA;
    }

    EXCEPTION_CATCH(
        (socket = std::make_shared<HcclSocket>(tag, netDevCtx_, remoteIp_, 0, HcclSocketRole::SOCKET_ROLE_SERVER)),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(socket);
    CHK_RET(socket->Init());

    HCCL_INFO(
        "[Accept]localIp[%s], remoteIp[%s], socketHandle[%p], tag[%s]", localIp_.GetReadableAddress(),
        remoteIp_.GetReadableAddress(), nicSocketHandle_, tag.c_str());

    s32 acceptTimeOutTmp = static_cast<s32>(acceptTimeOut);
    s32 timer = (acceptTimeOutTmp > 0 && acceptTimeOutTmp < GetExternalInputHcclLinkTimeOut()) ?
                    acceptTimeOutTmp :
                    GetExternalInputHcclLinkTimeOut();

    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(timer);
    u32 count = 0;

    while (1) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            if (acceptTimeOutTmp != 0) {
                HCCL_WARNING(
                    "[Get][Connection]topo exchange server get socket timeout, waiting continues! "
                    "timeout[%d s]",
                    timer);
            } else {
                HCCL_ERROR("[Get][Connection]topo exchange server get socket timeout! timeout[%d s]", timer);
            }
            return HCCL_E_TIMEOUT;
        }

        HcclSocketStatus status = socket->GetStatus();
        if (status == HcclSocketStatus::SOCKET_OK) {
            HCCL_DEBUG(
                "[Accept]socket is established. localIp[%s], remoteIp[%s]", socket->GetLocalIp().GetReadableIP(),
                socket->GetRemoteIp().GetReadableIP());
            return HCCL_SUCCESS;
        } else if (status == HcclSocketStatus::SOCKET_CONNECTING) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            // 日志过滤, 50次才打印一次
            if (count % 50 == 0) {
                HCCL_DEBUG("[Wait][LinkEstablish]socket is connecting ");
            }
            count++;
            continue;
        } else if (status == HcclSocketStatus::SOCKET_TIMEOUT) {
            return HCCL_E_TIMEOUT;
        } else {
            HCCL_ERROR("[Accept]get socket fail");
            return HCCL_E_TCP_CONNECT;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclSocket::Send(const void* data, u64 size)
{
    CHK_PTR_NULL(data);
    CHK_PTR_NULL(fdHandle_);
    CHK_RET(hrtRaSocketBlockSend(fdHandle_, data, size, [this]() -> bool {
        return this->GetStopFlag();
    }));
    return HCCL_SUCCESS;
}

HcclResult HcclSocket::Recv(void* recvBuf, u32 recvBufLen, u32 timeout)
{
    CHK_PTR_NULL(fdHandle_);
    CHK_PTR_NULL(recvBuf);
    CHK_RET(hrtRaSocketBlockRecv(
        fdHandle_, recvBuf, recvBufLen,
        [this]() -> bool {
            return this->GetStopFlag();
        },
        timeout));
    return HCCL_SUCCESS;
}

HcclResult HcclSocket::Send(const std::string& sendMsg)
{
    CHK_PTR_NULL(fdHandle_);
    u32 msgLen = sendMsg.length();
    u8 buff[MAX_MSG_STR_LEN] = {0};
    s32 sRet = strcpy_s(reinterpret_cast<char*>(buff), MAX_MSG_STR_LEN, sendMsg.c_str());
    if (sRet != 0) {
        HCCL_ERROR("[Send] Block send message length[%u] is illegal", msgLen);
        return HCCL_E_PARA;
    }

    // 与 HcclSocket::Recv(std::string &recvMsg) 对应, 发送的消息长度和接收的消息长度一致, 才能保证正常Recv
    CHK_RET(hrtRaSocketBlockSend(fdHandle_, buff, MAX_MSG_STR_LEN, [this]() -> bool {
        return this->GetStopFlag();
    }));
    return HCCL_SUCCESS;
}

HcclResult HcclSocket::Recv(std::string& recvMsg, u32 timeout)
{
    CHK_PTR_NULL(fdHandle_);
    recvMsg.clear();
    u8 recvBuf[MAX_MSG_STR_LEN] = {0};
    CHK_RET(hrtRaSocketBlockRecv(
        fdHandle_, reinterpret_cast<void*>(recvBuf), MAX_MSG_STR_LEN,
        [this]() -> bool {
            return this->GetStopFlag();
        },
        timeout));
    recvMsg.assign(reinterpret_cast<char*>(recvBuf));
    return HCCL_SUCCESS;
}

HcclResult HcclSocket::ISend(void* data, u64 size, u64& compSize)
{
    CHK_PTR_NULL(data);
    CHK_PTR_NULL(fdHandle_);
    if (sendStatus_ != 0)
        return HCCL_E_NETWORK;
    if (size > SOCKET_SEND_MAX_SIZE) {
        HCCL_ERROR(
            "[ISend]errNo[0x%016llx] ra socket send size is too large, "
            "data[%p], size[%llu Byte]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), data, size);
        return HCCL_E_PARA;
    }
    s32 ret = hrtRaSocketNonBlockSend(fdHandle_, data, size, &compSize);
    HCCL_DEBUG("[ISend]except size [%llu Byte], actual size [%llu Byte], ret[%d]", size, compSize, ret);

    if (ret != 0 && ret != SOCK_EAGAIN) {
        sendStatus_ = ret;
        HCCL_RUN_WARNING("[ISend]except size [%llu Byte], actual size [%llu Byte], ret[%d]", size, compSize, ret);
        return HCCL_E_NETWORK;
    }
    return HCCL_SUCCESS; // EAGAIN和success都要返回HCCL_SUCCESS
}

HcclResult HcclSocket::IRecv(void* recvBuf, u32 recvBufLen, u64& compSize)
{
    CHK_PTR_NULL(fdHandle_);
    CHK_PTR_NULL(recvBuf);
    if (recvStatus_ != 0)
        return HCCL_E_NETWORK;
    s32 ret = hrtRaSocketNonBlockRecv(fdHandle_, recvBuf, recvBufLen, &compSize);
    HCCL_DEBUG("[IRecv]except size [%u Byte], actual size [%u Byte], ret[%u]", recvBufLen, compSize, ret);

    if (ret != 0 && ret != SOCK_EAGAIN) {
        recvStatus_ = ret;
        HCCL_RUN_INFO("[IRecv]except size [%u Byte], actual size [%u Byte], ret[%u]", recvBufLen, compSize, ret);
        return HCCL_E_NETWORK;
    }
    return HCCL_SUCCESS; // EAGAIN和success都要返回HCCL_SUCCESS
}

HcclResult HcclSocket::SendAsync(const void* data, u64 size, u64* sentSize, void** reqHandle)
{
    CHK_PTR_NULL(fdHandle_);
    CHK_PTR_NULL(data);
    CHK_PTR_NULL(sentSize);
    CHK_PTR_NULL(reqHandle);
    CHK_PRT_RET(
        (size == 0) || (size > MAX_MSG_STR_LEN),
        HCCL_ERROR("[SendAsync]send size[%llu] is 0 or large than %u", size, MAX_MSG_STR_LEN), HCCL_E_PARA);

    s32 ret = hrtRaSocketSendAsync(fdHandle_, data, size, sentSize, reqHandle);
    if (ret == 0) {
        return HCCL_SUCCESS;
    }

    if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    }
    HCCL_ERROR("[SendAsync]RaSocketSendAsync failed, data[%p] size[%llu] ret[%d]", data, size, ret);
    return HCCL_E_NETWORK;
}

HcclResult HcclSocket::RecvAsync(void* recvBuf, u64 recvBufLen, u64* receivedSize, void** reqHandle)
{
    CHK_PTR_NULL(fdHandle_);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(receivedSize);
    CHK_PTR_NULL(reqHandle);
    CHK_PRT_RET(recvBufLen == 0, HCCL_ERROR("[RecvAsync]recvBufLen is 0"), HCCL_E_PARA);

    s32 ret = hrtRaSocketRecvAsync(fdHandle_, recvBuf, recvBufLen, receivedSize, reqHandle);
    if (ret == 0) {
        return HCCL_SUCCESS;
    }

    if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    }
    HCCL_ERROR("[RecvAsync]RaSocketRecvAsync failed, recvBuf[%p] recvBufLen[%llu] ret[%d]", recvBuf, recvBufLen, ret);
    return HCCL_E_NETWORK;
}

HcclResult HcclSocket::GetAsyncReqResult(void* reqHandle, HcclResult& reqResult)
{
    CHK_PTR_NULL(reqHandle);
    s32 asyncReqRet = 0;
    s32 ret = hrtRaSocketGetAsyncReqResult(reqHandle, &asyncReqRet);
    if (ret == 0) {
        reqResult
            = (asyncReqRet == 0) ? HCCL_SUCCESS : (asyncReqRet == SOCK_EAGAIN ? HCCL_E_AGAIN : HCCL_E_TCP_TRANSFER);
        return HCCL_SUCCESS;
    }

    if (ret == OTHERS_EAGAIN) {
        return HCCL_E_AGAIN;
    }
    HCCL_ERROR("[GetAsyncReqResult]RaSocketGetAsyncReqResult failed, ret[%d]", ret);
    return HCCL_E_NETWORK;
}

// static
bool HcclSocket::IsSupportAsync()
{
    bool isSupportRaSocketAsync = false;
    HcclResult ret = IsSupportHdcAsync(isSupportRaSocketAsync);
    if (ret != HCCL_SUCCESS) { // 失败时默认不支持异步收发
        HCCL_WARNING("[IsSupportAsync] IsSupportHdcAsync failed ret[%d]", ret);
    }
    return isSupportRaSocketAsync;
}

std::string HcclSocket::GetTag() const { return tag_; }

NicType HcclSocket::GetSocketType() const { return socketType_; }

HcclResult HcclSocket::GetNicSocketHandle(
    std::map<HcclIpAddress, IpSocket>& socketMap, const HcclIpAddress& ip, SocketHandle& nicSocketHandle)
{
    if (ip.IsInvalid()) {
        HCCL_ERROR("[Get][NicHandleInfo]phyId[%u] nicIp is invalid", localDevicePhyId_);
        return HCCL_E_PARA;
    }

    auto it = socketMap.find(ip);
    if (it == socketMap.end()) {
        HCCL_ERROR("[Get][NicHandleInfo]can not find nic socket handle, ip[%s]", ip.GetReadableAddress());
        return HCCL_E_PARA;
    } else {
        if (it->second.nicSocketHandle == nullptr) {
            HCCL_ERROR(
                "[Get][NicHandleInfo]get nic socket handle failed! phyId[%u] IP addr[%s]", localDevicePhyId_,
                ip.GetReadableAddress());
            return HCCL_E_PARA;
        }
        nicSocketHandle = it->second.nicSocketHandle;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclSocket::GetNicSocketHandle()
{
    if (nicSocketHandle_ != nullptr) {
        return HCCL_SUCCESS;
    }
    RaResourceInfo raResourceInfo;
    CHK_RET(NetworkManager::GetInstance(localDeviceLogicId_).GetRaResourceInfo(raResourceInfo));

    std::map<HcclIpAddress, IpSocket> tempSocketMap;

    if (socketType_ == NicType::DEVICE_NIC_TYPE) {
        tempSocketMap = raResourceInfo.nicSocketMap;
        HCCL_INFO(
            "[Get][NicHandleInfo]phyId[%u], nicSocketMap[%u] localIp[[%s]", localDeviceLogicId_, tempSocketMap.size(),
            localIp_.GetReadableAddress());
        CHK_RET(GetNicSocketHandle(tempSocketMap, localIp_, nicSocketHandle_));
    } else if (socketType_ == NicType::HOST_NIC_TYPE) {
        tempSocketMap = raResourceInfo.hostNetSocketMap;
        HCCL_INFO(
            "[Get][NicHandleInfo]phyId[%u], nicSocketMap[%u] localIp[[%s]", localDeviceLogicId_, tempSocketMap.size(),
            localIp_.GetReadableAddress());
        CHK_RET(GetNicSocketHandle(tempSocketMap, localIp_, nicSocketHandle_));
    } else if (socketType_ == NicType::VNIC_TYPE) {
        tempSocketMap = raResourceInfo.vnicSocketMap;
        HCCL_INFO(
            "[Get][NicHandleInfo]phyId[%u], vnicSocketMap size[%u] localIp[[%s]", localDeviceLogicId_,
            tempSocketMap.size(), localIp_.GetReadableAddress());
        CHK_RET(GetNicSocketHandle(tempSocketMap, localIp_, nicSocketHandle_));
    } else {
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclIpAddress HcclSocket::GetRemoteIp() const { return remoteIp_; }

u32 HcclSocket::GetRemotePort() const { return remotePort_; }

HcclIpAddress HcclSocket::GetLocalIp() const { return localIp_; }

u32 HcclSocket::GetLocalPort() const { return localPort_; }

HcclSocketRole HcclSocket::GetLocalRole() const { return localRole_; }

FdHandle HcclSocket::GetFdHandle() const { return fdHandle_; }

void HcclSocket::SetForceClose(bool forceClose) { forceClose_ = forceClose; }

HcclResult HcclSocket::SetStopFlag(bool value)
{
    stopFlag_.store(value);
    return HCCL_SUCCESS;
}

bool HcclSocket::GetStopFlag() { return stopFlag_.load(); }
} // namespace hccl
