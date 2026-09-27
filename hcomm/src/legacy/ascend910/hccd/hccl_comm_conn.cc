/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_comm_conn.h"
#include <algorithm>
#include "dlhal_function.h"
#include "hccl_comm_conn_mgr.h"
#include "transport_heterog_raw_roce.h"

using namespace std;

namespace hccl {

static const string CONNECT_TAG = "COMMCONN_";

HcclCommConn::HcclCommConn() {}

HcclCommConn::~HcclCommConn()
{
    HcclResult ret = HCCL_SUCCESS;
    if (role_ == SERVER_ROLE_SOCKET && isListen_) {
        (void)StopListen();
    }

    if (memBlocksManager_ != nullptr) {
        HcclResult ret
            = MrManager::GetInstance().ReleaseKey(memBlocksManager_->GetMemAddr(), memBlocksManager_->GetMemSize());
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "Comm connection ip[%s], ReleaseKey failed!",
                HcclIpAddress(localAddr_.info.tcp.ipv4Addr).GetReadableIP());
        }
    }

    if (transport_.get() != nullptr && rdmaHandle_ != nullptr) {
        (void)MrManager::GetInstance().DeInit(rdmaHandle_);
    }

    if (transport_.get() != nullptr) {
        transport_->Deinit();
    }

    // 用户使用Connect()但是底层链路未建链成功场景使用abort强行停止
    if (role_ == CLIENT_ROLE_SOCKET && socketInfo_.fdHandle == nullptr) {
        ret = hrtRaSocketNonBlockBatchAbort(&connectInfo_, 1);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("hrtRaSocketNonBlockBatchAbort failed");
        }
    }
    if (socketHandle_ != nullptr) {
        (void)hrtRaSocketDeInitRef(socketHandle_);
        socketHandle_ = nullptr;
    }

    if (rdmaHandle_ != nullptr) {
        (void)HrtRaRdmaDeInitRef(rdmaHandle_, NO_USE);
        rdmaHandle_ = nullptr;
    }

    HcclCommConnMgr::GetInstance().DeleteConnectCommMap(remoteAddr_);
}

HcclResult HcclCommConn::SetAddr(HcclAddr& bindAddr, u32 opType)
{
    if (opType == INIT_LOCAL_IP) {
        localAddr_ = bindAddr;
    } else if (opType == INIT_REMOTE_IP) {
        remoteAddr_ = bindAddr;
    } else {
        HCCL_ERROR("This op[%u] is not supported currently.", opType);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

// 在client端，由于hccp接口不支持，当前Bind接口不支持指定socket的本地port
HcclResult HcclCommConn::Bind(HcclAddr& bindAddr)
{
    HcclResult ret = HCCL_SUCCESS;
    // 增加一个锁，防止同一个comm出现并发情况
    lock_guard<mutex> lock(bindMutex_);
    if (socketHandle_ != nullptr && rdmaHandle_ != nullptr) {
        HCCL_ERROR("Duplicate bind, please check!");
        return HCCL_E_PARA;
    }

    CHK_RET(SetAddr(bindAddr, INIT_LOCAL_IP));

    u32& localIpv4Addr = localAddr_.info.tcp.ipv4Addr;
    HCCL_RUN_INFO(
        "HcclCommConn Bind localIpv4Addr[%s], port[%u]", HcclIpAddress(localIpv4Addr).GetReadableIP(),
        localAddr_.info.tcp.port);

    struct rdev nicRdevInfo {};
    nicRdevInfo.phyId = devId_;
    nicRdevInfo.family = AF_INET;
    nicRdevInfo.localIp.addr.s_addr = localIpv4Addr;

    if (socketHandle_ == nullptr) {
        ret = hrtRaSocketInitRef(NETWORK_PEER_ONLINE, nicRdevInfo, socketHandle_);
        CHK_PTR_NULL(socketHandle_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "hrtRaSocketInit failed! ip[%s], port[%u], ret[%d]", HcclIpAddress(localIpv4Addr).GetReadableIP(),
                localAddr_.info.tcp.port, ret);
            return HCCL_E_ROCE_CONNECT;
        }
    }

    ret = HrtRaRdmaInitRef(NETWORK_PEER_ONLINE, NO_USE, nicRdevInfo, rdmaHandle_);
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("HcclCommConn Bind rdma init need retry."), HCCL_E_AGAIN);
    CHK_PTR_NULL(rdmaHandle_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("hrtRaRdmaInit failed! ip[%s], ret[%d]", HcclIpAddress(localIpv4Addr).GetReadableIP(), ret);
        return HCCL_E_ROCE_CONNECT;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::Listen([[maybe_unused]] int backLog)
{
    if (isListen_) {
        HCCL_ERROR(
            "This conn has been listened ip[%s], port[%u]", HcclIpAddress(localAddr_.info.tcp.ipv4Addr).GetReadableIP(),
            localAddr_.info.tcp.port);
        return HCCL_E_PARA;
    }

    if (UNLIKELY(role_ == CLIENT_ROLE_SOCKET)) {
        HCCL_ERROR("this HcclCommConn has been configured as client, cannot use listen as server.");
        return HCCL_E_INTERNAL;
    }

    CHK_PTR_NULL(socketHandle_);
    struct SocketListenInfoT serverInfo;
    serverInfo.socketHandle = socketHandle_;
    serverInfo.port = localAddr_.info.tcp.port;
    HCCL_RUN_INFO(
        "HcclCommConn Listen localIpv4Addr[%s], port[%u]", HcclIpAddress(localAddr_.info.tcp.ipv4Addr).GetReadableIP(),
        localAddr_.info.tcp.port);
    HcclResult ret = hrtRaSocketNonBlockListenStart(&serverInfo, 1);
    std::string errormessage = "The IP address "
                               + std::string(HcclIpAddress(localAddr_.info.tcp.ipv4Addr).GetReadableIP()) + " and port "
                               + std::to_string(localAddr_.info.tcp.port) + " have already been bound.";
    RPT_INPUT_ERR(
        ret == HCCL_E_UNAVAIL, "EI0019", std::vector<std::string>({"reason"}),
        std::vector<std::string>({errormessage}));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("HcclCommConn start listen socket fail. "), ret);
    CHK_RET(hrtRaSocketAcceptCreditAdd(&serverInfo, 1, MAX_CONCURRENCY_LINK_NUM));
    isListen_ = true;
    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::StopListen()
{
    struct SocketListenInfoT serverInfo;
    serverInfo.socketHandle = socketHandle_;
    serverInfo.port = localAddr_.info.tcp.port;
    CHK_RET(hrtRaSocketListenStop(&serverInfo, 1));
    isListen_ = false;
    HCCL_RUN_INFO(
        "HcclCommConn ip[%s] port[%u] StopListen success.", HcclIpAddress(localAddr_.info.tcp.ipv4Addr).GetReadableIP(),
        localAddr_.info.tcp.port);
    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::Accept(HcclAddr& acceptAddr, HcclCommConn*& acceptConn)
{
    HcclResult ret = HCCL_SUCCESS;
    AcceptCommConn acceptComConn;
    std::queue<AcceptCommConn> connHandleTmpQueue{};
    bool isNeedCreditAdd = false;
    u32 creditNum = 0;

    std::unique_lock<std::mutex> lock(connHandleQueueMutex_);
    if (connHandleQueue_.size() == MAX_CONCURRENCY_LINK_NUM) {
        HCCL_RUN_WARNING(
            "The maximum number of concurrent link setups is %u. cur link num[%u]", MAX_CONCURRENCY_LINK_NUM,
            connHandleQueue_.size());
        ret = HCCL_E_AGAIN;
    } else if (HcclCommConnMgr::GetInstance().IsExceedMaxLinkNum(SERVER_ROLE_SOCKET)) {
        HCCL_RUN_WARNING(
            "The maximum number of communication connections that can be created is %u.", MAX_CONN_LINK_NUM);
        ret = HCCL_E_AGAIN;
    } else {
        ret = PrepareSocketInfoForServer(acceptComConn.socketInfo);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        ret = GetSocket(acceptComConn.socketInfo);
        if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
            HCCL_ERROR("HcclCommConn Accept GetSocket fail error[%d]", ret);
            return HCCL_E_TCP_CONNECT;
        } else if (ret == HCCL_SUCCESS) {
            HCCL_RUN_INFO(
                "Server Got new socket, ipv4Addr[%s] socketHandle[%p] fdHandle[%p]",
                HcclIpAddress(acceptComConn.socketInfo.remoteIp.addr.s_addr).GetReadableIP(),
                acceptComConn.socketInfo.socketHandle, acceptComConn.socketInfo.fdHandle);
            acceptComConn.newCommConn = new (nothrow) HcclCommConn();
            CHK_PTR_NULL(acceptComConn.newCommConn);
            acceptComConn.newCommConn->SetStartTime();
            connHandleQueue_.push(acceptComConn);
        }
    }

    while (!connHandleQueue_.empty()) {
        acceptComConn = connHandleQueue_.front();
        connHandleQueue_.pop();
        ret = acceptComConn.newCommConn->InitTransport(role_, localAddr_, acceptComConn.socketInfo);
        if (ret == HCCL_SUCCESS) {
            acceptConn = acceptComConn.newCommConn;
            acceptComConn.newCommConn = nullptr;
            acceptAddr = acceptConn->GetRemoteAddr();
            isNeedCreditAdd = true;
            creditNum++;
            HCCL_RUN_INFO(
                "Server Got new socket finally, ipv4Addr[%s], port[%u]",
                HcclIpAddress(acceptAddr.info.tcp.ipv4Addr).GetReadableIP(), acceptAddr.info.tcp.port);
            break;
        } else if (ret != HCCL_E_AGAIN) {
            HCCL_RUN_WARNING(
                "Accept Error Result[%d], Need Reset Conn ipv4Addr[%s]", ret,
                HcclIpAddress(acceptComConn.socketInfo.remoteIp.addr.s_addr).GetReadableIP());
            CHK_RET(ResetCurrentErrorConnection(acceptComConn.newCommConn));
            isNeedCreditAdd = true;
            creditNum++;
            break;
        } else {
            // 增加防吊死功能
            auto endTime = std::chrono::steady_clock::now();
            std::chrono::time_point<std::chrono::steady_clock> startTime;
            acceptComConn.newCommConn->GetStartTime(startTime);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            if (duration > ACCEPT_MAX_TIME) {
                HCCL_RUN_WARNING(
                    "accept time duration > %ums, Need Reset Conn ipv4Addr[%s]", ACCEPT_MAX_TIME,
                    HcclIpAddress(acceptComConn.socketInfo.remoteIp.addr.s_addr).GetReadableIP());
                CHK_RET(ResetCurrentErrorConnection(acceptComConn.newCommConn));
                isNeedCreditAdd = true;
                creditNum++;
                continue;
            }
            connHandleTmpQueue.push(acceptComConn);
        }
    }

    while (!connHandleTmpQueue.empty()) {
        connHandleQueue_.push(connHandleTmpQueue.front());
        connHandleTmpQueue.pop();
    }

    if (isNeedCreditAdd) {
        // 当建链成功、qp交换信息返回不可恢复错误、触发防吊死三种情况都需要进程accept credit add
        struct SocketListenInfoT serverInfo;
        serverInfo.socketHandle = socketHandle_;
        serverInfo.port = localAddr_.info.tcp.port;
        CHK_RET(hrtRaSocketAcceptCreditAdd(&serverInfo, 1, creditNum));
    }
    return ret;
}

HcclResult HcclCommConn::ResetCurrentErrorConnection(HcclCommConn*& newCommConn)
{
    if (newCommConn == nullptr) {
        HCCL_INFO("No Connection is being processed.");
        return HCCL_SUCCESS;
    }

    if (transport_ != nullptr) {
        transport_->SetForceClose();
    }
    delete newCommConn;
    newCommConn = nullptr;

    return HCCL_SUCCESS;
}

void HcclCommConn::SetForceClose()
{
    if (transport_ != nullptr) {
        transport_->SetForceClose();
    }
}

const HcclAddr& HcclCommConn::GetRemoteAddr() const { return remoteAddr_; }

HcclResult HcclCommConn::PrepareSocketInfoForServer(struct SocketInfoT& socketInfo)
{
    string linkTag = CONNECT_TAG + to_string(0) + "_" + to_string(localAddr_.info.tcp.ipv4Addr) + "_"
                     + to_string(localAddr_.info.tcp.port);

    socketInfo.socketHandle = socketHandle_;
    socketInfo.fdHandle = nullptr;
    socketInfo.status = CONNECT_FAIL;
    CHK_SAFETY_FUNC_RET(strncpy_s(socketInfo.tag, SOCK_CONN_TAG_SIZE, linkTag.c_str(), linkTag.length() + 1));
    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::GetSocket(struct SocketInfoT& socketInfo)
{
    u32 connectedNum = 0;

    HcclResult ret = hrtRaNonBlockGetSockets(role_, &socketInfo, 1, &connectedNum);
    if (ret == HCCL_SUCCESS) {
        if (connectedNum == 0) {
            ret = HCCL_E_AGAIN;
        } else if (connectedNum != 1 || socketInfo.status != CONNECT_OK || socketInfo.fdHandle == nullptr) {
            HCCL_ERROR(
                "GetSocket fail linkTag linkTag[%s], connectedNum[%u] != 1, status[%d] != CONNECT_OK, "
                "or fdHandle is nullptr",
                socketInfo.tag, connectedNum, socketInfo.status);
            return HCCL_E_TCP_CONNECT;
        }
    }

    if (ret == HCCL_E_AGAIN) {
        SaluSleep(DELAY_TIME);
    }

    return ret;
}

HcclResult HcclCommConn::PrepareConnectSocketInfoForClient(HcclAddr& bindAddr)
{
    CHK_RET(SetAddr(bindAddr, INIT_REMOTE_IP));

    HcclIpAddress remoteIp(remoteAddr_.info.tcp.ipv4Addr);
    string linkTag = CONNECT_TAG + to_string(0) + "_" + to_string(remoteAddr_.info.tcp.ipv4Addr) + "_"
                     + to_string(remoteAddr_.info.tcp.port);

    connectInfo_.socketHandle = socketHandle_;
    connectInfo_.remoteIp.addr = remoteIp.GetBinaryAddress().addr;
    connectInfo_.remoteIp.addr6 = remoteIp.GetBinaryAddress().addr6;
    connectInfo_.port = remoteAddr_.info.tcp.port;
    CHK_SAFETY_FUNC_RET(strncpy_s(connectInfo_.tag, SOCK_CONN_TAG_SIZE, linkTag.c_str(), linkTag.length() + 1));

    socketInfo_.socketHandle = socketHandle_;
    socketInfo_.fdHandle = nullptr;
    socketInfo_.remoteIp.addr.s_addr = remoteAddr_.info.tcp.ipv4Addr;
    socketInfo_.status = CONNECT_FAIL;
    CHK_SAFETY_FUNC_RET(strncpy_s(socketInfo_.tag, SOCK_CONN_TAG_SIZE, linkTag.c_str(), linkTag.length() + 1));
    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::InitMsgAndRequestBuffer()
{
    {
        lock_guard<mutex> lock(msgInfosMutex_);
        if (msgInfosMem_ == nullptr) {
            msgInfosMem_.reset(new (nothrow) LocklessRingMemoryAllocate<HcclMessageInfo>(RESOURCE_MEMORY_CAPACITY));
            CHK_SMART_PTR_NULL(msgInfosMem_);
            CHK_RET(msgInfosMem_->Init());
            HCCL_INFO("InitRecvMsgBuffer Success!");
        }
    }

    {
        lock_guard<mutex> lock(reqInfosMutex_);
        if (reqInfosMem_ == nullptr) {
            reqInfosMem_.reset(new (nothrow) LocklessRingMemoryAllocate<HcclRequestInfo>(RESOURCE_MEMORY_CAPACITY));
            CHK_SMART_PTR_NULL(reqInfosMem_);
            CHK_RET(reqInfosMem_->Init());
            HCCL_INFO("InitRequestBuffer Success!");
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::InitMemBlocksAndRecvWrMem()
{
    u32 memBlockNum = MEM_BLOCK_CAPACITY; // MEM_BLOCK_NUM_BIGER
    u32 info = 0;
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    CHK_RET(hrtDrvGetPlatformInfo(&info));

    // 初始化信封内存
    if (memBlocksManager_ == nullptr) {
        memBlocksManager_.reset(new (nothrow) HeterogMemBlocksManager());
        CHK_SMART_PTR_NULL(memBlocksManager_);
        CHK_RET(memBlocksManager_->Init(memBlockNum));
    }

    // 初始化wr内存
    {
        lock_guard<mutex> lock(recvWrInfosMutex_);
        if (recvWrInfosMem_ == nullptr) {
            recvWrInfosMem_.reset(new (nothrow) LocklessRingMemoryAllocate<RecvWrInfo>(RESOURCE_MEMORY_CAPACITY));
            CHK_SMART_PTR_NULL(recvWrInfosMem_);
            CHK_RET(recvWrInfosMem_->Init());
        }
    }

    // 注册mr
    CHK_RET(MrManager::GetInstance().GetKey(
        memBlocksManager_->GetMemAddr(), memBlocksManager_->GetMemSize(), transportResourceInfo_.lkey));
    HCCL_INFO("InitMemBlocksAndRecvWrMem Success!");

    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::InitTransport(u32 role, HcclAddr& localAddr, SocketInfoT& tmpInfo)
{
    if (transport_ != nullptr) {
        return transport_->CheckAndPushBuildLink();
    }

    if (role == SERVER_ROLE_SOCKET) {
        role_ = role;
        localAddr_ = localAddr;
        remoteAddr_.info.tcp.ipv4Addr = tmpInfo.remoteIp.addr.s_addr;
        remoteAddr_.info.tcp.port = 0; // 不感知对端端口号，默认填0

        struct rdev nicRdevInfo {};
        nicRdevInfo.phyId = devId_;
        nicRdevInfo.family = AF_INET;
        nicRdevInfo.localIp.addr.s_addr = localAddr_.info.tcp.ipv4Addr;
        CHK_RET(hrtRaSocketInitRef(NETWORK_PEER_ONLINE, nicRdevInfo, socketHandle_));
        CHK_RET(HrtRaRdmaInitRef(NETWORK_PEER_ONLINE, NO_USE, nicRdevInfo, rdmaHandle_));
    }

    if (localAddr_.type != HCCL_ADDR_TYPE_ROCE) {
        HCCL_ERROR("HcclCommConn: This type[%d] is not supported currently.", localAddr_.type);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_RET(MrManager::GetInstance().Init(rdmaHandle_));
    CHK_RET(InitMsgAndRequestBuffer());
    CHK_RET(InitMemBlocksAndRecvWrMem());

    const string& linkTag = CONNECT_TAG;
    HcclIpAddress selfIp(localAddr_.info.tcp.ipv4Addr);
    HcclIpAddress peerIp(remoteAddr_.info.tcp.ipv4Addr);

    HCCL_RUN_INFO(
        "role[%u], local ipv4[%s], port[%u], remote ipv4[%s], port[%u] init TransportRoce", role_,
        HcclIpAddress(localAddr_.info.tcp.ipv4Addr).GetReadableIP(), localAddr_.info.tcp.port,
        HcclIpAddress(remoteAddr_.info.tcp.ipv4Addr).GetReadableIP(), remoteAddr_.info.tcp.port);

    transportResourceInfo_.isRawConn = true;
    EXCEPTION_CATCH(
        (transport_ = make_unique<TransportHeterogRawRoce>(
             linkTag, selfIp, peerIp, remoteAddr_.info.tcp.port, localAddr_.info.tcp.port, transportResourceInfo_)),
        return HCCL_E_PTR);

    CHK_SMART_PTR_NULL(transport_);
    CHK_RET(transport_->Init(tmpInfo, rdmaHandle_, &MrManager::GetInstance()));

    return transport_->CheckAndPushBuildLink();
}

HcclResult HcclCommConn::Connect(HcclAddr& connectAddr)
{
    if (UNLIKELY(isListen_)) {
        HCCL_ERROR("this HcclCommConn has been listened as server, cannot use connect as client.");
        return HCCL_E_INTERNAL;
    }

    HcclResult ret = HCCL_SUCCESS;
    switch (connectState_) {
        case OpStatus::START:
            role_ = CLIENT_ROLE_SOCKET;
            ret = PrepareConnectSocketInfoForClient(connectAddr);
            if (ret != HCCL_SUCCESS) {
                break;
            }
            [[fallthrough]];
        case OpStatus::CONNECT:
            connectState_ = OpStatus::CONNECT;
            ret = hrtRaSocketNonBlockBatchConnect(&connectInfo_, 1);
            if (ret != HCCL_SUCCESS) {
                break;
            }
            [[fallthrough]];
        case OpStatus::GETSOCKET:
            connectState_ = OpStatus::GETSOCKET;
            ret = GetSocket(socketInfo_);
            if (ret != HCCL_SUCCESS) {
                break;
            }
            [[fallthrough]];
        case OpStatus::BUILDTRANSPORT:
            connectState_ = OpStatus::BUILDTRANSPORT;
            ret = InitTransport(role_, localAddr_, socketInfo_);
            if (ret == HCCL_SUCCESS) {
                connectState_ = OpStatus::END;
            }
            break;
        case OpStatus::END:
            HCCL_WARNING(
                "Connect: This conn has been Connected ip[%s]",
                HcclIpAddress(localAddr_.info.tcp.ipv4Addr).GetReadableIP());
            break;
        default:
            HCCL_ERROR("Connect: op Invalid connectState[%u].", connectState_);
            return HCCL_E_INTERNAL;
    }

    HCCL_DEBUG("Connect: op connectState[%u] ret[%d].", connectState_, ret);
    return ret;
}

HcclResult HcclCommConn::Isend(const void* buf, int count, HcclDataType dataType, HcclRequest& request)
{
    CheckDataType(dataType);

    if ((buf == nullptr) && (count != 0)) {
        HCCL_ERROR("[Check][Buffer]errNo[0x%016llx] or count[%d] is invalid", HCCL_ERROR_CODE(HCCL_E_PARA), count);
        return HCCL_E_PARA;
    }

    CHK_PRT_RET(
        transport_ == nullptr,
        HCCL_ERROR("[Get][transportPtr]errNo[0x%016llx] transportPtr is nullptr", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    TransportEndPointInfo srcEp(0, DEFAULT_LOCAL_RANK, DEFAULT_TAG);
    TransportEndPointInfo dstEp(0, DEFAULT_REMOTE_RANK, DEFAULT_TAG);
    TransportEndPointParam epParam(srcEp, dstEp);

    TransData sendData(reinterpret_cast<u64>(buf), reinterpret_cast<u64>(nullptr), count, dataType, false, 0);
    HcclRequestInfo* requestHandle = nullptr;
    CHK_RET(transport_->Isend(sendData, epParam, requestHandle));
    request = requestHandle;
    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::Improbe(int& flag, HcclMessage& msg, HcclStatus& status)
{
    CHK_PRT_RET(
        transport_ == nullptr,
        HCCL_ERROR("[Get][transportPtr]errNo[0x%016llx] transportPtr is nullptr", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    TransportEndPointInfo srcEp(0, DEFAULT_REMOTE_RANK, DEFAULT_TAG);
    TransportEndPointInfo dstEp(0, DEFAULT_LOCAL_RANK, DEFAULT_TAG);
    TransportEndPointParam epParam(srcEp, dstEp);
    HcclMessageInfo* msgHandle = nullptr;

    transport_->Improbe(epParam, flag, msgHandle, status);
    msg = msgHandle;
    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::Imrecv(void* buf, int count, HcclDataType dataType, HcclMessage msg, HcclRequest& request)
{
    CheckDataType(dataType);

    HcclMessageInfo* msgHandle = static_cast<HcclMessageInfo*>(msg);
    CHK_PTR_NULL(msgHandle);
    CHK_PRT_RET(
        transport_ == nullptr,
        HCCL_ERROR("[Get][transportPtr]errNo[0x%016llx] transportPtr is nullptr", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    HcclRequestInfo* requestHandle = nullptr;
    TransData recvData(reinterpret_cast<u64>(nullptr), reinterpret_cast<u64>(buf), count, dataType);
    CHK_RET(transport_->Imrecv(recvData, *msgHandle, requestHandle));
    request = requestHandle;
    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::ImrecvScatter(
    void* buf[], int count[], int bufCount, HcclDataType datatype, HcclMessage msg, HcclRequest& request)
{
    CheckDataType(datatype);

    HcclMessageInfo* msgHandle = static_cast<HcclMessageInfo*>(msg);
    CHK_PTR_NULL(msgHandle);
    CHK_PRT_RET(
        transport_ == nullptr,
        HCCL_ERROR("[Get][transportPtr]errNo[0x%016llx] transportPtr is nullptr", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    HcclRequestInfo* requestHandle = nullptr;
    CHK_RET(transport_->ImrecvScatter(buf, count, bufCount, datatype, *msgHandle, requestHandle));
    request = requestHandle;
    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::Test(HcclRequest requestHandle, s32& flag, HcclStatus& compState)
{
    HcclRequestInfo* request = reinterpret_cast<HcclRequestInfo*>(requestHandle);
    CHK_PTR_NULL(request->transportHandle);

    TransportHeterog* transportPtr = reinterpret_cast<TransportHeterog*>(request->transportHandle);
    return transportPtr->Test(*request, flag, compState);
}

HcclResult HcclCommConn::CheckDataType(const HcclDataType dataType)
{
    if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8)) {
        HCCL_ERROR(
            "[Check][DataType]errNo[0x%016llx] data type[%s] not supported", HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT),
            GetDataTypeEnumStr(dataType).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommConn::SocketForceClose(SocketInfoT& socketInfo)
{
    if (socketInfo.socketHandle == nullptr || socketInfo.fdHandle == nullptr) {
        HCCL_ERROR(
            "SocketForceClose socketInfo is invalid socketHandle[%p] fdHandle[%p]", socketInfo.socketHandle,
            socketInfo.fdHandle);
        return HCCL_E_PARA;
    }

    SocketCloseInfoT conns[1]{};
    conns[0].socketHandle = socketInfo.socketHandle;
    conns[0].fdHandle = socketInfo.fdHandle;
    conns[0].disuseLinger = static_cast<s32>(true);

    HcclResult ret = hrtRaSocketBatchClose(conns, 1);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "SocketForceClose ra socket batch close failed socketHandle[%p] fdHandle[%p]", socketInfo.socketHandle,
            socketInfo.fdHandle);
        return ret;
    }
    socketInfo.socketHandle = nullptr;
    socketInfo.fdHandle = nullptr;
    return HCCL_SUCCESS;
}

void HcclCommConn::SetStartTime() { startTime_ = chrono::steady_clock::now(); }

void HcclCommConn::GetStartTime(std::chrono::time_point<std::chrono::steady_clock>& startTime)
{
    startTime = startTime_;
}
} // namespace hccl
