/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "zero_copy_memory_agent.h"
#include <string>
#include "acl/acl_rt.h"
#include "hccl_network_pub.h"
#include "adapter_hccp_common.h"
#include "adapter_rts_common.h"
#include "snapshot_control.h"

namespace hccl {
using namespace std;

const string STR_IPC_MEM_EXCHANGE = "IpcMemExchange";
constexpr u32 IPC_MEMORY_EXCHANGE_LENGTH = 64; // Bytes
constexpr u32 USLEEP_ONE_THOUSAND = 1000;
constexpr int INNER_THREAD_LOOP_US = 500;

std::shared_mutex ZeroCopyMemoryAgent::g_addressMgrMutex_;
std::unique_ptr<ZeroCopyAddressMgr> ZeroCopyMemoryAgent::addressMgr_ = nullptr;

template <typename T>
HcclResult ConstructData(u8*& exchangeDataPtr, u32& exchangeDataBlankSize, T& value)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &value, sizeof(T)));
    exchangeDataPtr += sizeof(T);
    exchangeDataBlankSize -= sizeof(T);
    return HCCL_SUCCESS;
}

/* copy 变长数据 */
HcclResult ConstructData(u8*& exchangeDataPtr, u32& exchangeDataBlankSize, const void* ptr, size_t len)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, ptr, len));
    exchangeDataPtr += len;
    exchangeDataBlankSize -= len;
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult ParseData(u8*& exchangeDataPtr, u32& exchangeDataBlankSize, T& value)
{
    CHK_PRT_RET(
        exchangeDataBlankSize < sizeof(T),
        HCCL_ERROR("[ParseData] blankSize is [%u] less than [%lu]", exchangeDataBlankSize, sizeof(T)), HCCL_E_INTERNAL);

    CHK_SAFETY_FUNC_RET(memcpy_s(&value, sizeof(T), exchangeDataPtr, sizeof(T)));
    exchangeDataPtr += sizeof(T);
    exchangeDataBlankSize -= sizeof(T);
    return HCCL_SUCCESS;
}

ZeroCopyMemoryAgent::ZeroCopyMemoryAgent(
    const std::unique_ptr<HcclSocketManager>& socketManager, u32 devicePhyId, s32 deviceLogicId,
    const HcclIpAddress& localVnicIp, const std::vector<RankInfo>& rankInfoList, RankId userRank, bool useSuperPodMode,
    const std::string& identifier)
    : initiated_(false),
      socketManager_(socketManager),
      devicePhyId_(devicePhyId),
      deviceLogicId_(deviceLogicId),
      localVnicIp_(localVnicIp),
      rankInfoList_(rankInfoList),
      userRank_(userRank),
      rankSize_(rankInfoList.size()),
      useSuperPodMode_(useSuperPodMode),
      identifier_(identifier)
{}

// 创建vnic socket连接，启动recv 接收线程
// 每个rank 都启动listen，并且都和对端connect
HcclResult ZeroCopyMemoryAgent::Init()
{
    isSingleRank_ = (rankInfoList_.size() == 1);
    CHK_PRT_RET(isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][Init] single rank communicator"), HCCL_SUCCESS);
    std::unique_lock<std::mutex> lock(commRefCntLock_);

    {
        std::unique_lock<std::shared_mutex> wlock(g_addressMgrMutex_);
        if (addressMgr_ == nullptr) {
            addressMgr_ = std::make_unique<ZeroCopyAddressMgr>();
            HCCL_RUN_INFO("[ZeroCopyMemoryAgent][%s]init addressMgr_ success.", __func__);
        }
        CHK_RET(addressMgr_->IncreCommRefCnt());
    }

    CHK_RET(EstablishSockets());

    exchangeDataForSend_.resize(IPC_MEMORY_EXCHANGE_LENGTH * ZERO_COPY_MEMORY_AGENT_SEND_QUEUE_SIZE, 0);
    for (const auto& kv : mapDevPhyIdconnectedSockets_) {
        exchangeDataForAck_[kv.first].resize(IPC_MEMORY_EXCHANGE_LENGTH, 0);
        sendMgrs_[kv.first].reqDataSize_ = IPC_MEMORY_EXCHANGE_LENGTH;
        recvMgrs_[kv.first].receivedData_.resize(
            ZERO_COPY_MEMORY_AGENT_RECV_QUEUE_SIZE, std::vector<u8>(IPC_MEMORY_EXCHANGE_LENGTH, 0));
    }

    CHK_RET(InitInnerThread());

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::InitInnerThread()
{
    threadRun_ = true;
    innerThread_.reset(new (std::nothrow) std::thread(&ZeroCopyMemoryAgent::InnerThread, std::ref(*this)));
    CHK_SMART_PTR_NULL(innerThread_);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::EstablishSockets()
{
    CHK_PRT_RET((vnicPortCtx_ != nullptr), HCCL_ERROR("[ZeroCopyMemoryAgent][Init] already initd"), HCCL_E_PARA);
    CHK_RET(HcclNetOpenDev(&vnicPortCtx_, NicType::VNIC_TYPE, devicePhyId_, deviceLogicId_, localVnicIp_));
    CHK_PTR_NULL(vnicPortCtx_);

    isSocketSupportAsync_ = HcclSocket::IsSupportAsync();
    HCCL_RUN_INFO("[ZeroCopyMemoryAgent][Init] isSocketSupportAsync[%d]", isSocketSupportAsync_);

    for (size_t i = 0; i < rankInfoList_.size(); i++) {
        if (rankInfoList_[i].devicePhyId == static_cast<s32>(devicePhyId_)) {
            continue;
        }
        HcclRankLinkInfo remoteLinkInfo;
        RankInfo dstRankInfo = rankInfoList_[i];
        remoteLinkInfo.userRank = dstRankInfo.userRank;
        remoteLinkInfo.devicePhyId = dstRankInfo.devicePhyId;
        remoteLinkInfo.ip = HcclIpAddress(dstRankInfo.devicePhyId);
        if (useSuperPodMode_) {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_SDID, dstRankInfo.superDeviceId, remoteLinkInfo.ip));
        } else {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, dstRankInfo.devicePhyId, remoteLinkInfo.ip));
        }
        // 通信域未分配端口则使用默认端口
        remoteLinkInfo.port
            = dstRankInfo.deviceVnicPort == HCCL_INVALID_PORT ? HETEROG_CCL_PORT : dstRankInfo.deviceVnicPort;
        remoteLinkInfo.socketsPerLink = 1;
        string newTag = GenerateSocketTag(devicePhyId_, rankInfoList_[i].devicePhyId);
        std::vector<std::shared_ptr<HcclSocket>> tmpSockets;
        HcclResult ret
            = socketManager_->CreateSingleLinkSocket(newTag, vnicPortCtx_, remoteLinkInfo, tmpSockets, false, true);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Create][DestSockets]Create single link sockets failed, "
                "local rank[%u], remote rank[%u]",
                userRank_, i),
            ret);
        if (tmpSockets.size() != 1) {
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][CreateVnic] socket number[%llu] is not 1 as expected!", tmpSockets.size());
            return HCCL_E_INTERNAL;
        }
        // 设置强制断链为关闭，避免进程退出时recv失败
        tmpSockets[0]->SetForceClose(false);
        mapDevPhyIdconnectedSockets_[remoteLinkInfo.devicePhyId] = (tmpSockets[0]);
        mapDevPhyId2RankId_[remoteLinkInfo.devicePhyId] = remoteLinkInfo.userRank;
    }

    for (const auto& kv : mapDevPhyIdconnectedSockets_) {
        CHK_PRT_RET(
            socketManager_->WaitLinkEstablish(kv.second) != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][EstablishSockets] tag[%s] socket establish failed", kv.second->GetTag().c_str()),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

std::string ZeroCopyMemoryAgent::GenerateSocketTag(u32 localRank, u32 remoteRank)
{
    u32 small = localRank;
    u32 large = remoteRank;

    if (localRank > remoteRank) {
        small = remoteRank;
        large = localRank;
    }

    // Socket构造规则：前缀 + identifier + small + large
    std::string tag
        = STR_IPC_MEM_EXCHANGE + "_" + identifier_ + "_" + std::to_string(small) + ":" + std::to_string(large);
    return tag;
}

HcclResult ZeroCopyMemoryAgent::SendRequestSync(RequestType requestType, const std::vector<u8>& req, u32 remoteDevPhyId)
{
    HcclResult ret;
    if (remoteDevPhyId != INVALID_VALUE_RANKID) {
        std::unique_lock<std::mutex> lock(sendMutex_); // send 存在多线调用，需要锁保护
        ret = mapDevPhyIdconnectedSockets_[remoteDevPhyId]->Send(req.data(), IPC_MEMORY_EXCHANGE_LENGTH);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][SendRequestSync] Send %s to remote[%u] failed",
                GetReadableRequestType(requestType), remoteDevPhyId),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    std::unique_lock<std::mutex> lock(sendMutex_);
    for (const auto& kv : mapDevPhyIdconnectedSockets_) {
        CHK_PRT_RET(
            kv.second->Send(req.data(), IPC_MEMORY_EXCHANGE_LENGTH) != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][SendRequestSync] Send %s to remote[%u] failed",
                GetReadableRequestType(requestType), kv.first),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::SendRequest(RequestType requestType, const std::vector<u8>& req, u32 remoteDevPhyId)
{
    HCCL_INFO(
        "[ZeroCopyMemoryAgent][SendRequest] requestType[%s] remote[%u]", GetReadableRequestType(requestType),
        remoteDevPhyId);

    if (!isSocketSupportAsync_) { // socket不支持异步收发的场景
        return SendRequestSync(requestType, req, remoteDevPhyId);
    }

    bool isAck = IsAckRequestType(requestType);
    if (remoteDevPhyId != INVALID_VALUE_RANKID) {
        sendMgrs_[remoteDevPhyId].AddRequest(isAck, req);
    } else {
        for (auto& kv : sendMgrs_) {
            kv.second.AddRequest(isAck, req);
        }
    }

    // 唤醒内部io线程
    std::unique_lock<std::mutex> lock(sendMutex_);
    hasSendRequest_ = true;
    sendCv_.notify_all();
    return HCCL_SUCCESS;
}

void ZeroCopyMemoryAgent::RequestBatchSendAsync()
{
    HcclResult ret;
    for (auto& kv : sendMgrs_) {
        auto& sendMgr = kv.second;
        if ((sendMgr.lastSendHandle_ != nullptr) || (!sendMgr.hasReq_[0] && !sendMgr.hasReq_[1])) {
            // 前回发送未完成 或者 没有待发送的数据
            continue;
        }

        if (mapDevPhyIdconnectedSockets_.find(kv.first) == mapDevPhyIdconnectedSockets_.end()) {
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][RequestBatchSendAsync] remote[%u] not found in"
                "mapDevPhyIdconnectedSockets_",
                kv.first);
            continue;
        }
        auto& socket = mapDevPhyIdconnectedSockets_[kv.first];
        if (sendMgr.sentSize_ == 0) {                       // 非断点续传
            if (sendMgr.hasReq_[0] && sendMgr.hasReq_[1]) { // 合并发送
                u8* ptr = const_cast<u8*>(sendMgr.reqDatas_[1]->data()) + IPC_MEMORY_EXCHANGE_LENGTH;
                u32 leftSize = IPC_MEMORY_EXCHANGE_LENGTH;
                if (ConstructData(ptr, leftSize, sendMgr.reqDatas_[0]->data(), IPC_MEMORY_EXCHANGE_LENGTH)
                    == HCCL_SUCCESS) {
                    sendMgr.hasReq_[0] = false;
                    sendMgr.currIndex_ = 1;
                    sendMgr.reqDataSize_ = IPC_MEMORY_EXCHANGE_LENGTH + IPC_MEMORY_EXCHANGE_LENGTH;
                } else {
                    sendMgr.currIndex_ = 0;
                    sendMgr.reqDataSize_ = IPC_MEMORY_EXCHANGE_LENGTH;
                }
            } else {
                sendMgr.currIndex_ = sendMgr.hasReq_[0] ? 0 : 1;
                sendMgr.reqDataSize_ = IPC_MEMORY_EXCHANGE_LENGTH;
            }
        }
        const std::vector<u8>* req = sendMgr.reqDatas_[sendMgr.currIndex_];
        sendMgr.lastSendSize_ = 0; // 用于ra上报发送的数据量
        ret = socket->SendAsync(
            req->data() + sendMgr.sentSize_, sendMgr.reqDataSize_ - sendMgr.sentSize_, &sendMgr.lastSendSize_,
            &sendMgr.lastSendHandle_);
        if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) { // 发送失败的场景
            RequestType requestType = *reinterpret_cast<const RequestType*>(req->data());
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][RequestBatchSendAsync] failed, ret[%d] remote[%u] requestType[%s] "
                "sentSize[%llu]",
                ret, kv.first, GetReadableRequestType(requestType), sendMgr.sentSize_);
        }
    }
}

void ZeroCopyMemoryAgent::CheckBatchSendAsyncResult()
{
    HcclResult ret;
    HcclResult lastSendRet;
    for (auto& kv : sendMgrs_) {
        auto& sendMgr = kv.second;
        if (sendMgr.lastSendHandle_ == nullptr) { // 没有正在执行的异步send
            continue;
        }

        if (mapDevPhyIdconnectedSockets_.find(kv.first) == mapDevPhyIdconnectedSockets_.end()) {
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][CheckBatchSendAsyncResult] remote[%u] not found in"
                "mapDevPhyIdconnectedSockets_",
                kv.first);
            continue;
        }
        auto& socket = mapDevPhyIdconnectedSockets_[kv.first];
        ret = socket->GetAsyncReqResult(sendMgr.lastSendHandle_, lastSendRet);
        if (ret != HCCL_SUCCESS) {
            CHK_PRT_CONT(
                ret != HCCL_E_AGAIN,
                HCCL_ERROR(
                    "[ZeroCopyMemoryAgent][CheckBatchSendAsyncResult]GetAsyncReqResult failed, ret[%d] remote[%u]", ret,
                    kv.first));
            continue;
        }

        sendMgr.lastSendHandle_ = nullptr;
        if ((lastSendRet != HCCL_SUCCESS) && (sendMgr.lastSendSize_ == 0)) {
            CHK_PRT_CONT(
                lastSendRet != HCCL_E_AGAIN, HCCL_ERROR(
                                                 "[ZeroCopyMemoryAgent][CheckBatchSendAsyncResult]SendAsync failed, "
                                                 "result[%d] remote[%u] sentSize[%llu]",
                                                 lastSendRet, kv.first, sendMgr.sentSize_));
            continue;
        }

        sendMgr.sentSize_ += sendMgr.lastSendSize_;      // 下次从中断的地方开始重发
        if (sendMgr.sentSize_ == sendMgr.reqDataSize_) { // request发送完成
            sendMgr.sentSize_ = 0;
            sendMgr.hasReq_[sendMgr.currIndex_] = false;
            HCCL_DEBUG(
                "[ZeroCopyMemoryAgent][CheckBatchSendAsyncResult]SendAsync success, requestType[%s] remote[%u]",
                GetReadableRequestType(
                    *reinterpret_cast<const RequestType*>(sendMgr.reqDatas_[sendMgr.currIndex_]->data())),
                kv.first);
        }
    }
}

void ZeroCopyMemoryAgent::RequestBatchRecvAsync()
{
    HcclResult ret;
    for (auto& kv : recvMgrs_) {
        auto& recvMgr = kv.second;
        if ((recvMgr.lastRecvHandle_ != nullptr) || // 前回接收未完成
            ((receivedBarrierClose_.count(kv.first) != 0) && (receivedBarrierCloseAck_.count(kv.first) != 0))) {
            // 该socket已经收到BarrierClose与BarrierCloseAck报文，因此不允许再进行其他数据接收了
            continue;
        }

        if (mapDevPhyIdconnectedSockets_.find(kv.first) == mapDevPhyIdconnectedSockets_.end()) {
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][RequestBatchRecvAsync] remote[%u] not found in"
                "mapDevPhyIdconnectedSockets_",
                kv.first);
            continue;
        }
        auto& socket = mapDevPhyIdconnectedSockets_[kv.first];
        std::vector<u8>& req = recvMgr.receivedData_[recvMgr.recvIndex_];
        recvMgr.lastRecvSize_ = 0; // 用于ra上报接收的数据量
        ret = socket->RecvAsync(
            req.data() + recvMgr.receivedSize_, IPC_MEMORY_EXCHANGE_LENGTH - recvMgr.receivedSize_,
            &recvMgr.lastRecvSize_, &recvMgr.lastRecvHandle_);
        CHK_PRT_CONT(
            (ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN),
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][RequestBatchRecvAsync] RecvAsync failed, ret[%d] remote[%u] receivedSize[%llu]",
                ret, kv.first, recvMgr.receivedSize_));
    }
}

void ZeroCopyMemoryAgent::CheckBatchRecvAsyncResult()
{
    HcclResult ret;
    HcclResult lastRecvRet;
    for (auto& kv : recvMgrs_) {
        auto& recvMgr = kv.second;
        if (recvMgr.lastRecvHandle_ == nullptr) { // 没有正在异步接收
            continue;
        }

        if (mapDevPhyIdconnectedSockets_.find(kv.first) == mapDevPhyIdconnectedSockets_.end()) {
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][CheckBatchRecvAsyncResult] remote[%u] not found in"
                "mapDevPhyIdconnectedSockets_",
                kv.first);
            continue;
        }
        auto& socket = mapDevPhyIdconnectedSockets_[kv.first];
        ret = socket->GetAsyncReqResult(recvMgr.lastRecvHandle_, lastRecvRet);
        if (ret != HCCL_SUCCESS) {
            CHK_PRT_CONT(
                ret != HCCL_E_AGAIN,
                HCCL_ERROR(
                    "[ZeroCopyMemoryAgent][CheckBatchRecvAsyncResult] GetAsyncReqResult failed, ret[%d] remote[%u]",
                    ret, kv.first));
            continue;
        }

        recvMgr.lastRecvHandle_ = nullptr;
        if ((lastRecvRet != HCCL_SUCCESS) && (recvMgr.lastRecvSize_ == 0)) {
            CHK_PRT_CONT(
                lastRecvRet != HCCL_E_AGAIN, HCCL_WARNING(
                                                 "[ZeroCopyMemoryAgent][CheckBatchRecvAsyncResult] RecvAsync failed, "
                                                 "result[%d] remote[%u] lastRecvSize[%llu]",
                                                 lastRecvRet, kv.first, recvMgr.lastRecvSize_));
            continue;
        }

        recvMgr.receivedSize_ += recvMgr.lastRecvSize_;
        if (recvMgr.receivedSize_ == IPC_MEMORY_EXCHANGE_LENGTH) {
            recvMgr.receivedSize_ = 0;
            RecvRequest(recvMgr, kv.first);
            ioRecvWaiting_ = true; // 后面高概率还有数据要收（ack与request合并场景），loop不等待
        } else {
            // request没收全，loop不等待
            ioRecvWaiting_ = (recvMgr.receivedSize_ > 0);
        }
    }
}

inline void ZeroCopyMemoryAgent::RecvRequest(ZeroCopyMemoryAgentRecvMgr& recvMgr, u32 remoteDevicePhyId)
{
    std::vector<u8>& req = recvMgr.receivedData_[recvMgr.recvIndex_];
    RequestType requestType = *reinterpret_cast<RequestType*>(req.data());
    HCCL_DEBUG(
        "[ZeroCopyMemoryAgent][RecvRequest] recv requestType[%s] remote[%u]", GetReadableRequestType(requestType),
        remoteDevicePhyId);

    if (IsAckRequestType(requestType)) { // 收到ACK时，直接优先处理
        u32 remoteRank = mapDevPhyId2RankId_[remoteDevicePhyId];
        CHK_PRT_CONT(
            ParseReceivedRequest(req, remoteRank) != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][ParseReceivedRequest] failed requestType[%s] remote[%u]",
                GetReadableRequestType(requestType), remoteDevicePhyId));
        return;
    }

    recvMgr.recvIndex_ = (recvMgr.recvIndex_ + 1) % ZERO_COPY_MEMORY_AGENT_RECV_QUEUE_SIZE; // 准备下一次接收
    hasReceivedRequest_ = true;
}

void ZeroCopyMemoryAgent::ParseReceivedRequests()
{
    if (!hasReceivedRequest_) {
        return;
    }
    hasReceivedRequest_ = false;

    for (auto& kv : recvMgrs_) {
        u32 remoteRank = mapDevPhyId2RankId_[kv.first];
        auto& recvMgr = kv.second;
        while (recvMgr.praseIndex_ != recvMgr.recvIndex_) {
            std::vector<u8>& req = recvMgr.receivedData_[recvMgr.praseIndex_];
            CHK_PRT_CONT(
                ParseReceivedRequest(req, remoteRank) != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[ZeroCopyMemoryAgent][ParseReceivedRequest] failed to parse requestType[%s] remote[%u]",
                    GetReadableRequestType(*reinterpret_cast<RequestType*>(req.data())), kv.first));
            recvMgr.praseIndex_++;
            if (recvMgr.praseIndex_ == ZERO_COPY_MEMORY_AGENT_RECV_QUEUE_SIZE) {
                recvMgr.praseIndex_ = 0;
            }
        }
    }
}

void ZeroCopyMemoryAgent::RequestBatchRecvSync()
{
    HcclResult ret;
    for (auto& kv : recvMgrs_) {
        auto& recvMgr = kv.second;
        if ((receivedBarrierClose_.count(kv.first) != 0) && (receivedBarrierCloseAck_.count(kv.first) != 0)) {
            // 该socket已经收到BarrierClose与BarrierCloseAck报文，因此不允许再进行其他数据接收了
            continue;
        }

        if (mapDevPhyIdconnectedSockets_.find(kv.first) == mapDevPhyIdconnectedSockets_.end()) {
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][RequestBatchRecvSync] remote[%u] not found in"
                "mapDevPhyIdconnectedSockets_",
                kv.first);
            continue;
        }
        auto& socket = mapDevPhyIdconnectedSockets_[kv.first];
        std::vector<u8>& req = recvMgr.receivedData_[0];
        recvMgr.lastRecvSize_ = 0;
        ret = socket->IRecv(
            req.data() + recvMgr.receivedSize_, IPC_MEMORY_EXCHANGE_LENGTH - recvMgr.receivedSize_,
            recvMgr.lastRecvSize_);
        CHK_PRT_CONT(
            (ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN),
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][RequestBatchRecvSync] IRecv failed, ret[%d] remote[%u] receivedSize[%llu]", ret,
                kv.first, recvMgr.receivedSize_));

        recvMgr.receivedSize_ += recvMgr.lastRecvSize_;
        if (recvMgr.receivedSize_ == IPC_MEMORY_EXCHANGE_LENGTH) {
            recvMgr.receivedSize_ = 0;
            ret = ParseReceivedRequest(req, mapDevPhyId2RankId_[kv.first]);
            CHK_PRT_CONT(ret != HCCL_SUCCESS, HCCL_ERROR("[ZeroCopyMemoryAgent][ParseReceivedRequest] failed"));
        }
    }
}

void ZeroCopyMemoryAgent::InnerThread()
{
    // 新线程，更新一下使用的设备
    if (hrtSetDevice(deviceLogicId_) != HCCL_SUCCESS) {
        HCCL_ERROR("[ZeroCopyMemoryAgent][InnerThread] set device failed");
        return;
    }

    while (threadRun_) {
        CheckSnapshotStatus();
        if (isPaused_) {
            SaluSleep(USLEEP_ONE_THOUSAND);
            continue;
        }

        if (isSocketSupportAsync_) {
            CheckBatchSendAsyncResult();
            RequestBatchSendAsync();

            CheckBatchRecvAsyncResult();
            RequestBatchRecvAsync();

            ParseReceivedRequests();

            std::unique_lock<std::mutex> lock(sendMutex_);
            if (!ioRecvWaiting_ && !hasSendRequest_) {
                sendCv_.wait_for(lock, std::chrono::microseconds(INNER_THREAD_LOOP_US));
            }
            hasSendRequest_ = false;
            ioRecvWaiting_ = false;
        } else {
            RequestBatchRecvSync();
            SaluSleep(USLEEP_ONE_THOUSAND);
        }
    }

    if (hrtResetDevice(deviceLogicId_) != HCCL_SUCCESS) {
        HCCL_ERROR("[ZeroCopyMemoryAgent][InnerThread] reset device failed");
        return;
    }
}

HcclResult ZeroCopyMemoryAgent::SetRemoteTgid()
{
    if (remotePids_.size() == mapDevPhyIdconnectedSockets_.size()) {
        HCCL_INFO("[ZeroCopyMemoryAgent][SetRemoteTgid] tgid exchange is ok");
        return HCCL_SUCCESS;
    }
    remotePids_.clear();

    u8* exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::SET_REMOTE_BARE_TGID;

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    CHK_RET(SendRequest(requestType, exchangeDataForSend_));

    CHK_RET(WaitForAllRemoteComplete(RequestType::SET_REMOTE_BARE_TGID_ACK));
    if (remotePids_.size() != mapDevPhyIdconnectedSockets_.size()) {
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][SetRemoteTgid] tgid exchange failed recv pids count[%lu]", remotePids_.size());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::DeInit()
{
    CHK_PRT_RET(isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][DeInit] single rank communicator"), HCCL_SUCCESS);
    std::unique_lock<std::mutex> lock(commRefCntLock_);
    threadRun_ = false;
    if (innerThread_) {
        if (innerThread_->joinable()) {
            innerThread_->join(); // 等待线程执行后释放资源
        }
    }
    innerThread_ = nullptr;

    if (vnicPortCtx_ != nullptr) {
        HcclNetCloseDev(vnicPortCtx_);
        vnicPortCtx_ = nullptr;
    }

    std::unique_lock<std::shared_mutex> wlock(g_addressMgrMutex_);
    if (addressMgr_ == nullptr) {
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][%s]addressMgr_ is nullptr, no need to deinit. local rank[u32]", __func__, userRank_);
        return HCCL_E_INTERNAL;
    }
    CHK_RET(addressMgr_->DecreCommRefCnt());
    if (addressMgr_->GetCommRefCnt() == 0) {
        addressMgr_.reset();
        HCCL_RUN_INFO("[ZeroCopyMemoryAgent][%s]Release addressMgr_", __func__);
    }
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::SetMemoryRange(void* virPtr, size_t size, size_t alignment, uint64_t flags)
{
    CHK_PRT_RET(
        isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][SetMemoryRange] single rank communicator"), HCCL_SUCCESS);
    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_PRT_RET(
            addressMgr_->SetMemoryRange(devicePhyId_, virPtr, size) != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][SetMemoryRange] invalid set ptr[%p] size[%lu] alignment[%lu] flags[%lu]", virPtr,
                size, alignment, flags),
            HCCL_E_PARA);
    }

    HCCL_INFO(
        "[ZeroCopyMemoryAgent][SetMemoryRange] basePtr[%p] size[%lu] alignment[%lu] flag[%lu]", virPtr, size, alignment,
        flags);
    u8* exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::SET_MEMORY_RANGE;

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    u64 addr = reinterpret_cast<u64>(virPtr);
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, size));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, alignment));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, flags));

    CHK_RET(SendRequest(requestType, exchangeDataForSend_));

    CHK_RET(WaitForAllRemoteComplete(RequestType::SET_MEMORY_RANGE_ACK));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::UnsetMemoryRange(void* virPtr)
{
    CHK_PRT_RET(
        isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][UnsetMemoryRange] single rank communicator"), HCCL_SUCCESS);
    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_PRT_RET(
            !addressMgr_->IsAddressSet(devicePhyId_, virPtr),
            HCCL_ERROR("[ZeroCopyMemoryAgent][UnsetMemoryRange] ptr[%p] is not set memory", virPtr), HCCL_E_PARA);
        CHK_RET(addressMgr_->UnsetMemoryRange(devicePhyId_, virPtr));
    }

    HCCL_INFO("[ZeroCopyMemoryAgent][UnsetMemoryRange] basePtr[%p]", virPtr);
    u8* exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::UNSET_MEMORY_RANGE;
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    u64 addr = reinterpret_cast<u64>(virPtr);
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));

    CHK_RET(SendRequest(requestType, exchangeDataForSend_));

    CHK_RET(WaitForAllRemoteComplete(RequestType::UNSET_MEMORY_RANGE_ACK));
    return HCCL_SUCCESS;
}

HcclResult
ZeroCopyMemoryAgent::ActivateCommMemory(void* virPtr, size_t size, size_t offset, void* memHandle, uint64_t flags)
{
    CHK_PRT_RET(
        isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][ActivateCommMemory] single rank communicator"), HCCL_SUCCESS);
    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_PRT_RET(
            !addressMgr_->IsInSetAddressRange(devicePhyId_, virPtr, size),
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][ActivateCommMemory] input ptr[%p] size[%lu] is not in set address range", virPtr,
                size),
            HCCL_E_PARA);
        CHK_PRT_RET(
            addressMgr_->IsOverlapWithActivateAddr(virPtr, size),
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][ActivateCommMemory] input ptr[%p] size[%lu] overlap with activate memory",
                virPtr, size),
            HCCL_E_PARA);
    }

    HCCL_INFO(
        "[ZeroCopyMemoryAgent][ActivateCommMemory] virPtr[%p] size[%lu] offset[%lu] memHandle[%p], flags[%lu]", virPtr,
        size, offset, memHandle, flags);
    CHK_RET(SetRemoteTgid());

    uint64_t shareableHandle;
    aclrtMemHandleType handleType = ACL_MEM_HANDLE_TYPE_NONE;
    aclError ret = ACL_SUCCESS;
    ret = aclrtMemExportToShareableHandle(memHandle, handleType, 0, &shareableHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][ActivateCommMemory] aclrtMemExportToShareableHandle handle[%p] type[%d] flags[%llu] "
            "failed, ret[%d]",
            memHandle, handleType, 0, ret),
        HCCL_E_RUNTIME);
    ret = aclrtMemSetPidToShareableHandle(shareableHandle, remotePids_.data(), remotePids_.size());
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][ActivateCommMemory] aclrtMemSetPidToShareableHandle shareableHandl[%llu]",
            " failed, ret[%d]", shareableHandle, ret),
        HCCL_E_RUNTIME);

    HCCL_INFO(
        "[ZeroCopyMemoryAgent][ActivateCommMemory] dev[%u] export shareableHandle[%lu]", devicePhyId_, shareableHandle);
    u8* exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::ACTIVATE_COMM_MEMORY;
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    u64 addr = reinterpret_cast<u64>(virPtr);
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, size));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, offset));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, shareableHandle));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, flags));

    CHK_RET(SendRequest(requestType, exchangeDataForSend_));

    CHK_RET(WaitForAllRemoteComplete(RequestType::ACTIVATE_COMM_MEMORY_ACK));
    std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
    CHK_RET(CheckAddressMgrWithoutLock());
    CHK_RET(addressMgr_->ActivateCommMemoryAddr(virPtr, size));

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::DeactivateCommMemory(void* virPtr)
{
    CHK_PRT_RET(
        isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][DeactivateCommMemory] single rank communicator"), HCCL_SUCCESS);
    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_PRT_RET(
            !addressMgr_->IsActivateCommMemoryAddr(virPtr, 1),
            HCCL_ERROR("[ZeroCopyMemoryAgent][DeactivateCommMemory] input ptr[%p] is not activate", virPtr),
            HCCL_E_PARA);

        HCCL_INFO("[ZeroCopyMemoryAgent][DeactivateCommMemory] virPtr[%p]", virPtr);
        CHK_RET(addressMgr_->DeactivateCommMemoryAddr(virPtr));
    }

    u8* exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::DEACTIVATE_COMM_MEMORY;
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    u64 addr = reinterpret_cast<u64>(virPtr);
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));

    CHK_RET(SendRequest(requestType, exchangeDataForSend_));

    CHK_RET(WaitForAllRemoteComplete(RequestType::DEACTIVATE_COMM_MEMORY_ACK));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::BarrierClose()
{
    CHK_PRT_RET(isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][BarrierClose] single rank communicator"), HCCL_SUCCESS);

    HCCL_RUN_INFO("[ZeroCopyMemoryAgent][BarrierClose] [%s] ready to barrier close", identifier_.c_str());
    u8* exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::BARRIER_CLOSE;
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    CHK_RET(SendRequest(requestType, exchangeDataForSend_));

    CHK_RET(WaitForAllRemoteComplete(RequestType::BARRIER_CLOSE_ACK));

    return HCCL_SUCCESS;
}

bool ZeroCopyMemoryAgent::IsActivateCommMemoryAddr(void* virPtr, u64 length)
{
    std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
    if (addressMgr_ == nullptr) {
        HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent is not init.", __func__);
        return false;
    }
    return addressMgr_->IsActivateCommMemoryAddr(virPtr, length);
}

HcclResult ZeroCopyMemoryAgent::GetRingBufferAddr(u64& bufferPtr, u64& headPtr, u64& tailPtr)
{
    std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
    CHK_RET(CheckAddressMgrWithoutLock());
    addressMgr_->GetRingBufferAddr(bufferPtr, headPtr, tailPtr);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::CheckAddressMgrWithoutLock()
{
    CHK_PRT_RET(
        addressMgr_ == nullptr,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent "
            "is not init.",
            __func__),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::WaitForAllRemoteComplete(RequestType requestType)
{
    bool useBarrier = NeedBarrier(requestType);
    if (useBarrier) {
        reqMsgDeliverCnt_++;
    }

    u32 expectedNum = mapDevPhyIdconnectedSockets_.size();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    std::unique_lock<std::mutex> lock(dfxMutex_);
    waitCompleteCv_.wait_for(lock, timeout);
    if ((reqMsgCounter_[static_cast<int>(requestType)] == expectedNum)
        && (!useBarrier || (useBarrier && reqMsgDeliverCnt_ <= reqMsgFinishCnt_))) {
        reqMsgCounter_[static_cast<int>(requestType)] = 0;
        reqMsgFinishedRanks_[static_cast<int>(requestType)].clear();
        return HCCL_SUCCESS;
    }

    HCCL_ERROR(
        "[Wait][RemoteComplete %s] dev[%u] errNo[0x%016llx] timeout[%d s] completeCount[%u] %s",
        GetReadableRequestType(requestType), devicePhyId_, HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), timeout,
        reqMsgCounter_[static_cast<int>(requestType)].load(), DumpFinishInfo(requestType).c_str());
    reqMsgCounter_[static_cast<int>(requestType)] = 0;
    reqMsgFinishedRanks_[static_cast<int>(requestType)].clear();
    return HCCL_E_TCP_TRANSFER;
}

HcclResult ZeroCopyMemoryAgent::ParseSetMemoryRange(u8*& exchangeDataPtr, u32& exchangeDataBlankSize)
{
    CHK_RET(CheckAddressMgrWithoutLock());
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u64 addr;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, addr));

    size_t size;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, size));

    size_t alignment;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, alignment));

    uint64_t flags;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, flags));

    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        devicePhyId >= maxDeviceNum,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][ParseSetMemoryRange] devicePhyId[%u] is exceed max device num[%u]", devicePhyId,
            maxDeviceNum),
        HCCL_E_PARA);

    void* remoteAddrBase = reinterpret_cast<void*>(addr);
    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_PRT_RET(
            addressMgr_->IsAddressSet(devicePhyId, remoteAddrBase),
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][ParseSetMemoryRange] devicePhyId[%u] had set addr [%p]", devicePhyId,
                remoteAddrBase),
            HCCL_E_PARA);
    }

    void* devPtr = nullptr;
    void* devAddr = nullptr;
    aclError ret = aclrtReserveMemAddress(&devPtr, size, alignment, devAddr, flags);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][ParseSetMemoryRange] rtReserve Memory failed, "
            "return[%d], devPtr[%p] size[%llu] alignment[%llu] devAddr[%p] flags[%llu]",
            ret, devPtr, size, alignment, devAddr, flags),
        HCCL_E_RUNTIME);

    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_RET(addressMgr_->AddLocalIpc2RemoteAddr(devicePhyId, devPtr, reinterpret_cast<void*>(addr), size));
    }

    CHK_RET(SendAckAfterParse(RequestType::SET_MEMORY_RANGE, RequestType::SET_MEMORY_RANGE_ACK, devicePhyId));

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::SendAckAfterParse(
    RequestType requestType, RequestType ackType, u32 remoteDevicePhyId, void* extraData, u64 extraDataLen)
{
    u8* exchangeDataAckPtr = exchangeDataForAck_[remoteDevicePhyId].data();
    u32 exchangeDataAckBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    CHK_RET(ConstructData(exchangeDataAckPtr, exchangeDataAckBlankSize, ackType));

    CHK_RET(ConstructData(exchangeDataAckPtr, exchangeDataAckBlankSize, devicePhyId_));

    if (extraData != nullptr && extraDataLen != 0) {
        CHK_RET(ConstructData(exchangeDataAckPtr, exchangeDataAckBlankSize, extraData, extraDataLen));
    }

    // 不需要进行barrier，那么我们每处理一个请求就回复一个请求
    if (!NeedBarrier(requestType)) {
        CHK_PRT_RET(
            SendRequest(ackType, exchangeDataForAck_[remoteDevicePhyId], remoteDevicePhyId) != HCCL_SUCCESS,
            HCCL_WARNING("[ZeroCopyMemoryAgent][SendAckAfterParse] failed, remote[%u]", remoteDevicePhyId),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    // 需要进行barrier的请求，我们先统计一下收到的请求数目，等于链接数才算收完所有
    u32 expectedNum = mapDevPhyIdconnectedSockets_.size();
    std::unique_lock<std::mutex> dfxLock(dfxMutex_);
    u32 counter = ++reqMsgCounter_[static_cast<int>(requestType)];
    HCCL_INFO(
        "[ZeroCopyMemoryAgent][SendAckAfterParse] requestType[%d] counter %u expect %u", requestType, counter,
        expectedNum);
    if (counter < expectedNum) {
        return HCCL_SUCCESS;
    } else {
        reqMsgCounter_[static_cast<int>(requestType)] = 0;
        reqMsgFinishCnt_++;
        // 释放锁后再发送，避免阻塞SendRequest
        dfxLock.unlock();

        // 我们统一将所有的请求一次性都发送过去
        CHK_PRT_RET(
            SendRequest(ackType, exchangeDataForAck_[remoteDevicePhyId]) != HCCL_SUCCESS,
            HCCL_WARNING("[ZeroCopyMemoryAgent][SendAckAfterParse] failed, remote[all]"), HCCL_E_INTERNAL);
    }

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseRemoteAck(RequestType requestType, u32 remoteRank)
{
    bool useBarrier = NeedBarrier(requestType);
    std::unique_lock<std::mutex> dfxLock(dfxMutex_);
    reqMsgFinishedRanks_[static_cast<int>(requestType)].insert(remoteRank);
    u32 counter = ++reqMsgCounter_[static_cast<int>(requestType)];
    if ((counter == mapDevPhyIdconnectedSockets_.size())
        && (!useBarrier || (useBarrier && reqMsgDeliverCnt_ <= reqMsgFinishCnt_))) {
        waitCompleteCv_.notify_all();
    }
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseUnsetMemoryRange(u8*& exchangeDataPtr, u32& exchangeDataBlankSize)
{
    CHK_RET(CheckAddressMgrWithoutLock());
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u64 addr;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, addr));

    LocalIpc2RemoteAddr mapAddr;
    void* remoteAddr = reinterpret_cast<void*>(addr);
    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_PRT_RET(
            addressMgr_->GetLocalIpc2RemoteAddr(devicePhyId, remoteAddr, mapAddr) != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][ParseUnsetMemoryRange] device[%u] not set addr [%p]", devicePhyId, remoteAddr),
            HCCL_E_PARA);
        CHK_RET(addressMgr_->DelLocalIpc2RemoteAddr(devicePhyId, reinterpret_cast<void*>(mapAddr.remoteAddr)));
    }

    void* devPtr = reinterpret_cast<void*>(mapAddr.localIpcAddr);
    aclError ret = aclrtReleaseMemAddress(devPtr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][ParseUnsetMemoryRange]rtRelease Memory failed, "
            "return[%d], devPtr[%p]",
            ret, devPtr),
        HCCL_E_RUNTIME);

    CHK_RET(SendAckAfterParse(RequestType::UNSET_MEMORY_RANGE, RequestType::UNSET_MEMORY_RANGE_ACK, devicePhyId));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseBareTgid(u8*& exchangeDataPtr, u32& exchangeDataBlankSize)
{
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    // 获取本端的ack，然后通过ack返回给对端
    int32_t tgid = 0;
    aclError ret = aclrtDeviceGetBareTgid(&tgid);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[ZeroCopyMemoryAgent][ParseBareTgid] get tgid failed, ret[%d]", ret),
        HCCL_E_RUNTIME);

    HCCL_INFO(
        "[ZeroCopyMemoryAgent][ParseBareTgid] dev[%u] tgid[%d] to remoteDev[%u]", devicePhyId_, tgid, devicePhyId);
    CHK_RET(SendAckAfterParse(
        RequestType::SET_REMOTE_BARE_TGID, RequestType::SET_REMOTE_BARE_TGID_ACK, devicePhyId, &tgid, sizeof(tgid)));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseBareTgidAck(u8*& exchangeDataPtr, u32& exchangeDataBlankSize)
{
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u32 tgid;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, tgid));

    HCCL_INFO("[ZeroCopyMemoryAgent][ParseBareTgidAck] recv dev[%u] tgid[%u]", devicePhyId, tgid);
    remotePids_.emplace_back(tgid);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseBarrierCloseAck(u8*& exchangeDataPtr, u32& exchangeDataBlankSize)
{
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u32 tgid;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, tgid));

    receivedBarrierCloseAck_.insert(devicePhyId);
    HCCL_RUN_INFO(
        "[ZeroCopyMemoryAgent][ParseBarrierCloseAck] [%s] recv dev[%u] barrier close ack, so we stop this socket's "
        "recv",
        identifier_.c_str(), devicePhyId, tgid);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseActivateCommMemory(u8*& exchangeDataPtr, u32& exchangeDataBlankSize)
{
    CHK_RET(CheckAddressMgrWithoutLock());
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u64 addr;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, addr));

    size_t size;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, size));

    size_t offset;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, offset));

    size_t shareableHandle;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, shareableHandle));

    size_t flags;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, flags));

    LocalIpc2RemoteAddr mapAddr;
    void* remoteAddr = reinterpret_cast<void*>(addr);
    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_PRT_RET(
            (addressMgr_->GetLocalIpc2RemoteAddr(devicePhyId, remoteAddr, mapAddr) != HCCL_SUCCESS),
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][ParseActivateCommMemory] address may not be reserved in device[%u]",
                devicePhyId),
            HCCL_E_PARA);
    }

    HCCL_INFO(
        "[ZeroCopyMemoryAgent][ParseActivateCommMemory] prepare import from dev[%u] shareableHandle[%llu]", devicePhyId,
        shareableHandle);
    u64 actualAddr = mapAddr.localIpcAddr + (addr - mapAddr.remoteAddr);
    void* devPtr = reinterpret_cast<void*>(actualAddr);
    CHK_PRT_RET(
        actualAddr + size > mapAddr.localIpcAddr + mapAddr.length,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][ParseActivateCommMemory] remote addr[0x%lx] size[%llu] exceed memory range", addr,
            size),
        HCCL_E_PARA);
    aclError ret = ACL_SUCCESS;
    void* pHandle = nullptr;
    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_PRT_RET(
            addressMgr_->IsOverlapWithActivateAddr(devPtr, size),
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][ParseActivateCommMemory] remote addr[0x%lx] size[%llu] devPtr[%p] is overlap",
                addr, size, devPtr),
            HCCL_E_PARA);
        CHK_RET(addressMgr_->ActivateCommMemoryAddr(devPtr, size));
    }
    ret = aclrtMemImportFromShareableHandle(shareableHandle, deviceLogicId_, &pHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][ParseActivateCommMemory] import shareableHandle[%llu] dev[%d] failed, ret[%d]",
            shareableHandle, deviceLogicId_, ret),
        HCCL_E_RUNTIME);

    ret = aclrtMapMem(devPtr, size, offset, pHandle, flags);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][ParseActivateCommMemory] map dev[%p] size[%llu] offset[%llu] handle[%p]",
            " flag[%llu] failed, ret[%d]", devPtr, size, offset, pHandle, flags, ret),
        HCCL_E_RUNTIME);

    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_RET(addressMgr_->AddRemoteImportAddr(devPtr, pHandle));
    }

    CHK_RET(SendAckAfterParse(RequestType::ACTIVATE_COMM_MEMORY, RequestType::ACTIVATE_COMM_MEMORY_ACK, devicePhyId));

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseDeactivateCommMemory(u8*& exchangeDataPtr, u32& exchangeDataBlankSize)
{
    CHK_RET(CheckAddressMgrWithoutLock());
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u64 addr;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, addr));

    LocalIpc2RemoteAddr mapAddr;
    void* remoteAddr = reinterpret_cast<void*>(addr);
    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_PRT_RET(
            (addressMgr_->GetLocalIpc2RemoteAddr(devicePhyId, remoteAddr, mapAddr) != HCCL_SUCCESS),
            HCCL_ERROR(
                "[ZeroCopyMemoryAgent][ParseDeactivateCommMemory] address [%p] not be set in device[%u]", remoteAddr,
                devicePhyId),
            HCCL_E_PARA);
    }

    u64 actualAddr = mapAddr.localIpcAddr + (addr - mapAddr.remoteAddr);
    void* devPtr = reinterpret_cast<void*>(actualAddr);
    void* handle = nullptr;
    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_RET(addressMgr_->DeactivateCommMemoryAddr(devPtr));
        CHK_RET(addressMgr_->GetRemoteImportAddr(devPtr, handle));
    }

    aclError ret = ACL_SUCCESS;
    ret = aclrtUnmapMem(devPtr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][ParseDeactivateCommMemory] aclrtUnmapMem dev[%p] failed, ret[%d]", devPtr, ret),
        HCCL_E_RUNTIME);
    ret = aclrtFreePhysical(handle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[ZeroCopyMemoryAgent][ParseDeactivateCommMemory] aclrtFreePhysical handle[%p] failed, ret[%d]", handle,
            ret),
        HCCL_E_RUNTIME);

    {
        std::shared_lock<std::shared_mutex> rlock(g_addressMgrMutex_);
        CHK_RET(CheckAddressMgrWithoutLock());
        CHK_RET(addressMgr_->DelRemoteImportAddr(devPtr));
    }

    CHK_RET(
        SendAckAfterParse(RequestType::DEACTIVATE_COMM_MEMORY, RequestType::DEACTIVATE_COMM_MEMORY_ACK, devicePhyId));

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseBarrierClose(u8*& exchangeDataPtr, u32& exchangeDataBlankSize)
{
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));
    HCCL_INFO("[ZeroCopyMemoryAgent][ParseBarrierClose] recv dev[%u] barrier close", devicePhyId);

    receivedBarrierClose_.insert(devicePhyId);
    CHK_RET(SendAckAfterParse(RequestType::BARRIER_CLOSE, RequestType::BARRIER_CLOSE_ACK, devicePhyId));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseReceivedRequest(std::vector<u8>& receivedData, u32 remoteRank)
{
    u8* exchangeDataPtr = receivedData.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    HcclResult ret = HCCL_SUCCESS;
    switch (requestType) {
        case RequestType::SET_MEMORY_RANGE:
            ret = ParseSetMemoryRange(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::UNSET_MEMORY_RANGE:
            ret = ParseUnsetMemoryRange(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::ACTIVATE_COMM_MEMORY:
            ret = ParseActivateCommMemory(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::DEACTIVATE_COMM_MEMORY:
            ret = ParseDeactivateCommMemory(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::SET_REMOTE_BARE_TGID:
            ret = ParseBareTgid(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::BARRIER_CLOSE:
            ret = ParseBarrierClose(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::SET_REMOTE_BARE_TGID_ACK:
            ret = ParseBareTgidAck(exchangeDataPtr, exchangeDataBlankSize);
            ParseRemoteAck(requestType, remoteRank);
            break;
        case RequestType::SET_MEMORY_RANGE_ACK:
        case RequestType::UNSET_MEMORY_RANGE_ACK:
        case RequestType::ACTIVATE_COMM_MEMORY_ACK:
        case RequestType::DEACTIVATE_COMM_MEMORY_ACK:
            ParseRemoteAck(requestType, remoteRank);
            break;
        case RequestType::BARRIER_CLOSE_ACK:
            ret = ParseBarrierCloseAck(exchangeDataPtr, exchangeDataBlankSize);
            ParseRemoteAck(requestType, remoteRank);
            break;
        default:
            HCCL_ERROR("[Parse][ReceivedRequest] invalid RequestType[%d]", requestType);
            ret = HCCL_E_INTERNAL;
            break;
    }
    return ret;
}

std::string ZeroCopyMemoryAgent::DumpFinishInfo(RequestType requestType)
{
    auto& finishedRanks = reqMsgFinishedRanks_[static_cast<int>(requestType)];

    std::string msg = "Expect [";
    for (auto& info : rankInfoList_) {
        msg += std::to_string(info.userRank) + " ";
    }

    msg += "] Actual [";
    for (auto& rank : finishedRanks) {
        msg += std::to_string(rank) + " ";
    }

    msg += "]";
    finishedRanks.clear();

    return msg;
}

bool ZeroCopyMemoryAgent::IsPaused() const { return !threadRun_ || isPaused_; }

bool ZeroCopyMemoryAgent::IsResumed() const { return !threadRun_ || !isPaused_; }

void ZeroCopyMemoryAgent::CheckSnapshotStatus()
{
    auto snapshotStatus = SnapshotControl::GetInstance(deviceLogicId_).GetStatus();
    if (isPaused_ && snapshotStatus == SnapshotStatus::POST_SNAPSHOT) {
        isPaused_ = false;
        HCCL_RUN_INFO(
            "[ZeroCopyMemoryAgent][CheckSnapshotStatus] detect snapshot post-processing, "
            "zero-copy memory agent is resumed, deviceLogicId[%d].",
            deviceLogicId_);
    } else if (!isPaused_ && snapshotStatus == SnapshotStatus::PRE_SNAPSHOT) {
        isPaused_ = true;
        HCCL_RUN_INFO(
            "[ZeroCopyMemoryAgent][CheckSnapshotStatus] detect snapshot pre-processing, "
            "zero-copy memory agent is paused, deviceLogicId[%d].",
            deviceLogicId_);
    }
}

} // namespace hccl
