/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_heterog_event_roce.h"

#include "log.h"
#include "adapter_hal.h"
#include "dlhal_function.h"

using namespace std;
namespace hccl {
constexpr u32 MAX_CQECOUNT_ALLLINK = 64;
constexpr s32 PROTOCOL_TYPE = 0;

HcclReceivedEnvelope TransportHeterogEventRoce::gReceivedEnvelopes;
std::mutex TransportHeterogEventRoce::gReceivedEnvelopesMutex;

std::vector<std::atomic<int>> TransportHeterogEventRoce::gCqeCounterPerEvent(MAX_CQECOUNT_ALLLINK);
std::vector<std::vector<void*>> TransportHeterogEventRoce::gAllLinkVec(MAX_CQECOUNT_ALLLINK);
std::mutex TransportHeterogEventRoce::gAllLinkVecSendCompMutex;
std::mutex TransportHeterogEventRoce::gAllLinkVecRecvReqMutex;
std::mutex TransportHeterogEventRoce::gAllLinkVecRecvCompMutex;

std::mutex TransportHeterogEventRoce::gPollTagRqLock;
std::mutex TransportHeterogEventRoce::gPollDataRqLock;
std::mutex TransportHeterogEventRoce::gPollDataSqLock;
std::mutex TransportHeterogEventRoce::gEschedAckMutex;

u32 TransportHeterogEventRoce::gEschedAckRef = 0;
u32 TransportHeterogEventRoce::gAllLinkInitCount = 0;
u32 TransportHeterogEventRoce::recvRequestEvent = 0;
u32 TransportHeterogEventRoce::sendCompletionEvent = 0;
u32 TransportHeterogEventRoce::recvCompletionEvent = 0;

constexpr u32 RECV_WQE_BATCH_NUM = 8 * 1024;
constexpr u32 RECV_WQE_NUM_THRESHOLD = 4 * 1024;
constexpr u32 RECV_WQE_BATCH_SUPPLEMENT = 2 * 1024;
constexpr u32 MAX_WR_NUM = 1023;
constexpr s32 TAG_QP_APPEND = 1;
constexpr s32 DATA_QP_APPEND = 2;

atomic<u32> g_tagRecvWqeNum;  // qp0上的recv wqe的数量,recv端消耗
atomic<u32> g_dataRecvWqeNum; // qp1上的recv wqe的数量,send端消耗
std::shared_mutex TransportHeterogEventRoce::gQpnMapMutex;
map<u32, TransportHeterogEventRoce*> TransportHeterogEventRoce::gQpnToTransportMap; // tag qpn和transport映射
map<u32, atomic<u32>> TransportHeterogEventRoce::gQpnToSqMaxWrMap;                  // data qpn和sq max wr深度映射
bool TransportHeterogEventRoce::gNeedRepoEvent = true;

void TransportHeterogEventRoce::EschedAckCallbackRecvRequest(
    unsigned int devId, unsigned int subeventId, u8* msg, unsigned int msgLen)
{
    (void)subeventId;
    (void)msg;
    (void)msgLen;
    TransportHeterogEventRoce::EschedAckCallback(devId, HCCL_EVENT_RECV_REQUEST_MSG);
}

void TransportHeterogEventRoce::EschedAckCallbackSendCompletion(
    unsigned int devId, unsigned int subeventId, u8* msg, unsigned int msgLen)
{
    (void)subeventId;
    (void)msg;
    (void)msgLen;
    TransportHeterogEventRoce::EschedAckCallback(devId, HCCL_EVENT_SEND_COMPLETION_MSG);
}

void TransportHeterogEventRoce::EschedAckCallbackRecvCompletion(
    unsigned int devId, unsigned int subeventId, u8* msg, unsigned int msgLen)
{
    (void)subeventId;
    (void)msg;
    (void)msgLen;
    TransportHeterogEventRoce::EschedAckCallback(devId, HCCL_EVENT_RECV_COMPLETION_MSG);
}

TransportHeterogEventRoce::TransportHeterogEventRoce(
    const std::string& transTag, HcclIpAddress& selfIp, HcclIpAddress& peerIp, u32 peerPort, u32 selfPort,
    const TransportResourceInfo& transportResourceInfo)
    : TransportHeterogRoce(transTag, selfIp, peerIp, peerPort, selfPort, transportResourceInfo)
{
    tagQpInfo_.srq = transportResourceInfo.tagSrqInfo.srq;
    tagQpInfo_.srqCq = transportResourceInfo.tagSrqInfo.srqCq;
    tagQpInfo_.srqContext = transportResourceInfo.tagSrqInfo.context;
    dataQpInfo_.srq = transportResourceInfo.dataSrqInfo.srq;
    dataQpInfo_.srqCq = transportResourceInfo.dataSrqInfo.srqCq;
    dataQpInfo_.srqContext = transportResourceInfo.dataSrqInfo.context;
    srqInit_ = ((tagQpInfo_.srq != nullptr) && (dataQpInfo_.srq != nullptr));
}
TransportHeterogEventRoce::TransportHeterogEventRoce(const TransportResourceInfo& transportResourceInfo)
    : TransportHeterogRoce(transportResourceInfo)
{
    tagQpInfo_.srq = transportResourceInfo.tagSrqInfo.srq;
    tagQpInfo_.srqCq = transportResourceInfo.tagSrqInfo.srqCq;
    tagQpInfo_.srqContext = transportResourceInfo.tagSrqInfo.context;
    dataQpInfo_.srq = transportResourceInfo.dataSrqInfo.srq;
    dataQpInfo_.srqCq = transportResourceInfo.dataSrqInfo.srqCq;
    dataQpInfo_.srqContext = transportResourceInfo.dataSrqInfo.context;
    srqInit_ = ((tagQpInfo_.srq != nullptr) && (dataQpInfo_.srq != nullptr));
}

TransportHeterogEventRoce::~TransportHeterogEventRoce()
{
    HcclResult ret = Deinit();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("TransportHeterogEventRoce:: destructor Deinit fail.");
    }
}

HcclResult TransportHeterogEventRoce::RegisterEschedAckCallback()
{
    std::lock_guard<std::mutex> lock(gEschedAckMutex);
    if (gEschedAckRef == 0) {
        CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());

        recvRequestEvent = HCCL_EVENT_RECV_REQUEST_MSG;
        sendCompletionEvent = HCCL_EVENT_SEND_COMPLETION_MSG;
        recvCompletionEvent = HCCL_EVENT_RECV_COMPLETION_MSG;

        CHK_RET(hrtHalEschedRegisterAckFunc(HCCL_EVENT_RECV_REQUEST_MSG, EschedAckCallbackRecvRequest));
        CHK_RET(hrtHalEschedRegisterAckFunc(HCCL_EVENT_SEND_COMPLETION_MSG, EschedAckCallbackSendCompletion));
        CHK_RET(hrtHalEschedRegisterAckFunc(HCCL_EVENT_RECV_COMPLETION_MSG, EschedAckCallbackRecvCompletion));

        gCqeCounterPerEvent[HCCL_EVENT_RECV_REQUEST_MSG] = 0;
        gCqeCounterPerEvent[HCCL_EVENT_SEND_COMPLETION_MSG] = 0;
        gCqeCounterPerEvent[HCCL_EVENT_RECV_COMPLETION_MSG] = 0;
    }
    gEschedAckRef++;

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::DeregisterEschedAckCallback()
{
    std::lock_guard<std::mutex> lock(gEschedAckMutex);
    if (gEschedAckRef > 0) {
        gEschedAckRef--;
    } else if (gEschedAckRef == 0) {
        HCCL_WARNING("TransportHeterogEventRoce:: EschedAckCallback has been deregistered.");
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::InitAllLinkVec()
{
    gAllLinkInitCount++;
    std::unique_lock<std::mutex> lockRecvReq(gAllLinkVecRecvReqMutex);
    gAllLinkVec[HCCL_EVENT_RECV_REQUEST_MSG].push_back(this);
    lockRecvReq.unlock();

    std::unique_lock<std::mutex> lockSendComp(gAllLinkVecSendCompMutex);
    gAllLinkVec[HCCL_EVENT_SEND_COMPLETION_MSG].push_back(this);
    lockSendComp.unlock();

    std::unique_lock<std::mutex> lockRecvComp(gAllLinkVecRecvCompMutex);
    gAllLinkVec[HCCL_EVENT_RECV_COMPLETION_MSG].push_back(this);
    lockRecvComp.unlock();

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::DeinitAllLinkVec()
{
    if (gAllLinkInitCount > 0) {
        std::unique_lock<std::mutex> lockRecvReq(gAllLinkVecRecvReqMutex);
        gAllLinkVec[HCCL_EVENT_RECV_REQUEST_MSG].clear();
        lockRecvReq.unlock();

        std::unique_lock<std::mutex> lockSendComp(gAllLinkVecSendCompMutex);
        gAllLinkVec[HCCL_EVENT_SEND_COMPLETION_MSG].clear();
        lockSendComp.unlock();

        std::unique_lock<std::mutex> lockRecvComp(gAllLinkVecRecvCompMutex);
        gAllLinkVec[HCCL_EVENT_RECV_COMPLETION_MSG].clear();
        lockRecvComp.unlock();
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::EraseTransportFromAllLinkVec(void* transportPtr)
{
    std::unique_lock<std::mutex> lockRecvReq(gAllLinkVecRecvReqMutex);
    auto recvReqIter = find(
        gAllLinkVec[HCCL_EVENT_RECV_REQUEST_MSG].begin(), gAllLinkVec[HCCL_EVENT_RECV_REQUEST_MSG].end(), transportPtr);
    if (recvReqIter != gAllLinkVec[HCCL_EVENT_RECV_REQUEST_MSG].end()) {
        gAllLinkVec[HCCL_EVENT_RECV_REQUEST_MSG].erase(recvReqIter);
    }
    lockRecvReq.unlock();

    std::unique_lock<std::mutex> lockSendComp(gAllLinkVecSendCompMutex);
    auto sendCompIter = find(
        gAllLinkVec[HCCL_EVENT_SEND_COMPLETION_MSG].begin(), gAllLinkVec[HCCL_EVENT_SEND_COMPLETION_MSG].end(),
        transportPtr);
    if (sendCompIter != gAllLinkVec[HCCL_EVENT_SEND_COMPLETION_MSG].end()) {
        gAllLinkVec[HCCL_EVENT_SEND_COMPLETION_MSG].erase(sendCompIter);
    }
    lockSendComp.unlock();

    std::unique_lock<std::mutex> lockRecvComp(gAllLinkVecRecvCompMutex);
    auto recvCompIter = find(
        gAllLinkVec[HCCL_EVENT_RECV_COMPLETION_MSG].begin(), gAllLinkVec[HCCL_EVENT_RECV_COMPLETION_MSG].end(),
        transportPtr);
    if (recvCompIter != gAllLinkVec[HCCL_EVENT_RECV_COMPLETION_MSG].end()) {
        gAllLinkVec[HCCL_EVENT_RECV_COMPLETION_MSG].erase(recvCompIter);
    }
    lockRecvComp.unlock();

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::Init()
{
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    CHK_RET(RegisterEschedAckCallback());
    CHK_RET(TransportHeterogRoce::Init());

    CHK_RET(InitAllLinkVec());
    isDeinited_ = false;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::Deinit()
{
    if (isDeinited_) {
        return HCCL_SUCCESS;
    }
    CHK_RET(DeregisterEschedAckCallback());
    CHK_RET(DeinitAllLinkVec());

    CHK_RET(TransportHeterogRoce::Deinit());
    isDeinited_ = true;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::Isend(
    const TransData& sendData, const TransportEndPointParam& epParam, HcclRequestInfo*& request)
{
    HcclResult ret = TransportHeterogRoce::Isend(sendData, epParam, request);
    if (ret != HCCL_SUCCESS && request != nullptr) {
        CHK_RET(FreeRequest(*request));
    }
    return ret;
}

HcclResult TransportHeterogEventRoce::Improbe(
    const TransportEndPointParam& epParam, s32& matched, HcclMessageInfo*& msg, HcclStatus& status)
{
    CHK_RET(TransportHeterogRoce::Improbe(epParam, matched, msg, status));
    if (matched == HCCL_IMPROBE_COMPLETED) {
        gCqeCounterPerEvent[recvRequestEvent].fetch_sub(1);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::Imrecv(const TransData& recvData, HcclMessageInfo& msg, HcclRequestInfo*& request)
{
    CHK_RET(TransportHeterogRoce::Imrecv(recvData, msg, request));
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::Test(HcclRequestInfo& request, s32& flag, HcclStatus& compState)
{
    // 建链未完成时，继续推进建链流程；
    if (GetState() != ConnState::CONN_STATE_COMPLETE) {
        CHK_RET(ConnectAsync());
    }

    if (GetState() == ConnState::CONN_STATE_COMPLETE || GetState() == ConnState::CONN_STATE_FLUSH_QUEUE) {
        if (request.transportRequest.requestType == HcclRequestType::HCCL_REQUEST_SEND) {
            CHK_RET(PullSendStatus());
        } else if (request.transportRequest.requestType == HcclRequestType::HCCL_REQUEST_RECV) {
            CHK_RET(PullRecvStatus());
        } else {
            HCCL_ERROR("[HcclTest] requestType[%u] is invalid", request.transportRequest.requestType);
            return HCCL_E_PARA;
        }
    }

    return QueryRequestStatus(request, flag, compState);
}

HcclResult TransportHeterogEventRoce::PullRecvRequestStatus(bool allowNotify)
{
    std::unique_lock<std::mutex> lock(gPollTagRqLock, std::defer_lock);
    if (lock.try_lock()) {
        CHK_RET(TransportHeterogRoce::PullRecvRequestStatus(allowNotify));
    } else {
        if ((allowNotify) && (gCqeCounterPerEvent[recvRequestEvent] <= 0)) {
            CHK_RET(hrtIbvReqNotifyCq(tagQpInfo_.recvCq, 0));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::PullSendStatus(bool allowNotify)
{
    std::unique_lock<std::mutex> lock(gPollDataRqLock, std::defer_lock);
    if (lock.try_lock()) {
        CHK_RET(TransportHeterogRoce::PullSendStatus(allowNotify));
    } else {
        if ((allowNotify) && (gCqeCounterPerEvent[sendCompletionEvent] <= 0)) {
            CHK_RET(hrtIbvReqNotifyCq(dataQpInfo_.recvCq, 0));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::PullRecvStatus(bool allowNotify)
{
    std::unique_lock<std::mutex> lock(gPollDataSqLock, std::defer_lock);
    if (lock.try_lock()) {
        CHK_RET(TransportHeterogRoce::PullRecvStatus(allowNotify));
    } else {
        if ((allowNotify) && (gCqeCounterPerEvent[recvCompletionEvent] <= 0)) {
            CHK_RET(hrtIbvReqNotifyCq(dataQpInfo_.sendCq, 0));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::ParseErrorTagSqe(const struct ibv_wc* wc, int index)
{
    CHK_RET(TransportHeterogRoce::ParseErrorTagSqe(wc, index));
    gCqeCounterPerEvent[sendCompletionEvent].fetch_add(1);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::ParseTagRqes(const struct ibv_wc* wc, int num)
{
    if (srqInit_) {
        CHK_RET(ParseTagSrqes(wc, num));
    } else {
        CHK_RET(TransportHeterogRoce::ParseTagRqes(wc, num));
    }

    gCqeCounterPerEvent[recvRequestEvent].fetch_add(num);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::ParseDataRqes(const struct ibv_wc* wc, int num)
{
    if (srqInit_) {
        CHK_RET(ParseDataSrqes(wc, num));
    } else {
        CHK_RET(TransportHeterogRoce::ParseDataRqes(wc, num));
    }
    gCqeCounterPerEvent[sendCompletionEvent].fetch_add(num);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::ParseDataSqes(const struct ibv_wc* wc, int num)
{
    CHK_RET(TransportHeterogRoce::ParseDataSqes(wc, num));
    gCqeCounterPerEvent[recvCompletionEvent].fetch_add(num);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::QueryRequestStatus(HcclRequestInfo& request, s32& flag, HcclStatus& compState)
{
    flag = HCCL_TEST_INCOMPLETED;
    u32 eventType;
    if (request.transportRequest.requestType == HcclRequestType::HCCL_REQUEST_SEND) {
        eventType = sendCompletionEvent;
    } else if (request.transportRequest.requestType == HcclRequestType::HCCL_REQUEST_RECV) {
        eventType = recvCompletionEvent;
    } else {
        HCCL_ERROR("[QueryRequestStatus]requestType is invalid! requestType[%u]", request.transportRequest.requestType);
        return HCCL_E_PARA;
    }

    if (gCqeCounterPerEvent[eventType] > 0) {
        CHK_RET(TransportHeterogRoce::QueryRequestStatus(request, flag, compState));
        if (flag == HCCL_TEST_COMPLETED) {
            gCqeCounterPerEvent[eventType].fetch_sub(1);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::PullRecvRequestStatus(void* transportHandle)
{
    CHK_PTR_NULL(transportHandle);
    TransportHeterogEventRoce* transportPtr = reinterpret_cast<TransportHeterogEventRoce*>(transportHandle);
    if (transportPtr->GetState() == ConnState::CONN_STATE_COMPLETE) {
        CHK_RET(transportPtr->PullRecvRequestStatus(true));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::PullSendStatus(void* transportHandle)
{
    CHK_PTR_NULL(transportHandle);
    TransportHeterogEventRoce* transportPtr = reinterpret_cast<TransportHeterogEventRoce*>(transportHandle);
    if (transportPtr->GetState() == ConnState::CONN_STATE_COMPLETE
        || transportPtr->GetState() == ConnState::CONN_STATE_FLUSH_QUEUE) {
        CHK_RET(transportPtr->PullSendStatus(true));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::PullRecvStatus(void* transportHandle)
{
    CHK_PTR_NULL(transportHandle);
    TransportHeterogEventRoce* transportPtr = reinterpret_cast<TransportHeterogEventRoce*>(transportHandle);
    if (transportPtr->GetState() == ConnState::CONN_STATE_COMPLETE) {
        CHK_RET(transportPtr->PullRecvStatus(true));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::UpdateStatus(u32 eventId)
{
    if (gCqeCounterPerEvent[eventId] > 0) {
        return HCCL_SUCCESS;
    }
    // 检查所有tag的对应cq中有没有cqe，如果没有则继续轮询下一个cq，如果检查到某一cq中存在cqe则退出循环。
    if (eventId == HCCL_EVENT_RECV_REQUEST_MSG) {
        std::unique_lock<std::mutex> lockRecvReq(gAllLinkVecRecvReqMutex);
        for (auto& iterLink : gAllLinkVec[HCCL_EVENT_RECV_REQUEST_MSG]) {
            CHK_RET(PullRecvRequestStatus(iterLink));
            if ((gCqeCounterPerEvent[HCCL_EVENT_RECV_REQUEST_MSG] > 0)) {
                return HCCL_SUCCESS;
            }
        }
        lockRecvReq.unlock();
    } else if (eventId == HCCL_EVENT_SEND_COMPLETION_MSG) {
        std::unique_lock<std::mutex> lockSendComp(gAllLinkVecSendCompMutex);
        for (auto& iterLink : gAllLinkVec[HCCL_EVENT_SEND_COMPLETION_MSG]) {
            CHK_RET(PullSendStatus(iterLink));
            if ((gCqeCounterPerEvent[HCCL_EVENT_SEND_COMPLETION_MSG] > 0)) {
                return HCCL_SUCCESS;
            }
        }
        lockSendComp.unlock();
    } else if (eventId == HCCL_EVENT_RECV_COMPLETION_MSG) {
        std::unique_lock<std::mutex> lockRecvComp(gAllLinkVecRecvCompMutex);
        for (auto& iterLink : gAllLinkVec[HCCL_EVENT_RECV_COMPLETION_MSG]) {
            CHK_RET(PullRecvStatus(iterLink));
            if ((gCqeCounterPerEvent[HCCL_EVENT_RECV_COMPLETION_MSG] > 0)) {
                return HCCL_SUCCESS;
            }
        }
        lockRecvComp.unlock();
    }
    return HCCL_SUCCESS;
}

void TransportHeterogEventRoce::EschedAckCallback(u32 devId, u32 eventId)
{
    if (!gNeedRepoEvent) {
        HCCL_DEBUG("TransportHeterogEventRoce no need submit event.");
        return;
    }

    HCCL_DEBUG("EventCallback start. devId:%u, eventId:%u.", devId, eventId);
    HcclUs startut = TIME_NOW();

    if (UpdateStatus(eventId) != HCCL_SUCCESS) {
        HCCL_ERROR("poll all cqes failed. event id:%u", eventId);
        return;
    }
    if (gCqeCounterPerEvent[eventId] != 0) {
        hrtHalSubmitEvent(devId, eventId);
    }

    HcclUs endut = TIME_NOW();
    HCCL_INFO(
        "EschedAckCallback cost time: %lld us, event id: %u, devId:%u, compCount:%d.", DURATION_US(endut - startut),
        eventId, devId, gCqeCounterPerEvent[eventId].load());
    return;
}

HcclResult TransportHeterogEventRoce::CreateCqAndQp()
{
    HCCL_INFO("TransportHeterogEventRoce CreateCqAndQp. gNeedRepoEvent[%d]", gNeedRepoEvent);
    if (gNeedRepoEvent) {
        CHK_RET(CreateQpWithSharedCq(nicRdmaHandle_, selfIp_, peerIp_, -1, recvRequestEvent, tagQpInfo_));
        CHK_RET(CreateQpWithSharedCq(
            nicRdmaHandle_, selfIp_, peerIp_, recvCompletionEvent, sendCompletionEvent, dataQpInfo_));
    } else {
        tagQpAppend_ = TAG_QP_APPEND;
        dataQpAppend_ = DATA_QP_APPEND;
        CHK_RET(CreateQpWithSharedCq(
            nicRdmaHandle_, selfIp_, peerIp_, -1, -1, tagQpInfo_, tagQpAppend_, MAX_SCATTER_BUF_NUM));
        CHK_RET(CreateQpWithSharedCq(
            nicRdmaHandle_, selfIp_, peerIp_, -1, -1, dataQpInfo_, dataQpAppend_, MAX_SCATTER_BUF_NUM));
    }

    if (srqInit_) {
        std::unique_lock<std::shared_mutex> lock(gQpnMapMutex);
        gQpnToTransportMap[tagQpInfo_.qp->qp_num] = this;
        gQpnToSqMaxWrMap[dataQpInfo_.qp->qp_num] = MAX_WR_NUM;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::DestroyCqAndQp()
{
    HCCL_INFO("TransportHeterogEventRoce DestroyCqAndQp.");
    CHK_RET(DestroyQpWithSharedCq(tagQpInfo_, tagQpAppend_));
    tagQpInfo_ = QpInfo();
    CHK_RET(DestroyQpWithSharedCq(dataQpInfo_, dataQpAppend_));
    dataQpInfo_ = QpInfo();
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::InitSrqRecvWqe()
{
    CHK_RET(IssueRecvWqe(tagQpInfo_.srq, RECV_WQE_BATCH_NUM));
    g_tagRecvWqeNum = RECV_WQE_BATCH_NUM;

    CHK_RET(IssueRecvWqe(dataQpInfo_.srq, RECV_WQE_BATCH_NUM));
    g_dataRecvWqeNum = RECV_WQE_BATCH_NUM;
    HCCL_INFO("InitSrqRecvWqe success.");
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::InitTagRecvWqe()
{
    if (!srqInit_) {
        CHK_RET(TransportHeterogRoce::InitTagRecvWqe());
        return HCCL_SUCCESS;
    }

    CHK_RET(CheckTagRecvWqe());
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::InitDataRecvWqe()
{
    if (!srqInit_) {
        CHK_RET(TransportHeterogRoce::InitDataRecvWqe());
        return HCCL_SUCCESS;
    }
    CHK_RET(CheckDataRecvWqe());
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::SendFlowControl()
{
    if (!srqInit_) {
        CHK_RET(TransportHeterogRoce::SendFlowControl());
        return HCCL_SUCCESS;
    }

    CHK_RET(CheckDataRecvWqe());
    u32 sqMaxWrMap = gQpnToSqMaxWrMap[dataQpInfo_.qp->qp_num].load();
    if (sqMaxWrMap <= 0) {
        CHK_RET(PullSendStatus());
        HCCL_RUN_INFO("Flow control is activated, because sqMaxWrMap[%u] <= 0", sqMaxWrMap);

        return HCCL_E_AGAIN;
    }
    gQpnToSqMaxWrMap[dataQpInfo_.qp->qp_num]--;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::CheckTagRecvWqe()
{
    if (g_tagRecvWqeNum <= RECV_WQE_NUM_THRESHOLD) {
        CHK_RET(IssueRecvWqe(tagQpInfo_.srq, RECV_WQE_BATCH_SUPPLEMENT));
        g_tagRecvWqeNum += RECV_WQE_BATCH_SUPPLEMENT;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::SupplyTagRecvWqe()
{
    if (!srqInit_) {
        CHK_RET(TransportHeterogRoce::SupplyTagRecvWqe());
        return HCCL_SUCCESS;
    }

    g_tagRecvWqeNum--;
    CHK_RET(CheckTagRecvWqe());

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::CheckDataRecvWqe()
{
    if (g_dataRecvWqeNum <= RECV_WQE_NUM_THRESHOLD) {
        CHK_RET(IssueRecvWqe(dataQpInfo_.srq, RECV_WQE_BATCH_SUPPLEMENT));
        g_dataRecvWqeNum += RECV_WQE_BATCH_SUPPLEMENT;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::SupplyDataRecvWqe()
{
    if (!srqInit_) {
        CHK_RET(TransportHeterogRoce::SupplyDataRecvWqe());
        return HCCL_SUCCESS;
    }

    g_dataRecvWqeNum--;
    CHK_RET(CheckDataRecvWqe());

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::IssueRecvWqe(struct ibv_srq* srq, u32 num)
{
    list<void*> blockList(num, nullptr);
    CHK_RET(AllocMemBlocks(blockList));

    auto iter = blockList.begin();
    struct ibv_recv_wr* nextRqWr = nullptr;
    struct ibv_recv_wr rqWr[num];
    struct ibv_sge sgeList[num];
    for (int i = num - 1; i >= 0; i--) {
        CHK_PTR_NULL(*iter);
        u64 wrId = 0;
        CHK_RET(GenerateRecvWrId(*iter, wrId));

        rqWr[i].wr_id = wrId;
        rqWr[i].next = nextRqWr;
        rqWr[i].sg_list = &sgeList[i];
        rqWr[i].num_sge = 1;
        sgeList[i].addr = reinterpret_cast<uint64_t>(*iter);
        sgeList[i].length = MEM_BLOCK_SIZE;
        sgeList[i].lkey = blockMemLkey_;

        nextRqWr = &rqWr[i];
        iter++;
    }

    struct ibv_recv_wr* badRqWr = nullptr;
    CHK_RET(hrtIbvPostSrqRecv(srq, &rqWr[0], &badRqWr));
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::ParseTagSrqes(const struct ibv_wc* wc, int num)
{
    for (int i = 0; i < num; i++) {
        HCCL_INFO(
            "rq cqe info: wrId[%llu] status[%u] opcode[%u] qpn[%u].", wc[i].wr_id, wc[i].status, wc[i].opcode,
            wc[i].qp_num);
        CHK_PRT_RET(wc[i].status != 0, HCCL_ERROR("rdma send failed, cqe status[%u].", wc[i].status), HCCL_E_INTERNAL);
        RecvWrInfo* info = reinterpret_cast<RecvWrInfo*>(wc[i].wr_id);
        CHK_PTR_NULL(info);

        CHK_RET(SupplyTagRecvWqe());

        HcclEnvelope* envelope = reinterpret_cast<HcclEnvelope*>(info->buf);
        CHK_PTR_NULL(envelope);

        HCCL_INFO(
            "recv request: tag:%d srcRank:%u dstRank:%u status:%u msn:0x%016llx count:%d.", envelope->epParam.src.tag,
            envelope->epParam.src.rank, envelope->epParam.dst.rank, wc[i].status, envelope->msn,
            envelope->transData.count);

        HcclEnvelopeSummary envelopSummary(*envelope, wc[i].status);
        {
            std::shared_lock<std::shared_mutex> lock(gQpnMapMutex);
            if (gQpnToTransportMap.count(wc[i].qp_num) != 0) {
                gQpnToTransportMap[wc[i].qp_num]->SaveEnvelope(envelopSummary);
            } else {
                HCCL_ERROR(
                    "The transport does not exist, wrId[%llu] status[%u] opcode[%u] qpn[%u]", wc[i].wr_id, wc[i].status,
                    wc[i].opcode, wc[i].qp_num);
                return HCCL_E_PTR;
            }
        }

        CHK_RET(FreeMemBlock(info->buf));
        CHK_RET(FreeRecvWrId(wc[i].wr_id));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogEventRoce::ParseDataSrqes(const struct ibv_wc* wc, int num)
{
    for (int i = 0; i < num; i++) {
        HCCL_INFO("rq cqe info: wrId[%llu] status[%u] opcode[%u].", wc[i].wr_id, wc[i].status, wc[i].opcode);
        CHK_PRT_RET(
            wc[i].status != 0, HCCL_ERROR("rdma poll data rq failed, cqe status[%u].", wc[i].status), HCCL_E_INTERNAL);
        RecvWrInfo* info = reinterpret_cast<RecvWrInfo*>(wc[i].wr_id);
        CHK_PTR_NULL(info);
        HcclRequestInfo* wrPtr = reinterpret_cast<HcclRequestInfo*>(*reinterpret_cast<u64*>(info->buf));
        CHK_PTR_NULL(wrPtr);
        CHK_RET(SupplyDataRecvWqe());
        wrPtr->transportRequest.status = wc[i].status;
        {
            std::shared_lock<std::shared_mutex> lock(gQpnMapMutex);
            if (gQpnToSqMaxWrMap.count(wc[i].qp_num) != 0) {
                gQpnToSqMaxWrMap[wc[i].qp_num]++;
            } else {
                HCCL_ERROR(
                    "The qpn does not exist, wrId[%llu] status[%u] opcode[%u] qpn[%u]", wc[i].wr_id, wc[i].status,
                    wc[i].opcode, wc[i].qp_num);
                return HCCL_E_PTR;
            }
        }

        CHK_RET(DeregMr(
            reinterpret_cast<void*>(wrPtr->transportRequest.transData.srcBuf),
            static_cast<u64>(
                wrPtr->transportRequest.transData.count * SIZE_TABLE[wrPtr->transportRequest.transData.dataType])));
        CHK_RET(FreeMemBlock(info->buf));
        CHK_RET(FreeRecvWrId(wc[i].wr_id));
        HCCL_INFO(
            "send completion: tag:%d peerRank:%u status:%d msn:0x%016llx request:%p.",
            wrPtr->transportRequest.epParam.src.tag, wrPtr->transportRequest.epParam.src.rank,
            wrPtr->transportRequest.status, wrPtr->transportRequest.msn, wrPtr);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
