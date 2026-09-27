/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_HETEROG_EVENT_ROCE_PUB_H
#define TRANSPORT_HETEROG_EVENT_ROCE_PUB_H
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <vector>
#include <queue>
#include <stack>
#include <vector>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "dispatcher_pub.h"
#include "heterog_mem_blocks_manager_pub.h"
#include "adapter_verbs.h"
#include "adapter_hal.h"
#include "transport_heterog_roce_pub.h"

namespace hccl {
class TransportHeterogEventRoce : public TransportHeterogRoce {
public:
    explicit TransportHeterogEventRoce(
        const std::string& transTag, HcclIpAddress& selfIp, HcclIpAddress& peerIp, u32 peerPort, u32 selfPort,
        const TransportResourceInfo& transportResourceInfo);
    explicit TransportHeterogEventRoce(const TransportResourceInfo& transportResourceInfo);
    ~TransportHeterogEventRoce() override;
    HcclResult Init() override;
    HcclResult Deinit() override;
    HcclResult
    Isend(const TransData& sendData, const TransportEndPointParam& epParam, HcclRequestInfo*& request) override;
    HcclResult
    Improbe(const TransportEndPointParam& epParam, s32& matched, HcclMessageInfo*& msg, HcclStatus& status) override;
    HcclResult Imrecv(const TransData& recvData, HcclMessageInfo& msg, HcclRequestInfo*& request) override;
    HcclResult Test(HcclRequestInfo& request, s32& flag, HcclStatus& compState) override;
    HcclResult InitSrqRecvWqe();

protected:
    HcclResult PullRecvRequestStatus(bool allowNotify = false) override;
    HcclResult PullSendStatus(bool allowNotify = false) override;
    HcclResult PullRecvStatus(bool allowNotify = false) override;
    HcclResult ParseErrorTagSqe(const struct ibv_wc* wc, int index) override;
    HcclResult ParseTagRqes(const struct ibv_wc* wc, int num) override;
    HcclResult ParseDataRqes(const struct ibv_wc* wc, int num) override;
    HcclResult ParseDataSqes(const struct ibv_wc* wc, int num) override;
    HcclResult QueryRequestStatus(HcclRequestInfo& request, s32& flag, HcclStatus& compState) override;
    HcclResult CreateCqAndQp() override;
    HcclResult DestroyCqAndQp() override;
    HcclResult SendFlowControl() override;
    HcclResult InitTagRecvWqe() override;
    HcclResult InitDataRecvWqe() override;
    HcclResult SupplyTagRecvWqe() override;
    HcclResult SupplyDataRecvWqe() override;

private:
    static HcclResult PullRecvRequestStatus(void* transportHandle);
    static HcclResult PullSendStatus(void* transportHandle);
    static HcclResult PullRecvStatus(void* transportHandle);
    static HcclResult UpdateStatus(u32 eventId);
    static void EschedAckCallback(u32 devId, u32 eventId);
    static void EschedAckCallbackRecvRequest(unsigned int devId, unsigned int subeventId, u8* msg, unsigned int msgLen);
    static void
    EschedAckCallbackRecvCompletion(unsigned int devId, unsigned int subeventId, u8* msg, unsigned int msgLen);
    static void
    EschedAckCallbackSendCompletion(unsigned int devId, unsigned int subeventId, u8* msg, unsigned int msgLen);

    static HcclReceivedEnvelope gReceivedEnvelopes;
    static std::mutex gReceivedEnvelopesMutex;
    static std::vector<std::atomic<int>> gCqeCounterPerEvent;
    static std::vector<std::vector<void*>> gAllLinkVec;
    static std::mutex gAllLinkVecSendCompMutex;
    static std::mutex gAllLinkVecRecvReqMutex;
    static std::mutex gAllLinkVecRecvCompMutex;

    static std::mutex gPollTagRqLock;
    static std::mutex gPollDataRqLock;
    static std::mutex gPollDataSqLock;
    static std::mutex gEschedAckMutex;

    static u32 gEschedAckRef;
    static u32 gAllLinkInitCount;

    s32 tagQpAppend_{0};
    s32 dataQpAppend_{0};

private:
    HcclResult RegisterEschedAckCallback();
    HcclResult DeregisterEschedAckCallback();
    HcclResult InitAllLinkVec();
    HcclResult DeinitAllLinkVec();
    HcclResult EraseTransportFromAllLinkVec(void* transportPtr);

    static u32 recvRequestEvent;
    static u32 sendCompletionEvent;
    static u32 recvCompletionEvent;

    HcclResult IssueRecvWqe(struct ibv_srq* srq, u32 num);
    HcclResult ParseTagSrqes(const struct ibv_wc* wc, int num);
    HcclResult ParseDataSrqes(const struct ibv_wc* wc, int num);
    HcclResult CheckTagRecvWqe();
    HcclResult CheckDataRecvWqe();
    static std::shared_mutex gQpnMapMutex;
    static std::map<u32, TransportHeterogEventRoce*> gQpnToTransportMap;
    static std::map<u32, std::atomic<u32>> gQpnToSqMaxWrMap;
    static bool gNeedRepoEvent;
    bool srqInit_{};
    bool isDeinited_{false};
};
} // namespace hccl
#endif
