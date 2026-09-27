/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_HETEROG_ROCE_PUB_H
#define TRANSPORT_HETEROG_ROCE_PUB_H
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
#include "adapter_verbs.h"
#include "adapter_hal.h"
#include "mr_manager.h"
#include "transport_heterog_pub.h"
#include "transport_ibverbs_pub.h"
#include "es_private.h"

namespace hccl {
using EventSendFlag = struct EventSendFlagDef {
    std::atomic_flag flag;
    std::atomic<int> counter;
    EventSendFlagDef() : flag(ATOMIC_FLAG_INIT), counter(0) {}
};

constexpr u32 HCCL_POLL_CQ_ONETIME = 1;
constexpr u32 REMOTE_RDMA_SIGNAL_SIZE = 2;

class TransportHeterogRoce : public TransportHeterog {
public:
    explicit TransportHeterogRoce(
        const std::string& transTag, HcclIpAddress& selfIp, HcclIpAddress& peerIp, u32 peerPort, u32 selfPort,
        const TransportResourceInfo& transportResourceInfo);
    explicit TransportHeterogRoce(const TransportResourceInfo& transportResourceInfo);
    ~TransportHeterogRoce() override;
    HcclResult Init() override;
    HcclResult Deinit() override;
    HcclResult
    Isend(const TransData& sendData, const TransportEndPointParam& epParam, HcclRequestInfo*& request) override;
    HcclResult Send(const TransData& sendData, const TransportEndPointParam& epParam) override;
    HcclResult
    Improbe(const TransportEndPointParam& epParam, s32& matched, HcclMessageInfo*& msg, HcclStatus& status) override;
    HcclResult Imrecv(
        const TransData& recvData, HcclMessageInfo& msg, HcclRequestInfo*& request, bool flag,
        bool needRecordFlag) override;
    HcclResult Imrecv(const TransData& recvData, HcclMessageInfo& msg, HcclRequestInfo*& request) override;
    HcclResult Iwrite(const TransData& sendData, const HcclEnvelope& envelope, HcclRequestInfo*& request) override;
    HcclResult Test(HcclRequestInfo& request, s32& flag, HcclStatus& compState) override;
    HcclResult SendEnvelope(HcclEnvelope& envelopInfo, void* stream = nullptr);
    HcclResult PollCq(QpInfo& qpInfo, bool isSend, s32& num, struct ibv_wc* wc);
    HcclResult CreateNotifyValueBuffer();
    HcclResult GetNotifySize();
    HcclResult CreatSignalMesg();
    HcclResult ExchangeSignalMesg();
    HcclResult DoorBellSend(const s32 qpMode, const SendWrRsp& opRsp, void* stream = nullptr);
    HcclResult DeleteNotifyValueBuffer();
    HcclResult RecordNotify(Stream& stream, RdmaNotifyOp type, u64 wrId);

    HcclResult RecordNotifyWithReq(Stream& stream, RdmaNotifyOp type, HcclRequestInfo*& request);

    HcclResult RecoverNotifyMsg(HcclRdmaSignalInfo* remoteRdmaSignal, u64 signalNum);
    HcclResult GetRemoteIsendDoneSignal(std::shared_ptr<LocalIpcNotify>& signal) override;
    HcclResult GetRemoteImrecvDoneSignal(std::shared_ptr<LocalIpcNotify>& signal) override;
    HcclResult Improbe(
        const TransportEndPointParam& epParam, s32& matched, HcclMessageInfo*& msg, HcclStatus& status,
        bool& flag) override;
    HcclResult PsRdmaDbSend(uint32_t dbindex, uint64_t dbinfo, rtStream_t stream);
    HcclResult CreateHostMemForNotify(DeviceMem& devMem, u64 size, u32 value, bool needMap);
    HcclResult CreateDevMemForNotify(DeviceMem& devMem, u64 size, u32 value);

    HcclResult RegMr(void* mem, u64 size, u32& lkey, bool isTagQpHandle = true);
    HcclResult DeregMr(void* mem, u64 size, bool isTagQpHandle = true);
    void GetLinkTag(std::string& tag) override;

    constexpr static u32 RECV_WQE_BATCH_NUM = MEM_BLOCK_RECV_WQE_BATCH_NUM;
    DeviceMem notifyMem_;
    std::vector<s8*> hostMemPtr_;
    std::array<MemMsg, static_cast<u32>(MemType::MEM_TYPE_RESERVED)> notifyMemMsg_;
    QpInfo tagQpInfo_;
    QpInfo dataQpInfo_;

protected:
    virtual HcclResult CreateCqAndQp();
    virtual HcclResult DestroyCqAndQp();
    virtual HcclResult PullRecvRequestStatus(bool allowNotify = false);
    virtual HcclResult PullSendStatus(bool allowNotify = false);
    virtual HcclResult PullRecvStatus(bool allowNotify = false);
    virtual HcclResult ParseErrorTagSqe(const struct ibv_wc* wc, int index);
    virtual HcclResult ParseTagRqes(const struct ibv_wc* wc, int num);
    virtual HcclResult ParseDataRqes(const struct ibv_wc* wc, int num);
    virtual HcclResult ParseDataSqes(const struct ibv_wc* wc, int num);
    virtual HcclResult QueryRequestStatus(HcclRequestInfo& request, s32& flag, HcclStatus& compState);
    HcclResult GetSocketInfos(std::vector<std::vector<HcclSocketInfo>>& socketInfos);
    virtual HcclResult InitTagRecvWqe();
    virtual HcclResult InitDataRecvWqe();
    virtual HcclResult SupplyTagRecvWqe();
    virtual HcclResult SupplyDataRecvWqe();
    virtual HcclResult SendFlowControl();
    virtual HcclResult PreQpConnect();
    HcclResult FreeRecvWrId(u64 wrId);
    HcclResult FreeMemBlock(void* block);
    void SaveEnvelope(HcclEnvelopeSummary& envelope);
    HcclResult AllocMemBlocks(std::list<void*>& blockList);
    HcclResult GenerateRecvWrId(void* recvBuf, u64& wrId);
    bool GetSavedEnvelope(HcclEnvelopeSummary& envelope);

    void* nicRdmaHandle_;
    MrManager* mrManager_;
    MrManager* dataQpMrManager_{nullptr};
    u32 blockMemLkey_;
    u32 recvWqeBatchNum_;
    u32 recvWqeBatchThreshold_;
    u32 recvWqeBatchSupplement_;

    struct ibv_send_wr envelopeWr_; // 信封发送的wr模板
    struct ibv_send_wr dataReadWr_; // 数据读取的wr模板
    struct ibv_send_wr dataWriteWr_;
    struct ibv_send_wr notifyWriteWr_;
    struct ibv_send_wr dataAckWr_; // ACK发送的wr模板
    struct ibv_sge envelopeSge_;
    struct ibv_sge dataReadSge_;
    struct ibv_sge dataWriteSge_;
    struct ibv_sge notifyWriteSge_;
    struct ibv_sge dataReadSgeArry_[MAX_SCATTER_BUF_NUM + 1];
    struct ibv_sge dataAckSge_;

    u64 hostAddrBegin_{};
    u64 devAddrBegin_{};
    s32 deviceLogicId_;
    s32 remoteDeviceId_{0};
    s32 recvPid_{0};
    FdHandle fdHandle_{nullptr};
    u32 notifySize_{0};
    s32 access_;
    std::shared_ptr<LocalIpcNotify> remoteIsendDoneSignal_{nullptr};
    std::shared_ptr<LocalIpcNotify> remoteImrecvDoneSignal_{nullptr};
    const u64 notifyValueSize_{LARGE_PAGE_MEMORY_MIN_SIZE};
    HcclRdmaSignalInfo rdmaSignal_[REMOTE_RDMA_SIGNAL_SIZE]{};

private:
    HcclResult Wait(HcclRequestInfo& request, s32& flag);
    HcclResult PullSendOrRecvStatus(const HcclRequestInfo& request);

    HcclResult LoopStateProcess() override;
    HcclResult EnterStateProcess(ConnState nextState) override;

    HcclResult IssueRecvWqe(struct ibv_qp* qp, u32 num);
    HcclResult GetNetworkResource();
    HcclResult GetQpStatus(bool& completed);
    HcclResult QpConnect(bool& completed);
    HcclResult FlushSendQueue(bool& completed);
    HcclResult RoceConnectSocket(SocketConnectInfoT conn[], u32 num, bool& completed);
    void GetTransportResourceInfo(const TransportResourceInfo& transportResourceInfo);
    HcclResult MrManagerInit();
    HcclResult MemBlocksManagerInit();
    HcclResult MemBlocksManagerDeInit();
    HcclResult MrManagerDeInit();
    HcclResult PreHdcResource();
    HcclResult CreateRdmaSignal(
        std::shared_ptr<LocalIpcNotify>& localNotify, HcclRdmaSignalInfo& rdmaSignalInfo, MemType notifyType);
    bool IsRamdHandleLevelMr();

    std::atomic<u32> tagRecvWqeNum_;     // qp0上的recv wqe的数量,recv端消耗
    std::atomic<u32> dataRecvWqeNum_;    // qp1上的recv wqe的数量,send端消耗
    std::atomic<u32> dataRecvWqeExpNum_; // qp1上的recv wqe的预期数量

    const std::unique_ptr<HeterogMemBlocksManager>& memBlocksManager_;
    std::unique_ptr<HeterogMemBlocksManager> tagMemBlocksManager_;
    const std::unique_ptr<LocklessRingMemoryAllocate<RecvWrInfo>>& pRecvWrInfosMem_;

    std::queue<HcclEnvelope> envelopeBacklogQueue_;
    std::mutex envelopeBacklogQueueLock_;

    std::mutex envelopeQueMutex_;
    std::queue<HcclEnvelopeSummary> envelopeQue_;

    u32 memBlockNum_;
    void* deviceEvePtr_;
    u32 deviceEveLkey_;
    bool useDevMem_;
    bool isGlobalMrmanagerInit_;
    u32 hdcHostWqeBatchNum_;
    std::vector<void*> devMemPtrs_{};
    std::list<void*> wqeBlockLists_{};
    bool isRawConn_{false};
    bool isDeinited_{false};
};
} // namespace hccl
#endif
