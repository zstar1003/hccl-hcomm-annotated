/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_ROCE_MEM_H
#define TRANSPORT_ROCE_MEM_H

#include <array>

#include "transport_mem.h"
#include "private_types.h"
#include "local_ipc_notify.h"
#include "rma_buffer_mgr.h"
#include "local_rdma_rma_buffer.h"
#include "remote_rdma_rma_buffer.h"

namespace hccl {
constexpr u32 REMOTE_RDMA_SIGNAL_SIZE = 1;
constexpr u64 MAX_RDMA_WQE_SIZE = 2ULL * 1024 * 1024 * 1024; // RDMA最大WQE限制, 2G限制是RDMA导致
constexpr u64 NORMAL_PAGE_SIZE = 4 * 1024;                   // 普通内存大小，减少内存占用

class TransportRoceMem : public TransportMem {
public:
    using RemoteRdmaRmaBufferMgr = hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, std::shared_ptr<RemoteRdmaRmaBuffer>>;

    TransportRoceMem(
        const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx, const HcclDispatcher& dispatcher,
        AttrInfo& attrInfo, bool aicpuUnfoldMode = false);
    ~TransportRoceMem() override;
    HcclResult
    ExchangeMemDesc(const RmaMemDescs& localMemDescs, RmaMemDescs& remoteMemDescs, u32& actualNumOfRemote) override;
    HcclResult EnableMemAccess(const RmaMemDesc& remoteMemDesc, RmaMem& remoteMem) override;
    HcclResult DisableMemAccess(const RmaMemDesc& remoteMemDesc) override;
    HcclResult SetSocket(const std::shared_ptr<HcclSocket>& socket) override;
    HcclResult Connect(s32 timeoutSec) override;
    HcclResult Write(const HcclBuf& remoteMem, const HcclBuf& localMem, const rtStream_t& stream) override;
    HcclResult Read(const HcclBuf& localMem, const HcclBuf& remoteMem, const rtStream_t& stream) override;
    HcclResult Write(const RmaOpMem& remoteMem, const RmaOpMem& localMem, const rtStream_t& stream) override;
    HcclResult Read(const RmaOpMem& localMem, const RmaOpMem& remoteMem, const rtStream_t& stream) override;
    HcclResult AddOpFence(const rtStream_t& stream) override;

    HcclResult
    GetTransInfo(HcclQpInfoV2& qpInfo, u32* lkey, u32* rkey, HcclBuf* localMem, HcclBuf* remoteMem, u32 num) override;
    HcclResult WaitOpFence(const rtStream_t& stream) override;

    HcclResult BatchWrite(
        const std::vector<MemDetails>& remoteMems, const std::vector<MemDetails>& localMems, Stream& stream) override;
    HcclResult BatchRead(
        const std::vector<MemDetails>& localMems, const std::vector<MemDetails>& remoteMems, Stream& stream) override;
    HcclResult AddOpFence(const MemDetails& localFenceMem, const MemDetails& remoteFenceMem, Stream& stream) override;

private:
    enum class RdmaOp { OP_WRITE = 0, OP_SEND = 2, OP_READ = 4 };

    enum class SupportStatus { INIT = 0, SUPPORT = 1, NOT_SUPPORT = 2 };
    HcclResult CreateCqAndQp();
    HcclResult WaitQPLinkComplete(s32 timeoutSec);
    HcclResult DoorBellSend(const s32 qpMode, WrInfo& sendWrInfo, const SendWrRsp& opRsp, rtStream_t stream);
    HcclResult DestroyCqAndQp();
    HcclResult CreatSignalMesg();
    HcclResult GetNotifySize();
    HcclResult RecoverNotifyMsg(MemMsg* remoteRdmaSignal, u64 signalNum);
    HcclResult ConnectImpl(s32 timeoutSec);
    HcclResult GetQpStatus();
    HcclResult QpConnect(s32 timeoutSec);
    HcclResult
    CreateRdmaSignal(std::shared_ptr<LocalIpcNotify>& localNotify, MemMsg& rdmaSignalInfo, MemType notifyType);
    HcclResult GetRdmaHandle();
    HcclResult ExchangeNotifyValueBuffer(s32 timeoutSec);
    HcclResult RdmaDbSend(u32 dbindex, u64 dbinfo, const struct SendWr& sendWr, rtStream_t stream);
    HcclResult CreateNotifyValueBuffer();
    HcclResult TransportRdmaWithType(
        const RmaBufferSlice& localRmaBufferSlice, const RmaBufferSlice& remoteRmaBufferSlice, const rtStream_t& stream,
        const RdmaOp& rdmaOp);
    HcclResult TransportIpc(
        const RmaBufferSlice& dstRmaBufferSlice, const RmaBufferSlice& srcRmaBufferSlice, const rtStream_t& stream);
    HcclResult CheckRaSendNormalWrlistSupport();

    HcclResult FillRmaBufferSlice(
        const HcclBuf& localMem, const HcclBuf& remoteMem, RmaBufferSlice& localRmaBufferSlice,
        RmaBufferSlice& remoteRmaBufferSlice);

    HcclResult FillRmaBufferSlice(
        const RmaOpMem& localMem, const RmaOpMem& remoteMem, RmaBufferSlice& localRmaBufferSlice,
        RmaBufferSlice& remoteRmaBufferSlice);

    HcclResult GetQpInfo(HcclQpInfoV2& qpInfo);
    HcclResult GetMemInfo(u32& lkey, u32& rkey, HcclBuf& localMem, HcclBuf& remoteMem);
    HcclResult GetOpFence(u32& lkey, u32& rkey, HcclBuf& localMem, HcclBuf& remoteMem);
    HcclResult CheckRdmaVal(void);

    RemoteRdmaRmaBufferMgr remoteRdmaRmaBufferMgr_{};

    std::array<MemMsg, static_cast<u32>(MemType::MEM_TYPE_RESERVED)> notifyMemMsg_;
    QpInfo dataQpInfo_{};
    s32 access_{RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_WRITE | RA_ACCESS_REMOTE_READ};
    void* nicRdmaHandle_{nullptr};
    struct AiQpInfo aiQpInfo_ {}; // struct ibv_qp
    std::shared_ptr<LocalIpcNotify> remoteIsendDoneSignal_{nullptr};
    MemMsg rdmaSignal_[REMOTE_RDMA_SIGNAL_SIZE];
    MrHandle rdmaSignalMrHandle_{nullptr};
    MrHandle notifyValueMemMrHandle_{nullptr};
    u32 notifySize_{0};
    s32 deviceLogicId_{-1};
    s32 devicePhyId_{-1};
    s32 index_{-1};
    static std::atomic<uint64_t> sendWrHandle;
    std::chrono::seconds maxTimeOut_{0};
    const u64 notifyValueSize_{NORMAL_PAGE_SIZE};
    DeviceMem notifyMem_{};
    SupportStatus isSupportRaSendNormalWrlist_ = SupportStatus::INIT;
    u32 trafficClass_;
    u32 serviceLevel_;
};
} // namespace hccl
#endif
