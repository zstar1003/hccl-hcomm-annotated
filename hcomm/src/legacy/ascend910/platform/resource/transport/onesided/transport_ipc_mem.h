/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_IPC_MEM_H
#define TRANSPORT_IPC_MEM_H

#include "transport_mem.h"
#include "rma_buffer_mgr.h"
#include "local_ipc_rma_buffer.h"
#include "remote_ipc_rma_buffer.h"

namespace hccl {
class TransportIpcMem : public TransportMem {
public:
    using RemoteIpcRmaBufferMgr = hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, std::shared_ptr<RemoteIpcRmaBuffer>>;

    struct ProcessInfo {
        u32 pid;
        u32 sdid;     // super device id
        u32 serverId; // server id
    };

    TransportIpcMem(
        const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx, const HcclDispatcher& dispatcher,
        AttrInfo& attrInfo, bool aicpuUnfoldMode = false);
    ~TransportIpcMem() override;
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
    HcclResult TransportIpc(
        const RmaBufferSlice& dstRmaBufferSlice, const RmaBufferSlice& srcRmaBufferSlice, const rtStream_t& stream);

    HcclResult FillRmaBufferSlice(
        const HcclBuf& localMem, const HcclBuf& remoteMem, RmaBufferSlice& localRmaBufferSlice,
        RmaBufferSlice& remoteRmaBufferSlice);

    HcclResult FillRmaBufferSlice(
        const RmaOpMem& localMem, const RmaOpMem& remoteMem, RmaBufferSlice& localRmaBufferSlice,
        RmaBufferSlice& remoteRmaBufferSlice);

    HcclResult GetMemInfo(u32& lkey, u32& rkey, HcclBuf& localMem, HcclBuf& remoteMem);
    RemoteIpcRmaBufferMgr remoteIpcRmaBufferMgr_{};
    u32 sdid_;
    u32 serverId_;
};
} // namespace hccl
#endif
