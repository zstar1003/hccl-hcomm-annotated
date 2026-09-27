/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_DEVICE_ROCE_MEM_H
#define TRANSPORT_DEVICE_ROCE_MEM_H

#include "transport_mem.h"
#include <atomic>
#include <chrono>

namespace hccl {
class TransportDeviceRoceMem : public TransportMem {
public:
    TransportDeviceRoceMem(
        const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx, const HcclDispatcher& dispatcher,
        AttrInfo& attrInfo, bool aicpuUnfoldMode, const HcclQpInfoV2& qpInfo);
    ~TransportDeviceRoceMem() override;

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
    enum class RdmaOp { OP_WRITE = 0, OP_READ = 4 };

    template <typename T>
    inline T CeilDiv(T left, T right)
    {
        if (right == 0) {
            return left;
        }
        return (left + right - 1) / right;
    }

    HcclResult BatchOp(
        Stream& stream, const std::vector<MemDetails>& localMems, const std::vector<MemDetails>& remoteMems,
        bool isRead, bool fence);
    HcclResult FillMemDetails(
        std::vector<MemDetails>& localMemList, std::vector<MemDetails>& remoteMemList, MemDetails& localMem,
        MemDetails& remoteMem);
    HcclResult DoorBellSend(Stream& stream, u64 dbInfo, u32 wrDataLen, bool fence);
    HcclResult BatchPostSend(
        Stream& stream, u64& dbInfo, std::vector<MemDetails>& localMemList, std::vector<MemDetails>& remoteMemList,
        bool isRead, bool fence, u32& wqeCount, u64& wrDataLen);
    HcclResult PostSend(
        Stream& stream, u64& dbInfo, MemDetails* localMems, MemDetails* remoteMems, u32 memNum, bool isRead, bool fence,
        u32& wqeCount, u64& wrDataLen);
    HcclResult
    RdmaPostSend(u64& dbInfo, MemDetails* localMems, MemDetails* remoteMems, u32 memNum, RdmaOp opCode, bool fence);

    static std::atomic<u64> wrIdOffset_;

    const u64 MAX_RDMA_WQE_SIZE = 2ULL * 1024 * 1024 * 1024; // RDMA最大WQE限制, 2G限制是RDMA导致
    const u32 SEND_WR_LEN = 64;
    const std::chrono::microseconds timeout_;
    HcclQpInfoV2 qpInfo_{};
};
} // namespace hccl
#endif
