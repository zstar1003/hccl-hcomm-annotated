/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef P2P_MEM_TRANSPORT_LITE_H
#define P2P_MEM_TRANSPORT_LITE_H

#include <vector>
#include <memory>
#include <unordered_map>
#include "base_transport_lite_impl.h"
#include "notify_lite.h"
#include "task_param.h"
#include "rmt_rma_buf_slice_lite.h"
#include "rma_conn_lite.h"
#include "kernel_param_lite.h"

namespace Hccl {

class P2PTransportLiteImpl : public BaseTransportLiteImpl {
public:
    explicit P2PTransportLiteImpl(
        std::vector<char>& uniqueId,
        std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback);

    P2PTransportLiteImpl(std::vector<char>& uniqueId);
    void Init(std::vector<char>& uniqueId);

    ~P2PTransportLiteImpl() override;

    std::string Describe() const override;

    Buffer GetRmtBuffer(u32 index) override;

    HcclResult BuildLocRmaBufferLite(const uintptr_t addr, const size_t size, RmaBufferLite& rmaBufferLite) override;

    void Post(u32 index, const StreamLite& stream) override;

    void Wait(u32 index, const StreamLite& stream) override;

    void WaitWithTimeout(u32 index, const StreamLite& stream, u32 timeout) override;

    void Read(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream) override;

    void ReadReduce(
        const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream) override;

    void BatchTransfer(
        const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt,
        const std::vector<TransferOp>& transferOp, const StreamLite& stream) override;

private:
    u32 notifyNum{0};
    u32 bufferNum{0};

    struct RmtP2PNotifyLite {
        u64 addr;
        u64 size;
        u32 id;
        std::string Describe() const
        {
            return StringFormat("RmtP2PNotifyLite[addr=0x%llx, size=0x%llx, id=%u]", addr, size, id);
        }
    };

    struct P2PBufLite {
        u64 addr;
        u64 size;
        std::string Describe() const { return StringFormat("P2PBufLite[addr=0x%llx, size=0x%llx]", addr, size); }
    };

    std::vector<RmtP2PNotifyLite> rmtNotifyVec;
    std::vector<P2PBufLite> rmtBufferVec;
    std::vector<P2PBufLite> locBufferVec;

    std::vector<std::unique_ptr<NotifyLite>> locNotifyVec;

    std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback_{nullptr};

    void ParseLocNotifyVec(std::vector<char>& data);

    void ParseRmtNotifyVec(std::vector<char>& data, std::vector<RmtP2PNotifyLite>& vec) const;

    void ParseRmtBufferVec(std::vector<char>& data, std::vector<P2PBufLite>& vec) const;

    void BuildNotifyRecordTask(const StreamLite& stream, u64 rmtNotifyAddr);

    void BuildNotifyWaitTask(const StreamLite& stream, u32 notifyId);

    void BuildP2PRead(const StreamLite& stream, const RmaBufferLite& loc, const Buffer& rmt);

    void
    BuildP2PReadReduce(const StreamLite& stream, const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn);
};

} // namespace Hccl
#endif
