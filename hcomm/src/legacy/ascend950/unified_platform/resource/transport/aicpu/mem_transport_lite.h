/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEM_TRANSPORT_LITE_H
#define MEM_TRANSPORT_LITE_H
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include "task_param.h"
#include "base_transport_lite_impl.h"
#include "stream_lite.h"
#include "rma_buffer_lite.h"
#include "buffer.h"
#include "kernel_param_lite.h"

namespace Hccl {

class MemTransportLite {
public:
    explicit MemTransportLite(
        std::vector<char>& uniqueId,
        std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback);

    std::string Describe() const;

    using TransferOp = struct TransferOp;

    Buffer GetRmtBuffer(u32 index) { return impl->GetRmtBuffer(index); }

    void Post(u32 index, const StreamLite& stream) { impl->Post(index, stream); }

    void Wait(u32 index, const StreamLite& stream) { impl->Wait(index, stream); }

    void Read(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream) { impl->Read(loc, rmt, stream); }

    void Write(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream) { impl->Write(loc, rmt, stream); }

    void ReadReduce(const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream)
    {
        impl->ReadReduce(loc, rmt, reduceIn, stream);
    }

    void WriteReduce(const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream)
    {
        impl->WriteReduce(loc, rmt, reduceIn, stream);
    }

    void WriteWithNotify(
        const RmaBufferLite& loc, const Buffer& rmt, const WithNotifyIn& withNotify, const StreamLite& stream)
    {
        impl->WriteWithNotify(loc, rmt, withNotify, stream);
    }

    void WriteReduceWithNotify(
        const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const WithNotifyIn& withNotify,
        const StreamLite& stream)
    {
        impl->WriteReduceWithNotify(loc, rmt, reduceIn, withNotify, stream);
    }

    void BatchOneSidedRead(
        const std::vector<RmaBufSliceLite>& loc, const std::vector<RmtRmaBufSliceLite>& rmt, const StreamLite& stream)
    {
        impl->BatchOneSidedRead(loc, rmt, stream);
    }

    void BatchOneSidedWrite(
        const std::vector<RmaBufSliceLite>& loc, const std::vector<RmtRmaBufSliceLite>& rmt, const StreamLite& stream)
    {
        impl->BatchOneSidedWrite(loc, rmt, stream);
    }

    void BatchTransfer(
        const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt,
        const std::vector<BaseTransportLiteImpl::TransferOp>& transferOp, const StreamLite& stream)
    {
        impl->BatchTransfer(loc, rmt, transferOp, stream);
    }

private:
    TransportType type;

    std::unique_ptr<BaseTransportLiteImpl> impl;
};

} // namespace Hccl
#endif
