/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_MEM_TRANSPORT_LITE_H
#define BASE_MEM_TRANSPORT_LITE_H

#include <memory>
#include <vector>
#include <unordered_map>
#include "stream_lite.h"
#include "buffer.h"
#include "internal_exception.h"
#include "rma_buffer_lite.h"
#include "mem_transport_common.h"
#include "rmt_rma_buf_slice_lite.h"
#include "task_param.h"
namespace Hccl {

inline HcclReduceOp ConvertReduceOpToHcclReduceOp(ReduceOp reduceOp)
{
    static std::map<ReduceOp, HcclReduceOp> reduceTypeMap
        = {{ReduceOp::SUM, HcclReduceOp::HCCL_REDUCE_SUM},
           {ReduceOp::PROD, HcclReduceOp::HCCL_REDUCE_PROD},
           {ReduceOp::MAX, HcclReduceOp::HCCL_REDUCE_MAX},
           {ReduceOp::MIN, HcclReduceOp::HCCL_REDUCE_MIN}};
    if (UNLIKELY(reduceTypeMap.find(reduceOp) == reduceTypeMap.end())) {
        THROW<InternalException>(StringFormat("reduceOp[%u] is invalid", reduceOp));
    }
    return reduceTypeMap[reduceOp];
}

MAKE_ENUM(
    TransferType, WRITE, WRITE_REDUCE, WRITE_WITH_NOTIFY, WRITE_REDUCE_WITH_NOTIFY, READ, READ_REDUCE, NOTIFY_RECORD)

class BaseTransportLiteImpl {
public:
    BaseTransportLiteImpl() = default;

    virtual ~BaseTransportLiteImpl() = default;

    virtual std::string Describe() const { return "BaseTransportLiteImpl"; }

    struct TransferOp {
        TransferType transType;
        ReduceIn reduceIn;
    };

    virtual Buffer GetRmtBuffer(u32 index)
    {
        (void)index;
        return Buffer(0, 0);
    }

    virtual HcclResult BuildLocRmaBufferLite(const uintptr_t addr, const size_t size, RmaBufferLite& rmaBufferLite)
    {
        (void)addr;
        (void)size;
        rmaBufferLite = RmaBufferLite(0, 0, 0, 0);
        return HCCL_SUCCESS;
    }

    virtual void Post(u32 index, const StreamLite& stream)
    {
        (void)index;
        (void)stream;
    }

    virtual void Wait(u32 index, const StreamLite& stream)
    {
        (void)index;
        (void)stream;
    }

    virtual void WaitWithTimeout(u32 index, const StreamLite& stream, u32 timeout)
    {
        (void)index;
        (void)stream;
        (void)timeout;
    }

    virtual void Read(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream)
    {
        (void)loc;
        (void)rmt;
        (void)stream;
    }

    virtual void Write(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream)
    {
        (void)loc;
        (void)rmt;
        (void)stream;
    }

    virtual void
    ReadReduce(const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream)
    {
        (void)loc;
        (void)rmt;
        (void)reduceIn;
        (void)stream;
    }

    virtual void
    WriteReduce(const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream)
    {
        (void)loc;
        (void)rmt;
        (void)reduceIn;
        (void)stream;
    }

    virtual void WriteWithNotify(
        const RmaBufferLite& loc, const Buffer& rmt, const WithNotifyIn& withNotify, const StreamLite& stream)
    {
        (void)loc;
        (void)rmt;
        (void)withNotify;
        (void)stream;
    }

    virtual void WriteReduceWithNotify(
        const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const WithNotifyIn& withNotify,
        const StreamLite& stream)
    {
        (void)loc;
        (void)rmt;
        (void)reduceIn;
        (void)withNotify;
        (void)stream;
    }

    virtual void BatchOneSidedWrite(
        const std::vector<RmaBufSliceLite>& loc, const std::vector<RmtRmaBufSliceLite>& rmt, const StreamLite& stream)
    {
        (void)loc;
        (void)rmt;
        (void)stream;
    }

    virtual void BatchOneSidedRead(
        const std::vector<RmaBufSliceLite>& loc, const std::vector<RmtRmaBufSliceLite>& rmt, const StreamLite& stream)
    {
        (void)loc;
        (void)rmt;
        (void)stream;
    }

    virtual void BatchTransfer(
        const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt,
        const std::vector<TransferOp>& transferOp, const StreamLite& stream)
    {
        (void)loc;
        (void)rmt;
        (void)transferOp;
        (void)stream;
    }

    virtual void Drain(const StreamLite& stream) { (void)stream; }

    virtual HcclResult Fence() { return HCCL_SUCCESS; }

protected:
private:
};

} // namespace Hccl
#endif
