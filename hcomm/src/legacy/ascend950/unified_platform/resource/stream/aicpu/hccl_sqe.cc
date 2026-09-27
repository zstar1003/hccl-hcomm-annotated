/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "hccl_sqe.h"
#include "exception_util.h"
#include "not_support_exception.h"

namespace Hccl {

u32 GetAddrLow(u64 addr) { return static_cast<u32>(addr & LOW32_BIT_MASK); }

u32 GetAddrHigh(u64 addr) { return static_cast<u32>((addr & HIGH32_BIT_MASK) >> UINT32_BIT_NUM); }

HcclNotifyWaitSqe::HcclNotifyWaitSqe()
{
    sqe = std::make_unique<RtStarsNotifySqe>();
    sqe->header.type = static_cast<uint8_t>(RtStarsSqeType::RT_STARS_SQE_TYPE_NOTIFY_WAIT);
    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
}

void HcclNotifyWaitSqe::Config(u16 streamId, u16 taskId, u64 notifyId)
{
    sqe->header.rtStreamId = streamId;
    sqe->camelBack = notifyId;
    sqe->header.taskId = taskId;
    HCCL_INFO("[SQE] notify wait: notifyId=%llu, streamId=%u, taskId=%u.", notifyId, streamId, taskId);
}

u64 HcclNotifyWaitSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

HcclNotifyRecordSqe::HcclNotifyRecordSqe()
{
    sqe = std::make_unique<RtStarsNotifySqe>();
    sqe->header.type = static_cast<uint8_t>(RtStarsSqeType::RT_STARS_SQE_TYPE_NOTIFY_RECORD);
    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
}
void HcclNotifyRecordSqe::Config(u16 streamId, u16 taskId, u64 notifyId)
{
    sqe->header.rtStreamId = streamId;
    sqe->camelBack = notifyId;
    sqe->header.taskId = taskId;
    HCCL_INFO("[SQE] notify record: notifyId=%llu, streamId=%u, taskId=%u.", notifyId, streamId, taskId);
}
u64 HcclNotifyRecordSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

HcclWriteValueSqe::HcclWriteValueSqe()
{
    sqe = std::make_unique<RtStarsWriteValueSqe>();
    sqe->header.type = static_cast<uint8_t>(RtStarsSqeType::RT_STARS_SQE_TYPE_WRITE_VALUE);
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->awsize = RtStarsWriteValueSizeType::RT_STARS_WRITE_VALUE_SIZE_TYPE_32BIT;
    sqe->writeValuePart0 = 1U;
    sqe->subType = RtStarsWriteValueSubType::RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_NO_PCIE;
}

void HcclWriteValueSqe::Config(u16 streamId, u16 taskId, u64 notifyWRAddr)
{
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->writeAddrLow = GetAddrLow(notifyWRAddr);
    sqe->writeAddrHigh = GetAddrHigh(notifyWRAddr) & MASK_17_BIT;
    HCCL_INFO("[SQE] write value: writePtr=0x%llx, streamId=%u, taskId=%u.", notifyWRAddr, streamId, taskId);
}

u64 HcclWriteValueSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

HcclSdmaSqe::HcclSdmaSqe()
{
    sqe = std::make_unique<RtStarsMemcpyAsyncSqe>();
    sqe->header.type = static_cast<uint8_t>(RtStarsSqeType::RT_STARS_SQE_TYPE_SDMA);
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->qos = 6; // 6 is HCCL QoS
}

void HcclSdmaSqe::Config(
    u16 streamId, u16 taskId, const u64 src, u32 length, RtDataType rtDataType, RtReduceKind rtReduceOp, const u64 dst,
    u32 partId)
{
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    const bool isReduce
        = ((rtReduceOp == RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_ADD)
           || (rtReduceOp == RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_MAX)
           || (rtReduceOp == RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_MIN)
           || (rtReduceOp == RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_EQUAL));
    sqe->opcode = isReduce ? GetSdmaOpCode(static_cast<u32>(rtReduceOp), static_cast<u8>(rtDataType)) : 0U;
    HCCL_INFO(
        "[SQE]MemcpySqe copyKind=%u,Opcode=0x%x, streamId=%u, len=%u, src:0x%llx, dst:0x%llx",
        static_cast<u32>(rtReduceOp), static_cast<u32>(sqe->opcode), streamId, length, src, dst);
    sqe->length = length;

    // extract common function for this
    sqe->srcAddrLow = GetAddrLow(src);
    sqe->srcAddrHigh = GetAddrHigh(src);
    sqe->dstAddrLow = GetAddrLow(dst);
    sqe->dstAddrHigh = GetAddrHigh(dst);

    sqe->partid = partId;
}

u8 HcclSdmaSqe::GetSdmaOpCode(u32 copyKind, u8 copyDataType) const
{
    const u8 memcpyDataType = ConvertToMemcpyDataType(copyDataType);
    const u8 opType = ConvertToMemcpyOpType(copyKind);
    // opcode: 高4bit为datatype，低4bit为optype
    return memcpyDataType | opType;
}

// change name: convert
u8 HcclSdmaSqe::ConvertToMemcpyDataType(u8 copyDataType) const
{
    u8 opcode;
    switch (static_cast<RtDataType>(copyDataType)) {
        case RtDataType::RT_DATA_TYPE_INT8: {
            opcode = static_cast<u8>(RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT8);
            break;
        }
        case RtDataType::RT_DATA_TYPE_INT16: {
            opcode = static_cast<u8>(RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT16);
            break;
        }
        case RtDataType::RT_DATA_TYPE_INT32: {
            opcode = static_cast<u8>(RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT32);
            break;
        }
        case RtDataType::RT_DATA_TYPE_FP16: {
            opcode = static_cast<u8>(RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP16);
            break;
        }
        case RtDataType::RT_DATA_TYPE_FP32: {
            opcode = static_cast<u8>(RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP32);
            break;
        }
        case RtDataType::RT_DATA_TYPE_BFP16: {
            opcode = static_cast<u8>(RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_BFP16);
            break;
        }
        default: {
            // Should not run here.
            // Only for code style, 0x80 is reserved value of STRAS opcode.
            MACRO_THROW(
                NotSupportException, StringFormat("DataType=%u do not support.", static_cast<u32>(copyDataType)));
            break;
        }
    }
    return opcode;
}

u8 HcclSdmaSqe::ConvertToMemcpyOpType(u32 copyKind) const
{
    u8 opcode;
    switch (static_cast<RtReduceKind>(copyKind)) {
        case RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_ADD: {
            opcode = static_cast<u8>(RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_ADD);
            break;
        }
        case RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_MAX: {
            opcode = static_cast<u8>(RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_MAX);
            break;
        }
        case RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_MIN: {
            opcode = static_cast<u8>(RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_MIN);
            break;
        }
        case RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_EQUAL: {
            opcode = static_cast<u8>(RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_EQUAL);
            break;
        }
        default: {
            MACRO_THROW(NotSupportException, StringFormat("Type out of range: copyKind=%u", copyKind));
            break;
        }
    }
    return opcode;
}

u64 HcclSdmaSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

} // namespace Hccl
