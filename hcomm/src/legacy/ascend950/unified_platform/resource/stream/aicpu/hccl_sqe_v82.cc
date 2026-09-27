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
#include "hccl_sqe_v82.h"
#include "exception_util.h"
#include "not_support_exception.h"

namespace Hccl {

HcclUBDmaDBSqe::HcclUBDmaDBSqe()
{
    sqe = std::make_unique<Rt91095StarsUbdmaDBmodeSqe>();
    (void)memset_s(sqe.get(), sizeof(Rt91095StarsUbdmaDBmodeSqe), 0, sizeof(Rt91095StarsUbdmaDBmodeSqe));
    sqe->header.wrCqe = 0U;
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
    sqe->header.lock = 0U;
    sqe->header.unlock = 0U;
    sqe->header.ie = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;
    sqe->header.preP = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;
    sqe->header.postP = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;
    sqe->mode = Rt91095UbDmaSqeMode::RT_91095_SQE_DOORBELL_MODE;
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->sqeLength = 0U;
}

void HcclUBDmaDBSqe::Config(u16 streamId, u16 taskId, u16 jettyid, u8 funcId, u16 piValue, u16 dieId)
{
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;

    sqe->doorbellNum = 1U;
    sqe->jettyId1 = jettyid;
    sqe->funcId1 = funcId;
    sqe->piValue1 = piValue;
    sqe->dieId1 = dieId;
    HCCL_INFO(
        "[SQE]HcclUBDmaDBSqe streamId=%u, taskId=%u, jettyid=%u, funcId=%u, dieId=%u, piValue=%u", streamId, taskId,
        jettyid, funcId, dieId, piValue);
}

u64 HcclUBDmaDBSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

HcclUBNotifyWaitSqe::HcclUBNotifyWaitSqe()
{
    sqe = std::make_unique<Rt91095StarsNotifySqe>();
    (void)memset_s(sqe.get(), sizeof(Rt91095StarsNotifySqe), 0, sizeof(Rt91095StarsNotifySqe));
    sqe->header.ie = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;
    sqe->header.preP = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;
    sqe->header.postP = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;
    sqe->header.wrCqe = 0U;
    sqe->header.headUpdate = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;

    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe->cntFlag = false;
    sqe->clrFlag = true;
    sqe->waitModeBit = 0U;
    sqe->recordModeBit = 0U;
    sqe->cntValue = 0U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_SINGLE_NOTIFY_WAIT);
}

void HcclUBNotifyWaitSqe::Config(u16 streamId, u16 taskId, u64 notifyId)
{
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->notifyId = notifyId;

    HCCL_INFO("[SQE]HcclUBNotifyWaitSqe streamId=%u, taskId=%u, notifyId=%llu", streamId, taskId, notifyId);
}

u64 HcclUBNotifyWaitSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

HcclUBNotifyRecordSqe::HcclUBNotifyRecordSqe()
{
    sqe = std::make_unique<Rt91095StarsNotifySqe>();
    (void)memset_s(sqe.get(), sizeof(Rt91095StarsNotifySqe), 0, sizeof(Rt91095StarsNotifySqe));
    sqe->header.ie = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;
    sqe->header.preP = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;
    sqe->header.postP = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;
    sqe->header.wrCqe = 0U;
    sqe->header.headUpdate = RtStarsSqeIntDirType::RT_STARS_SQE_INT_DIR_NO;
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->clrFlag = 0U;
    sqe->cntFlag = 0U;
    sqe->cntValue = 0U;
    sqe->waitModeBit = 0U;
    sqe->recordModeBit = 0U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_SINGLE_NOTIFY_RECORD);
}

void HcclUBNotifyRecordSqe::Config(u16 streamId, u16 taskId, u64 notifyId)
{
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->notifyId = notifyId;
    HCCL_INFO("[SQE]HcclUBNotifyRecordSqe streamId=%u, taskId=%u, notifyId=%llu", streamId, taskId, notifyId);
}

u64 HcclUBNotifyRecordSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

HcclUBCntNotifyNto1RecordSqe::HcclUBCntNotifyNto1RecordSqe()
{
    sqe = std::make_unique<Rt91095StarsNotifySqe>();
    (void)memset_s(sqe.get(), sizeof(Rt91095StarsNotifySqe), 0, sizeof(Rt91095StarsNotifySqe));
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->clrFlag = false;
    sqe->cntFlag = true;
    sqe->recordModeBit = 0x2U; // rtCntNotifyRecordMode_t::RECORD_WRITE_BIT_MODE
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_RECORD);
}

void HcclUBCntNotifyNto1RecordSqe::Config(u16 streamId, u16 taskId, u64 notifyId, u32 cntValue)
{
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    HCCL_INFO(
        "[SQE]HcclUBCntNotifyNto1RecordSqe streamId=%u, taskId=%u, notifyId=%llu, cntValue=%u", streamId, taskId,
        notifyId, cntValue);
}

u64 HcclUBCntNotifyNto1RecordSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

HcclUBCntNotify1toNWaitSqe::HcclUBCntNotify1toNWaitSqe()
{
    sqe = std::make_unique<Rt91095StarsNotifySqe>();
    (void)memset_s(sqe.get(), sizeof(Rt91095StarsNotifySqe), 0, sizeof(Rt91095StarsNotifySqe));
    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe->cntFlag = true;
    sqe->clrFlag = true;
    sqe->bitmap = 1U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_WAIT);
}

void HcclUBCntNotify1toNWaitSqe::Config(u16 streamId, u16 taskId, u64 notifyId, u32 cntValue)
{
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    HCCL_INFO(
        "[SQE]HcclUBCntNotify1toNWaitSqe streamId=%u, taskId=%u, notifyId=%llu, cntValue=%u", streamId, taskId,
        notifyId, cntValue);
}

u64 HcclUBCntNotify1toNWaitSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

HcclUBCntNotifyNto1WaitSqe::HcclUBCntNotifyNto1WaitSqe()
{
    sqe = std::make_unique<Rt91095StarsNotifySqe>();
    (void)memset_s(sqe.get(), sizeof(Rt91095StarsNotifySqe), 0, sizeof(Rt91095StarsNotifySqe));
    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe->cntFlag = true;
    sqe->clrFlag = true;
    sqe->waitModeBit = 0x1U; // rtCntNotifyWaitMode_t::WAIT_EQUAL_MODE
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_WAIT);
}

void HcclUBCntNotifyNto1WaitSqe::Config(u16 streamId, u16 taskId, u64 notifyId, u32 cntValue)
{
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    HCCL_INFO(
        "[SQE]HcclUBCntNotifyNto1WaitSqe streamId=%u, taskId=%u, notifyId=%llu, cntValue=%u", streamId, taskId,
        notifyId, cntValue);
}

u64 HcclUBCntNotifyNto1WaitSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

HcclUBCntNotify1toNRecordSqe::HcclUBCntNotify1toNRecordSqe()
{
    sqe = std::make_unique<Rt91095StarsNotifySqe>();
    (void)memset_s(sqe.get(), sizeof(Rt91095StarsNotifySqe), 0, sizeof(Rt91095StarsNotifySqe));
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->clrFlag = false;
    sqe->cntFlag = true;
    sqe->recordModeBit = 0x0U; // rtCntNotifyRecordMode_t::RECORD_STORE_MODE
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_RECORD);
}

void HcclUBCntNotify1toNRecordSqe::Config(u16 streamId, u16 taskId, u64 notifyId, u32 cntValue)
{
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    HCCL_INFO(
        "[SQE]HcclUBCntNotify1toNRecordSqe streamId=%u, taskId=%u, notifyId=%llu, cntValue=%u", streamId, taskId,
        notifyId, cntValue);
}

u64 HcclUBCntNotify1toNRecordSqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

HcclUBMemcpySqe::HcclUBMemcpySqe()
{
    sqe = std::make_unique<Rt91095StarsMemcpySqe>();
    (void)memset_s(sqe.get(), sizeof(Rt91095StarsMemcpySqe), 0, sizeof(Rt91095StarsMemcpySqe));
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
    sqe->header.lock = 0U;
    sqe->header.unlock = 0U;
    sqe->header.ie = 0U;
    sqe->header.wrCqe = 0U;
    sqe->header.ptrMode = 0U;
    sqe->header.rttMode = 0U;
    sqe->header.headUpdate = 0U;
    sqe->header.sqeProf = 0U;
    sqe->header.numBlocks = 0U;

    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;

    sqe->ie2 = 0U;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->qos = 0U;
    sqe->dns = 1U;
    sqe->sro = 0U;
    sqe->dro = 0U;
    sqe->mapamPartId = 0U; // 这里走的memcpy，如果走withcfg,需要传入qoscfg
    sqe->mpamns = 0U;
    sqe->stride = 0U;
    sqe->compEn = 0U;
    sqe->pmg = 0U;
    sqe->res1 = 0U;
    sqe->res2 = 0U;
    sqe->res3 = 0U;
    sqe->res4 = 0U;

    sqe->d2dOffsetFlag = 0U;
    sqe->u.strideMode0.srcOffsetLow = 0U;
    sqe->u.strideMode0.dstOffsetLow = 0U;
    sqe->u.strideMode0.srcOffsetHigh = 0U;
    HCCL_INFO("[SQE]HcclUBMemcpySqe construct end");
}

void HcclUBMemcpySqe::Config(
    u16 streamId, u16 taskId, RtDataType rtDataType, RtReduceKind rtReduceOp, u64 count, const u64* src, const u64* dst,
    u32 partId)
{
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    const bool isReduce
        = ((rtReduceOp == RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_ADD)
           || (rtReduceOp == RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_MAX)
           || (rtReduceOp == RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_MIN)
           || (rtReduceOp == RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_EQUAL));
    sqe->opcode = isReduce ? GetUBOpCode(static_cast<u32>(rtReduceOp), static_cast<u8>(rtDataType)) : 0U;

    sqe->u.strideMode0.lengthMove = count;
    sqe->u.strideMode0.srcAddrLow = static_cast<uint32_t>(static_cast<uint64_t>(*src) & 0x00000000ffffffffU);
    sqe->u.strideMode0.srcAddrHigh
        = static_cast<uint32_t>((static_cast<uint64_t>(*src) & 0xffffffff00000000U) >> UINT32_BIT_NUM);
    sqe->u.strideMode0.dstAddrLow = static_cast<uint32_t>(static_cast<uint64_t>(*dst) & 0x00000000ffffffffU);
    sqe->u.strideMode0.dstAddrHigh
        = static_cast<uint32_t>((static_cast<uint64_t>(*dst) & 0xffffffff00000000U) >> UINT32_BIT_NUM);
    sqe->mapamPartId = partId;

    HCCL_INFO(
        "[SQE]HcclUBMemcpySqe dataType=%u,rtReduceOp =%u, count=%llu, src=%p, dst=%p, partId=%u, streamId=%u, "
        "taskId=%u",
        rtDataType, rtReduceOp, count, src, dst, partId, streamId, taskId);
    HCCL_INFO(
        "[SQE]HcclUBMemcpySqe sqe->opcode=%u sqe->u.strideMode0.srcAddrLow=0x%x, "
        "sqe->u.strideMode0.srcAddrHigh=0x%x,sqe->u.strideMode0.dstAddrLow=0x%x, "
        "sqe->u.strideMode0.dstAddrHigh=0x%x",
        sqe->opcode, sqe->u.strideMode0.srcAddrLow, sqe->u.strideMode0.srcAddrHigh, sqe->u.strideMode0.dstAddrLow,
        sqe->u.strideMode0.dstAddrHigh);
}

u64 HcclUBMemcpySqe::GetSqe() { return ReinterpretAs<u64>(sqe.get()); }

// change name: convert
u8 HcclUBMemcpySqe::ConvertToMemcpyDataType(u8 copyDataType) const
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

u8 HcclUBMemcpySqe::ConvertToMemcpyOpType(u32 copyKind) const
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

u8 HcclUBMemcpySqe::GetUBOpCode(u32 copyKind, u8 copyDataType) const
{
    const u8 memcpyDataType = ConvertToMemcpyDataType(copyDataType);
    const u8 opType = ConvertToMemcpyOpType(copyKind);
    // opcode: 高4bit为datatype，低4bit为optype
    return memcpyDataType | opType;
}

} // namespace Hccl
