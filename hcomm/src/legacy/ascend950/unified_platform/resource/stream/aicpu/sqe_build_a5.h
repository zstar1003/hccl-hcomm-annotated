/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_SQE_BUILD_A5_H
#define HCCLV2_SQE_BUILD_A5_H
#include "types.h"
#include "ub_jetty_lite.h"
#include "sqe_v82.h"
#include "log.h"

namespace Hccl {

constexpr u32 LOW_BITS = 16;

u32 GetKernelExecTimeoutFromEnvConfig();

extern thread_local uint8_t g_sqeProfBit;
void SetSqeProfilingEnabled(bool isEnabled);

inline void SetSqeHeaderTaskFields(void* sqe, u32 taskId)
{
    auto header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqe);
    header->rtStreamId = static_cast<uint16_t>(taskId);
    header->taskId = static_cast<uint16_t>(taskId >> LOW_BITS);
}

inline void BuildA5SqeNotifyWait(u32 streamId, u32 taskId, u32 notifyId, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe->header.wrCqe = 0U;

    sqe->cntFlag = false;
    sqe->clrFlag = true;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_SINGLE_NOTIFY_WAIT);
    sqe->kernelCredit
        = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT; // wait任务需要设置为0xff，否则会触发rtsq的超时机制，即使record了也wait不到
    sqe->notifyId = notifyId;
    sqe->timeout = GetKernelExecTimeoutFromEnvConfig();
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeNotifyWait(u32 streamId, u32 taskId, u32 notifyId, u32 timeout, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe->header.wrCqe = 0U;

    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    sqe->cntFlag = false;
    sqe->clrFlag = true;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_SINGLE_NOTIFY_WAIT);
    sqe->notifyId = notifyId;
    sqe->timeout = timeout;
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeNotifyRecord(u32 streamId, u32 taskId, u32 notifyId, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    sqe->header.wrCqe = 0U;

    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_SINGLE_NOTIFY_RECORD);
    sqe->notifyId = notifyId;
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeCnt1toNNotifyRecord(u32 streamId, u32 taskId, u32 notifyId, u32 cntValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    sqe->header.wrCqe = 0U;

    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->clrFlag = false;
    sqe->cntFlag = true;
    sqe->recordModeBit = 0x0U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_RECORD);
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeCnt1toNNotifyWait(u32 streamId, u32 taskId, u32 notifyId, u32 cntValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe->header.wrCqe = 0U;

    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    sqe->cntFlag = true;
    sqe->clrFlag = true;
    sqe->bitmap = 1U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_WAIT);
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeCntNto1NotifyRecord(u32 streamId, u32 taskId, u32 notifyId, u32 cntValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    sqe->header.wrCqe = 0U;
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;

    sqe->clrFlag = false;
    sqe->cntFlag = true;
    sqe->recordModeBit = 0x2U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_RECORD);
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeCntNto1NotifyWait(u32 streamId, u32 taskId, u32 notifyId, u32 cntValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe->header.wrCqe = 0U;

    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    sqe->cntFlag = true;
    sqe->clrFlag = true;
    sqe->waitModeBit = 0x1U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_WAIT);
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeSdmaCopy(
    u32 streamId, u32 taskId, u64 dstAddr, u64 srcAddr, u32 size, u32 partId, u32 opcode, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsMemcpySqe* sqe = (Rt91095StarsMemcpySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
    sqe->header.wrCqe = 0U;

    sqe->opcode = opcode;
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->mapamPartId = partId;

    sqe->u.strideMode0.lengthMove = size;
    sqe->u.strideMode0.srcAddrLow = static_cast<uint32_t>(srcAddr & 0x00000000ffffffffU);
    sqe->u.strideMode0.srcAddrHigh = static_cast<uint32_t>((srcAddr & 0xffffffff00000000U) >> 32);
    sqe->u.strideMode0.dstAddrLow = static_cast<uint32_t>(dstAddr & 0x00000000ffffffffU);
    sqe->u.strideMode0.dstAddrHigh = static_cast<uint32_t>((dstAddr & 0xffffffff00000000U) >> 32);
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void
BuildA5SqeUbDbSend(u32 streamId, u32 taskId, const UbJettyLiteId& jettyLiteId, u16 piValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsUbdmaDBmodeSqe* sqe = (Rt91095StarsUbdmaDBmodeSqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);

    sqe->mode = Rt91095UbDmaSqeMode::RT_91095_SQE_DOORBELL_MODE;
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->doorbellNum = 1U;
    sqe->jettyId1 = jettyLiteId.GetJettyId();
    sqe->funcId1 = jettyLiteId.GetFuncId();
    sqe->piValue1 = piValue;
    sqe->dieId1 = jettyLiteId.GetDieId();
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeP2pWriteValue(u32 streamId, u32 taskId, u64 remoteAddr, u32 writeValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsWriteValueSqe* sqe = reinterpret_cast<Rt91095StarsWriteValueSqe*>(sqeIn);
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE);

    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->writeAddrLow = remoteAddr & MASK_32_BIT;
    sqe->writeAddrHigh = (remoteAddr >> UINT32_BIT_NUM) & MASK_17_BIT;
    sqe->awsize = RtStarsWriteValueSizeType::RT_STARS_WRITE_VALUE_SIZE_TYPE_32BIT;
    sqe->writeValuePart[0] = writeValue;
    sqe->va = 1;
    sqe->header.sqeProf = g_sqeProfBit;
}

u32 GetKernelExecTimeoutFromEnvConfig();

void BuildA5SqeCCoreNotifyWait(u32 streamId, u32 taskId, u64 waitAddr, u64 actAddr, bool last, uint8_t* const sqeIn);

void BuildA5SqeCCoreNotifyRecord(u32 streamId, u32 taskId, u64 writeAddr, u64 valueAddr, uint8_t* const sqeIn);

void BuildA5SqeRdmaDbSend(u32 streamId, u32 taskId, u64 dbAddr, u64 dbValue, uint8_t* const sqeIn);

} // namespace Hccl

#endif
