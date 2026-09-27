/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/aicpu_hccl_sqcqv2.h"
#include "hccl_common.h"
#include "dispatcher_task_types.h"
namespace {
enum class SdmaReduceOpcode {
    SDMA_OPCODE_NOT_ATOMIC = 0,
    SDMA_OPCODE_INT_16 = 1,
    SDMA_OPCODE_FLOAT_32 = 2,
    SDMA_OPCODE_FLOAT_16 = 3
};

uint8_t ReduceOpcode(uint8_t copyDataType)
{
    uint8_t opcode;
    switch (copyDataType) {
        case ACL_INT16: {
            opcode = static_cast<uint8_t>(SdmaReduceOpcode::SDMA_OPCODE_INT_16);
            break;
        }
        case ACL_FLOAT16: {
            opcode = static_cast<uint8_t>(SdmaReduceOpcode::SDMA_OPCODE_FLOAT_16);
            break;
        }
        case ACL_FLOAT: {
            opcode = static_cast<uint8_t>(SdmaReduceOpcode::SDMA_OPCODE_FLOAT_32);
            break;
        }
        default: {
            opcode = static_cast<uint8_t>(SdmaReduceOpcode::SDMA_OPCODE_FLOAT_32);
            break;
        }
    }
    return opcode;
}
} // namespace

void AddOneNotifyWaitSqeV2(
    uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType,
    const dfx::DfxTimeOutConfig& dfxTimeOutConfig)
{
    (void)dfxTimeOutConfig;
    *sqeType = SqeType::NOTIFY_SQE_V2;
    rtStarsNotifySqeV2_t* const sqe = (rtStarsNotifySqeV2_t*)sqeIn;
    sqe->header.type = RT_HW_STARS_SQE_TYPE_NOTIFY_WAIT;
    sqe->kernel_credit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    sqe->header.rt_stream_id = streamId;
    sqe->notify_id = notifyId;
    sqe->header.task_id = taskId;
    HCCL_INFO("[SQE] notify wait: notifyId=%lu, streamId=%u, taskId=%u.", notifyId, streamId, taskId);
}

void AddOneRecordSqeV2(uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::NOTIFY_SQE_V2;
    rtStarsNotifySqeV2_t* const sqe = (rtStarsNotifySqeV2_t*)sqeIn;
    sqe->header.type = RT_HW_STARS_SQE_TYPE_NOTIFY_RECORD;
    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->header.rt_stream_id = streamId;
    sqe->notify_id = notifyId;
    sqe->header.task_id = taskId;
    HCCL_INFO("[SQE] notify record: notifyId=%lu, streamId=%u, taskId=%u.", notifyId, streamId, taskId);
}

void AddOneWriteValueRecordSqeV2(
    uint16_t streamId, uint16_t taskId, u64 notifyWRAddr, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::WRITE_VALUE_SQE_V2;
    rtStarsWriteValueSqeV2_t* const sqe = (rtStarsWriteValueSqeV2_t*)sqeIn;
    sqe->header.type = RT_HW_STARS_SQE_TYPE_WRITE_VALUE;
    sqe->header.rt_stream_id = streamId;
    sqe->header.task_id = taskId;

    sqe->awsize = RT_STARS_WRITE_VALUE_SIZE_TYPE_64BIT;
    sqe->awprot = 2; /* 2: b:010 unprivileged access non-secure access instruction access */

    sqe->write_val[0] = 1U;
    sqe->reg_addr_low = static_cast<uint32_t>(notifyWRAddr & MASK_32_BIT);
    sqe->reg_addr_high = static_cast<uint16_t>(notifyWRAddr >> UINT32_BIT_NUM);
    HCCL_INFO("[SQE] write value: writePtr=0x%lx, streamId=%u, task_id=%u.", notifyWRAddr, streamId, taskId);
}

void AddOneMemcpySqeV2(
    uint16_t streamId, uint16_t taskId, const void* src, uint32_t length, const aclDataType runtimeDataType,
    aclrtReduceKind rtReduceOp, const void* dst, uint32_t partId, uint32_t ssid, uint32_t devId, u64 overflowAddr,
    uint8_t linkType, const uint8_t* sqeIn, uint8_t* sqeType, uint32_t hcclQos)
{
    (void)partId;
    (void)linkType;
    *sqeType = SqeType::MEMCPY_ASYNC_SQE_V2;
    rtStarsMemcpyAsyncSqeV2_t* const sqe = (rtStarsMemcpyAsyncSqeV2_t*)sqeIn;
    uint16_t smmuStreamId;
    if (devId == 0) {
        smmuStreamId = 0x7F45;
    } else {
        smmuStreamId = 0xBF45;
    }

    sqe->type = RT_HW_STARS_SQE_TYPE_SDMA;

    sqe->rt_stream_id = streamId;
    sqe->task_id = taskId;

    sqe->sro = 1U;
    sqe->dro = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->sssv = 1U;
    sqe->dssv = 1U;

    sqe->src_streamid = static_cast<uint16_t>(smmuStreamId);
    sqe->dst_streamid = static_cast<uint16_t>(smmuStreamId);

    sqe->src_substreamid = ssid;
    sqe->dst_substreamid = ssid;

    sqe->length = length;
    sqe->kernel_credit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    sqe->src_addr_low = static_cast<uint32_t>(reinterpret_cast<u64>(src) & 0x00000000ffffffffU);
    sqe->src_addr_high = static_cast<uint32_t>((reinterpret_cast<u64>(src) & 0xffffffff00000000U) >> UINT32_BIT_NUM);
    sqe->dst_addr_low = static_cast<uint32_t>(reinterpret_cast<u64>(dst) & 0x00000000ffffffffU);
    sqe->dst_addr_high = static_cast<uint32_t>((reinterpret_cast<u64>(dst) & 0xffffffff00000000U) >> UINT32_BIT_NUM);
    sqe->overflow_en = 1U;
    sqe->overflow_addr_low = static_cast<uint32_t>(reinterpret_cast<u64>(overflowAddr) & 0x00000000ffffffffU);
    sqe->overflow_addr_high
        = static_cast<uint32_t>((reinterpret_cast<u64>(overflowAddr) & 0xffffffff00000000U) >> UINT32_BIT_NUM);
    if (linkType == static_cast<uint8_t>(hccl::LinkType::LINK_SIO)
        || linkType == static_cast<uint8_t>(hccl::LinkType::LINK_ONCHIP)) {
        hcclQos = SDMA_QOS_DEFAULT;
    }
    sqe->qos = hcclQos;
    const bool isReduce = (rtReduceOp == ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM);
    sqe->opcode
        = isReduce ? ReduceOpcode(runtimeDataType) : static_cast<uint8_t>(SdmaReduceOpcode::SDMA_OPCODE_NOT_ATOMIC);
    HCCL_INFO(
        "[SQE]MemcpySqe copyKind=%u,Opcode=0x%x, streamId=%u, len=%u, src:%p, dst:%p src_substreamid:%u "
        "dst_substreamid:%u src_streamid:%x dst_streamid:%x overflowAddr:%llx sqe->linkType=%u hcclQos=%u",
        static_cast<uint32_t>(rtReduceOp), static_cast<uint32_t>(sqe->opcode), streamId, length, src, dst,
        sqe->src_substreamid, sqe->dst_substreamid, sqe->src_streamid, sqe->dst_streamid, overflowAddr,
        static_cast<unsigned int>(linkType), hcclQos);
}

void AddOneEventResetSqeV2(
    uint16_t streamId, int32_t eventId, uint16_t taskId, int64_t phyChipId, int64_t phyDieId, u64 addr,
    const uint8_t* sqeIn, uint8_t* sqeType)
{
    (void)phyChipId;
    (void)phyDieId;
    *sqeType = SqeType::WRITE_VALUE_SQE_V2;
    rtStarsWriteValueSqeV2_t* const sqe = (rtStarsWriteValueSqeV2_t*)sqeIn;
    sqe->header.type = RT_STARS_SQE_TYPE_WRITE_VALUE;

    sqe->header.rt_stream_id = streamId;
    sqe->header.task_id = taskId;

    sqe->awprot = 0x2; /* b:010 unprivileged access non-secure access instruction access */
    sqe->awsize = RT_STARS_WRITE_VALUE_SIZE_TYPE_64BIT; /* 64bit */

    sqe->reg_addr_low = static_cast<uint32_t>(addr & MASK_32_BIT);
    sqe->reg_addr_high = static_cast<uint32_t>((addr >> UINT32_BIT_NUM) & MASK_17_BIT);
    HCCL_INFO("[SQE] event_reset: eventId=%u, streamId=%u, taskId=%u, addr:%p", eventId, streamId, taskId, addr);
}

void AddOneEventRecordSqeV2(uint16_t streamId, int32_t eventId, uint16_t taskId, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::EVENT_SQE_V2;
    rtStarsEventSqeV2_t* const sqe = (rtStarsEventSqeV2_t*)sqeIn;
    sqe->type = RT_STARS_SQE_TYPE_EVENT_RECORD;

    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;

    sqe->rt_stream_id = streamId;
    sqe->event_id = static_cast<uint16_t>(eventId);
    sqe->task_id = taskId;

    HCCL_INFO("[SQE] event record: eventId=%d, streamId=%u, taskId=%u.", eventId, streamId, taskId);
}

void AddOneEventWaitSqeV2(uint16_t streamId, int32_t eventId, uint16_t taskId, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::EVENT_SQE_V2;
    rtStarsEventSqeV2_t* const sqe = (rtStarsEventSqeV2_t*)sqeIn;
    sqe->type = RT_STARS_SQE_TYPE_EVENT_WAIT;
    sqe->kernel_credit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;

    sqe->rt_stream_id = streamId;
    sqe->event_id = eventId;
    sqe->task_id = taskId;
    HCCL_INFO("[SQE] event wait: eventId=%d, streamId=%u, taskId=%u", eventId, streamId, taskId);
}
