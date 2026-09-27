/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/aicpu_hccl_sqcqv1.h"
#include "hccl_common.h"
#include "dispatcher_task_types.h"
#include <unordered_map>

namespace {
bool ChipIsHaveStars() { return true; }

uint8_t ReduceOpcodeHigh(uint8_t copyDataType)
{
    uint8_t opcode;
    switch (copyDataType) {
        case ACL_INT8: {
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT8);
            break;
        }
        case ACL_INT16: {
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT16);
            break;
        }
        case ACL_INT32: {
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT32);
            break;
        }
        case ACL_FLOAT16: {
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP16);
            break;
        }
        case ACL_FLOAT: {
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP32);
            break;
        }
        case ACL_BF16: {
            if (ChipIsHaveStars()) {
                opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_BFP16);
            } else {
                HCCL_ERROR("DataType=%u do not support.", static_cast<uint32_t>(copyDataType));
                opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_RESERVED);
            }
            break;
        }
        default: {
            // Should not run here.
            // if not support, it will return RT_ERROR_FEATURE_NOT_SUPPORT at context.cc's reduce ability check.
            // Only for code style, 0x80 is reserved value of STRAS opcode.
            HCCL_ERROR("DataType=%u do not support.", static_cast<uint32_t>(copyDataType));
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_RESERVED);
            break;
        }
    }
    return opcode;
}

uint8_t ReduceOpcodeLow(uint32_t copyKind)
{
    uint8_t opcode;
    switch (copyKind) {
        case ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM: {
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_ADD);
            break;
        }
        case ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX: {
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_MAX);
            break;
        }
        case ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN: {
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_MIN);
            break;
        }
        case ACL_RT_MEMCPY_SDMA_AUTOMATIC_EQUAL: {
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_EQUAL);
            break;
        }
        default: {
            HCCL_ERROR("Type out of range: copyKind=%u", copyKind);
            opcode = static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_RESERVED);
            break;
        }
    }
    return opcode;
}

uint8_t GetOpcodeForReduce(uint32_t copyKind, uint8_t copyDataType)
{
    const uint8_t opcodeHigh = ReduceOpcodeHigh(copyDataType);
    const uint8_t opcodeLow = ReduceOpcodeLow(copyKind);
    if ((static_cast<int32_t>(opcodeHigh) == RT_STARS_MEMCPY_ASYNC_OP_RESERVED)
        || (static_cast<int32_t>(opcodeLow) == RT_STARS_MEMCPY_ASYNC_OP_RESERVED)) {
        // Should not run here. 0x80 is reserved value of STRAS opcode
        return static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_RESERVED);
    } else {
        return opcodeHigh | opcodeLow;
    }
}

std::unordered_map<uint8_t, uint8_t> RT2HCCL_REDUCE_OP_MAP = {
    {static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_CPY), static_cast<uint8_t>(HCCL_REDUCE_RESERVED)},
    {static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_ADD), static_cast<uint8_t>(HCCL_REDUCE_SUM)},
    {static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_MAX), static_cast<uint8_t>(HCCL_REDUCE_MAX)},
    {static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_MIN), static_cast<uint8_t>(HCCL_REDUCE_MIN)},
    {static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_EQUAL), static_cast<uint8_t>(HCCL_REDUCE_RESERVED)},
};

} // namespace

void TranslateOpcode(uint8_t opCode, uint8_t& reduceType)
{
    reduceType = RT2HCCL_REDUCE_OP_MAP[opCode & 0x0F]; // opCode的低4位表示reduce类型
    HCCL_DEBUG("[TranslateOpcode] opCode=%u, reduceType=%u.", opCode, reduceType);
}

void AddOneNotifyWaitSqeV1(
    uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType,
    const dfx::DfxTimeOutConfig& dfxTimeOutConfig)
{
    *sqeType = SqeType::NOTIFY_SQE;
    rtStarsNotifySqeV1_t* const sqe = (rtStarsNotifySqeV1_t*)sqeIn;
    sqe->header.type = RT_STARS_SQE_TYPE_NOTIFY_WAIT;
    const auto& credit_and_time_out = GetTimeOutValue(dfxTimeOutConfig);
    sqe->kernel_credit = static_cast<uint8_t>(credit_and_time_out.first);
    sqe->timeout = credit_and_time_out.second;
    sqe->header.rtStreamId = streamId;
    sqe->notify_id = notifyId;
    sqe->header.taskId = taskId;
    HCCL_INFO(
        "[SQE] notify wait: notifyId=%lu, streamId=%u, taskId=%u, "
        "kernel_credit %u, timeout %u s.",
        notifyId, streamId, taskId, sqe->kernel_credit, sqe->timeout);
}

void AddOneRecordSqeV1(uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::NOTIFY_SQE;
    rtStarsNotifySqeV1_t* const sqe = (rtStarsNotifySqeV1_t*)sqeIn;
    sqe->header.type = RT_STARS_SQE_TYPE_NOTIFY_RECORD;
    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->header.rtStreamId = streamId;
    sqe->notify_id = notifyId;
    sqe->header.taskId = taskId;
    HCCL_INFO(
        "[SQE] notify record: notifyId=%lu, streamId=%u, taskId=%u, "
        "kernel_credit %u, timeout %u s.",
        notifyId, streamId, taskId, sqe->kernel_credit, sqe->timeout);
}

void AddOneWriteValueRecordSqeV1(
    uint16_t streamId, uint16_t taskId, u64 notifyWRAddr, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::WRITE_VALUE_SQE;
    rtStarsWriteValueSqe_t* const sqe = (rtStarsWriteValueSqe_t*)sqeIn;
    sqe->header.type = RT_STARS_SQE_TYPE_WRITE_VALUE;
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->awsize = RT_STARS_WRITE_VALUE_SIZE_TYPE_32BIT;
    sqe->write_value_part0 = 1U;
    sqe->sub_type = RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_NO_PCIE;
    sqe->write_addr_low = static_cast<uint32_t>(notifyWRAddr & MASK_32_BIT);
    sqe->write_addr_high = static_cast<uint32_t>((notifyWRAddr >> UINT32_BIT_NUM) & MASK_17_BIT);
    HCCL_INFO("[SQE] write value: writePtr=0x%lx, streamId=%u, taskId=%u.", notifyWRAddr, streamId, taskId);
}

void AddOneMemcpySqeV1(
    uint16_t streamId, uint16_t taskId, const void* src, uint32_t length, const aclDataType runtimeDataType,
    aclrtReduceKind rtReduceOp, const void* dst, uint32_t partId, uint32_t ssid, uint32_t devId, u64 overflowAddr,
    uint8_t linkType, const uint8_t* sqeIn, uint8_t* sqeType, uint32_t hcclQos)
{
    (void)ssid;
    (void)devId;
    (void)overflowAddr;
    *sqeType = SqeType::MEMCPY_ASYNC_SQE;
    rtStarsMemcpyAsyncSqe_t* const sqe = (rtStarsMemcpyAsyncSqe_t*)sqeIn;

    u32 len, srcAddrLow, srcAddrHigh, dstAddrLow, dstAddrHigh;
    if (length != 0U || src != nullptr || dst != nullptr) {
        len = length;
        srcAddrLow = static_cast<uint32_t>(reinterpret_cast<u64>(src) & 0x00000000ffffffffU);
        srcAddrHigh = static_cast<uint32_t>((reinterpret_cast<u64>(src) & 0xffffffff00000000U) >> UINT32_BIT_NUM);
        dstAddrLow = static_cast<uint32_t>(reinterpret_cast<u64>(dst) & 0x00000000ffffffffU);
        dstAddrHigh = static_cast<uint32_t>((reinterpret_cast<u64>(dst) & 0xffffffff00000000U) >> UINT32_BIT_NUM);
    } else {
        len = sqe->length;
        srcAddrLow = sqe->src_addr_low;
        srcAddrHigh = sqe->src_addr_high;
        dstAddrLow = sqe->dst_addr_low;
        dstAddrHigh = sqe->dst_addr_high;
        (void)memset_s(sqe, sizeof(rtStarsMemcpyAsyncSqe_t), 0, sizeof(rtStarsMemcpyAsyncSqe_t));
    }

    sqe->header.type = RT_STARS_SQE_TYPE_SDMA;
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->kernel_credit = dfx::kCreditTimeDefault;
    const bool isReduce
        = ((rtReduceOp == ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM) || (rtReduceOp == ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX)
           || (rtReduceOp == ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN) || (rtReduceOp == ACL_RT_MEMCPY_SDMA_AUTOMATIC_EQUAL));

    sqe->opcode = isReduce ? GetOpcodeForReduce(rtReduceOp, runtimeDataType) : 0U;
    if (linkType == static_cast<uint8_t>(hccl::LinkType::LINK_SIO)
        || linkType == static_cast<uint8_t>(hccl::LinkType::LINK_ONCHIP)) {
        hcclQos = SDMA_QOS_DEFAULT;
    }
    HCCL_INFO(
        "[SQE]MemcpySqe copyKind=%u,Opcode=0x%x, streamId=%u, len=%u, src:%p, dst:%p, sqe->linkType=%u, hcclQos=%u",
        static_cast<uint32_t>(rtReduceOp), static_cast<uint32_t>(sqe->opcode), streamId, length, src, dst,
        static_cast<unsigned int>(linkType), hcclQos);

    sqe->length = len;
    sqe->src_addr_low = srcAddrLow;
    sqe->src_addr_high = srcAddrHigh;
    sqe->dst_addr_low = dstAddrLow;
    sqe->dst_addr_high = dstAddrHigh;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->qos = hcclQos;
    sqe->partid = partId;
    sqe->linkType = linkType;
}

void AddOneRdmaDbSendSqeV1(
    uint16_t streamId, uint16_t taskId, uint64_t dbInfo, uint64_t dbAddr, uint32_t length, uint8_t rdmaType,
    const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::RDMA_DB_SEND_SQE;
    rtStarsWriteValueSqe_t* const sqe = (rtStarsWriteValueSqe_t*)sqeIn;

    sqe->header.type = RT_STARS_SQE_TYPE_WRITE_VALUE;
    sqe->header.ie = RT_STARS_SQE_INT_DIR_NO;
    sqe->header.preP = RT_STARS_SQE_INT_DIR_NO;
    sqe->header.postP = RT_STARS_SQE_INT_DIR_NO;
    sqe->header.wrCqe = 0U;
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;

    sqe->va = 0U;
    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->awsize = RT_STARS_WRITE_VALUE_SIZE_TYPE_64BIT;

    sqe->sub_type = RT_STARS_WRITE_VALUE_SUB_TYPE_RDMA_DB_SEND;

    if (dbAddr == 0ULL) {
        sqe->header.type = RT_STARS_SQE_TYPE_INVALID;
        return;
    }
    sqe->write_value_part0 = static_cast<uint32_t>(dbInfo & MASK_32_BIT);
    sqe->write_value_part1 = static_cast<uint32_t>(dbInfo >> UINT32_BIT_NUM);
    sqe->write_addr_low = static_cast<uint32_t>(dbAddr & MASK_32_BIT);
    sqe->write_addr_high = static_cast<uint32_t>((dbAddr >> UINT32_BIT_NUM) & MASK_17_BIT);
    sqe->rdmaWrLenth = length; // wr len
    sqe->rdmaType = static_cast<uint32_t>(rdmaType);
    HCCL_INFO(
        "[SQE]RdmaDbSend: length=%u, rdmaType=%u, dbAddr=0x%lx, streamId=%u, taskId=%u.", length, rdmaType, dbAddr,
        streamId, taskId);
}

void AddOneEventResetSqeV1(
    uint16_t streamId, int32_t eventId, uint16_t taskId, int64_t phyChipId, int64_t phyDieId, u64 eventAddr,
    const uint8_t* sqeIn, uint8_t* sqeType)
{
    (void)eventAddr;
    *sqeType = SqeType::WRITE_VALUE_SQE;
    rtStarsWriteValueSqe_t* const sqe = (rtStarsWriteValueSqe_t*)sqeIn;
    sqe->header.type = RT_STARS_SQE_TYPE_WRITE_VALUE;

    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;

    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->res7 = static_cast<uint32_t>(eventId);
    sqe->sub_type = RT_STARS_WRITE_VALUE_SUB_TYPE_EVENT_RESET;

    const u64 eventTableId = static_cast<u64>(eventId) / STARS_EVENT_NUM_OF_SINGLE_TABLE;

    /* same as eventid % STARS_EVENT_NUM_OF_SINGLE_TABLE */
    const u64 eventNum = (static_cast<u64>(eventId)) & 0xFFFUL;
    // 默认devType  为DevType::DEV_TYPE_COUNT  stream->Device_()->GetPhyChipId() 默认0 stream->Device_()->GetPhyDieId()
    // 默认0
    u64 base = static_cast<u64>(
        RT_STARS_BASE_ADDR + (RT_CHIP_ADDR_OFFSET * static_cast<u64>(phyChipId))
        + (RT_DIE_ADDR_OFFSET * static_cast<u64>(phyDieId)) + STARS_EVENT_BASE_ADDR);
    const u64 addr = base + (eventTableId * STARS_EVENT_TABLE_OFFSET) + (eventNum * STARS_EVENT_OFFSET);
    sqe->write_addr_low = static_cast<uint32_t>(addr & MASK_32_BIT);
    sqe->write_addr_high = static_cast<uint32_t>((addr >> UINT32_BIT_NUM) & MASK_17_BIT);
    HCCL_INFO("[SQE] event_reset: eventId=%u, streamId=%u, taskId=%u, addr:%p", eventId, streamId, taskId, addr);
}

void AddOneEventRecordSqeV1(uint16_t streamId, int32_t eventId, uint16_t taskId, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::EVENT_SQE;
    rtStarsEventSqe_t* const sqe = (rtStarsEventSqe_t*)sqeIn;
    sqe->header.type = RT_STARS_SQE_TYPE_EVENT_RECORD;
    sqe->header.wrCqe = 1U; // 1: set wrCqe
    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;

    // eventRecordTaskInfo->waitCqflag  是否为同步task
    if (false) {
        streamId |= RT_SYNC_TASK_FLAG;
    }

    sqe->header.rtStreamId = streamId;
    sqe->eventId = static_cast<uint16_t>(eventId);
    sqe->header.taskId = taskId;
    HCCL_INFO("[SQE] event record: eventId=%d, streamId=%u, taskId=%u.", eventId, streamId, taskId);
}

void AddOneEventWaitSqeV1(uint16_t streamId, int32_t eventId, uint16_t taskId, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::EVENT_SQE;
    rtStarsEventSqe_t* const sqe = (rtStarsEventSqe_t*)sqeIn;
    sqe->header.type = RT_STARS_SQE_TYPE_EVENT_WAIT;
    sqe->kernel_credit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;

    sqe->header.rtStreamId = streamId;
    sqe->eventId = eventId;
    sqe->header.taskId = taskId;
    HCCL_INFO("[SQE] event wait: eventId=%d, streamId=%u, taskId=%u", eventId, streamId, taskId);
}

void AddOneFlipPlaceHolderSqeV1(
    uint16_t streamId, uint16_t flipNum, uint16_t taskId, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::FLIP_PLACEHOLDER_SQE;
    rtStarsPlaceHolderSqe_t* const sqe = (rtStarsPlaceHolderSqe_t*)sqeIn;
    sqe->header.type = RT_STARS_SQE_TYPE_PLACE_HOLDER;
    sqe->header.ie = 0U;
    sqe->header.preP = 1U;
    sqe->header.postP = 0U;
    sqe->header.wrCqe = 0U;
    sqe->header.reserved = 0U;
    sqe->header.blockDim = RT_TASK_TYPE_FLIP; // task type
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->u.flip_task_info.flipNumReport = flipNum;

    HCCL_INFO("[SQE] placeholder: flipNum=%d, streamId=%u, taskId=%u", flipNum, streamId, taskId);
}

void AddOneCacheMemcpyPlaceHolderSqeV1(
    uint16_t streamId, uint16_t taskId, const void* src, const void* dst, uint8_t linkType, const uint8_t* sqeIn,
    uint8_t* sqeType, uint32_t hcclQos)
{
    *sqeType = SqeType::CACHE_MEMCPY_PLACEHOLDER_SQE;
    SetCachePlaceholderHeaderV1(streamId, taskId, sqeIn);

    rtStarsPlaceHolderSqe_t* const sqe = (rtStarsPlaceHolderSqe_t*)sqeIn;
    constexpr uint64_t uintBitWidth = 32;
    uint64_t srcAddr = reinterpret_cast<uint64_t>(src);
    uint64_t dstAddr = reinterpret_cast<uint64_t>(dst);
    sqe->u.cache_memcpy_task_info.src_addr_high = static_cast<uint32_t>(srcAddr >> uintBitWidth);
    sqe->u.cache_memcpy_task_info.src_addr_low = static_cast<uint32_t>(srcAddr & 0xFFFFFFFFULL);
    sqe->u.cache_memcpy_task_info.dst_addr_high = static_cast<uint32_t>(dstAddr >> uintBitWidth);
    sqe->u.cache_memcpy_task_info.dst_addr_low = static_cast<uint32_t>(dstAddr & 0xFFFFFFFFULL);
    sqe->u.cache_memcpy_task_info.kernel_credit = dfx::kCreditTimeDefault;
    sqe->u.cache_memcpy_task_info.linkType = linkType;

    HCCL_INFO(
        "[SQE] cache-memcpy placeholder: streamId=%u, taskId=%u, srcAddr=0x%016llx, dstAddr=0x%016llx, linkType=%u, "
        "hcclQos=%u",
        streamId, taskId, src, dst, linkType, hcclQos);

    // 适配QoS
    if (linkType == static_cast<uint8_t>(hccl::LinkType::LINK_SIO)
        || linkType == static_cast<uint8_t>(hccl::LinkType::LINK_ONCHIP)) {
        hcclQos = SDMA_QOS_DEFAULT;
    }
    HCCL_INFO(
        "[AddOneCacheMemcpyPlaceHolderSqeV1] sqe->linkType=%u hcclQos=%u", static_cast<unsigned int>(linkType),
        static_cast<unsigned int>(hcclQos));
    sqe->u.cache_memcpy_task_info.qos = hcclQos;
}

void AddOneCacheNotifyWaitPlaceholderSqeV1(
    uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType,
    const dfx::DfxTimeOutConfig& dfxTimeOutConfig)
{
    *sqeType = SqeType::CACHE_NOTIFY_PLACEHOLDER_SQE;
    SetCachePlaceholderHeaderV1(streamId, taskId, sqeIn);

    rtStarsPlaceHolderSqe_t* const sqe = (rtStarsPlaceHolderSqe_t*)sqeIn;
    sqe->u.cache_notify_task_info.is_wait = 1; // NotifyWait
    const auto& credit_and_time_out = GetTimeOutValue(dfxTimeOutConfig);
    sqe->u.cache_notify_task_info.kernel_credit = static_cast<uint8_t>(credit_and_time_out.first);
    sqe->u.cache_notify_task_info.timeout = credit_and_time_out.second;
    sqe->u.cache_notify_task_info.notify_id = notifyId;

    HCCL_INFO(
        "[SQE] cache-notify placeholder (wait): notifyId=%lu, streamId=%u, taskId=%u, u.kernel_credit %u, timeout %u "
        "s.",
        notifyId, streamId, taskId, sqe->u.cache_notify_task_info.kernel_credit, sqe->u.cache_notify_task_info.timeout);
}

void AddOneCacheNotifyRecordPlaceholderSqeV1(
    uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::CACHE_NOTIFY_PLACEHOLDER_SQE;
    SetCachePlaceholderHeaderV1(streamId, taskId, sqeIn);

    rtStarsPlaceHolderSqe_t* const sqe = (rtStarsPlaceHolderSqe_t*)sqeIn;
    sqe->u.cache_notify_task_info.is_wait = 0; // NotifyRecord
    sqe->u.cache_notify_task_info.notify_id = notifyId;

    HCCL_INFO(
        "[SQE] cache-notify placeholder (record): notifyId=%lu, streamId=%u, taskId=%u, "
        "kernel_credit %u.",
        notifyId, streamId, taskId, sqe->kernel_credit);
}

void AddOneCacheWriteValuePlaceholderSqeV1(
    uint16_t streamId, uint16_t taskId, u64 notifyWRAddr, const uint8_t* sqeIn, uint8_t* sqeType)
{
    *sqeType = SqeType::CACHE_WRITE_VALUE_PLACEHOLDER_SQE;
    SetCachePlaceholderHeaderV1(streamId, taskId, sqeIn);

    rtStarsPlaceHolderSqe_t* const sqe = (rtStarsPlaceHolderSqe_t*)sqeIn;
    sqe->u.cache_write_value_task_info.write_addr_low = static_cast<uint32_t>(notifyWRAddr & MASK_32_BIT);
    sqe->u.cache_write_value_task_info.write_addr_high
        = static_cast<uint32_t>((notifyWRAddr >> UINT32_BIT_NUM) & MASK_17_BIT);

    HCCL_INFO("[SQE] cache-write placeholder: writePtr=0x%lx, streamId=%u, taskId=%u.", notifyWRAddr, streamId, taskId);
}

void AddOneCacheMemcpyRecordPlaceholderSqeV1(
    uint16_t streamId, uint16_t taskId, const void* src, uint32_t length, const aclDataType runtimeDataType,
    aclrtReduceKind rtReduceOp, const void* dst, uint32_t partId, uint32_t ssid, uint32_t devId, u64 overflowAddr,
    uint8_t linkType, const uint8_t* sqeIn, uint8_t* sqeType, uint32_t hcclQos)
{
    (void)ssid;
    (void)devId;
    (void)overflowAddr;
    *sqeType = SqeType::CACHE_MEMCPY_RECORD_PLACEHOLDER_SQE;
    rtStarsPlaceHolderSqe_t* const sqe = (rtStarsPlaceHolderSqe_t*)sqeIn;

    u32 len, srcAddrLow, srcAddrHigh, dstAddrLow, dstAddrHigh;
    if (length != 0U || src != nullptr || dst != nullptr) {
        len = length;
        srcAddrLow = static_cast<uint32_t>(reinterpret_cast<u64>(src) & 0x00000000ffffffffU);
        srcAddrHigh = static_cast<uint32_t>((reinterpret_cast<u64>(src) & 0xffffffff00000000U) >> UINT32_BIT_NUM);
        dstAddrLow = static_cast<uint32_t>(reinterpret_cast<u64>(dst) & 0x00000000ffffffffU);
        dstAddrHigh = static_cast<uint32_t>((reinterpret_cast<u64>(dst) & 0xffffffff00000000U) >> UINT32_BIT_NUM);
    } else {
        len = sqe->u.cache_memcpy_record_task_info.length;
        srcAddrLow = sqe->u.cache_memcpy_record_task_info.src_addr_low;
        srcAddrHigh = sqe->u.cache_memcpy_record_task_info.src_addr_high;
        dstAddrLow = sqe->u.cache_memcpy_record_task_info.dst_addr_low;
        dstAddrHigh = sqe->u.cache_memcpy_record_task_info.dst_addr_high;
        (void)memset_s(sqe, sizeof(rtStarsPlaceHolderSqe_t), 0, sizeof(rtStarsPlaceHolderSqe_t));
    }

    // 用于placeholder
    SetCachePlaceholderHeaderV1(streamId, taskId, sqeIn);

    // 保存memcpy-record SQE的相关信息
    sqe->u.cache_memcpy_record_task_info.kernel_credit = dfx::kCreditTimeDefault;
    const bool isReduce
        = ((rtReduceOp == ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM) || (rtReduceOp == ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX)
           || (rtReduceOp == ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN) || (rtReduceOp == ACL_RT_MEMCPY_SDMA_AUTOMATIC_EQUAL));
    sqe->u.cache_memcpy_record_task_info.opcode = isReduce ? GetOpcodeForReduce(rtReduceOp, runtimeDataType) : 0U;
    HCCL_INFO(
        "[SQE] cache-write-memcpy placeholder: copyKind=%u,Opcode=0x%x, streamId=%u, len=%u, src:%p, dst:%p, hcclQos: "
        "%u",
        static_cast<uint32_t>(rtReduceOp), static_cast<uint32_t>(sqe->u.cache_memcpy_record_task_info.opcode), streamId,
        length, src, dst, hcclQos);
    sqe->u.cache_memcpy_record_task_info.length = len;
    sqe->u.cache_memcpy_record_task_info.src_addr_low = srcAddrLow;
    sqe->u.cache_memcpy_record_task_info.src_addr_high = srcAddrHigh;
    sqe->u.cache_memcpy_record_task_info.dst_addr_low = dstAddrLow;
    sqe->u.cache_memcpy_record_task_info.dst_addr_high = dstAddrHigh;
    sqe->u.cache_memcpy_record_task_info.partid = partId;
    sqe->u.cache_memcpy_record_task_info.linkType = linkType;

    // 适配qos
    if (linkType == static_cast<uint8_t>(hccl::LinkType::LINK_SIO)
        || linkType == static_cast<uint8_t>(hccl::LinkType::LINK_ONCHIP)) {
        hcclQos = SDMA_QOS_DEFAULT;
    }
    HCCL_INFO(
        "[AddOneCacheMemcpyRecordPlaceholderSqeV1] sqe->linkType=%u hcclQos=%u", static_cast<unsigned int>(linkType),
        static_cast<unsigned int>(hcclQos));
    sqe->u.cache_memcpy_record_task_info.qos = hcclQos;
}

void SetCachePlaceholderHeaderV1(uint16_t streamId, uint16_t taskId, const uint8_t* sqeIn)
{
    rtStarsPlaceHolderSqe_t* const sqe = (rtStarsPlaceHolderSqe_t*)sqeIn;
    sqe->header.type = RT_STARS_SQE_TYPE_PLACE_HOLDER;
    sqe->header.ie = 0U;
    sqe->header.preP = 0U; // 不需要STARS_FW参与任何预处理
    sqe->header.postP = 0U;
    sqe->header.wrCqe = 0U;
    sqe->header.reserved = 0U;
    // NOTE: task type在preP阶段被TASK_FW使用, 而此placeholder无preP阶段, 设置为RT_TASK_TYPE_FLIP不影响功能
    sqe->header.blockDim = RT_TASK_TYPE_FLIP;
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    return;
}

std::pair<uint64_t, uint64_t> GetTimeOutValue(const dfx::DfxTimeOutConfig& dfxTimeOutConfig)
{
    if (dfxTimeOutConfig.useCredit) {
        HCCL_DEBUG("Use hard sync with [%lu s]", dfxTimeOutConfig.sqeCreditTimeOut);
        return {dfxTimeOutConfig.sqeCreditTimeOut, dfx::kTimeOutTimeInvalid};
    }
    HCCL_DEBUG("Use soft sync with [%lu s]", dfxTimeOutConfig.sqeTimeOutTimeOut);
    return {dfx::kCreditTimeInvalid, dfxTimeOutConfig.sqeTimeOutTimeOut};
}
