/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/aicpu_sqe_context.h"

#include <sstream>
#include <unordered_map>

#include "common/aicpu_hccl_common.h"
#include "utils/hccl_aicpu_utils.h"
#include "securec.h"

namespace {
void ParseNotifySqe(const uint8_t* sqeLocal, uint32_t addInfo, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsNotifySqeV1_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->notifyId = sqe->notify_id;
    info->remoteRank = addInfo;
}
void ParseWriteValueSqe(const uint8_t* sqeLocal, uint32_t addInfo, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsWriteValueSqe_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->subType = sqe->sub_type;
    info->eventId = sqe->res7;
    info->addr1High = sqe->write_addr_high;
    info->addr1Low = sqe->write_addr_low;
    info->remoteRank = addInfo;
    info->length = sqe->rdmaWrLenth;            // rdma wr len
    info->taskRelated.rdmaType = sqe->rdmaType; // rdma type
}
void ParseFlipPlaceHolderSqe(const uint8_t* sqeLocal, uint32_t addInfo, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->length = sqe->u.flip_task_info.flipNumReport;
    info->remoteRank = addInfo;
}
void ParseCacheMemcpyPlaceholderSqe(const uint8_t* sqeLocal, uint32_t addInfo, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    // 注意: cache hit后重新生成的placeholder SQE中的src/dst addr为0 (无需刷新, 因为addr字段只在第一次cache miss时使用,
    // 用于确定memory type和target rank)
    info->addr1High = sqe->u.cache_memcpy_task_info.src_addr_high;
    info->addr1Low = sqe->u.cache_memcpy_task_info.src_addr_low;
    info->addr2High = sqe->u.cache_memcpy_task_info.dst_addr_high;
    info->addr2Low = sqe->u.cache_memcpy_task_info.dst_addr_low;
    info->remoteRank = addInfo >> 16; // 16 bit
    info->dataType = static_cast<uint16_t>(addInfo);
    info->taskRelated.linkType = static_cast<uint8_t>(sqe->u.cache_memcpy_task_info.linkType); // linkType
}
void ParseCacheNotifyPlaceholderSqe(const uint8_t* sqeLocal, uint32_t addInfo, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->notifyId = sqe->u.cache_notify_task_info.notify_id;
    info->remoteRank = addInfo;
}
void ParseCacheWriteValuePlaceholderSqe(const uint8_t* sqeLocal, uint32_t addInfo, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->subType = RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_NO_PCIE;
    info->addr1High = sqe->u.cache_write_value_task_info.write_addr_high;
    info->addr1Low = sqe->u.cache_write_value_task_info.write_addr_low;
    info->remoteRank = addInfo;
}
void ParseCacheMemcpyRecordPlaceholderSqe(const uint8_t* sqeLocal, uint32_t addInfo, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->opCode = sqe->u.cache_memcpy_record_task_info.opcode;
    info->length = sqe->u.cache_memcpy_record_task_info.length;
    info->addr1High = sqe->u.cache_memcpy_record_task_info.src_addr_high;
    info->addr1Low = sqe->u.cache_memcpy_record_task_info.src_addr_low;
    info->addr2High = sqe->u.cache_memcpy_record_task_info.dst_addr_high;
    info->addr2Low = sqe->u.cache_memcpy_record_task_info.dst_addr_low;
    info->partId = sqe->u.cache_memcpy_record_task_info.partid;
    info->remoteRank = addInfo >> 16; // 16 bit
    info->dataType = static_cast<uint16_t>(addInfo);
    info->taskRelated.linkType = static_cast<uint8_t>(sqe->u.cache_memcpy_record_task_info.linkType); // linkType
}
void ParseEventSqe(const uint8_t* sqeLocal, uint32_t /* addInfo */, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsEventSqe_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->eventId = sqe->eventId;
}
void ParseMemcpyAsyncSqe(const uint8_t* sqeLocal, uint32_t addInfo, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsMemcpyAsyncSqe_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rtStreamId;
    info->taskId = sqe->header.taskId;
    info->opCode = sqe->opcode;
    info->length = sqe->length;
    info->addr1High = sqe->src_addr_high;
    info->addr1Low = sqe->src_addr_low;
    info->addr2High = sqe->dst_addr_high;
    info->addr2Low = sqe->dst_addr_low;
    info->partId = sqe->partid;
    info->remoteRank = addInfo >> 16; // 16 bit
    info->dataType = static_cast<uint16_t>(addInfo);
    info->taskRelated.linkType = static_cast<uint8_t>(sqe->linkType); // linkType
}
void ParseCcoreWaitStartSqe(const uint8_t* sqeLocal, uint32_t addInfo, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsCcoreWaitStartSqe_t*>(sqeLocal);
    info->type = sqe->sqeHeader.type;
    info->streamId = sqe->sqeHeader.rtStreamId;
    info->taskId = sqe->sqeHeader.taskId;
    info->addr1High = sqe->ldrImm2.immdAddrHigh;
    info->addr1Low = sqe->ldrImm2.immdAddrLow;
    info->condValue = addInfo >> 16; // 16 bit
    info->isLast = addInfo & 1;
}
void ParseCcoreWriteValueSqe(const uint8_t* sqeLocal, uint32_t addInfo, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsCcoreWriteValueSqe_t*>(sqeLocal);
    info->type = sqe->sqeHeader.type;
    info->streamId = sqe->sqeHeader.rtStreamId;
    info->taskId = sqe->sqeHeader.taskId;
    info->addr1High = sqe->lhwi1.immd;
    info->addr1Low = sqe->llwi1.immdHigh;
    info->addr2High = sqe->llwi1.immdLow;
    info->condValue = addInfo;
}
void ParseNotifySqeV2(const uint8_t* sqeLocal, uint32_t /* addInfo */, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsNotifySqeV2_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rt_stream_id;
    info->taskId = sqe->header.task_id;
    info->notifyId = sqe->notify_id;
}
void ParseWriteValueSqeV2(const uint8_t* sqeLocal, uint32_t /* addInfo */, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsWriteValueSqeV2_t*>(sqeLocal);
    info->type = sqe->header.type;
    info->streamId = sqe->header.rt_stream_id;
    info->taskId = sqe->header.task_id;
    info->addr1High = sqe->reg_addr_high;
    info->addr1Low = sqe->reg_addr_low;
}
void ParseEventSqeV2(const uint8_t* sqeLocal, uint32_t /* addInfo */, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsEventSqeV2_t*>(sqeLocal);
    info->type = sqe->type;
    info->streamId = sqe->rt_stream_id;
    info->taskId = sqe->task_id;
    info->eventId = sqe->event_id;
}
void ParseMemcpyAsyncSqeV2(const uint8_t* sqeLocal, uint32_t /* addInfo */, SqeInfo* info)
{
    auto sqe = reinterpret_cast<const rtStarsMemcpyAsyncSqeV2_t*>(sqeLocal);
    info->type = sqe->type;
    info->streamId = sqe->rt_stream_id;
    info->taskId = sqe->task_id;
    info->opCode = sqe->opcode;
    info->length = sqe->length;
    info->addr1High = sqe->src_addr_high;
    info->addr1Low = sqe->src_addr_low;
    info->addr2High = sqe->dst_addr_high;
    info->addr2Low = sqe->dst_addr_low;
}
} // namespace

HcclResult SqeContextUtils::QuerySqeInfo(const uint8_t* sqeLocal, uint8_t sqeType, uint32_t addInfo, SqeInfo* info)
{
    static const std::unordered_map<uint8_t, void (*)(const uint8_t*, uint32_t, SqeInfo*)> funcMap
        = {{SqeType::NOTIFY_SQE, ParseNotifySqe},
           {SqeType::WRITE_VALUE_SQE, ParseWriteValueSqe},
           {SqeType::EVENT_SQE, ParseEventSqe},
           {SqeType::MEMCPY_ASYNC_SQE, ParseMemcpyAsyncSqe},
           {SqeType::CCORE_WAIT_START_SQE, ParseCcoreWaitStartSqe},
           {SqeType::CCORE_WRITE_VALUE_SQE, ParseCcoreWriteValueSqe},
           {SqeType::NOTIFY_SQE_V2, ParseNotifySqeV2},
           {SqeType::WRITE_VALUE_SQE_V2, ParseWriteValueSqeV2},
           {SqeType::EVENT_SQE_V2, ParseEventSqeV2},
           {SqeType::MEMCPY_ASYNC_SQE_V2, ParseMemcpyAsyncSqeV2},
           {SqeType::RDMA_DB_SEND_SQE, ParseWriteValueSqe},
           {SqeType::FLIP_PLACEHOLDER_SQE, ParseFlipPlaceHolderSqe},
           {SqeType::CACHE_MEMCPY_PLACEHOLDER_SQE, ParseCacheMemcpyPlaceholderSqe},
           {SqeType::CACHE_NOTIFY_PLACEHOLDER_SQE, ParseCacheNotifyPlaceholderSqe},
           {SqeType::CACHE_WRITE_VALUE_PLACEHOLDER_SQE, ParseCacheWriteValuePlaceholderSqe},
           {SqeType::CACHE_MEMCPY_RECORD_PLACEHOLDER_SQE, ParseCacheMemcpyRecordPlaceholderSqe}};
    auto it = funcMap.find(sqeType);
    if (it == funcMap.cend()) {
        HCCL_WARNING("sqetype:%u is unsupported", sqeType);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(info);
    (it->second)(sqeLocal, addInfo, info);
    info->valid = 1;
    return HCCL_SUCCESS;
}

std::string SqeContextUtils::RtsqTaskTypeToStr(uint8_t type)
{
    switch (type) {
        case RT_STARS_SQE_TYPE_NOTIFY_WAIT:
            return "NOTIFY WAIT";
        case RT_STARS_SQE_TYPE_NOTIFY_RECORD:
            return "NOTIFY RECORD";
        case RT_STARS_SQE_TYPE_WRITE_VALUE:
            return "WRITE VALUE";
        case RT_STARS_SQE_TYPE_SDMA:
            return "SDMA";
        case RT_STARS_SQE_TYPE_COND:
            return "COND";
        default:
            return std::to_string(type);
    }
}
