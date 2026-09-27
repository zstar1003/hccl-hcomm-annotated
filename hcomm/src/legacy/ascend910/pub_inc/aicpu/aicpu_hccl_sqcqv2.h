/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_SQCQV2_H__
#define __AICPU_HCCL_SQCQV2_H__
#include <memory>
#include <vector>

#include "ascend_hal.h"
#include "log.h"
#include "aicpu_hccl_sqcq.h"
enum rtStarsSqeTypeV2 {
    RT_HW_STARS_SQE_TYPE_AICORE = 0,        // AICORE
    RT_HW_STARS_SQE_TYPE_AICPU = 1,         // AICPU
    RT_HW_STARS_SQE_TYPE_AIV = 1,           // AIV
    RT_HW_STARS_SQE_TYPE_PLACE_HOLDER = 3,  // PLACE_HOLDER
    RT_HW_STARS_SQE_TYPE_EVENT_RECORD = 4,  // EVENT_RECORD
    RT_HW_STARS_SQE_TYPE_EVENT_WAIT = 5,    // EVENT_WAIT
    RT_HW_STARS_SQE_TYPE_NOTIFY_RECORD = 6, // NOTIFY_RECORD
    RT_HW_STARS_SQE_TYPE_NOTIFY_WAIT = 7,   // NOTIFY_WAIT
    RT_HW_STARS_SQE_TYPE_WRITE_VALUE = 8,   // for EVENT_RESET task
    RT_HW_STARS_SQE_TYPE_SDMA = 9,          // SDMA
    RT_HW_STARS_SQE_TYPE_MAX = 10           // MAX
};

#pragma pack(push)
#pragma pack(1)
struct rtStarsSqeHeaderV2_t {
    uint16_t type : 6;
    uint16_t graph_lock : 1;
    uint16_t graph_unlock : 1;
    uint16_t ie : 1;
    uint16_t pre_p : 1;
    uint16_t post_p : 1;
    uint16_t wr_cqe : 1;
    uint16_t res0 : 2;
    uint16_t l2_lock : 1;
    uint16_t l2_unlock : 1;
    uint16_t res1; // num_blocks or res
    uint16_t rt_stream_id;
    uint16_t task_id;
};
// 内部替换为event的notify
struct rtStarsNotifySqeV2_t {
    rtStarsSqeHeaderV2_t header;

    uint32_t notify_id : 10;
    uint32_t res2 : 22;
    uint16_t res3;
    uint8_t kernel_credit;
    uint8_t res4;
    uint32_t res[12];
};

struct rtStarsWriteValueSqeV2_t {
    rtStarsSqeHeaderV2_t header;

    uint32_t reg_addr_low;
    uint16_t reg_addr_high;
    uint16_t awsize : 3;
    uint8_t snoop : 1;
    uint8_t res2 : 4;
    uint8_t awcache : 4;
    uint8_t awprot : 3;
    uint8_t VA : 1;
    uint32_t write_val[8];
    uint32_t res[4];
};

struct rtStarsMemcpyAsyncSqeV2_t {
    uint16_t type : 8;
    uint16_t ie : 1;
    uint16_t pre_p : 1;
    uint16_t post_p : 1;
    uint16_t wr_cqe : 1;
    uint16_t res0 : 2;
    uint16_t l2_lock : 1;
    uint16_t l2_unlock : 1;
    uint16_t res1;
    uint16_t rt_stream_id;
    uint16_t task_id;

    uint32_t res2;
    uint32_t res3 : 16;
    uint32_t kernel_credit : 8;
    uint32_t res4 : 8;

    uint32_t opcode : 8;
    uint32_t ie_dma : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t overflow_en : 1; // overflow使能标识下面携带了overflowAddr
    uint32_t res5 : 12;

    uint16_t src_streamid;
    uint16_t src_substreamid;
    uint16_t dst_streamid;
    uint16_t dst_substreamid;
    uint32_t length;

    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;
    uint32_t overflow_addr_low; // 填入overflowAddr
    uint32_t overflow_addr_high;
};

struct rtStarsEventSqeV2_t {
    uint16_t type : 8;
    uint16_t ie : 1;
    uint16_t pre_p : 1;
    uint16_t post_p : 1;
    uint16_t wr_cqe : 1;
    uint16_t res0 : 4;
    uint16_t flag; // event record flag
    uint16_t rt_stream_id;
    uint16_t task_id;
    uint16_t event_id : 10;
    uint32_t res2 : 22;
    uint16_t res3;
    uint8_t kernel_credit;
    uint8_t res4;
    uint32_t offset; // event record timeline offset
    uint32_t res6[12];
};
#pragma pack(pop)

extern void AddOneNotifyWaitSqeV2(
    uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType,
    const dfx::DfxTimeOutConfig& dfxTimeOutConfig);
extern void AddOneRecordSqeV2(uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType);
extern void AddOneWriteValueRecordSqeV2(
    uint16_t streamId, uint16_t taskId, u64 notifyWRAddr, const uint8_t* sqeIn, uint8_t* sqeType);
extern void AddOneMemcpySqeV2(
    uint16_t streamId, uint16_t taskId, const void* src, uint32_t length, const aclDataType runtimeDataType,
    aclrtReduceKind rtReduceOp, const void* dst, uint32_t partId, uint32_t ssid, uint32_t devId, u64 overflowAddr,
    uint8_t linkType, const uint8_t* sqeIn, uint8_t* sqeType, uint32_t hcclQos);
extern void AddOneEventResetSqeV2(
    uint16_t streamId, int32_t eventId, uint16_t taskId, int64_t phyChipId, int64_t phyDieId, u64 addr,
    const uint8_t* sqeIn, uint8_t* sqeType);
extern void
AddOneEventRecordSqeV2(uint16_t streamId, int32_t eventId, uint16_t taskId, const uint8_t* sqeIn, uint8_t* sqeType);
extern void
AddOneEventWaitSqeV2(uint16_t streamId, int32_t eventId, uint16_t taskId, const uint8_t* sqeIn, uint8_t* sqeType);
#endif // __AICPU_HCCL_SQCQV2_HPP__
