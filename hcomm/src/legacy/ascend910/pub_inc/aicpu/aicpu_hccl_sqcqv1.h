/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_SQCQV1_H__
#define __AICPU_HCCL_SQCQV1_H__

#include <memory>
#include <vector>

#include "ascend_hal.h"
#include "log.h"
#include "aicpu_hccl_sqcq.h"

#pragma pack(push)
#pragma pack(1)
struct rtStarsNotifySqeV1_t {
    rtStarsSqeHeader_t header;

    uint32_t notify_id : 13;
    uint32_t res2 : 19;

    uint16_t res3;
    uint8_t kernel_credit;
    uint8_t res4;
    uint32_t timeout;
    uint32_t res5[11];
};
#pragma pack(pop)

extern void TranslateOpcode(uint8_t opCode, uint8_t& reduceType);

extern void AddOneNotifyWaitSqeV1(
    uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType,
    const dfx::DfxTimeOutConfig& dfxTimeOutConfig);

extern void AddOneRecordSqeV1(uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType);

extern void AddOneWriteValueRecordSqeV1(
    uint16_t streamId, uint16_t taskId, u64 notifyWRAddr, const uint8_t* sqeIn, uint8_t* sqeType);

extern void AddOneMemcpySqeV1(
    uint16_t streamId, uint16_t taskId, const void* src, uint32_t length, const aclDataType runtimeDataType,
    aclrtReduceKind rtReduceOp, const void* dst, uint32_t partId, uint32_t ssid, uint32_t devId, u64 overflowAddr,
    uint8_t linkType, const uint8_t* sqeIn, uint8_t* sqeType, uint32_t hcclQos);

extern void AddOneRdmaDbSendSqeV1(
    uint16_t streamId, uint16_t taskId, uint64_t dbInfo, uint64_t dbAddr, uint32_t length, uint8_t rdmaType,
    const uint8_t* sqeIn, uint8_t* sqeType);

extern void AddOneEventResetSqeV1(
    uint16_t streamId, int32_t eventId, uint16_t taskId, int64_t phyChipId, int64_t phyDieId, u64 eventAddr,
    const uint8_t* sqeIn, uint8_t* sqeType);

extern void
AddOneEventRecordSqeV1(uint16_t streamId, int32_t eventId, uint16_t taskId, const uint8_t* sqeIn, uint8_t* sqeType);

extern void
AddOneEventWaitSqeV1(uint16_t streamId, int32_t eventId, uint16_t taskId, const uint8_t* sqeIn, uint8_t* sqeType);

extern void AddOneFlipPlaceHolderSqeV1(
    uint16_t streamId, uint16_t flipNum, uint16_t taskId, const uint8_t* sqeIn, uint8_t* sqeType);

extern void AddOneCacheMemcpyPlaceHolderSqeV1(
    uint16_t streamId, uint16_t taskId, const void* src, const void* dst, uint8_t linkType, const uint8_t* sqeIn,
    uint8_t* sqeType, uint32_t hcclQos);

extern void AddOneCacheNotifyWaitPlaceholderSqeV1(
    uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType,
    const dfx::DfxTimeOutConfig& dfxTimeOutConfig);

extern void AddOneCacheNotifyRecordPlaceholderSqeV1(
    uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType);

extern void AddOneCacheWriteValuePlaceholderSqeV1(
    uint16_t streamId, uint16_t taskId, u64 notifyWRAddr, const uint8_t* sqeIn, uint8_t* sqeType);

extern void AddOneCacheMemcpyRecordPlaceholderSqeV1(
    uint16_t streamId, uint16_t taskId, const void* src, uint32_t length, const aclDataType runtimeDataType,
    aclrtReduceKind rtReduceOp, const void* dst, uint32_t partId, uint32_t ssid, uint32_t devId, u64 overflowAddr,
    uint8_t linkType, const uint8_t* sqeIn, uint8_t* sqeType, uint32_t hcclQos);

extern void SetCachePlaceholderHeaderV1(uint16_t streamId, uint16_t taskId, const uint8_t* sqeIn);

// 分别返回硬件和软件的超时时长，二者一般只有一个值是有效的
std::pair<uint64_t, uint64_t> GetTimeOutValue(const dfx::DfxTimeOutConfig& dfxTimeOutConfig);
#endif // __AICPU_HCCL_SQCQV1_HPP__
