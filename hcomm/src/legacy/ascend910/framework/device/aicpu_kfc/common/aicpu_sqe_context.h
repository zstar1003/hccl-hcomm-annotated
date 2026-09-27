/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SQE_AICPU_CONTEXT_H__
#define __SQE_AICPU_CONTEXT_H__

#include "hccl_common.h"
#include "common/aicpu_kfc_def.h"
#include "common/sqe_context_utils.h"

constexpr uint32_t AC_SQE_MAX_CNT = 2048U; // 一次mc2算子一个流上最大下发sqe数量

struct SqeLocalRingBuffer {
    uint8_t localBuff[AC_SQE_SIZE * AC_SQE_MAX_CNT]; // local buffer
    uint8_t sqeType[AC_SQE_MAX_CNT];                 // 记录SQE类型,用于后续解析
    uint32_t addInfo[AC_SQE_MAX_CNT];                // 记录额外信息
    uint64_t profTimestap[AC_SQE_MAX_CNT]{0};        // profiling上报
    uint16_t tailSqeTaskId;                          // 最后一个sqe对应的taskId
    uint16_t tailSqeIdx;                             // 最后一个sqe对应的数组idx
    uint16_t sqeCnt;                                 // 当前轮保存的sqe数量(下发后重置)
    uint32_t sqHead;
    uint32_t sqTail;
    uint16_t filpNum;
};

struct SqeContext {
    SqeLocalRingBuffer* buffPtr; // SqeLocalRingBuffer[AC_MAX_RANK_NUM]
    int32_t clusterId;
};

class AicpuSqeContext {
public:
    static void InitSqeContext();
    static void SyncVariable();
    static void SaveVariable();
    static HcclResult
    GetNextSqeBufferAddr(uint32_t streamId, uint8_t*& sqeBufferAddr, uint8_t*& sqeTypeAddr, uint16_t& taskId);
    static HcclResult RecordAddInfo(uint32_t streamId, uint32_t addInfo);
    static HcclResult QuerySqeInfoByHead(uint32_t streamId, uint32_t sqHead, SqeInfo* info);
    static HcclResult QuerySqeInfoByTaskId(uint32_t streamId, uint16_t taskId, SqeInfo* info);
    static HcclResult ClearLocalBuff();
    static HcclResult ClearCurBuff(uint32_t streamid, uint32_t leftBound = 0);
    static HcclResult ModifyBuffer(uint32_t streamid);
    static std::string GetString(const SqeInfo& sqeInfo);
    static HcclResult AddFlipTask(uint32_t streamId);
};

extern SqeContext* GetSqeContext();

#endif // __MC2_AICPU_CONTEXT_HPP__
