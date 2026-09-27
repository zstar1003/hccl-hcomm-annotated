/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SQE_AICPU_CONTEXT_UTILS_H__
#define __SQE_AICPU_CONTEXT_UTILS_H__

#include "common/aicpu_hccl_def.h"

struct SqeInfo {
    uint32_t addr1High = 0; // write value addr/sdma src addr/cond op addr
    uint32_t addr1Low = 0;
    uint32_t addr2High = 0; // sdma dst addr/cond op value addr
    uint32_t addr2Low = 0;
    uint32_t sqeHeadIdx = 0; // 当前sqe或当前sqe组的起始idx
    uint32_t notifyId = 0;
    uint32_t length = 0; // sdma length
    uint32_t partId = 0; // sdma
    uint32_t remoteRank = 0;
    uint32_t dataType = 0;
    uint16_t streamId = 0;
    uint16_t eventId = 0;
    uint16_t taskId = 0;
    uint16_t condValue = 0; // cond op value
    uint8_t isLast = 0;     // cond op isLast
    uint8_t opCode = 0;     // sdma 类型
    uint8_t sqeNum = 0;     // 当前sqe组长度 默认0表示单个sqe
    uint8_t type = 0;
    uint8_t subType = 0;
    uint8_t valid = 0; // 是否有效标记位
    union TaskRelatedType {
        uint8_t rdmaType; // rdma类型 是payload还是notify
        uint8_t linkType; // 链路类型 SIO
    } taskRelated;
    uint8_t reverse[9] = {0};
}; // 64B

class SqeContextUtils {
public:
    static std::string RtsqTaskTypeToStr(uint8_t type);
    static HcclResult QuerySqeInfo(const uint8_t* sqeLocal, uint8_t sqeType, uint32_t addInfo, SqeInfo* info);
};

#endif // __SQE_AICPU_CONTEXT_UTILS_H__
