/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SQE_A5_H
#define SQE_A5_H

#include <cstdint>

// 当 legacy sqe_v82.h 已被包含时，复用其定义避免重定义冲突
#ifndef HCCLV2_SQE_V82_H

namespace Hccl {

enum class Rt91095StarsSqeType : uint8_t {
    RT_91095_SQE_TYPE_NOTIFY_RECORD = 6,
    RT_91095_SQE_TYPE_NOTIFY_WAIT = 7,
    RT_91095_SQE_TYPE_WRITE_VALUE = 8,
    RT_91095_SQE_TYPE_UBDMA = 9,
    RT_91095_SQE_TYPE_SDMA = 11,
};

struct Rt91095StarsSqeHeader {
    uint8_t type : 6;
    uint8_t lock : 1;
    uint8_t unlock : 1;
    uint8_t ie : 1;
    uint8_t preP : 1;
    uint8_t postP : 1;
    uint8_t wrCqe : 1;
    uint8_t ptrMode : 1;
    uint8_t rttMode : 1;
    uint8_t headUpdate : 1;
    uint8_t sqeProf : 1; // SQE profiling标记(未开启L1时为0)；A5不启用，恒为0
    uint16_t numBlocks;
    uint16_t rtStreamId;
    uint16_t taskId;
};

struct RtMemcpyStride00 {
    uint16_t dstStreamId;
    uint16_t dstSubStreamId;
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;
    uint32_t lengthMove;
    uint32_t srcOffsetLow;
    uint32_t dstOffsetLow;
    uint16_t srcOffsetHigh;
    uint16_t dstOffsetHigh;
};

struct Rt91095StarsMemcpySqe {
    Rt91095StarsSqeHeader header;
    uint32_t res1;
    uint16_t res2;
    uint8_t kernelCredit;
    uint8_t res3;
    uint32_t opcode : 8;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t stride : 2;
    uint32_t ie2 : 1;
    uint32_t compEn : 1;
    uint32_t res4 : 14;
    uint16_t sqeId;
    uint8_t mapamPartId;
    uint8_t mpamns : 1;
    uint8_t pmg : 2;
    uint8_t qos : 4;
    uint8_t d2dOffsetFlag : 1;
    uint16_t srcStreamId;
    uint16_t srcSubStreamId;
    union {
        RtMemcpyStride00 strideMode0;
    } u;
};

struct Rt91095StarsNotifySqe {
    Rt91095StarsSqeHeader header;
    uint32_t notifyId : 17;
    uint32_t res2 : 13;
    uint32_t cntFlag : 1;
    uint32_t clrFlag : 1;
    uint16_t subType;
    uint8_t kernelCredit;
    uint8_t res4 : 5;
    uint8_t sqeLength : 3;
    uint32_t cntValue;
    uint32_t waitModeBit : 2;
    uint32_t recordModeBit : 3;
    uint32_t bitmap : 1;
    uint32_t res5 : 26;
    uint32_t timeout;
    uint32_t exeResult;
    uint32_t res7[8];
};

struct Rt91095StarsWriteValueSqe {
    Rt91095StarsSqeHeader header;
    uint32_t res1;
    uint16_t res2;
    uint8_t kernelCredit;
    uint8_t res3;
    uint32_t writeAddrLow;
    uint32_t writeAddrHigh : 17;
    uint32_t res4 : 3;
    uint32_t awsize : 3;
    uint32_t snoop : 1;
    uint32_t awcache : 4;
    uint32_t awprot : 3;
    uint32_t va : 1;
    uint32_t res5;
    uint32_t subType;
    uint32_t writeValuePart[8];
};

} // namespace Hccl

#endif // HCCLV2_SQE_V82_H

#endif // SQE_A5_H
