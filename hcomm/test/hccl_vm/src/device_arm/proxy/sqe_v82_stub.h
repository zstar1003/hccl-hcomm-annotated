/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SQE_V85_STUB_H
#define SQE_V85_STUB_H
#include <cstdint>

enum class Rt91095StarsSqeType {
    RT_91095_SQE_TYPE_AIC = 0,           // AIC
    RT_91095_SQE_TYPE_AIV = 1,           // AIV
    RT_91095_SQE_TYPE_FUSION = 2,        // FUSION
    RT_91095_SQE_TYPE_PLACE_HOLDER = 3,  // PLACE_HOLDER
    RT_91095_SQE_TYPE_AICPU_H = 4,       // AICPU_H
    RT_91095_SQE_TYPE_AICPU_D = 5,       // AICPU_D
    RT_91095_SQE_TYPE_NOTIFY_RECORD = 6, // NOTIFY_RECORD
    RT_91095_SQE_TYPE_NOTIFY_WAIT = 7,   // NOTIFY_WAIT
    RT_91095_SQE_TYPE_WRITE_VALUE = 8,   // WRITE_VALUE
    RT_91095_SQE_TYPE_UBDMA = 9,         // UBDMA
    RT_91095_SQE_TYPE_ASYNCDMA = 10,     // ASYNCDMA
    RT_91095_SQE_TYPE_SDMA = 11,         // SDMA
    RT_91095_SQE_TYPE_VPC = 12,          // VPC
    RT_91095_SQE_TYPE_JPEGE = 13,        // JPEGE
    RT_91095_SQE_TYPE_JPEGD = 14,        // JPEGD
    RT_91095_SQE_TYPE_CMO = 15,          // CMO
    RT_91095_SQE_TYPE_COND = 20,         // condition
    RT_91095_SQE_TYPE_END = 21,
};

struct Rt91095StarsSqeHeader {
    /* word0 */
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
    uint8_t reserved : 1;
    uint16_t numBlocks;

    /* word1 */
    uint16_t rtStreamId;
    uint16_t taskId;
};

struct RtMemcpyStride00 {
    /* word7 */
    uint16_t dstStreamId;
    uint16_t dstSubStreamId;

    /* word8-9 */
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;

    /* word10-11 */
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;

    /* word12 */
    uint32_t lengthMove;

    /* word13-15 */
    uint32_t srcOffsetLow;
    uint32_t dstOffsetLow;
    uint16_t srcOffsetHigh;
    uint16_t dstOffsetHigh;
};

struct RtMemcpyStride01 {
    /* word7 */
    uint16_t dstStreamId;
    uint16_t dstSubStreamId;

    /* word8-9 */
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;

    /* word10-11 */
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;

    /* word12 */
    uint32_t lengthMove;

    /* word13-15 */
    uint32_t srcStrideLength;
    uint32_t dstStrideLength;
    uint32_t strideNum;
};

struct RtMemcpyStride10 {
    /* word7 */
    uint16_t numOuter;
    uint16_t numInner;

    /* word8-9 */
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;

    /* word10-11 */
    uint32_t strideOuter;
    uint32_t strideInner;

    /* word12 */
    uint32_t lengthInner;

    /* word13-15 */
    uint32_t reserved[3];
};

struct Rt91095StarsMemcpySqe {
    /* word0-1 */
    Rt91095StarsSqeHeader header;

    /* word2 */
    uint32_t res1;

    /* word3 */
    uint16_t res2;
    uint8_t kernelCredit;
    uint8_t res3;

    /* word4 */
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

    /* word5 */
    uint16_t sqeId;
    uint8_t mapamPartId;
    uint8_t mpamns : 1;
    uint8_t pmg : 2;
    uint8_t qos : 4;
    uint8_t d2dOffsetFlag : 1; // use reserved filed

    /* word6 */
    uint16_t srcStreamId;
    uint16_t srcSubStreamId;

    /* word7-15 */
    union {
        RtMemcpyStride00 strideMode0;
        RtMemcpyStride01 strideMode1;
        RtMemcpyStride10 strideMode2;
    } u;
};

struct Rt91095StarsNotifySqe {
    /* word0-1 */
    Rt91095StarsSqeHeader header;

    /* word2 */
    uint32_t notifyId : 17;
    uint32_t res2 : 13;
    uint32_t cntFlag : 1;
    uint32_t clrFlag : 1;

    /* word3 */
    uint16_t subType; // This field is reserved and used by software.
    uint8_t kernelCredit;
    uint8_t res4 : 5;
    uint8_t sqeLength : 3;

    /* word4 */
    uint32_t cntValue;

    /* word5 */
    uint32_t waitModeBit : 2;   // bit 0:equal, bit 1:bigger
    uint32_t recordModeBit : 3; // bit 0:add, bit 1:write, bit 2:clear
    uint32_t bitmap : 1;        // only use for conut notify wait, 1 means comapre with count value by bit
    uint32_t res5 : 26;

    /* word6 */
    uint32_t timeout; // This field is reserved and used by software.

    /* word7 */
    uint32_t exeResult; // for Two-phase operator

    /* word8-15 */
    uint32_t res7[8];
};

struct Rt91095StarsUbdmaDBmodeSqe {
    /* word0-1 */
    Rt91095StarsSqeHeader header;

    /* word2 */
    uint16_t mode : 1;
    uint16_t doorbellNum : 2;
    uint16_t res0 : 13;
    uint16_t res1;

    /* word3 */
    uint16_t res2;
    uint8_t kernelCredit;
    uint8_t res3 : 5;
    uint8_t sqeLength : 3;

    /* word4 */
    uint32_t jettyId1 : 16;
    uint32_t res4 : 9;
    uint32_t funcId1 : 7;

    /* word5 */
    uint16_t piValue1;
    uint16_t res5 : 15;
    uint16_t dieId1 : 1;

    /* word6 */
    uint32_t jettyId2 : 16;
    uint32_t res6 : 9;
    uint32_t funcId2 : 7;

    /* word7 */
    uint16_t piValue2;
    uint16_t res7 : 15;
    uint16_t dieId2 : 1;

    /* word8-15 */
    uint32_t res8[8];
};

#endif
