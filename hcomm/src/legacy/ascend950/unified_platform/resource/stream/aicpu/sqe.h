/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_SQE_H
#define HCCLV2_SQE_H

#include <cstdint>

#include "enum_factory.h"
#include "hccl/base.h"

namespace Hccl {

constexpr u32 AC_SQE_SIZE = 64U;
constexpr u32 AC_SQE_MAX_CNT = 256U; // 一轮通信在一个流上最大下发sqe数量
constexpr u8 RT_STARS_DEFAULT_KERNEL_CREDIT = 254U;
constexpr u8 RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT = 255U;

constexpr u32 UINT32_BIT_NUM = 32U;
constexpr u32 MASK_32_BIT = 0xFFFFFFFFU;
constexpr u32 MASK_17_BIT = 0x0001FFFFU;

constexpr u64 LOW32_BIT_MASK = 0x00000000ffffffffU;
constexpr u64 HIGH32_BIT_MASK = 0xffffffff00000000U;

enum class RtStarsSqeType {
    RT_STARS_SQE_TYPE_FFTS = 0,          // FFTS
    RT_STARS_SQE_TYPE_AICPU = 1,         // AICPU
    RT_STARS_SQE_TYPE_PLACE_HOLDER = 3,  // PLACE_HOLDER
    RT_STARS_SQE_TYPE_EVENT_RECORD = 4,  // EVENT_RECORD
    RT_STARS_SQE_TYPE_EVENT_WAIT = 5,    // EVENT_WAIT
    RT_STARS_SQE_TYPE_NOTIFY_RECORD = 6, // NOTIFY_RECORD
    RT_STARS_SQE_TYPE_NOTIFY_WAIT = 7,   // NOTIFY_WAIT
    RT_STARS_SQE_TYPE_WRITE_VALUE = 8,   // for EVENT_RESET task
    RT_STARS_SQE_TYPE_SDMA = 11,         // SDMA
    RT_STARS_SQE_TYPE_VPC = 12,          // VPC
    RT_STARS_SQE_TYPE_JPEGE = 13,        // JPEGE
    RT_STARS_SQE_TYPE_JPEGD = 14,        // JPEGD
    RT_STARS_SQE_TYPE_DSA = 15,
    RT_STARS_SQE_TYPE_ROCCE = 16,    // RoCCE
    RT_STARS_SQE_TYPE_PCIE_DMA = 17, // PCIE_DMA
    RT_STARS_SQE_TYPE_RESV = 18,     // reserve
    RT_STARS_SQE_TYPE_CDQM = 19,     // CDQM
    RT_STARS_SQE_TYPE_COND = 20,     // condition
    RT_STARS_SQE_TYPE_END = 21,
    RT_STARS_SQE_TYPE_VIR_TYPE = 0xFF // DVPP virtual SQE TYPE
};

MAKE_ENUM(
    RtStarsWriteValueSizeType, RT_STARS_WRITE_VALUE_SIZE_TYPE_8BIT, RT_STARS_WRITE_VALUE_SIZE_TYPE_16BIT,
    RT_STARS_WRITE_VALUE_SIZE_TYPE_32BIT, RT_STARS_WRITE_VALUE_SIZE_TYPE_64BIT, RT_STARS_WRITE_VALUE_SIZE_TYPE_128BIT,
    RT_STARS_WRITE_VALUE_SIZE_TYPE_256BIT)

MAKE_ENUM(
    RtStarsWriteValueSubType, RT_STARS_WRITE_VALUE_SUB_TYPE_DEFAULT, RT_STARS_WRITE_VALUE_SUB_TYPE_EVENT_RESET,
    RT_STARS_WRITE_VALUE_SUB_TYPE_RDMA_DB_SEND, RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_NO_PCIE,
    RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_PCIE, RT_STARS_WRITE_VALUE_SUB_TYPE_MAX)

enum class RtReduceKind {
    RT_MEMCPY_SDMA_AUTOMATIC_ADD = 10, // D2D, SDMA inline reduce, include 1P, and P2P
    RT_MEMCPY_SDMA_AUTOMATIC_MAX = 11,
    RT_MEMCPY_SDMA_AUTOMATIC_MIN = 12,
    RT_MEMCPY_SDMA_AUTOMATIC_EQUAL = 13,
    RT_REDUCE_KIND_END
};

enum class RtDataType {
    RT_DATA_TYPE_FP32 = 0,    // fp32
    RT_DATA_TYPE_FP16 = 1,    // fp16
    RT_DATA_TYPE_INT16 = 2,   // int16
    RT_DATA_TYPE_INT4 = 3,    // int4
    RT_DATA_TYPE_INT8 = 4,    // int8
    RT_DATA_TYPE_INT32 = 5,   // int32
    RT_DATA_TYPE_BFP16 = 6,   // bfp16
    RT_DATA_TYPE_BFP32 = 7,   // bfp32
    RT_DATA_TYPE_UINT8 = 8,   // uint8
    RT_DATA_TYPE_UINT16 = 9,  // uint16
    RT_DATA_TYPE_UINT32 = 10, // uint32
    RT_DATA_TYPE_END
};

enum class RtStarsMemcpyAsyncDataType {
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT8 = 0x00,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT16 = 0x10,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT32 = 0x20,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_UINT8 = 0x30,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_UINT16 = 0x40,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_UINT32 = 0x50,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP16 = 0x60,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP32 = 0x70,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_BFP16 = 0x80,
    RT_STARS_MEMCPY_ASYNC_OP_RESERVED = 0xf0
};

enum class RtStarsMemcpyAsyncOperationKind {
    RT_STARS_MEMCPY_ASYNC_OP_KIND_CPY = 0x00,
    RT_STARS_MEMCPY_ASYNC_OP_KIND_ADD = 0x01,
    RT_STARS_MEMCPY_ASYNC_OP_KIND_MAX = 0x02,
    RT_STARS_MEMCPY_ASYNC_OP_KIND_MIN = 0x03,
    RT_STARS_MEMCPY_ASYNC_OP_KIND_EQUAL = 0x04
};

struct RtStarsSqeHeader {
    uint8_t type : 6;
    uint8_t l1Lock : 1;
    uint8_t l1Unlock : 1;

    uint8_t ie : 2;
    uint8_t preP : 2;
    uint8_t postP : 2;
    uint8_t wrCqe : 1;
    uint8_t reserved : 1;

    uint16_t numBlocks; // numBlocks or res

    uint16_t rtStreamId;
    uint16_t taskId;
};

struct RtStarsWriteValueSqe {
    RtStarsSqeHeader header;

    uint32_t res3;

    uint32_t res4 : 16;
    uint32_t kernelCredit : 8;
    uint32_t res5 : 8;

    uint32_t writeAddrLow;

    uint32_t writeAddrHigh : 17;
    uint32_t res6 : 3;
    uint32_t awsize : 3;
    uint32_t snoop : 1;
    uint32_t awcache : 4;
    uint32_t awprot : 3;
    uint32_t va : 1; // 1 /* 1: virtual address; 0: phy addr */

    uint32_t res7; // eventId for event reset task
    uint32_t subType;

    uint32_t writeValuePart0;
    uint32_t writeValuePart1;
    uint32_t writeValuePart2;
    uint32_t writeValuePart3;
    uint32_t writeValuePart4;
    uint32_t writeValuePart5;
    uint32_t writeValuePart6;
    uint32_t writeValuePart7;
};

struct RtStarsMemcpyAsyncSqe {
    RtStarsSqeHeader header;

    uint32_t res3;
    /* *******12 bytes********* */

    uint16_t res4;
    uint8_t kernelCredit;
    uint8_t ptrMode : 1;
    uint8_t res5 : 7;
    /* *******16 bytes********* */

    uint32_t opcode : 8;
    uint32_t ie2 : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t res6 : 4;
    /* *******20 bytes********* */

    uint16_t srcStreamid;
    uint16_t srcSubStreamid;
    uint16_t dstStreamid;
    uint16_t dstSubStreamid;
    /* *******28 bytes********* */

    uint32_t length;
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;

    uint32_t resLast[4];
};

struct RtStarsNotifySqe {
    RtStarsSqeHeader header;

    uint32_t camelBack : 13;
    uint32_t res2 : 19;

    uint16_t res3;
    uint8_t kernelCredit;
    uint8_t res4;
    uint32_t timeout;
    uint32_t res5[11];
};
} // namespace Hccl
#endif
