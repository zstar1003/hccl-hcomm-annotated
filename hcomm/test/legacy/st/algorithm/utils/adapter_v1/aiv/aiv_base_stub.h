/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_BASE_STUB_H
#define AIV_BASE_STUB_H
#include <stdint.h>

#ifdef CHECKER_V1
#include "stream_pub.h"
#endif

typedef unsigned char __uint8_t__;
typedef __uint8_t__ uint8_t;

#define __global__
#define __aicore__
#define __gm__
#define __restrict__
#define GM_ADDR __gm__ uint8_t* __restrict__
#define __inout_pipe__(pipe)

#ifdef assert
#undef assert
#endif

namespace AscendC {

extern int64_t block_idx;
extern int64_t block_num;

using half = int16_t;
#ifndef __ARM_NEON
using bfloat16_t = int16_t;
#endif

const uint8_t DEFAULT_DATA_COPY_NBURST = 1;
const uint8_t DEFAULT_DATA_COPY_STRIDE = 0;
const uint32_t ONE_CORE_DUMP_SIZE = 1024;

typedef enum {
    PIPE_S = 0,    // Scalar Pipe
    PIPE_MTE2 = 1, // OUT ->{L1, L0{A,B}, UB}
    PIPE_MTE3 = 2, // UB ->{OUT,L1}
    PIPE_ALL,
} pipe_t;

enum class HardEvent : uint8_t {
    // src_dst
    MTE2_MTE1,
    MTE1_MTE2,
    MTE1_M,
    M_MTE1,
    MTE2_V,
    V_MTE2,
    MTE3_V,
    V_MTE3,
    M_V,
    V_M,
    V_V,
    MTE3_MTE1,
    MTE1_MTE3,
    MTE1_V,
    MTE2_M,
    M_MTE2,
    V_MTE1,
    M_FIX,
    FIX_M,
    MTE3_MTE2,
    MTE2_MTE3,
    S_V,
    V_S,
    S_MTE2,
    MTE2_S,
    S_MTE3,
    MTE3_S,
    MTE2_FIX,
    FIX_MTE2,
    FIX_S,
    M_S,
    FIX_MTE3,
    MTE1_FIX,
    FIX_MTE1,
    FIX_FIX,
    MAX,
};

enum class DumpType : uint8_t {
    DUMP_DEFAULT = 0,
    DUMP_SCALAR,
    DUMP_TENSOR,
    DUMP_SHAPE,
    DUMP_ASSERT,
    DUMP_META,
    DUMP_TIME_STAMP,
    DUMP_SIMT,
};

extern __aicore__ inline int64_t GetBlockIdx() { return block_idx; };
extern __aicore__ inline int64_t GetBlockNum() { return block_num; };
extern __aicore__ inline void InitDump(bool dump, uint8_t* dumpAddr, u32 dumpSize) { return; };
extern __aicore__ inline void PRINTF(const char* __restrict format, ...)
{
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
};
extern __aicore__ inline void PrintfImpl(DumpType dumpType, const char* __restrict format, ...)
{
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
};
extern __aicore__ inline void trap() { return; };
} // namespace AscendC

#endif
