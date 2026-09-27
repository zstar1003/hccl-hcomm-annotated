/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_C_BASE_STUB_H
#define ASCEND_C_BASE_STUB_H

#include <cstdint>
#include <string>

typedef unsigned char __uint8_t__;
typedef __uint8_t__ uint8_t;

#define __global__
#define __aicore__
#define __gm__
#define __restrict__
#define GM_ADDR __gm__ uint8_t* __restrict__
#define __inout_pipe__(pipe)

#define Std std

#ifdef assert
#undef assert
#endif

namespace AscendC {
extern int64_t block_num;
extern thread_local int64_t block_idx;

extern __aicore__ inline int64_t GetBlockIdx() { return block_idx; }
extern __aicore__ inline int64_t GetBlockNum() { return block_num; }

using half = int16_t;
using bfloat16_t = int16_t;
using fp8_e4m3fn_t = int8_t;
using fp8_e5m2_t = int8_t;
using fp8_e8m0_t = int8_t;
using hifloat8_t = int8_t;

typedef enum {
    PIPE_S = 0,    // Scalar Pipe
    PIPE_MTE2 = 1, // MTE2 Pipe: GM->UB
    PIPE_MTE3 = 2, // MTE3 Pipe: UB->GM
    PIPE_ALL,
} pipe_t;

std::string GetPipeName(pipe_t pipe);

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

using TEventID = int8_t;

enum class TPosition : uint8_t {
    GM,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    CO1,
    CO2,
    VECIN,
    VECOUT,
    VECCALC,
    LCM = VECCALC,
    SPM,
    SHM = SPM,
    TSCM,
    C2PIPE2GM,
    C2PIPE2LOCAL,
    MAX,
};

using QuePosition = TPosition;
} // namespace AscendC

#endif // ASCEND_C_BASE_STUB_H
