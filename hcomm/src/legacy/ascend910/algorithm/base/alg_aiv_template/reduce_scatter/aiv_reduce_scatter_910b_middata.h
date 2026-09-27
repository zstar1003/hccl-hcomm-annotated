/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base.h"

using namespace AscendC;

class AivReduceScatterMid910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterMid910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void AivReduceScatterMid910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint64_t count = len;

    // 用16个flagsize
    bool ifPingpong = (tag % 2 == 0);
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T* inputGm = (__gm__ T*)input;
    __gm__ T* outputGm = (__gm__ T*)output;
    __gm__ T* cclGmSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
    __gm__ T* cclGmOther = (__gm__ T*)(GM_IN[blockIdx_] + dataOffset);
    if (blockIdx_ != rank_) {
        CpGM2GM(cclGmSelf + blockIdx_ * count, inputGm + blockIdx_ * count, count);
        // 卡内同步
        WaitNv1(tag, rank_, AivNotifyType::DataSignal, 0, ifPingpong);
        pipe_barrier(PIPE_ALL);
        Record(tag, blockIdx_, AivNotifyType::DataSignal, 0, ifPingpong);
        Wait(tag, blockIdx_, AivNotifyType::DataSignal, 0, ifPingpong);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(outputGm, cclGmOther + rank_ * count, count, true, reduceOp_);
    } else {
        CpGM2GM(outputGm, inputGm + rank_ * count, count);
        // 卡内同步
        pipe_barrier(PIPE_ALL);
        Record1vN(tag, CommPattern::intraRank, AivNotifyType::DataSignal, 0, ifPingpong);
    }
}

template <typename T>
__aicore__ inline void aiv_reduce_scatter_910b_middata(KERNEL_ARGS_DEF)
{
    AivReduceScatterMid910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag);
    op.TailCounter();
}
