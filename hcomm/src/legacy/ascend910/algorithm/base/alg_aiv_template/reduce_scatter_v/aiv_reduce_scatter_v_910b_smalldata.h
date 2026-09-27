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

class AivReduceScatterVSmall910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterVSmall910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, int32_t tag, ExtraArgs& extraArgs);
};

template <typename T>
__aicore__ inline void
AivReduceScatterVSmall910B::Process(GM_ADDR input, GM_ADDR output, int32_t tag, ExtraArgs& extraArgs)
{
    // 共用16个flag
    bool ifPingpong = (tag % 2 == 0);
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[blockIdx_] + dataOffset);
    __gm__ T* outputGM = (__gm__ T*)output;

    if (blockIdx_ != rank_) {
        GlobalTensor<T> cclGTOther;
        cclGTOther.SetGlobalBuffer(cclGMOther, extraArgs.sendCounts[rank_]);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, extraArgs.sendCounts[rank_]);

        CpGM2GM(
            cclGMSelf + extraArgs.sendDispls[blockIdx_], inputGM + extraArgs.sendDispls[blockIdx_],
            extraArgs.sendCounts[blockIdx_]);
        // 卡间同步
        pipe_barrier(PIPE_ALL);
        Record(tag, blockIdx_, AivNotifyType::DataSignal, 0, ifPingpong);
        Wait(tag, blockIdx_, AivNotifyType::DataSignal, 0, ifPingpong);
        pipe_barrier(PIPE_ALL);
        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, cclGTOther[extraArgs.sendDispls[rank_]], extraArgs.sendCounts[rank_]);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();

        WaitNv1(tag, rank_, AivNotifyType::DataSignal, 0, ifPingpong);
        pipe_barrier(PIPE_ALL);
        SetAtomicOp<T>(reduceOp_);
        DataCopyUB2GM(outputGT, localOut, extraArgs.sendCounts[rank_]);
        SetAtomicNone();

        inOutQue.FreeTensor(localOut);

    } else {
        CpGM2GM(outputGM, inputGM + extraArgs.sendDispls[rank_], extraArgs.sendCounts[rank_]);
        pipe_barrier(PIPE_ALL);
        // 卡内同步
        Record1vN(tag, CommPattern::intraRank, AivNotifyType::DataSignal, 0, ifPingpong);
    }
}

template <typename T>
__aicore__ inline void aiv_reduce_scatter_v_910b_smalldata(EXTERN_KERNEL_ARGS_DEF)
{
    AivReduceScatterVSmall910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process<T>(input, output, tag, extraArgs);
    op.TailCounter();
}
