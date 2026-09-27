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

class AivAllReduceSmall910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceSmall910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void AivAllReduceSmall910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint64_t count = len;

    // 用4个flag
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;
    bool ifPingpong = (tag % 2 == 0);

    if (blockIdx_ == rank_) {
        __gm__ T* inputGM = (__gm__ T*)input;
        __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
        __gm__ T* outputGM = (__gm__ T*)output;

        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM, count);
        GlobalTensor<T> cclGTSelf;
        cclGTSelf.SetGlobalBuffer(cclGMSelf, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, count);

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT, count);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(cclGTSelf, localOut, count);

        PipeBarrier<PIPE_MTE3>();

        // 卡间同步
        Record1vN(tag, CommPattern::interRank, AivNotifyType::DataSignal, 0, ifPingpong);

        DataCopyUB2GM(outputGT, localOut, count);
        inOutQue.FreeTensor(localOut);

        PipeBarrier<PIPE_MTE3>();

        // 卡内同步
        Record1vN(tag, CommPattern::intraRank, AivNotifyType::DataSignal, 0, ifPingpong);
    } else {
        __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[blockIdx_] + dataOffset);
        __gm__ T* outputGM = (__gm__ T*)output;

        GlobalTensor<T> cclGTOther;
        cclGTOther.SetGlobalBuffer(cclGMOther, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, count);

        // 卡间同步
        WaitNv1(tag, blockIdx_, AivNotifyType::DataSignal, 0, ifPingpong);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, cclGTOther, count);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();

        // 卡内同步
        WaitNv1(tag, rank_, AivNotifyType::DataSignal, 0, ifPingpong);
        PipeBarrier<PIPE_ALL>();

        SetAtomicOp<T>(reduceOp_);
        DataCopyUB2GM(outputGT, localOut, count);
        SetAtomicNone();

        inOutQue.FreeTensor(localOut);
    }
}

template <typename T>
__aicore__ inline void aiv_all_reduce_910b_smalldata(KERNEL_ARGS_DEF)
{
    AivAllReduceSmall910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag);
    op.TailCounter();
}
