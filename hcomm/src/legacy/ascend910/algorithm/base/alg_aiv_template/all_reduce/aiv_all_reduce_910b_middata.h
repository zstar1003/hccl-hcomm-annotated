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

class AivAllReduceMid910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceMid910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void AivAllReduceMid910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t padCount = 32 / sizeof(T);
    uint64_t avgLengthPerRank = CeilDiv(len, rankSize_);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerRank, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;
    bool ifPingpong = (tag % 2 == 0);

    uint64_t count = 0;

    // 用9个flag
    __gm__ T* inputGm = (__gm__ T*)input;
    __gm__ T* outputGm = (__gm__ T*)output;
    uint32_t dataOffset = (tag % 2 == 0) ? 0 : AIV_PING_PONG_SIZE;
    __gm__ T* cclGmSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
    __gm__ T* cclGmOther = (__gm__ T*)(GM_IN[blockIdx_] + dataOffset);

    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t clearOffset = multiOffset + DOUBLE * DOUBLE * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE
                          + DOUBLE * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE
                          + (NUM_BLOCKS_FOUR_PER_RANK_A3)*ATOMIC_FLAG_SIZE;

    if (blockIdx_ == rank_) {
        SetSignalValue((__gm__ int32_t*)(GM_OUT[rank_] + OffSet + clearOffset), localSetTensor, 0);
        PipeBarrier<PIPE_ALL>();
    }

    // LocalCopy
    uint64_t gmOffset = blockIdx_ * avgLengthPerSlice;

    count = CalActualCount(blockIdx_, sliceCount, avgLengthPerSlice, tailLength);
    CpGM2GM(cclGmSelf + gmOffset, inputGm + gmOffset, count);
    PipeBarrier<PIPE_ALL>();

    if (blockIdx_ == rank_) {
        Record1vN(tag, CommPattern::intraRank, AivNotifyType::DataSignal, 0, ifPingpong);
    } else {
        Record(tag, blockIdx_, AivNotifyType::DataSignal, 0, ifPingpong);
        WaitNv1(tag, rank_, AivNotifyType::DataSignal, 0, ifPingpong);
        Wait(tag, blockIdx_, AivNotifyType::DataSignal, 0, ifPingpong);

        count = CalActualCount(rank_, sliceCount, avgLengthPerSlice, tailLength);

        uint64_t gmOffset = rank_ * avgLengthPerSlice;

        PipeBarrier<PIPE_ALL>();
        CpGM2GM(cclGmSelf + gmOffset, cclGmOther + gmOffset, count, true, reduceOp_);

        PipeBarrier<PIPE_MTE3>();

        // 本aiv reduce完成
        RecordNv1(tag, rank_, AivNotifyType::DataSignal, 0, ifPingpong);
    }

    // 每个aiv读相应对端的flag
    WaitSignalValue(
        (__gm__ int32_t*)(GM_OUT[blockIdx_] + OffSet + clearOffset), localCheckTensor, (rankSize_ - 1) * tag);

    // AllGather
    gmOffset = blockIdx_ * avgLengthPerSlice;
    count = CalActualCount(blockIdx_, sliceCount, avgLengthPerSlice, tailLength);

    PipeBarrier<PIPE_ALL>();
    CpGM2GM(outputGm + gmOffset, cclGmOther + gmOffset, count);

    return;
}

template <typename T>
__aicore__ inline void aiv_all_reduce_910b_middata(KERNEL_ARGS_DEF)
{
    AivAllReduceMid910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag);
    op.TailCounter();
}
