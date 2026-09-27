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

class AivReduceScatterDeterSmall910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterDeterSmall910B() {}

    __aicore__ inline int64_t GetDeterministicRankOffset(int64_t x);

    template <typename T>
    __aicore__ inline void
    SumByPairs(int64_t x, int64_t count, int32_t tag, __gm__ T* cclGMSelf, int64_t flagOffset2st);

    template <typename T>
    __aicore__ inline void
    GatherReduce(int64_t x, int64_t count, int32_t tag, __gm__ T* cclGMSelf, int64_t flagOffset2st);

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize);

    template <typename T>
    __aicore__ inline void ProcessProxy(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize);

    template <typename T>
    __aicore__ inline void
    ProcessSingleRanksizeCore(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize);
};

__aicore__ inline int64_t AivReduceScatterDeterSmall910B::GetDeterministicRankOffset(int64_t x)
{
    int64_t tmp = 1;
    while (!(x & 1)) {
        x >>= 1;
        tmp <<= 1;
    }
    return tmp;
}

template <typename T>
__aicore__ inline void AivReduceScatterDeterSmall910B::SumByPairs(
    int64_t x, int64_t count, int32_t tag, __gm__ T* cclGMSelf, int64_t flagOffset2st)
{
    SetSignalValue((__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2st), localSetTensor, tag);

    if (x != 0) {
        int64_t multiple = GetDeterministicRankOffset(x);
        int64_t target = x - multiple;

        if (x & 1) {
            WaitSignalValue(
                (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2st - multiple * FLAG_SIZE), localCheckTensor, tag);
            CpGM2GM<T>(cclGMSelf + target * count, cclGMSelf + x * count, count, true, reduceOp_);
            PipeBarrier<PIPE_ALL>();
            SetSignalValue(
                (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2st + (rankSize_ * multiple) * FLAG_SIZE), localSetTensor,
                tag);
        } else {
            int64_t OffsetACK = rankSize_ * (multiple / DOUBLE) * FLAG_SIZE;
            WaitSignalValue(
                (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2st + OffsetACK - (multiple / DOUBLE) * FLAG_SIZE),
                localCheckTensor, tag);
            int64_t multipleTemp = multiple;
            while (x + multipleTemp / DOUBLE >= rankSize_) {
                multipleTemp /= DOUBLE;
            }
            if (multipleTemp > 1) {
                int64_t OffsetACKX = rankSize_ * (multipleTemp / DOUBLE) * FLAG_SIZE;
                WaitSignalValue(
                    (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2st + OffsetACKX + (multipleTemp / DOUBLE) * FLAG_SIZE),
                    localCheckTensor, tag);
            }
            CpGM2GM<T>(cclGMSelf + target * count, cclGMSelf + x * count, count, true, reduceOp_);
            PipeBarrier<PIPE_ALL>();
            SetSignalValue(
                (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2st + rankSize_ * multiple * FLAG_SIZE), localSetTensor,
                tag);
        }
    }
}

template <typename T>
__aicore__ inline void AivReduceScatterDeterSmall910B::GatherReduce(
    int64_t x, int64_t count, int32_t tag, __gm__ T* cclGMSelf, int64_t flagOffset2st)
{
    if (rankSize_ >= DETERMINISTIC_RANKSIZE) {
        SumByPairs(x, count, tag, cclGMSelf, flagOffset2st);
    } else {
        if (x == 0) {
            SetSignalValue((__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2st), localSetTensor, tag);
        } else {
            // 等待前一个核reduce完成
            WaitSignalValue((__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2st - FLAG_SIZE), localCheckTensor, tag);

            CpGM2GM(cclGMSelf, cclGMSelf + x * count, count, true, reduceOp_);
            PipeBarrier<PIPE_ALL>();

            // 告诉下一个核我reduce完成
            SetSignalValue((__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2st), localSetTensor, tag);
        }
    }
}

template <typename T>
__aicore__ inline void
AivReduceScatterDeterSmall910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize)
{
    int64_t count = len;
    int64_t allCount = count * rankSize_;
    int64_t blockNumPerGroup = rankSize_;
    int64_t x = blockIdx_ % blockNumPerGroup; // x means target rank
    int64_t flagOffsetBasic = separateOffset + BASE_FLAG_OFFSET * AIV_REDUCE_SCATTER_DETER_910B_SMALLDATA;

    uint32_t flagOffsetBase = ((tag % 2 == 0) ? 0 : 6 * rankSize_ * FLAG_SIZE) + flagOffsetBasic;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[x] + dataOffset);
    __gm__ T* outputGM = (__gm__ T*)output;

    int64_t flagOffset1st = flagOffsetBase + x * FLAG_SIZE;
    int64_t flagOffset2st = flagOffsetBase + (rankSize_ + x) * FLAG_SIZE;
    // 第一组 先从input拷贝到cclbuffer
    if (blockIdx_ < blockNumPerGroup) {
        CpGM2GM(cclGMSelf + x * count, inputGM + x * count, count);
        PipeBarrier<PIPE_ALL>();
        SetSignalValue((__gm__ int32_t*)(GM_OUT[rank_] + flagOffsetBase + blockIdx_ * FLAG_SIZE), localSetTensor, tag);
    }
    // 第二组 等待第一组完成，拷贝cclbuffer到cllbuffer后半部分, 并进行reduce
    else {
        WaitSignalValue((__gm__ int32_t*)(GM_OUT[x] + flagOffsetBase + rank_ * FLAG_SIZE), localCheckTensor, tag);
        CpGM2GM(cclGMSelf + allCount + x * count, cclGMOther + rank_ * count, count);
        PipeBarrier<PIPE_ALL>();
        GatherReduce(x, count, tag, cclGMSelf + allCount, flagOffset2st);
    }
    // 第一组 拷贝cclbuffer到output
    PipeBarrier<PIPE_ALL>();
    if (blockIdx_ < blockNumPerGroup) {
        int64_t flagOffsetLast = flagOffsetBase + (rankSize_ + rankSize_ - 1) * FLAG_SIZE;
        if (rankSize_ >= DETERMINISTIC_RANKSIZE) {
            if (rankSize_ < 5) {
                flagOffsetLast = flagOffsetBase + (rankSize_ + 2) * FLAG_SIZE + rankSize_ * 2 * FLAG_SIZE;
            } else {
                flagOffsetLast = flagOffsetBase + (rankSize_ + 4) * FLAG_SIZE + rankSize_ * 4 * FLAG_SIZE;
            }
        }

        if (x == rank_) {
            WaitSignalValue((__gm__ int32_t*)(GM_OUT[x] + flagOffsetLast), localCheckTensor, tag);
            CpGM2GM(outputGM, cclGMSelf + allCount, count);
        }
    }
}

template <typename T>
__aicore__ inline void AivReduceScatterDeterSmall910B::ProcessSingleRanksizeCore(
    GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize)
{
    int64_t count = len;
    int64_t allCount = count * rankSize_;
    int64_t blockNumPerGroup = rankSize_;
    int64_t x = blockIdx_ % blockNumPerGroup;
    int64_t flagOffsetBasic = separateOffset + BASE_FLAG_OFFSET * AIV_REDUCE_SCATTER_DETER_910B_SMALLDATA;

    uint32_t flagOffsetBase = ((tag % 2 == 0) ? 0 : 6 * rankSize_ * FLAG_SIZE) + flagOffsetBasic;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[x] + dataOffset);
    __gm__ T* outputGM = (__gm__ T*)output;

    int64_t flagOffset1st = flagOffsetBase + x * FLAG_SIZE;
    int64_t flagOffset2st = flagOffsetBase + (rankSize_ + x) * FLAG_SIZE;

    // 先从input拷贝到cclbuffer
    CpGM2GM(cclGMSelf + x * count, inputGM + x * count, count);
    PipeBarrier<PIPE_ALL>();
    SetSignalValue((__gm__ int32_t*)(GM_OUT[rank_] + flagOffsetBase + blockIdx_ * FLAG_SIZE), localSetTensor, tag);

    // 拷贝cclbuffer到cllbuffer后半部分, 并进行reduce
    WaitSignalValue((__gm__ int32_t*)(GM_OUT[x] + flagOffsetBase + rank_ * FLAG_SIZE), localCheckTensor, tag);
    CpGM2GM(cclGMSelf + allCount + x * count, cclGMOther + rank_ * count, count);
    PipeBarrier<PIPE_ALL>();
    GatherReduce(x, count, tag, cclGMSelf + allCount, flagOffset2st);
    PipeBarrier<PIPE_ALL>();

    // 拷贝cclbuffer到output
    if (x == rank_) {
        int64_t flagOffsetLast = flagOffsetBase + (2 * rankSize_ - 1) * FLAG_SIZE;
        if (rankSize_ >= DETERMINISTIC_RANKSIZE) {
            if (rankSize_ < 5) {
                flagOffsetLast = flagOffsetBase + (rankSize_ + 2) * FLAG_SIZE + rankSize_ * 2 * FLAG_SIZE;
            } else {
                flagOffsetLast = flagOffsetBase + (rankSize_ + 4) * FLAG_SIZE + rankSize_ * 4 * FLAG_SIZE;
            }
        }
        WaitSignalValue((__gm__ int32_t*)(GM_OUT[x] + flagOffsetLast), localCheckTensor, tag);
        CpGM2GM(outputGM, cclGMSelf + allCount, count);
    }
}

template <typename T>
__aicore__ inline void AivReduceScatterDeterSmall910B::ProcessProxy(
    GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize)
{
    if (numBlocks_ == rankSize_) {
        ProcessSingleRanksizeCore<T>(input, output, len, tag, bufferSize);
    } else {
        Process<T>(input, output, len, tag, bufferSize);
    }
}

template <typename T>
__aicore__ inline void aiv_reduce_scatter_deter_910b_smalldata(KERNEL_ARGS_DEF)
{
    AivReduceScatterDeterSmall910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.ProcessProxy<T>(input, output, len, tag, bufferSize);
    op.TailCounter();
}
