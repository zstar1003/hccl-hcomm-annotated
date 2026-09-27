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

class AivAllReduceRdmaMid910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceRdmaMid910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, int32_t aivRdmaStep);

    template <typename T>
    __aicore__ inline void ReduceScatter(
        __gm__ T* inputGM, __gm__ T* cclGMSelf, __gm__ T* cclGMOther, uint64_t sliceCount, uint64_t avgLengthPerSlice,
        uint64_t tailLength, int32_t tag);

    template <typename T>
    __aicore__ inline void AllGather(
        __gm__ T* outputGM, __gm__ T* cclGMSelf, __gm__ T* cclGMOther, uint64_t sliceCount, uint64_t avgLengthPerSlice,
        uint64_t tailLength, int32_t tag);
};

template <typename T>
__aicore__ inline void
AivAllReduceRdmaMid910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, int32_t aivRdmaStep)
{
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[blockIdx_]);
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    uint64_t avgLengthPerRank = CeilDiv(len, rankSize_);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerRank, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    if (aivRdmaStep == 0) {
        ReduceScatter(inputGM, cclGMSelf, cclGMOther, sliceCount, avgLengthPerSlice, tailLength, tag);
    }
    if (aivRdmaStep == 2) {
        AllGather(outputGM, cclGMSelf, cclGMOther, sliceCount, avgLengthPerSlice, tailLength, tag);
    }
    return;
}

template <typename T>
__aicore__ inline void AivAllReduceRdmaMid910B::ReduceScatter(
    __gm__ T* inputGM, __gm__ T* cclGMSelf, __gm__ T* cclGMOther, uint64_t sliceCount, uint64_t avgLengthPerSlice,
    uint64_t tailLength, int32_t tag)
{
    // reduce scatter，数据从input输入，inputMem+0作为buffer，结果放在原位
    if (blockIdx_ == rank_) {
        int64_t curCount = CalActualCount(blockIdx_, sliceCount, avgLengthPerSlice, tailLength);

        // 本地拷贝 & 卡间同步
        CpGM2GM(cclGMSelf + avgLengthPerSlice * blockIdx_, inputGM + avgLengthPerSlice * blockIdx_, curCount);
        pipe_barrier(PIPE_ALL);
        Record1vN(tag, CommPattern::intraRank);
    } else {
        int64_t curCount = CalActualCount(blockIdx_, sliceCount, avgLengthPerSlice, tailLength);

        // 本地拷贝 & 卡间同步
        CpGM2GM(cclGMSelf + avgLengthPerSlice * blockIdx_, inputGM + avgLengthPerSlice * blockIdx_, curCount);
        pipe_barrier(PIPE_ALL);
        Record(tag, blockIdx_, AivNotifyType::ACK); // 本卡该片数据已经可以被跨片读取

        // 检查对端数据就绪且本端就绪 & 跨片搬运
        curCount = CalActualCount(rank_, sliceCount, avgLengthPerSlice, tailLength);

        Wait(tag, blockIdx_, AivNotifyType::ACK);
        WaitNv1(tag, rank_);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(
            cclGMSelf + avgLengthPerSlice * rank_, cclGMOther + avgLengthPerSlice * rank_, curCount, true, reduceOp_);
    }
    return;
}

template <typename T>
__aicore__ inline void AivAllReduceRdmaMid910B::AllGather(
    __gm__ T* outputGM, __gm__ T* cclGMSelf, __gm__ T* cclGMOther, uint64_t sliceCount, uint64_t avgLengthPerSlice,
    uint64_t tailLength, int32_t tag)
{
    if (blockIdx_ == rank_) {
        int64_t curCount = CalActualCount(blockIdx_, sliceCount, avgLengthPerSlice, tailLength);

        // 本地拷贝 & 卡间同步
        Record1vN(tag, CommPattern::interRank);
        CpGM2GM(outputGM + avgLengthPerSlice * blockIdx_, cclGMSelf + avgLengthPerSlice * blockIdx_, curCount);
    } else {
        int64_t curCount = CalActualCount(blockIdx_, sliceCount, avgLengthPerSlice, tailLength);

        // 检查对端就绪 & 跨片拷贝
        WaitNv1(tag, blockIdx_);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(outputGM + (blockIdx_ * avgLengthPerSlice), cclGMOther + blockIdx_ * avgLengthPerSlice, curCount);
        pipe_barrier(PIPE_ALL);

        // 末尾同步
        // 本卡已读完GetBlockIdx()号对端上的rank号数据
        Record(tag, blockIdx_, AivNotifyType::DataSignal);
        pipe_barrier(PIPE_ALL);
        // 检查本卡上是否有GetBlockIdx()号对端的读完标记
        Wait(tag, blockIdx_, AivNotifyType::DataSignal);
    }
    return;
}

template <typename T>
__aicore__ inline void aiv_all_reduce_910b_rdma_middata(KERNEL_ARGS_DEF)
{
    AivAllReduceRdmaMid910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag, aivRdmaStep);
    op.TailCounter();
}
