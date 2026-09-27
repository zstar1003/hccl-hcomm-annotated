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

class AivAllReduceSmallGraph910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceSmallGraph910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint32_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void AivAllReduceSmallGraph910B::Process(GM_ADDR input, GM_ADDR output, uint32_t len, int32_t tag)
{
    uint32_t count = len;

    if (blockIdx_ == rank_) {
        __gm__ T* inputGM = (__gm__ T*)input;
        __gm__ T* outputGM = (__gm__ T*)output;

        CpGM2GM(outputGM, inputGM, count);

        PipeBarrier<PIPE_MTE3>();

        // 卡内同步
        Record1vN(tag, CommPattern::intraRank);
    } else {
        __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[blockIdx_]);
        __gm__ T* outputGM = (__gm__ T*)output;
        // 告诉对端可以从本端拉走数据
        Record(tag, blockIdx_, AivNotifyType::ACK);
        Wait(tag, blockIdx_, AivNotifyType::ACK);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<T> cclGTOther;
        cclGTOther.SetGlobalBuffer(cclGMOther, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, count);

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, cclGTOther, count);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();

        // 卡内同步
        WaitNv1(tag, rank_);

        PipeBarrier<PIPE_ALL>();

        SetAtomicOp<T>(reduceOp_);
        DataCopyUB2GM(outputGT, localOut, count);
        SetAtomicNone();

        inOutQue.FreeTensor(localOut);

        PipeBarrier<PIPE_ALL>();

        // 本端告诉对端已经拉走数据
        Record(tag, blockIdx_, AivNotifyType::DataSignal);
        Wait(tag, blockIdx_, AivNotifyType::DataSignal);
    }
}

template <typename T>
__aicore__ inline void aiv_all_reduce_910b_smalldata_graph(KERNEL_ARGS_DEF)
{
    AivAllReduceSmallGraph910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag);
    op.TailCounter();
}

__aicore__ inline void sk_all_reduce_910b_smalldata_graph(SUPERKERNEL_ARGS_DEF)
{
    AivAllReduceSmallGraph910B op;
    op.Init(SUPERKERNEL_CLASS_INIT, 0, false);
#ifdef HCCL_DTYPE_INT8
    op.Process<int8_t>(input, output, op.len_, op.tag_);
#elif defined HCCL_DTYPE_INT16
    op.Process<int16_t>(input, output, op.len_, op.tag_);
#elif defined HCCL_DTYPE_INT32
    op.Process<int32_t>(input, output, op.len_, op.tag_);
#elif defined HCCL_DTYPE_FP16
    op.Process<half>(input, output, op.len_, op.tag_);
#elif defined HCCL_DTYPE_FP32
    op.Process<float>(input, output, op.len_, op.tag_);
#elif defined HCCL_DTYPE_BFP16
    op.Process<bfloat16_t>(input, output, op.len_, op.tag_);
#elif defined HCCL_DTYPE_UINT8
    op.Process<uint8_t>(input, output, op.len_, op.tag_);
#elif defined HCCL_DTYPE_UINT16
    op.Process<uint16_t>(input, output, op.len_, op.tag_);
#elif defined HCCL_DTYPE_UINT32
    op.Process<uint32_t>(input, output, op.len_, op.tag_);
#else
#endif
}
