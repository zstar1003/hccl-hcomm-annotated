/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base.h"

using namespace AscendC;

class AivReduceScatter910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatter910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void AivReduceScatter910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    localSetTensor.SetValue(0, tag);
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;

    uint32_t blockNumPerGroup = numBlocks_ / rankSize_; // numBlocks_需要能被rankSize_整除
    uint32_t blockIdxInGroup = blockIdx_ % blockNumPerGroup;

    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    uint64_t avgLengthPerBlock = CeilDiv(len, blockNumPerGroup);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    uint64_t count = CalActualCount(blockIdxInGroup, sliceCount, avgLengthPerSlice, tailLength);
    uint64_t blockOffset = blockIdxInGroup * avgLengthPerSlice;
    uint32_t dstRank = blockIdx_ / blockNumPerGroup;

    GlobalTensor<int32_t> globalSet;
    __gm__ int32_t* ctrlFlagsGML
        = (__gm__ int32_t*)(GM_OUT[rank_] + multiOffset
                            + (2 * NUM_BLOCKS_FOUR_PER_RANK_A3 + blockIdxInGroup) * ATOMIC_FLAG_SIZE);
    globalSet.SetGlobalBuffer(ctrlFlagsGML, UB_FLAG_PAD_COUNT);

    if (dstRank == rank_) {
        CpGM2GM(outputGM + blockOffset, (__gm__ T*)(inputGM + rank_ * len + blockOffset), count);
        pipe_barrier(PIPE_MTE3);
        DataCopy(globalSet, localSetTensor, UB_FLAG_PAD_COUNT);
    } else {
        __gm__ int32_t* ctrlFlagsGMX
            = (__gm__ int32_t*)(GM_OUT[dstRank] + (rankSize_ * FLAG_BUF_NUM * blockIdxInGroup + rank_) * FLAG_SIZE);
        __gm__ int32_t* ctrlFlagsGM
            = (__gm__ int32_t*)(GM_OUT[rank_] + (rankSize_ * FLAG_BUF_NUM * blockIdxInGroup + dstRank) * FLAG_SIZE);
        globalSet.SetGlobalBuffer(ctrlFlagsGMX, UB_FLAG_PAD_COUNT);
        DataCopy(globalSet, localSetTensor, UB_FLAG_PAD_COUNT);
        WaitSignalValue(ctrlFlagsGM, localCheckTensor, tag);
        WaitSignalValue(ctrlFlagsGML, localCheckTensor, tag);

        PipeBarrier<PIPE_ALL>();

        CpGM2GM(
            outputGM + blockOffset, (__gm__ T*)(GM_IN[dstRank]) + rank_ * len + blockOffset, count, true, reduceOp_);

        ctrlFlagsGMX
            = (__gm__ int32_t*)(GM_OUT[dstRank]
                                + (rankSize_ * FLAG_BUF_NUM * blockIdxInGroup + rank_ + rankSize_) * FLAG_SIZE);
        ctrlFlagsGM
            = (__gm__ int32_t*)(GM_OUT[rank_]
                                + (rankSize_ * FLAG_BUF_NUM * blockIdxInGroup + dstRank + rankSize_) * FLAG_SIZE);
        pipe_barrier(PIPE_MTE3);
        globalSet.SetGlobalBuffer(ctrlFlagsGMX, UB_FLAG_PAD_COUNT);
        DataCopy(globalSet, localSetTensor, UB_FLAG_PAD_COUNT);
        WaitSignalValue(ctrlFlagsGM, localCheckTensor, tag);
    }

    return;
}

__aicore__ inline void sk_reduce_scatter_910B(SUPERKERNEL_ARGS_DEF)
{
    AivReduceScatter910B op;
    op.Init(SUPERKERNEL_CLASS_INIT, 0, true);
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
#else
#endif
}
