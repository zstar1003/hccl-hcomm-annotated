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

class AivAllGatherSmall91093 : public AivCommBase {
public:
    __aicore__ inline AivAllGatherSmall91093() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);

    template <typename T>
    __aicore__ inline void ProcessSmall(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);

    template <typename T>
    __aicore__ inline void ProcessBig(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);

private:
    LocalTensor<int32_t> localFlagTensor;
};

template <typename T>
__aicore__ inline void AivAllGatherSmall91093::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    if (len * sizeof(T) <= AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE) {
        ProcessSmall<T>(input, output, len, tag);
    } else {
        ProcessBig<T>(input, output, len, tag);
    }
}

template <typename T>
__aicore__ inline void AivAllGatherSmall91093::ProcessSmall(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t blockNumPerGroup = numBlocks_ / rankSize_; // numBlocks_需要能被rankSize_整除
    uint32_t blockIdxInGroup = blockIdx_ % blockNumPerGroup;
    localFlagTensor = localFlagBuf.Get<int32_t>();

    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    uint64_t avgLengthPerBlock = CeilDiv(len, blockNumPerGroup);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    uint64_t count = CalActualCount(blockIdxInGroup, sliceCount, avgLengthPerSlice, tailLength);
    uint64_t blockOffset = blockIdxInGroup * avgLengthPerSlice;
    uint32_t dstRank = blockIdx_ / blockNumPerGroup;

    // 共用2个flag
    uint32_t flagOffset = ((tag % 2 == 1) ? 0 : pingpongOffset) + multiOffset
                          + NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * 2
                          + NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[dstRank] + dataOffset);
    __gm__ T* outputGM = (__gm__ T*)output;

    if (dstRank != rank_) {
        GlobalTensor<int32_t> globalCheck;
        globalCheck.SetGlobalBuffer(
            (__gm__ int32_t*)(GM_OUT[dstRank] + flagOffset + blockIdxInGroup * ATOMIC_FLAG_SIZE), UB_FLAG_PAD_COUNT);
        while (true) {
            DataCopy(localFlagTensor[8], globalCheck, UB_FLAG_PAD_COUNT);
            SyncFunc<HardEvent::MTE2_S>();
            if (localFlagTensor[8].GetValue(0) == tag) {
                break;
            }
        }
        SyncFunc<HardEvent::S_MTE2>();

        CpGM2GM(outputGM + dstRank * len + blockOffset, cclGMOther + blockOffset, count);
        // 卡间同步
    } else {
        CpGM2GM(cclGMSelf + blockOffset, inputGM + blockOffset, count);
        // 卡间同步
        GlobalTensor<int32_t> globalSet;
        globalSet.SetGlobalBuffer(
            (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset + blockIdxInGroup * ATOMIC_FLAG_SIZE), UB_FLAG_PAD_COUNT);
        localFlagTensor.SetValue(0, tag);
        PipeBarrier<PIPE_MTE3>();
        SyncFunc<HardEvent::S_MTE3>();
        DataCopy(globalSet, localFlagTensor, UB_FLAG_PAD_COUNT);

        CpGM2GM(outputGM + len * rank_ + blockOffset, inputGM + blockOffset, count); // 与上独立
    }
}

template <typename T>
__aicore__ inline void AivAllGatherSmall91093::ProcessBig(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
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

    uint32_t flagOffset = blockIdxInGroup * rankSize_ * FLAG_SIZE * 3;
    __gm__ int32_t* ctrlFlagsGMX = (__gm__ int32_t*)(GM_OUT[dstRank] + rank_ * FLAG_SIZE + flagOffset);
    __gm__ int32_t* ctrlFlagsGM = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset + dstRank * FLAG_SIZE);
    GlobalTensor<int32_t> globalSet;

    if (dstRank == rank_) {
        CpGM2GM(outputGM + rank_ * len + blockOffset, (__gm__ T*)(inputGM + blockOffset), count);
    } else {
        globalSet.SetGlobalBuffer(ctrlFlagsGMX, UB_FLAG_PAD_COUNT);
        DataCopy(globalSet, localSetTensor, UB_FLAG_PAD_COUNT);
        WaitSignalValue(ctrlFlagsGM, localCheckTensor, tag);
        PipeBarrier<PIPE_ALL>();

        CpGM2GM(outputGM + dstRank * len + blockOffset, (__gm__ T*)(GM_IN[dstRank]) + blockOffset, count);

        ctrlFlagsGMX = (__gm__ int32_t*)(GM_OUT[dstRank] + rank_ * FLAG_SIZE + flagOffset + rankSize_ * FLAG_SIZE);
        ctrlFlagsGM = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset + dstRank * FLAG_SIZE + rankSize_ * FLAG_SIZE);
        PipeBarrier<PIPE_MTE3>();
        globalSet.SetGlobalBuffer(ctrlFlagsGMX, UB_FLAG_PAD_COUNT);
        DataCopy(globalSet, localSetTensor, UB_FLAG_PAD_COUNT);
        WaitSignalValue(ctrlFlagsGM, localCheckTensor, tag);
    }
    return;
}

template <typename T>
__aicore__ inline void aiv_all_gather_91093_smalldata(KERNEL_ARGS_DEF)
{
    AivAllGatherSmall91093 op;
    if (len * sizeof(T) > AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE) {
        op.Init(KERNEL_CLASS_INIT, true);
    } else {
        op.Init(KERNEL_CLASS_INIT, false);
    }
    op.HeadCounter();
    op.Process<T>(input, output, len, tag);
    op.TailCounter();
}

__aicore__ inline void sk_all_gather_91093_smalldata(SUPERKERNEL_ARGS_DEF)
{
    AivAllGatherSmall91093 op;
    op.Init(SUPERKERNEL_CLASS_INIT, AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE);
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
#elif defined HCCL_DTYPE_INT64
    op.Process<int64_t>(input, output, op.len_, op.tag_);
#elif defined HCCL_DTYPE_UINT64
    op.Process<uint64_t>(input, output, op.len_, op.tag_);
#elif defined HCCL_DTYPE_FP64
    op.Process<double>(input, output, op.len_, op.tag_);
#else
#endif
}
