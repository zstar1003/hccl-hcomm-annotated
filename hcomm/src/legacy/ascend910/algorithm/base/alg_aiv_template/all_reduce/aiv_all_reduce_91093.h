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

class AivAllReduce91093 : public AivCommBase {
public:
    __aicore__ inline AivAllReduce91093() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);

    template <typename T>
    __aicore__ inline void ProcessSmall(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);

    template <typename T>
    __aicore__ inline void ProcessBig(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void AivAllReduce91093::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    if (len * sizeof(T) <= AIV_A3_ALL_REDUCE_GRAPH_GUIYI_SIZE) {
        ProcessSmall<T>(input, output, len, tag);
    } else {
        ProcessBig<T>(input, output, len, tag);
    }
}

template <typename T>
__aicore__ inline void AivAllReduce91093::ProcessSmall(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
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
    bool ifPingpong = (tag % 2 == 0);
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    // 用4个flag
    if (dstRank == rank_) {
        __gm__ T* inputGM = (__gm__ T*)input;
        __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
        __gm__ T* outputGM = (__gm__ T*)output;

        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM + blockOffset, count);
        GlobalTensor<T> cclGTSelf;
        cclGTSelf.SetGlobalBuffer(cclGMSelf + blockOffset, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM + blockOffset, count);

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT, count);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(cclGTSelf, localOut, count);

        PipeBarrier<PIPE_MTE3>();

        // 卡间同步
        Record1vN(tag, CommPattern::interRank, AivNotifyType::DataSignal, blockIdxInGroup, ifPingpong);

        DataCopyUB2GM(outputGT, localOut, count);
        inOutQue.FreeTensor(localOut);

        PipeBarrier<PIPE_MTE3>();

        // 卡内同步
        Record1vN(tag, CommPattern::intraRank, AivNotifyType::DataSignal, blockIdxInGroup, ifPingpong);
    } else {
        __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[dstRank] + dataOffset);
        __gm__ T* outputGM = (__gm__ T*)output;

        GlobalTensor<T> cclGTOther;
        cclGTOther.SetGlobalBuffer(cclGMOther + blockOffset, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM + blockOffset, count);

        // 卡间同步
        WaitNv1(tag, dstRank, AivNotifyType::DataSignal, blockIdxInGroup, ifPingpong);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, cclGTOther, count);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();

        // 卡内同步
        WaitNv1(tag, rank_, AivNotifyType::DataSignal, blockIdxInGroup, ifPingpong);
        PipeBarrier<PIPE_ALL>();

        SetAtomicOp<T>(reduceOp_);
        DataCopyUB2GM(outputGT, localOut, count);
        SetAtomicNone();

        inOutQue.FreeTensor(localOut);
    }
}

template <typename T>
__aicore__ inline void AivAllReduce91093::ProcessBig(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t blockNumPerGroup = numBlocks_ / rankSize_;
    uint32_t blockIdxInGroup = blockIdx_ % blockNumPerGroup;
    uint32_t dstRank = blockIdx_ / blockNumPerGroup;
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);

    uint64_t avgLengthPerBlock = CeilDiv(len, numBlocks_);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    uint64_t count = 0;
    // 使用19个flag
    __gm__ T* outputGm = (__gm__ T*)output;
    __gm__ T* cclGmSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGmOther = (__gm__ T*)(GM_IN[dstRank]);

    // 本卡已进入算子，通知其他卡可以搬运，使用第1个flag
    Record(tag, dstRank, AivNotifyType::ACK, blockIdxInGroup);

    // 确认对端已经将对应的数据拉走
    Wait(tag, dstRank, AivNotifyType::ACK, blockIdxInGroup);
    PipeBarrier<PIPE_ALL>();

    // ReduceScatter
    if (dstRank != rank_) {
        uint32_t sliceIdx = rank_ * blockNumPerGroup + blockIdxInGroup;
        count = CalActualCount(sliceIdx, sliceCount, avgLengthPerSlice, tailLength);

        uint64_t gmOffset = sliceIdx * avgLengthPerSlice;

        CpGM2GM(cclGmSelf + gmOffset, cclGmOther + gmOffset, count, true, reduceOp_);

        PipeBarrier<PIPE_MTE3>();

        // 本aiv reduce完成，使用第2个flag
        RecordNv1(tag, rank_, AivNotifyType::ACK, blockIdxInGroup);
    }

    // 全卡同步
    PipeBarrier<PIPE_ALL>();
    if (dstRank == rank_) {
        // check 本端aiv 所有reduce结果是否完成
        Wait1vN((rankSize_ - 1) * tag, CommPattern::intraRank, true, AivNotifyType::ACK, blockIdxInGroup);
        SyncFunc<HardEvent::MTE3_S>();

        // 告诉别人自己已经加完所有卡了，使用第3个flag
        Record1vN(tag, CommPattern::interRank, AivNotifyType::ACK, blockIdxInGroup);
        SyncFunc<HardEvent::MTE3_MTE2>();
    } else {
        // 每个aiv读相应对端的flag
        WaitNv1(tag, dstRank, AivNotifyType::ACK, blockIdxInGroup);
    }
    PipeBarrier<PIPE_ALL>();

    // AllGather
    uint32_t sliceIdx = dstRank * blockNumPerGroup + blockIdxInGroup;
    uint64_t gmOffset = sliceIdx * avgLengthPerSlice;
    count = CalActualCount(sliceIdx, sliceCount, avgLengthPerSlice, tailLength);
    CpGM2GM(outputGm + gmOffset, cclGmOther + gmOffset, count);
    PipeBarrier<PIPE_ALL>();
    // 通知对端，自己已经把对端的那片数据拉回来了
    Record(tag, dstRank, AivNotifyType::DataSignal, blockIdxInGroup);
    // 确认对端已经将对应的数据拉走
    Wait(tag, dstRank, AivNotifyType::DataSignal, blockIdxInGroup);
    return;
}

template <typename T>
__aicore__ inline void aiv_all_reduce_91093(KERNEL_ARGS_DEF)
{
    AivAllReduce91093 op;
    op.Init(KERNEL_CLASS_INIT, len * sizeof(T) > AIV_A3_ALL_REDUCE_GRAPH_GUIYI_SIZE);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag);
    op.TailCounter();
}

__aicore__ inline void sk_all_reduce_91093(SUPERKERNEL_ARGS_DEF)
{
    AivAllReduce91093 op;
    op.Init(SUPERKERNEL_CLASS_INIT, AIV_A3_ALL_REDUCE_GRAPH_GUIYI_SIZE);
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
