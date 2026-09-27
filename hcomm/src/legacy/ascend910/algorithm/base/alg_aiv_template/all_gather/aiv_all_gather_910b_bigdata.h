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

class AivAllGatherBig910B : public AivCommBase {
public:
    __aicore__ inline AivAllGatherBig910B() {}

    template <typename T>
    __aicore__ inline void
    MemcpyWithFlagWrap(__gm__ T* cclGmSelf, __gm__ T* cclGmOther, uint64_t count, int32_t dstRank, int32_t tag);

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t totalLen);

    template <typename T>
    __aicore__ inline void ProcessProxy(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t totalLen);

    template <typename T>
    __aicore__ inline void
    ProcessSingleRanksizeCore(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t totalLen);

    template <typename T>
    __aicore__ inline void ClearFlag(uint32_t flagOffsetBase);
};

template <typename T>
__aicore__ inline void AivAllGatherBig910B::ClearFlag(uint32_t flagOffsetBase)
{
    // 用10个flag
    uint32_t flagOffsetCount = flagOffsetBase;
    __gm__ int32_t* ctrlFlagsGM = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffsetBase + rank_ * FLAG_SIZE);
    if (blockIdx_ < rankSize_ && blockIdx_ == rank_) {
        SetSignalValue(ctrlFlagsGM, localSetTensor, 0);
    }
}

template <typename T>
__aicore__ inline void AivAllGatherBig910B::MemcpyWithFlagWrap(
    __gm__ T* cclGmSelf, __gm__ T* cclGmOther, uint64_t count, int32_t dstRank, int32_t tag)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);

    while (true) {
        if (processedBatchCount >= CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE)) {
            break;
        }
#ifndef OPEN_HCCL_TEST
        uint64_t localFlagValueX = CountWait(dstRank, dstRank);
#else
        LocalTensor<int32_t> localFlagX = flagInQue.AllocTensor<int32_t>();
        uint64_t localFlagValueX = GetSignalValueWithExpected(
            (int32_t*)(GM_OUT[dstRank] + countOffset + dstRank * FLAG_SIZE), localFlagX,
            CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE) + tag);
        flagInQue.FreeTensor(localFlagX);
#endif

        if (localFlagValueX <= tag) {
            continue;
        }

        uint64_t preparedBatchCount = localFlagValueX - tag;
        if (processedBatchCount >= preparedBatchCount) {
            continue;
        }

        uint64_t curSize = (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;
        if (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > avgSizePerSlice) {
            curSize = avgSizePerSlice - processedBatchCount * UB_DB_DATA_BATCH_SIZE;
        }

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

        uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
        CpGM2GM(cclGmSelf + curProcessedOffset, cclGmOther + curProcessedOffset, curSize / sizeof(T));
        PipeBarrier<PIPE_ALL>();
        processedBatchCount = preparedBatchCount;
    }
}

template <typename T>
__aicore__ inline void
AivAllGatherBig910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t totalLen)
{
    uint32_t avgLengthPerSlice = len;
    uint32_t avgSizePerSlice = avgLengthPerSlice * sizeof(T);

    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = blockIdx_ >= rankSize_ ? blockIdx_ - rankSize_ : blockIdx_;

    // 用10个flag
    __gm__ T* inputGm = (__gm__ T*)input;
    __gm__ T* outputGm = (__gm__ T*)output;
    __gm__ T* cclGmSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGmOther = (__gm__ T*)(GM_IN[targetRank]);

    if (blockIdx_ < blockNumPerGroup) {
        int32_t outputOffset = targetRank * totalLen;
        if (blockIdx_ == rank_) {
            CpGM2GMWithFlagWrap(cclGmSelf, inputGm, avgLengthPerSlice, rank_, 8, tag);
            // 所有对端都取走数据
            PipeBarrier<PIPE_ALL>();
            Wait1vN((rankSize_ - 1) * tag, CommPattern::intraRank, true);
        } else {
            MemcpyWithFlagWrap(outputGm + outputOffset, cclGmOther, len, targetRank, tag);
            PipeBarrier<PIPE_ALL>();
            Record(tag, targetRank, AivNotifyType::DataSignal);
            PipeBarrier<PIPE_ALL>();
            Wait(tag, targetRank, AivNotifyType::DataSignal);
            PipeBarrier<PIPE_ALL>();
            // 是否要加清零的参数
            RecordNv1(tag, rank_);
            PipeBarrier<PIPE_ALL>();
        }
    } else {
        CpGM2GM(outputGm + rank_ * totalLen, inputGm, avgLengthPerSlice);
    }
    return;
}

template <typename T>
__aicore__ inline void AivAllGatherBig910B::ProcessSingleRanksizeCore(
    GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t totalLen)
{
    uint32_t avgLengthPerSlice = len;
    uint32_t avgSizePerSlice = avgLengthPerSlice * sizeof(T);
    uint32_t targetRank = blockIdx_;

    // 用10个flag
    __gm__ T* inputGm = (__gm__ T*)input;
    __gm__ T* outputGm = (__gm__ T*)output;
    __gm__ T* cclGmSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGmOther = (__gm__ T*)(GM_IN[targetRank]);

    int32_t outputOffset = targetRank * totalLen;
    if (blockIdx_ == rank_) {
        CpGM2GMWithFlagWrap(cclGmSelf, inputGm, avgLengthPerSlice, rank_, 8, tag);
        PipeBarrier<PIPE_ALL>();
        CpGM2GM(outputGm + rank_ * totalLen, inputGm, avgLengthPerSlice);
        // 所有对端都取走数据
        PipeBarrier<PIPE_ALL>();
        Wait1vN((rankSize_ - 1) * tag, CommPattern::intraRank, true);
    } else {
        MemcpyWithFlagWrap(outputGm + outputOffset, cclGmOther, len, targetRank, tag);
        PipeBarrier<PIPE_ALL>();
        Record(tag, targetRank, AivNotifyType::DataSignal);
        PipeBarrier<PIPE_ALL>();
        Wait(tag, targetRank, AivNotifyType::DataSignal);
        PipeBarrier<PIPE_ALL>();
        // 是否要加清零的参数
        RecordNv1(tag, rank_);
        PipeBarrier<PIPE_ALL>();
    }
    return;
}

template <typename T>
__aicore__ inline void
AivAllGatherBig910B::ProcessProxy(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t totalLen)
{
    if (numBlocks_ == rankSize_) {
        ProcessSingleRanksizeCore<T>(input, output, len, tag, totalLen);
    } else {
        Process<T>(input, output, len, tag, totalLen);
    }
}

template <typename T>
__aicore__ inline void aiv_all_gather_910b_bigdata(KERNEL_ARGS_DEF)
{
    AivAllGatherBig910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    uint64_t maxCountPerLoop = bufferSize / UB_ALIGN_SIZE * UB_ALIGN_SIZE / sizeof(T);
    uint64_t countLeft = len;

    GM_ADDR curInput = input;
    GM_ADDR curOutput = output;
    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        // 执行kernel
        op.ProcessProxy<T>(curInput, curOutput, curCount, curTag, len);

        countLeft -= curCount;
        curInput += curSize;
        curOutput += curSize;
        curTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
    op.TailCounter();
}
