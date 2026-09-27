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

class AivReduceScatterBig910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterBig910B() {}
    template <typename T>
    __aicore__ inline void
    ReduceWithFlagWrap(__gm__ T* cclGmSelf, __gm__ T* cclGmOther, uint64_t count, int32_t dstRank, int32_t tag);

    template <typename T>
    __aicore__ inline void
    Process(GM_ADDR input, GM_ADDR output, uint64_t len, uint64_t maxCount, int32_t tag, uint64_t totallen);

    template <typename T>
    __aicore__ inline void
    ProcessProxy(GM_ADDR input, GM_ADDR output, uint64_t len, uint64_t maxCount, int32_t tag, uint64_t totallen);

    template <typename T>
    __aicore__ inline void ProcessSingleRanksizeCore(
        GM_ADDR input, GM_ADDR output, uint64_t len, uint64_t maxCount, int32_t tag, uint64_t totallen);
};

template <typename T>
__aicore__ inline void AivReduceScatterBig910B::ReduceWithFlagWrap(
    __gm__ T* cclGmSelf, __gm__ T* cclGmOther, uint64_t count, int32_t dstRank, int32_t tag)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);

    while (true) {
        if (processedBatchCount >= CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE)) {
            break;
        }

#ifndef OPEN_HCCL_TEST
        int32_t localFlag = CountWait(rank_, rank_);
        int32_t RemoteFlag = CountWait(dstRank, rank_);
#else
        LocalTensor<int32_t> localFlagX = flagInQue.AllocTensor<int32_t>();
        LocalTensor<int32_t> localFlagY = flagInQue.AllocTensor<int32_t>();
        int32_t localFlag = GetSignalValueWithExpected(
            (int32_t*)(GM_OUT[rank_] + countOffset + rank_ * FLAG_SIZE), localFlagX,
            CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE) + tag);
        int32_t RemoteFlag = GetSignalValueWithExpected(
            (int32_t*)(GM_OUT[dstRank] + countOffset + rank_ * FLAG_SIZE), localFlagY,
            CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE) + tag);
        flagInQue.FreeTensor(localFlagX);
        flagInQue.FreeTensor(localFlagY);
#endif

        int64_t localFlagValue = localFlag - tag;
        int64_t RemoteFlagValue = RemoteFlag - tag;

        if (localFlagValue <= 0 || RemoteFlagValue <= 0) {
            continue;
        }

        uint64_t preparedBatchCount = (localFlagValue <= RemoteFlagValue) ? localFlagValue : RemoteFlagValue;
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
        CpGM2GM(cclGmSelf + curProcessedOffset, cclGmOther + curProcessedOffset, curSize / sizeof(T), true, reduceOp_);

        processedBatchCount = preparedBatchCount;
    }
}

template <typename T>
__aicore__ inline void AivReduceScatterBig910B::Process(
    GM_ADDR input, GM_ADDR output, uint64_t len, uint64_t maxCount, int32_t tag, uint64_t totallen)
{
    uint64_t avgLengthPerSlice = len;
    uint64_t avgSizePerSlice = avgLengthPerSlice * sizeof(T);

    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = blockIdx_ >= rankSize_ ? blockIdx_ - rankSize_ : blockIdx_;

    __gm__ T* inputGm = (__gm__ T*)input;
    __gm__ T* outputGm = (__gm__ T*)output;
    __gm__ T* cclGmSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGmOther = (__gm__ T*)(GM_IN[targetRank]);

    if (blockIdx_ < blockNumPerGroup) {
        uint64_t inputOffset = targetRank * totallen;
        uint64_t cclGmSelfOffset = targetRank * maxCount;

        if (targetRank != rank_) {
            CpGM2GMWithFlagWrap(
                cclGmSelf + cclGmSelfOffset, inputGm + inputOffset, avgLengthPerSlice, targetRank, 8, tag);
            // 确定对端已经拉走数据
            pipe_barrier(PIPE_ALL);
            Wait(tag, targetRank, AivNotifyType::DataSignal);
        }
    } else if (targetRank != rank_) {
        uint64_t cclGmOtherOffset = rank_ * maxCount;

        ReduceWithFlagWrap(outputGm, cclGmOther + cclGmOtherOffset, len, targetRank, tag);

        pipe_barrier(PIPE_ALL);
        // 通知对端已把数据拉走
        Record(tag, targetRank, AivNotifyType::DataSignal);
        pipe_barrier(PIPE_ALL);
        // 通知本端已相加
        RecordNv1(tag, rank_);
    } else {
        uint64_t inputOffset = rank_ * totallen;

        CpGM2GMWithFlagWrap(outputGm, inputGm + inputOffset, avgLengthPerSlice, rank_, 8, tag);
        // 确认已加完
        Wait1vN((rankSize_ - 1) * tag, CommPattern::intraRank);
    }

    return;
}

template <typename T>
__aicore__ inline void AivReduceScatterBig910B::ProcessSingleRanksizeCore(
    GM_ADDR input, GM_ADDR output, uint64_t len, uint64_t maxCount, int32_t tag, uint64_t totallen)
{
    uint64_t avgLengthPerSlice = len;
    uint64_t avgSizePerSlice = avgLengthPerSlice * sizeof(T);
    uint32_t targetRank = blockIdx_;

    __gm__ T* inputGm = (__gm__ T*)input;
    __gm__ T* outputGm = (__gm__ T*)output;
    __gm__ T* cclGmSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGmOther = (__gm__ T*)(GM_IN[targetRank]);

    uint64_t inputOffset = targetRank * totallen;
    uint64_t cclGmSelfOffset = targetRank * maxCount;

    if (targetRank != rank_) {
        CpGM2GMWithFlagWrap(cclGmSelf + cclGmSelfOffset, inputGm + inputOffset, avgLengthPerSlice, targetRank, 8, tag);
        pipe_barrier(PIPE_ALL);

        uint64_t cclGmOtherOffset = rank_ * maxCount;
        ReduceWithFlagWrap(outputGm, cclGmOther + cclGmOtherOffset, len, targetRank, tag);
        pipe_barrier(PIPE_ALL);

        RecordNv1(tag, rank_);
    } else {
        uint64_t inputOffset = rank_ * totallen;

        CpGM2GMWithFlagWrap(outputGm, inputGm + inputOffset, avgLengthPerSlice, rank_, 8, tag);
        // 确认已加完
        Wait1vN((rankSize_ - 1) * tag, CommPattern::intraRank);
    }

    return;
}

template <typename T>
__aicore__ inline void AivReduceScatterBig910B::ProcessProxy(
    GM_ADDR input, GM_ADDR output, uint64_t len, uint64_t maxCount, int32_t tag, uint64_t totallen)
{
    if (numBlocks_ == rankSize_) {
        ProcessSingleRanksizeCore<T>(input, output, len, maxCount, tag, totallen);
    } else {
        Process<T>(input, output, len, maxCount, tag, totallen);
    }
}

template <typename T>
__aicore__ inline void aiv_reduce_scatter_910b_bigdata(KERNEL_ARGS_DEF)
{
    AivReduceScatterBig910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    uint64_t maxCountPerLoop = bufferSize / UB_ALIGN_SIZE * UB_ALIGN_SIZE / rankSize / sizeof(T);
    uint64_t countLeft = len;

    GM_ADDR curInput = input;
    GM_ADDR curOutput = output;
    int32_t curTag = (tag << 15);
    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        // 执行kernel
        op.ProcessProxy<T>(curInput, curOutput, curCount, maxCountPerLoop, curTag, len);

        countLeft -= curCount;
        curInput += curSize;
        curOutput += curSize;
        curTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
    op.TailCounter();
}
