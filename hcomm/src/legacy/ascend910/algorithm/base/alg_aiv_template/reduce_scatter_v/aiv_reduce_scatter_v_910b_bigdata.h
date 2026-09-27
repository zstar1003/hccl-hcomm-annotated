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

class AivReduceScatterVBig910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterVBig910B() {}
    template <typename T>
    __aicore__ inline void
    ReduceWithFlagWrap(__gm__ T* cclGmSelf, __gm__ T* cclGmOther, uint64_t count, int32_t targetRank, int32_t tag);

    template <typename T>
    __aicore__ inline void Process(
        GM_ADDR input, GM_ADDR output, uint64_t selfOffset, uint64_t othersOffset, uint64_t len, uint64_t lenSelf,
        uint64_t maxCount, int32_t tagLeft, int32_t tagSelf, ExtraArgs& extraArgs);
};

template <typename T>
__aicore__ inline void AivReduceScatterVBig910B::ReduceWithFlagWrap(
    __gm__ T* cclGmSelf, __gm__ T* cclGmOther, uint64_t count, int32_t targetRank, int32_t tag)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);

    while (true) {
        if (processedBatchCount >= CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE)) {
            break;
        }

#ifndef OPEN_HCCL_TEST
        int64_t localFlagValue = CountWait(rank_, rank_) - tag;
        int64_t RemoteFlagValue = CountWait(targetRank, rank_) - tag;
#else
        LocalTensor<int32_t> localFlagX = flagInQue.AllocTensor<int32_t>();
        LocalTensor<int32_t> localFlagY = flagInQue.AllocTensor<int32_t>();
        int64_t localFlagValue = GetSignalValueWithExpected(
                                     (int32_t*)(GM_OUT[rank_] + countOffset + rank_ * FLAG_SIZE), localFlagX,
                                     CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE) + tag)
                                 - tag;
        int64_t RemoteFlagValue = GetSignalValueWithExpected(
                                      (int32_t*)(GM_OUT[targetRank] + countOffset + rank_ * FLAG_SIZE), localFlagY,
                                      CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE) + tag)
                                  - tag;
        flagInQue.FreeTensor(localFlagX);
        flagInQue.FreeTensor(localFlagY);
#endif

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
__aicore__ inline void AivReduceScatterVBig910B::Process(
    GM_ADDR input, GM_ADDR output, uint64_t selfOffset, uint64_t othersOffset, uint64_t len, uint64_t lenSelf,
    uint64_t maxCount, int32_t tagLeft, int32_t tagSelf, ExtraArgs& extraArgs)
{
    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = blockIdx_ >= rankSize_ ? blockIdx_ - rankSize_ : blockIdx_;

    __gm__ T* inputGm = (__gm__ T*)(input + othersOffset);
    __gm__ T* inputGmSelf = (__gm__ T*)(input + selfOffset);
    __gm__ T* outputGmSelf = (__gm__ T*)(output + selfOffset);

    __gm__ T* cclGmSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGmOther = (__gm__ T*)(GM_IN[targetRank]);

    if (blockIdx_ < blockNumPerGroup) {
        int32_t inputOffset = extraArgs.sendDispls[targetRank];
        int32_t cclGmSelfOffset = targetRank * maxCount;

        if (targetRank != rank_) {
            CpGM2GMWithFlagWrap(cclGmSelf + cclGmSelfOffset, inputGm + inputOffset, len, targetRank, 8, tagLeft);
            // 确定对端已经拉走数据
            pipe_barrier(PIPE_ALL);
            Wait(tagLeft, targetRank, AivNotifyType::DataSignal);
        }
    } else if (targetRank != rank_) {
        uint32_t cclGmOtherOffset = rank_ * maxCount;
        ReduceWithFlagWrap(outputGmSelf, cclGmOther + cclGmOtherOffset, lenSelf, targetRank, tagSelf);

        pipe_barrier(PIPE_ALL);
        // 通知对端已把数据拉走
        Record(tagSelf, targetRank, AivNotifyType::DataSignal);
        pipe_barrier(PIPE_ALL);
        // 通知本端已相加
        RecordNv1(tagSelf, rank_);
    } else {
        int32_t inputOffset = extraArgs.sendDispls[rank_];
        CpGM2GMWithFlagWrap(outputGmSelf, inputGmSelf + inputOffset, lenSelf, rank_, 8, tagSelf);
        // 确认已加完
        pipe_barrier(PIPE_ALL);
        Wait1vN((rankSize_ - 1) * tagSelf, CommPattern::intraRank);
    }
    return;
}

template <typename T>
__aicore__ inline void aiv_reduce_scatter_v_910b_bigdata(EXTERN_KERNEL_ARGS_DEF)
{
    AivReduceScatterVBig910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    uint64_t countLeft;
    uint64_t maxCountPerLoop = bufferSize / UB_ALIGN_SIZE * UB_ALIGN_SIZE / rankSize / sizeof(T);
    if (op.blockIdx_ >= rankSize) {
        countLeft = extraArgs.sendCounts[op.blockIdx_ - rankSize];
    } else {
        countLeft = extraArgs.sendCounts[op.blockIdx_];
    }
    uint64_t countSelf = extraArgs.sendCounts[rank];
    GM_ADDR curInput = input;
    GM_ADDR curOutput = output;

    int32_t curTagLeft = (tag << TAG_MOVE_LEFT_BITS);
    int32_t curTagSelf = curTagLeft;

    uint64_t selfOffset = 0;
    uint64_t othersOffset = 0;
    while (countLeft > 0 || countSelf > 0) {
        if (op.blockIdx_ == rank || (op.blockIdx_ >= rankSize && countSelf <= 0)
            || (op.blockIdx_ < rankSize && countLeft <= 0)) {
            break;
        }
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curCountSelf = (countSelf > maxCountPerLoop) ? maxCountPerLoop : countSelf;
        uint64_t curSize = curCount * sizeof(T);
        uint64_t curSizeSelf = curCountSelf * sizeof(T);
        // 执行kernel
        op.Process<T>(
            curInput, curOutput, selfOffset, othersOffset, curCount, curCountSelf, maxCountPerLoop, curTagLeft,
            curTagSelf, extraArgs);
        countLeft -= curCount;
        countSelf -= curCountSelf;
        othersOffset += curSize;
        selfOffset += curSizeSelf;
        curTagLeft += curSize / UB_DB_DATA_BATCH_SIZE + 1;
        curTagSelf += curSizeSelf / UB_DB_DATA_BATCH_SIZE + 1;
    }
    op.TailCounter();
}
