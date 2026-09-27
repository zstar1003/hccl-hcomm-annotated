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

class AivReduceScatterBigGraph910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterBigGraph910B() {}

    template <typename T>
    __aicore__ inline void ReduceWithFlagWrap(__gm__ T* cclGmSelf, __gm__ T* cclGmOther, uint64_t count, int32_t tag);

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void
AivReduceScatterBigGraph910B::ReduceWithFlagWrap(__gm__ T* cclGmSelf, __gm__ T* cclGmOther, uint64_t count, int32_t tag)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);

    while (true) {
        if (processedBatchCount >= CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE)) {
            break;
        }

#ifndef OPEN_HCCL_TEST
        int32_t localFlag = CountWait(rank_, rank_);
#else
        LocalTensor<int32_t> localFlagX = flagInQue.AllocTensor<int32_t>();
        int32_t localFlag = GetSignalValueWithExpected(
            (int32_t*)(GM_OUT[rank_] + countOffset + rank_ * FLAG_SIZE), localFlagX,
            CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE) + tag);
        flagInQue.FreeTensor(localFlagX);
#endif
        uint64_t localFlagValue;
        if (localFlag <= tag) {
            continue;
        } else {
            localFlagValue = localFlag - tag;
        }
        if (localFlagValue == 0) {
            continue;
        }
        uint64_t preparedBatchCount = localFlagValue;
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
__aicore__ inline void AivReduceScatterBigGraph910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t avgLengthPerSlice = len;
    uint32_t avgSizePerSlice = avgLengthPerSlice * sizeof(T);

    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = blockIdx_ >= rankSize_ ? blockIdx_ - rankSize_ : blockIdx_; // 0-7

    __gm__ T* inputGm = (__gm__ T*)input;
    __gm__ T* outputGm = (__gm__ T*)output;
    __gm__ T* cclGmSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGmOther = (__gm__ T*)(GM_IN[targetRank]);

    int32_t inputOffset = targetRank * avgLengthPerSlice;

    if (blockIdx_ == rank_) {
        // 拷贝相应的数据到output
        uint64_t freq = avgSizePerSlice >= 2 * 1024 * 1024 ? 4 : 16;
        CpGM2GMWithFlagWrap(outputGm, inputGm + inputOffset, avgLengthPerSlice, rank_, freq, tag);

        // 确认本端全部reduce完成
        Wait1vN((rankSize_ - 1) * tag, CommPattern::intraRank);
    } else if (targetRank != rank_) {
        // 确定可以从对端拉数据
        Record(tag, targetRank, AivNotifyType::ACK);
        Wait(tag, targetRank, AivNotifyType::ACK);

        uint32_t cclGmOtherOffset = rank_ * avgLengthPerSlice;
        ReduceWithFlagWrap(outputGm, cclGmOther + cclGmOtherOffset, len, tag);

        // 通知对端数据已经拉走
        // 是否要加个check
        PipeBarrier<PIPE_ALL>();
        Record(tag, targetRank, AivNotifyType::DataSignal);
        Wait(tag, targetRank, AivNotifyType::DataSignal);
        PipeBarrier<PIPE_ALL>();
        // 通知本端reduce完成
        RecordNv1(tag, rank_);
    }
    return;
}

template <typename T>
__aicore__ inline void aiv_reduce_scatter_910b_bigdata_graph(KERNEL_ARGS_DEF)
{
    AivReduceScatterBigGraph910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    tag = tag << TAG_MOVE_LEFT_BITS;
    op.Process<T>(input, output, len, tag);
    op.TailCounter();
}
