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

class AivAllGatherVBig910B : public AivCommBase {
public:
    __aicore__ inline AivAllGatherVBig910B() {}

    template <typename T>
    __aicore__ inline void
    MemcpyWithFlagWrap(__gm__ T* cclGmSelf, __gm__ T* cclGmOther, uint64_t count, int32_t dstRank, int32_t tag);

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t curCount, ExtraArgs& extraArgs, int32_t tag);
};

template <typename T>
__aicore__ inline void AivAllGatherVBig910B::MemcpyWithFlagWrap(
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

        uint64_t curSize = (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > avgSizePerSlice) ?
                               avgSizePerSlice - processedBatchCount * UB_DB_DATA_BATCH_SIZE :
                               (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

        uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
        CpGM2GM(cclGmSelf + curProcessedOffset, cclGmOther + curProcessedOffset, curSize / sizeof(T));

        processedBatchCount = preparedBatchCount;
    }
}

template <typename T>
__aicore__ inline void
AivAllGatherVBig910B::Process(GM_ADDR input, GM_ADDR output, uint64_t curCount, ExtraArgs& extraArgs, int32_t tag)
{
    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = blockIdx_ >= rankSize_ ? blockIdx_ - rankSize_ : blockIdx_;

    __gm__ T* inputGm = (__gm__ T*)input;
    __gm__ T* outputGm = (__gm__ T*)output;
    __gm__ T* cclGmSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGmOther = (__gm__ T*)(GM_IN[targetRank]);

    if (blockIdx_ < blockNumPerGroup) {
        if (blockIdx_ == rank_) { // 把数据从UserIn 搬运到 CCLIn，同时检测有多少个核在搬运这个数据
            CpGM2GMWithFlagWrap(cclGmSelf, inputGm, curCount, rank_, 8, tag);
            // 所有对端都取走数据
            pipe_barrier(PIPE_ALL);
            Wait1vN((rankSize_ - 1) * tag, CommPattern::interRank, true);
        } else {
            MemcpyWithFlagWrap(outputGm + extraArgs.recvDispls[targetRank], cclGmOther, curCount, targetRank, tag);
            pipe_barrier(PIPE_ALL);
            RecordNv1(tag, targetRank);
        }
    } else {
        CpGM2GM(outputGm + extraArgs.recvDispls[rank_], inputGm, curCount);
    }
    return;
}

template <typename T>
__aicore__ inline void aiv_all_gather_v_910b_bigdata(EXTERN_KERNEL_ARGS_DEF)
{
    AivAllGatherVBig910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    uint64_t maxCountPerLoop = bufferSize / UB_ALIGN_SIZE * UB_ALIGN_SIZE / sizeof(T);
    uint64_t countLeft;
    if (op.blockIdx_ < rankSize) {
        countLeft = extraArgs.recvCounts[op.blockIdx_];
    } else {
        countLeft = extraArgs.recvCounts[rank];
    }

    GM_ADDR curInput = input;
    GM_ADDR curOutput = output;
    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);

    while (countLeft > 0) {
        uint64_t curCount = countLeft > maxCountPerLoop ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        // 执行kernel
        op.Process<T>(curInput, curOutput, curCount, extraArgs, curTag);

        countLeft -= curCount;
        curInput += curSize;
        curOutput += curSize;
        curTag += maxCountPerLoop * sizeof(T) / UB_DB_DATA_BATCH_SIZE + 1; // 确认按最大值增加tag的合理性
    }
    op.TailCounter();
}
