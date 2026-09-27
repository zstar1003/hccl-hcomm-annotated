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
#include "aiv_crossnode_91093_base.h"

using namespace AscendC;

class AivReduceScatterCrossNode91093 : public AivCrossNode91093Base {
public:
    __aicore__ inline AivReduceScatterCrossNode91093() {}

    template <typename T>
    __aicore__ inline void Process(
        GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
        uint64_t bufferSize, uint64_t len);
};

template <typename T>
__aicore__ inline void AivReduceScatterCrossNode91093::Process(
    GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
    uint64_t avgBufferCount, uint64_t len)
{
    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)buffIn0;

    // RS需要先保证input->output完成，再做remote copy进行原子累加
    if (localCopyCores) {
        CpGM2GM(outputGM + blockOffset, inputGM + rank_ * len + blockOffset, countPerCore);
        PipeBarrier<PIPE_ALL>();
    }

    // localcopy后的卡内核间同步，多等一（Case1/2目标核做完localcopy后告知本卡其他核）
    SingleRecordBatchWaitCoreLevel(tag, localCopyCores);

    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    uint64_t curOffset = 0;
    uint64_t curCount;
    uint64_t curBlockOffset;
    uint64_t curBlockCclOffset = blockOffsetMid > blockOffsetTail ? blockOffsetMid : blockOffsetTail;
    uint32_t bufferLoopNum = (len + avgBufferCount - 1) / avgBufferCount;

    for (uint32_t loop = 0; loop < bufferLoopNum; loop++) {
        if (loop == bufferLoopNum - 1) { // 最后一轮ccl填充
            curCount = countTail;
            curBlockOffset = blockOffsetTail;
        } else {
            curCount = countMid;
            curBlockOffset = blockOffsetMid;
        }
        PipeBarrier<PIPE_ALL>();

        // localcopy
        for (uint32_t i = 0; i < numTargets; i++) {
            if (targetRanks[i] != rank_) {
                uint64_t localSendOffset = len * targetRanks[i];
                uint64_t localRecvOffset = avgBufferCount * targetRanks[i];
                CpGM2GM(
                    cclGMSelf + localRecvOffset + curBlockCclOffset,
                    inputGM + localSendOffset + curOffset + curBlockOffset, curCount);
            }
        }

        PipeBarrier<PIPE_ALL>();

        // 首次卡间同步
        BatchRecordWait(curTag, buffersOut);

        PipeBarrier<PIPE_ALL>();

        // 读对端ccl到usrout
        for (uint32_t i = 0; i < numTargets; i++) {
            if (targetRanks[i] != rank_) {
                __gm__ T* cclGMOther = (__gm__ T*)(buffersIn[i]);
                uint64_t remoteSendOffset = avgBufferCount * rank_;
                CpGM2GM(
                    outputGM + curOffset + curBlockOffset, cclGMOther + remoteSendOffset + curBlockCclOffset, curCount,
                    true, reduceOp_);
            }
        }

        PipeBarrier<PIPE_ALL>();

        // 结尾卡间同步
        BatchRecordWait(curTag, buffersOut, AivNotifyType::DataSignal);

        curTag += 1;
        curOffset += avgBufferCount;
    }
}

template <typename T>
__aicore__ inline void aiv_reduce_scatter_crossnode_91093(KERNEL_ARGS_DEF_A3)
{
    AivReduceScatterCrossNode91093 op;

    // 每张卡的CCLBuffer大小为bufferSize，平均分给ranksize块，每块的大小
    uint64_t avgBufferCount = (uint64_t)bufferSize / rankSize / sizeof(T);

    op.Init<T>(buffOut0, buffOut1, rank, rankSize, avgBufferCount, len, reduceOp, tag, step, numBlocks, true);
    op.InitOpCounter(headCountMem, tailCountMem, addOneMem, SIZE_OF_INT32, isEnableCounter);
    op.HeadCounter();
    op.Process<T>(buffIn0, buffOut0, buffOut1, input, output, tag, avgBufferCount, len);
    op.TailCounter();
}
