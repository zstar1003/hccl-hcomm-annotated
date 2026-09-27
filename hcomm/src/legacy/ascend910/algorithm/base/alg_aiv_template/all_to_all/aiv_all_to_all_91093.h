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

class AivAll2All91093 : public AivCrossNode91093Base {
public:
    __aicore__ inline AivAll2All91093() {}

    template <typename T>
    __aicore__ inline void Process(
        GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
        uint64_t bufferSize, uint64_t len);
};

template <typename T>
__aicore__ inline void AivAll2All91093::Process(
    GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
    uint64_t bufferSize, uint64_t len)
{
    // 每张卡的CCLBuffer大小为bufferSize，平均分给ranksize块，每块的大小
    uint64_t avgBufferCount = bufferSize / rankSize_ / sizeof(T);

    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)buffIn0;

    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    uint64_t remainCount = len;
    uint64_t curOffset = 0;
    uint32_t bufferLoopNum = (len + avgBufferCount - 1) / avgBufferCount;
    for (uint32_t loop = 0; loop < bufferLoopNum; loop++) {
        uint64_t curCount = remainCount > avgBufferCount ? avgBufferCount : remainCount;
        PipeBarrier<PIPE_ALL>();

        // 每次最多处理avgBufferCount
        for (uint32_t i = 0; i < numTargets; i++) {
            uint64_t localSendOffset = len * targetRanks[i];
            uint64_t localRecvOffset = avgBufferCount * targetRanks[i];
            CpGM2GM(cclGMSelf + localRecvOffset, inputGM + localSendOffset + curOffset, curCount);
        }

        PipeBarrier<PIPE_ALL>();

        // localcopy后的同步
        BatchRecordWait(curTag, buffersOut);

        PipeBarrier<PIPE_ALL>();

        // 读对端ccl到usrout
        for (uint32_t i = 0; i < numTargets; i++) {
            __gm__ T* cclGMOther = (__gm__ T*)(buffersIn[i]);

            uint64_t remoteSendOffset = avgBufferCount * rank_;
            uint64_t localRecvOffset = len * targetRanks[i];
            CpGM2GM(outputGM + localRecvOffset + curOffset, cclGMOther + remoteSendOffset, curCount);
        }

        PipeBarrier<PIPE_ALL>();

        // read后的同步
        BatchRecordWait(curTag, buffersOut, AivNotifyType::DataSignal);

        curTag += 1;
        curOffset += curCount;
        remainCount -= curCount;
    }

    // 最后一个核做localcopy
    if (blockIdx_ == numBlocks_ - 1) {
        CpGM2GM(outputGM + rank_ * len, inputGM + rank_ * len, len);
    }
}

template <typename T>
__aicore__ inline void aiv_all_to_all_91093(KERNEL_ARGS_DEF)
{
    AivAll2All91093 op;
    op.Init(buffOut0, buffOut1, rank, rankSize, tag, numBlocks, isOpBase, true);
    op.InitOpCounter(headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter);
    op.HeadCounter();
    op.Process<T>(buffIn0, buffOut0, buffOut1, input, output, tag, bufferSize, len);
    op.TailCounter();
}
