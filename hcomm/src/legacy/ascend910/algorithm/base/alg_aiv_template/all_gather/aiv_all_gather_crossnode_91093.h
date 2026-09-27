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

class AivAllGatherCrossNode91093 : public AivCrossNode91093Base {
public:
    __aicore__ inline AivAllGatherCrossNode91093() {}

    template <typename T>
    __aicore__ inline void Process(
        GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
        uint64_t bufferCount, uint64_t len);
};

template <typename T>
__aicore__ inline void AivAllGatherCrossNode91093::Process(
    GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
    uint64_t bufferCount, uint64_t len)
{
    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)buffIn0;

    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    uint64_t curOffset = 0;
    uint64_t curCount;
    uint64_t curBlockOffset;
    uint32_t bufferLoopNum = (len + bufferCount - 1) / bufferCount;
    uint64_t curCCLOffset = blockOffsetMid >= blockOffsetTail ? blockOffsetMid : blockOffsetTail;

    for (uint32_t loop = 0; loop < bufferLoopNum; loop++) {
        if (loop == bufferLoopNum - 1) { // 最后一轮ccl填充
            curCount = countTail;
            curBlockOffset = blockOffsetTail;
        } else {
            curCount = countMid;
            curBlockOffset = blockOffsetMid;
        }

        PipeBarrier<PIPE_ALL>();

        if (localCopyCores) {
            CpGM2GM(cclGMSelf + curCCLOffset, inputGM + curOffset + curBlockOffset, curCount);
            PipeBarrier<PIPE_ALL>();
        }

        // 首次卡间同步，多等一（Case1/2目标核做完localcopy后告知其他卡所有remotecopy的核它完成了）
        SingleRecordBatchWait(curTag, buffersOut, localCopyCores);

        PipeBarrier<PIPE_ALL>();

        // 读对端ccl到usrout
        for (uint32_t i = 0; i < numTargets; i++) {
            __gm__ T* cclGMOther = (__gm__ T*)(buffersIn[i]);

            uint64_t localRecvOffset = len * targetRanks[i];
            CpGM2GM(outputGM + localRecvOffset + curOffset + curBlockOffset, cclGMOther + curCCLOffset, curCount);
        }

        PipeBarrier<PIPE_ALL>();

        // 结尾卡间同步，多等多（所有卡等待其他卡的remotecopy完成）
        BatchRecordWait(curTag, buffersOut, AivNotifyType::DataSignal);

        if (loop != bufferLoopNum - 1) {
            // 卡内核间同步，避免下一轮last core做localcopy时抢跑
            BatchRecordSingleWaitCoreLevel(curTag, localCopyCores);
            curTag += 1;
            curOffset += bufferCount;
        }
    }
}

template <typename T>
__aicore__ inline void aiv_all_gather_crossnode_91093(KERNEL_ARGS_DEF_A3)
{
    AivAllGatherCrossNode91093 op;

    // 每张卡的CCLBuffer大小为bufferSize; bufferSize中能装下的数据个数为bufferCount
    uint64_t bufferCount = (uint64_t)bufferSize / sizeof(T);

    op.Init<T>(buffOut0, buffOut1, rank, rankSize, bufferCount, len, reduceOp, tag, step, numBlocks, true);
    op.InitOpCounter(headCountMem, tailCountMem, addOneMem, SIZE_OF_INT32, isEnableCounter);
    op.HeadCounter();
    op.Process<T>(buffIn0, buffOut0, buffOut1, input, output, tag, bufferCount, len);
    op.TailCounter();
}
