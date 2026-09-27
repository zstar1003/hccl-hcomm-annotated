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

class AivAll2AllV91093 : public AivCrossNode91093Base {
public:
    __aicore__ inline AivAll2AllV91093() {}

    __aicore__ inline void BatchRecordWaitV(
        int32_t curTag, GM_ADDR* buffersOut, bool* needTx, bool* needRx, AivNotifyType notifyType = AivNotifyType::ACK);

    template <typename T>
    __aicore__ inline void Process(
        GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
        uint64_t bufferSize, ExtraArgsV2* extraArgs);
};

__aicore__ inline void AivAll2AllV91093::BatchRecordWaitV(
    int32_t curTag, GM_ADDR* buffersOut, bool* needTx, bool* needRx, AivNotifyType notifyType)
{
    // tx
    for (uint32_t i = 0; i < numTargets; i++) {
        if (!needTx[i]) {
            continue;
        }
        Record(curTag, buffersOut[i], notifyType);
    }
    // rx and clear
    for (uint32_t i = 0; i < numTargets; i++) {
        if (!needRx[i]) {
            continue;
        }
        Wait(curTag, targetRanks[i], notifyType);
    }
}

template <typename T>
__aicore__ inline void AivAll2AllV91093::Process(
    GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
    uint64_t bufferSize, ExtraArgsV2* extraArgs)
{
    // 每张卡的CCLBuffer大小为bufferSize，平均分给ranksize块，每块的大小
    uint64_t avgBufferCount = bufferSize / rankSize_ / sizeof(T);

    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)buffIn0;

    uint32_t cclReadyFlagOffset = 0;
    uint32_t finalAckFlagOffset = rankSize_ * FLAG_SIZE;

    // 准备参数，buffer地址和最大收发count
    uint64_t sendCounts[MAX_TARGET_NUM] = {};
    uint64_t recvCounts[MAX_TARGET_NUM] = {};
    uint64_t sendDispls[MAX_TARGET_NUM] = {};
    uint64_t recvDispls[MAX_TARGET_NUM] = {};
    uint64_t maxCount = 0;
    bool needSend[MAX_TARGET_NUM] = {0};
    bool needRead[MAX_TARGET_NUM] = {0};

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = targetRanks[i];
        sendCounts[i] = extraArgs->sendCounts[targetRank];
        recvCounts[i] = extraArgs->recvCounts[targetRank];
        sendDispls[i] = extraArgs->sendDispls[targetRank];
        recvDispls[i] = extraArgs->recvDispls[targetRank];

        maxCount = sendCounts[i] > maxCount ? sendCounts[i] : maxCount;
        maxCount = recvCounts[i] > maxCount ? recvCounts[i] : maxCount;
    }
    uint32_t bufferLoopNum = (maxCount + avgBufferCount - 1) / avgBufferCount;

    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    for (uint32_t loop = 0; loop < bufferLoopNum; loop++) {
        PipeBarrier<PIPE_ALL>();

        // 每次最多处理avgBufferCount
        for (uint32_t i = 0; i < numTargets; i++) {
            // 记录是否需要同步
            needSend[i] = (sendCounts[i] > 0);
            needRead[i] = (recvCounts[i] > 0);

            uint64_t localSendOffset = sendDispls[i];
            uint64_t localSendCount = sendCounts[i] > avgBufferCount ? avgBufferCount : sendCounts[i];
            uint64_t localRecvOffset = avgBufferCount * targetRanks[i];
            CpGM2GM(cclGMSelf + localRecvOffset, inputGM + localSendOffset, localSendCount);
            sendDispls[i] += localSendCount;
            sendCounts[i] -= localSendCount;
        }

        PipeBarrier<PIPE_ALL>();

        // localcopy后的同步
        BatchRecordWaitV(curTag, buffersOut, needSend, needRead);

        PipeBarrier<PIPE_ALL>();

        // 读对端ccl到usrout
        for (uint32_t i = 0; i < numTargets; i++) {
            __gm__ T* cclGMOther = (__gm__ T*)(buffersIn[i]);

            uint64_t remoteSendOffset = avgBufferCount * rank_;
            uint64_t localRecvOffset = recvDispls[i];
            uint64_t remoteSendCount = recvCounts[i] > avgBufferCount ? avgBufferCount : recvCounts[i];
            CpGM2GM(outputGM + localRecvOffset, cclGMOther + remoteSendOffset, remoteSendCount);
            recvDispls[i] += remoteSendCount;
            recvCounts[i] -= remoteSendCount;
        }

        PipeBarrier<PIPE_ALL>();

        // read后的同步
        BatchRecordWaitV(curTag, buffersOut, needRead, needSend, AivNotifyType::DataSignal);

        curTag += 1;
    }

    // 最后一个核做localcopy
    if (blockIdx_ == numBlocks_ - 1) {
        uint64_t sendCount = extraArgs->sendCounts[rank_];
        uint64_t sendOffset = extraArgs->sendDispls[rank_];
        uint64_t recvOffset = extraArgs->recvDispls[rank_];
        CpGM2GM(outputGM + recvOffset, inputGM + sendOffset, sendCount);
    }
}

template <typename T>
__aicore__ inline void aiv_all_to_all_v_91093(KERNEL_ARGS_DEF, ExtraArgsV2* extraArgs)
{
    AivAll2AllV91093 op;
    op.Init(buffOut0, buffOut1, rank, rankSize, tag, numBlocks, isOpBase, true);
    op.InitOpCounter(headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter);
    op.HeadCounter();
    op.Process<T>(buffIn0, buffOut0, buffOut1, input, output, tag, bufferSize, extraArgs);
    op.TailCounter();
}
