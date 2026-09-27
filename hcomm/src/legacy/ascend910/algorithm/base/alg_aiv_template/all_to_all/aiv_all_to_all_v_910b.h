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

class AivAll2AllV910B : public AivCommBase {
public:
    __aicore__ inline AivAll2AllV910B(bool isAlltoAllVC) : isAlltoAllVC_(isAlltoAllVC) {}

    template <typename T>
    __aicore__ inline void
    Process(GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t bufferSize, ExtraArgs& extraArgs);

    template <typename T>
    __aicore__ inline void ProcessAllToAllV910B(
        GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t bufferSize, ExtraArgs& extraArgs,
        bool isAlltoAllVC = false);

private:
    bool isAlltoAllVC_;
};

template <typename T>
__aicore__ inline void
AivAll2AllV910B::Process(GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t bufferSize, ExtraArgs& extraArgs)
{
    ProcessAllToAllV910B<T>(input, output, tag, bufferSize, extraArgs, isAlltoAllVC_);
}

template <typename T>
__aicore__ inline void AivAll2AllV910B::ProcessAllToAllV910B(
    GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t bufferSize, ExtraArgs& extraArgs, bool isAlltoAllVC)
{
    uint32_t targetRank = (blockIdx_ >= rankSize_ ? blockIdx_ - rankSize_ : blockIdx_); // 0-2*rankSize

    // 每张卡的CCLBuffer大小为bufferSize，平均分给ranksize块，每块的大小
    uint64_t avgBufferSize = bufferSize / rankSize_;
    uint64_t maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    uint64_t maxLoopCount = avgBufferSize / sizeof(T) / maxCountPerLoop; // 一块小CCLbuffer能够搬运几次ub大小才填满
    uint64_t avgBufferCount = maxLoopCount * maxCountPerLoop;

    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[targetRank]);
    tag = tag << TAG_MOVE_LEFT_BITS;

    if (blockIdx_ < rankSize_) { // 前rankSize个aiv负责userin->cclin
        uint64_t localSendOffset = 0;
        if (isAlltoAllVC) {
            for (uint32_t i = 0; i < blockIdx_; i++) {
                localSendOffset += extraArgs.sendCountMatrix[rank_ * rankSize_ + i];
            }
        } else {
            localSendOffset = extraArgs.sendDispls[targetRank];
        }
        uint64_t localSendCount = isAlltoAllVC ? extraArgs.sendCountMatrix[rank_ * rankSize_ + blockIdx_] :
                                                 extraArgs.sendCounts[targetRank];
        uint64_t localRecvOffset = avgBufferCount * blockIdx_; // userin搬到ccl的偏移

        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM + localSendOffset, localSendCount);
        // ccl只有avgBufferCount大小，可能小于localSendCount
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(cclGMSelf + localRecvOffset, avgBufferCount);

        uint64_t flushFrequency = 8;
        uint64_t curBatchCount = 0;
        uint64_t curInOffset = 0;
        uint64_t curOutOffset = 0;
        while (localSendCount > 0) {
            if ((curBatchCount % maxLoopCount == 0) && (curBatchCount != 0)) {
                PipeBarrier<PIPE_ALL>();
                // 告诉写数据的aiv读完一小块cclbuffer
                curOutOffset = 0;
                Wait(tag + curBatchCount, targetRank, AivNotifyType::DataSignal);
            }
            curBatchCount += 1;

            uint64_t curCount = localSendCount > maxCountPerLoop ? maxCountPerLoop : localSendCount;

            LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
            DataCopyGM2UB(localIn, inputGT[curInOffset], curCount);
            inOutQue.EnQue(localIn);
            LocalTensor<T> localOut = inOutQue.DeQue<T>();

            // 如果curBatchCount超过了maxLoopCount，则重新从这小块ccl的起始开始放
            DataCopyUB2GM(outputGT[curOutOffset], localOut, curCount);
            inOutQue.FreeTensor(localOut);

            localSendCount -= curCount;
            curInOffset += curCount;
            curOutOffset += curCount;

            if (curBatchCount % flushFrequency == 0 || curBatchCount % maxLoopCount == 0 || localSendCount == 0) {
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

                CountRecord(tag, curBatchCount, blockIdx_);
            }
        }
        PipeBarrier<PIPE_ALL>();
        Wait(tag, targetRank, AivNotifyType::Done);
        PipeBarrier<PIPE_ALL>();

        // 检查其他卡对本卡该aiv的依赖，清空计数

    } else {                                                // 后rankSize个aiv负责cclother->usrout
        uint64_t remoteSendOffset = avgBufferCount * rank_; // ccl读到usrout的偏移

        // 本端output接收远端ccl的数据偏移，远端卡号为GetBlockIdx()，可能为本rank
        uint64_t localRecvOffset = 0; // 本端output接收远端ccl的数据偏移，目标远端卡号为GetBlockIdx()，可能为本rank
        if (isAlltoAllVC) {
            for (uint32_t i = 0; i < targetRank; i++) {
                localRecvOffset += extraArgs.sendCountMatrix[i * rankSize_ + rank_];
            }
        } else {
            localRecvOffset = extraArgs.recvDispls[targetRank];
        }

        // 远端ccl发送给本端output的数据量，远端可能为本rank
        uint64_t remoteSendCount = isAlltoAllVC ? extraArgs.sendCountMatrix[targetRank * rankSize_ + rank_] :
                                                  extraArgs.recvCounts[targetRank];
        uint64_t remoteSendSize = remoteSendCount * sizeof(T);

        uint64_t processedBatchCount = 0;

#ifdef OPEN_HCCL_TEST
        int32_t iter = 1;
#endif
        while (true) {
            if (processedBatchCount >= CeilDiv(remoteSendSize, UB_DB_DATA_BATCH_SIZE)) {
                break;
            }
#ifndef OPEN_HCCL_TEST
            int32_t localFlag = CountWait(targetRank, rank_);
#else
            int32_t flagValue = std::min(CeilDiv(remoteSendSize, UB_DB_DATA_BATCH_SIZE), maxLoopCount * iter++);
            LocalTensor<int32_t> localFlagX = flagInQue.AllocTensor<int32_t>();
            int32_t localFlag = GetSignalValueWithExpected(
                (int32_t*)(GM_OUT[targetRank] + countOffset + rank_ * FLAG_SIZE), localFlagX, flagValue + tag);
            flagInQue.FreeTensor(localFlagX);
#endif

            if (localFlag < tag) {
                continue;
            }

            uint64_t preparedBatchCount = localFlag - tag;

            // 还没有数据准备好，或者没有新的数据准备好，继续等
            if (preparedBatchCount == 0 || processedBatchCount >= preparedBatchCount) {
                continue;
            }

            uint64_t curSize = (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;
            if (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > remoteSendSize) {
                curSize = remoteSendSize - processedBatchCount * UB_DB_DATA_BATCH_SIZE;
            }

            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

            uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
            uint64_t curProcessedInOffset = curProcessedOffset % avgBufferCount;
            CpGM2GM(
                outputGM + localRecvOffset + curProcessedOffset, cclGMOther + remoteSendOffset + curProcessedInOffset,
                curSize / sizeof(T));

            processedBatchCount = preparedBatchCount;

            // 如果processedBatchCount超过了maxLoopCount，则需要做一次同步，并且重新开始用这块CCL
            if (processedBatchCount % maxLoopCount == 0) {
                PipeBarrier<PIPE_ALL>();
                // 告诉写数据的aiv读完一小块cclbuffer
                Record(tag + processedBatchCount, targetRank, AivNotifyType::DataSignal);
            }
        }

        // 通知对端，自己已经把对端的那片数据拉回来了
        PipeBarrier<PIPE_ALL>();
        Record(tag, targetRank, AivNotifyType::Done);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void aiv_all_to_all_v_910b(EXTERN_KERNEL_ARGS_DEF)
{
    AivAll2AllV910B op(false);
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    op.Process<T>(input, output, tag, bufferSize, extraArgs);
    op.TailCounter();
}

template <typename T>
__aicore__ inline void aiv_all_to_all_vc_910b(EXTERN_KERNEL_ARGS_DEF)
{
    AivAll2AllV910B op(true);
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    op.Process<T>(input, output, tag, bufferSize, extraArgs);
    op.TailCounter();
}
