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

class AivAll2All91093Single : public AivCommBase {
public:
    __aicore__ inline AivAll2All91093Single() {}

    template <typename T>
    __aicore__ inline void Process(
        GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t remoteSendOffset, uint64_t localRecvOffset,
        uint64_t remoteSendCount);

    template <typename T>
    __aicore__ inline void ProcessSmall(
        GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t remoteSendOffset, uint64_t localRecvOffset,
        uint64_t remoteSendCount);

    template <typename T>
    __aicore__ inline void ProcessBig(
        GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t remoteSendOffset, uint64_t localRecvOffset,
        uint64_t remoteSendCount);
};

template <typename T>
__aicore__ inline void AivAll2All91093Single::Process(
    GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t remoteSendOffset, uint64_t localRecvOffset,
    uint64_t remoteSendCount)
{
    if (remoteSendCount * sizeof(T) <= AIV_A3_ALL_TO_ALL_GRAPH_GUIYI_SIZE) {
        ProcessSmall<T>(input, output, tag, remoteSendOffset, localRecvOffset, remoteSendCount);
    } else {
        ProcessBig<T>(input, output, tag, remoteSendOffset, localRecvOffset, remoteSendCount);
    }
}

template <typename T>
__aicore__ inline void AivAll2All91093Single::ProcessBig(
    GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t remoteSendOffset, uint64_t localRecvOffset,
    uint64_t remoteSendCount)
{
    uint32_t blockNumPerGroup = numBlocks_ / rankSize_;
    uint32_t blockIdxInGroup = blockIdx_ % blockNumPerGroup;
    uint32_t dstRank = blockIdx_ / blockNumPerGroup;
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);

    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[dstRank]);

    // 使用96个flag
    uint64_t blockRecvCount = 0;
    uint64_t blockRecvOffset = 0;
    CalBlockCountAndOffset(
        remoteSendCount, blockNumPerGroup, blockIdxInGroup, padCount, blockRecvCount, blockRecvOffset);

    // 确认对端已经准备好
    Record(tag, dstRank, AivNotifyType::ACK, blockIdxInGroup);
    Wait(tag, dstRank, AivNotifyType::ACK, blockIdxInGroup);
    PipeBarrier<PIPE_ALL>();

    CpGM2GM(
        outputGM + localRecvOffset + blockRecvOffset, cclGMOther + remoteSendOffset + blockRecvOffset, blockRecvCount);
    PipeBarrier<PIPE_ALL>();

    // 确认对端已经读完本端
    Record(tag, dstRank, AivNotifyType::DataSignal, blockIdxInGroup);
    Wait(tag, dstRank, AivNotifyType::DataSignal, blockIdxInGroup);

    return;
}

template <typename T>
__aicore__ inline void AivAll2All91093Single::ProcessSmall(
    GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t remoteSendOffset, uint64_t localRecvOffset,
    uint64_t remoteSendCount)
{
    uint32_t blockNumPerGroup = numBlocks_ / rankSize_;
    uint32_t blockIdxInGroup = blockIdx_ % blockNumPerGroup;
    uint32_t dstRank = blockIdx_ / blockNumPerGroup;
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);

    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;
    bool ifPingpong = (tag % 2 == 0);

    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[dstRank] + dataOffset);

    // 使用96个flag
    uint64_t blockRecvCount = 0;
    uint64_t blockRecvOffset = 0;
    CalBlockCountAndOffset(
        remoteSendCount, blockNumPerGroup, blockIdxInGroup, padCount, blockRecvCount, blockRecvOffset);

    // localcopy
    CpGM2GM(cclGMSelf + localRecvOffset + blockRecvOffset, inputGM + localRecvOffset + blockRecvOffset, blockRecvCount);

    PipeBarrier<PIPE_ALL>();

    // 卡间同步，确认对端已经准备好
    Record(tag, dstRank, AivNotifyType::DataSignal, blockIdxInGroup, ifPingpong);
    Wait(tag, dstRank, AivNotifyType::DataSignal, blockIdxInGroup, ifPingpong);

    PipeBarrier<PIPE_ALL>();

    CpGM2GM(
        outputGM + localRecvOffset + blockRecvOffset, cclGMOther + remoteSendOffset + blockRecvOffset, blockRecvCount);

    return;
}

template <typename T>
__aicore__ inline void aiv_all_to_all_vc_91093_single_graph(KERNEL_ARGS_DEF, ExtraArgs* extraArgs)
{
    AivAll2All91093Single op;
    op.Init(KERNEL_CLASS_INIT, true);

    uint32_t blockNumPerGroup = op.numBlocks_ / rankSize;
    uint32_t dstRank = op.blockIdx_ / blockNumPerGroup;

    uint64_t remoteSendOffset = 0;
    for (uint32_t i = 0; i < rank; i++) {
        remoteSendOffset += extraArgs->sendCountMatrix[dstRank * rankSize + i];
    }
    uint64_t localRecvOffset = 0;
    for (uint32_t i = 0; i < dstRank; i++) {
        localRecvOffset += extraArgs->sendCountMatrix[i * rankSize + rank];
    }
    uint64_t remoteSendCount = extraArgs->sendCountMatrix[dstRank * rankSize + rank];

    op.HeadCounter();
    op.ProcessBig<T>(input, output, tag, remoteSendOffset, localRecvOffset, remoteSendCount);
    op.TailCounter();
}

template <typename T>
__aicore__ inline void aiv_all_to_all_91093_single(KERNEL_ARGS_DEF)
{
    AivAll2All91093Single op;
    op.Init(KERNEL_CLASS_INIT, true);

    uint32_t blockNumPerGroup = op.numBlocks_ / rankSize;
    uint32_t dstRank = op.blockIdx_ / blockNumPerGroup;

    uint64_t remoteSendOffset = rank * len;
    uint64_t localRecvOffset = dstRank * len;
    uint64_t remoteSendCount = len;

    op.HeadCounter();
    op.Process<T>(input, output, tag, remoteSendOffset, localRecvOffset, remoteSendCount);
    op.TailCounter();
}

__aicore__ inline void sk_all_to_all_91093_single(SUPERKERNEL_ARGS_DEF)
{
    AivAll2All91093Single op;
    op.Init(SUPERKERNEL_CLASS_INIT, AIV_A3_ALL_TO_ALL_GRAPH_GUIYI_SIZE);
    uint32_t blockNumPerGroup = op.numBlocks_ / op.rankSize_;
    uint32_t dstRank = op.blockIdx_ / blockNumPerGroup;

    uint64_t remoteSendOffset = op.rank_ * op.len_;
    uint64_t localRecvOffset = dstRank * op.len_;
    uint64_t remoteSendCount = op.len_;
#ifdef HCCL_DTYPE_INT8
    op.Process<int8_t>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_INT16
    op.Process<int16_t>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_INT32
    op.Process<int32_t>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_FP16
    op.Process<half>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_FP32
    op.Process<float>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_BFP16
    op.Process<bfloat16_t>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_UINT8
    op.Process<uint8_t>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_UINT16
    op.Process<uint16_t>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_UINT32
    op.Process<uint32_t>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_INT64
    op.Process<int64_t>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_UINT64
    op.Process<uint64_t>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#elif defined HCCL_DTYPE_FP64
    op.Process<double>(input, output, op.tag_, remoteSendOffset, localRecvOffset, remoteSendCount);
#else
#endif
}
