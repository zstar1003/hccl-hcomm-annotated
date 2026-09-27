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

class AivAll2AllVCGraph910B : public AivCommBase {
public:
    __aicore__ inline AivAll2AllVCGraph910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, int32_t tag, ExtraArgs& extraArgs);
};

template <typename T>
__aicore__ inline void AivAll2AllVCGraph910B::Process(GM_ADDR input, GM_ADDR output, int32_t tag, ExtraArgs& extraArgs)
{
    uint32_t targetRank = blockIdx_; // 0-rankSize

    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[targetRank]);

    // 共使用2组flag
    uint32_t initAckFlagOffset = 0;
    uint32_t finalAckFlagOffset = rankSize_ * FLAG_SIZE;

    // 本卡已进入算子，通知其他卡可以搬运，使用第1个flag
    Record(tag, targetRank, AivNotifyType::ACK);
    Wait(tag, targetRank, AivNotifyType::ACK);
    PipeBarrier<PIPE_ALL>();
    uint64_t remoteSendOffset = 0; // 远端usrin发送给本端output的数据偏移，远端卡号为GetBlockIdx()，可能为本rank
    for (uint32_t i = 0; i < rank_; i++) {
        remoteSendOffset += extraArgs.sendCountMatrix[targetRank * rankSize_ + i];
    }

    uint64_t localRecvOffset = 0; // 本端output接收远端usrin的数据偏移，目标远端卡号为GetBlockIdx()，可能为本rank
    for (uint32_t i = 0; i < targetRank; i++) {
        localRecvOffset += extraArgs.sendCountMatrix[i * rankSize_ + rank_];
    }

    // 远端ccl发送给本端output的数据量，远端可能为本rank
    uint64_t remoteSendCount = extraArgs.sendCountMatrix[targetRank * rankSize_ + rank_];

    CpGM2GM(outputGM + localRecvOffset, cclGMOther + remoteSendOffset, remoteSendCount);
    PipeBarrier<PIPE_ALL>();

    Record(tag, targetRank, AivNotifyType::DataSignal);
    Wait(tag, targetRank, AivNotifyType::DataSignal);
    return;
}

template <typename T>
__aicore__ inline void aiv_all_to_all_vc_910b_graph(EXTERN_KERNEL_ARGS_DEF)
{
    AivAll2AllVCGraph910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    op.Process<T>(input, output, tag, extraArgs);
    op.TailCounter();
}
