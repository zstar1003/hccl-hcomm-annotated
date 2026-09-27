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

class AivAll2AllRdma910B : public AivCommBase {
public:
    __aicore__ inline AivAll2AllRdma910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, int32_t tag, int64_t sendCount, int32_t serverNum);
};

template <typename T>
__aicore__ inline void
AivAll2AllRdma910B::Process(GM_ADDR input, GM_ADDR output, int32_t tag, int64_t sendCount, int32_t serverNum)
{
    if (blockIdx_ >= rankSize_) {
        return;
    }
    uint32_t targetRank = blockIdx_; // 每个aicore处理的rank

    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[targetRank]);

    if (blockIdx_ == rank_) {
        // 前同步，记录当前rank就绪
        Record1vN(tag, CommPattern::interRank);
    } else {
        // 检查对端就绪 & 跨片拷贝
        WaitNv1(tag, blockIdx_);
    }
    pipe_barrier(PIPE_ALL);

    for (uint32_t i = 0; i < serverNum; i++) {
        uint64_t toRank = i * rankSize_ + blockIdx_; // toRank表示rank给对端同一平面的rank的发送数据进行遍历
        uint64_t remoteRank = i * rankSize_ + rank_; // remoteRank表示对端根据重排后计算的数据偏移

        uint64_t rdmaSendLocalOffset = toRank * sendCount;
        uint64_t rdmaSendRemoteOffset = remoteRank * sendCount;

        CpGM2GM(cclGMOther + rdmaSendRemoteOffset, inputGM + rdmaSendLocalOffset, sendCount);
    }

    // 末尾同步
    // 本卡已完成GetBlockIdx()号对端上的rank号的数据发送
    pipe_barrier(PIPE_ALL);
    Record(tag, blockIdx_, AivNotifyType::DataSignal);
    pipe_barrier(PIPE_ALL);
    // 检查本卡上是否已接收到所有对端发送的数据
    Wait(tag, blockIdx_, AivNotifyType::DataSignal);

    return;
}

template <typename T>
__aicore__ inline void aiv_all_to_all_rdma_910b(KERNEL_ARGS_DEF)
{
    AivAll2AllRdma910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process<T>(input, output, tag, len, serverNum);
    op.TailCounter();
}
