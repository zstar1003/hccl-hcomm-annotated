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

class AivAllGatherSmall910B : public AivCommBase {
public:
    __aicore__ inline AivAllGatherSmall910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void AivAllGatherSmall910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    // 共用2个flag
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[blockIdx_] + dataOffset);
    __gm__ T* outputGM = (__gm__ T*)output;
    bool ifPingpong = (tag % 2 == 0);

    uint64_t count = len;

    if (blockIdx_ != rank_) {
        WaitNv1(tag, blockIdx_, AivNotifyType::DataSignal, 0, ifPingpong);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(outputGM + blockIdx_ * count, cclGMOther, count);

        // 卡间同步
    } else {
        CpGM2GM(cclGMSelf, inputGM, count);
        // 卡间同步
        pipe_barrier(PIPE_ALL);
        Record1vN(tag, CommPattern::interRank, AivNotifyType::DataSignal, 0, ifPingpong);
        CpGM2GM(outputGM + count * rank_, cclGMSelf, count);
    }
}

template <typename T>
__aicore__ inline void aiv_all_gather_910b_smalldata(KERNEL_ARGS_DEF)
{
    AivAllGatherSmall910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag);
    op.TailCounter();
}
