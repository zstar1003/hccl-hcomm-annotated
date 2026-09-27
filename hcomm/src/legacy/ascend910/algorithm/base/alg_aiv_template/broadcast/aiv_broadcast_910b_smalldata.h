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

class AivBroadcastSmall910B : public AivCommBase {
public:
    __aicore__ inline AivBroadcastSmall910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint32_t root);
};

template <typename T>
__aicore__ inline void
AivBroadcastSmall910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint32_t root)
{
    // 小数据量的情况下只用一个核搬运数据
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMRoot = (__gm__ T*)(GM_IN[root]); // root卡的cclbuffer
    __gm__ T* outputGM = (__gm__ T*)output;

    if (blockIdx_ > 0) {
        return;
    }
    if (rank_ == root) {
        // 当前卡为root时，将root的数据搬到cclbuffer
        CpGM2GM(cclGMRoot, inputGM, len);
        PipeBarrier<PIPE_ALL>();
        // 告诉其他卡可以取数据了
        Record1vN(tag, CommPattern::interRank, AivNotifyType::DataSignal);
        for (uint32_t remoteRank = 0; remoteRank < rankSize_; remoteRank += 1) {
            // 等每个对端rank拿走自己的数据
            if (remoteRank == root)
                continue;
            Wait(tag, remoteRank, AivNotifyType::Done);
        }
        PipeBarrier<PIPE_ALL>();
    } else {
        WaitNv1(tag, root, AivNotifyType::DataSignal);
        PipeBarrier<PIPE_ALL>();
        CpGM2GM(outputGM, cclGMRoot, len); // 数据量小于190k时，ub一次搬运，直接从root卡的cclbuffer传数据到本卡的输出
        PipeBarrier<PIPE_ALL>();
        // 置标志位，表示自己拿走了这个数据
        Record(tag, root, AivNotifyType::Done);
    }
}
template <typename T>
__aicore__ inline void aiv_broadcast_910b_smalldata(KERNEL_ARGS_DEF)
{
    AivBroadcastSmall910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.Process<T>(input, output, len, tag, root); // 只用1个核的情况
}
