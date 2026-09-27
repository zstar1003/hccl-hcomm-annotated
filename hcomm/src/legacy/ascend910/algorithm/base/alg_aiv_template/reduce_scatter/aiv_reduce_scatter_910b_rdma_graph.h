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

class AivReduceScatterRdmaGraph910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterRdmaGraph910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint32_t serverNum);
};

template <typename T>
__aicore__ inline void
AivReduceScatterRdmaGraph910B::Process(GM_ADDR input, GM_ADDR output, uint64_t count, int32_t tag, uint32_t serverNum)
{
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[blockIdx_]);

    // reduce scatter，数据从input输入，inputMem+0作为buffer，结果放在原位，标记放在inputMem末尾flag区的起始位置
    uint32_t flagBaseOffset = 0;
    uint32_t flagOffsetOut = flagBaseOffset + blockIdx_ * FLAG_INTERVAL; // 给其他卡的标记，数据已在buffer中就绪
    uint32_t flagOffsetRemote = flagBaseOffset + rank_ * FLAG_INTERVAL; // 本卡aiv需要读的其他卡的标记
    uint32_t flagOffsetIn = flagBaseOffset + rank_ * FLAG_INTERVAL; // 给本卡其他aiv的标记，数据已在output中，可以累加
    uint32_t LengthPerPlane = serverNum * count;
    uint32_t LengthPerServer = rankSize_ * count;

    if (blockIdx_ == rank_) {
        for (uint32_t i = 0; i < serverNum; i++) { // 循环处理每个服务器需要的数据
            CpGM2GM(
                cclGMSelf + blockIdx_ * LengthPerPlane + i * count, inputGM + blockIdx_ * count + i * LengthPerServer,
                count);
        }
        // 本地拷贝 & 卡间同步
        pipe_barrier(PIPE_ALL);
        Record1vN(tag, CommPattern::intraRank); // 本卡该片数据已可以被跨片读取（也可累加）
    } else {
        for (uint32_t i = 0; i < serverNum; i++) { // 循环处理每个服务器需要的数据
            CpGM2GM(
                cclGMSelf + blockIdx_ * LengthPerPlane + i * count, inputGM + blockIdx_ * count + i * LengthPerServer,
                count);
        }
        // 本地拷贝 & 卡间同步
        pipe_barrier(PIPE_ALL);
        Record(tag, blockIdx_, AivNotifyType::ACK); // 本卡该片数据已可以被跨片读取
        // 检查对端数据就绪且本端就绪 & 跨片搬运
        pipe_barrier(PIPE_ALL);
        Wait(tag, blockIdx_, AivNotifyType::ACK);
        WaitNv1(tag, blockIdx_);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(
            cclGMSelf + LengthPerPlane * rank_, cclGMOther + LengthPerPlane * rank_, count * serverNum, true,
            reduceOp_);
        pipe_barrier(PIPE_ALL);
        Record(tag, blockIdx_, AivNotifyType::DataSignal); // 本卡该片数据已可以被跨片读取
        Wait(tag, blockIdx_, AivNotifyType::DataSignal);
    }
    return;
}

template <typename T>
__aicore__ inline void aiv_reduce_scatter_910b_rdma_graph(KERNEL_ARGS_DEF)
{
    AivReduceScatterRdmaGraph910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag, serverNum);
    op.TailCounter();
}
