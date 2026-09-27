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

class AivReduceScatterRdma910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterRdma910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint32_t serverNum);
};

template <typename T>
__aicore__ inline void
AivReduceScatterRdma910B::Process(GM_ADDR input, GM_ADDR output, uint64_t count, int32_t tag, uint32_t serverNum)
{
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[blockIdx_]);

    // reduce scatter，数据从input输入，inputMem+0作为buffer，结果放在原位，标记放在inputMem末尾flag区的起始位置
    uint32_t LengthPerPlane = serverNum * count;
    uint32_t LengthPerServer = rankSize_ * count;

    if (blockIdx_ == rank_) {                      // 本rank对应的block,将数据从input搬至ccl
        for (uint32_t i = 0; i < serverNum; i++) { // 循环处理每个服务器需要的数据
            CpGM2GM(
                cclGMSelf + blockIdx_ * LengthPerPlane + i * count, inputGM + blockIdx_ * count + i * LengthPerServer,
                count);
        }
        // 本地拷贝 & 卡间同步
        pipe_barrier(PIPE_ALL);
        Record1vN(tag, CommPattern::intraRank); // 本卡该片数据已可以被跨片读取（也可累加）
    } else { // 其余block,先将数据从input搬至ccl，再从其他卡的ccl读数据至本卡ccl
        for (uint32_t i = 0; i < serverNum; i++) { // 循环处理每个服务器需要的数据
            CpGM2GM(
                cclGMSelf + blockIdx_ * LengthPerPlane + i * count, inputGM + blockIdx_ * count + i * LengthPerServer,
                count);
        }
        // 本地拷贝 & 卡间同步
        pipe_barrier(PIPE_ALL);
        Record(tag, blockIdx_, AivNotifyType::ACK); // 本卡该片数据已可以被跨片读取

        // 检查对端数据就绪且本端就绪 & 跨片搬运
        Wait(tag, blockIdx_, AivNotifyType::ACK);
        WaitNv1(tag, rank_);
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
__aicore__ inline void aiv_reduce_scatter_910b_rdma(KERNEL_ARGS_DEF)
{
    AivReduceScatterRdma910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag, serverNum);
    op.TailCounter();
}
