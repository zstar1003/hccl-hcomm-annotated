/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_GATHER_910B_RDMA_H
#define AIV_ALL_GATHER_910B_RDMA_H

#include "aiv_communication_base.h"
using namespace AscendC;

#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__

template <typename T>
class AivAllGather910BRdma : public AivCommBase {
public:
    FORCE_INLINE_AICORE AivAllGather910BRdma() {}

    /**
     *  8个核就够拉整个8个不同卡cclOut到userOut了
     */

    FORCE_INLINE_AICORE void
    Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize, uint64_t serverNum)
    {
        if (blockIdx_ >= rankSize_) {
            return;
        }

        if (blockIdx_ == rank_) {
            // 本卡该片数据已经可以被跨片读取
            Record1vN(tag, CommPattern::interRank);
        } else {
            WaitNv1(tag, blockIdx_);
        }
        pipe_barrier(PIPE_ALL);

        // todo:1、serverNum需要赋值。 2、len是inputCount 还是inputSize还是 output相关？
        for (int i = 0; i < serverNum; i++) {
            int64_t sendSize = len * sizeof(T);
            int64_t sendSizeOffset = i * len * sizeof(T);
            int64_t receiveSizeOffset = (i * rankSize_ + blockIdx_) * len * sizeof(T);
            CpGM2GM<T>(
                (__gm__ T*)((__gm__ char*)output + receiveSizeOffset),
                (__gm__ T*)((__gm__ char*)(GM_IN[blockIdx_]) + sendSizeOffset), len);
        }
        pipe_barrier(PIPE_ALL);
        // 尾同步，每个卡搬完完后要进行标记。要确保所有卡都搬完再退出。
        if (blockIdx_ == rank_) {
            Wait1vN(tag * (rankSize_ - 1), CommPattern::interRank);
        } else {
            RecordNv1(tag, blockIdx_);
        }
    }
};

template <typename T>
FORCE_INLINE_AICORE void aiv_all_gather_910b_rdma(KERNEL_ARGS_DEF)
{
    AivAllGather910BRdma<T> op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process(input, output, len, tag, bufferSize, serverNum);
    op.TailCounter();
}
#endif // AIV_ALL_GATHER_910B_RDMA_H
