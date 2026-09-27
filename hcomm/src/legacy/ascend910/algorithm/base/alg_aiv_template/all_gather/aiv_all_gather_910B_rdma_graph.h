/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_GATHER_910B_RDMA_GRAPH_H
#define AIV_ALL_GATHER_910B_RDMA_GRAPH_H

#include "aiv_communication_base.h"

using namespace AscendC;

#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__

template <typename T>
class AivAllGather910BRdmaGraph : public AivCommBase {
public:
    FORCE_INLINE_AICORE AivAllGather910BRdmaGraph() {}

    /**
     *  8个核就够拉整个8个不同卡cclOut到userOut了
     */
    FORCE_INLINE_AICORE void
    Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize, uint64_t serverNum)
    {
        if (blockIdx_ >= rankSize_) {
            return;
        }

        if (blockIdx_ == 0) {
            // 本卡该片数据已经可以被跨片读取
            SetSignalValue((__gm__ int32_t*)(GM_IN[rank_]), localSetTensor, tag);
        }
        WaitSignalValue((__gm__ int32_t*)(GM_IN[blockIdx_]), localCheckTensor, tag);

        for (int i = 0; i < serverNum; i++) {
            if (blockIdx_ == rank_) {
                break;
            }
            int64_t receiveSizeOffset = (i * rankSize_ + blockIdx_) * len * sizeof(T);
            CpGM2GM<T>(
                (__gm__ T*)((__gm__ char*)output + receiveSizeOffset),
                (__gm__ T*)((__gm__ char*)(GM_OUT[blockIdx_]) + receiveSizeOffset), len);
        }

        if (blockIdx_ == 0) {
            SetSignalValue((__gm__ int32_t*)(GM_IN[rank_]) + 8, localSetTensor, tag);
        }
        WaitSignalValue((__gm__ int32_t*)(GM_IN[blockIdx_]) + 8, localCheckTensor, tag);
    }
};

template <typename T>
FORCE_INLINE_AICORE void aiv_all_gather_910b_rdma_graph(KERNEL_ARGS_DEF)
{
    AivAllGather910BRdmaGraph<T> op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process(input, output, len, tag, bufferSize, serverNum);
    op.TailCounter();
}
#endif // AIV_ALL_GATHER_910B_RDMA_GRAPH_H
