/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_PARAM_H
#define ALG_PARAM_H

#include <cstdint>
#include <cstddef>

struct AlgParam {
    uint32_t algType;
    uint32_t planeNum;
    uint32_t rankSize;
    uint32_t rootRank;
};

namespace ops_hccl {
struct OpParam {
    void* resCtx = nullptr;
    void* hcclComm = nullptr;
    void* inputPtr = nullptr;
    void* outputPtr = nullptr;
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    uint32_t root = 0;
    uint32_t userRank = 0;
    int reduceType = 0;
    int opType = 0;
    void* stream = nullptr;
};
} // namespace ops_hccl

#endif
