/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_ALLTOALL_H__
#define __AICPU_ALLTOALL_H__

#include "aicpu_algorithm.h"

class AicpuAllToAll : public AicpuAlgorithm {
public:
    explicit AicpuAllToAll(AicpuComContext* ctx) : AicpuAlgorithm(ctx) {}
    ~AicpuAllToAll() override = default;

    HcclResult RunAlgorithm(
        HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType,
        u64 strideLen = 0, AivAicpuOpParam* nextTask = nullptr) override;

private:
    HcclResult RunAllToAllWinIn(void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen);
    HcclResult RunAllToAll(void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen);
};

#endif
