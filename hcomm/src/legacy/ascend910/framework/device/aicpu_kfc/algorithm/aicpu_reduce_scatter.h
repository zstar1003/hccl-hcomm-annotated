/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_REDUCE_SCATTER_H__
#define __AICPU_REDUCE_SCATTER_H__

#include "aicpu_algorithm.h"

class AicpuReduceScatter : public AicpuAlgorithm {
public:
    explicit AicpuReduceScatter(AicpuComContext* ctx) : AicpuAlgorithm(ctx) {}
    ~AicpuReduceScatter() override = default;

    HcclResult RunAlgorithm(
        HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType,
        u64 strideLen = 0, AivAicpuOpParam* nextTask = nullptr) override;

private:
    HcclResult RunReduceScatterWriteMode(
        HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen);
    HcclResult RunReduceScatterReadMode(
        HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType);

    std::vector<Slice> PrepareMeshSlice(u64 dataSize, uint32_t rankNum);
    HcclResult RunDeterministicReduceScatter(
        HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideCount);
    HcclResult RunDeterministicReduceScatterLocal(
        HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType, u64 strideLen);
    HcclResult RunCurrentDeterministicReduceScatter(
        HcclReduceOp opType, uint8_t* curInputPtr, uint8_t* curOutputPtr, u64 strideSize, u64 curSize,
        HcclDataType dataType);

    // 910_93
    HcclResult GenRingTask(
        HcclReduceOp opType, u64 sndAddr, u64 rcvAddr, u64 curSize, u64 scatterSize, HcclDataType dataType,
        uint32_t streamId, bool isClockwise, uint32_t step) const;
    HcclResult RunDoubleRingReduceScatter(
        HcclReduceOp opType, u64 sendBuffer, u64 recvBuffer, u64 dataCount, HcclDataType dataType) const;
    HcclResult RunSwitchReduceScatter(
        HcclReduceOp opType, u64 sendBuffer, u64 recvBuffer, u64 dataCount, HcclDataType dataType) const;
};

#endif
