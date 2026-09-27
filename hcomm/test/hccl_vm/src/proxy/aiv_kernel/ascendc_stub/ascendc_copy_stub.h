/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_C_COPY_STUB_H
#define ASCEND_C_COPY_STUB_H

#include "ai_core_stub.h"
#include "aiv_task.h"
#include "ascendc_base_stub.h"
#include "ascendc_memory_stub.h"
#include "sim_log.h"

namespace AscendC {
const uint8_t DEFAULT_DATA_COPY_NBURST = 1;
const uint8_t DEFAULT_DATA_COPY_STRIDE = 0;

struct DataCopyParams {
    __aicore__ DataCopyParams() {}

    __aicore__
    DataCopyParams(const uint16_t count, const uint16_t len, const uint16_t srcStrideIn, const uint16_t dstStrideIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint16_t blockLen = 0;
    uint16_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t dstStride = DEFAULT_DATA_COPY_STRIDE;
};

struct DataCopyExtParams {
    __aicore__ DataCopyExtParams() {}

    __aicore__ DataCopyExtParams(
        const uint16_t count, const uint32_t len, const uint32_t srcStrideIn, const uint32_t dstStrideIn,
        const uint32_t rsvIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          rsv(rsvIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint32_t blockLen = 0;
    uint32_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t rsv = 0; // reserved information
};

template <typename T>
struct DataCopyPadExtParams {
    __aicore__ DataCopyPadExtParams() {}

    __aicore__
    DataCopyPadExtParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue, T padValue)
        : isPad(isPadValue),
          leftPadding(leftPadValue),
          rightPadding(rightPadValue),
          paddingValue(padValue)
    {}

    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    T paddingValue = 0;
};

template <typename T>
__aicore__ void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
{
    const auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }

    const uint64_t len = static_cast<uint64_t>(count) * sizeof(T);
    // src DataSlice
    const AivSim::AivDataSlice srcSlice(AivSim::AivBufferType::UB, src.GetPhyAddr(), len);
    const AivSim::RankId srcRank = AivSim::AivKernelExecutor::GetInstance().GetRankId();
    // dst DataSlice
    AivSim::RankId dstRank = UINT32_MAX;
    const auto dstAddr = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
    const AivSim::AivDataSlice dstSlice
        = AivSim::AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(dstAddr, len, &dstRank);
    if (dstRank == UINT32_MAX) {
        HCCL_VM_ERROR("DataCopy resolved data slice failed, addr=0x{:x} len={:d}", dstAddr, len);
        return;
    }

    std::shared_ptr<AivSim::AivTask> task;
    if (aiv->GetAtomicOp() == AivSim::ReduceOp::REDUCE_RESERVED) {
        task = std::make_shared<AivSim::AivTaskMemCopy>(srcRank, srcSlice, dstRank, dstSlice);
    } else {
        const uint32_t dataType = AivSim::AivKernelExecutor::GetInstance().GetCurOp().dataType;
        task = std::make_shared<AivSim::AivTaskReduce>(
            srcRank, srcSlice, dstRank, dstSlice, dataType, static_cast<uint32_t>(aiv->GetAtomicOp()));
    }
    aiv->AppendMTE3(task);
}

template <typename T>
__aicore__ void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const uint32_t count)
{
    const auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }

    const uint64_t len = static_cast<uint64_t>(count) * sizeof(T);
    // dst DataSlice
    const AivSim::AivDataSlice dstSlice(AivSim::AivBufferType::UB, dst.GetPhyAddr(), len);
    const AivSim::RankId dstRank = AivSim::AivKernelExecutor::GetInstance().GetRankId();
    // src DataSlice
    AivSim::RankId srcRank = UINT32_MAX;
    const auto srcAddr = reinterpret_cast<uint64_t>(src.GetPhyAddr());
    const AivSim::AivDataSlice srcSlice
        = AivSim::AivKernelExecutor::GetInstance().ResolveGlobalDataSlice(srcAddr, len, &srcRank);
    if (srcRank == UINT32_MAX) {
        HCCL_VM_ERROR("DataCopy resolved data slice failed, addr=0x{:x} len={:d}", srcAddr, len);
        return;
    }

    auto task = std::make_shared<AivSim::AivTaskMemCopy>(srcRank, srcSlice, dstRank, dstSlice);
    aiv->AppendMTE2(task);
}

__aicore__ inline void SetAtomicNone()
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    aiv->SetAtomicOp();
}

template <typename T>
__aicore__ void SetAtomicAdd()
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    aiv->SetAtomicOp(AivSim::ReduceOp::REDUCE_SUM);
}

template <typename T>
__aicore__ void SetAtomicMax()
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    aiv->SetAtomicOp(AivSim::ReduceOp::REDUCE_MAX);
}

template <typename T>
__aicore__ void SetAtomicMin()
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    aiv->SetAtomicOp(AivSim::ReduceOp::REDUCE_MIN);
}

template <typename T>
__aicore__ void Duplicate(const LocalTensor<T>& dstLocal, const T& scalar, const int32_t& count)
{}
} // namespace AscendC

#endif // ASCEND_C_COPY_STUB_H
