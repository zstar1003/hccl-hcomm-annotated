/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_COPY_STUB_H
#define AIV_COPY_STUB_H
#include "aiv_base_stub.h"
#include "aiv_memory_stub.h"

namespace AscendC {
// 数据拷贝相关
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
__aicore__ void DataCopy(
    const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const uint32_t calCount,
    bool isGenfromSync = false);

template <typename T>
__aicore__ void DataCopy(
    const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const uint32_t calCount,
    bool isGenfromSync = false);

template <typename T>
__aicore__ void DataCopyPad(
    const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const DataCopyExtParams& dataCopyParams,
    bool isGenfromSync = false);

template <typename T>
__aicore__ void DataCopyPad(
    const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const DataCopyExtParams& dataCopyParams,
    const DataCopyPadExtParams<T>& padParams, bool isGenfromSync = false);

__aicore__ void SetAtomicNone();

// set_atomic_add
template <typename T>
__aicore__ void SetAtomicAdd();

// set_atomic_max
template <typename T>
__aicore__ void SetAtomicMax();

// set_atomic_min
template <typename T>
__aicore__ void SetAtomicMin();

template <typename T>
__aicore__ void Duplicate(const LocalTensor<T>& dstLocal, const T& scalar, const int32_t& count);
} // namespace AscendC

#endif
