/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_GLOBAL_TENSOR_STUB_H
#define AIV_GLOBAL_TENSOR_STUB_H

#include "ascendc_base_stub.h"
#include "sim_log.h"

namespace AscendC {
template <typename T>
class GlobalTensor {
public:
    __aicore__ GlobalTensor() = default;
    __aicore__ ~GlobalTensor() = default;

    __aicore__ void SetGlobalBuffer(__gm__ T* buffer, uint64_t bufferSize)
    {
        ptr_ = buffer;
        size_ = bufferSize;
    }

    __aicore__ void SetGlobalBuffer(__gm__ T* buffer)
    {
        ptr_ = buffer;
        size_ = UINT64_MAX;
    }

    __aicore__ __inout_pipe__(S) T GetValue(const uint64_t offset) const
    {
        if (offset >= size_) {
            HCCL_VM_ERROR("GlobalTensor GetValue out-of-bounds, offset={:d}", offset);
        }
        return *(ptr_ + offset);
    }

    __aicore__ GlobalTensor operator[](const uint64_t offset) const
    {
        if (offset >= size_) {
            HCCL_VM_ERROR("LocalTensor operator[] out-of-bounds, offset={:d}", offset);
        }
        T* newPtr = ptr_ + offset;
        uint64_t newSize = size_ - offset;
        GlobalTensor<T> newTensor{};
        newTensor.SetGlobalBuffer(newPtr, newSize);
        return newTensor;
    }

    __aicore__ const __gm__ T* GetPhyAddr() const { return ptr_; }
    __aicore__ void SetPtr(void* ptr) { ptr_ = static_cast<T*>(ptr); }

    __aicore__ void SetSize(const uint32_t size) { size_ = size; }
    __aicore__ uint64_t GetSize() const { return size_ == UINT64_MAX ? 0 : size_; }
    __aicore__ uint64_t GetLength() const { return size_ == UINT64_MAX ? 0 : size_ * sizeof(T); }

private:
    T* ptr_{nullptr};
    uint64_t size_{0};
};
} // namespace AscendC

#endif // AIV_GLOBAL_TENSOR_STUB_H
