/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_LOCAL_TENSOR_STUB_H
#define AIV_LOCAL_TENSOR_STUB_H

#include "ai_core_stub.h"
#include "aiv_task.h"
#include "ascendc_base_stub.h"
#include "sim_log.h"

namespace AscendC {
template <typename T>
class LocalTensor {
public:
    __aicore__ LocalTensor() = default;
    __aicore__ LocalTensor(TPosition pos, void* ptr, uint32_t size) : ptr_(static_cast<T*>(ptr)), size_(size) {}
    __aicore__ LocalTensor(TPosition pos, uint32_t addr, uint32_t size) : ptr_(reinterpret_cast<T*>(addr)), size_(size)
    {}
    __aicore__ ~LocalTensor() = default;

    __aicore__ __inout_pipe__(S) void SetValue(const uint32_t index, const T value) const
    {
        HCCL_VM_ERROR("LocalTensor SetValue not support !");
    }

    __aicore__ __inout_pipe__(S) T GetValue(const uint32_t offset) const
    {
        if (offset >= size_) {
            HCCL_VM_ERROR("LocalTensor GetValue out-of-bounds, offset={:d}", offset);
        }
        return *(ptr_ + offset);
    }

    __aicore__ LocalTensor operator[](const uint32_t offset) const
    {
        if (offset >= size_) {
            HCCL_VM_ERROR("LocalTensor operator[] out-of-bounds, offset={:d}", offset);
        }
        T* newPtr = ptr_ + offset;
        uint32_t newSize = size_ - offset;
        return LocalTensor<T>{TPosition::LCM, newPtr, newSize};
    }

    __aicore__ uint64_t GetPhyAddr() const { return reinterpret_cast<uint64_t>(ptr_); }
    __aicore__ const T* GetPtr() const { return ptr_; }
    __aicore__ void SetPtr(void* ptr) { ptr_ = static_cast<T*>(ptr); }

    __aicore__ void SetSize(const uint32_t size) { size_ = size; }
    __aicore__ uint32_t GetSize() const { return size_; }
    __aicore__ uint32_t GetLength() const { return size_ * sizeof(T); } // LocalTensor数据长度，单位为字节

private:
    T* ptr_{nullptr};
    uint32_t size_{0}; // 元素个数
};
} // namespace AscendC

#endif // AIV_LOCAL_TENSOR_STUB_H
