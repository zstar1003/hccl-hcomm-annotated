/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_host.h"
#include "adapter_rts.h"

namespace hccl {
HostMem::HostMem(void* ptr, u64 size, bool owner, bool isRtsMem)
    : ptr_(ptr),
      size_(size),
      owner_(owner),
      isRtsMem_(isRtsMem)
{}

HostMem::HostMem(const HostMem& that) : ptr_(that.ptr()), size_(that.size_), owner_(false), isRtsMem_(that.isRtsMem_) {}

HostMem::HostMem(HostMem&& that) : ptr_(that.ptr_), size_(that.size_), owner_(that.owner_), isRtsMem_(that.isRtsMem_)
{
    that.ptr_ = nullptr;
    that.size_ = 0;
    that.owner_ = false;
}

HostMem::~HostMem()
{
    if (owner_ && ptr_) {
        HCCL_DEBUG("size_[%llu Byte]", size_);

        if (isRtsMem_) {
            HcclResult ret = hrtFreeHost(ptr_);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("rt_free error, ret[%d]", ret);
            }
        } else {
            delete[] static_cast<u8*>(ptr_);
        }
    }
}

void HostMem::free()
{
    if (ptr_) {
        HCCL_DEBUG("[HostMem][free] size_[%llu Byte]", size_);

        if (isRtsMem_) {
            HcclResult ret = hrtFreeHost(ptr_);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("[HostMem][free] rt_free error, ret[%d]", ret);
            }
        } else {
            delete[] static_cast<u8*>(ptr_);
        }
        ptr_ = nullptr;
    }
}

HostMem HostMem::alloc(u64 size, bool isRtsMem)
{
    void* ptr = nullptr;
    if (isRtsMem) {
        HcclResult ret = hrtMallocHost(&ptr, size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HostMem][Alloc]rtMallocHost error, ret[%d], size[%llu Byte]", ret, size);
        }
    } else {
        ptr = new (std::nothrow) u8[size];
    }
    if (ptr == nullptr) {
        HCCL_WARNING("HostMem alloc ptr null");
    }
    HostMem mem(ptr, size, true, isRtsMem);
    return mem;
}

HostMem HostMem::create(void* ptr, u64 size)
{
    HostMem mem(ptr, size, false);
    return mem;
}

HostMem& HostMem::operator=(const HostMem& that)
{
    if (&that != this) {
        ptr_ = that.ptr();
        size_ = that.size_;
        owner_ = false;
        isRtsMem_ = that.isRtsMem_;
    }

    return *this;
}

HostMem HostMem::operator=(HostMem&& that)
{
    if (&that != this) {
        ptr_ = that.ptr_;
        size_ = that.size_;
        owner_ = that.owner_;
        isRtsMem_ = that.isRtsMem_;
    }

    that.ptr_ = nullptr;
    that.size_ = 0;
    that.owner_ = false;

    return *this;
}

HostMem HostMem::range(u64 offset, u64 size) const
{
    HostMem mem;
    if (ptr_ != nullptr && (offset + size) <= size_) {
        mem = HostMem(static_cast<void*>(static_cast<s8*>(ptr_) + offset), size, false);
    } else {
        HCCL_WARNING("HostMem range[%llu] size[%llu Byte] error or ptr null", offset + size, size_);
    }
    return mem;
}
} // namespace hccl
