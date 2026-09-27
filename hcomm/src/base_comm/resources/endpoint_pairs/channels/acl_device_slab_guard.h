/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_ACL_DEVICE_SLAB_GUARD_H
#define CHANNEL_ACL_DEVICE_SLAB_GUARD_H

#include "adapter_rts.h"
#include "log.h"

namespace hcomm {

class AclDeviceSlabGuard {
public:
    AclDeviceSlabGuard() = default;
    ~AclDeviceSlabGuard()
    {
        if (ptr_ != nullptr) {
            HcclResult ret = hrtFree(ptr_);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("[AclDeviceSlabGuard] hrtFree failed, ptr[%p], size[%zu], ret[%d]", ptr_, size_, ret);
            }
        }
    }

    void Reset(void* ptr, size_t size)
    {
        ptr_ = ptr;
        size_ = size;
    }

    void* Release()
    {
        void* ptr = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return ptr;
    }

private:
    void* ptr_{nullptr};
    size_t size_{0};
};

} // namespace hcomm

#endif // CHANNEL_ACL_DEVICE_SLAB_GUARD_H
