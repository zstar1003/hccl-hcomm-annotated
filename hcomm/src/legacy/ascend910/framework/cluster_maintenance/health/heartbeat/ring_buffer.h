/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RINGBUFFER_H
#define HCCL_RINGBUFFER_H

#include "hccl/hccl_types.h"
#include "log.h"

namespace hccl {

class RingBuffer {
public:
    RingBuffer() {}
    ~RingBuffer()
    {
        if (capacity_ > 0 && data_ != nullptr) {
            delete[] data_;
            data_ = nullptr;
            capacity_ = 0;
        }
    }
    HcclResult Init(u32 capacity)
    {
        if (initialized_) {
            return HCCL_SUCCESS;
        }
        capacity_ = capacity;
        if (capacity_ <= 0) {
            HCCL_ERROR("[RingBuffer] capacity[%u] must be greater than 0", capacity_);
            return HCCL_E_PARA;
        }

        data_ = new (std::nothrow) u8[capacity_];
        CHK_PTR_NULL(data_);
        initialized_ = true;

        return HCCL_SUCCESS;
    }
    HcclResult PushSeg(u8* src, u32 count)
    {
        CHK_PTR_NULL(src);
        if (size_ > capacity_ || count > (capacity_ - size_)) {
            HCCL_ERROR("[RingBuffer] Not Enough Space");
            return HCCL_E_PARA;
        }

        u32 rest = capacity_ - tail_;
        // 如果队尾到右边界的空间足够入队，则直接入队
        // 否则，从队尾入rest个元素后，再从左边界入队
        if (rest >= count) {
            s32 sRet = memcpy_s(data_ + tail_, count, src, count);
            CHK_PRT_RET(sRet != EOK, HCCL_ERROR("memcpy_s failed, errorno[%d], size[%u]", sRet, count), HCCL_E_MEMORY);
        } else {
            s32 sRet = memcpy_s(data_ + tail_, rest, src, rest);
            CHK_PRT_RET(sRet != EOK, HCCL_ERROR("memcpy_s failed, errorno[%d], size[%u]", sRet, rest), HCCL_E_MEMORY);
            sRet = memcpy_s(data_, count - rest, src + rest, count - rest);
            CHK_PRT_RET(
                sRet != EOK, HCCL_ERROR("memcpy_s failed, errorno[%d], size[%u]", sRet, count - rest), HCCL_E_MEMORY);
        }
        size_ += count;
        tail_ = (tail_ + count) % capacity_;

        return HCCL_SUCCESS;
    }
    HcclResult PopSeg(u32 count)
    {
        if (size_ < count) {
            HCCL_ERROR("[RingBuffer] Not Enough Element");
            return HCCL_E_PARA;
        }
        size_ -= count;
        head_ = (head_ + count) % capacity_;

        return HCCL_SUCCESS;
    }
    HcclResult GetSeg(u8* dst, u32 count) const
    {
        CHK_PTR_NULL(dst);
        if (size_ < count) {
            HCCL_ERROR("[RingBuffer] Not Enough Element");
            return HCCL_E_PARA;
        }

        u32 rest = capacity_ - head_;
        // 如果队头到右边界的元素足够出队，则直接出队
        // 否则，从队头出rest个元素后，再从左边界出队
        if (rest >= count) {
            s32 sRet = memcpy_s(dst, count, data_ + head_, count);
            CHK_PRT_RET(sRet != EOK, HCCL_ERROR("memcpy_s failed, errorno[%d], size[%u]", sRet, count), HCCL_E_MEMORY);
        } else {
            s32 sRet = memcpy_s(dst, rest, data_ + head_, rest);
            CHK_PRT_RET(sRet != EOK, HCCL_ERROR("memcpy_s failed, errorno[%d], size[%u]", sRet, rest), HCCL_E_MEMORY);
            sRet = memcpy_s(dst + rest, count - rest, data_, count - rest);
            CHK_PRT_RET(
                sRet != EOK, HCCL_ERROR("memcpy_s failed, errorno[%d], size[%u]", sRet, count - rest), HCCL_E_MEMORY);
        }

        return HCCL_SUCCESS;
    }
    u32 Size() const { return size_; }

private:
    u32 capacity_ = 0;
    u32 head_ = 0;
    u32 tail_ = 0;
    u32 size_ = 0;
    u8* data_ = nullptr;
    bool initialized_ = false;
};

} // namespace hccl
#endif // HCCL_RINGBUFFER_H
