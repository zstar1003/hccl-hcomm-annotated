/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFX_CIRCULAR_QUEUE_H
#define DFX_CIRCULAR_QUEUE_H
#include "cast_utils.h"
#include <cstring>
#include "securec.h"
#include "hccl/base.h"
#include "log.h"
#include "res_pub.h"

namespace Hccl {

template <typename T, u32 CAPACITY>
class DfxCircularQueue {
    static constexpr u32 ITEM_SIZE = sizeof(T);

public:
    DfxCircularQueue() : begin_(0), end_(0), count_(0)
    {
        (void)memset_sp(buffer_, sizeof(buffer_), 0, sizeof(buffer_));
    }

    u16 GetBegin() const { return begin_; }

    u16 GetEnd() const { return end_; }

    void MarkAllRead()
    {
        begin_ = end_;
        count_ = 0;
    }

    void* NextSlot()
    {
        void* slot = buffer_ + end_ * ITEM_SIZE;
        if (count_ == CAPACITY) {
            begin_ = static_cast<u16>((begin_ + 1) % CAPACITY);
        }
        end_ = static_cast<u16>((end_ + 1) % CAPACITY);
        if (count_ < CAPACITY) {
            count_++;
        }
        return slot;
    }

    constexpr u32 GetCapacity() const { return CAPACITY; }

    u16 GetCount() const { return count_; }

    bool IsEmpty() const { return count_ == 0; }

    bool IsFull() const { return count_ == CAPACITY; }

    T* GetSlot(u16 index) const { return ReinterpretAs<T*>(buffer_ + static_cast<size_t>(index) * ITEM_SIZE); }

private:
    mutable u8 buffer_[CAPACITY * ITEM_SIZE];
    u16 begin_;
    u16 end_;
    u16 count_;
};

using TaskInfoCircularQueue = DfxCircularQueue<DfxTaskInfo, DFX_TASK_INFO_QUEUE_CAPACITY>;
using DfxOpInfoCircularQueue = DfxCircularQueue<DfxDfxOpInfo, DFX_OP_INFO_QUEUE_CAPACITY>;

} // namespace Hccl
#endif // DFX_CIRCULAR_QUEUE_H
