/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_T_PIPE_H
#define AIV_T_PIPE_H

#include <atomic>
#include <cstdint>

#include "ascendc_base_stub.h"
#include "local_tensor_stub.h"
#include "t_buf_stub.h"
#include "t_que_bind_stub.h"

namespace AscendC {
class TPipe {
public:
    __aicore__ TPipe() = default;
    __aicore__ ~TPipe() = default;

    template <class T>
    __aicore__ bool InitBuffer(T& que, uint8_t num, uint32_t len)
    {
        // todo 临时简单方案
        for (uint32_t i = 0; i < num; i++) {
            TensorMem tensorMem;
            tensorMem.ptr = reinterpret_cast<void*>(i * len);
            tensorMem.len = len;
            que.blocks_.push_back(tensorMem);
        }
        que.tensorLen_ = len;
        que.srcEventID_ = AllocEventID<HardEvent::MAX>();
        que.dstEventID_ = AllocEventID<HardEvent::MAX>();
        return true;
    }

    template <TPosition pos>
    __aicore__ bool InitBuffer(TBuf<pos>& buf, uint32_t len)
    {
        // todo 临时简单方案
        buf.block_.ptr = 0;
        buf.block_.len = len;
        return true;
    }

    __aicore__ void Reset() { eventIdGen_.store(0); }

    // 事件管理: 当前先简单实现, EventID的数量限制先不支持
    template <HardEvent evt>
    __aicore__ TEventID AllocEventID()
    {
        return eventIdGen_.fetch_add(1);
    }
    template <HardEvent evt>
    __aicore__ void ReleaseEventID(TEventID id)
    {}

private:
    std::atomic<TEventID> eventIdGen_{0}; // EventID Generator

    void* startPtr = 0;
    void* endPtr = 0;
    void* curPtr = 0;
    uint64_t usedLen = 0;
    uint32_t usedNum = 0;
};
} // namespace AscendC

#endif // AIV_T_PIPE_H
