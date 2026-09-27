/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_T_QUE_BIND_STUB_H
#define AIV_T_QUE_BIND_STUB_H

#include <vector>

#include "ascendc_base_stub.h"
#include "ascendc_sync_stub.h"
#include "local_tensor_stub.h"

namespace AscendC {
struct TensorMem {
    void* ptr{nullptr};
    uint64_t len{0};
    bool isUse{false};
    bool inQue{false};
    bool needWait{false};
};

template <TPosition src, TPosition dst, int32_t depth, auto mask = 0>
class TQueBind {
public:
    __aicore__ TQueBind() = default;
    __aicore__ ~TQueBind() = default;

    template <typename T>
    __aicore__ LocalTensor<T> AllocTensor()
    {
        LocalTensor<T> tmp;
        for (auto& mem : blocks_) {
            if (!mem.isUse && !mem.inQue) {
                mem.isUse = true;
                tmp.SetPtr(mem.ptr);
                tmp.SetSize(mem.len / sizeof(T));

                if (mem.needWait) {
                    WaitFlag<HardEvent::MTE3_MTE2>(dstEventID_); // todo
                }
                mem.needWait = true;
                break;
            }
        }
        return tmp;
    }

    template <typename T>
    __aicore__ void FreeTensor(LocalTensor<T>& tensor)
    {
        for (auto& mem : blocks_) {
            if (mem.ptr == tensor.GetPtr() && mem.isUse) {
                mem.isUse = false;
                break;
            }
        }
        tensor.SetPtr(nullptr);
        tensor.SetPtr(0);
        SetFlag<HardEvent::MTE3_MTE2>(dstEventID_); // todo
    }

    template <typename T>
    __aicore__ bool EnQue(const LocalTensor<T>& tensor)
    {
        for (auto& mem : blocks_) {
            if (mem.ptr == tensor.GetPtr() && !mem.inQue) {
                mem.inQue = true;
                SetFlag<HardEvent::MTE2_MTE3>(srcEventID_); // todo
                return true;
            }
        }
        return false;
    }

    template <typename T>
    __aicore__ LocalTensor<T> DeQue()
    {
        LocalTensor<T> tmp;
        for (auto& mem : blocks_) {
            if (mem.isUse && mem.inQue) {
                mem.inQue = false;
                tmp.SetPtr(mem.ptr);
                tmp.SetSize(mem.len / sizeof(T));
                break;
            }
        }
        WaitFlag<HardEvent::MTE2_MTE3>(srcEventID_); // todo
        return tmp;
    }

    __aicore__ uint32_t GetMemBlockNum() const { return blocks_.size(); }

private:
    std::vector<TensorMem> blocks_{}; // 内存块, InitBuffer时分配
    uint64_t tensorLen_{0};           // 每个内存块的大小，单位为字节

    // 正向同步事件(事件类型暂时固定), InitBuffer时分配ID
    HardEvent srcEvent_{HardEvent::MTE2_MTE3}; // todo
    TEventID srcEventID_{0};
    // 反向同步事件(事件类型暂时固定), InitBuffer时分配ID
    HardEvent dstEvent_{HardEvent::MTE3_MTE2}; // todo
    TEventID dstEventID_{0};

    friend class TPipe;
};
} // namespace AscendC

#endif // AIV_T_QUE_BIND_STUB_H
