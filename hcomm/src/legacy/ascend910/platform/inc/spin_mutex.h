/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SPIN_MUTEX_H
#define HCCL_SPIN_MUTEX_H

#include <mutex>
#include <atomic>

namespace hccl {

class SpinMutex {
public:
    SpinMutex() = default;
    ~SpinMutex() = default;
    // delete copy and move constructors and assign operators
    SpinMutex(SpinMutex const&) = delete;            // Copy construct
    SpinMutex(SpinMutex&&) = delete;                 // Move construct
    SpinMutex& operator=(SpinMutex const&) = delete; // Copy assign
    SpinMutex& operator=(SpinMutex&&) = delete;      // Move assign
    void lock()
    {
        bool expected = false;
        while (!flag.compare_exchange_strong(expected, true)) {
            expected = false;
        }
    }
    void unlock() { flag.store(false); }

private:
    std::atomic<bool> flag = ATOMIC_VAR_INIT(false);
};
} // namespace hccl
#endif // HCCL_SPIN_MUTEX_H
