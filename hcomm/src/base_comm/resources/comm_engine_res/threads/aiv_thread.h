/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_THREAD_H
#define AIV_THREAD_H

#include <vector>
#include "thread.h"

namespace hccl {
/**
 * @note 职责：AIV通信引擎的Thread的C++派生类，内部对应AIV block，及thread间的同步Notify。
 */
class AivThread : public Thread {
public:
    AivThread();
    ~AivThread();

    uint32_t GetNotifyNum() const override;

private:
    uint32_t notifyNum_{};
};
} // namespace hccl
#endif // AIV_THREAD_H
