/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_COMM_POOL_POLICY_H
#define SIM_COMM_POOL_POLICY_H

#include <cstddef>

namespace sim {
// HcclCommPool 复用区的准入策略，主机侧与设备侧共用的纯判定。阈值与上界集中在此定义。
struct CommPoolPolicy {
    static constexpr size_t kBigBlockThreshold = 200ULL * 1024 * 1024; // 200MB，对齐 CCL buffer 默认
    // 复用区规格 4GB，大块可申请和可寻址的上界（含等于）。超过 4GB 由 ExceedsCeiling 报错。
    static constexpr size_t kPoolSize = 4ULL * 1024 * 1024 * 1024; // 4GB

    // 复用区共享内存名。
    static constexpr const char* kPoolName = "HcclCommPool";

    // 本次申请是否应引流到复用区：仅校验模式开且 threshold <= size <= poolSize。
    static bool
    ShouldRedirect(size_t size, bool checkOnlyMode, size_t threshold = kBigBlockThreshold, size_t poolSize = kPoolSize)
    {
        return checkOnlyMode && size >= threshold && size <= poolSize;
    }

    // 本次申请是否超出复用区上界、应报错：仅仅校验模式下拦 size > poolSize。
    static bool ExceedsCeiling(size_t size, bool checkOnlyMode, size_t poolSize = kPoolSize)
    {
        return checkOnlyMode && size > poolSize;
    }
};
} // namespace sim

#endif // SIM_COMM_POOL_POLICY_H
