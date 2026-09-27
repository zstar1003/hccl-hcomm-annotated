/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_PROFILING_H
#define TASK_PROFILING_H

#include "task_profiling_pub.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdint>

namespace hccl {
/* 一共1K个notify，每个8字节，地址和13位的1相与之后右移3位后就可以转换成
    0~1023的唯一的ID */
#define NOTIFY_ADDR_TO_ID(addr) ((((u64)(uintptr_t)(addr)) & 0x1FFF) >> 3)

constexpr u32 RESERVED_SIZE_STRING = 4096; // 4KB

/* * 统一按照PCIe的带宽来计算, SDMA的固定开销按照0.6us(<512KB), 1.5us(>512KB)计算
    PCIe DMA实测带宽为19.3GB */
constexpr u32 DURATION_PRECISION = 3;
constexpr u32 DURATION_INIT_VALUE = 0;

constexpr u32 DURATION_SDMA_FIXED_THRESHOLD = 1024 * 512;   // （魔鬼数字解释）512 * 1024  512K
constexpr double DURATION_SDMA_FIXED_THRESHOLD_BELOW = 0.6; // 小数据量的时候SDMA的固定开销为0.6us
constexpr double DURATION_SDMA_FIXED_THRESHOLD_ABOVE = 1.5; // 大数据量的时候SDMA的固定开销为0.6us
constexpr double DURATION_SDMA_BANDWIDTH_MB = 19.3 * 1000;  // （魔鬼数字解释）19.3 * 1000

/* * RDMA的固定开销按照7us计算(实测7us)
            RDMA实测大包单流带宽为 12.5 GB(刨去协议头, 取12GB) */
constexpr double DURATION_RDMA_FIXED = 7;
constexpr double DURATION_RDMA_BANDWIDTH_MB = 12 * 1000; // (魔鬼数字解释) 12 * 1000=12k

/* * 统一按照10GB计算, CCE reduce的固定开销按照0.6us计算
    (理论值0.5 + dim * 8) */
constexpr double DURATION_CCE_FIXED = 0.6;
constexpr double DURATION_CCE_BANDWIDTH_MB = 10 * 1000; // (魔鬼数字解释) 10 * 1000=10k

/* * 统一按1us算(片间1us, 片内0.5us), Notify Wait按0.02us估计 */
constexpr double DURATION_NOTIFY_RECORD = 1;
constexpr double DURATION_NOTIFY_WAIT = 0.02;

constexpr u64 MULTIPLIER_S2NS = 1000 * 1000 * 1000; // 秒转换成纳秒 1000 * 1000 * 1000
constexpr u64 MULTIPLIER_MS2NS = 1000 * 1000;       // 毫秒转换成纳秒 1000 * 1000
constexpr u64 MULTIPLIER_US2NS = 1000;              // 微妙转换成纳秒 1000
} // namespace hccl

#endif /* * TASK_PROFILING_H */
