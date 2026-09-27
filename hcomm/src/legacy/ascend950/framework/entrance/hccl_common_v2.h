/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMMON_V2_H
#define HCCL_COMMON_V2_H

#include <chrono>
#include "hccl/base.h"
#include "hccl_log_keywords.h"

constexpr s32 INVALID_INT = 0xFFFFFFFF;
constexpr u32 INVALID_VALUE_RANKSIZE = 0xFFFFFFFF; // rank size非法值

constexpr u32 MAX_MODULE_DEVICE_NUM = 65;

constexpr u32 ROOTINFO_INDENTIFIER_MAX_LENGTH = 128;
constexpr char HCCL_WORLD_GROUP[] = "hccl_world_group";
constexpr u32 GROUP_NAME_MAX_LEN = 127;                // 最大的group name 长度
constexpr u32 RANKTABLE_MAX_SIZE = 1024 * 1024 * 1024; // rankTable max size 1G

using HcclUs = std::chrono::steady_clock::time_point;
constexpr uint32_t HCCL_ALG_MESH = 0b1U;

using aclrtStream = void*;
using HcclComm = void*;

#define DURATION_US(x) (std::chrono::duration_cast<std::chrono::microseconds>(x))
#define TAKE_TIME_US(x, y) (DURATION_US(x) - DURATION_US(y))
#define TIME_NOW() ({ std::chrono::steady_clock::now(); })

enum class HcclTopoLevel {
    HCCL_TOPO_L0 = 0,
    HCCL_TOPO_L1,
    HCCL_TOPO_MAX,
};
#endif
