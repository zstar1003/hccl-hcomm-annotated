/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_LLT_COMMON_H
#define HCCL_LLT_COMMON_H

#include "base.h"
#include <chrono>

namespace checker {

using char_t = char;
#ifdef HCCL_ALG_ANALYZER_DAVID
using RankId = s32;
#else
using RankId = u32;
#endif
using ServerId = u32;
using SuperPodId = u32;
using QId = u32;
using BlockId = u32;

using HcclUs = std::chrono::steady_clock::time_point;
#define DURATION_US(x) (std::chrono::duration_cast<std::chrono::microseconds>(x))
#define TIME_NOW() ({ std::chrono::steady_clock::now(); })

} // namespace checker

#endif
