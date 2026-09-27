/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_PROF_CYCLE_TIME_H
#define HCOMM_PROF_CYCLE_TIME_H

#include <cstdint>

namespace hcomm {
/**
 * @brief 获取 prof 系统时间戳，封装 DfxDlProfFunction 的调用，
 *        避免上层模块直接依赖 coll_communicator_mgr 头文件。
 * @return 当前系统时间戳
 */
uint64_t GetProfCycleTime();
} // namespace hcomm

#endif // HCOMM_PROF_CYCLE_TIME_H
