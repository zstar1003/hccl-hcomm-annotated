/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROFFILING_COMMAND_HANDLE_H
#define PROFFILING_COMMAND_HANDLE_H
#include "hccl/hccl_types.h"
#include "stream_pub.h"
#include "hccl_common.h"
#include "common/aicpu_hccl_def.h"

namespace hccl {
using Prof_Status = uint32_t;
const Prof_Status PROF_SUCCESS = 0x0;
const Prof_Status PROF_FAILED = 0xFFFFFFFF;
#define ADPROF_TASK_TIME_L0 0x00000008ULL
#define ADPROF_TASK_TIME_L1 0x00000010ULL
#define ADPROF_TASK_TIME_L2 0x00000020ULL
int32_t DeviceCommandHandle(uint32_t profType, void* data, uint32_t len);
} // namespace hccl
#endif
