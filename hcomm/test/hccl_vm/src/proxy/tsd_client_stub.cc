/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "TSD_STUB"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "sim_log.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

uint32_t TsdCapabilityGet(const uint32_t logicDeviceId, const int32_t type, const uint64_t ptr)
{
    (void)logicDeviceId;
    (void)type;
    (void)ptr;
    HCCL_VM_WARN("is empty");
    return 0;
}
#ifdef __cplusplus
}
#endif // __cplusplus
