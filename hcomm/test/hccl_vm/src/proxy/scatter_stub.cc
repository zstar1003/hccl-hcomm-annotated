/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 为确保Scatter正常执行打的临时桩，需要适时移除

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "SCATTER_STUB"

#include <cstdio>
#include <cstring>

#include "hccl/dtype_common.h"
#include "sim_log.h"

namespace ops_hccl {
bool RunIndependentOpExpansion(DevType deviceType)
{
    (void)deviceType;
    HCCL_VM_INFO("return true");
    return true;
}

bool IsAiCpuMode(DevType deviceType, u32 rankSize)
{
    (void)deviceType;
    (void)rankSize;
    const char* env = getenv("HCCL_OP_EXPANSION_MODE");
    HCCL_VM_INFO("env = {}", env);
    // 判断是否是AI_CPU
    if (env != nullptr && strcmp(env, "AI_CPU") == 0) {
        return true;
    }
    return false;
}
} // namespace ops_hccl
