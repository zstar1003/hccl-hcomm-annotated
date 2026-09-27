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
#undef HCCL_VM_MODULE
#define HCCL_VM_MODULE "HCCL_LOG"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>

#include "sim_log.h"

constexpr size_t LOG_MSG_BUFFER_SIZE = 1024;

std::once_flag log_initialized_flag;

#ifdef __cplusplus
extern "C" {
#endif

int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return static_cast<int32_t>(true);
}

void DlogPrintStub(int level, int moduleId, const char* msgBuffer)
{
    std::call_once(log_initialized_flag, []() {
        if (g_logger != nullptr) {
            return;
        }
        LogConfig config = LoadLogConfig("proxy");
        InitLogger(config);
    });

    switch (level) {
        case 0:
            HCCL_VM_DEBUG("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        case 1:
            HCCL_VM_INFO("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        case 2:
            HCCL_VM_WARN("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        case 3:
            HCCL_VM_ERROR("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        default:
            HCCL_VM_TRACE("[{}]{}", moduleId & 0xFF, msgBuffer);
    }
}

void DlogRecord(int moduleId, int level, const char* fmt, ...)
{
    char buffer[LOG_MSG_BUFFER_SIZE] = {0};
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    DlogPrintStub(level, moduleId, buffer);
}

void _Z10DlogRecordiiPKcz(int moduleId, int level, const char* fmt, ...)
{
    char buffer[LOG_MSG_BUFFER_SIZE] = {0};
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    DlogPrintStub(level, moduleId, buffer);
}

#ifdef __cplusplus
}
#endif
