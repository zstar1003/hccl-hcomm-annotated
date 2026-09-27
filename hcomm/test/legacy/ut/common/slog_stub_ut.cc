/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlog_pub.h"
#include <stdint.h>
#include <syslog.h>
#include <stdio.h>
#include <cstdlib>

static int GetLogLevelFromEnv()
{
    const char* env = getenv("HCCL_UT_LOG_LEVEL");
    if (env != nullptr) {
        int val = atoi(env);
        if (val >= 0 && val <= 3) {
            return val;
        }
    }

    return DLOG_ERROR;
}
static int g_UT_LOG_LEVEL = GetLogLevelFromEnv();

int CheckLogLevel(int moduleId, int logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return 1;
}

int HcclCheckLogLevel(int moduleId, int logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return 1;
}

bool IsErrorToWarn() { return false; }

void SetErrWarnSwitch(bool flag) { (void)flag; }

void DlogRecord(int moduleId, int level, const char* fmt, ...)
{
    (void)moduleId;
    (void)fmt;
    if (level >= g_UT_LOG_LEVEL) {
        va_list list;
        va_start(list, fmt);
        vprintf(fmt, list);
        va_end(list);
    }
}

void DlogInner(int moduleId, int level, const char* fmt, ...)
{
    (void)moduleId;
    (void)fmt;
    if (level >= g_UT_LOG_LEVEL) {
        va_list list;
        va_start(list, fmt);
        vprintf(fmt, list);
        va_end(list);
    }
}
