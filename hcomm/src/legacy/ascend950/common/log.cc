/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include <dlog_pub.h>

namespace Hccl {
#ifndef CCL_KERNEL
int HcclCheckLogLevel(int logLevel)
{
    if (logLevel == HCCL_LOG_RUN_INFO) {
        return CheckLogLevel(static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_INFO);
    } else if (logLevel == HCCL_LOG_RUN_WARN) {
        return CheckLogLevel(static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_WARN);
    } else {
        return CheckLogLevel(HCCL, logLevel);
    }
}

bool CheckDebugLogLevel() { return CheckLogLevel(HCCL, DLOG_DEBUG) == 1; }

bool CheckInfoLogLevel() { return CheckLogLevel(HCCL, DLOG_INFO) == 1; }

void CallDlogInvalidType(int level, int errCode, std::string file, int line)
{
    if (level == HCCL_LOG_RUN_INFO) {
        LOG_FUNC(
            static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_INFO, "[%s:%d][Log] Invalid LogType:Mod[%s],Type[%u]\n",
            file.c_str(), line, "HCCL", errCode);
    } else if (level == HCCL_LOG_RUN_WARN) {
        LOG_FUNC(
            static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_WARN, "[%s:%d][Log] Invalid LogType:Mod[%s],Type[%u]\n",
            file.c_str(), line, "HCCL", errCode);
    } else {
        LOG_FUNC(HCCL, level, "[%s:%d][Log] Invalid LogType:Mod[%s],Type[%u]\n", file.c_str(), line, "HCCL", errCode);
    }
}

void CallDlogNoSzFormat(int level, int errCode, std::string file, int line)
{
    if (level == HCCL_LOG_RUN_INFO) {
        LOG_FUNC(
            static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_INFO, "[%s:%d]errNo[0x%016llx] ptr of szFormat is null\n",
            file.c_str(), line, errCode);
    } else if (level == HCCL_LOG_RUN_WARN) {
        LOG_FUNC(
            static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_WARN, "[%s:%d]errNo[0x%016llx] ptr of szFormat is null\n",
            file.c_str(), line, errCode);
    } else {
        LOG_FUNC(HCCL, level, "[%s:%d]errNo[0x%016llx] ptr of szFormat is null\n", file.c_str(), line, errCode);
    }
}

void CallDlogMemError(int level, std::string file, int line)
{
    if (level == HCCL_LOG_RUN_INFO) {
        LOG_FUNC(
            static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_INFO, "[%s:%d]memset stack log buffer to 0 failed.\n",
            file.c_str(), line);
    } else if (level == HCCL_LOG_RUN_WARN) {
        LOG_FUNC(
            static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_WARN, "[%s:%d]memset stack log buffer to 0 failed.\n",
            file.c_str(), line);
    } else {
        LOG_FUNC(HCCL, level, "[%s:%d]memset stack log buffer to 0 failed.\n", file.c_str(), line);
    }
}

void CallDlogPrintError(int level, std::string file, int line)
{
    if (level == HCCL_LOG_RUN_INFO) {
        LOG_FUNC(static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_INFO, "[%s:%d]snprintf_s failed.\n", file.c_str(), line);
    } else if (level == HCCL_LOG_RUN_WARN) {
        LOG_FUNC(static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_WARN, "[%s:%d]snprintf_s failed.\n", file.c_str(), line);
    } else {
        LOG_FUNC(HCCL, level, "[%s:%d]snprintf_s failed.\n", file.c_str(), line);
    }
}

void CallDlog(int level, int sysCallBack, const char* buffer, std::string file, int line)
{
    if (level == HCCL_LOG_RUN_INFO) {
        LOG_FUNC(
            static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_INFO, "[%s:%d][%u]%s\n", file.c_str(), line, sysCallBack,
            buffer);
    } else if (level == HCCL_LOG_RUN_WARN) {
        LOG_FUNC(
            static_cast<u32>(HCCL) | RUN_LOG_MASK, DLOG_WARN, "[%s:%d][%u]%s\n", file.c_str(), line, sysCallBack,
            buffer);
    } else {
        LOG_FUNC(HCCL, level, "[%s:%d][%u]%s\n", file.c_str(), line, sysCallBack, buffer);
    }
}
#endif
} // namespace Hccl
