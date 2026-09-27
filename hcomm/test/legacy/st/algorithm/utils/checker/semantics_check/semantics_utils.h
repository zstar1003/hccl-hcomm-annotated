/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SEMANTICS_UTILS_H
#define SEMANTICS_UTILS_H

#include "log.h"
#include "data_dumper.h"
#include "analysis_result.pb.h"

namespace checker {

std::string GenMsg(const char* fileName, s32 lineNum, const char* fmt, ...);

#define DUMP_AND_ERROR(...)                                        \
    do {                                                           \
        HCCL_ERROR(__VA_ARGS__);                                   \
        std::string msg = GenMsg(__FILE__, __LINE__, __VA_ARGS__); \
        DataDumper::Global()->AddErrorString(msg);                 \
    } while (0)

#define CHECKER_WARNING_LOG(format, ...)                                            \
    do {                                                                            \
        if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_WARN, INVLID_MOUDLE_ID))) {         \
            HCCL_LOG_PRINT(INVLID_MOUDLE_ID, HCCL_LOG_WARN, format, ##__VA_ARGS__); \
        }                                                                           \
    } while (0)

#define CHECKER_WARNING(...)                                       \
    do {                                                           \
        CHECKER_WARNING_LOG(__VA_ARGS__);                          \
        std::string msg = GenMsg(__FILE__, __LINE__, __VA_ARGS__); \
        DataDumper::Global()->AddErrorString(msg);                 \
    } while (0)
} // namespace checker

#endif
