/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_ERROR_MANAGER_PUB_H
#define HCCL_INC_ADAPTER_ERROR_MANAGER_PUB_H

#include <cstdint>
#include <string>
#include <vector>

constexpr char HCCL_RPT_CODE[] = "EI9999"; // 每个组件有固定的标识号码
constexpr size_t const LIMIT_PER_MESSAGE = 1024U;
using ErrContextPub = struct Context_Pub {
    uint64_t work_stream_id = 0; // default value 0, invalid value
    uint64_t reserved[7] = {0};
};

ErrContextPub hrtErrMGetErrorContextPub(void);
void hrtErrMSetErrorContextPub(ErrContextPub errorContextPub);

// 以下方法不直接使用只在宏定义中使用
__attribute__((weak)) void
RptInputErr(std::string error_code, std::vector<std::string> key, std::vector<std::string> value);

__attribute__((weak)) void RptInnerErrPrt(const char* fmt, ...);

#define RPT_INPUT_ERR(result, error_code, key, value)     \
    do {                                                  \
        if (UNLIKELY(result) && RptInputErr != nullptr) { \
            RptInputErr(error_code, key, value);          \
        }                                                 \
    } while (0)

#define RPT_ENV_ERR(result, error_code, key, value)       \
    do {                                                  \
        if (UNLIKELY(result) && RptInputErr != nullptr) { \
            RptInputErr(error_code, key, value);          \
        }                                                 \
    } while (0)

#define RPT_INNER_ERR_PRT(fmt, ...)             \
    do {                                        \
        if (RptInnerErrPrt != nullptr) {        \
            RptInnerErrPrt(fmt, ##__VA_ARGS__); \
        }                                       \
    } while (0)

#define RPT_CALL_ERR(result, fmt, ...)                       \
    do {                                                     \
        if (UNLIKELY(result) && RptInnerErrPrt != nullptr) { \
            RptInnerErrPrt(fmt, ##__VA_ARGS__);              \
        }                                                    \
    } while (0)

#define RPT_CALL_ERR_PRT(fmt, ...)              \
    do {                                        \
        if (RptInnerErrPrt != nullptr) {        \
            RptInnerErrPrt(fmt, ##__VA_ARGS__); \
        }                                       \
    } while (0)

#endif
