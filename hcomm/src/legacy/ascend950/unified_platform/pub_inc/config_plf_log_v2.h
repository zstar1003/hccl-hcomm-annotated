/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONFIG_PLF_LOG_V2_H
#define CONFIG_PLF_LOG_V2_H

#include "log.h"
#include "plf_debug_config.h"

// config要求传入宏名字作为日志打印关键字，不可以传入其他变量或常量
#define PLF_CONFIG_INFO(config, format, ...)                                                                       \
    do {                                                                                                           \
        if (UNLIKELY((Hccl::GetPlfDebugConfigValue() & (config)) != 0)) {                                          \
            const char* configName = #config;                                                                      \
            LOG_FUNC(                                                                                              \
                static_cast<u32>(HCCL) | RUN_LOG_MASK, HCCL_LOG_INFO, "[%s:%d] [%u] [%s] [%s]: " format, __FILE__, \
                __LINE__, static_cast<u32>(syscall(SYS_gettid)), configName, __func__, ##__VA_ARGS__);             \
        } else {                                                                                                   \
            HCCL_INFO(format, ##__VA_ARGS__);                                                                      \
        }                                                                                                          \
    } while (0)

#define PLF_CONFIG_DEBUG(config, format, ...)                                                                 \
    do {                                                                                                      \
        if (UNLIKELY((Hccl::GetPlfDebugConfigValue() & (config)) != 0)) {                                     \
            const char* configName = #config;                                                                 \
            LOG_FUNC(                                                                                         \
                static_cast<u32>(HCCL) | RUN_LOG_MASK, HCCL_LOG_INFO, "[%s:%d] [%u] [%s]: " format, __FILE__, \
                __LINE__, static_cast<u32>(syscall(SYS_gettid)), configName, ##__VA_ARGS__);                  \
        } else {                                                                                              \
            HCCL_DEBUG(format, ##__VA_ARGS__);                                                                \
        }                                                                                                     \
    } while (0)

#endif // CONFIG_PLF_LOG_V2_H
