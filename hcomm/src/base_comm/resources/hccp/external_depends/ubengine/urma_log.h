/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef URMA_LOG_H
#define URMA_LOG_H
#include <stdbool.h>
#include <urma_types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOG_FORMAT_IDX 4 /* index of 'format' of urma_log */
#define LOG_VA_ARG_IDX 5 /* index of variable argument of urma_log */

int urma_log_init(void);
void urma_getenv_log_level(void);
bool urma_log_drop(urma_vlog_level_t level);
void __attribute__((format(printf, LOG_FORMAT_IDX, LOG_VA_ARG_IDX))) urma_log(const char *function, int line,
    urma_vlog_level_t level, const char *format, ...);
const char *urma_get_level_print(urma_vlog_level_t level);
urma_vlog_level_t urma_log_get_level_from_string(const char *level_string);

#define URMA_LOG(l, ...)                                                                                               \
    if (!urma_log_drop(URMA_VLOG_LEVEL_##l)) {                                                                         \
        urma_log(__func__, __LINE__, URMA_VLOG_LEVEL_##l, __VA_ARGS__);                                                \
    }

#define URMA_LOG_INFO(...) URMA_LOG(INFO, __VA_ARGS__)

#define URMA_LOG_ERR(...) URMA_LOG(ERR, __VA_ARGS__)

#define URMA_LOG_WARN(...) URMA_LOG(WARNING, __VA_ARGS__)

#define URMA_LOG_DEBUG(...) URMA_LOG(DEBUG, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif
