/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file context_ascendc_log.h
 * \brief
 */

#ifndef __CONTEXT_ASCENDC_LOG_H__
#define __CONTEXT_ASCENDC_LOG_H__
#include <csignal>
#include "dlog_pub.h"

#define ASCENDC_MODULE_NAME static_cast<int32_t>(ASCENDCKERNEL)

#ifdef __cplusplus
extern "C" {
#endif

#define ASCENDC_ASSERT(cond, ret, msg) \
    do {                               \
        if (!(cond)) {                 \
            msg;                       \
            ret;                       \
        }                              \
    } while (0)

#define CXT_ASCENDC_LOGE(format, ...)                                                      \
    do {                                                                                   \
        dlog_error(ASCENDC_MODULE_NAME, "[%s] " format "\n", __FUNCTION__, ##__VA_ARGS__); \
    } while (0)

#define CXT_ASCENDC_LOGD(format, ...)                                                      \
    do {                                                                                   \
        dlog_debug(ASCENDC_MODULE_NAME, "[%s] " format "\n", __FUNCTION__, ##__VA_ARGS__); \
    } while (0)

#define CXT_ASCENDC_LOGW(format, ...)                                                     \
    do {                                                                                  \
        dlog_warn(ASCENDC_MODULE_NAME, "[%s] " format "\n", __FUNCTION__, ##__VA_ARGS__); \
    } while (0)

#define CXT_ASCENDC_LOGI(format, ...)                                                     \
    do {                                                                                  \
        dlog_info(ASCENDC_MODULE_NAME, "[%s] " format "\n", __FUNCTION__, ##__VA_ARGS__); \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif // __CONTEXT_ASCENDC_LOG_H__
