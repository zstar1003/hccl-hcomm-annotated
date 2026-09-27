/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef USER_LOG_H
#define USER_LOG_H

#include <stdio.h>
#include <pwd.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <dlog_pub.h>

#define CHK_PRT_RETURN(result, exeLog, ret)                                                                            \
    do {                                                                                                               \
        if (result) {                                                                                                  \
            exeLog;                                                                                                    \
            return (ret);                                                                                              \
        }                                                                                                              \
    } while (0)

#define DEBUG_LEVEL 0
#define INFO_LEVEL 1
#define WARN_LEVEL 2
#define ERROR_LEVEL 3
#define EVENT_LEVEL 16

#ifdef DRV_HOST
#include "drv_log_user.h"
#define roce_err(fmt, ...) DRV_ERR(HAL_MODULE_TYPE_NET, fmt, ##__VA_ARGS__)
#define roce_warn(fmt, ...) DRV_WARN(HAL_MODULE_TYPE_NET, fmt, ##__VA_ARGS__)
#define roce_info(fmt, ...) DRV_INFO(HAL_MODULE_TYPE_NET, fmt, ##__VA_ARGS__)
#define roce_dbg(fmt, ...) DRV_DEBUG(HAL_MODULE_TYPE_NET, fmt, ##__VA_ARGS__)
#define roce_run_info(fmt, ...) DRV_NOTICE(HAL_MODULE_TYPE_NET, fmt, ##__VA_ARGS__)
#else
#ifdef LOG_HOST
#define HCCPDlogForC(moduleId, level, fmt, ...)                                                                        \
    do {                                                                                                               \
        if (CheckLogLevel(moduleId, level) == 1) {                                                                     \
            DlogRecord(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);                             \
        }                                                                                                              \
    } while (0)

/* HCCP module */
#define hccp_err(fmt, args...)                                                                                         \
    HCCPDlogForC(HCCP, ERROR_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_warn(fmt, args...)                                                                                        \
    HCCPDlogForC(HCCP, WARN_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_info(fmt, args...)                                                                                        \
    HCCPDlogForC(HCCP, INFO_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_dbg(fmt, args...)                                                                                         \
    HCCPDlogForC(HCCP, DEBUG_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)

#define hccp_run_err(fmt, args...)                                                                                     \
    HCCPDlogForC(HCCP | RUN_LOG_MASK, ERROR_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_warn(fmt, args...)                                                                                    \
    HCCPDlogForC(HCCP | RUN_LOG_MASK, WARN_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_info(fmt, args...)                                                                                    \
    HCCPDlogForC(HCCP | RUN_LOG_MASK, INFO_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_dbg(fmt, args...)                                                                                     \
    HCCPDlogForC(HCCP | RUN_LOG_MASK, DEBUG_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)

#define hccp_event(fmt, args...)                                                                                       \
    HCCPDlogForC(HCCP, EVENT_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)

/* NET module */
#define roce_err(fmt, args...) HCCPDlogForC(NET, ERROR_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_warn(fmt, args...) HCCPDlogForC(NET, WARN_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_info(fmt, args...) HCCPDlogForC(NET, INFO_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_dbg(fmt, args...) HCCPDlogForC(NET, DEBUG_LEVEL, "%s : " fmt, __func__, ##args)

#define roce_run_err(fmt, args...) HCCPDlogForC(NET | RUN_LOG_MASK, ERROR_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_run_warn(fmt, args...) HCCPDlogForC(NET | RUN_LOG_MASK, WARN_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_run_info(fmt, args...) HCCPDlogForC(NET | RUN_LOG_MASK, INFO_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_run_dbg(fmt, args...) HCCPDlogForC(NET | RUN_LOG_MASK, DEBUG_LEVEL, "%s : " fmt, __func__, ##args)

#define roce_event(fmt, args...) HCCPDlogForC(NET, EVENT_LEVEL, "%s : " fmt, __func__, ##args)
#else
#define hccp_dlog(moduleId, level, fmt, ...)                                                                           \
    do {                                                                                                               \
        if (CheckLogLevel(moduleId, level) == 1) {                                                                     \
            DlogRecord(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);                             \
        }                                                                                                              \
    } while (0)
/* HCCP module */
#define hccp_err(fmt, args...) hccp_dlog(HCCP, ERROR_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_warn(fmt, args...) hccp_dlog(HCCP, WARN_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_info(fmt, args...) hccp_dlog(HCCP, INFO_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_dbg(fmt, args...) hccp_dlog(HCCP, DEBUG_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)

#define hccp_run_err(fmt, args...)                                                                                     \
    hccp_dlog(HCCP | RUN_LOG_MASK, ERROR_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_warn(fmt, args...)                                                                                    \
    hccp_dlog(HCCP | RUN_LOG_MASK, WARN_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_info(fmt, args...)                                                                                    \
    hccp_dlog(HCCP | RUN_LOG_MASK, INFO_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_dbg(fmt, args...)                                                                                     \
    hccp_dlog(HCCP | RUN_LOG_MASK, DEBUG_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)

#define hccp_event(fmt, args...)                                                                                       \
    hccp_dlog(HCCP, EVENT_LEVEL, "tid:%d,%s : " fmt, syscall(__NR_gettid), __func__, ##args)

/* NET module */
#define roce_err(fmt, args...) hccp_dlog(NET, ERROR_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_warn(fmt, args...) hccp_dlog(NET, WARN_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_info(fmt, args...) hccp_dlog(NET, INFO_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_dbg(fmt, args...) hccp_dlog(NET, DEBUG_LEVEL, "%s : " fmt, __func__, ##args)

#define roce_run_err(fmt, args...) hccp_dlog(NET | RUN_LOG_MASK, ERROR_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_run_warn(fmt, args...) hccp_dlog(NET | RUN_LOG_MASK, WARN_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_run_info(fmt, args...) hccp_dlog(NET | RUN_LOG_MASK, INFO_LEVEL, "%s : " fmt, __func__, ##args)
#define roce_run_dbg(fmt, args...) hccp_dlog(NET | RUN_LOG_MASK, DEBUG_LEVEL, "%s : " fmt, __func__, ##args)

#define roce_event(fmt, args...) hccp_dlog(NET, "%s : " fmt, __func__, ##args)
#endif
#endif
#endif
