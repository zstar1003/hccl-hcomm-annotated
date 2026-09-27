/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <unistd.h>
#include <sys/syscall.h>
#include "slog.h"
#include "slog_api.h"

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

#ifdef DRV_HOST
#include "dmc_user_interface.h"
#define roce_err(fmt, ...) DRV_ERR(HAL_MODULE_TYPE_NET, fmt, ##__VA_ARGS__)
#define roce_warn(fmt, ...) DRV_WARN(HAL_MODULE_TYPE_NET, fmt, ##__VA_ARGS__)
#define roce_info(fmt, ...) DRV_INFO(HAL_MODULE_TYPE_NET, fmt, ##__VA_ARGS__)
#define roce_dbg(fmt, ...) DRV_DEBUG(HAL_MODULE_TYPE_NET, fmt, ##__VA_ARGS__)
#define roce_run_info(fmt, ...) DRV_NOTICE(HAL_MODULE_TYPE_NET, fmt, ##__VA_ARGS__)

#define net_err roce_err
#define net_warn roce_warn
#define net_info roce_info
#define net_dbg roce_dbg
#define net_run_info roce_run_info
#else
#ifdef LOG_HOST
/* fix slog compatibility issue: rs will be packaged into runtime/opp and deployed on the device */
#define UsrDlogForC(moduleId, level, fmt, ...)                                                                         \
    do {                                                                                                               \
        if (CheckLogLevelForC(moduleId, level) == 1) {                                                                 \
            if (DlogRecordForC == NULL) {                                                                              \
                DlogInnerForC(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);                      \
            } else {                                                                                                   \
                DlogRecordForC(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);                     \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

/* HCCP module */
#define hccp_err(fmt, args...) UsrDlogForC(HCCP, ERROR_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_warn(fmt, args...) UsrDlogForC(HCCP, WARN_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_info(fmt, args...) UsrDlogForC(HCCP, INFO_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_dbg(fmt, args...) UsrDlogForC(HCCP, DEBUG_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_err(fmt, args...)                                                                                     \
    UsrDlogForC(HCCP | RUN_LOG_MASK, ERROR_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_warn(fmt, args...)                                                                                    \
    UsrDlogForC(HCCP | RUN_LOG_MASK, WARN_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_info(fmt, args...)                                                                                    \
    UsrDlogForC(HCCP | RUN_LOG_MASK, INFO_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_dbg(fmt, args...)                                                                                     \
    UsrDlogForC(HCCP | RUN_LOG_MASK, DEBUG_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)

/* ROCE module */
#define roce_err(fmt, args...) UsrDlogForC(ROCE, ERROR_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_warn(fmt, args...) UsrDlogForC(ROCE, WARN_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_info(fmt, args...) UsrDlogForC(ROCE, INFO_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_dbg(fmt, args...) UsrDlogForC(ROCE, DEBUG_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_run_err(fmt, args...) UsrDlogForC(ROCE | RUN_LOG_MASK, ERROR_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_run_warn(fmt, args...) UsrDlogForC(ROCE | RUN_LOG_MASK, WARN_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_run_info(fmt, args...) UsrDlogForC(ROCE | RUN_LOG_MASK, INFO_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_run_dbg(fmt, args...) UsrDlogForC(ROCE | RUN_LOG_MASK, DEBUG_LEVEL, "%s: " fmt, __func__, ##args)

/* NET module */
#define net_err(fmt, args...) UsrDlogForC(NET, ERROR_LEVEL, "%s: " fmt, __func__, ##args)
#define net_warn(fmt, args...) UsrDlogForC(NET, WARN_LEVEL, "%s: " fmt, __func__, ##args)
#define net_info(fmt, args...) UsrDlogForC(NET, INFO_LEVEL, "%s: " fmt, __func__, ##args)
#define net_dbg(fmt, args...) UsrDlogForC(NET, DEBUG_LEVEL, "%s: " fmt, __func__, ##args)
#define net_run_err(fmt, args...) UsrDlogForC(NET | RUN_LOG_MASK, ERROR_LEVEL, "%s: " fmt, __func__, ##args)
#define net_run_warn(fmt, args...) UsrDlogForC(NET | RUN_LOG_MASK, WARN_LEVEL, "%s: " fmt, __func__, ##args)
#define net_run_info(fmt, args...) UsrDlogForC(NET | RUN_LOG_MASK, INFO_LEVEL, "%s: " fmt, __func__, ##args)
#define net_run_dbg(fmt, args...) UsrDlogForC(NET | RUN_LOG_MASK, DEBUG_LEVEL, "%s: " fmt, __func__, ##args)
#else
/* fix slog compatibility issue: rs will be packaged into runtime/opp and deployed on the device */
#define usr_dlog(moduleId, level, fmt, ...)                                                                            \
    do {                                                                                                               \
        if (CheckLogLevel(moduleId, level) == 1) {                                                                     \
            if (DlogRecord == NULL) {                                                                                  \
                DlogInner(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);                          \
            } else {                                                                                                   \
                DlogRecord(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);                         \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

/* sub dlog */
#define usr_sub_dlog(moduleId, submodule, level, fmt, ...)                                                             \
    do {                                                                                                               \
        if (CheckLogLevel(moduleId, level) == 1) {                                                                     \
            if (DlogRecord == NULL) {                                                                                  \
                DlogInner(moduleId, level, "[%s:%d][%s]" fmt, __FILE__, __LINE__, submodule, ##__VA_ARGS__);           \
            } else {                                                                                                   \
                /* DlogSub is a macro wrapper of DlogRecord */                                                         \
                DlogSub(moduleId, submodule, level, fmt, ##__VA_ARGS__);                                               \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

/* HCCP module */
#define hccp_err(fmt, args...) usr_dlog(HCCP, ERROR_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_warn(fmt, args...) usr_dlog(HCCP, WARN_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_info(fmt, args...) usr_dlog(HCCP, INFO_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_dbg(fmt, args...) usr_dlog(HCCP, DEBUG_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_err(fmt, args...)                                                                                     \
    usr_dlog(HCCP | RUN_LOG_MASK, ERROR_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_warn(fmt, args...)                                                                                    \
    usr_dlog(HCCP | RUN_LOG_MASK, WARN_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_info(fmt, args...)                                                                                    \
    usr_dlog(HCCP | RUN_LOG_MASK, INFO_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)
#define hccp_run_dbg(fmt, args...)                                                                                     \
    usr_dlog(HCCP | RUN_LOG_MASK, DEBUG_LEVEL, "tid:%d,%s: " fmt, syscall(__NR_gettid), __func__, ##args)

/* ROCE module */
#define roce_err(fmt, args...) usr_dlog(ROCE, ERROR_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_warn(fmt, args...) usr_dlog(ROCE, WARN_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_info(fmt, args...) usr_dlog(ROCE, INFO_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_dbg(fmt, args...) usr_dlog(ROCE, DEBUG_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_run_err(fmt, args...) usr_dlog(ROCE | RUN_LOG_MASK, ERROR_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_run_warn(fmt, args...) usr_dlog(ROCE | RUN_LOG_MASK, WARN_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_run_info(fmt, args...) usr_dlog(ROCE | RUN_LOG_MASK, INFO_LEVEL, "%s: " fmt, __func__, ##args)
#define roce_run_dbg(fmt, args...) usr_dlog(ROCE | RUN_LOG_MASK, DEBUG_LEVEL, "%s: " fmt, __func__, ##args)

/* fix slog compatibility issue: NET module log with ROCE module */
#define net_err(fmt, args...) usr_dlog(ROCE, ERROR_LEVEL, "%s: " fmt, __func__, ##args)
#define net_warn(fmt, args...) usr_dlog(ROCE, WARN_LEVEL, "%s: " fmt, __func__, ##args)
#define net_info(fmt, args...) usr_dlog(ROCE, INFO_LEVEL, "%s: " fmt, __func__, ##args)
#define net_dbg(fmt, args...) usr_dlog(ROCE, DEBUG_LEVEL, "%s: " fmt, __func__, ##args)
#define net_run_err(fmt, args...) usr_dlog(ROCE | RUN_LOG_MASK, ERROR_LEVEL, "%s: " fmt, __func__, ##args)
#define net_run_warn(fmt, args...) usr_dlog(ROCE | RUN_LOG_MASK, WARN_LEVEL, "%s: " fmt, __func__, ##args)
#define net_run_info(fmt, args...) usr_dlog(ROCE | RUN_LOG_MASK, INFO_LEVEL, "%s: " fmt, __func__, ##args)
#define net_run_dbg(fmt, args...) usr_dlog(ROCE | RUN_LOG_MASK, DEBUG_LEVEL, "%s: " fmt, __func__, ##args)
#endif /* LOG_HOST */
#endif /* DRV_HOST */
#endif /* USER_LOG_H */
