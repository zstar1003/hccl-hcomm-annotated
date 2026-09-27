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
#include <pwd.h>
#include <unistd.h>
#include "config_log.h"

#define CHK_PRT_RETURN(result, exeLog, ret) \
    do {                                    \
        if (result) {                       \
            exeLog;                         \
            return (ret);                   \
        }                                   \
    } while (0)

#define CHK_PRT_GOTO(result, exeLog, label) \
    do {                                    \
        if (result) {                       \
            exeLog;                         \
            goto label;                     \
        }                                   \
    } while (0)

/* HCCP module */
#define hccp_err(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define hccp_warn(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define hccp_info(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define hccp_dbg(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define hccp_event(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define hccp_event_with_user(fmt, args...)                                                          \
    fprintf(                                                                                        \
        stderr, "%s, pid(%d), %s(%d), user(%s): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, \
        getpwuid(getuid())->pw_name, ##args)
#define hccp_run_info(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define hccp_run_warn(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)

/* ROCE module */
#define roce_err(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define roce_warn(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define roce_info(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define roce_dbg(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define roce_event(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define roce_run_info(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)

/* NET module */
#define net_err(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define net_warn(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define net_info(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define net_dbg(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define net_run_err(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define net_run_warn(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define net_run_info(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)
#define net_run_dbg(fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)

/* dlog */
#define usr_dlog(moduleId, level, fmt, ...)

/* sub dlog */
#define usr_sub_dlog(moduleId, submodule, level, fmt, args...) \
    fprintf(stderr, "%s, pid(%d), %s(%d): " fmt "\n", __TIME__, getpid(), __func__, __LINE__, ##args)

enum {
    SLOG,  /**< Slog */
    IDEDD, /**< IDE daemon device */
    IDEDH, /**< IDE daemon host */
    HCCL,  /**< HCCL */
    FMK,   /**< Adapter */
    CCU,
    HIAIENGINE, /**< Matrix */
    DVPP,       /**< DVPP */
    RUNTIME,    /**< Runtime */
    CCE,        /**< CCE */
#if (OS_TYPE == LINUX)
    HDC, /**< HDC */
#else
    HDCL,
#endif
    DRV, /**< Driver */
    NET,
    MDCFUSION,     /**< Mdc fusion */
    MDCLOCATION,   /**< Mdc location */
    MDCPERCEPTION, /**< Mdc perception */
    MDCFSM,
    MDCCOMMON,
    MDCMONITOR,
    MDCBSWP,    /**< MDC base software platform */
    MDCDEFAULT, /**< MDC undefine */
    MDCSC,      /**< MDC spatial cognition */
    MDCPNC,
    MLL,      /**< abandon */
    DEVMM,    /**< Dlog memory managent */
    KERNEL,   /**< Kernel */
    LIBMEDIA, /**< Libmedia */
    CCECPU,   /**< aicpu shedule */
    ASCENDDK, /**< AscendDK */
    ROS,      /**< ROS */
    HCCP,
    ROCE,
    TEFUSION,
    PROFILING, /**< Profiling */
    DP,        /**< Data Preprocess */
    APP,       /**< User Application */
    TS,        /**< TS module */
    TSDUMP,    /**< TSDUMP module */
    AICPU,     /**< AICPU module */
    LP,        /**< LP module */
    TDT,       /**< tsdaemon or aicpu shedule */
    FE,
    MD,
    MB,
    ME,
    IMU,
    IMP,
    GE, /**< Fmk */
    MDCFUSA,
    CAMERA,
    ASCENDCL,
    TEEOS,
    ISP,
    SIS,
    HSM,
    DSS,
    PROCMGR,
    BBOX,
    AIVECTOR,
    TBE,
    FV,
    MDCMAP,
    TUNE,
    HSS, /**< helper */
    FFTS,
    OP,
    UDF,
    HICAID,
    TSYNC,
    CCU_USR,
    INVLID_MOUDLE_ID
};

#define DEBUG_LEVEL 0
#define INFO_LEVEL 1
#define WARN_LEVEL 2
#define ERROR_LEVEL 3

#endif
