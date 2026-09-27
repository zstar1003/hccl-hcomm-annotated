/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_SLOG_H
#define NET_SLOG_H
#include <stdio.h>
#include <dlfcn.h>
#include <pwd.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <dlog_pub.h>

#define DEBUG_LEVEL 0
#define INFO_LEVEL 1
#define WARN_LEVEL 2
#define ERROR_LEVEL 3
#define EVENT_LEVEL 16

struct nslb_slog_api_ops {
    /* slog ops api */
    void (*DlogRecord)(int moduleId, int level, const char *fmt, ...);
    int (*CheckLogLevel)(int moduleId, int logLevel);
};

extern struct nslb_slog_api_ops g_nslb_slog_api_ops;

int NetCoOpenSlogSo(void);
int NetCoCloseSlogSo(void);
int NetCoSlogApiInit(void);

#define NSLB_DLOG(isRunLog, level, fmt, ...)                                                                           \
    do {                                                                                                               \
        if (g_nslb_slog_api_ops.DlogRecord != NULL && g_nslb_slog_api_ops.CheckLogLevel != NULL) {                     \
            if (g_nslb_slog_api_ops.CheckLogLevel(HCCP, level) == 1) {                                                 \
                g_nslb_slog_api_ops.DlogRecord(HCCP, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);         \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

/* NETCO module */
#define nslb_err(fmt, args...) NSLB_DLOG(0, ERROR_LEVEL, "%s(%d) : " fmt, __func__, __LINE__, ##args)
#define nslb_warn(fmt, args...) NSLB_DLOG(0, WARN_LEVEL, "%s(%d) : " fmt, __func__, __LINE__, ##args)
#define nslb_info(fmt, args...) NSLB_DLOG(0, INFO_LEVEL, "%s(%d) : " fmt, __func__, __LINE__, ##args)
#define nslb_dbg(fmt, args...) NSLB_DLOG(0, DEBUG_LEVEL, "%s(%d) : " fmt, __func__, __LINE__, ##args)

#define nslb_run_err(fmt, args...) NSLB_DLOG(1, ERROR_LEVEL, "%s(%d) : " fmt, __func__, __LINE__, ##args)
#define nslb_run_warn(fmt, args...) NSLB_DLOG(1, WARN_LEVEL, "%s(%d) : " fmt, __func__, __LINE__, ##args)
#define nslb_run_info(fmt, args...) NSLB_DLOG(1, INFO_LEVEL, "%s(%d) : " fmt, __func__, __LINE__, ##args)
#define nslb_run_dbg(fmt, args...) NSLB_DLOG(1, DEBUG_LEVEL, "%s(%d) : " fmt, __func__, __LINE__, ##args)

#endif