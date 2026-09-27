/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_ADDR_INFO_LOG_H
#define TOPO_ADDR_INFO_LOG_H

#include "topo_addr_info_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 日志模块 ID：HCCL (3) */
#define TOPO_MODULE_ID (3)

/* 日志级别常量 */
#define TOPO_LOG_DEBUG 0
#define TOPO_LOG_INFO 1
#define TOPO_LOG_WARN 2
#define TOPO_LOG_ERROR 3

/* 日志掩码：INFO/WARN 需带 RUN_LOG_MASK 路由到 run/ 目录 */
#define RUN_LOG_MASK (0x01000000U)
#define TOPO_RUN_MASK (TOPO_MODULE_ID | RUN_LOG_MASK) /* 0x01000003 */

/* ── 日志函数指针（由 TopoLogInit 通过 dlopen + dlsym 填充） ── */
extern void (*g_topo_DlogRecord)(int moduleId, int level, const char* fmt, ...);
extern int (*g_topo_CheckLogLevel)(int moduleId, int logLevel);

/**
 * 初始化日志：dlopen("libunified_dlog.so") + dlsym
 * 可在进程启动后任一时间调用，失败不会阻塞后续流程。
 * 多次调用安全，只有首次生效。
 */
void TopoLogInit(void);

/* ── 日志宏（函数指针为 NULL 时安全跳过） ── */

#define TOPO_ERR(fmt, ...)                                                                                        \
    do {                                                                                                          \
        if (g_topo_DlogRecord != NULL) {                                                                          \
            g_topo_DlogRecord(                                                                                    \
                TOPO_MODULE_ID, TOPO_LOG_ERROR, "[%s:%d][%s] " fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
        }                                                                                                         \
    } while (0)

#define TOPO_WARN(fmt, ...)                                                                                     \
    do {                                                                                                        \
        if (g_topo_DlogRecord != NULL && g_topo_CheckLogLevel != NULL                                           \
            && g_topo_CheckLogLevel(TOPO_RUN_MASK, TOPO_LOG_WARN)) {                                            \
            g_topo_DlogRecord(                                                                                  \
                TOPO_RUN_MASK, TOPO_LOG_WARN, "[%s:%d][%s] " fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
        }                                                                                                       \
    } while (0)

#define TOPO_INFO(fmt, ...)                                                                                     \
    do {                                                                                                        \
        if (g_topo_DlogRecord != NULL && g_topo_CheckLogLevel != NULL                                           \
            && g_topo_CheckLogLevel(TOPO_RUN_MASK, TOPO_LOG_INFO)) {                                            \
            g_topo_DlogRecord(                                                                                  \
                TOPO_RUN_MASK, TOPO_LOG_INFO, "[%s:%d][%s] " fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
        }                                                                                                       \
    } while (0)

/* ── 检查并返回宏（依赖 TOPO_ERR 日志） ── */

/* 条件为真则打 ERROR 日志并返回指定 TopoAddrResult */
#define TOPO_RET_IF(cond, ret, fmt, ...)  \
    do {                                  \
        if ((cond)) {                     \
            TOPO_ERR(fmt, ##__VA_ARGS__); \
            return (ret);                 \
        }                                 \
    } while (0)

/* 空指针检查 (返回 TOPO_ERR_PTR) */
#define TOPO_RET_PTR_NULL(ptr) TOPO_RET_IF((ptr) == NULL, TOPO_ERR_PTR, "ptr [%s] is NULL", #ptr)

/* 安全函数返回值检查 (sprintf_s / strcpy_s < 0 则返回 TOPO_ERR_INTERNAL) */
#define TOPO_RET_SAFE(call)                             \
    do {                                                \
        if ((call) < 0) {                               \
            TOPO_ERR("safety func [%s] failed", #call); \
            return TOPO_ERR_INTERNAL;                   \
        }                                               \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif /* TOPO_ADDR_INFO_LOG_H */
