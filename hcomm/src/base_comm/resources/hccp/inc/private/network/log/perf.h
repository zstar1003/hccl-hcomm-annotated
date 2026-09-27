/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PERF_H
#define PERF_H

#include <time.h>
#include <stdint.h>
#include "user_log.h"
#include "config_log.h"

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

/* 单调时钟纳秒，用于计算调用耗时（不受系统时间调整影响） */
static inline uint64_t hccp_perf_clock_ns(void)
{
    struct timespec ts = {0, 0};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* 性能探针：包裹有返回值的调用，返回其原值，命中掩码时额外打印耗时 */
#define PERF_TRACE(moduleType, call)                                                                                   \
    ({                                                                                                                 \
        uint64_t _pt_start = 0;                                                                                        \
        uint64_t _pt_mask = HccpGetDebugConfig();                                                                      \
        if (unlikely(_pt_mask & (1ULL << (moduleType)))) {                                                             \
            _pt_start = hccp_perf_clock_ns();                                                                          \
        }                                                                                                              \
        __auto_type _pt_ret = (call);                                                                                  \
        if (unlikely(_pt_mask & (1ULL << (moduleType)))) {                                                             \
            uint64_t _pt_cost = hccp_perf_clock_ns() - _pt_start;                                                      \
            hccp_run_info("cost [%6llu] ns", (unsigned long long)_pt_cost);                                            \
        }                                                                                                              \
        _pt_ret;                                                                                                       \
    })

/* 性能探针：包裹无返回值（void）的调用，语义同 PERF_TRACE */
#define PERF_TRACE_VOID(moduleType, call)                                                                              \
    do {                                                                                                               \
        uint64_t _pt_start = 0;                                                                                        \
        uint64_t _pt_mask = HccpGetDebugConfig();                                                                      \
        if (unlikely(_pt_mask & (1ULL << (moduleType)))) {                                                             \
            _pt_start = hccp_perf_clock_ns();                                                                          \
        }                                                                                                              \
        (call);                                                                                                        \
        if (unlikely(_pt_mask & (1ULL << (moduleType)))) {                                                             \
            uint64_t _pt_cost = hccp_perf_clock_ns() - _pt_start;                                                      \
            hccp_run_info("cost [%6llu] ns", (unsigned long long)_pt_cost);                                            \
        }                                                                                                              \
    } while (0)

#endif /* PERF_H */
