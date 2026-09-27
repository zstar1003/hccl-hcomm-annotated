/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_ADDR_INFO_PERF_H
#define TOPO_ADDR_INFO_PERF_H

#include <time.h>
#include "topo_addr_info_log.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── 耗时打点宏（纳秒精度，通过 TOPO_INFO 输出） ── */

#define TOPO_PERF_BEGIN(name)         \
    struct timespec _tp_##name##_bgn; \
    clock_gettime(CLOCK_MONOTONIC_RAW, &_tp_##name##_bgn)

#define TOPO_PERF_END(name)                                                                            \
    do {                                                                                               \
        struct timespec _tp_##name##_end;                                                              \
        clock_gettime(CLOCK_MONOTONIC_RAW, &_tp_##name##_end);                                         \
        long long _tp_##name##_ns = (_tp_##name##_end.tv_sec - _tp_##name##_bgn.tv_sec) * 1000000000LL \
                                    + (_tp_##name##_end.tv_nsec - _tp_##name##_bgn.tv_nsec);           \
        TOPO_INFO("[perf] " #name " took %lld ns", _tp_##name##_ns);                                   \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif /* TOPO_ADDR_INFO_PERF_H */
