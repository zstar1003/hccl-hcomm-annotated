/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_ASSERT_H
#define BKF_ASSERT_H

#include "vos_assert.h"
#include "vrp_typdef.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*F_ASSERT_OUTFUNC)(const char *fmt, ...);
static F_ASSERT_OUTFUNC g_AssertOutFunc = NULL;

static inline void BkfSetLogFunc(F_ASSERT_OUTFUNC func)
{
    g_AssertOutFunc = func;
}

static inline void BkfLogAssertInfo(char *funcName, uint16_t line)
{
    if (g_AssertOutFunc) {
        g_AssertOutFunc("\nAssert: func:%s line:%u\n", funcName, line);
    }
    return;
}

#define BKF_ASSERT(expr)                                                                                               \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            VOS_ASSERT(0);                                                                                             \
            BkfLogAssertInfo((char *)__func__, __LINE__);                                                              \
        }                                                                                                              \
    } while (0)

#define BKF_ASSERT_WITHLOG(expr, file, line, log)                                                                      \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            VOS_ASSERT(0);                                                                                             \
        }                                                                                                              \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif
