/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_prog.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

STATIC const int32_t g_BkfFibo[] = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 988, 1596};

int32_t BkfFibo32Get(int32_t idx)
{
    if ((idx >= 0) && (idx < (int32_t)BKF_GET_ARY_COUNT(g_BkfFibo))) {
        return g_BkfFibo[idx];
    } else {
        return -1;
    }
}

STATIC int32_t BkfFibo32GetValid(int32_t *itorInOut, int32_t max, int32_t i)
{
    int32_t temp = i;

    while (g_BkfFibo[temp] > max) {
        temp--;
        if (temp < 0) {
            return -1;
        }
    }

    *itorInOut = temp;
    return g_BkfFibo[temp];
}
int32_t BkfFibo32GetNextValid(int32_t *itorInOut, int32_t max)
{
    int32_t i;

    if ((itorInOut == VOS_NULL) || (max < g_BkfFibo[0])) {
        return -1;
    }

    i = *itorInOut;
    if (i < 0) {
        i = 0;
    } else if (i < ((int32_t)BKF_GET_ARY_COUNT(g_BkfFibo) - 1)) {
        i++;
    } else {
        i = (int32_t)BKF_GET_ARY_COUNT(g_BkfFibo) - 1;
    }

    return BkfFibo32GetValid(itorInOut, max, i);
}

int32_t BkfFibo32GetPrevValid(int32_t *itorInOut, int32_t max)
{
    int32_t i;

    if ((itorInOut == VOS_NULL) || (max < g_BkfFibo[0])) {
        return -1;
    }

    i = *itorInOut;
    if (i > ((int32_t)BKF_GET_ARY_COUNT(g_BkfFibo) - 1)) {
        i = (int32_t)(BKF_GET_ARY_COUNT(g_BkfFibo) - 1);
    } else if (i > 0) {
        i--;
    } else {
        i = 0;
    }

    return BkfFibo32GetValid(itorInOut, max, i);
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
