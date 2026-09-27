/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "vos_base.h"
#include "vos_typdef.h"
#include "vos_errno.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOS_SYSTIME_BASE_YEAR 1900
#define VOS_SYSTIME_TIME_FACTOR 1000
uint32_t VOS_SystimeGet(VOS_SYSTM_S *pstSystime)
{
    if (pstSystime == VOS_NULL) {
        return VOS_ERROR;
    }

    /* 获取系统时间 */
    struct timeval tv;
    struct timezone tz;
    int ret = gettimeofday(&tv, &tz);
    if (ret != 0) {
        return VOS_ERROR;
    }

    /* 获得日历时间 */
    struct tm temp;
    struct tm *t;
    t = gmtime_r(&tv.tv_sec, &temp);
    if (t == VOS_NULL) {
        return VOS_ERROR;
    }
    pstSystime->usYear = t->tm_year + VOS_SYSTIME_BASE_YEAR;
    pstSystime->ucMonth = t->tm_mon + 1;
    pstSystime->ucDate = t->tm_mday;
    pstSystime->ucHour = t->tm_hour;
    pstSystime->ucMinute = t->tm_min;
    pstSystime->ucSecond = t->tm_sec;
    pstSystime->uiMillSec = (uint32_t)(tv.tv_usec / VOS_SYSTIME_TIME_FACTOR);
    pstSystime->ucWeek = t->tm_wday;
    return VOS_OK;
}

#ifdef __cplusplus
}
#endif
