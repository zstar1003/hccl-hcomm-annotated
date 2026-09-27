/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_timer_adapt.h"

#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <execinfo.h>
#include <time.h>

#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

void DbgGetFuncName(void *funcAddr, char *funcName, size_t size)
{
    void *addrBuffer[2] = {NULL};
    char **funcStr = NULL;

    if (funcAddr == NULL) {
        *funcName = '\0';
        return;
    }

    addrBuffer[0] = funcAddr;
    funcStr = (void *)backtrace_symbols(addrBuffer, 1);
    if (funcStr == NULL) {
        *funcName = '\0';
        return;
    }

    (void)SchFuncNameDeal(funcName, size, funcStr[0]);

    free(funcStr);

    return;
}

/* 获取从系统启动到当前时刻的时间，单位: 毫秒 */
uint32_t SystimeGetMilliSec(uint64_t *milliSec)
{
    struct timespec curTime;

    if (clock_gettime(CLOCK_MONOTONIC, &curTime) != 0) {
        return -1;
    }
    *milliSec = ((uint64_t)curTime.tv_sec) * 1000 + (((uint64_t)curTime.tv_nsec) / 1000 / 1000); /* 时间转秒倍数1000 */

    return 0;
}

/* 从UDF移植时，封装UDF的基础函数 */
uint32_t SchFuncNameDeal(char *funcName, size_t size, const char *funcStr)
{
    uint32_t cnt = 0;
    uint32_t len;
    uint32_t i = 0;

    len = strlen(funcStr);

    for (; (*funcStr != '(') && (i < len); funcStr++) {
        i++;
    }

    if (i == len) {
        return -1;
    }

    funcStr++;
    while ((*funcStr != ')') && (*funcStr != '+')) {
        *funcName++ = (*funcStr++);
        cnt++;
        if (size == (cnt + 1)) {
            break;
        }
    }

    *funcName = '\0';
    return 0;
}

/* 当前时间戳字符串获取，格式为"yyyy-mm-dd hh:mm:ss" */
void UdfTimeStrGet(char *timeStr, size_t timeStrSize)
{
    time_t timeResult;
    struct tm localTime;

    (void)time(&timeResult);
    (void)localtime_r(&timeResult, &localTime);
    (void)strftime(timeStr, timeStrSize, "%Y-%m-%d %T", &localTime);

    return;
}

#ifdef __cplusplus
}
#endif
