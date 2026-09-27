/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sys/time.h>
#include "vos_base.h"
#include "vos_errno.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CPUTICK_PER_SEC 1000000
#define CPUTICK_BITNUM32 (32)
uint32_t VOS_CpuTick2MsEx(VOS_CPUTICK_S *cputick, uint32_t *milliSecsHigh, uint32_t *milliSecsLow)
{
    if (cputick == NULL || milliSecsHigh == NULL || milliSecsLow == NULL) {
        return VOS_ERROR;
    }
    uint64_t ullTick = cputick->uiHigh;
    ullTick = ullTick << CPUTICK_BITNUM32;
    ullTick += cputick->uiLow;
    uint64_t ullMs = ullTick / CPUTICK_PER_SEC;
    *milliSecsHigh = (uint32_t)(ullMs >> CPUTICK_BITNUM32);
    *milliSecsLow = (uint32_t)(ullMs & 0xffffffff);
    return VOS_OK;
}

void VOS_GetCpuTick(uint32_t *puiLow, uint32_t *puiHigh)
{
    if (puiLow == NULL || puiHigh == NULL) {
        return;
    }

    struct timeval tv;
    gettimeofday(&tv, NULL);

    unsigned long long nowtick = (unsigned long long)tv.tv_sec * CPUTICK_PER_SEC;
    nowtick += ((unsigned long long)tv.tv_usec);
    *puiHigh = nowtick >> CPUTICK_BITNUM32;
    *puiLow = nowtick & 0xffffffff;
    return;
}

#ifdef __cplusplus
}
#endif
