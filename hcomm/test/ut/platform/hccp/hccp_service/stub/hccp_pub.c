/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <sys/time.h>
#include "hccp_pub.h"
#include "rs.h"
int HccpInit(unsigned int chipId, pid_t pid, int hdcType, unsigned int whiteListStatus) { return 0; }
int HccpDeinit(unsigned int chipId) { return 0; }

#define MS_PER_SECOND_F 1000.0
#define MS_PER_SECOND_I 1000

void RsGetCurTime(struct timeval* time)
{
    int ret;

    ret = gettimeofday(time, NULL);
    if (ret) {
        memset(time, 0, sizeof(struct timeval));
    }

    return;
}

void HccpTimeInterval(struct timeval* end_time, struct timeval* start_time, float* msec)
{
    /* if low position is sufficient, then borrow one from the high position */
    if (end_time->tv_usec < start_time->tv_usec) {
        end_time->tv_sec -= 1;
        end_time->tv_usec += MS_PER_SECOND_I * MS_PER_SECOND_I;
    }

    *msec = (end_time->tv_sec - start_time->tv_sec) * MS_PER_SECOND_F
            + (end_time->tv_usec - start_time->tv_usec) / MS_PER_SECOND_F;

    return;
}

void RsApiDeinit(void) {}

int RsApiInit(void) { return 0; }

enum ProductType RsGetProductType(int devId) { return PRODUCT_TYPE_950; }
