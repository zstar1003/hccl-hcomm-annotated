/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_ADP_PING_H
#define RA_ADP_PING_H

#include "hccp_ping.h"
#include "ra_rs_comm.h"

struct RsPingOps {
    int (*pingInit)(struct PingInitAttr *attr, struct PingInitInfo *info, unsigned int *devIndex);
    int (*pingTargetAdd)(struct RaRsDevInfo *rdev, struct PingTargetInfo *target);
    int (*pingTaskStart)(struct RaRsDevInfo *rdev, struct PingTaskAttr *attr);
    int (*pingGetResults)(struct RaRsDevInfo *rdev, struct PingTargetCommInfo target[], unsigned int *num,
        struct PingResultInfo result[]);
    int (*pingTaskStop)(struct RaRsDevInfo *rdev);
    int (*pingTargetDel)(struct RaRsDevInfo *rdev, struct PingTargetCommInfo target[], unsigned int *num);
    int (*pingDeinit)(struct RaRsDevInfo *rdev);
};

int RaRsPingInit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsPingTargetAdd(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsPingTaskStart(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsPingGetResults(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsPingTaskStop(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsPingTargetDel(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsPingDeinit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);

#endif // RA_ADP_PING_H
