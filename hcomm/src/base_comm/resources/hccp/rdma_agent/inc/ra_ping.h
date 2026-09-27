/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_PING_H
#define RA_PING_H

#include <pthread.h>
#include "hccp_ping.h"
#include "hccp_common.h"

struct RaPingHandle {
    enum ProtocolTypeT protocol;
    union PingDev dev;
    uint32_t bufferSize;

    struct RaPingOps *pingOps;
    pthread_mutex_t mutex;
    uint32_t taskCnt;
    uint32_t targetCnt;

    uint32_t devIndex;
    unsigned int phyId;
};

struct RaPingOps {
    int (*raPingInit)(struct RaPingHandle *pingHandle, struct PingInitAttr *initAttr, struct PingInitInfo *initInfo);
    int (*raPingTargetAdd)(struct RaPingHandle *pingHandle, struct PingTargetInfo target[], uint32_t num);
    int (*raPingTaskStart)(struct RaPingHandle *pingHandle, struct PingTaskAttr *attr);
    int (*raPingGetResults)(struct RaPingHandle *pingHandle, struct PingTargetResult target[], uint32_t *num);
    int (*raPingTargetDel)(struct RaPingHandle *pingHandle, struct PingTargetCommInfo target[], uint32_t num);
    int (*raPingTaskStop)(struct RaPingHandle *pingHandle);
    int (*raPingDeinit)(struct RaPingHandle *pingHandle);
};
#endif // RA_PING_H
