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
#include <sys/epoll.h>
#include <errno.h>
#include "hccp_ping.h"
#include "ra_rs_comm.h"
#include "ra_comm.h"

int RsPingHandleInit(unsigned int chipId, int hdc_type, unsigned int white_list_status) { return 0; }

int RsPingHandleDeinit(unsigned int chipId) { return 0; }

int RsPingInit(struct PingInitAttr* attr, struct PingInitInfo* info, unsigned int* rdevIndex) { return 0; }

int RsPingTargetAdd(struct RaRsDevInfo* rdev, struct PingTargetInfo* target) { return 0; }

int RsPingTaskStart(struct RaRsDevInfo* rdev, struct PingTaskAttr* attr) { return 0; }

int RsPingGetResults(
    struct RaRsDevInfo* rdev, struct PingTargetCommInfo target[], unsigned int* num, struct PingResultInfo result[])
{
    return 0;
}

int RsPingTaskStop(struct RaRsDevInfo* rdev) { return 0; }

int RsPingTargetDel(struct RaRsDevInfo* rdev, struct PingTargetCommInfo target[], unsigned int* num) { return 0; }

int RsPingDeinit(struct RaRsDevInfo* rdev) { return 0; }
