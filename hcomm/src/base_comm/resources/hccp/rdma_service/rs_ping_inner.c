/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "user_log.h"
#include "rs_ping_inner.h"

uint32_t RsPingGetTripTime(struct RsPingTimestamp *timestamp)
{
    uint64_t localInterval = ((timestamp->tvSec4 - timestamp->tvSec1) * RS_PING_SEC_TO_USEC + timestamp->tvUsec4) -
                             timestamp->tvUsec1;
    uint64_t remoteInterval = ((timestamp->tvSec3 - timestamp->tvSec2) * RS_PING_SEC_TO_USEC + timestamp->tvUsec3) -
                              timestamp->tvUsec2;

    hccp_dbg("t1:{%llu, %llu} t4:{%llu, %llu} t4-t1:%llu, t2:{%llu, %llu} t3:{%llu, %llu} t3-t2:%llu, rtt:%u",
        timestamp->tvSec1, timestamp->tvUsec1, timestamp->tvSec4, timestamp->tvUsec4, localInterval, timestamp->tvSec2,
        timestamp->tvUsec2, timestamp->tvSec3, timestamp->tvUsec3, remoteInterval,
        (uint32_t)(localInterval - remoteInterval));

    return (uint32_t)(localInterval - remoteInterval);
}
