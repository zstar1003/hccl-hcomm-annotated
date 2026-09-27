/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_PING_H
#define RS_PING_H

#include "ra_rs_comm.h"
#include "hccp_ping.h"

RS_ATTRI_VISI_DEF int RsPingHandleInit(unsigned int chipId, int hdcType, unsigned int whiteListStatus);
RS_ATTRI_VISI_DEF int RsPingHandleDeinit(unsigned int chipId);
RS_ATTRI_VISI_DEF int RsPingInit(struct PingInitAttr *attr, struct PingInitInfo *info, unsigned int *devIndex);
RS_ATTRI_VISI_DEF int RsPingTargetAdd(struct RaRsDevInfo *rdev, struct PingTargetInfo *target);
RS_ATTRI_VISI_DEF int RsPingTaskStart(struct RaRsDevInfo *rdev, struct PingTaskAttr *attr);
RS_ATTRI_VISI_DEF int RsPingGetResults(struct RaRsDevInfo *rdev, struct PingTargetCommInfo target[], unsigned int *num,
    struct PingResultInfo result[]);
RS_ATTRI_VISI_DEF int RsPingTaskStop(struct RaRsDevInfo *rdev);
RS_ATTRI_VISI_DEF int RsPingTargetDel(struct RaRsDevInfo *rdev, struct PingTargetCommInfo target[], unsigned int *num);
RS_ATTRI_VISI_DEF int RsPingDeinit(struct RaRsDevInfo *rdev);
#endif // RS_PING_H
