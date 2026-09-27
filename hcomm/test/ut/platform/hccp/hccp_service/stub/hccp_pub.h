/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __HCCP_PUB_H__
#define __HCCP_PUB_H__
#include <stdbool.h>
int HccpInit(unsigned int chipId, pid_t pid, int hdcType, unsigned int whiteListStatus);
int HccpDeinit(unsigned int chipId);
void RsGetCurTime(struct timeval* time);
void HccpTimeInterval(struct timeval* end_time, struct timeval* start_time, float* msec);
bool RsGetIsRdmaSupported(int devId);
#endif
