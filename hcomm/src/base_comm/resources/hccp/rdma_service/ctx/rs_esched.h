/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_ESCHED_H
#define RS_ESCHED_H

#include "rs_inner.h"

#define ESCHED_GRP_TS_HCCP 0
#define ESCHED_THREAD_ID_TS_HCCP 0
#define ESCHED_THREAD_TRY_TIME 100
#define ESCHED_THREAD_USLEEP_TIME 10000

struct RsEschedInfo {
    unsigned int threadStatus;
};

int RsEschedInit(struct rs_cb *rscb);
void RsEschedDeinit(enum ProtocolTypeT protocol);
#endif // RS_ESCHED_H
