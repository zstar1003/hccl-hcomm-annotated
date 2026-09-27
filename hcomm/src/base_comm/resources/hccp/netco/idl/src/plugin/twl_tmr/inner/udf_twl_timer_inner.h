/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_TWL_TIMER_INNER_H
#define UDF_TWL_TIMER_INNER_H

#include <stdint.h>
#include <stdbool.h>
#include "udf_twl_timer.h"
#include "udf_base_timer.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32_t UdfTimerInstGetRemainTime(UdfTmrInstHandle tmrInst, uint32_t *remainMs);

uint32_t UdfTimerInstStart(UdfTmrInstHandle tmrInst);

uint32_t UdfTimerInstStop(UdfTmrInstHandle tmrInst);

uint32_t UdfTimerInstCreateNotStart(UdfTwlTimerHandle handle, TimerDestructProc func, UdfTmrInstParam *timerParam,
    UdfTmrInstHandle *tmrInstHandle);

#ifdef __cplusplus
}
#endif

#endif
