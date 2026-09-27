/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __VOS_CPUTICK
#define __VOS_CPUTICK

#include "vos_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// 将 CPU TICK计数值转换为毫秒数(精度更高)。
uint32_t VOS_CpuTick2MsEx(VOS_CPUTICK_S *cputick, uint32_t *milliSecsHigh, uint32_t *milliSecsLow);
void VOS_GetCpuTick(uint32_t *puiLow, uint32_t *puiHigh);
/**
 * @ingroup vos_cputick
 * 获取cputick的值，更多说明请参考#VOS_GetCpuTick
 */
#define VOS_CpuTickGet(p) VOS_GetCpuTick(&(p)->uiLow, &(p)->uiHigh)

#ifdef __cplusplus
}
#endif

#endif /* __VOS_CPUTICK */
