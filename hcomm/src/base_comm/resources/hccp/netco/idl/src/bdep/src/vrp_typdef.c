/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "vrp_typdef.h"

#ifdef __cplusplus
extern "C" {
#endif
/* 对编译/运行环境进行测试。如果出错，需要review定义出错的原因 */
BDEP_STATIC_ASSERT(sizeof(char) == 1);
BDEP_STATIC_ASSERT(sizeof(uint8_t) == 1);

BDEP_STATIC_ASSERT(sizeof(int16_t) == 2);
BDEP_STATIC_ASSERT(sizeof(uint16_t) == 2);

BDEP_STATIC_ASSERT(sizeof(int32_t) == 4);
BDEP_STATIC_ASSERT(sizeof(uint32_t) == 4);

BDEP_STATIC_ASSERT(sizeof(int64_t) == 8);
BDEP_STATIC_ASSERT(sizeof(uint64_t) == 8);

BDEP_STATIC_ASSERT(sizeof(BOOL) == 4);

BDEP_STATIC_ASSERT(sizeof(intptr_t) == sizeof(void *));
BDEP_STATIC_ASSERT(sizeof(uintptr_t) == sizeof(void *));

BDEP_STATIC_ASSERT(sizeof(unsigned char) == 1);

void vrp_typedef_func_foo123_(void)
{
}

void vrp_typedef_func_bar456_(uint32_t i)
{
    (void)i;
}

#ifdef __cplusplus
}
#endif
