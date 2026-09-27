/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VOS_TYPDEF_H
#define VOS_TYPDEF_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
/* 静态断言 */
#define BDEP_STATIC_ASSERT(X) typedef char BDepStaticAssert__[(!!(X)) * 2 - 1]

typedef uint32_t VOS_BOOL; /* 原先dopra中定义为unsigned int，长度不确定。修改 */

#define VOS_NULL_PTR NULL

enum VOS_BOOL_DEFINE {
    VOS_FALSE = 0, /**< BOOL 型的逻辑假 */
    VOS_TRUE = 1   /**< BOOL 型的逻辑真 */
};

#define VOS_NULL NULL

#ifdef __cplusplus
}
#endif

#endif
