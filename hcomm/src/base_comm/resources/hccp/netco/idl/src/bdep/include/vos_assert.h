/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __VOS_ASSERT_H
#define __VOS_ASSERT_H

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(VOS_BUILD_RELEASE) && (VOS_BUILD_RELEASE == 1)
#define VOS_ASSERT(exp) ((void)0)
#elif defined(VOS_BUILD_DEBUG) && (VOS_BUILD_DEBUG == 1)
#define VOS_ASSERT(exp) assert(exp)
#else
#define VOS_ASSERT(exp) ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* __VOS_ASSERT_H */
