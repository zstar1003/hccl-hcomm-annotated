/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @defgroup vos_base 公共定义模块 通用错误码
 * @ingroup bare
 */

#ifndef __VOS_ERRNO_H__
#define __VOS_ERRNO_H__

#include "vos_typdef.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VOS_YES
#define VOS_YES 1 /* 真 */
#endif

#ifndef VOS_NO
#define VOS_NO 0 /* 假 */
#endif

#ifndef VOS_ERR
#define VOS_ERR 1 /* 出错 For VRP VOS adaptation */
#endif
#define VOS_ERROR (~0) /* Compatible V1 */

/* Common Error Code */
#define VOS_OK 0 /* 正确 */

#ifdef __cplusplus
}
#endif

#endif /* __VOS_ERRNO_H__ */
