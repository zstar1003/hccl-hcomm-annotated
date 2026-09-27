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
 * @defgroup util Util
 */
/**
 * @defgroup v_avl AVL
 * @ingroup util
 */
#ifndef AVL_ADAPT_H
#define AVL_ADAPT_H

#ifdef __cplusplus
extern "C" {
#endif /* --cplusplus */

#define AVL_NULL_PTR 0L

#define AVL_TRUE 1
#define AVL_FALSE 0

/* 兼容SSP VXWORKS */
#if defined(VOS_OS_VER) && defined(VOS_VXWORKS) && (VOS_OS_VER == VOS_VXWORKS)
#define AVL_INT32_TO_LONG 1
#endif

/* 兼容SSP VXWORKS */
#ifndef VOS_PACKEND_ZERO
#define VOS_PACKEND_ZERO 1
#endif
#if defined(VOS_PACKEND_ZERO) && defined(VOS_PACK_END) && (VOS_PACKEND_ZERO == VOS_PACK_END)
#define AVL_PACKEND_ZERO 1
#endif

#ifdef __cplusplus
}
#endif /* --cplusplus */

#endif
