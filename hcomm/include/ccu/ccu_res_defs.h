/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RES_DEFS_H
#define CCU_RES_DEFS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief CCU资源描述符句柄类型
 */
typedef uint64_t HcommCcuResDescHandle;

/**
 * @brief CCU资源类型枚举
 */
typedef enum {
    HCOMM_CCU_RES_TYPE_INVALID = -1,    ///< 无效资源类型
    HCOMM_CCU_RES_TYPE_LOOP = 0,        ///< Loop资源
    HCOMM_CCU_RES_TYPE_CCU_BUF = 1,     ///< CCU Buffer资源
    HCOMM_CCU_RES_TYPE_VARIABLE = 2,    ///< Variable资源
    HCOMM_CCU_RES_TYPE_ADDRESS = 3,     ///< Address资源
    HCOMM_CCU_RES_TYPE_EVENT = 4,       ///< Event资源
    HCOMM_CCU_RES_TYPE_CCU_THREAD = 5,  ///< CCU Thread资源
    HCOMM_CCU_RES_TYPE_INSTRUCTION = 6, ///< Instruction资源
    HCOMM_CCU_RES_TYPE_CASC_CNT = 7     ///< Casc Count资源
} HcommCcuResType;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CCU_RES_DEFS_H
