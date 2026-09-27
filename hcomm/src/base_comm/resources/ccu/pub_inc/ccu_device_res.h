/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_PRIMITIVES_H
#define CCU_PRIMITIVES_H

#include "ccu_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum {
    CCU_DEFAULT = 0, /**< default, now is scheduler mode */
    CCU_SCHED = 1,   /**< scheduler mode, with less loop resource */
    CCU_MS = 2,      /**< memory slices mode, with more loop resource */
    CCU_UNUSED = 254,
    CCU_RESERVED = 255 /**< reserved */
} CcuInstanceType;

/**
 * @brief 按 CCU 实例类型创建 ccu 实例（兼容 hccl 旧版本）。
 * @param[in] insType CCU实例类型。
 * @param[out] ccuInsHandle 创建成功后返回的 CCU 实例句柄，不可为 nullptr。
 * @return CcuResult。
 */
extern CcuResult HcommCcuInsCreateLegacy(const CcuInstanceType insType, CcuInsHandle* ccuInsHandle);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CCU_PRIMITIVES_H
