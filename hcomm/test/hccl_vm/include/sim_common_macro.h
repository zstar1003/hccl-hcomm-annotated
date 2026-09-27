/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_COMMON_MACRO_H
#define SIM_COMMON_MACRO_H

#include <iostream>
#include "sim_log.h"

#define UNLIKELY(x) __builtin_expect(!!(x), 0)

/* 检查函数返回值, 并返回指定错误码 */
#define HCCLVM_CHK_RET(call)                                                      \
    do {                                                                          \
        auto hcclRet = call;                                                      \
        if (UNLIKELY(hcclRet != 0)) {                                             \
            HCCL_VM_INFO("call trace: hcclRet -> {}", static_cast<int>(hcclRet)); \
            return hcclRet;                                                       \
        }                                                                         \
    } while (0)

/* 检查指针, 若指针为NULL, 则记录日志, 并返回错误 */
#define HCCLVM_CHK_PTR(ptr)                                                                                         \
    do {                                                                                                            \
        if (UNLIKELY((ptr) == nullptr)) {                                                                           \
            HCCL_VM_INFO(                                                                                           \
                "errNo[0x{:x}], ptr [{}] is nullptr, return HCCL_E_PTR", static_cast<unsigned int>(HCCL_SIM_E_PTR), \
                #ptr);                                                                                              \
            return HCCL_SIM_E_PTR;                                                                                  \
        }                                                                                                           \
    } while (0)

#endif // SIM_COMMON_MACRO_H
