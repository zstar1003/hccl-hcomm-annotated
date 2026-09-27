/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_LOG_H
#define CCU_LOG_H

#include "log.h"
#include "exception_handler.h"

#define HCCL_TO_CCU_RET(hcclRet) static_cast<CcuResult>(hcclRet)

#define CCU_CHK_RES_UNAVAIL(ccuRet) (ccuRet == CCU_E_UNAVAIL)

/* 检查函数返回值, 并返回指定错误码 */
#define CCU_CHK_RET(call)                                                                                 \
    do {                                                                                                  \
        CcuResult ccuRet = HCCL_TO_CCU_RET(call);                                                         \
        if (UNLIKELY(ccuRet != CCU_SUCCESS)) {                                                            \
            if (ccuRet == static_cast<int32_t>(HcclResult::HCCL_E_AGAIN) || ccuRet == CCU_E_DRV_BUSY) {   \
                HCCL_WARNING("[%s]call trace: ccuRet -> %d", __func__, ccuRet);                           \
            } else if (ccuRet == CCU_E_UNAVAIL) {                                                         \
                HCCL_WARNING("[%s]call trace: ccuRet resources are unavailable -> %d", __func__, ccuRet); \
            } else {                                                                                      \
                HCCL_ERROR("[%s]call trace: ccuRet -> %d", __func__, ccuRet);                             \
            }                                                                                             \
            return ccuRet;                                                                                \
        }                                                                                                 \
    } while (0)

#define CCU_CHK_PTR_NULL(ptr)                                                                                      \
    do {                                                                                                           \
        if (UNLIKELY((ptr) == nullptr)) {                                                                          \
            HCCL_ERROR(                                                                                            \
                "[%s]errNo[0x%016llx]ptr [%s] is nullptr, return CCU_E_PTR", __func__, HCCL_ERROR_CODE(CCU_E_PTR), \
                #ptr);                                                                                             \
            return CCU_E_PTR;                                                                                      \
        }                                                                                                          \
    } while (0)

// 宏定义，用于包装 C 接口函数的异常处理
#define CCU_EXCEPTION_HANDLE_BEGIN try {
#define CCU_EXCEPTION_HANDLE_END_INFO(func_name) \
    }                                            \
    catch (...) { return HCCL_TO_CCU_RET(hccl::ExceptionHandler::HandleException(func_name)); }

#define CCU_EXCEPTION_HANDLE_END CCU_EXCEPTION_HANDLE_END_INFO(__func__)

#endif // CCU_LOG_H
