/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_LAUNCH_H
#define CCU_LAUNCH_H

#include "hccl_types.h"
#include "ccu_types.h"
#include "hccl_res.h"

#ifndef HCOMM_WEAK_SYMBOL
#define HCOMM_WEAK_SYMBOL __attribute__((weak))
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle) HCOMM_WEAK_SYMBOL;

extern CcuResult HcommCcuKernelRegister(
    CcuInsHandle insHandle, uint32_t dieId, const char* kernelFuncName, const void* kernelFunc, const void** kernelArgs,
    uint32_t argNum, CcuKernelHandle* kernelHandle) HCOMM_WEAK_SYMBOL;

extern CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle) HCOMM_WEAK_SYMBOL;

extern CcuResult HcommCcuGetTaskArgsNum(CcuKernelHandle kernelHandle, uint32_t* taskArgsNum);

extern CcuResult HcommCcuKernelLaunch(
    ThreadHandle threadHandle, CcuKernelHandle kernelHandle, const void* taskArgs, uint32_t argNum) HCOMM_WEAK_SYMBOL;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CCU_LAUNCH_H
