/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_MC2_EX_H
#define HCCL_MC2_EX_H

#include "mc2_data_type.h"
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult HcclGetCommHandleByCtx(void* ctx, void** opHandle);
HcclResult HcclReleaseComm(void* opHandle);
HcclResult HcclGetTaskStatus(void* opHandle, HcclTaskStatus* status);
HcclResult HcclCheckFinishByStream(void* opHandle);
HcclResult HcclPrintTaskExceptionAllComm(void* opHandle);
HcclResult HcclLaunchCcoreWait(void* opHandle, uint64_t waitAddr, uint32_t turnNum, uint64_t turnNumAddr, bool isLast);
HcclResult HcclLaunchCcorePost(void* opHandle, uint64_t recordAddr, uint32_t turnNum, uint64_t turnNumAddr);
HcclResult HcclLaunchOp(void* opHandle, HcclOpData* data);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_MC2_EX_H
