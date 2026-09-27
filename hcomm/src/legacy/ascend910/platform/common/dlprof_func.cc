/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlprof_func.h"
#include "log.h"

namespace hccl {
DlProfFunc& DlProfFunc::GetInstance()
{
    static DlProfFunc hcclDlProfFunction;
    return hcclDlProfFunction;
}

DlProfFunc::DlProfFunc() { DlProfFunctionStubInit(); }

DlProfFunc::~DlProfFunc()
{
    if (handle_ != nullptr) {
        (void)dlclose(handle_);
        handle_ = nullptr;
    }
}

uint64_t HcclMsprofSysCycleTimeStub()
{
    HCCL_WARNING("Entry HcclMsprofSysCycleTimeStub");
    return 0;
}

void DlProfFunc::DlProfFunctionStubInit() { dlMsprofSysCycleTime = (uint64_t(*)(void))HcclMsprofSysCycleTimeStub; }

HcclResult DlProfFunc::DlProfFunctionInterInit()
{
    dlMsprofSysCycleTime = (uint64_t(*)(void))dlsym(handle_, "MsprofSysCycleTime");
    CHK_SMART_PTR_NULL(dlMsprofSysCycleTime);

    return HCCL_SUCCESS;
}

HcclResult DlProfFunc::DlProfFunctionInit()
{
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = dlopen("libprofapi.so", RTLD_NOW);
    }
    if (handle_ != nullptr) {
        CHK_RET(DlProfFunctionInterInit());
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
