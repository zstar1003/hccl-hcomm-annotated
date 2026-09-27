/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlprof_func_v2.h"
#include <dlfcn.h>
#include "log.h"
#include "exception_util.h"

namespace Hccl {
DlProfFunc& DlProfFunc::GetInstance()
{
    static DlProfFunc hcclDlProfFunction;
    return hcclDlProfFunction;
}

bool DlProfFunc::isStubMode() const { return false; }

DlProfFunc::DlProfFunc()
{
    if (isStubMode()) {
        DlProfFunctionStubInit();
    } else {
        DlProfFunctionInit();
    }
}

DlProfFunc::~DlProfFunc()
{
    if (handle_ != nullptr) {
        (void)dlclose(handle_);
        handle_ = nullptr;
    }
}

static uint64_t HcclMsprofSysCycleTimeStub()
{
    HCCL_WARNING("Entry HcclMsprofSysCycleTimeStub");
    return 0;
}

void DlProfFunc::DlProfFunctionStubInit() { dlMsprofSysCycleTime = &HcclMsprofSysCycleTimeStub; }

HcclResult DlProfFunc::DlProfFunctionInterInit()
{
    CHECK_NULLPTR(handle_, "[DlProfFunc::DlProfFunctionInterInit] handle_ is nullptr!");
    dlMsprofSysCycleTime = (uint64_t(*)(void))dlsym(handle_, "MsprofSysCycleTime");
    CHK_PTR_NULL(dlMsprofSysCycleTime);

    return HCCL_SUCCESS;
}

HcclResult DlProfFunc::DlProfFunctionInit()
{
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = dlopen("libprofapi.so", RTLD_NOW);
    }
    if (handle_ != nullptr) {
        CHK_RET(DlProfFunctionInterInit());
    }
    initializedFlag_ = true;
    return HCCL_SUCCESS;
}
} // namespace Hccl
