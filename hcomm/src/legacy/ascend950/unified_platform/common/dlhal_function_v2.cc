/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlhal_function_v2.h"

#include <string>
#include <map>
#include "log.h"
#include <dlfcn.h>

namespace Hccl {
DlHalFunctionV2& DlHalFunctionV2::GetInstance()
{
    static DlHalFunctionV2 hcclDlHalFunction;
    return hcclDlHalFunction;
}

DlHalFunctionV2::DlHalFunctionV2() : handle_(nullptr) {}

DlHalFunctionV2::~DlHalFunctionV2()
{
    if (handle_ != nullptr) {
        (void)dlclose(handle_);
        handle_ = nullptr;
    }
}

HcclResult DlHalFunctionV2::DlHalFunctionEschedInit()
{
    HCCL_INFO("DlHalFunctionEschedInit start");
    dlHalEschedSubmitEvent = (drvError_t(*)(unsigned int, struct event_summary*))dlsym(handle_, "halEschedSubmitEvent");
    CHK_SMART_PTR_NULL(dlHalEschedSubmitEvent);

    dlHalDrvQueryProcessHostPid = (drvError_t(*)(int, unsigned int*, unsigned int*, unsigned int*, unsigned int*))dlsym(
        handle_, "drvQueryProcessHostPid");
    CHK_SMART_PTR_NULL(dlHalDrvQueryProcessHostPid);

    dlHalGetDeviceInfo = (drvError_t(*)(uint32_t, int32_t, int32_t, int64_t*))dlsym(handle_, "halGetDeviceInfo");
    CHK_SMART_PTR_NULL(dlHalGetDeviceInfo);

    HCCL_INFO("DlHalFunction::DlHalFunctionEschedInit end");

    return HCCL_SUCCESS;
}

HcclResult DlHalFunctionV2::DlHalFunctionInit()
{
    HCCL_INFO("DlHalFunctionInit start");
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = dlopen("libascend_hal.so", RTLD_NOW);
        const char* errMsg = dlerror();
        CHK_PRT_RET(
            handle_ == nullptr,
            HCCL_ERROR(
                "dlopen [%s] failed, %s", "libascend_hal.so",
                (errMsg == nullptr) ? "please check the file exist or permission denied." : errMsg),
            HCCL_E_OPEN_FILE_FAILURE);
    }

    CHK_RET(DlHalFunctionEschedInit());
    HCCL_INFO("DlHalFunction::DlHalFunctionInit end");
    return HCCL_SUCCESS;
}
} // namespace Hccl
