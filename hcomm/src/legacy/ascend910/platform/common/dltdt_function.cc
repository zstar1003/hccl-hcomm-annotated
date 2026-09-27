/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <map>
#include "hccl_dl.h"
#include "log.h"
#include "dltdt_function.h"

namespace hccl {
DlTdtFunction& DlTdtFunction::GetInstance()
{
    static DlTdtFunction hcclDlTdtFunction;
    return hcclDlTdtFunction;
}

DlTdtFunction::DlTdtFunction() : handle_(nullptr) {}

DlTdtFunction::~DlTdtFunction()
{
    if (handle_ != nullptr) {
        (void)HcclDlclose(handle_);
        handle_ = nullptr;
    }
}
HcclResult DlTdtFunction::DlTdtFunctionInit()
{
#ifndef HCCD
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = HcclDlopen("libtsdclient.so", RTLD_NOW);
        const char* errMsg = dlerror();
        CHK_PRT_RET(
            handle_ == nullptr,
            HCCL_ERROR(
                "dlopen [%s] failed, %s", "libtsdclient.so",
                (errMsg == nullptr) ? "please check the file exist or permission denied." : errMsg),
            HCCL_E_OPEN_FILE_FAILURE);
    }
    dlTsdCapabilityGet
        = (uint32_t(*)(uint32_t deviceLogicId, int32_t type, uint64_t ptr))HcclDlsym(handle_, "TsdCapabilityGet");
    CHK_SMART_PTR_NULL(dlTsdCapabilityGet);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[DlTdtFunctionInit]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult DlTdtFunction::DlTdtFunctionHeterogInit()
{
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = HcclDlopen("libtsdclient.so", RTLD_NOW);
        const char* errMsg = dlerror();
        CHK_PRT_RET(
            handle_ == nullptr,
            HCCL_ERROR(
                "dlopen [%s] failed, %s", "libtsdclient.so",
                (errMsg == nullptr) ? "please check the file exist or permission denied." : errMsg),
            HCCL_E_OPEN_FILE_FAILURE);
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
