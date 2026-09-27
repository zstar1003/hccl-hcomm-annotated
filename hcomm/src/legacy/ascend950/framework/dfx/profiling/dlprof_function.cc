/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlprof_function_v2.h"
#include "log.h"

namespace Hccl {
#define UNUSED(x) (void)(x)

DlProfFunction& DlProfFunction::GetInstance()
{
    static DlProfFunction hcclDlProfFunction;
    return hcclDlProfFunction;
}

DlProfFunction::DlProfFunction() { DlProfFunctionStubInit(); }

DlProfFunction::~DlProfFunction()
{
    if (handle_ != nullptr) {
        (void)dlclose(handle_);
        handle_ = nullptr;
    }
}

static int32_t MsprofRegisterCallbackStub(uint32_t moduleId, ProfCommandHandle handle)
{
    UNUSED(moduleId);
    UNUSED(handle);
    HCCL_WARNING("Entry MsprofRegisterCallbackStub");
    return 0;
}

static int32_t MsprofRegTypeInfoStub(uint16_t level, uint32_t typeId, const char* typeName)
{
    UNUSED(level);
    UNUSED(typeId);
    UNUSED(typeName);
    HCCL_WARNING("Entry MsprofRegTypeInfoStub");
    return 0;
}

static int32_t MsprofReportApiStub(uint32_t agingFlag, const MsprofApi* api)
{
    UNUSED(agingFlag);
    UNUSED(api);
    HCCL_WARNING("Entry MsprofReportApiStub");
    return 0;
}

static int32_t MsprofReportCompactInfoStub(uint32_t agingFlag, const void* data, uint32_t length)
{
    UNUSED(agingFlag);
    UNUSED(data);
    UNUSED(length);
    HCCL_WARNING("Entry MsprofReportCompactInfoStub");
    return 0;
}

static int32_t MsprofReportAdditionalInfoStub(uint32_t agingFlag, const void* data, uint32_t length)
{
    UNUSED(agingFlag);
    UNUSED(data);
    UNUSED(length);
    HCCL_WARNING("Entry MsprofReportAdditionalInfoStub");
    return 0;
}

static int32_t MsprofReportBatchAdditionalInfoStub(uint32_t agingFlag, const void* data, uint32_t length)
{
    UNUSED(agingFlag);
    UNUSED(data);
    UNUSED(length);
    HCCL_WARNING("Entry MsprofReportBatchAdditionalInfoStub");
    return 0;
}

static uint64_t MsprofStr2IdStub(const char* hashInfo, uint32_t length)
{
    UNUSED(hashInfo);
    UNUSED(length);
    HCCL_WARNING("Entry MsprofStr2IdStub");
    return 0;
}

static uint64_t MsprofSysCycleTimeStub()
{
    HCCL_WARNING("Entry MsprofSysCycleTimeStub");
    return 0;
}

void DlProfFunction::DlProfFunctionStubInit()
{
    dlMsprofRegisterCallback = static_cast<int32_t (*)(uint32_t, ProfCommandHandle)>(MsprofRegisterCallbackStub);
    dlMsprofRegTypeInfo = static_cast<int32_t (*)(uint16_t, uint32_t, const char*)>(MsprofRegTypeInfoStub);
    dlMsprofReportApi = static_cast<int32_t (*)(uint32_t, const MsprofApi*)>(MsprofReportApiStub);
    dlMsprofReportCompactInfo = static_cast<int32_t (*)(uint32_t, const void*, uint32_t)>(MsprofReportCompactInfoStub);
    dlMsprofReportAdditionalInfo
        = static_cast<int32_t (*)(uint32_t, const void*, uint32_t)>(MsprofReportAdditionalInfoStub);
    dlMsprofReportBatchAdditionalInfo
        = static_cast<int32_t (*)(uint32_t, const void*, uint32_t)>(MsprofReportBatchAdditionalInfoStub);
    dlMsprofStr2Id = static_cast<uint64_t (*)(const char*, uint32_t)>(MsprofStr2IdStub);
    dlMsprofSysCycleTime = static_cast<uint64_t (*)(void)>(MsprofSysCycleTimeStub);
}

HcclResult DlProfFunction::DlProfFunctionInterInit()
{
    dlMsprofRegisterCallback = (int32_t(*)(uint32_t, ProfCommandHandle))dlsym(handle_, "MsprofRegisterCallback");
    CHK_PTR_NULL(dlMsprofRegisterCallback);

    dlMsprofRegTypeInfo = (int32_t(*)(uint16_t, uint32_t, const char*))dlsym(handle_, "MsprofRegTypeInfo");
    CHK_PTR_NULL(dlMsprofRegTypeInfo);

    dlMsprofReportApi = (int32_t(*)(uint32_t, const MsprofApi*))dlsym(handle_, "MsprofReportApi");
    CHK_PTR_NULL(dlMsprofReportApi);

    dlMsprofReportCompactInfo
        = (int32_t(*)(uint32_t, const VOID_PTR, uint32_t))dlsym(handle_, "MsprofReportCompactInfo");
    CHK_PTR_NULL(dlMsprofReportCompactInfo);

    dlMsprofReportAdditionalInfo
        = (int32_t(*)(uint32_t, const VOID_PTR, uint32_t))dlsym(handle_, "MsprofReportAdditionalInfo");
    CHK_PTR_NULL(dlMsprofReportAdditionalInfo);

    dlMsprofReportBatchAdditionalInfo
        = (int32_t(*)(uint32_t, const VOID_PTR, uint32_t))dlsym(handle_, "MsprofReportBatchAdditionalInfo");
    if (dlMsprofReportBatchAdditionalInfo == nullptr) {
        HCCL_INFO("[DlProfFunction] MsprofReportBatchAdditionalInfo not found, batch report disabled");
    }

    dlMsprofStr2Id = (uint64_t(*)(const char*, uint32_t))dlsym(handle_, "MsprofStr2Id");
    CHK_PTR_NULL(dlMsprofStr2Id);

    dlMsprofSysCycleTime = (uint64_t(*)(void))dlsym(handle_, "MsprofSysCycleTime");
    CHK_PTR_NULL(dlMsprofSysCycleTime);

    return HCCL_SUCCESS;
}

HcclResult DlProfFunction::DlProfFunctionInit()
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
