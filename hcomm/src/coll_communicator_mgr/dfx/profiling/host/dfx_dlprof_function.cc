/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfx_dlprof_function.h"
#include "log.h"

namespace Hccl {

DfxDlProfFunction& DfxDlProfFunction::GetInstance()
{
    static DfxDlProfFunction hcclDfxDlProfFunction;
    return hcclDfxDlProfFunction;
}

DfxDlProfFunction::DfxDlProfFunction() { DfxDlProfFunctionStubInit(); }

DfxDlProfFunction::~DfxDlProfFunction()
{
    if (handle_ != nullptr) {
        (void)dlclose(handle_);
        handle_ = nullptr;
    }
}

static int32_t MsprofRegisterCallbackStub([[maybe_unused]] uint32_t moduleId, [[maybe_unused]] ProfCommandHandle handle)
{
    HCCL_WARNING("Entry MsprofRegisterCallbackStub");
    return 0;
}

static int32_t MsprofRegTypeInfoStub(
    [[maybe_unused]] uint16_t level, [[maybe_unused]] uint32_t typeId, [[maybe_unused]] const char* typeName)
{
    HCCL_WARNING("Entry MsprofRegTypeInfoStub");
    return 0;
}

static int32_t MsprofReportApiStub([[maybe_unused]] uint32_t agingFlag, [[maybe_unused]] const MsprofApi* api)
{
    HCCL_WARNING("Entry MsprofReportApiStub");
    return 0;
}

static int32_t MsprofReportCompactInfoStub(
    [[maybe_unused]] uint32_t agingFlag, [[maybe_unused]] const void* data, [[maybe_unused]] uint32_t length)
{
    HCCL_WARNING("Entry MsprofReportCompactInfoStub");
    return 0;
}

static int32_t MsprofReportAdditionalInfoStub(
    [[maybe_unused]] uint32_t agingFlag, [[maybe_unused]] const void* data, [[maybe_unused]] uint32_t length)
{
    HCCL_WARNING("Entry MsprofReportAdditionalInfoStub");
    return 0;
}

static int32_t MsprofReportBatchAdditionalInfoStub(
    [[maybe_unused]] uint32_t agingFlag, [[maybe_unused]] const void* data, [[maybe_unused]] uint32_t length)
{
    HCCL_WARNING("Entry MsprofReportBatchAdditionalInfoStub");
    return 0;
}

static uint64_t MsprofStr2IdStub([[maybe_unused]] const char* hashInfo, [[maybe_unused]] uint32_t length)
{
    HCCL_WARNING("Entry MsprofStr2IdStub");
    return 0;
}

static uint64_t MsprofSysCycleTimeStub()
{
    HCCL_WARNING("Entry MsprofSysCycleTimeStub");
    return 0;
}

void DfxDlProfFunction::DfxDlProfFunctionStubInit()
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

HcclResult DfxDlProfFunction::DfxDlProfFunctionInterInit()
{
    auto tmpRegisterCallback
        = reinterpret_cast<int32_t (*)(uint32_t, ProfCommandHandle)>(dlsym(handle_, "MsprofRegisterCallback"));
    CHK_PTR_NULL(tmpRegisterCallback);
    auto tmpRegTypeInfo
        = reinterpret_cast<int32_t (*)(uint16_t, uint32_t, const char*)>(dlsym(handle_, "MsprofRegTypeInfo"));
    CHK_PTR_NULL(tmpRegTypeInfo);
    auto tmpReportApi = reinterpret_cast<int32_t (*)(uint32_t, const MsprofApi*)>(dlsym(handle_, "MsprofReportApi"));
    CHK_PTR_NULL(tmpReportApi);
    auto tmpReportCompactInfo
        = reinterpret_cast<int32_t (*)(uint32_t, const VOID_PTR, uint32_t)>(dlsym(handle_, "MsprofReportCompactInfo"));
    CHK_PTR_NULL(tmpReportCompactInfo);
    auto tmpReportAdditionalInfo = reinterpret_cast<int32_t (*)(uint32_t, const VOID_PTR, uint32_t)>(
        dlsym(handle_, "MsprofReportAdditionalInfo"));
    CHK_PTR_NULL(tmpReportAdditionalInfo);
    auto tmpReportBatchAdditionalInfo = reinterpret_cast<int32_t (*)(uint32_t, const VOID_PTR, uint32_t)>(
        dlsym(handle_, "MsprofReportBatchAdditionalInfo"));
    if (tmpReportBatchAdditionalInfo == nullptr) {
        HCCL_INFO("[DfxDlProfFunction] MsprofReportBatchAdditionalInfo not found, batch report disabled");
    }
    auto tmpStr2Id = reinterpret_cast<uint64_t (*)(const char*, uint32_t)>(dlsym(handle_, "MsprofStr2Id"));
    CHK_PTR_NULL(tmpStr2Id);
    auto tmpSysCycleTime = reinterpret_cast<uint64_t (*)(void)>(dlsym(handle_, "MsprofSysCycleTime"));
    CHK_PTR_NULL(tmpSysCycleTime);

    // 全部 dlsym 成功后才覆盖，失败时保留 stub 不被置为 nullptr
    dlMsprofRegisterCallback = tmpRegisterCallback;
    dlMsprofRegTypeInfo = tmpRegTypeInfo;
    dlMsprofReportApi = tmpReportApi;
    dlMsprofReportCompactInfo = tmpReportCompactInfo;
    dlMsprofReportAdditionalInfo = tmpReportAdditionalInfo;
    if (tmpReportBatchAdditionalInfo != nullptr) {
        dlMsprofReportBatchAdditionalInfo = tmpReportBatchAdditionalInfo;
    }
    dlMsprofStr2Id = tmpStr2Id;
    dlMsprofSysCycleTime = tmpSysCycleTime;
    return HCCL_SUCCESS;
}

HcclResult DfxDlProfFunction::DfxDlProfFunctionInit()
{
    if (initializedFlag_.load(std::memory_order_acquire)) {
        return HCCL_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (initializedFlag_.load(std::memory_order_relaxed)) {
        return HCCL_SUCCESS;
    }
    if (handle_ == nullptr) {
        handle_ = dlopen("libprofapi.so", RTLD_NOW);
    }
    if (handle_ == nullptr) {
        HCCL_ERROR("[DfxDlProfFunction][Init] dlopen libprofapi.so failed: %s", dlerror());
        return HCCL_E_INTERNAL;
    }
    CHK_RET(DfxDlProfFunctionInterInit());
    initializedFlag_.store(true, std::memory_order_release);
    return HCCL_SUCCESS;
}
} // namespace Hccl
