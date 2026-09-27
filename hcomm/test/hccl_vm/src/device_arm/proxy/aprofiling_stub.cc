/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "aprof_pub.h"
#include "sim_log.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

uint64_t MsprofStr2Id(const char* hashInfo, size_t length)
{
    (void)hashInfo;
    (void)length;
    HCCL_VM_INFO("[APROF] stub");
    return 1;
}

int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char* typeName)
{
    (void)level;
    (void)typeId;
    (void)typeName;
    HCCL_VM_INFO("[APROF] stub");
    return 0;
}

uint64_t MsprofSysCycleTime()
{
    HCCL_VM_INFO("[APROF] stub");
    return 1;
}

int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle)
{
    (void)moduleId;
    (void)handle;
    HCCL_VM_INFO("[APROF] stub");
    return 0;
}

int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi* api)
{
    (void)agingFlag;
    (void)api;
    HCCL_VM_INFO("[APROF] stub");
    return 0;
}

int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    (void)agingFlag;
    (void)data;
    (void)length;
    HCCL_VM_INFO("[APROF] stub");
    return 0;
}

int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    (void)agingFlag;
    (void)data;
    (void)length;
    HCCL_VM_INFO("[APROF] stub");
    return 0;
}

int32_t MsprofReportBatchAdditionalInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length)
{
    (void)nonPersistantFlag;
    (void)data;
    (void)length;
    HCCL_VM_INFO("[APROF] stub");
    return 0;
}

int32_t AdprofReportAdditionalInfo(uint32_t agingFlag, const void* data, uint32_t length)
{
    (void)agingFlag;
    (void)data;
    (void)length;
    HCCL_VM_INFO("[APROF] stub");
    return 0;
}

int32_t AdprofReportBatchAdditionalInfo(uint32_t nonPersistantFlag, const void* data, uint32_t length)
{
    (void)nonPersistantFlag;
    (void)data;
    (void)length;
    HCCL_VM_INFO("[APROF] stub");
    return 0;
}

uint64_t AdprofGetHashId(const char* hashInfo, size_t length)
{
    (void)hashInfo;
    (void)length;
    HCCL_VM_INFO("[APROF] stub");
    return 1;
}

int32_t AdprofCheckFeatureIsOn(uint64_t feature)
{
    (void)feature;
    HCCL_VM_INFO("[APROF] stub");
    return 0;
}

#ifdef __cplusplus
}
#endif // __cplusplus
