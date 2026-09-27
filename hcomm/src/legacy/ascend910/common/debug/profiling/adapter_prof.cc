/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_prof.h"
#include "externalinput_pub.h"
#include "dlrt_function.h"
using namespace hccl;

HcclResult hrtMsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle)
{
    s32 ret = DlProfFunction::GetInstance().dlMsprofRegisterCallback(moduleId, handle);
    HCCL_INFO("Call MsprofRegisterCallback, return value[%d], Params: moduleId[%u]", ret, moduleId);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR("[Register][CtrlCallback]MsprofRegisterCallback fail, return[%d], moduleId[%u]", ret, moduleId),
        HCCL_E_PROFILING);
    return HCCL_SUCCESS;
}

HcclResult hrtMsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char* typeName)
{
    s32 ret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(level, typeId, typeName);
    HCCL_INFO("Call MsprofRegTypeInfo, return value[%d], Params: typeId[%u] typeName = [%s]", ret, typeId, typeName);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Call MsprofRegTypeInfo fail, return[%d]", ret), HCCL_E_PROFILING);
    return HCCL_SUCCESS;
}

HcclResult hrtMsprofReportApi(uint32_t agingFlag, const MsprofApi* api)
{
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportApi(agingFlag, api);
    HCCL_INFO("Call MsprofReportApi, return value[%d], Params: agingFlag[%u]", ret, agingFlag);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Call MsprofReportApi fail, return[%d]", ret), HCCL_E_PROFILING);
    return HCCL_SUCCESS;
}

HcclResult hrtMsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportCompactInfo(agingFlag, data, length);
    HCCL_INFO("Call MsprofReportCompactInfo, return value[%d], Params: agingFlag[%u]", ret, agingFlag);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Call MsprofReportCompactInfo fail, return[%d]", ret), HCCL_E_PROFILING);
    return HCCL_SUCCESS;
}

HcclResult hrtMsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportAdditionalInfo(agingFlag, data, length);
    HCCL_INFO("Call MsprofReportAdditionalInfo, return value[%d], Params: agingFlag[%u]", ret, agingFlag);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Call MsprofReportAdditionalInfo fail, return[%d]", ret), HCCL_E_PROFILING);
    return HCCL_SUCCESS;
}

uint64_t hrtMsprofGetHashId(const char* hashInfo, uint32_t length)
{
    if (hashInfo == nullptr || length == 0) {
        HCCL_WARNING("HashData hashInfo is empty.");
        return INVALID_U64;
    }
    u64 ret = DlProfFunction::GetInstance().dlMsprofStr2Id(hashInfo, length);
    return ret;
}

uint64_t hrtMsprofSysCycleTime(void)
{
    if (!GetIfProfile()) {
        return 0;
    }
    u64 ret = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    HCCL_DEBUG("Call MsprofSysCycleTime, return value[%u]", ret);
    return ret;
}

HcclResult hrtProfRegisterCtrlCallback(uint32_t logId, rtProfCtrlHandle callback)
{
    rtError_t ret = DlRtFunction::GetInstance().dlrtProfRegisterCtrlCallback(logId, callback);
    HCCL_DEBUG("Call rtProfRegisterCtrlCallback, return value[%d], Params: logId[%u].", ret, logId);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Register][CtrlCallback]errNo[0x%016llx] rtProf Register CtrlCallback"
            " fail, return[%d], rt logId[%u]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, logId),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}
