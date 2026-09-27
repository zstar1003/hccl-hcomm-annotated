/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_trace.h"
#include "dltrace_function.h"
#include "log.h"
using namespace hccl;

HcclResult hrtOpenTrace()
{
    HcclResult ret = DlTraceFunction::GetInstance().DlTraceFunctionInit();
    HCCL_INFO("Call TraceCreate, return value[%d]", ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Call TraceCreate, return value[%d]", ret), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void hrtTraceDestroy(TraHandle handle)
{
    if (DlTraceFunction::GetInstance().dlAtraceDestroy != nullptr) {
        DlTraceFunction::GetInstance().dlAtraceDestroy(handle);
    } else {
        DlTraceFunction::GetInstance().dlUtraceDestroy(handle);
    }
    HCCL_INFO("Call TraceDestroy, Params:handle[%p]", handle);
    return;
}

HcclResult hrtTraceSubmit(TraHandle handle, const void* buffer, u32 bufSize)
{
    HcclResult ret = HCCL_E_RESERVED;
    if (DlTraceFunction::GetInstance().dlAtraceSubmit != nullptr) {
        ret = DlTraceFunction::GetInstance().dlAtraceSubmit(handle, buffer, bufSize);
    } else {
        ret = DlTraceFunction::GetInstance().dlUtraceSubmit(handle, buffer, bufSize);
    }
    CHK_PRT_RET(
        ret != 0, HCCL_ERROR("Call TraceSubmit, return value[%d], Params:handle[%p]", ret, handle), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult hrtTraceCreateWithAttr(const char* objName, TraHandle& handle)
{
    CHK_PTR_NULL(objName);
    TraHandle ret = TRACE_INVALID_HANDLE;
    if (DlTraceFunction::GetInstance().dlAtraceCreateWithAttr != nullptr) {
        TraceAttr hcclAtraceAttr = {};
        hcclAtraceAttr.exitSave = true;
        hcclAtraceAttr.msgNum = DEFAULT_ATRACE_MSG_NUM;
        hcclAtraceAttr.msgSize = DEFAULT_ATRACE_MSG_SIZE;
        ret = DlTraceFunction::GetInstance().dlAtraceCreateWithAttr(TRACER_TYPE_SCHEDULE, objName, &hcclAtraceAttr);
    } else {
        TraceAttr hcclUtraceAttr = {};
        hcclUtraceAttr.exitSave = false;
        hcclUtraceAttr.msgNum = DEFAULT_ATRACE_MSG_NUM;
        hcclUtraceAttr.msgSize = DEFAULT_ATRACE_MSG_SIZE;
        ret = DlTraceFunction::GetInstance().dlUtraceCreateWithAttr(TRACER_TYPE_SCHEDULE, objName, &hcclUtraceAttr);
    }
    HCCL_INFO("Call TraceCreateWithAttr, return value[%p], Params: objName[%s]", ret, objName);
    CHK_PRT_RET(
        ret == TRACE_INVALID_HANDLE, HCCL_WARNING("Call TraceCreateWithAttr, return value[%lld]", ret),
        HCCL_E_INTERNAL);
    handle = ret;
    return HCCL_SUCCESS;
}

HcclResult hrtTraceSetGlobalAttr(const TraceGlobalAttr* attr)
{
    TraHandle ret = TRACE_INVALID_HANDLE;
    if (DlTraceFunction::GetInstance().dlAtraceSetGlobalAttr != nullptr) {
        ret = DlTraceFunction::GetInstance().dlAtraceSetGlobalAttr(attr);
    } else {
        ret = DlTraceFunction::GetInstance().dlUtraceSetGlobalAttr(attr);
    }
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Call TraceSetGlobalAttr, return value[%d]", ret), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult hrtTraceSave(TracerType tracerType, bool syncFlag)
{
    TraHandle ret = TRACE_INVALID_HANDLE;
    if (DlTraceFunction::GetInstance().dlAtraceSave != nullptr) {
        ret = DlTraceFunction::GetInstance().dlAtraceSave(tracerType, syncFlag);
    } else {
        ret = DlTraceFunction::GetInstance().dlUtraceSave(tracerType, syncFlag);
    }
    CHK_PRT_RET(ret != 0, HCCL_ERROR("Call traceSave, return value[%d]", ret), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}
