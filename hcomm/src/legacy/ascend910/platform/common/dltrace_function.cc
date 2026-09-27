/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dltrace_function.h"
#include "hccl_dl.h"
#include "log.h"

namespace hccl {

DlTraceFunction& DlTraceFunction::GetInstance()
{
    static DlTraceFunction hcclDlTraceFunction;
    return hcclDlTraceFunction;
}

DlTraceFunction::DlTraceFunction() : handle_(nullptr) {}

DlTraceFunction::~DlTraceFunction()
{
    if (handle_ != nullptr) {
        (void)HcclDlclose(handle_);
        handle_ = nullptr;
    }
}

HcclResult DlTraceFunction::DlUTraceFunctionInterInit()
{
    dlUtraceDestroy = (void (*)(TraHandle handle))HcclDlsym(handle_, "UtraceDestroy");
    CHK_SMART_PTR_NULL(dlUtraceDestroy);
    dlUtraceSubmit
        = (HcclResult(*)(TraHandle handle, const void* buffer, u32 bufSize))HcclDlsym(handle_, "UtraceSubmit");
    CHK_SMART_PTR_NULL(dlUtraceSubmit);
    dlUtraceCreateWithAttr = (TraHandle(*)(int tracerType, const char* objName, const TraceAttr* attr))HcclDlsym(
        handle_, "UtraceCreateWithAttr");
    CHK_SMART_PTR_NULL(dlUtraceCreateWithAttr);
    dlUtraceSetGlobalAttr = (TraStatus(*)(const TraceGlobalAttr* attr))HcclDlsym(handle_, "UtraceSetGlobalAttr");
    CHK_SMART_PTR_NULL(dlUtraceSetGlobalAttr);
    dlUtraceSave = (TraStatus(*)(TracerType tracerType, bool syncFlag))HcclDlsym(handle_, "UtraceSave");
    CHK_SMART_PTR_NULL(dlUtraceSave);
    return HCCL_SUCCESS;
}

HcclResult DlTraceFunction::DlATraceFunctionInterInit()
{
    dlAtraceDestroy = (void (*)(TraHandle handle))HcclDlsym(handle_, "AtraceDestroy");
    CHK_SMART_PTR_NULL(dlAtraceDestroy);
    dlAtraceSubmit
        = (HcclResult(*)(TraHandle handle, const void* buffer, u32 bufSize))HcclDlsym(handle_, "AtraceSubmit");
    CHK_SMART_PTR_NULL(dlAtraceSubmit);
    dlAtraceCreateWithAttr = (TraHandle(*)(int tracerType, const char* objName, const TraceAttr* attr))HcclDlsym(
        handle_, "AtraceCreateWithAttr");
    CHK_SMART_PTR_NULL(dlAtraceCreateWithAttr);
    dlAtraceSetGlobalAttr = (TraStatus(*)(const TraceGlobalAttr* attr))HcclDlsym(handle_, "AtraceSetGlobalAttr");
    dlAtraceSave = (TraStatus(*)(TracerType tracerType, bool syncFlag))HcclDlsym(handle_, "AtraceSave");
    CHK_SMART_PTR_NULL(dlAtraceSave);
    return HCCL_SUCCESS;
}

HcclResult DlTraceFunction::DlTraceFunctionInit()
{
    HcclResult ret = HCCL_SUCCESS;
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = HcclDlopen("libascend_trace.so", RTLD_NOW);
        const char* errMsg_atrace = dlerror();
        if (handle_ == nullptr) {
            HCCL_WARNING(
                "dlopen [%s] failed, %s", "libascend_trace.so",
                (errMsg_atrace == nullptr) ? "please check the file exist or permission denied." : errMsg_atrace);

            handle_ = HcclDlopen("libutrace.so", RTLD_NOW);
            const char* errMsg_utrace = dlerror();
            CHK_PRT_RET(
                handle_ == nullptr,
                HCCL_ERROR(
                    "dlopen [%s] failed, %s", "libutrace.so",
                    (errMsg_utrace == nullptr) ? "please check the file exist or permission denied." : errMsg_utrace),
                HCCL_E_OPEN_FILE_FAILURE);
            ret = DlUTraceFunctionInterInit();
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[DlTraceFunctionInit] DlUTraceFunctionInterInit fail, ret[%d], destroy handle.", ret);
                (void)HcclDlclose(handle_);
                handle_ = nullptr;
                return ret;
            }
        } else {
            ret = DlATraceFunctionInterInit();
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[DlTraceFunctionInit] DlATraceFunctionInterInit fail, ret[%d], destroy handle.", ret);
                (void)HcclDlclose(handle_);
                handle_ = nullptr;
                return ret;
            }
        }
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
