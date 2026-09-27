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
#include "sal_pub.h"
#include "externalinput.h"
#include "adapter_trace.h"
#include "atrace_types.h"
#include "hccl_trace_info.h"
using namespace std;
namespace hccl {
constexpr u32 TRACE_MAX_MSG_SIZE = 111;

HcclTraceInfo::HcclTraceInfo() : index(0), handle(-1) { hcclTraceType_ = HcclTraceType::HostTraceType; }

HcclTraceInfo::HcclTraceInfo(const UtraceAttr& utraceAttr) : utraceAttr_(utraceAttr), index(0), handle(-1)
{
    hcclTraceType_ = HcclTraceType::DeviceTraceType;
}

HcclTraceInfo::~HcclTraceInfo() {}

HcclResult HcclTraceInfo::Init(std::string& logInfo)
{
    if (hcclTraceType_ == HcclTraceType::DeviceTraceType && !utraceAttr_.utraceStatusFlag) {
        return HCCL_SUCCESS;
    }
    if (handle != -1) {
        return HCCL_SUCCESS;
    }
    /* 申请trace资源信息 */
    CHK_RET(hrtOpenTrace());
    if (hrtTraceCreateWithAttr(logInfo.c_str(), handle) != HCCL_SUCCESS) {
        HCCL_RUN_INFO("[HcclTraceInfo]atrace create handle failed, Save logs to run info");
        handle = -1;
    }
    if (hcclTraceType_ == HcclTraceType::DeviceTraceType) {
        TraceGlobalAttr traceGlobalAttr = {};
        traceGlobalAttr.saveMode = 1; // 1代表 发送到远程并保存
        traceGlobalAttr.deviceId = utraceAttr_.deviceid;
        traceGlobalAttr.pid = utraceAttr_.pid;
        CHK_RET(hrtTraceSetGlobalAttr(&traceGlobalAttr));
    }
    return HCCL_SUCCESS;
}

void HcclTraceInfo::DeInit()
{
    if (handle != -1
        && ((hcclTraceType_ == HcclTraceType::DeviceTraceType && utraceAttr_.utraceStatusFlag)
            || hcclTraceType_ == HcclTraceType::HostTraceType)) {
        /* 销毁当前trace句柄 */
        hrtTraceDestroy(handle);
    }
}

HcclResult HcclTraceInfo::SaveTraceInfo(std::string& logInfo, AtraceOption option)
{
    if (hcclTraceType_ == HcclTraceType::DeviceTraceType && !utraceAttr_.utraceStatusFlag) {
        return HCCL_SUCCESS;
    }

    if (UNLIKELY(handle == -1)) {
        HCCL_WARNING("%s", logInfo.c_str());
        return HCCL_SUCCESS;
    }

    if (GetExternalInputHcclEnableEntryLog() && hcclTraceType_ == HcclTraceType::HostTraceType) {
        HCCL_RUN_INFO("%s", logInfo.c_str());
        return HCCL_SUCCESS;
    }
    std::string outLogInfo = "";
    if (hcclTraceType_ == HcclTraceType::HostTraceType) {
        if (option == AtraceOption::Opbasekey) {
            outLogInfo = to_string(SalGetTid()) + "_" + to_string(index) + "_" + logInfo;
        } else {
            outLogInfo = to_string(SalGetTid()) + "_" + logInfo;
        }
    } else {
        outLogInfo = logInfo;
    }

    /* 输出Trace信息 */
    u32 len = TRACE_MAX_MSG_SIZE; // atrace 兼顾性能的单条日志长度
    u32 pos = 0;
    u32 totLen = outLogInfo.length();
    u32 submitLen = 0;
    const u8* startPos = reinterpret_cast<const u8*>(outLogInfo.c_str());
    while (pos < totLen) {
        submitLen = (pos + len <= totLen) ? len : (totLen - pos);
        CHK_RET(hrtTraceSubmit(handle, startPos, submitLen));
        startPos += submitLen;
        pos += submitLen;
    }
    if (option == AtraceOption::Opbasekey && hcclTraceType_ == HcclTraceType::HostTraceType) {
        index++;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclTraceInfo::SavealgtypeTraceInfo(std::string& alg, const std::string& tag)
{
    std::string logInfo;
    if (alg == "") {
        logInfo = tag + " AlgType is set by user or the server number is one";
    } else {
        logInfo = tag + " AlgType is: " + alg;
    }
    CHK_RET_AND_PRINT_IDE(SaveTraceInfo(logInfo, AtraceOption::Algtype), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclTraceInfo::Flush()
{
    if (hcclTraceType_ == HcclTraceType::DeviceTraceType && !utraceAttr_.utraceStatusFlag) {
        return HCCL_SUCCESS;
    }

    if (UNLIKELY(handle == -1)) {
        return HCCL_SUCCESS;
    }
    TracerType tracerType = TracerType::TRACER_TYPE_SCHEDULE;
    bool syncFlag = true;
    CHK_RET(hrtTraceSave(tracerType, syncFlag));
    return HCCL_SUCCESS;
}
} // namespace hccl
