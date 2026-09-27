/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "atrace_pub.h"
#include "orion_adapter_trace.h"
using namespace std;
namespace Hccl {

constexpr u32 MAX_LOG_TIMEOUT_MS = 1000;
std::chrono::steady_clock::time_point lastLogTimeTrace{};

// 抑制日志刷屏，同一类型日志超时前只打印一次
bool CheckLogTime(std::chrono::steady_clock::time_point& lastTime)
{
    auto nowTime = std::chrono::steady_clock::now();
    if (nowTime - lastTime <= std::chrono::milliseconds(MAX_LOG_TIMEOUT_MS)) {
        return false;
    }
    lastTime = nowTime;
    return true;
}

intptr_t TraceCreate(const char* objName)
{
    TraceAttr hcclAtraceAttr = {};
    hcclAtraceAttr.exitSave = true;
    hcclAtraceAttr.msgNum = DEFAULT_ATRACE_MSG_NUM;
    hcclAtraceAttr.msgSize = DEFAULT_ATRACE_MSG_SIZE;

    TraHandle traHandle = 0;
    traHandle = AtraceCreateWithAttr(TRACER_TYPE_SCHEDULE, objName, &hcclAtraceAttr);
    if (traHandle == TRACE_INVALID_HANDLE && CheckLogTime(lastLogTimeTrace)) {
        HCCL_ERROR(
            "[TraceCreate]errNo[0x%016llx] rt trace create failed. return[%d]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_INTERNAL), traHandle);
    }
    return traHandle;
}

bool TraceSubmit(intptr_t handle, const void* buffer, uint32_t bufSize)
{
    auto ret = AtraceSubmit(handle, buffer, bufSize);
    if (ret != 0) {
        HCCL_ERROR(
            "[TraceSubmit]errNo[0x%016llx] rt trace submit failed. return[%d]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_INTERNAL), ret);
        return false;
    }
    return true;
}

void TraceDestroy(intptr_t handle) { AtraceDestroy(handle); }

} // namespace Hccl
