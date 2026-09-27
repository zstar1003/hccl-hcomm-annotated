/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SRC_DlTraceFunction_H
#define HCCL_SRC_DlTraceFunction_H

#include <functional>
#include <mutex>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "atrace_pub.h"

namespace hccl {
class DlTraceFunction {
public:
    virtual ~DlTraceFunction();
    static DlTraceFunction& GetInstance();
    HcclResult DlTraceFunctionInit();

    std::function<void(TraHandle handle)> dlAtraceDestroy{};
    std::function<HcclResult(TraHandle handle, const void* buffer, u32 bufSize)> dlAtraceSubmit{};
    std::function<TraHandle(int tracerType, const char* objName, const TraceAttr* attr)> dlAtraceCreateWithAttr{};
    std::function<TraStatus(const TraceGlobalAttr* attr)> dlAtraceSetGlobalAttr{};
    std::function<TraStatus(TracerType tracerType, bool syncFlag)> dlAtraceSave{};

    std::function<void(TraHandle handle)> dlUtraceDestroy{};
    std::function<HcclResult(TraHandle handle, const void* buffer, u32 bufSize)> dlUtraceSubmit{};
    std::function<TraHandle(int tracerType, const char* objName, const TraceAttr* attr)> dlUtraceCreateWithAttr{};
    std::function<TraStatus(const TraceGlobalAttr* attr)> dlUtraceSetGlobalAttr{};
    std::function<TraStatus(TracerType tracerType, bool syncFlag)> dlUtraceSave{};

private:
    void* handle_;
    std::mutex handleMutex_;
    DlTraceFunction(const DlTraceFunction&);
    DlTraceFunction& operator=(const DlTraceFunction&);
    DlTraceFunction();
    HcclResult DlATraceFunctionInterInit();
    HcclResult DlUTraceFunctionInterInit();
};
} // namespace hccl
#endif
