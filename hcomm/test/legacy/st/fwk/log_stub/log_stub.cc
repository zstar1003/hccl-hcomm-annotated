/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "slog.h"
#include "slog_api.h"
#include "atrace_pub.h"
int CheckLogLevel(int moduleId, int logLevel) { return 0; }

void DlogRecord(int moduleId, int level, const char* fmt, ...) {}

void DlogInner(int moduleId, int level, const char* fmt, ...) {}

TraHandle AtraceCreateWithAttr(TracerType tracerType, const char* objName, const TraceAttr* attr) { return 0; }

TraStatus AtraceSubmit(TraHandle handle, const void* buffer, uint32_t bufSize) { return 0; }

void AtraceDestroy(TraHandle handle) { return; }
