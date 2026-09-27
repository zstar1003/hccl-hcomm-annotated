/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_TRACE_H
#define HCCL_INC_ADAPTER_TRACE_H

#include <hccl/hccl_types.h>
#include <hccl/base.h>
#include "atrace_types.h"

HcclResult hrtOpenTrace();
void hrtTraceDestroy(TraHandle handle);
HcclResult hrtTraceSubmit(TraHandle handle, const void* buffer, u32 bufSize);
HcclResult hrtTraceCreateWithAttr(const char* objName, TraHandle& handle);
HcclResult hrtTraceSetGlobalAttr(const TraceGlobalAttr* attr);
HcclResult hrtTraceSave(TracerType tracerType, bool syncFlag);
#endif
