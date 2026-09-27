/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_ATRACE_H
#define HCCL_INC_ADAPTER_ATRACE_H

#include <types.h>
#include <chrono>
namespace Hccl {

bool CheckLogTime(std::chrono::steady_clock::time_point& lastTime);
intptr_t TraceCreate(const char* objName);
bool TraceSubmit(intptr_t handle, const void* buffer, uint32_t bufSize);
void TraceDestroy(intptr_t handle);
extern std::chrono::steady_clock::time_point lastLogTimeTrace; // 抑制日志刷屏时间戳

} // namespace Hccl
#endif
