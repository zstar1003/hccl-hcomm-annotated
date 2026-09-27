/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADAPTER_V2_SAL_STUB_H
#define ADAPTER_V2_SAL_STUB_H

namespace Hccl {

extern HcclResult sal_memset(void* dest, size_t destMaxSize, int c, size_t count);
s32 SalGetPid();
s32 SalGetTid();
s32 sal_vsnprintf(char* strDest, size_t destMaxSize, size_t count, const char* format, va_list argList);

} // namespace Hccl

#endif
