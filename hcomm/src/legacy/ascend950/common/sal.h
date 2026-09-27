/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_SAL_PUB_H
#define HCCL_INC_SAL_PUB_H

#include <string>
#include "types.h"

namespace Hccl {

constexpr int HCCL_BASE_DECIMAL = 10; // 10进制字符串转换

void SaluSleep(u32 usec);

void SalSleep(u32 sec);

std::string SalGetEnv(const char* name);

HcclResult SalStrToULong(const std::string str, int base, u32& val);

s32 SalGetTid();

u64 SalGetCurrentTimestamp();

u64 GetCurAicpuTimestamp();

void SetThreadName(const std::string& threadStr);

#ifndef CCL_LLT
inline void AsmCntvc(uint64_t& cntvct)
{
#if defined __aarch64__
    asm volatile("mrs %0, cntvct_el0" : "=r"(cntvct));
#else
    cntvct = 0;
#endif
}
#endif

inline u64 ProfGetCurCpuTimestamp()
{
#ifndef CCL_LLT
    uint64_t cntvct;
    AsmCntvc(cntvct);
    return cntvct;
#endif
    return 0;
}

} // namespace Hccl
#endif // !HCCL_INC_SAL_PUB_H
