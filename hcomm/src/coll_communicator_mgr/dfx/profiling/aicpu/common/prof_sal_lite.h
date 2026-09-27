/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROF_SAL_LITE_H
#define PROF_SAL_LITE_H

#include <sys/syscall.h>
#include <cstdint>

namespace Hccl {

inline int32_t SalGetTidLite()
{
    thread_local int32_t cachedTid = static_cast<int32_t>(syscall(SYS_gettid));
    return cachedTid;
}

inline uint64_t ProfGetCurCpuTimestampLite()
{
#ifndef CCL_LLT
    uint64_t cntvct = 0;
#if defined __aarch64__
    asm volatile("mrs %0, cntvct_el0" : "=r"(cntvct));
#endif
    return cntvct;
#endif
    return 0;
}

} // namespace Hccl

#endif // PROF_SAL_LITE_H
