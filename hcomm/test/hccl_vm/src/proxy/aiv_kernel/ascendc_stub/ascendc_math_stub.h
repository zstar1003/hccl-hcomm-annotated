/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_ASCENDC_MATH_STUB_H
#define HCCL_VM_ASCENDC_MATH_STUB_H

#include <cstdint>
#include "ascendc_base_stub.h"
#include "ascendc_memory_stub.h"

namespace AscendC {
template <typename T>
__aicore__ inline void
Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)
{}

template <typename T>
__aicore__ inline void
Max(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)
{}

template <typename T>
__aicore__ inline void
Min(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)
{}
} // namespace AscendC

#endif // HCCL_VM_ASCENDC_MATH_STUB_H
