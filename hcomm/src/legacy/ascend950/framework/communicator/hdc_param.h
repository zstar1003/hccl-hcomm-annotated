/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_HDC_PARAM_H
#define HCCLV2_HDC_PARAM_H

#include "types.h"

namespace Hccl {
constexpr u32 HCCLV2_HDC_TYPE_D2H = 0;
constexpr u32 HCCLV2_HDC_TYPE_H2D = 1;

struct HDCommunicateParams {
    u64 hostAddr{0};
    u64 deviceAddr{0};
    u64 readCacheAddr{0};
    u32 devMemSize{0};
    u32 buffLen{0};
    u32 flag{0};
};

} // namespace Hccl
#endif // HCCLV2_HDC_PARAM_H
