/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CONST_VAL_H
#define HCCLV2_CONST_VAL_H
namespace Hccl {
constexpr u32 CANN_VERSION_MAX_LEN = 50; // Cann版本信息的最大长度为50
constexpr u32 SOC_VERSION_MAX_LEN = 32;  // soc version的最大长度为32
constexpr u32 INVALID_U32 = UINT32_MAX;
constexpr u64 INVALID_U64 = UINT64_MAX;
constexpr s32 INVALID_RANKID = INT32_MAX;
} // namespace Hccl
#endif
