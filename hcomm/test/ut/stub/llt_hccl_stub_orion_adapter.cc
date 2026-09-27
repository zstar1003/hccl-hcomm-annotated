/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"

extern "C" {

// Stub implementation for HrtSetXpuDevice
HcclResult HrtSetXpuDevice(uint32_t devType, const uint32_t devId)
{
    (void)devType;
    (void)devId;
    return HCCL_SUCCESS;
}

// Stub implementation for HrtResetXpuDevice
HcclResult HrtResetXpuDevice(uint32_t devType, const uint32_t devId)
{
    (void)devType;
    (void)devId;
    return HCCL_SUCCESS;
}
}
