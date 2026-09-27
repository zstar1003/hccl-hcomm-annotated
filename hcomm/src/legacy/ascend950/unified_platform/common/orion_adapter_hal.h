/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ORION_ADAPTER_HAL_H
#define ORION_ADAPTER_HAL_H

#include <hccl/hccl_types.h>

namespace Hccl {
HcclResult HrtHalDrvQueryProcessHostPid(
    int pid, unsigned int* chipId, unsigned int* vfid, unsigned int* hostPid, unsigned int* cpType);
HcclResult HrtHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t& value);
HcclResult HrtHalHostRegister(void* srcPtr, uint64_t size, uint32_t flag, int32_t deviceLogicId, void** dstPtr);
HcclResult HrtHalHostUnregister(void* ptr, int32_t deviceLogicId);
} // namespace Hccl
#endif
