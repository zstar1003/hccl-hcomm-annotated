/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ORION_ADAPTER_RTS_H
#define HCCLV2_ORION_ADAPTER_RTS_H

#include <string>
#include <unordered_map>
#include "types.h"
#include "dev_type.h"
#include "rt_external.h"
#include "orion_adapter_rts.h"

namespace Hccl {

DevType HrtGetDeviceType();
void HrtGetSocVer(char_t* chipVer, const u32 size);
void HrtUbDevQueryInfo(rtUbDevQueryCmd cmd, void* devInfo);
u32 HrtGetDevicePhyIdByUserDevId(u32 deviceLogicId);
void* HrtMalloc(u64 size, aclrtMemType_t memType);
void HrtFree(void* devPtr);
void HrtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t count, rtMemcpyKind_t kind);
s32 HrtGetDevice();

class HcclMainboardId;
HcclResult HrtGetMainboardId(uint32_t deviceLogicId, HcclMainboardId& hcclMainboardId);

} // namespace Hccl

#endif // HCCLV2_ORION_ADAPTER_RTS_H
