/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_KERNEL_ENTRANCE_H
#define HCCLV2_KERNEL_ENTRANCE_H

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include "log.h"

extern std::unordered_map<std::string, void*> g_taskExpDevMemMap;
extern std::mutex g_taskExpDevMemMapMutex;
extern "C" {
__attribute__((visibility("default"))) uint32_t HcclKernelEntrance(void* args);

__attribute__((visibility("default"))) uint32_t HcclUpdateCommKernelEntrance(void* args);

__attribute__((visibility("default"))) uint32_t HcclDpuTaskexpShmemRestore(void* args);
};

#endif // HCCL_KERNEL_ENTRANCE_H
