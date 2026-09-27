/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_INTERFACE_H
#define DPU_INTERFACE_H

#include <cstdint>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include "task_service.h"

extern std::mutex g_serMapMutex;
extern std::unordered_map<std::string, std::unordered_map<uint32_t, std::unique_ptr<Hccl::TaskService>>>
    g_taskServiceMap;
extern std::unordered_map<std::string, std::unordered_map<uint32_t, void*>> g_taskExpMemMap;
extern "C" {
__attribute__((visibility("default"))) uint32_t RunDpuRpcSrvLaunch(const uint64_t args);
}

namespace Hccl {
constexpr uint8_t TASK_TERMINATE_RESPONSE = 3;

struct DpuKernelLaunchParam {
    u64 memorySize;
    void* deviceMem;
    void* hostMem;
    uint32_t deviceId;
    std::string commId;
    void* taskExpMem;
};

} // namespace Hccl

#endif // DPU_INTERFACE_H
