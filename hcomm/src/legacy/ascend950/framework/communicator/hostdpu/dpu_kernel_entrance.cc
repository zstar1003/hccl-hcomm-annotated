/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_kernel_entrance.h"
#include "log.h"
#include "acl/acl_rt.h"

using namespace Hccl;
using u64 = unsigned long long;
std::unordered_map<std::string, std::unordered_map<uint32_t, std::unique_ptr<Hccl::TaskService>>> g_taskServiceMap;
std::unordered_map<std::string, std::unordered_map<uint32_t, void*>> g_taskExpMemMap;
std::mutex g_serMapMutex;
extern "C" {
__attribute__((visibility("default"))) uint32_t RunDpuRpcSrvLaunch(const uint64_t args)
{
    HCCL_INFO("[%s] Launch Dpu Kernel: 0x%lx", __func__, args);
    if (args == 0) {
        HCCL_ERROR("[%s] args is null.", __func__);
        return HCCL_E_PARA;
    }

    if (reinterpret_cast<uint64_t>(args) + sizeof(DpuKernelLaunchParam) < reinterpret_cast<uint64_t>(args)) {
        HCCL_ERROR("[%s] Invalid args address.", __func__);
        return HCCL_E_PARA;
    }
    // 解析参数信息
    DpuKernelLaunchParam* params = reinterpret_cast<DpuKernelLaunchParam*>(args);

    HCCL_RUN_INFO(
        "[%s] DpuKernelLaunchParam{commId:%s; memorySize:%lu; deviceMem:%p; hostMem:%p, taskExpMem:%p; devId:%u}",
        __func__, params->commId.c_str(), params->memorySize, params->deviceMem, params->hostMem, params->taskExpMem,
        params->deviceId);

    if (params->memorySize == 0) {
        HCCL_ERROR("[%s] memorySize is 0.", __func__);
        return HCCL_E_PARA;
    }
    if (params->deviceMem == nullptr || params->hostMem == nullptr || params->taskExpMem == nullptr) {
        HCCL_ERROR(
            "[%s] deviceMem[%p] or hostMem[%p] or taskExpMem[%p] is nullptr.", __func__, params->deviceMem,
            params->hostMem, params->taskExpMem);
        return HCCL_E_PARA;
    }

    // 实例化TaskService
    std::unique_ptr<Hccl::TaskService> taskService = std::make_unique<Hccl::TaskService>(
        params->deviceMem, params->memorySize, params->hostMem, params->memorySize, params->commId, params->deviceId);

    aclError ret = aclrtSetDevice(params->deviceId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] set device fail. DeviceId: %u.", __func__, params->deviceId);
        return HCCL_E_RUNTIME;
    }

    // 设置到通信域中保存 map<commId, map<devid, TaskService>>与map<commId, map<devid,taskExpMem>>
    HCCL_INFO("[%s] save TaskService", __func__);
    Hccl::TaskService* svcPtr = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_serMapMutex);
        g_taskServiceMap[params->commId][params->deviceId] = std::move(taskService);
        g_taskExpMemMap[params->commId][params->deviceId] = params->taskExpMem;
        svcPtr = g_taskServiceMap[params->commId][params->deviceId].get();
    }

    // Run
    HCCL_INFO("[%s] start to TaskRun", __func__);
    HcclResult hcclRet = svcPtr->TaskRun();
    if (hcclRet != HCCL_SUCCESS) {
        uint8_t newFlag = TASK_TERMINATE_RESPONSE;
        errno_t cpyRet = memcpy_s(static_cast<uint8_t*>(params->deviceMem), sizeof(newFlag), &newFlag, sizeof(newFlag));
        if (cpyRet != EOK) {
            HCCL_ERROR("set exit flag failed: %d", cpyRet);
            return HCCL_E_INTERNAL;
        }
    }

    // End
    return hcclRet;
}
}
