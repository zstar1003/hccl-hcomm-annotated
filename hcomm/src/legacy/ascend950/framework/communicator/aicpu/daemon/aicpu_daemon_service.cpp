/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_daemon_service.h"
#include <algorithm>
#include "sal.h"
#include "log.h"

namespace Hccl {

constexpr u32 TEN_MILLISECOND_OF_USLEEP = 10000;
std::mutex AicpuDaemonService::mutexForFuncs_;

AicpuDaemonService& AicpuDaemonService::GetInstance()
{
    static AicpuDaemonService daemonService;
    return daemonService;
}

void AicpuDaemonService::ServiceRun(void* info)
{
    HCCL_RUN_INFO("Start background thread");
    auto commandToBackGroud = static_cast<CommandToBackGroud*>(info);
    while (true) {
        if (*commandToBackGroud == CommandToBackGroud::Stop) {
            HCCL_RUN_INFO("Background thread returned");
            break;
        }

        std::unique_lock<std::mutex> lock(mutexForFuncs_);
        for (auto& func : daemonFuncs) {
            func->Call();
            if (needBreak) {
                break;
            }
        }
        lock.unlock();
        if (needBreak) {
            HCCL_RUN_INFO("Background thread needBreak");
            break;
        }

        SaluSleep(TEN_MILLISECOND_OF_USLEEP);
    }
    HCCL_RUN_INFO("Exit background thread");
}

void AicpuDaemonService::ServiceStop(void* info) const
{
    auto commandToBackGroud = static_cast<CommandToBackGroud*>(info);
    *commandToBackGroud = CommandToBackGroud::Stop;
    HCCL_INFO("Stop background thread");
}

void AicpuDaemonService::Register(DaemonFunc* daemonFunc)
{
    if (daemonFunc == nullptr) {
        HCCL_ERROR("[AicpuDaemonService][Register] daemonFunc is nullptr");
        return;
    }
    std::unique_lock<std::mutex> lock(mutexForFuncs_);
    if (std::find(daemonFuncs.begin(), daemonFuncs.end(), daemonFunc) != daemonFuncs.end()) {
        HCCL_INFO("Background thread daemonFunc already registered, daemonFunc[%p]", daemonFunc);
        return;
    }
    daemonFuncs.push_back(daemonFunc);
    HCCL_INFO("Background thread register daemonFunc, daemonFunc[%p]", daemonFunc);
}

void AicpuDaemonService::Unregister(DaemonFunc* daemonFunc)
{
    if (daemonFunc == nullptr) {
        HCCL_ERROR("[AicpuDaemonService][Unregister] daemonFunc is nullptr");
        return;
    }
    std::unique_lock<std::mutex> lock(mutexForFuncs_);
    auto it = std::remove(daemonFuncs.begin(), daemonFuncs.end(), daemonFunc);
    if (it != daemonFuncs.end()) {
        daemonFuncs.erase(it, daemonFuncs.end());
        HCCL_INFO("Background thread unregister daemonFunc, daemonFunc[%p]", daemonFunc);
    }
}

void AicpuDaemonService::Break()
{
    needBreak = true;
    HCCL_INFO("Background thread received break");
}
} // namespace Hccl
