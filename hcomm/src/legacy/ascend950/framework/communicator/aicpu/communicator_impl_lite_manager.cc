/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "communicator_impl_lite_manager.h"
#include "aicpu_daemon_service.h"
#include "aicpu_comm_destroy_func.h"
#include "ns_recovery_handler_func.h"
#include "task_exception_func.h"
#include "task_exception_handler_lite.h"
namespace Hccl {

CommunicatorImplLiteMgr::CommunicatorImplLiteMgr()
{
    HCCL_INFO("CommunicatorImplLiteMgr:: start");
    static auto commandToBackGroud = CommandToBackGroud::Default;
    HCCL_INFO("CommunicatorImplLiteMgr:: gen daemon service run func");
    static auto daemonServiceRun = [](void* info) {
        AicpuDaemonService::GetInstance().ServiceRun(info);
    };
    HCCL_INFO("CommunicatorImplLiteMgr:: gen daemon service stop func");
    static auto daemonServiceStop = [](void* info) {
        AicpuDaemonService::GetInstance().ServiceStop(info);
    };

    // 注册守护进程函数
    AicpuDaemonService::GetInstance().Register(&TaskExceptionFunc::GetInstance());
    AicpuDaemonService::GetInstance().Register(&AicpuCommDestroyFunc::GetInstance());
    AicpuDaemonService::GetInstance().Register(&NsRecoveryHandlerFunc::GetInstance());
    // 注册TaskExceptionFunc回调函数
    TaskExceptionHandlerLite::GetInstance();

    // 启动背景线程
    if (StartMC2MaintenanceThread != nullptr) {
        StartMC2MaintenanceThread(daemonServiceRun, &commandToBackGroud, daemonServiceStop, &commandToBackGroud);
        HCCL_INFO("[CommunicatorImplLiteMgr] start BackGround thread success.");
    } else {
        HCCL_WARNING("Aicpu api StartMC2MaintenanceThread func is nullptr");
    }

    HCCL_INFO("CommunicatorImplLiteMgr::end");
}

CommunicatorImplLiteMgr::~CommunicatorImplLiteMgr() { HCCL_INFO("CommunicatorImplLiteMgr Destroy"); }

CommunicatorImplLiteMgr& CommunicatorImplLiteMgr::GetInstance()
{
    static CommunicatorImplLiteMgr communicatorLiteMgr;
    return communicatorLiteMgr;
}

CommunicatorImplLite* CommunicatorImplLiteMgr::Get(const u32 commIdIndex)
{
    std::lock_guard<std::mutex> lock(serialMutex);
    // 通过commIdIndex查找communicatorImplLites中是否存在，不存在再处理资源
    auto iter = communicatorImplLites.find(commIdIndex);
    if (iter != communicatorImplLites.end()) {
        HCCL_INFO("CommunicatorImplLiteMgr::find commIdIndex [%u] in communicatorImplLites", commIdIndex);
        unique_lock<mutex> aicpuLock(communicatorImplLites[commIdIndex]->GetAicpuMc2Mutex());
        communicatorImplLites[commIdIndex]->SetIsFirstUsedToFalse();
        aicpuLock.unlock();
        return communicatorImplLites[commIdIndex].get();
    }

    try {
        communicatorImplLites[commIdIndex] = make_unique<CommunicatorImplLite>(commIdIndex);
    } catch (...) {
        HCCL_ERROR("new CommunicatorImplLite failed, commIdIndex[%u]", commIdIndex);
        return nullptr;
    }

    return communicatorImplLites[commIdIndex].get();
}

std::vector<CommunicatorImplLite*> CommunicatorImplLiteMgr::GetAll()
{
    std::lock_guard<std::mutex> lock(serialMutex);
    std::vector<CommunicatorImplLite*> vec;
    for (auto iter = communicatorImplLites.begin(); iter != communicatorImplLites.end(); iter++) {
        if (iter->second != nullptr) {
            vec.push_back(iter->second.get());
        }
    }
    return vec;
}

void CommunicatorImplLiteMgr::DestroyComm(u32 commIdIndex)
{
    std::lock_guard<std::mutex> lock(serialMutex);
    HCCL_INFO("Destroy comm start commIdIndex[%u]", commIdIndex);
    communicatorImplLites.erase(commIdIndex);
    HCCL_INFO("Destroy comm success commIdIndex[%u]", commIdIndex);
}
} // namespace Hccl
