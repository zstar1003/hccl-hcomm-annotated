/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_aicpu_destroy_func.h"
#include "coll_comm_aicpu_mgr.h"
#include <shared_mutex>
#include "kernel_entrance.h"

namespace hccl {
CollCommAicpuDestroyFunc& CollCommAicpuDestroyFunc::GetInstance()
{
    static CollCommAicpuDestroyFunc func;
    return func;
}

void CollCommAicpuDestroyFunc::Call()
{
    if (stopCall_ == true) {
        return;
    }

    HcclResult ret = Process();
    if (ret != HCCL_SUCCESS) {
        stopCall_ = true;
        HCCL_ERROR("[%s]Process fail, set stopCall_[%d] ret[%d]", __func__, stopCall_, ret);
    }
}

HcclResult CollCommAicpuDestroyFunc::Process()
{
    std::vector<std::string> destroyComm;
    {
        std::shared_lock<std::shared_mutex> rwlock(CollCommAicpuMgr::GetInstance().GetMutex());

        std::vector<std::pair<std::string, CollCommAicpu*>> aicpuCommInfo;
        CHK_RET(CollCommAicpuMgr::GetInstance().GetAllComms(aicpuCommInfo));

        for (auto& commInfo : aicpuCommInfo) {
            CollCommAicpu* aicpuComm = commInfo.second;
            CHK_PTR_NULL(aicpuComm);

            if (aicpuComm->GetCommmStatus() == HcclCommStatus::HCCL_COMM_STATUS_INVALID) {
                continue;
            }

            Hccl::KfcCommand cmd = Hccl::KfcCommand::NONE;
            CHK_RET(aicpuComm->BackGroundGetCmd(cmd));
            if (cmd != Hccl::KfcCommand::DESTROY_AICPU_COMM) {
                continue;
            }
            destroyComm.push_back(aicpuComm->GetIdentifier());
            CHK_RET(aicpuComm->BackGroundSetStatus(Hccl::KfcStatus::DESTROY_AICPU_COMM_DONE));

            {
                std::lock_guard<std::mutex> lock(g_taskExpDevMemMapMutex);
                auto it = g_taskExpDevMemMap.find(aicpuComm->GetIdentifier());
                if (it != g_taskExpDevMemMap.end()) {
                    g_taskExpDevMemMap.erase(aicpuComm->GetIdentifier()); // 清理dpu taskexception共享内存
                }
            }

            HCCL_RUN_INFO(
                "[%s]group[%s] Recv DESTROY_AICPU_COMM cmd and set DESTROY_AICPU_COMM_DONE", __func__,
                aicpuComm->GetIdentifier().c_str());
        }
    }

    for (std::string& groupName : destroyComm) {
        (void)(CollCommAicpuMgr::GetInstance().DestroyComm(groupName));
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
