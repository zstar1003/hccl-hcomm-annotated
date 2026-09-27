/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_engine_res_aicpu_mgr.h"
#include "log.h"

CommEngineResAicpuMgr::CommEngineResAicpuMgr(
    hccl::HcclCommDfxLite& dfx, std::function<HcclResult(bool)> checkExecStatusCallback)
{
    threadMgr_ = std::make_unique<ThreadAicpuMgr>(dfx, std::move(checkExecStatusCallback));
    notifyMgr_ = std::make_unique<NotifyAicpuMgr>();
    HCCL_INFO("[CommEngineResAicpuMgr][%s] constructed", __func__);
}

CommEngineResAicpuMgr::~CommEngineResAicpuMgr() { HCCL_INFO("[CommEngineResAicpuMgr][%s] destroyed", __func__); }

HcclResult CommEngineResAicpuMgr::InitThreads(ThreadMgrAicpuParam* param)
{
    CHK_SMART_PTR_NULL(threadMgr_);
    return threadMgr_->InitThreads(param);
}

HcclResult CommEngineResAicpuMgr::NotifyFree(NotifyMgrAicpuParam* param)
{
    CHK_SMART_PTR_NULL(notifyMgr_);
    return notifyMgr_->NotifyFree(param);
}

HcclResult CommEngineResAicpuMgr::NotifyAlloc(NotifyMgrAicpuParam* param)
{
    CHK_SMART_PTR_NULL(notifyMgr_);
    return notifyMgr_->NotifyAlloc(param);
}

void CommEngineResAicpuMgr::ReserveNotifyCapacity(size_t n)
{
    if (notifyMgr_) {
        notifyMgr_->ReserveNotifyCapacity(n);
    }
}

const std::vector<std::shared_ptr<hccl::Thread>>& CommEngineResAicpuMgr::GetAllThread()
{
    return threadMgr_->GetAllThread();
}

std::shared_mutex& CommEngineResAicpuMgr::GetThreadMutex() { return threadMgr_->GetThreadMutex(); }
