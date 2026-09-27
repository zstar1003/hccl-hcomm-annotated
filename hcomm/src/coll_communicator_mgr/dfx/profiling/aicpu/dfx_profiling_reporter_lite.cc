/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfx_profiling_reporter_lite.h"
#include "aicpu_ts_thread.h"

namespace Hccl {
DfxProfilingReporterLite::DfxProfilingReporterLite(DfxProfilingHandlerLite* profilingHandlerLite)
    : profilingHandlerLite_(profilingHandlerLite)
{}

DfxProfilingReporterLite::~DfxProfilingReporterLite() {}

HcclResult DfxProfilingReporterLite::Init()
{
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    if (UNLIKELY(profilingHandlerLite_ == nullptr)) {
        HCCL_ERROR("[DfxProfilingReporterLite][Init] profilingHandlerLite is nullptr.");
        return HCCL_E_PTR;
    }
    initializedFlag_ = true;
    return HCCL_SUCCESS;
}

void DfxProfilingReporterLite::ReportStreamTask(TaskInfoCircularQueue* taskQueue, const DfxCommContext& ctx)
{
    if (taskQueue == nullptr) {
        return;
    }
    if (!profilingHandlerLite_->GetProfL1State()) {
        taskQueue->MarkAllRead(); // L1关闭时跳过上报，但仍需标记已读，防止L1重新开启后重复上报
        return;
    }
    profilingHandlerLite_->ReportStreamTaskDetails(*taskQueue, ctx);
    taskQueue->MarkAllRead(); // 将队列begin推到end，标记已上报的task为已读，下次只遍历新增task
}

void DfxProfilingReporterLite::ReportAllTasksLog(const std::vector<hccl::Thread*>& threads) const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    for (auto* thread : threads) {
        auto* aicpuThread = static_cast<hccl::AicpuTsThread*>(thread);
        if (aicpuThread == nullptr) {
            continue;
        }
        auto* taskQueue = aicpuThread->GetTaskInfos();
        if (taskQueue == nullptr || taskQueue->IsEmpty()) {
            continue;
        }
        profilingHandlerLite_->ReportStreamTaskDetailsLog(*taskQueue);
    }
}

void DfxProfilingReporterLite::ReportAllTasks(const std::vector<hccl::Thread*>& threads, const DfxCommContext& ctx)
{
    ReportAllTasksLog(threads);
    if (!profilingHandlerLite_->GetProfL1State()) {
        HCCL_DEBUG("[DfxProfilingReporterLite][ReportAllTasks] L1 is off, MarkAllRead and skip report.");
        for (auto* thread : threads) {
            auto* aicpuThread = static_cast<hccl::AicpuTsThread*>(thread);
            if (aicpuThread == nullptr) {
                continue;
            }
            auto* taskQueue = aicpuThread->GetTaskInfos();
            if (taskQueue != nullptr) {
                taskQueue->MarkAllRead(); // L1关闭时跳过上报，但仍需标记已读，防止L1重新开启后重复上报关闭期间的task
            }
        }
        return;
    }

    for (auto* thread : threads) {
        auto* aicpuThread = static_cast<hccl::AicpuTsThread*>(thread);
        if (aicpuThread == nullptr) {
            continue;
        }
        ReportStreamTask(aicpuThread->GetTaskInfos(), ctx);
    }
}

void DfxProfilingReporterLite::UpdateProfStat() { profilingHandlerLite_->UpdateProfSwitch(); }

} // namespace Hccl
