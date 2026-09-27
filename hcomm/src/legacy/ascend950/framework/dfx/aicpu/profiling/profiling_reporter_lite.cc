/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_reporter_lite.h"

namespace Hccl {
constexpr size_t TASK_INFO_BATCH_RESERVE_SIZE = 8192;
ProfilingReporterLite::ProfilingReporterLite(
    MirrorTaskManagerLite* mirrorTaskMgrLite, ProfilingHandlerLite* profilingHandlerLite, [[maybe_unused]] bool isIndop)
    : mirrorTaskMgrLite_(mirrorTaskMgrLite),
      profilingHandlerLite_(profilingHandlerLite)
{}

ProfilingReporterLite::~ProfilingReporterLite() {}

HcclResult ProfilingReporterLite::Init()
{
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    if (UNLIKELY(mirrorTaskMgrLite_ == nullptr || profilingHandlerLite_ == nullptr)) {
        HCCL_ERROR("[ProfilingReporterLite][Init] mirrorTaskMgrLite or profilingHandlerLite is nullptr.");
        return HCCL_E_PTR;
    }
    mirrorTaskMgrLite_->RegFullyCallBack([this]() {
        ReportAllTasks();
    });
    initializedFlag_ = true;
    return HCCL_SUCCESS;
}

/*
 *  (*currQueue) == Queue<std::unique_ptr<TaskInfo>> = QUEUE
 *  QUEUE.Begin() =std::shared_ptr<Iterator<unique_ptr<taskInfo>>
 *  *QUEUE.Begin() = Iterator<unique_ptr<taskInfo>
 *  *(*QUEUE.Begin()) = unique_ptr<taskInfo>
 *  *(*(*QUEUE.Begin())) = taskInfo;
 *  taskInfo.push_back((*(*((*currQueue).Begin())));
 */
// 所有的迭代器都是make_shared 永不为空  底层修改之后 需求再看下
void ProfilingReporterLite::ReportAllTasksLog() const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    for (auto it = mirrorTaskMgrLite_->Begin(); it != mirrorTaskMgrLite_->End(); ++it) {
        u32 streamId = it->first;
        Queue<std::unique_ptr<TaskInfo>>* currQueue = it->second.queue.get();
        if (currQueue == nullptr || currQueue->Begin() == nullptr || currQueue->Tail() == nullptr) {
            continue;
        }
        bool logAll = (lastPoses_.find(streamId) == lastPoses_.end());
        for (auto logIter = currQueue->Begin(); *logIter != *currQueue->End(); ++(*logIter)) {
            if (*(*logIter) == nullptr) {
                continue;
            }
            if (!logAll
                && *logIter
                       == *lastPoses_.at(
                           streamId)) { //  找到旧 Tail 位置后设为 logAll=true，continue 跳过该位置，后续全打印
                logAll = true;
                continue;
            }
            if (!logAll) {
                continue;
            }
            TaskInfo task = (*(*(*logIter)));
            HCCL_INFO("[ProfilingReporterLite][ReportAllTasks] %s", task.Describe().c_str());
        }
    }
}

void ProfilingReporterLite::ReportAllTasks()
{
    ReportAllTasksLog();
    if (ProfilingHandlerLite::GetInstance().GetProfL1State() == false) {
        HCCL_DEBUG("[ProfilingReporterLite][ReportAllTasks] GetProfL1State is false, UpdateAllLastPos and skip report");
        UpdateAllLastPos();
        return;
    }

    std::vector<TaskInfo*> taskInfo;
    taskInfo.reserve(TASK_INFO_BATCH_RESERVE_SIZE);
    for (auto it = mirrorTaskMgrLite_->Begin(); it != mirrorTaskMgrLite_->End(); ++it) {
        u32 streamId = it->first;
        Queue<std::unique_ptr<TaskInfo>>* currQueue = it->second.queue.get();
        if (currQueue == nullptr || (*(*(currQueue->Begin()))) == nullptr || (*(*(currQueue->Tail()))) == nullptr) {
            HCCL_WARNING("[ProfilingReporterLite][ReportAllTasks] currQueue is nullptr, continue to next task.");
            continue;
        }
        if (lastPoses_.find(streamId) == lastPoses_.end()) {
            taskInfo.emplace_back((*currQueue->Begin())->get());
            lastPoses_[streamId] = currQueue->Begin();
        }
        auto endPos = currQueue->Tail();
        auto iter = lastPoses_[streamId];
        ++(*iter);
        for (; (*(iter)) != (*(currQueue->End())); ++(*(iter))) {
            taskInfo.emplace_back((*iter)->get());
        }
        lastPoses_[streamId] = endPos;
    }
    ProfilingHandlerLite::GetInstance().ReportHcclTaskDetails(taskInfo);
}

void ProfilingReporterLite::UpdateProfStat(void) const { ProfilingHandlerLite::GetInstance().UpdateProfSwitch(); }

void ProfilingReporterLite::UpdateAllLastPos()
{
    for (auto it = mirrorTaskMgrLite_->Begin(); it != mirrorTaskMgrLite_->End(); ++it) {
        u32 streamId = it->first;
        Queue<std::unique_ptr<TaskInfo>>* currQueue = it->second.queue.get(); // 一旦有streamid 必有queue 必不为空

        auto endPos = currQueue->Tail();
        lastPoses_[streamId] = endPos;
    }
}

} // namespace Hccl
