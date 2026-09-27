/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mirror_task_manager_lite.h"

namespace Hccl {

MirrorTaskManagerLite::MirrorTaskManagerLite() {}

void MirrorTaskManagerLite::RegFullyCallBack(std::function<void()> callBack)
{
    fullyCallBack_ = callBack;
    return;
}

void MirrorTaskManagerLite::RegGetRemoteRankCallBack(std::function<u32(u64)> callBack)
{
    getRemoteRankCallback_ = callBack;
    return;
}

HcclResult MirrorTaskManagerLite::AddTaskInfo(u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle)
{
    u32 remoteRankId = getRemoteRankCallback_ ? getRemoteRankCallback_(handle) : INVALID_U32;
    PrintTaskLog(streamId, taskId, taskParam, remoteRankId);

    auto it = streamQueues_.find(streamId);
    if (UNLIKELY(it == streamQueues_.end())) {
        auto cq = std::make_unique<CircularQueue<std::unique_ptr<TaskInfo>>>(MAX_AICPU_CIRCULAR_QUEUE_LENGTH);
        auto entry = StreamQueueEntry{std::move(cq), MAX_AICPU_CIRCULAR_QUEUE_LENGTH, 0};
        it = streamQueues_.emplace(streamId, std::move(entry)).first;
    }

    auto& entry = it->second;
    if (UNLIKELY(entry.taskNum == entry.capacity)) {
        fullyCallBack_();
        entry.taskNum = 0;
    }

    auto& taskInfo = entry.queue->GetAndUpdate();
    if (taskInfo == nullptr) {
        taskInfo = std::make_unique<Hccl::TaskInfo>(
            streamId, taskId, INVALID_U32, taskParam, currDfxOpInfo_, taskParam.isMaster);
    } else {
        taskInfo->streamId_ = streamId;
        taskInfo->taskId_ = taskId;
        taskInfo->taskParam_ = taskParam;
        taskInfo->dfxOpInfo_ = currDfxOpInfo_;
        taskInfo->remoteRank_ = INVALID_U32;
        taskInfo->isMaster_ = taskParam.isMaster;
    }

    taskInfo->channelHandle_ = handle;
    taskInfo->getRemoteRankByHandle_ = getRemoteRankCallback_;
    entry.taskNum++;
    return HCCL_SUCCESS;
}

void MirrorTaskManagerLite::AddTaskInfo(std::unique_ptr<TaskInfo>&& taskInfo)
{
    if (UNLIKELY(taskInfo == nullptr)) {
        THROW<InternalException>(StringFormat("MirrorTaskManagerLite::AddTaskInfo taskInfo is nullptr"));
    }

    auto it = streamQueues_.find(taskInfo->streamId_);
    if (UNLIKELY(it == streamQueues_.end())) {
        auto cq = std::make_unique<CircularQueue<std::unique_ptr<TaskInfo>>>(MAX_AICPU_CIRCULAR_QUEUE_LENGTH);
        auto entry = StreamQueueEntry{std::move(cq), MAX_AICPU_CIRCULAR_QUEUE_LENGTH, 0};
        it = streamQueues_.emplace(taskInfo->streamId_, std::move(entry)).first;
    }

    auto& entry = it->second;
    if (UNLIKELY(entry.taskNum == entry.capacity)) {
        fullyCallBack_();
        entry.taskNum = 0;
    }

    entry.queue->Append(std::move(taskInfo));
    entry.taskNum++;
    return;
}

HcclResult MirrorTaskManagerLite::SetCurrDfxOpInfo(std::shared_ptr<DfxOpInfo> dfxOpInfo)
{
    CHK_PTR_NULL(dfxOpInfo);
    currDfxOpInfo_ = std::move(dfxOpInfo);
    return HCCL_SUCCESS;
}

std::shared_ptr<DfxOpInfo> MirrorTaskManagerLite::GetCurrDfxOpInfo() const { return currDfxOpInfo_; }

TaskInfoQueue* MirrorTaskManagerLite::GetQueue(u32 streamId) const
{
    auto it = streamQueues_.find(streamId);
    if (it == streamQueues_.end()) {
        HCCL_ERROR("MirrorTaskManagerLite::GetQueue streamId(sqId)[%u] out of range", streamId);
        return nullptr;
    }
    return it->second.queue.get();
}

TaskInfo* MirrorTaskManagerLite::GetTaskInfo(u32 streamId, u32 taskId) const
{
    TaskInfoQueue* queue = nullptr;
    try {
        queue = GetQueue(streamId);
    } catch (HcclException& e) {
        HCCL_ERROR("Hccl exception %s was caught.", e.what());
        return nullptr;
    }

    auto FindTask = [taskId](const std::unique_ptr<TaskInfo>& taskInfo) {
        return taskInfo->taskId_ == taskId;
    };

    auto task = *queue->Find(FindTask);
    if (task == *queue->End()) {
        return nullptr;
    };

    HCCL_INFO("[MirrorTaskManagerLite][GetTaskInfo]find streamId(sqId)[%u] taskId(sqeId)[%u]", streamId, taskId);

    return (*task).get();
}

std::unordered_map<u32, StreamQueueEntry>::iterator MirrorTaskManagerLite::Begin() { return streamQueues_.begin(); }

std::unordered_map<u32, StreamQueueEntry>::iterator MirrorTaskManagerLite::End() { return streamQueues_.end(); }

MirrorTaskManagerLite::~MirrorTaskManagerLite() {}

} // namespace Hccl
