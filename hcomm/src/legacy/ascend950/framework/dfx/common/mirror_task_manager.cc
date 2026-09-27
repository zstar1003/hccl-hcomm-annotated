/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mirror_task_manager.h"

namespace Hccl {

MirrorTaskManager::MirrorTaskManager(u32 devId, GlobalMirrorTasks* globalMirrorTasks, bool devUsed)
    : devId_(devId),
      globalMirrorTasks_(globalMirrorTasks),
      devUsed_(devUsed)
{}

void MirrorTaskManager::RegFullyCallBack(std::function<void()> callBack)
{
    fullyCallBack_ = callBack;
    return;
}

QueueType MirrorTaskManager::GetQueueType() const
{
    if (currDfxOpInfo_ == nullptr) {
        HCCL_WARNING("[MirrorTaskManager][%s]currDfxOpInfo_ is nullptr, return default Circular_Queue!", __func__);
        return QueueType::Circular_Queue;
    }
    QueueType queueType = QueueType::Vector_Queue;

    if (devUsed_ || isStaticGraphMode_ || (opMode_ == OpMode::OPBASE)) {
        queueType = QueueType::Circular_Queue;
    }
    return queueType;
}

void MirrorTaskManager::AddTaskInfo(std::unique_ptr<TaskInfo>&& taskInfo)
{
    if (UNLIKELY(taskInfo == nullptr)) {
        THROW<InternalException>(StringFormat("MirrorTaskManager::AddTaskInfo taskInfo is nullptr"));
    }
    bool needCallback = false;
    std::unique_lock<std::mutex> lock(profMutex);
    if (taskInfo->dfxOpInfo_ == nullptr) {
        taskInfo->dfxOpInfo_ = currDfxOpInfo_;
    }

    auto emplaceResult
        = streamQueues_.emplace(taskInfo->streamId_, MirrorStreamQueueEntry{nullptr, QueueType::Vector_Queue, 0});
    MirrorStreamQueueEntry* entryPtr = &emplaceResult.first->second;
    if (emplaceResult.second) {
        entryPtr->queueType = GetQueueType();
        entryPtr->queue = &(globalMirrorTasks_->CreateQueue(devId_, taskInfo->streamId_, entryPtr->queueType));
    }
    if (UNLIKELY(entryPtr->taskNum == entryPtr->queue->Capacity())) {
        needCallback = true;
        entryPtr->taskNum = 0;
    }

    if (needCallback && fullyCallBack_ != nullptr) {
        lock.unlock();
        fullyCallBack_();
        lock.lock();
        auto queueIt = streamQueues_.find(taskInfo->streamId_);
        if (queueIt == streamQueues_.end()) {
            THROW<InternalException>(StringFormat(
                "MirrorTaskManager::AddTaskInfo streamId[%u] not found after callback", taskInfo->streamId_));
        }
        entryPtr = &queueIt->second;
    }
    auto& slot = entryPtr->queue->GetAndUpdate();
    slot = std::move(taskInfo);
    entryPtr->taskNum++;
    return;
}

HcclResult MirrorTaskManager::AddTaskInfo(
    u32 streamId, u32 taskId, u32 remoteRankId, const TaskParam& taskParam, std::shared_ptr<DfxOpInfo> dfxOpInfo,
    bool isMaster, u32 tid)
{
    bool needCallback = false;
    std::unique_lock<std::mutex> lock(profMutex);
    if (dfxOpInfo == nullptr) {
        dfxOpInfo = currDfxOpInfo_;
    }

    auto emplaceResult = streamQueues_.emplace(streamId, MirrorStreamQueueEntry{nullptr, QueueType::Vector_Queue, 0});
    MirrorStreamQueueEntry* entryPtr = &emplaceResult.first->second;
    if (emplaceResult.second) {
        entryPtr->queueType = GetQueueType();
        entryPtr->queue = &(globalMirrorTasks_->CreateQueue(devId_, streamId, entryPtr->queueType));
    }
    if (UNLIKELY(entryPtr->taskNum == entryPtr->queue->Capacity())) {
        needCallback = true;
        entryPtr->taskNum = 0;
    }

    if (needCallback && fullyCallBack_ != nullptr) {
        lock.unlock();
        fullyCallBack_();
        lock.lock();
        auto queueIt = streamQueues_.find(streamId);
        if (queueIt == streamQueues_.end()) {
            HCCL_ERROR("[MirrorTaskManager][AddTaskInfo] streamId[%u] not found after callback", streamId);
            return HCCL_E_INTERNAL;
        }
        entryPtr = &queueIt->second;
    }

    auto& slot = entryPtr->queue->GetAndUpdate();
    if (UNLIKELY(slot == nullptr)) {
        slot = std::make_unique<TaskInfo>(streamId, taskId, remoteRankId, taskParam, dfxOpInfo, isMaster);
    } else {
        slot->streamId_ = streamId;
        slot->taskId_ = taskId;
        slot->taskParam_ = taskParam;
        slot->dfxOpInfo_ = dfxOpInfo;
        slot->remoteRank_ = remoteRankId;
        slot->isMaster_ = isMaster;
        slot->channelHandle_ = INVALID_U64;
        slot->getRemoteRankByHandle_ = nullptr;
    }
    slot->taskParam_.tid = tid;
    entryPtr->taskNum++;
    return HCCL_SUCCESS;
}

bool MirrorTaskManager::IsStaticGraphMode(const CollOperator& collOperator) const
{
    return (collOperator.staticAddr == false) && (collOperator.staticShape == false);
}

void MirrorTaskManager::SetCurrDfxOpInfo(std::shared_ptr<DfxOpInfo> dfxOpInfo)
{
    if (dfxOpInfo == nullptr) {
        HCCL_ERROR("[MirrorTaskManager][SetCurrDfxOpInfo]fail, dfxOpInfo is nullptr");
        return;
    }
    isStaticGraphMode_ = IsStaticGraphMode(dfxOpInfo->op_);
    opMode_ = dfxOpInfo->op_.opMode;
    currDfxOpInfo_ = std::move(dfxOpInfo);
    HCCL_INFO(
        "[MirrorTaskManager][SetCurrDfxOpInfo] Succeed, currDfxOpInfo_[%p], this[%p] !", currDfxOpInfo_.get(), this);
    return;
}

std::shared_ptr<DfxOpInfo> MirrorTaskManager::GetCurrDfxOpInfo() const { return currDfxOpInfo_; }

TaskInfoQueue* MirrorTaskManager::GetQueue(u32 streamId) const
{
    auto it = streamQueues_.find(streamId);
    if (it == streamQueues_.end()) {
        THROW<InternalException>(StringFormat("MirrorTaskManager::GetQueue streamId(sqId)[%u] out of range", streamId));
    }
    return it->second.queue;
}

std::unordered_map<u32, MirrorStreamQueueEntry>::iterator MirrorTaskManager::Begin() { return streamQueues_.begin(); }

std::unordered_map<u32, MirrorStreamQueueEntry>::iterator MirrorTaskManager::End() { return streamQueues_.end(); }

MirrorTaskManager::~MirrorTaskManager() {}

} // namespace Hccl
