/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "global_mirror_tasks.h"
#include <stdexcept>

namespace Hccl {

GlobalMirrorTasks GlobalMirrorTasks::ins_;

GlobalMirrorTasks::GlobalMirrorTasks() {}

GlobalMirrorTasks::~GlobalMirrorTasks() {}

GlobalMirrorTasks& GlobalMirrorTasks::Instance() { return ins_; }

u32 GlobalMirrorTasks::DevSize() const { return DEVICE_MAX_NUM; }

TaskInfoQueue* GlobalMirrorTasks::GetQueue(u32 devId, u32 streamId) const
{
    if (devId >= DEVICE_MAX_NUM) {
        HCCL_ERROR("GlobalMirrorTasks::GetQueue devId[%u] out of range", devId);
        THROW<InternalException>(StringFormat("GlobalMirrorTasks::GetQueue devId[%u] out of range", devId));
    }

    auto& devMap = taskMaps_[devId];
    auto streamIterator = devMap.find(streamId);
    if (streamIterator == devMap.end()) {
        HCCL_ERROR("GlobalMirrorTasks::GetQueue devId[%u], streamId(sqId)[%u] not found", devId, streamId);
        THROW<InternalException>(
            StringFormat("GlobalMirrorTasks::GetQueue devId[%u], streamId(sqId)[%u] not found", devId, streamId));
    }

    HCCL_INFO("[GlobalMirrorTasks][GetQueue]find devId[%u], streamId(sqId)[%u]", devId, streamId);

    return streamIterator->second.get();
}

TaskInfoQueue& GlobalMirrorTasks::CreateQueue(u32 devId, u32 streamId, QueueType type)
{
    if (devId >= DEVICE_MAX_NUM) {
        THROW<InternalException>(StringFormat(
            "GlobalMirrorTasks::CreateQueue devId[%u] out of range, streamId(sqId)[%u] ", devId, streamId));
    }

    auto& devMap = taskMaps_[devId];
    auto streamIterator = devMap.find(streamId);
    if (streamIterator != devMap.end()) {
        return *(streamIterator->second.get());
    }

    std::unique_ptr<TaskInfoQueue> newQueue;
    if (type == QueueType::Circular_Queue) {
        newQueue = std::make_unique<CircularQueue<std::unique_ptr<TaskInfo>>>(MAX_CIRCULAR_QUEUE_LENGTH);
        HCCL_INFO(
            "[GlobalMirrorTasks][CreateQueue]Create circular queue, devId[%u] streamId(sqId)[%u]", devId, streamId);
    } else {
        newQueue = std::make_unique<VectorQueue<std::unique_ptr<TaskInfo>>>();
        HCCL_INFO("[GlobalMirrorTasks][CreateQueue]Create vector queue, devId[%u] streamId(sqId)[%u]", devId, streamId);
    }

    devMap[streamId] = std::move(newQueue);

    return *devMap[streamId].get();
}

void GlobalMirrorTasks::DestroyQueue(u32 devId, u32 streamId)
{
    if (devId >= DEVICE_MAX_NUM) {
        THROW<InternalException>(StringFormat(
            "GlobalMirrorTasks::DestroyQueue devId[%u] out of range, streamId(sqId)[%u]", devId, streamId));
        return;
    }
    taskMaps_[devId].erase(streamId);
}

TaskInfo* GlobalMirrorTasks::GetTaskInfo(u32 devId, u32 streamId, u32 taskId) const
{
    TaskInfoQueue* queue = nullptr;
    try {
        queue = GetQueue(devId, streamId);
    } catch (HcclException& e) {
        return nullptr;
    }

    auto FindTask = [taskId](const std::unique_ptr<TaskInfo>& taskInfo) {
        return taskInfo->taskId_ == taskId;
    };

    auto task = queue->Find(FindTask);
    if (*task == *queue->End()) {
        return nullptr;
    };

    HCCL_INFO(
        "[GlobalMirrorTasks][GetTaskInfo]find devId[%u], streamId(sqId)[%u] taskId(sqeId)[%u]", devId, streamId,
        taskId);

    return (*(*task)).get();
}

TaskInfoQueueMap::iterator GlobalMirrorTasks::Begin(u32 devId)
{
    if (devId >= DEVICE_MAX_NUM) {
        THROW<InternalException>(StringFormat("GlobalMirrorTasks::Begin devId[%u] out of range", devId));
    }
    auto& devMap = taskMaps_[devId];
    return devMap.begin();
}

TaskInfoQueueMap::iterator GlobalMirrorTasks::End(u32 devId)
{
    if (devId >= DEVICE_MAX_NUM) {
        THROW<InternalException>(StringFormat("GlobalMirrorTasks::End devId[%u] out of range", devId));
    }
    auto& devMap = taskMaps_[devId];
    return devMap.end();
}

HcclResult GlobalMirrorTasks::FindTaskInfo(u32 devId, u32 streamId, u32 taskId, TaskInfo*& curTask) const
{
    HCCL_INFO("[%s]start, devId[%u] streamId(sqId)[%u] taskId(sqeId)[%u].", __func__, devId, streamId, taskId);
    CHK_PRT_RET(devId >= DEVICE_MAX_NUM, HCCL_ERROR("[%s]fail, devId[%u] out of range.", __func__, devId), HCCL_E_PARA);

    const TaskInfoQueueMap& devMap = taskMaps_[devId];
    auto streamIterator = devMap.find(streamId);
    if (streamIterator
        == devMap.end()) { // rts回调时不会判断异常task是否HCCL task，索引不到可能是其他组件task，此处不打印ERROR日志
        HCCL_RUN_INFO("[%s]devId[%u] streamId(sqId)[%u] not hccl task.", __func__, devId, streamId);
        return HCCL_E_NOT_FOUND;
    }

    TaskInfoQueue* queue = streamIterator->second.get();
    CHK_PTR_NULL(queue);

    auto FindTask = [taskId](const std::unique_ptr<TaskInfo>& taskInfo) {
        return taskInfo->taskId_ == taskId;
    };

    auto task = queue->Find(FindTask);
    if (*task == *queue->End() || *(*task) == nullptr) {
        HCCL_RUN_INFO(
            "[%s]devId[%u] streamId(sqId)[%u] taskId(sqeId)[%u] not hccl task.", __func__, devId, streamId, taskId);
        return HCCL_E_NOT_FOUND;
    };

    curTask = (*(*task)).get();
    HCCL_INFO("[%s]success, devId[%u] streamId(sqId)[%u] taskId(sqeId)[%u].", __func__, devId, streamId, taskId);
    return HCCL_SUCCESS;
}

} // namespace Hccl
