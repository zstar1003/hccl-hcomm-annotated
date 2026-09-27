/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_overflow.h"
#include "externalinput_pub.h"

namespace hccl {

constexpr u32 DUMPTASK_COUNT_UPPER_LIMIT = 1000; // 算子溢出的task最大值为1000，防止内存占用量过大
constexpr u32 SUBTASK_DEFAULT_VALUE = 10;        // subTaskType默认参数
constexpr u32 REDUCE_INLINE = 0;
constexpr u32 REDUCE_TBE = 1;

TaskOverflow::TaskOverflow(u32 deviceLogicId) : ProfilerBase(deviceLogicId) {}
TaskOverflow::~TaskOverflow() {}

uint32_t TaskOverflow::GetTaskName(TaskType taskType) const
{
    uint32_t taskNameId = SUBTASK_DEFAULT_VALUE;
    switch (taskType) {
        case TaskType::TASK_SDMA:
        case TaskType::TASK_RDMA:
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            break;
        case TaskType::TASK_REDUCE_INLINE:
            taskNameId = REDUCE_INLINE;
            break;
        case TaskType::TASK_REDUCE_TBE:
            taskNameId = REDUCE_TBE;
            break;
        default:
            break;
    }
    return taskNameId;
}

HcclResult TaskOverflow::Save(
    [[maybe_unused]] u32 captureStreamID, u32 streamID, u32 taskID, TaskType& taskType,
    const TaskParaReduce& paraReduce)
{
    if (GetExternalInputHcclDumpDebug()) {
        HCCL_DEBUG("HcclDumpInfo save start");
        HcclDumpInfo hcclDumpInfo{};
        hcclDumpInfo.task_id = taskID;
        hcclDumpInfo.stream_id = streamID;
        hcclDumpInfo.output_addr = const_cast<void*>(paraReduce.dst);
        hcclDumpInfo.output_size = static_cast<u64>(paraReduce.size);
        hcclDumpInfo.input_addr = const_cast<void*>(paraReduce.src);
        hcclDumpInfo.input_size = static_cast<u64>(paraReduce.size);
        hcclDumpInfo.sub_task_type = GetTaskName(taskType);

        std::unique_lock<std::mutex> lock(dumpInfoVetcorMutex_);
        // 防止上层未及时清理dumpInfoVetcor_信息，导致内存占满。
        if (dumpInfoVetcor_.size() < DUMPTASK_COUNT_UPPER_LIMIT) {
            dumpInfoVetcor_.push_back(hcclDumpInfo);
            return HCCL_SUCCESS;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TaskOverflow::Save(u32& streamID, u32& taskID, TaskType& taskType, const TaskParaReduce& paraReduce)
{
    return Save(streamID, streamID, taskID, taskType, paraReduce);
}

HcclResult TaskOverflow::Save(
    [[maybe_unused]] u32 captureStreamID, [[maybe_unused]] u32 streamID, [[maybe_unused]] u32 taskID,
    [[maybe_unused]] TaskType& taskType, [[maybe_unused]] const TaskParaDMA& paraDMA)
{
    return HCCL_SUCCESS;
}

HcclResult TaskOverflow::Save(
    [[maybe_unused]] u32& streamID, [[maybe_unused]] u32& taskID, [[maybe_unused]] TaskType& taskType,
    [[maybe_unused]] const TaskParaDMA& paraDMA)
{
    return HCCL_SUCCESS;
}

HcclResult TaskOverflow::Save(
    [[maybe_unused]] u32 captureStreamID, [[maybe_unused]] u32 streamID, [[maybe_unused]] u32 taskID,
    [[maybe_unused]] TaskType& taskType, [[maybe_unused]] const TaskParaNotify& paraNotify)
{
    return HCCL_SUCCESS;
}

HcclResult TaskOverflow::Save(
    [[maybe_unused]] u32& streamID, [[maybe_unused]] u32& taskID, [[maybe_unused]] TaskType& taskType,
    [[maybe_unused]] const TaskParaNotify& paraNotify)
{
    return HCCL_SUCCESS;
}

HcclResult TaskOverflow::Save(
    [[maybe_unused]] u32 captureStreamID, [[maybe_unused]] u32 streamID, [[maybe_unused]] u32 taskID,
    [[maybe_unused]] const void* descBuf, [[maybe_unused]] size_t descBufLen)
{
    return HCCL_SUCCESS;
}

HcclResult TaskOverflow::Save(
    [[maybe_unused]] u32 captureStreamID, [[maybe_unused]] u32 streamID, [[maybe_unused]] u32 taskID,
    [[maybe_unused]] const TaskParaAiv& paraAiv)
{
    return HCCL_SUCCESS;
}

HcclResult TaskOverflow::Save(
    [[maybe_unused]] u32 streamID, [[maybe_unused]] u32 taskID, [[maybe_unused]] const TaskParaAiv& paraAiv)
{
    return HCCL_SUCCESS;
}

HcclResult TaskOverflow::Save(
    [[maybe_unused]] u32& streamID, [[maybe_unused]] u32& taskID, [[maybe_unused]] const void* descBuf,
    [[maybe_unused]] size_t descBufLen)
{
    return HCCL_SUCCESS;
}

HcclResult TaskOverflow::GetandClearOverFlowTasks(std::vector<HcclDumpInfo>& hcclDumpInfoVector)
{
    hcclDumpInfoVector.assign(dumpInfoVetcor_.begin(), dumpInfoVetcor_.end());
    dumpInfoVetcor_.clear();

    return HCCL_SUCCESS;
}

HcclResult TaskOverflow::Flush() { return HCCL_SUCCESS; }

HcclResult TaskOverflow::Run([[maybe_unused]] const StepData& stepData) { return HCCL_SUCCESS; }

HcclResult TaskOverflow::SaveToLog([[maybe_unused]] const TaskParaHost& paraHost) { return HCCL_SUCCESS; }
} // namespace hccl
