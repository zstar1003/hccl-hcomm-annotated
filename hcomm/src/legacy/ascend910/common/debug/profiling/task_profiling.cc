/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include "profiling_manager.h"
#include "adapter_prof.h"
#include "task_profiling.h"

namespace hccl {
std::mutex TaskProfiling::mutex_;

TaskProfiling::TaskProfiling(u32 deviceLogicId_, u32 localRank_, u32 rankSize_, bool profilingOn)
    : ProfilerBase(deviceLogicId_),
      localRank_(localRank_),
      rankSize_(rankSize_),
      profilingOn_(profilingOn)
{}

TaskProfiling::~TaskProfiling() {}

u64 TaskProfiling::TimestampNanosecond() const
{
    // 此时间戳获取方式需要与runtime保持一致
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<u64>(ts.tv_sec * MULTIPLIER_S2NS + ts.tv_nsec);
}

double TaskProfiling::GetTaskTime(TaskType taskType, const TaskData& taskData) const
{
    double estimatedUs = DURATION_INIT_VALUE;
    double fixedUs = DURATION_INIT_VALUE;

    switch (taskType) {
        case TaskType::TASK_SDMA:
            /* * 统一按照PCIe的带宽来计算, SDMA的固定开销按照0.6us(<512KB), 1.5us(>512KB)计算
                PCIe DMA实测带宽为19.3GB */
            fixedUs = (taskData.DMA.size > DURATION_SDMA_FIXED_THRESHOLD) ? DURATION_SDMA_FIXED_THRESHOLD_ABOVE :
                                                                            DURATION_SDMA_FIXED_THRESHOLD_BELOW;
            estimatedUs = (taskData.DMA.size / DURATION_SDMA_BANDWIDTH_MB) + fixedUs;
            break;

        case TaskType::TASK_REDUCE_INLINE:
            /* * 统一按照PCIe的带宽来计算, SDMA的固定开销按照0.6us(<512KB), 1.5us(>512KB)计算
                PCIe DMA实测带宽为19.3GB */
            fixedUs = (taskData.DMA.size > DURATION_SDMA_FIXED_THRESHOLD) ? DURATION_SDMA_FIXED_THRESHOLD_ABOVE :
                                                                            DURATION_SDMA_FIXED_THRESHOLD_BELOW;
            estimatedUs = (taskData.Reduce.size / DURATION_SDMA_BANDWIDTH_MB) + fixedUs;
            break;

        case TaskType::TASK_RDMA:
            /* * RDMA的固定开销按照7us计算(实测7us)
                RDMA实测大包单流带宽为 12.5 GB(刨去协议头, 取12GB) */
            estimatedUs = (taskData.DMA.size / DURATION_RDMA_BANDWIDTH_MB) + DURATION_RDMA_FIXED;
            break;

        case TaskType::TASK_REDUCE_TBE:
            // 统一按照10GB计算, CCE reduce的固定开销按照0.6us计算(理论值0.5 + dim * 8)
            estimatedUs = (taskData.Reduce.size / DURATION_CCE_BANDWIDTH_MB) + DURATION_CCE_FIXED;
            break;

        case TaskType::TASK_NOTIFY_RECORD:
            // 统一按1us算(片间1us, 片内0.5us)
            estimatedUs = DURATION_NOTIFY_RECORD;
            break;

        case TaskType::TASK_NOTIFY_WAIT:
            // Notify Wait按0.02us估计
            estimatedUs = DURATION_NOTIFY_WAIT;
            break;

        default:
            break;
    }

    return estimatedUs;
}

ProfTaskType TaskProfiling::GetProfTaskType(TaskType taskType) const
{
    ProfTaskType type = ProfTaskType::TASK_INVALID;
    switch (taskType) {
        case TaskType::TASK_SDMA:
        case TaskType::TASK_RDMA:
            type = (taskType == TaskType::TASK_SDMA) ? (ProfTaskType::TASK_SDMA) : (ProfTaskType::TASK_RDMA);
            break;
        case TaskType::TASK_REDUCE_INLINE:
        case TaskType::TASK_REDUCE_TBE:
            type = (taskType == TaskType::TASK_REDUCE_INLINE) ? (ProfTaskType::TASK_REDUCE_INLINE) :
                                                                (ProfTaskType::TASK_REDUCE_TBE);
            break;
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            type = (taskType == TaskType::TASK_NOTIFY_RECORD) ? (ProfTaskType::TASK_NOTIFY_RECORD) :
                                                                (ProfTaskType::TASK_NOTIFY_WAIT);
            break;
        default:
            break;
    }

    return type;
}

uint32_t TaskProfiling::GetTransportType(TaskType taskType) const
{
    if (taskType == TaskType::TASK_SDMA || taskType == TaskType::TASK_REDUCE_INLINE
        || taskType == TaskType::TASK_NOTIFY_RECORD) {
        return static_cast<int32_t>(SimpleTaskType::SDMA);
    } else if (taskType == TaskType::TASK_RDMA) {
        return static_cast<int32_t>(SimpleTaskType::RDMA);
    } else {
        return static_cast<int32_t>(SimpleTaskType::LOCAL);
    }
}

uint32_t TaskProfiling::GetTaskRole(TaskType taskType) const
{
    if (taskType == TaskType::TASK_SDMA || taskType == TaskType::TASK_REDUCE_INLINE
        || taskType == TaskType::TASK_REDUCE_TBE || taskType == TaskType::TASK_NOTIFY_WAIT) {
        return static_cast<uint32_t>(TaskRole::DST);
    } else {
        return static_cast<uint32_t>(TaskRole::SRC);
    }
}

void TaskProfiling::GetTaskData(TaskType taskType, const TaskData& taskData, struct MsprofHcclInfo& taskInfo)
{
    switch (taskType) {
        case TaskType::TASK_SDMA:
            GetSdmaTaskData(taskType, taskData, taskInfo);
            break;
        case TaskType::TASK_RDMA:
            GetRdmaTaskData(taskType, taskData, taskInfo);
            break;
        case TaskType::TASK_REDUCE_INLINE:
        case TaskType::TASK_REDUCE_TBE:
            GetReduceTaskData(taskType, taskData, taskInfo);
            break;
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            GetNotifyTaskData(taskType, taskData, taskInfo);
            break;
        default:
            break;
    }
}

void TaskProfiling::GetSdmaTaskData(TaskType taskType, const TaskData& taskData, struct MsprofHcclInfo& taskInfo) const
{
    taskInfo.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskData.DMA.src));
    taskInfo.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskData.DMA.dst));
    taskInfo.dataSize = static_cast<u64>(taskData.DMA.size);
    taskInfo.notifyID = taskData.DMA.notifyID;
    taskInfo.linkType = static_cast<u32>(taskData.DMA.linkType);
    taskInfo.remoteRank
        = (taskData.DMA.remoteUserRank == INVALID_VALUE_RANKID) ? localRank_ : taskData.DMA.remoteUserRank;
    taskInfo.transportType = GetTransportType(taskType);
    taskInfo.role = GetTaskRole(taskType);
    taskInfo.durationEstimated = GetTaskTime(taskData.taskType, taskData);
    taskInfo.ctxId = taskData.DMA.ctxId;
}

void TaskProfiling::GetRdmaTaskData(TaskType taskType, const TaskData& taskData, struct MsprofHcclInfo& taskInfo) const
{
    taskInfo.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskData.DMA.src));
    taskInfo.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskData.DMA.dst));
    taskInfo.dataSize = static_cast<u64>(taskData.DMA.size);
    taskInfo.notifyID = taskData.DMA.notifyID;
    taskInfo.linkType = static_cast<u32>(taskData.DMA.linkType);
    taskInfo.remoteRank
        = (taskData.DMA.remoteUserRank == INVALID_VALUE_RANKID) ? localRank_ : taskData.DMA.remoteUserRank;
    taskInfo.transportType = GetTransportType(taskType);
    taskInfo.role = GetTaskRole(taskType);
    taskInfo.rdmaType = static_cast<u32>(taskData.DMA.rdmaType);
    taskInfo.durationEstimated = GetTaskTime(taskData.taskType, taskData);
    taskInfo.ctxId = taskData.DMA.ctxId;
}

void TaskProfiling::GetReduceTaskData(
    TaskType taskType, const TaskData& taskData, struct MsprofHcclInfo& taskInfo) const
{
    taskInfo.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskData.DMA.src));
    taskInfo.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskData.DMA.dst));
    taskInfo.dataSize = static_cast<u64>(taskData.Reduce.size);
    taskInfo.opType = opString[taskData.Reduce.op];
    taskInfo.dataType = dataTypeString[taskData.Reduce.dataType];
    taskInfo.linkType = static_cast<u32>(taskData.Reduce.linkType);
    taskInfo.remoteRank = taskData.Reduce.remoteUserRank;
    taskInfo.transportType = GetTransportType(taskType);
    taskInfo.role = GetTaskRole(taskType);
    taskInfo.durationEstimated = GetTaskTime(taskData.taskType, taskData);
    taskInfo.ctxId = taskData.Reduce.ctxId;
}

void TaskProfiling::GetNotifyTaskData(
    TaskType taskType, const TaskData& taskData, struct MsprofHcclInfo& taskInfo) const
{
    taskInfo.notifyID = taskData.Notify.notifyID;
    taskInfo.stage = taskData.Notify.stage;
    taskInfo.remoteRank = taskData.Notify.remoteUserRank;
    taskInfo.transportType = GetTransportType(taskType);
    taskInfo.role = GetTaskRole(taskType);
    taskInfo.durationEstimated = GetTaskTime(taskData.taskType, taskData);
    taskInfo.ctxId = taskData.Notify.ctxId;
}

HcclResult TaskProfiling::Run([[maybe_unused]] const StepData& stepData) { return HCCL_SUCCESS; }

HcclResult TaskProfiling::Run(const TaskData& taskData, bool isCapture)
{
    HCCLReportData hcclReportData{};
    auto& profilingManager = hccl::ProfilingManager::Instance();
    bool isSubscribed = profilingManager.GetAdditionInfoState();
    if (!isSubscribed && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isCapture) {
        return HCCL_SUCCESS;
    }
    std::unique_lock<std::mutex> lock(mutex_);

    std::unique_lock<std::mutex> streamLock(streamMutex_[deviceLogicId_]);
    if (streamRecordInfoMap_[deviceLogicId_].find(taskData.streamID) == streamRecordInfoMap_[deviceLogicId_].end()) {
        // 找不到对应的tag则认为该stream不参与profiling, 返回SUCCESS
        HCCL_DEBUG("streamID[%u] not found in profiler", taskData.streamID);
        hcclReportData.tag = "unknown";
        hcclReportData.profInfo.planeID = 0;
        hcclReportData.groupName = "unknown";
        hcclReportData.profInfo.workFlowMode = 0;
    } else {
        StreamRecordInfo& streamInfo = streamRecordInfoMap_[deviceLogicId_][taskData.streamID];
        hcclReportData.tag = streamInfo.tag;
        hcclReportData.profInfo.planeID = streamInfo.planeId;
        hcclReportData.groupName = tagGroupMap_[deviceLogicId_][hcclReportData.tag];
        hcclReportData.profInfo.workFlowMode = static_cast<uint32_t>(tagModeMap_[deviceLogicId_][hcclReportData.tag]);
    }
    streamLock.unlock();

    hcclReportData.ts = hrtMsprofSysCycleTime();
    ProfTaskType type = GetProfTaskType(taskData.taskType);
    hcclReportData.type = static_cast<int32_t>(type);
    std::string nameInfo = GetProfTaskOpName(type);
    hcclReportData.profInfo.itemId = hrtMsprofGetHashId(nameInfo.c_str(), nameInfo.length());
    hcclReportData.profInfo.localRank = localRank_;
    hcclReportData.profInfo.rankSize = rankSize_;
    GetTaskData(taskData.taskType, taskData, hcclReportData.profInfo);

    hcclReportData.profInfo.cclTag = hrtMsprofGetHashId(hcclReportData.tag.c_str(), hcclReportData.tag.length());
    hcclReportData.profInfo.groupName
        = hrtMsprofGetHashId(hcclReportData.groupName.c_str(), hcclReportData.groupName.length());

    HCCL_DEBUG(
        "ReportMsprofData:streamID[%u] tag[%s][%llu] group[%s][%llu]", taskData.streamID, hcclReportData.tag.c_str(),
        hcclReportData.profInfo.cclTag, hcclReportData.groupName.c_str(), hcclReportData.profInfo.groupName);

    DumpReportDataInfo(hcclReportData.type, hcclReportData.profInfo);
    CHK_RET(ReportMsprofData(hcclReportData));
    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::Run([[maybe_unused]] const std::string& opName, [[maybe_unused]] const std::string& tag) const
{
    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::Save(
    u32 captureStreamID, [[maybe_unused]] u32 streamID, u32 taskID, TaskType& taskType,
    const TaskParaReduce& paraReduce)
{
    TaskData taskData(captureStreamID, taskID, taskType, paraReduce);
    Run(taskData, true);
    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::Save(u32& streamID, u32& taskID, TaskType& taskType, const TaskParaReduce& paraReduce)
{
    TaskData taskData(streamID, taskID, taskType, paraReduce);
    Run(taskData);
    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::Save(
    u32 captureStreamID, [[maybe_unused]] u32 streamID, u32 taskID, TaskType& taskType, const TaskParaDMA& paraDMA)
{
    TaskData taskData(captureStreamID, taskID, taskType, paraDMA);
    Run(taskData, true);
    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::Save(u32& streamID, u32& taskID, TaskType& taskType, const TaskParaDMA& paraDMA)
{
    TaskData taskData(streamID, taskID, taskType, paraDMA);
    Run(taskData);
    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::Save(
    u32 captureStreamID, [[maybe_unused]] u32 streamID, u32 taskID, TaskType& taskType,
    const TaskParaNotify& paraNotify)
{
    TaskData taskData(captureStreamID, taskID, taskType, paraNotify);
    Run(taskData, true);
    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::Save(u32& streamID, u32& taskID, TaskType& taskType, const TaskParaNotify& paraNotify)
{
    TaskData taskData(streamID, taskID, taskType, paraNotify);
    Run(taskData);
    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::Save(
    [[maybe_unused]] u32 captureStreamID, [[maybe_unused]] u32 streamID, [[maybe_unused]] u32 taskID,
    [[maybe_unused]] const void* descBuf, [[maybe_unused]] size_t descBufLen)
{
    return HCCL_SUCCESS;
}

HcclResult
TaskProfiling::Save(u32 captureStreamID, u32 streamID, [[maybe_unused]] u32 taskID, const TaskParaAiv& paraAiv)
{
    HCCLReportData hcclReportData{};
    auto& profilingManager = hccl::ProfilingManager::Instance();
    bool isSubscribed = profilingManager.GetAdditionInfoState();
    bool isCapture = (captureStreamID != streamID);
    if (!isSubscribed && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isCapture) {
        return HCCL_SUCCESS;
    }
    std::unique_lock<std::mutex> lock(mutex_);

    std::unique_lock<std::mutex> streamLock(streamMutex_[deviceLogicId_]);

    if (streamRecordInfoMap_[deviceLogicId_].find(captureStreamID) == streamRecordInfoMap_[deviceLogicId_].end()) {
        // 找不到对应的tag则认为该stream不参与profiling, 返回SUCCESS
        HCCL_DEBUG("streamID[%u] not found in profiler", captureStreamID);
        hcclReportData.tag = "unknown";
        hcclReportData.profInfo.planeID = 0;
        hcclReportData.groupName = "unknown";
        hcclReportData.profInfo.workFlowMode = 0;
    } else {
        StreamRecordInfo& streamInfo = streamRecordInfoMap_[deviceLogicId_][captureStreamID];
        hcclReportData.tag = streamInfo.tag;
        hcclReportData.profInfo.planeID = streamInfo.planeId;
        hcclReportData.groupName = tagGroupMap_[deviceLogicId_][hcclReportData.tag];
        hcclReportData.profInfo.workFlowMode = static_cast<uint32_t>(tagModeMap_[deviceLogicId_][hcclReportData.tag]);
    }
    streamLock.unlock();

    hcclReportData.ts = hrtMsprofSysCycleTime();

    hcclReportData.type = static_cast<int32_t>(ProfTaskType::TASK_AIV);
    std::string nameInfo = GetCMDTypeEnumStr(paraAiv.cmdType) + "AivKernel";
    hcclReportData.profInfo.itemId = hrtMsprofGetHashId(nameInfo.c_str(), nameInfo.length());
    hcclReportData.profInfo.localRank = localRank_;
    hcclReportData.profInfo.rankSize = rankSize_;

    hcclReportData.profInfo.dataSize = paraAiv.size;
    hcclReportData.profInfo.cclTag = hrtMsprofGetHashId(hcclReportData.tag.c_str(), hcclReportData.tag.length());
    hcclReportData.profInfo.groupName
        = hrtMsprofGetHashId(hcclReportData.groupName.c_str(), hcclReportData.groupName.length());

    HCCL_DEBUG(
        "ReportMsprofData:streamID[%u] tag[%s][%llu] group[%s][%llu]", captureStreamID, hcclReportData.tag.c_str(),
        hcclReportData.profInfo.cclTag, hcclReportData.groupName.c_str(), hcclReportData.profInfo.groupName);

    DumpReportDataInfo(hcclReportData.type, hcclReportData.profInfo);
    CHK_RET(ReportMsprofData(hcclReportData));
    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::Save(u32 streamID, u32 taskID, const TaskParaAiv& paraAiv)
{
    return Save(streamID, streamID, taskID, paraAiv);
}

HcclResult TaskProfiling::Save(
    [[maybe_unused]] u32& streamID, [[maybe_unused]] u32& taskID, [[maybe_unused]] const void* descBuf,
    [[maybe_unused]] size_t descBufLen)
{
    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::SaveToLog(const TaskParaHost& paraHost)
{
    auto& profilingManager = hccl::ProfilingManager::Instance();
    bool isSubscribed = profilingManager.GetAdditionInfoState();
    if (!isSubscribed) {
        return HCCL_SUCCESS;
    }

    u32 streamID = paraHost.streamID;
    u32 taskID = paraHost.taskID;
    u64 len = paraHost.len;
    std::string tag = paraHost.tag;
    std::chrono::microseconds duration = paraHost.duration;

    double bandWidth = 0;
    if (duration.count() > 0) {
        double ratio = 1.0 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
        bandWidth = len / (duration.count() * ratio);
        double BytetoGB = 1 << 30;
        bandWidth /= BytetoGB;
    }

    HCCL_INFO(
        "[profiling][host TCP/RDMA] tag[%s], streamId[%u], taskId[%u], bandWidth[%f GB / s]", tag.c_str(), streamID,
        taskID, bandWidth);

    return HCCL_SUCCESS;
}

void TaskProfiling::DumpReportDataInfo(uint32_t type, const MsprofHcclInfo& profInfo)
{
    HCCL_DEBUG(
        "[DumpReportDataInfo] type[%u], itemId[%llu], cclTag[%llu], groupName[%llu], localRank[%u], remoteRank[%u], "
        "rankSize[%u], workFlowMode[%u], planeID[%u], ctxId[%u], notifyID[%llu], stage[%u], role[%u], "
        "durationEstimated[%f], srcAddr[%llu], dstAddr[%llu], dataSize[%llu Byte], opType[%u], dataType[%u], "
        "linkType[%u], "
        "transportType[%u], rdmaType[%u]",
        type, profInfo.itemId, profInfo.cclTag, profInfo.groupName, profInfo.localRank, profInfo.remoteRank,
        profInfo.rankSize, profInfo.workFlowMode, profInfo.planeID, profInfo.ctxId, profInfo.notifyID, profInfo.stage,
        profInfo.role, profInfo.durationEstimated, profInfo.srcAddr, profInfo.dstAddr, profInfo.dataSize,
        profInfo.opType, profInfo.dataType, profInfo.linkType, profInfo.transportType, profInfo.rdmaType);

    return;
}

HcclResult TaskProfiling::ReportMsprofData(HCCLReportData& hcclReportData)
{
    int32_t cb_ret;
    auto& profilingManager = hccl::ProfilingManager::Instance();
    cb_ret = profilingManager.CallMsprofReportAdditionInfo(
        static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO), hcclReportData.ts, &hcclReportData.profInfo,
        sizeof(hcclReportData.profInfo));
    CHK_PRT_RET((cb_ret != 0), HCCL_ERROR("[TaskProfiling] ReportData profiling failed."), HCCL_E_PARA);

    return HCCL_SUCCESS;
}

HcclResult TaskProfiling::Flush() { return HCCL_SUCCESS; }
} // namespace hccl
