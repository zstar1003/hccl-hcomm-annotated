/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_reporter.h"
#include "dlprof_function_v2.h"
#include "communicator_impl.h"
#include "comm_engine_utils.h"

namespace Hccl {
constexpr size_t TASK_INFO_BATCH_RESERVE_SIZE = 128;
std::array<ProfilingReporter::lastPosesMap, MAX_MODULE_DEVICE_NUM> ProfilingReporter::allLastPoses_{};
ProfilingReporter::ProfilingReporter(MirrorTaskManager* mirrorTaskMgr, ProfilingHandler* profilingHandler)
    : mirrorTaskMgr_(mirrorTaskMgr),
      profilingHandler_(profilingHandler)
{
    taskInfoBatch_.reserve(TASK_INFO_BATCH_RESERVE_SIZE);
}

ProfilingReporter::~ProfilingReporter() {}

HcclResult ProfilingReporter::Init()
{
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    if (mirrorTaskMgr_ == nullptr || profilingHandler_ == nullptr) {
        HCCL_ERROR("[ProfilingReporter][Init] mirrorTaskMgr or profilingHandler is nullptr.");
        return HCCL_E_PTR;
    }
    mirrorTaskMgr_->RegFullyCallBack([this]() {
        ReportCallBackAllTasks();
    });
    deviceLogicId_ = HrtGetDevice();
    if (deviceLogicId_ >= static_cast<s32>(MAX_MODULE_DEVICE_NUM) || deviceLogicId_ < 0) {
        HCCL_ERROR("[ProfilingReporter][Init] deviceLogicId_[%d] out of range", deviceLogicId_);
        return HCCL_E_INTERNAL;
    }
    initializedFlag_ = true;
    return HCCL_SUCCESS;
}

void ProfilingReporter::SetCurrDfxOpInfo(std::shared_ptr<DfxOpInfo> dfxOpInfo) const
{
    HCCL_INFO(
        "[ProfilingReporter][SetCurrDfxOpInfo] L1State[%d] L0State[%d]", profilingHandler_->GetHcclL1State(),
        profilingHandler_->GetHcclL0State());
    auto it = CMD_OP_TYPE_INFO_MAP.find(static_cast<HcclCMDType>(dfxOpInfo->op_.oldOpType));
    if (it == CMD_OP_TYPE_INFO_MAP.end()) {
        HCCL_WARNING("%s dfxOpInfo.opType[%u] is not supported.", __func__, dfxOpInfo->op_.oldOpType);
    } else {
        dfxOpInfo->op_.opType = it->second.first; // A3转A5
        dfxOpInfo->tag_ = it->second.second;      // A5转字符串    延后
    }

    HCCL_INFO(
        "[ProfilingReporter][SetCurrDfxOpInfo] dfxOpInfo->op_.oldOpType[%u] dfxOpInfo.opType[%u] tag_[%s]",
        dfxOpInfo->op_.oldOpType, dfxOpInfo->op_.opType, dfxOpInfo->tag_.c_str());
    dfxOpInfo->op_.reduceOp = Hccl::HcclReduceOpToReduceOp(static_cast<HcclReduceOp>(dfxOpInfo->op_.oldReduceOp));
    dfxOpInfo->op_.dataType = Hccl::HcclDataTypeToDataType(static_cast<HcclDataType>(dfxOpInfo->op_.oldDataType));
    mirrorTaskMgr_->SetCurrDfxOpInfo(dfxOpInfo);
}

void ProfilingReporter::ReportOp(uint64_t beginTime, bool cachedReq, bool opbased) const
{
    std::shared_ptr<DfxOpInfo> opInfo = mirrorTaskMgr_->GetCurrDfxOpInfo();
    if (opInfo == nullptr) {
        HCCL_WARNING("[ProfilingReporter::ReportOp] opInfo is nullptr, skip ReportOp!");
        return;
    }
    uint64_t endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    OpType opType = opInfo->op_.opType;
    bool isAiCpu = false;
    // 新老流程判断
    if (opInfo->isIndop_ == true) {
        if (opInfo->engine == COMM_ENGINE_AICPU_TS || opInfo->engine == COMM_ENGINE_AICPU) {
            HCCL_INFO(
                "[ProfilingReporter][ReportOp] ReportOp Aicpu,opInfo->engine:[%s]",
                GetEnumToString(GetCommEngineStatusStrMap(), opInfo->engine).c_str());
            isAiCpu = true;
        }
    } else {
        CommunicatorImpl* commImp = static_cast<CommunicatorImpl*>(opInfo->comm_);
        if (commImp == nullptr) {
            HCCL_WARNING("[ProfilingReporter::ReportOp] commImp is nullptr, skip ReportOp!");
            return;
        }
        isAiCpu = commImp->GetOpAiCpuTSFeatureFlag();
    }
    // 上报op信息
    opInfo->endTime_ = endTime;
    profilingHandler_->ReportHcclOp(*opInfo, cachedReq);

    // 单算子模式涉及HOST API信息上报 注意这个地方
    if (opbased) {
        profilingHandler_->ReportHostApi(opType, beginTime, endTime, cachedReq, isAiCpu);
    }
}

void ProfilingReporter::ReportAllTasksLog() const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    auto& curLastPoses = allLastPoses_[deviceLogicId_];
    for (auto it = mirrorTaskMgr_->Begin(); it != mirrorTaskMgr_->End(); ++it) {
        u32 streamId = it->first;
        Queue<std::unique_ptr<TaskInfo>>* currQueue = it->second.queue;
        if (currQueue == nullptr) {
            continue;
        }
        if (**(currQueue->Begin()) == nullptr) {
            continue;
        }
        if (curLastPoses.find(streamId) == curLastPoses.end() && currQueue->Begin() != nullptr) {
            TaskInfo* task = (*currQueue->Begin())->get();
            HCCL_INFO("[ProfilingReporter] ReportAllTasksLog, %s", task->Describe().c_str());
        }
        if (curLastPoses.find(streamId) == curLastPoses.end()) {
            continue;
        }
        bool pastLastPos = false;
        auto logIter = currQueue->Begin();
        for (; *logIter != *currQueue->End(); ++(*logIter)) {
            if (!pastLastPos && *logIter == *curLastPoses[streamId]) {
                pastLastPos = true;
                continue;
            }
            if (pastLastPos) {
                TaskInfo* task = (*logIter)->get();
                HCCL_INFO("[ProfilingReporter] ReportAllTasksLog, %s", task->Describe().c_str());
            }
        }
    }
}

void ProfilingReporter::ReportCallBackAllTasks(bool cachedReq) { ReportAllTasks(cachedReq); }

void ProfilingReporter::ReportAllTasks(bool cachedReq)
{
    std::lock_guard<std::mutex> lock(mirrorTaskMgr_->GetTaskMutex());
    ReportAllTasksLog();
    auto& curLastPoses = allLastPoses_[deviceLogicId_];
    taskInfoBatch_.clear();
    for (auto it = mirrorTaskMgr_->Begin(); it != mirrorTaskMgr_->End(); ++it) {
        u32 streamId = it->first;
        Queue<std::unique_ptr<TaskInfo>>* currQueue = it->second.queue;
        if (currQueue == nullptr || currQueue->Begin() == nullptr || currQueue->Tail() == nullptr) {
            HCCL_WARNING("[ProfilingReporter][ReportAllTasks] currQueue is nullptr, continue to next task.");
            continue;
        }
        if (*(*(currQueue->Begin())) == nullptr) {
            HCCL_WARNING(
                "[ProfilingReporter][ReportAllTasks] (*(*(currQueue->Begin())) is nullptr, continue to next task.");
            continue;
        }
        if (curLastPoses.find(streamId) == curLastPoses.end() && currQueue->Begin() != nullptr) {
            TaskInfo* task = (*currQueue->Begin())->get();
            profilingHandler_->ReportHcclTaskApi(
                task->taskParam_.taskType, task->taskParam_.beginTime, task->taskParam_.endTime, task->isMaster_,
                cachedReq, true);
            taskInfoBatch_.emplace_back(task);
            curLastPoses[streamId] = currQueue->Begin();
        }

        auto endPos = currQueue->Tail();
        auto iter = curLastPoses[streamId];
        ++(*(iter));
        for (; (*(iter)) != (*(currQueue->End())); ++(*(iter))) {
            TaskInfo* task = (*iter)->get();
            profilingHandler_->ReportHcclTaskApi(
                task->taskParam_.taskType, task->taskParam_.beginTime, task->taskParam_.endTime, task->isMaster_,
                cachedReq, true);
            taskInfoBatch_.emplace_back(task);
        }
        curLastPoses[streamId] = endPos;
    }
    if (!taskInfoBatch_.empty()) {
        profilingHandler_->ReportHcclTaskDetailsBatch(taskInfoBatch_, cachedReq);
    }
}

/* 中途打开profiling开关 */
void ProfilingReporter::UpdateProfStat(void)
{
    if (enableHcclL1_ == true) {
        return;
    }
    // 读取L1开关状态，更新reporter中的开关；
    bool newEnableHcclL1 = profilingHandler_->GetHcclL1State();
    if (enableHcclL1_ != newEnableHcclL1) {
        enableHcclL1_ = newEnableHcclL1;
        auto& curLastPoses = allLastPoses_[deviceLogicId_];
        for (auto it = mirrorTaskMgr_->Begin(); it != mirrorTaskMgr_->End(); ++it) {
            u32 streamId = it->first;
            if (it->second.queue == nullptr) {
                continue;
            }
            curLastPoses[streamId] = it->second.queue->Tail();
        }
    }
}

void ProfilingReporter::CallReportMc2CommInfo(
    const Stream& kfcStream, const Stream& stream, const std::vector<Stream*>& aicpuStreams, const std::string& id,
    RankId myRank, u32 rankSize, RankId rankInParentComm) const
{
    profilingHandler_->ReportHcclMC2CommInfo(kfcStream, stream, aicpuStreams, id, myRank, rankSize, rankInParentComm);
}

void ProfilingReporter::CallReportMc2CommInfo(
    const u32 kfcStreamId, const std::vector<u32>& aicpuStreamsId, const std::string& id, RankId myRank, u32 rankSize,
    RankId rankInParentComm) const
{
    profilingHandler_->ReportHcclMC2CommInfo(kfcStreamId, aicpuStreamsId, id, myRank, rankSize, rankInParentComm);
}

} // namespace Hccl
