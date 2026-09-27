/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_handler_lite.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "sal.h"
#include "task_info.h"
#include "task_param.h"
#include "communicator_impl_lite.h"
#include "dlprof_function_v2.h"
#include <mutex>
namespace Hccl {

static constexpr u32 aging = 1;
constexpr std::uint32_t HCCLINFO_REPORT_BATCH_NUM = 2;
ProfilingHandlerLite ProfilingHandlerLite::instance_;

ProfilingHandlerLite::ProfilingHandlerLite() {}

ProfilingHandlerLite::~ProfilingHandlerLite() {}

ProfilingHandlerLite& ProfilingHandlerLite::GetInstance() { return instance_; }

void ProfilingHandlerLite::SetCachedCclTag()
{
    for (const auto& item : CMD_OP_TYPE_INFO_MAP) {
        const string& tag = item.second.second;
        if (cachedCclTag_.find(tag) == cachedCclTag_.end()) {
            const std::string& cclTag = item.second.second;
            cachedCclTag_[tag] = GetProfHashId(cclTag.c_str(), cclTag.length());
        }
    }
}

void ProfilingHandlerLite::SetCachedGroupName(const std::string& groupName, u32 rankSize)
{
    cachedGroupName_ = GetProfHashId(groupName.c_str(), groupName.length());
    cachedRankSize_ = rankSize;
}

HcclResult ProfilingHandlerLite::Init()
{
    static std::mutex initMutex;
    std::lock_guard<std::mutex> lock(initMutex); // 互斥锁，避免多线程并发调用
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    cachedTid_ = SalGetTid();
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        if (AdprofReportAdditionalInfo != nullptr) {
            reportAdditionalInfo_ = AdprofReportAdditionalInfo;
        }
        if (AdprofGetHashId != nullptr) {
            getProfHashId_ = AdprofGetHashId;
        }
        if (AdprofReportBatchAdditionalInfo != nullptr) {
            reportBatchAdditionalInfo_ = AdprofReportBatchAdditionalInfo;
        }
    } else {
        if (MsprofReportAdditionalInfo != nullptr) {
            reportAdditionalInfo_ = [](uint32_t flag, const void* data, uint32_t len) -> int32_t {
                return MsprofReportAdditionalInfo(flag, const_cast<void*>(data), len);
            };
        }
        if (MsprofStr2Id != nullptr) {
            getProfHashId_ = MsprofStr2Id;
        }
        reportBatchAdditionalInfo_ = [](uint32_t flag, const void* data, uint32_t len) -> int32_t {
            return MsprofReportBatchAdditionalInfo(flag, const_cast<void*>(data), len);
        };
    }
    if (reportAdditionalInfo_ == nullptr) {
        HCCL_ERROR("[ProfilingHandlerLite][Init] reportAdditionalInfo_ is nullptr, profiling report will be skipped");
        return HCCL_E_PROFILING;
    }
    if (getProfHashId_ == nullptr) {
        HCCL_ERROR("[ProfilingHandlerLite][Init] getProfHashId_ is nullptr, profiling hash will be invalid");
        return HCCL_E_PROFILING;
    }
    for (auto i = 0; i < static_cast<int>(TaskParamType::__COUNT__); i++) {
        TaskParamType type(static_cast<TaskParamType::Value>(i));
        std::string nameInfo = GetProfTaskOpNameV2(type);
        taskTypeHashCache_[static_cast<uint32_t>(i)] = GetProfHashId(nameInfo.c_str(), nameInfo.length());
    }
    cachedAlgTypeHashId_ = GetProfHashId("AlgType::NHR", strlen("AlgType::NHR"));
    SetCachedCclTag();
    initializedFlag_ = true;
    return HCCL_SUCCESS;
}

void ProfilingHandlerLite::ReportHcclOpInfo(const DfxOpInfo& opInfo) const
{
    if (!GetProfL0State()) {
        HCCL_INFO("[ProfilingHandlerLite][ReportHcclOpInfo] l0 is false.");
        return;
    }
    if (aicpu::GetTaskAndStreamId == nullptr) {
        HCCL_WARNING("[ProfilingHandlerLite][ReportHcclOpInfo] GetTaskAndStreamId is nullptr.");
        return;
    }
    uint64_t taskId = 0U;
    uint32_t streamId = 0;
    if (aicpu::GetTaskAndStreamId(taskId, streamId) != aicpu::status_t::AICPU_ERROR_NONE) {
        THROW<InternalException>("[ProfilingHandler] Failed to get task id and stream id.");
    }
    if (taskId > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        THROW<InvalidParamsException>("[ProfilingHandler] taskId is larger than u32.");
    }
    MsprofAdditionalInfo reporterData{};
    reporterData.level = MSPROF_REPORT_AICPU_LEVEL;
    reporterData.type = MSPROF_REPORT_AICPU_HCCL_OP_INFO;
    reporterData.threadId = cachedTid_;
    reporterData.dataLen = sizeof(MsprofAicpuHCCLOPInfo);
    reporterData.timeStamp = ProfGetCurCpuTimestamp();
    auto* hcclOpInfo = reinterpret_cast<MsprofAicpuHCCLOPInfo*>(reporterData.data);
    hcclOpInfo->algType = cachedAlgTypeHashId_;

    hcclOpInfo->taskId = static_cast<uint32_t>(taskId);
    hcclOpInfo->streamId = streamId;
    hcclOpInfo->count = opInfo.op_.dataCount;
    hcclOpInfo->dataType = opInfo.op_.dataType;
    hcclOpInfo->groupName = cachedGroupName_;
    hcclOpInfo->ranksize = cachedRankSize_;
    HCCL_INFO(
        "[ProfilingHandlerLite][ReportHcclOpInfo] relay:%d, retry:%d, dataType:%s, algType:%llu, count:%llu, "
        "groupName:%llu, ranksize:%u, taskId:%u, streamId:%u",
        hcclOpInfo->relay, hcclOpInfo->retry, DataTypeToSerialString(hcclOpInfo->dataType).c_str(),
        static_cast<unsigned long long>(hcclOpInfo->algType), static_cast<unsigned long long>(hcclOpInfo->count),
        static_cast<unsigned long long>(hcclOpInfo->groupName), hcclOpInfo->ranksize, hcclOpInfo->taskId,
        static_cast<unsigned int>(hcclOpInfo->streamId));
    ReportAdditionInfo(reporterData);
}

void ProfilingHandlerLite::ReportHcclTaskDetailsLog(const std::vector<TaskInfo*>& taskInfo) const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    for (std::vector<Hccl::TaskInfo*>::size_type i = 0; i < taskInfo.size(); i++) {
        DumpTaskDetails(MsprofAicpuHcclTaskInfo{}, taskInfo[i]);
    }
}

bool ProfilingHandlerLite::FillBatchReporterData(
    uint32_t batchId, const MsprofAicpuHcclTaskInfo* taskInfos, MsprofAdditionalInfo& addInfo) const
{
    addInfo.level = MSPROF_REPORT_AICPU_LEVEL;
    addInfo.type = MSPROF_REPORT_AICPU_MC2_BATCH_HCCL_INFO;
    addInfo.threadId = cachedTid_;
    addInfo.timeStamp = 0;
    addInfo.dataLen = sizeof(MsprofAicpuHcclTaskInfo) * batchId;
    s32 sret = memcpy_s(addInfo.data, sizeof(addInfo.data), taskInfos, addInfo.dataLen);
    if (sret != 0) {
        HCCL_WARNING("[ProfilingHandlerLite][FillBatchReporterData] memcpy failed, sret[%d]", sret);
        return false;
    }
    return true;
}

bool ProfilingHandlerLite::ReportBatchAddInfo(
    uint32_t batchId, const MsprofAicpuHcclTaskInfo* taskInfos, MsprofAdditionalInfo* addInfoVec, uint32_t& addInfoIndx,
    uint32_t maxBatchNum, bool isLastBatch) const
{
    if (!FillBatchReporterData(batchId, taskInfos, addInfoVec[addInfoIndx])) {
        return false;
    }
    addInfoIndx++;
    if (addInfoIndx == maxBatchNum || isLastBatch) {
        if (reportBatchAdditionalInfo_(aging, addInfoVec, addInfoIndx * sizeof(MsprofAdditionalInfo)) != 0) {
            HCCL_WARNING("[ProfilingHandlerLite][ReportHcclTaskDetails] reportBatchAdditionalInfo failed");
            return false;
        }
        addInfoIndx = 0;
    }
    return true;
}

void ProfilingHandlerLite::ReportHcclTaskDetails(const std::vector<TaskInfo*>& taskInfo) const
{
    ReportHcclTaskDetailsLog(taskInfo);
    MsprofAicpuHcclTaskInfo taskInfos[HCCLINFO_REPORT_BATCH_NUM] = {};
    bool isSupportBatchReport = (reportBatchAdditionalInfo_ != nullptr);
    HCCL_INFO(
        "[ProfilingHandlerLite][ReportHcclTaskDetails] isSupportBatchReport: %s, taskInfo.size: %zu",
        isSupportBatchReport ? "true" : "false", taskInfo.size());
    constexpr int32_t MAX_BATCH_REPORT_NUM = 512;
    MsprofAdditionalInfo addInfoVec[MAX_BATCH_REPORT_NUM] = {};
    uint32_t addInfoIndx = 0;
    uint32_t batchId = 0;
    for (std::vector<Hccl::TaskInfo*>::size_type i = 0; i < taskInfo.size(); i++) {
        GetTaskDetailInfos(taskInfo[i], taskInfos[batchId++]);
        if (batchId == HCCLINFO_REPORT_BATCH_NUM || i == taskInfo.size() - 1) {
            if (!isSupportBatchReport) {
                MsprofAdditionalInfo reporterData{};
                if (!FillBatchReporterData(batchId, taskInfos, reporterData)) {
                    return;
                }
                ReportAdditionInfo(reporterData);
            } else {
                if (!ReportBatchAddInfo(
                        batchId, taskInfos, addInfoVec, addInfoIndx, MAX_BATCH_REPORT_NUM, i == taskInfo.size() - 1)) {
                    return;
                }
            }
            batchId = 0;
            errno_t ret = memset_s(taskInfos, sizeof(taskInfos), 0, sizeof(taskInfos));
            if (ret != EOK) {
                THROW<InternalException>(
                    "[ProfilingHandler] Failed to memset_s taskInfos, dest[%p], destMax[%zu], "
                    "c[%d], count[%zu], ret[%d].",
                    static_cast<void*>(taskInfos), sizeof(taskInfos), 0, sizeof(taskInfos), ret);
            }
        }
    }
}

void ProfilingHandlerLite::FillTaskParamDetails(const TaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const
{
    const auto& taskPara = it->taskParam_.taskPara;
    switch (it->taskParam_.taskType) {
        case TaskParamType::TASK_SDMA:
        case TaskParamType::TASK_RDMA:
        case TaskParamType::TASK_UB_INLINE_WRITE:
        case TaskParamType::TASK_WRITE_WITH_NOTIFY:
        case TaskParamType::TASK_UB:
            taskDetailsInfos.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.src));
            taskDetailsInfos.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.dst));
            taskDetailsInfos.dataSize = static_cast<u32>(taskPara.DMA.size);
            taskDetailsInfos.notifyID = taskPara.DMA.notifyID;
            taskDetailsInfos.linkType = static_cast<uint16_t>(taskPara.DMA.linkType);
            break;
        case TaskParamType::TASK_REDUCE_INLINE:
        case TaskParamType::TASK_REDUCE_TBE:
        case TaskParamType::TASK_UB_REDUCE_INLINE:
        case TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY:
            taskDetailsInfos.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.src));
            taskDetailsInfos.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.dst));
            taskDetailsInfos.dataSize = static_cast<u32>(taskPara.Reduce.size);
            taskDetailsInfos.notifyID = taskPara.Reduce.notifyID;
            taskDetailsInfos.dataType = static_cast<uint16_t>(taskPara.Reduce.dataType);
            taskDetailsInfos.linkType = static_cast<uint16_t>(taskPara.Reduce.linkType);
            taskDetailsInfos.opType = taskPara.Reduce.reduceOp;
            break;
        case TaskParamType::TASK_NOTIFY_RECORD:
        case TaskParamType::TASK_NOTIFY_WAIT:
            taskDetailsInfos.notifyID = taskPara.Notify.notifyID;
            break;
        default:
            break;
    }
}

void ProfilingHandlerLite::GetTaskDetailInfos(const TaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const
{
    constexpr u32 INVALID_LOCAL_RANK = UINT32_MAX;
    constexpr u32 INVALID_REMOTE_RANK = UINT32_MAX;
    taskDetailsInfos.localRank = INVALID_LOCAL_RANK;
    taskDetailsInfos.remoteRank = INVALID_REMOTE_RANK;
    if (it->dfxOpInfo_ != nullptr) {
        auto cacheIt = taskTypeHashCache_.find(static_cast<uint32_t>(it->taskParam_.taskType));
        taskDetailsInfos.itemId = (cacheIt != taskTypeHashCache_.end()) ? cacheIt->second : INVALID_U64;
        auto cclTagIt = cachedCclTag_.find(it->dfxOpInfo_->tag_);
        taskDetailsInfos.cclTag = (cclTagIt != cachedCclTag_.end()) ? cclTagIt->second : INVALID_U64;
        taskDetailsInfos.remoteRank = it->GetRemoteRankId();
        taskDetailsInfos.groupName = cachedGroupName_;
        taskDetailsInfos.rankSize = cachedRankSize_;
        taskDetailsInfos.localRank = it->dfxOpInfo_->op_.myRank;
        taskDetailsInfos.stage = 0;
    } else {
        HCCL_WARNING("[ProfilingHandlerLite::GetTaskDetailInfos] dfxOpInfo_ is nullptr!");
        taskDetailsInfos.remoteRank = taskDetailsInfos.localRank;
    }
    taskDetailsInfos.stage = 0;
    FillTaskParamDetails(it, taskDetailsInfos);
    taskDetailsInfos.timeStamp = ProfGetCurCpuTimestamp();
    taskDetailsInfos.durationEstimated = 0;
    taskDetailsInfos.taskId = it->taskId_;
    taskDetailsInfos.streamId = it->streamId_;
    taskDetailsInfos.planeID = 0;
    taskDetailsInfos.transportType = static_cast<int32_t>(SimpleTaskType::UB);
    if (taskDetailsInfos.remoteRank == INVALID_VALUE_RANKID) {
        taskDetailsInfos.transportType = static_cast<int32_t>(SimpleTaskType::LOCAL);
        taskDetailsInfos.remoteRank = taskDetailsInfos.localRank;
    }
    taskDetailsInfos.role = static_cast<uint32_t>(TaskRole::DST);
    taskDetailsInfos.workFlowMode = static_cast<uint32_t>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

void ProfilingHandlerLite::DumpTaskDetails(
    [[maybe_unused]] const MsprofAicpuHcclTaskInfo& taskDetailsInfos, const TaskInfo* taskInfo) const
{
    HCCL_INFO("[ProfilingHandlerLite] DumpTaskDetails %s", taskInfo->Describe().c_str());
}

void ProfilingHandlerLite::ReportMainStreamTask(const FlagTaskInfo& flagTaskInfo) const
{
    if (!GetProfL0State()) {
        HCCL_INFO("[ProfilingHandlerLite][ReportMainStreamTask] l0 is false.");
        return;
    }
    if (aicpu::GetTaskAndStreamId == nullptr) {
        HCCL_WARNING("[ProfilingHandlerLite][ReportMainStreamTask] aicpu::GetTaskAndStreamId is nullptr.");
        return;
    }
    uint64_t aicpuKernelTaskId = 0U;
    uint32_t aicpuKernelStreamId = 0;
    if (aicpu::GetTaskAndStreamId(aicpuKernelTaskId, aicpuKernelStreamId) != aicpu::status_t::AICPU_ERROR_NONE) {
        THROW<InternalException>("[ProfilingHandler] Failed to get task id and stream id.");
    }
    if (aicpuKernelTaskId > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        THROW<InvalidParamsException>("[ProfilingHandler] aicpuKernelTaskId is larger than u32.");
    }
    MsprofAdditionalInfo reporterData{};
    reporterData.level = MSPROF_REPORT_AICPU_LEVEL;
    reporterData.type = MSPROF_REPORT_AICPU_HCCL_FLAG_TASK;
    reporterData.threadId = cachedTid_;
    reporterData.dataLen = sizeof(MsprofAicpuHcclMainStreamTask);
    reporterData.timeStamp = ProfGetCurCpuTimestamp();
    constexpr uint32_t UINT16_BIT_WIDTH = std::numeric_limits<uint16_t>::digits;
    auto* flagtask = reinterpret_cast<MsprofAicpuHcclMainStreamTask*>(reporterData.data);
    flagtask->taskId = static_cast<uint16_t>(flagTaskInfo.taskId >> UINT16_BIT_WIDTH);
    flagtask->streamId = static_cast<uint16_t>(flagTaskInfo.taskId);
    flagtask->type = flagTaskInfo.type;
    uint32_t aicpuKernelTaskIdLow32 = static_cast<uint32_t>(aicpuKernelTaskId);
    flagtask->aicpuTaskId = static_cast<uint16_t>(aicpuKernelTaskIdLow32 >> UINT16_BIT_WIDTH);
    flagtask->aicpuStreamId = static_cast<uint16_t>(aicpuKernelTaskIdLow32);
    HCCL_INFO(
        "[ProfilingHandlerLite][ReportMainStreamTask] streamId:%u, taskId:%u, type:%u,"
        "aicpuStreamId:%u, aicpuTaskId:%u",
        flagtask->streamId, flagtask->taskId, flagtask->type, flagtask->aicpuStreamId, flagtask->aicpuTaskId);
    ReportAdditionInfo(reporterData);
}

void ProfilingHandlerLite::ReportAdditionInfo(const MsprofAdditionalInfo& reporterData) const
{
    if (reportAdditionalInfo_(aging, &reporterData, sizeof(MsprofAdditionalInfo)) != 0) {
        THROW<InternalException>("[ProfilingHandler] ReportAdditionalInfo failed.");
    }
}

void ProfilingHandlerLite::UpdateProfSwitch()
{
    IsL1fromOffToOn();
    IsProfSwitchOn(ProfilingLevel::L0);
    IsProfSwitchOn(ProfilingLevel::L1);
}

bool ProfilingHandlerLite::IsProfOn(uint64_t feature) const
{
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        if (AdprofCheckFeatureIsOn == nullptr) {
            return false;
        }
        return AdprofCheckFeatureIsOn(feature) > 0;
    } else {
        if (feature == ADPROF_TASK_TIME_L1) {
            return enableHcclL1_;
        } else if (feature == ADPROF_TASK_TIME_L0) {
            return enableHcclL0_;
        }
    }

    return false;
}

bool ProfilingHandlerLite::IsProfSwitchOn(ProfilingLevel level)
{
    bool res = false;
    if (level == ProfilingLevel::L0) {
        res = IsProfOn(ADPROF_TASK_TIME_L0);
        enableHcclL0_ = res;
    } else if (level == ProfilingLevel::L1) {
        res = IsProfOn(ADPROF_TASK_TIME_L1);
        enableHcclL1_ = res;
    }
    return res;
}

bool ProfilingHandlerLite::IsL1fromOffToOn()
{
    if (((!GetProfL1State()) && IsProfSwitchOn(ProfilingLevel::L1))) {
        HCCL_INFO("Profiling L1 switch from off to on.");
        return true;
    }
    return false;
}

void ProfilingHandlerLite::SetProL1On(bool val)
{
    HCCL_INFO("[%s] val = [%d]", __func__, val);
    enableHcclL1_ = val;
}

void ProfilingHandlerLite::SetProL0On(bool val)
{
    HCCL_INFO("[%s] val = [%d]", __func__, val);
    enableHcclL0_ = val;
}

uint64_t ProfilingHandlerLite::GetProfHashId(const char* name, uint32_t len) const
{
    if (name == nullptr || len == 0) {
        HCCL_WARNING("HashData is empty. name:%s, len:%u", name, len);
        return INVALID_U64;
    }
    return getProfHashId_(name, len);
}

} // namespace Hccl
