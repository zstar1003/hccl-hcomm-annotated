/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <algorithm>
#include "profiling_handler.h"
#include "dlprof_function_v2.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "sal.h"
#include "task_param.h"
#include "aprof_pub.h"
#include "orion_adapter_rts.h"
#include "communicator_impl.h"
#include "data_type.h"
namespace Hccl {
#define UNUSED(x) (void)(x)

constexpr uint16_t CCU_TYPE = 2;
constexpr uint32_t DPU_DEV_ID_MASK = (1U << 12);

ProfilingHandler ProfilingHandler::instance_;

ProfilingHandler::ProfilingHandler() {}

ProfilingHandler::~ProfilingHandler() {}

ProfilingHandler& ProfilingHandler::GetInstance() { return instance_; }

void ProfilingHandler::InitLog() const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    for (auto i = 0; i < TaskParamType::__COUNT__; ++i) {
        TaskParamType type(static_cast<TaskParamType::Value>(i));
        std::string nameInfo = type.Describe();
        uint64_t hashId = GetProfHashId(nameInfo.c_str(), nameInfo.length());
        HCCL_INFO("[TaskParamType] nameInfo[%s] ret[%llu]", nameInfo.c_str(), hashId);
    }
}

HcclResult ProfilingHandler::Init()
{
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    if (Hccl::DlProfFunction::GetInstance().DlProfFunctionInit() != HCCL_SUCCESS) {
        HCCL_ERROR("[ProfilingHandler][Init] DlProfFunctionInit failed.");
        return HCCL_E_INTERNAL;
    }
    ProfCommandHandle callback = CommandHandleWrapper;
    auto ret = DlProfFunction::GetInstance().dlMsprofRegisterCallback(HCCL, callback);
    if (ret != 0) {
        HCCL_ERROR(
            "[ProfilingHandler][Init]errNo[0x%016llx] Prof Register CtrlCallback fail, return[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
        return HCCL_E_RUNTIME;
    }
    cachedAlgTypeHashId_.store(GetProfHashId("AlgType::NHR", strlen("AlgType::NHR")));
    initializedFlag_ = true;
    InitLog();
    return HCCL_SUCCESS;
}

int32_t ProfilingHandler::CommandHandleWrapper(uint32_t rtType, void* data, uint32_t len)
{
    return instance_.CommandHandle(rtType, data, len);
}

void ProfilingHandler::ReportKernel() const {}

void ProfilingHandler::ReportHostApi(OpType opType, uint64_t beginTime, uint64_t endTime, bool cachedReq, bool isAiCpu)
{
    uint32_t threadId = SalGetTid();
    std::string profName(GetProfOpName(opType));
    if (isAiCpu) {
        profName += "AicpuKernel";
    }
    uint64_t cmdItemId = DlProfFunction::GetInstance().dlMsprofStr2Id(profName.c_str(), profName.length());
    ReportAclApi(opType, beginTime, endTime, cmdItemId, threadId, cachedReq);
    ReportNodeApi(beginTime, endTime, cmdItemId, threadId, cachedReq);
    ReportNodeBasicInfo(endTime, cmdItemId, threadId, cachedReq);
}

void ProfilingHandler::ReportHcclOp(const DfxOpInfo& opInfo, bool cachedReq)
{
    uint32_t threadId = SalGetTid();
    ReportHcclOpInfo(opInfo.endTime_, opInfo, threadId, cachedReq);
}

void ProfilingHandler::ReportHcclTaskApi(
    TaskParamType taskType, uint64_t beginTime, uint64_t endTime, bool isMasterStream, bool cachedReq, bool ignoreLevel)
{
    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type = isMasterStream ? MSPROF_REPORT_HCCL_MASTER_TYPE : MSPROF_REPORT_HCCL_SLAVE_TYPE;
    reporterData.threadId = SalGetTid();
    reporterData.beginTime = beginTime;
    reporterData.endTime = endTime;
    const std::string proName(GetProfTaskOpNameV2(taskType));
    reporterData.itemId = GetProfHashId(proName.c_str(), proName.length());
    if (taskType == TaskParamType::TASK_AICPU_KERNEL) {
        return;
    }
    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cachedTaskApiInfoMutex_);
        cachedTaskApiInfo_.push(reporterData);
    }
    if ((!enableHcclNode_) || (!ignoreLevel && !enableHcclL1_)) {
        return;
    }
    HCCL_INFO(
        "ReportHcclTaskApi, enableHcclNode_[%d],ignoreLevel[%d],enableHcclL1_[%d],taskType[%s], beginTime[%llu], "
        "endTime[%llu], isMasterStream[%d], cachedReq[%d]",
        enableHcclNode_, ignoreLevel, enableHcclL1_, proName.c_str(), beginTime, endTime, isMasterStream, cachedReq);
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportApi(1, &reporterData);
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportApi fail, return[%d]", ret);
    }
}

void ProfilingHandler::SetCachedCclTag()
{
    cachedNewCclTag_.clear();
    for (const auto& item : CMD_OP_TYPE_INFO_MAP) {
        const std::string& cclTag = item.second.second;
        cachedNewCclTag_[item.second.first] = GetProfHashId(cclTag.c_str(), cclTag.length());
    }
    if (cachedNewCclTag_.find(static_cast<uint32_t>(OpType::HCCLGROUPOP)) == cachedNewCclTag_.end()) {
        const std::string cclTag = "OpType::HCCLGROUPOP";
        cachedNewCclTag_[OpType::HCCLGROUPOP] = GetProfHashId(cclTag.c_str(), cclTag.length());
    }
}

uint32_t ProfilingHandler::GetTaskTypeValue(TaskParamType taskType) const
{
    switch (taskType) {
        case TaskParamType::TASK_DPU_INLINE_WRITE:
        case TaskParamType::TASK_DPU_NOTIFY_WAIT:
        case TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY:
        case TaskParamType::TASK_DPU_CHANNEL_FENCE:
            return static_cast<uint32_t>(ProfTaskType::TASK_DPU_HCCL_INFO);
        default:
            return static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO);
    }
}

void ProfilingHandler::FillTaskAdditionInfo(const TaskInfo& taskInfo, MsprofAdditionalInfo& reporterData) const
{
    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.threadId = SalGetTid();
    FillProfCommonInfo(taskInfo, reporterData);

    switch (taskInfo.taskParam_.taskType) {
        case TaskParamType::TASK_DPU_INLINE_WRITE:
        case TaskParamType::TASK_DPU_NOTIFY_WAIT:
        case TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY:
        case TaskParamType::TASK_DPU_CHANNEL_FENCE:
            reporterData.type = static_cast<uint32_t>(ProfTaskType::TASK_DPU_HCCL_INFO);
            FillDpuProfInfo(taskInfo, reporterData);
            break;
        default:
            reporterData.type = static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO);
            FillProfTaskSpecificInfo(taskInfo, reinterpret_cast<MsprofHcclInfo*>(reporterData.data));
            break;
    }
}

void ProfilingHandler::ReportHcclTaskDetails(const TaskInfo& taskInfo, bool cachedReq)
{
    if (enableHcclL1_ == false && !cachedReq) {
        return;
    }
    if (taskInfo.dfxOpInfo_ == nullptr) {
        HCCL_WARNING("[%s] dfxOpInfo_ is nullptr, skip ReportHcclTaskDetails!", __func__);
        return;
    }
    if (taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_WARNING("[%s] comm_ is nullptr, skip ReportHcclTaskDetails!", __func__);
        return;
    }
    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cacheTaskInfosMutex_);
        cacheTaskInfos_.push_back(taskInfo);
    }
    if (enableHcclL1_ == false) {
        return;
    }
    MsprofAdditionalInfo reporterData{};
    FillTaskAdditionInfo(taskInfo, reporterData);
    DumpHCCLReportData(taskInfo, reporterData);
    CallAdditionInfo(reporterData);
}

void ProfilingHandler::ReportHcclTaskDetailsBatchLog(const std::vector<TaskInfo*>& taskInfos) const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    for (const auto& taskInfo : taskInfos) {
        HCCL_INFO("ReportHcclTaskDetailsBatchLog, taskType[%u]", GetTaskTypeValue(taskInfo->taskParam_.taskType));
    }
}

void ProfilingHandler::ReportHcclTaskDetailsBatch(const std::vector<TaskInfo*>& taskInfos, bool cachedReq)
{
    if (taskInfos.empty()) {
        HCCL_WARNING("[ProfilingHandler::ReportHcclTaskDetailsBatch] taskInfos is empty!");
        return;
    }
    if (taskInfos[0]->dfxOpInfo_ == nullptr) {
        HCCL_WARNING("[ProfilingHandler::ReportHcclTaskDetailsBatch] taskInfo.dfxOpInfo_ is nullptr, skip!");
        return;
    }
    if (taskInfos[0]->dfxOpInfo_->comm_ == nullptr) {
        HCCL_WARNING("[ProfilingHandler::ReportHcclTaskDetailsBatch] taskInfo.dfxOpInfo_->comm_ is nullptr, skip!");
        return;
    }
    if (enableHcclL1_ == false && !cachedReq) {
        return;
    }
    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cacheTaskInfosMutex_);
        for (const auto& taskInfo : taskInfos) {
            cacheTaskInfos_.push_back(*taskInfo);
        }
    }
    if (enableHcclL1_ == false) {
        return;
    }
    ReportHcclTaskDetailsBatchLog(taskInfos);
    for (const auto& taskInfo : taskInfos) {
        MsprofAdditionalInfo reporterData{};
        FillTaskAdditionInfo(*taskInfo, reporterData);
        CallAdditionInfo(reporterData);
    }
}

void ProfilingHandler::CallAdditionInfo(MsprofAdditionalInfo& reporterData) const
{
    s32 ret
        = DlProfFunction::GetInstance().dlMsprofReportAdditionalInfo(1, &reporterData, sizeof(MsprofAdditionalInfo));
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportAdditionalInfo failed, return[%d]", ret);
    }
}

void ProfilingHandler::FillProfCommonInfo(const TaskInfo& taskInfo, MsprofAdditionalInfo& reporterData) const
{
    if (taskInfo.dfxOpInfo_ == nullptr) {
        HCCL_WARNING("[ProfilingHandler][%s]taskInfo.dfxOpInfo_ is nullptr, skip GetProfCommonInfo.", __func__);
        return;
    }
    reporterData.timeStamp = taskInfo.taskParam_.endTime;
    reporterData.dataLen = sizeof(MsprofHcclInfo);
    auto* profInfo = reinterpret_cast<MsprofHcclInfo*>(reporterData.data);
    *profInfo = MsprofHcclInfo();

    const auto& profName = GetProfTaskOpNameV2(taskInfo.taskParam_.taskType);
    profInfo->itemId = GetProfHashId(profName.c_str(), profName.length());
    auto newCclTagIt = cachedNewCclTag_.find(taskInfo.dfxOpInfo_->op_.opType);
    profInfo->cclTag = (newCclTagIt != cachedNewCclTag_.end()) ? newCclTagIt->second : INVALID_U64;
    const auto& opTag = taskInfo.dfxOpInfo_->op_.opTag;
    uint64_t groupName = GetProfHashId(opTag.c_str(), opTag.length());
    if (taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_WARNING("[ProfilingHandler][%s]taskInfo.dfxOpInfo_->comm_ is nullptr, skip GetProfCommonInfo.", __func__);
        return;
    }
    if (taskInfo.dfxOpInfo_->isIndop_ == true) {
        profInfo->groupName = groupName;
        profInfo->rankSize = taskInfo.dfxOpInfo_->rankSize_;
    } else {
        CommunicatorImpl* commImp = static_cast<CommunicatorImpl*>(taskInfo.dfxOpInfo_->comm_);
        if (commImp == nullptr) {
            HCCL_WARNING("[ProfilingHandler][%s]commImp is nullptr, skip GetProfCommonInfo.", __func__);
            return;
        }
        profInfo->groupName = groupName;
        profInfo->rankSize = commImp->GetRankSize();
    }
    profInfo->workFlowMode = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    profInfo->planeID = 0;
    profInfo->stage = 0;
    profInfo->role = static_cast<uint32_t>(TaskRole::DST);
    profInfo->durationEstimated = 0;
    profInfo->localRank = taskInfo.dfxOpInfo_->op_.myRank;
    profInfo->remoteRank = taskInfo.remoteRank_;
    profInfo->dataType = taskInfo.dfxOpInfo_->op_.dataType;
    profInfo->opType = taskInfo.dfxOpInfo_->op_.opType;
    profInfo->transportType = static_cast<int32_t>(SimpleTaskType::UB);
    if (profInfo->remoteRank == INVALID_VALUE_RANKID) {
        profInfo->transportType = static_cast<int32_t>(SimpleTaskType::LOCAL);
        profInfo->remoteRank = profInfo->localRank;
        profInfo->linkType = 0;
    }
}

void ProfilingHandler::FillProfTaskSpecificInfo(const TaskInfo& taskInfo, MsprofHcclInfo* profInfo) const
{
    const auto& taskType = taskInfo.taskParam_.taskType;
    const auto& taskPara = taskInfo.taskParam_.taskPara;
    switch (taskType) {
        case TaskParamType::TASK_SDMA:
        case TaskParamType::TASK_RDMA:
            profInfo->srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.src));
            profInfo->dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.dst));
            profInfo->dataSize = static_cast<u32>(taskPara.DMA.size);
            profInfo->notifyID = taskPara.DMA.notifyID;
            profInfo->linkType = static_cast<uint16_t>(taskPara.DMA.linkType);
            break;
        case TaskParamType::TASK_REDUCE_INLINE:
        case TaskParamType::TASK_REDUCE_TBE:
            profInfo->srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.src));
            profInfo->dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.dst));
            profInfo->dataSize = static_cast<u32>(taskPara.Reduce.size);
            profInfo->notifyID = taskPara.Reduce.notifyID;
            profInfo->linkType = static_cast<uint16_t>(taskPara.Reduce.linkType);
            break;
        case TaskParamType::TASK_NOTIFY_RECORD:
        case TaskParamType::TASK_NOTIFY_WAIT:
            profInfo->notifyID = taskPara.Notify.notifyID;
            break;
        case TaskParamType::TASK_CCU:
            HCCL_INFO("current taskType is TASK_CCU");
            ReportCcuInfo(taskInfo);
            break;
        default:
            break;
    }
}

void ProfilingHandler::ConvertHcclInfoToDpuTrack(MsprofAdditionalInfo& reporterData) const
{
    auto* profInfo = reinterpret_cast<MsprofHcclInfo*>(reporterData.data);
    uint64_t itemId = profInfo->itemId;
    uint64_t cclTag = profInfo->cclTag;
    uint64_t groupName = profInfo->groupName;
    uint32_t localRank = profInfo->localRank;
    uint32_t remoteRank = profInfo->remoteRank;
    uint32_t rankSize = profInfo->rankSize;
    uint32_t workFlowMode = profInfo->workFlowMode;
    uint16_t planeID = static_cast<uint16_t>(profInfo->planeID);
    uint32_t stage = profInfo->stage;
    uint32_t role = profInfo->role;
    double durationEstimated = profInfo->durationEstimated;

    reporterData.dataLen = sizeof(MsprofDpuHcclTrack);
    auto* dpuProfInfo = reinterpret_cast<MsprofDpuHcclTrack*>(reporterData.data);
    *dpuProfInfo = MsprofDpuHcclTrack();

    dpuProfInfo->itemId = itemId;
    dpuProfInfo->cclTag = cclTag;
    dpuProfInfo->groupName = groupName;
    dpuProfInfo->localRank = localRank;
    dpuProfInfo->remoteRank = remoteRank;
    dpuProfInfo->rankSize = rankSize;
    dpuProfInfo->workFlowMode = static_cast<uint8_t>(workFlowMode);
    dpuProfInfo->planeID = planeID;
    dpuProfInfo->stage = stage;
    dpuProfInfo->role = static_cast<uint8_t>(role);
    dpuProfInfo->durationEstimated = durationEstimated;
}

void ProfilingHandler::FillDpuTaskParaDetails(const TaskInfo& taskInfo, MsprofDpuHcclTrack* dpuProfInfo) const
{
    const auto& taskPara = taskInfo.taskParam_.taskPara;
    switch (taskInfo.taskParam_.taskType) {
        case TaskParamType::TASK_DPU_INLINE_WRITE:
        case TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY:
            dpuProfInfo->srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.src));
            dpuProfInfo->dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.dst));
            dpuProfInfo->dataSize = static_cast<u32>(taskPara.DMA.size);
            dpuProfInfo->notifyID = taskPara.DMA.notifyID;
            break;
        case TaskParamType::TASK_DPU_NOTIFY_WAIT:
        case TaskParamType::TASK_DPU_CHANNEL_FENCE:
            dpuProfInfo->notifyID = taskPara.Notify.notifyID;
            break;
        default:
            break;
    }
}

void ProfilingHandler::FillDpuProfInfo(const TaskInfo& taskInfo, MsprofAdditionalInfo& reporterData) const
{
    if (taskInfo.dfxOpInfo_ == nullptr) {
        HCCL_WARNING("[ProfilingHandler::FillDpuProfInfo] taskInfo.dfxOpInfo_ is nullptr, skip GetDpuProfInfo!");
        return;
    }
    ConvertHcclInfoToDpuTrack(reporterData);
    auto* dpuProfInfo = reinterpret_cast<MsprofDpuHcclTrack*>(reporterData.data);
    dpuProfInfo->dataType = static_cast<uint32_t>(taskInfo.dfxOpInfo_->op_.dataType);
    dpuProfInfo->opType = static_cast<uint32_t>(taskInfo.dfxOpInfo_->op_.opType);
    dpuProfInfo->transportType = static_cast<uint32_t>(SimpleTaskType::ROCE);
    dpuProfInfo->aicpu_task_id = taskInfo.taskParam_.aicpuTaskId;
    dpuProfInfo->npuDevId = taskInfo.taskParam_.npuDevId;
    dpuProfInfo->dpuDevId = DPU_DEV_ID_MASK;
    const auto& taskPara = taskInfo.taskParam_.taskPara;
    dpuProfInfo->linkType = static_cast<uint16_t>(taskPara.DMA.linkType);
    if (dpuProfInfo->remoteRank == INVALID_VALUE_RANKID) {
        dpuProfInfo->transportType = static_cast<uint32_t>(SimpleTaskType::LOCAL);
        dpuProfInfo->remoteRank = dpuProfInfo->localRank;
        dpuProfInfo->linkType = 0;
    }
    dpuProfInfo->taskId = taskInfo.taskId_;
    dpuProfInfo->streamId = taskInfo.streamId_;
    dpuProfInfo->timeStamp = taskInfo.taskParam_.beginTime;
    HCCL_INFO(
        "[FillDpuProfInfo]taskId[%u], streamId[%u], npuDevId[%u], dpuDevId[%u], "
        "starttime[%llu], endtime[%llu], aicputaskId[%u].",
        dpuProfInfo->taskId, dpuProfInfo->streamId, dpuProfInfo->npuDevId, dpuProfInfo->dpuDevId,
        dpuProfInfo->timeStamp, reporterData.timeStamp, dpuProfInfo->aicpu_task_id);
    FillDpuTaskParaDetails(taskInfo, dpuProfInfo);
}

void ProfilingHandler::GetHCCLReportData(const TaskInfo& taskInfo, HCCLReportData& hcclReportData) const
{
    if (taskInfo.dfxOpInfo_ == nullptr) {
        HCCL_ERROR("[ProfilingHandler]taskInfo.dfxOpInfo_ is nullptr");
        return;
    }
    hcclReportData.ts = taskInfo.taskParam_.endTime;
    const auto& profName = GetProfTaskOpNameV2(taskInfo.taskParam_.taskType);
    hcclReportData.profInfo.itemId = GetProfHashId(profName.c_str(), profName.length());
    auto newCclTagIt = cachedNewCclTag_.find(taskInfo.dfxOpInfo_->op_.opType);
    hcclReportData.profInfo.cclTag = (newCclTagIt != cachedNewCclTag_.end()) ? newCclTagIt->second : INVALID_U64;
    const auto& opTag = taskInfo.dfxOpInfo_->op_.opTag;
    uint64_t groupName = GetProfHashId(opTag.c_str(), opTag.length());
    if (taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[ProfilingHandler]taskInfo.dfxOpInfo_->comm_ is nullptr");
        return;
    }
    if (taskInfo.dfxOpInfo_->isIndop_ == true) {
        hcclReportData.profInfo.groupName = groupName;
        hcclReportData.profInfo.rankSize = taskInfo.dfxOpInfo_->rankSize_;
    } else {
        CommunicatorImpl* commImp = static_cast<CommunicatorImpl*>(taskInfo.dfxOpInfo_->comm_);
        if (commImp == nullptr) {
            HCCL_ERROR("[ProfilingHandler]commImp is nullptr");
            return;
        }
        hcclReportData.profInfo.groupName = groupName;
        hcclReportData.profInfo.rankSize = commImp->GetRankSize();
    }
    hcclReportData.profInfo.workFlowMode = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    hcclReportData.profInfo.planeID = 0;
    hcclReportData.profInfo.stage = 0;
    hcclReportData.profInfo.role = static_cast<uint32_t>(TaskRole::DST);
    hcclReportData.profInfo.durationEstimated = 0;
    hcclReportData.profInfo.localRank = taskInfo.dfxOpInfo_->op_.myRank;
    hcclReportData.profInfo.remoteRank = taskInfo.remoteRank_;
    hcclReportData.profInfo.dataType = taskInfo.dfxOpInfo_->op_.dataType;
    hcclReportData.profInfo.opType = taskInfo.dfxOpInfo_->op_.opType;
    hcclReportData.profInfo.transportType = static_cast<int32_t>(SimpleTaskType::UB);
    if (hcclReportData.profInfo.remoteRank == INVALID_VALUE_RANKID) {
        hcclReportData.profInfo.transportType = static_cast<int32_t>(SimpleTaskType::LOCAL);
        hcclReportData.profInfo.remoteRank = hcclReportData.profInfo.localRank;
        hcclReportData.profInfo.linkType = 0;
    }
    const auto& taskType = taskInfo.taskParam_.taskType;
    const auto& taskPara = taskInfo.taskParam_.taskPara;
    switch (taskType) {
        case TaskParamType::TASK_DPU_INLINE_WRITE:
        case TaskParamType::TASK_DPU_NOTIFY_WAIT:
        case TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY:
        case TaskParamType::TASK_DPU_CHANNEL_FENCE:
            hcclReportData.dpuProfInfo.itemId = hcclReportData.profInfo.itemId;
            hcclReportData.dpuProfInfo.cclTag = hcclReportData.profInfo.cclTag;
            hcclReportData.dpuProfInfo.groupName = hcclReportData.profInfo.groupName;
            hcclReportData.dpuProfInfo.localRank = hcclReportData.profInfo.localRank;
            hcclReportData.dpuProfInfo.remoteRank = hcclReportData.profInfo.remoteRank;
            hcclReportData.dpuProfInfo.rankSize = hcclReportData.profInfo.rankSize;
            hcclReportData.dpuProfInfo.workFlowMode = hcclReportData.profInfo.workFlowMode;
            hcclReportData.dpuProfInfo.planeID = hcclReportData.profInfo.planeID;
            hcclReportData.dpuProfInfo.stage = hcclReportData.profInfo.stage;
            hcclReportData.dpuProfInfo.role = hcclReportData.profInfo.role;
            hcclReportData.dpuProfInfo.durationEstimated = hcclReportData.profInfo.durationEstimated;
            hcclReportData.dpuProfInfo.dataType = static_cast<uint32_t>(taskInfo.dfxOpInfo_->op_.dataType);
            hcclReportData.dpuProfInfo.opType = static_cast<uint32_t>(taskInfo.dfxOpInfo_->op_.opType);
            hcclReportData.dpuProfInfo.transportType = static_cast<uint32_t>(SimpleTaskType::ROCE);
            hcclReportData.dpuProfInfo.aicpu_task_id = taskInfo.taskParam_.aicpuTaskId;
            hcclReportData.dpuProfInfo.npuDevId = taskInfo.taskParam_.npuDevId;
            hcclReportData.dpuProfInfo.dpuDevId = DPU_DEV_ID_MASK;
            hcclReportData.dpuProfInfo.linkType = static_cast<uint16_t>(taskPara.DMA.linkType);
            if (hcclReportData.dpuProfInfo.remoteRank == INVALID_VALUE_RANKID) {
                hcclReportData.dpuProfInfo.transportType = static_cast<uint32_t>(SimpleTaskType::LOCAL);
                hcclReportData.dpuProfInfo.remoteRank = hcclReportData.dpuProfInfo.localRank;
                hcclReportData.dpuProfInfo.linkType = 0;
            }
            hcclReportData.dpuProfInfo.taskId = taskInfo.taskId_;
            hcclReportData.dpuProfInfo.streamId = taskInfo.streamId_;
            hcclReportData.dpuProfInfo.timeStamp = taskInfo.taskParam_.beginTime;
            switch (taskType) {
                case TaskParamType::TASK_DPU_INLINE_WRITE:
                case TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY:
                    hcclReportData.dpuProfInfo.srcAddr
                        = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.src));
                    hcclReportData.dpuProfInfo.dstAddr
                        = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.dst));
                    hcclReportData.dpuProfInfo.dataSize = static_cast<u32>(taskPara.DMA.size);
                    hcclReportData.dpuProfInfo.notifyID = taskPara.DMA.notifyID;
                    break;
                case TaskParamType::TASK_DPU_NOTIFY_WAIT:
                case TaskParamType::TASK_DPU_CHANNEL_FENCE:
                    hcclReportData.dpuProfInfo.notifyID = taskPara.Notify.notifyID;
                    break;
                default:
                    break;
            }
            break;
        default:
            switch (taskType) {
                case TaskParamType::TASK_SDMA:
                case TaskParamType::TASK_RDMA:
                    hcclReportData.profInfo.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.src));
                    hcclReportData.profInfo.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.DMA.dst));
                    hcclReportData.profInfo.dataSize = static_cast<u32>(taskPara.DMA.size);
                    hcclReportData.profInfo.notifyID = taskPara.DMA.notifyID;
                    hcclReportData.profInfo.linkType = static_cast<uint16_t>(taskPara.DMA.linkType);
                    break;
                case TaskParamType::TASK_REDUCE_INLINE:
                case TaskParamType::TASK_REDUCE_TBE:
                    hcclReportData.profInfo.srcAddr
                        = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.src));
                    hcclReportData.profInfo.dstAddr
                        = static_cast<u64>(reinterpret_cast<uintptr_t>(taskPara.Reduce.dst));
                    hcclReportData.profInfo.dataSize = static_cast<u32>(taskPara.Reduce.size);
                    hcclReportData.profInfo.notifyID = taskPara.Reduce.notifyID;
                    hcclReportData.profInfo.linkType = static_cast<uint16_t>(taskPara.Reduce.linkType);
                    break;
                case TaskParamType::TASK_NOTIFY_RECORD:
                case TaskParamType::TASK_NOTIFY_WAIT:
                    hcclReportData.profInfo.notifyID = taskPara.Notify.notifyID;
                    break;
                default:
                    break;
            }
            break;
    }
}

void ProfilingHandler::DumpHCCLReportData(const TaskInfo& taskInfo, const MsprofAdditionalInfo& reporterData) const
{
    if (reporterData.type == static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO)) {
        auto* profInfo = reinterpret_cast<const MsprofHcclInfo*>(reporterData.data);
        HCCL_INFO(
            "MsprofAdditionalInfo profInfo: timeStamp[%llu], itemId[%llu], cclTag[%llu], groupName[%llu], "
            "localRank[%u], remoteRank[%u], rankSize[%u], workFlowMode[%u], planeID[%u], ctxId[%u], "
            "stage[%u], role[%u], durationEstimated[%f], taskType[%d]",
            reporterData.timeStamp, profInfo->itemId, profInfo->cclTag, profInfo->groupName, profInfo->localRank,
            profInfo->remoteRank, profInfo->rankSize, profInfo->workFlowMode, profInfo->planeID, profInfo->ctxId,
            profInfo->stage, profInfo->role, profInfo->durationEstimated,
            static_cast<int>(taskInfo.taskParam_.taskType));
        HCCL_INFO(
            "MsprofAdditionalInfo profInfo detail: srcAddr[%llu], dstAddr[%llu], dataSize[%llu], notifyID[%llu], "
            "linkType[%u], opType[%s], transportType[%u], dataType[%s], rdmaType[%u]",
            profInfo->srcAddr, profInfo->dstAddr, profInfo->dataSize, profInfo->notifyID, profInfo->linkType,
            OpTypeToSerialString(profInfo->opType).c_str(), profInfo->transportType,
            DataTypeToSerialString(profInfo->dataType).c_str(), profInfo->rdmaType);
    } else {
        auto* dpuProfInfo = reinterpret_cast<const MsprofDpuHcclTrack*>(reporterData.data);
        HCCL_INFO(
            "MsprofAdditionalInfo dpuProfInfo: timeStamp[%llu], itemId[%llu], cclTag[%llu], groupName[%llu], "
            "localRank[%u], remoteRank[%u], rankSize[%u], workFlowMode[%u], planeID[%u], "
            "stage[%u], role[%u], durationEstimated[%f], taskType[%d]",
            reporterData.timeStamp, dpuProfInfo->itemId, dpuProfInfo->cclTag, dpuProfInfo->groupName,
            dpuProfInfo->localRank, dpuProfInfo->remoteRank, dpuProfInfo->rankSize, dpuProfInfo->workFlowMode,
            dpuProfInfo->planeID, dpuProfInfo->stage, dpuProfInfo->role, dpuProfInfo->durationEstimated,
            static_cast<int>(taskInfo.taskParam_.taskType));
        HCCL_INFO(
            "MsprofAdditionalInfo dpuProfInfo detail: srcAddr[%llu], dstAddr[%llu], dataSize[%llu], notifyID[%llu], "
            "linkType[%u], opType[%s], transportType[%u], dataType[%s], rdmaType[%u], "
            "taskId[%u], aicpu_task_id[%u], streamId[%u], npuDevId[%u], dpuDevId[%u], timeStamp[%llu]",
            dpuProfInfo->srcAddr, dpuProfInfo->dstAddr, dpuProfInfo->dataSize, dpuProfInfo->notifyID,
            dpuProfInfo->linkType, OpTypeToSerialString(dpuProfInfo->opType).c_str(), dpuProfInfo->transportType,
            DataTypeToSerialString(dpuProfInfo->dataType).c_str(), dpuProfInfo->rdmaType, dpuProfInfo->taskId,
            dpuProfInfo->aicpu_task_id, dpuProfInfo->streamId, dpuProfInfo->npuDevId, dpuProfInfo->dpuDevId,
            dpuProfInfo->timeStamp);
    }
}

void ProfilingHandler::LogCcuTaskInfo(
    const CcuProfilingInfo& info, const TaskInfo& taskInfo, uint64_t itemId, uint64_t groupName, u32 rankId,
    u32 ranksize) const
{
    HCCL_INFO(
        "[ProfilingHandler]GetCcuTaskInfo, ccuTaskInfo data is: version[%u], workFlowMode[%u], itemId[%llu], "
        "groupName[%llu], rankId[%u], ranksize[%u], streamId[%u], taskId[%u], dieId[%u], "
        "missionId[%u],instrId[%u]",
        0, static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), itemId, groupName, rankId, ranksize,
        taskInfo.streamId_, taskInfo.taskId_, info.dieId, info.missionId, info.instrId);
}

void ProfilingHandler::LogCcuWaitSignalInfo(
    const CcuProfilingInfo& info, const TaskInfo& taskInfo, uint64_t itemId, uint64_t groupName, u32 rankId,
    u32 ranksize) const
{
    HCCL_INFO(
        "[ProfilingHandler]GetCcuWaitSignalInfo, waitSignalInfo data is: version[%u], itemId[%llu], groupName[%llu], "
        "rankId[%u], ranksize[%u], workFlowMode[%u], streamId[%u], taskId[%u], dieId[%u],instrId[%u],missionId[%u], "
        "ckeId[%u],mask[%u]",
        0, itemId, groupName, rankId, ranksize, static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE),
        taskInfo.streamId_, taskInfo.taskId_, info.dieId, info.instrId, info.missionId, info.ckeId, info.mask);
}

void ProfilingHandler::LogCcuGroupInfo(
    const CcuProfilingInfo& info, const TaskInfo& taskInfo, uint64_t itemId, uint64_t groupName, u32 rankId,
    u32 ranksize) const
{
    HCCL_INFO(
        "[ProfilingHandler]GetCcuGroupInfo, ccuGroupInfo data is: version[%u], itemId[%llu], "
        "groupName[%llu], rankId[%u], ranksize[%u], workFlowMode[%u], streamId[%u], taskId[%u], "
        "dieId[%u],instrId[%u],missionId[%u], dataSize[%llu]",
        0, itemId, groupName, rankId, ranksize, static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE),
        taskInfo.streamId_, taskInfo.taskId_, info.dieId, info.instrId, info.missionId, info.dataSize);
    if (info.reduceOpType != INVALID_TYPE_VALUE) {
        HCCL_INFO("ccuGroupInfo reduceOpType is [%d]", static_cast<int>(info.reduceOpType));
    }
    if (info.inputDataType != INVALID_TYPE_VALUE) {
        HCCL_INFO("ccuGroupInfo inputDataType is [%d]", static_cast<int>(info.inputDataType));
    }
    if (info.outputDataType != INVALID_TYPE_VALUE) {
        HCCL_INFO("ccuGroupInfo outputDataType is [%d]", static_cast<int>(info.outputDataType));
    }
    for (auto i = 0; i < CCU_MAX_CHANNEL_NUM; i++) {
        if (info.channelId[i] != INVALID_VALUE_CHANNELID && info.remoteRankId[i] != INVALID_VALUE_RANKID) {
            HCCL_INFO(
                "[ProfilingHandler]GetCcuGroupInfo, ccuGroupInfo data is: channelId[%d] = %u, "
                "remoteRankId[%d] = %u",
                i, info.channelId[i], i, info.remoteRankId[i]);
        }
    }
}

void ProfilingHandler::ReportCcuInfoLog(const TaskInfo& taskInfo) const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    if (!enableHcclL1_) {
        return;
    }
    if (taskInfo.taskParam_.ccuDetailInfo == nullptr || taskInfo.dfxOpInfo_ == nullptr) {
        return;
    }
    auto ccuDetailInfo = taskInfo.taskParam_.ccuDetailInfo;
    for (const auto& info : *ccuDetailInfo) {
        uint64_t itemId = GetProfHashId(info.name.c_str(), info.name.length());
        uint64_t groupName
            = GetProfHashId(taskInfo.dfxOpInfo_->op_.opTag.c_str(), taskInfo.dfxOpInfo_->op_.opTag.length());
        u32 rankId = 0;
        u32 ranksize = 0;
        if (taskInfo.dfxOpInfo_->isIndop_ == true) {
            rankId = taskInfo.dfxOpInfo_->op_.myRank;
            ranksize = taskInfo.dfxOpInfo_->rankSize_;
        } else {
            CommunicatorImpl* commImp = static_cast<CommunicatorImpl*>(taskInfo.dfxOpInfo_->comm_);
            rankId = commImp->GetIdIndex();
            ranksize = commImp->GetRankSize();
        }
        if (info.type == 0) {
            LogCcuTaskInfo(info, taskInfo, itemId, groupName, rankId, ranksize);
        } else if (info.type == 1) {
            LogCcuWaitSignalInfo(info, taskInfo, itemId, groupName, rankId, ranksize);
        } else if (info.type == CCU_TYPE) {
            LogCcuGroupInfo(info, taskInfo, itemId, groupName, rankId, ranksize);
        }
    }
}

void ProfilingHandler::ReportCcuInfo(const TaskInfo& taskInfo) const
{
    ReportCcuInfoLog(taskInfo);
    if (taskInfo.taskParam_.ccuDetailInfo == nullptr) {
        HCCL_WARNING("[ProfilingHandler]ReportCcuInfo ccuDetailInfo is nullptr.");
        return;
    }
    if (taskInfo.dfxOpInfo_ == nullptr) {
        HCCL_WARNING("[ProfilingHandler]ReportCcuInfo dfxOpInfo_ is nullptr, skip ReportCcuInfo.");
        return;
    }
    auto ccuDetailInfo = taskInfo.taskParam_.ccuDetailInfo;
    for (const auto& info : *ccuDetailInfo) {
        if (info.type == 0 && enableHcclL1_) {
            GetCcuTaskInfo(taskInfo, info);
        } else if (info.type == 1 && enableHcclL1_) {
            GetCcuWaitSignalInfo(taskInfo, info);
        } else if (info.type == CCU_TYPE && enableHcclL1_) {
            GetCcuGroupInfo(taskInfo, info);
        }
    }
}

void ProfilingHandler::GetCcuTaskInfo(const TaskInfo& taskInfo, const CcuProfilingInfo& info) const
{
    uint64_t timestamp = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    MsprofAdditionalInfo reporterData{};
    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_CCU_TASK_INFO;
    reporterData.threadId = SalGetTid();
    reporterData.dataLen = sizeof(MsprofCcuTaskInfo);
    reporterData.timeStamp = timestamp;
    auto* ccuTaskInfo = reinterpret_cast<MsprofCcuTaskInfo*>(reporterData.data);
    ccuTaskInfo->version = 0;
    ccuTaskInfo->workFlowMode = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    ccuTaskInfo->itemId = GetProfHashId(info.name.c_str(), info.name.length());
    uint64_t groupName = GetProfHashId(taskInfo.dfxOpInfo_->op_.opTag.c_str(), taskInfo.dfxOpInfo_->op_.opTag.length());
    ccuTaskInfo->groupName = groupName;
    if (taskInfo.dfxOpInfo_->isIndop_ == true) {
        ccuTaskInfo->rankId = taskInfo.dfxOpInfo_->op_.myRank;
        ccuTaskInfo->ranksize = taskInfo.dfxOpInfo_->rankSize_;
    } else {
        CommunicatorImpl* commImp = static_cast<CommunicatorImpl*>(taskInfo.dfxOpInfo_->comm_);
        ccuTaskInfo->rankId = commImp->GetIdIndex();
        ccuTaskInfo->ranksize = commImp->GetRankSize();
    }

    ccuTaskInfo->streamId = taskInfo.streamId_;
    ccuTaskInfo->taskId = taskInfo.taskId_;
    ccuTaskInfo->dieId = info.dieId;
    ccuTaskInfo->missionId = info.missionId;
    ccuTaskInfo->instrId = info.instrId;
    ReportAdditionInfo(reporterData);
}

void ProfilingHandler::GetCcuGroupInfo(const TaskInfo& taskInfo, const CcuProfilingInfo& info) const
{
    MsprofAdditionalInfo reporterData{};
    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_CCU_GROUP_INFO;
    reporterData.threadId = SalGetTid();
    reporterData.dataLen = sizeof(MsprofCcuGroupInfo);
    reporterData.timeStamp = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    auto* ccuGroupInfo = reinterpret_cast<MsprofCcuGroupInfo*>(reporterData.data);
    ccuGroupInfo->version = 0;
    ccuGroupInfo->itemId = GetProfHashId(info.name.c_str(), info.name.length());
    uint64_t groupName = GetProfHashId(taskInfo.dfxOpInfo_->op_.opTag.c_str(), taskInfo.dfxOpInfo_->op_.opTag.length());
    ccuGroupInfo->groupName = groupName;
    if (taskInfo.dfxOpInfo_->isIndop_ == true) {
        ccuGroupInfo->rankId = taskInfo.dfxOpInfo_->op_.myRank;
        ccuGroupInfo->ranksize = taskInfo.dfxOpInfo_->rankSize_;
    } else {
        CommunicatorImpl* commImp = static_cast<CommunicatorImpl*>(taskInfo.dfxOpInfo_->comm_);
        ccuGroupInfo->rankId = commImp->GetIdIndex();
        ccuGroupInfo->ranksize = commImp->GetRankSize();
    }
    ccuGroupInfo->workFlowMode = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    ccuGroupInfo->streamId = taskInfo.streamId_;
    ccuGroupInfo->taskId = taskInfo.taskId_;
    ccuGroupInfo->dieId = info.dieId;
    ccuGroupInfo->instrId = info.instrId;
    ccuGroupInfo->missionId = info.missionId;
    ccuGroupInfo->reduceOpType = info.reduceOpType;
    ccuGroupInfo->inputDataType = info.inputDataType;
    ccuGroupInfo->outputDataType = info.outputDataType;
    ccuGroupInfo->dataSize = info.dataSize;
    std::copy(info.channelId, info.channelId + CCU_MAX_CHANNEL_NUM, ccuGroupInfo->channelId);
    std::copy(info.remoteRankId, info.remoteRankId + CCU_MAX_CHANNEL_NUM, ccuGroupInfo->remoteRankId);
    ReportAdditionInfo(reporterData);
}

void ProfilingHandler::DumpCcuGroupInfo([[maybe_unused]] const MsprofCcuGroupInfo& ccuGroupInfo) const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    // HCCL_INFOs migrated to ReportCcuInfoLog
}

void ProfilingHandler::GetCcuWaitSignalInfo(const TaskInfo& taskInfo, const CcuProfilingInfo& info) const
{
    uint64_t timestamp = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    MsprofAdditionalInfo reporterData{};
    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_CCU_WAIT_SIGNAL_INFO;
    reporterData.threadId = SalGetTid();
    reporterData.dataLen = sizeof(MsprofCcuWaitSignalInfo);
    reporterData.timeStamp = timestamp;
    auto* waitSignalInfo = reinterpret_cast<MsprofCcuWaitSignalInfo*>(reporterData.data);
    waitSignalInfo->version = 0;
    waitSignalInfo->itemId = GetProfHashId(info.name.c_str(), info.name.length());
    uint64_t groupName = GetProfHashId(taskInfo.dfxOpInfo_->op_.opTag.c_str(), taskInfo.dfxOpInfo_->op_.opTag.length());
    waitSignalInfo->groupName = groupName;
    if (taskInfo.dfxOpInfo_->isIndop_ == true) {
        waitSignalInfo->rankId = taskInfo.dfxOpInfo_->op_.myRank;
        waitSignalInfo->ranksize = taskInfo.dfxOpInfo_->rankSize_;
    } else {
        CommunicatorImpl* commImp = static_cast<CommunicatorImpl*>(taskInfo.dfxOpInfo_->comm_);
        waitSignalInfo->rankId = commImp->GetIdIndex();
        waitSignalInfo->ranksize = commImp->GetRankSize();
    }
    waitSignalInfo->workFlowMode = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    waitSignalInfo->streamId = taskInfo.streamId_;
    waitSignalInfo->taskId = taskInfo.taskId_;
    waitSignalInfo->dieId = info.dieId;
    waitSignalInfo->instrId = info.instrId;
    waitSignalInfo->missionId = info.missionId;
    waitSignalInfo->ckeId = info.ckeId;
    waitSignalInfo->mask = info.mask;
    std::copy(info.channelId, info.channelId + CCU_MAX_CHANNEL_NUM, waitSignalInfo->channelId);
    std::copy(info.remoteRankId, info.remoteRankId + CCU_MAX_CHANNEL_NUM, waitSignalInfo->remoteRankId);
    ReportAdditionInfo(reporterData);
}

void ProfilingHandler::ReportAclApi(
    uint32_t cmdType, uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId, bool cachedReq)
{
    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_ACL_LEVEL;
    reporterData.type = static_cast<int32_t>(cmdType) + MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE;
    reporterData.threadId = threadId;
    reporterData.beginTime = beginTime;
    reporterData.endTime = endTime;
    reporterData.itemId = cmdItemId;

    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cachedAclApiInfoMutex_);
        HCCL_INFO(
            "ReportAclApi MSPROF_REPORT_ACL_LEVEL type[%u] beginTime[%llu]", reporterData.type, reporterData.beginTime);
        cachedAclApiInfo_.push(reporterData);
    }
    HCCL_INFO(
        "[ProfilingHandler][ReportAclApi], reporterData data is: level[%u], type[%u], threadId[%u], "
        "beginTime[%llu], endTime[%llu], itemId[%llu]",
        reporterData.level, reporterData.type, reporterData.threadId, reporterData.beginTime, reporterData.endTime,
        reporterData.itemId);
    if (!enableHostApi_) {
        return;
    }

    s32 ret = DlProfFunction::GetInstance().dlMsprofReportApi(1, &reporterData);
    if (ret != 0) {
        THROW<InternalException>("Call dlMsprofReportApi failed, return[%d]", ret);
    }
}

void ProfilingHandler::ReportNodeApi(
    uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId, bool cachedReq)
{
    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    reporterData.threadId = threadId;
    reporterData.beginTime = beginTime;
    reporterData.endTime = endTime;
    reporterData.itemId = cmdItemId;

    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cachedTaskApiInfoMutex_);
        cachedTaskApiInfo_.push(reporterData);
    }
    HCCL_INFO(
        "[ProfilingHandler][ReportNodeApi], reporterData data is: level[%u], type[%u], threadId[%u], "
        "beginTime[%llu], endTime[%llu], itemId[%llu]",
        reporterData.level, reporterData.type, reporterData.threadId, reporterData.beginTime, reporterData.endTime,
        reporterData.itemId);
    if (!enableHostApi_) {
        return;
    }

    s32 ret = DlProfFunction::GetInstance().dlMsprofReportApi(1, &reporterData);
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportApi failed, return[%d]", ret);
    }
}

void ProfilingHandler::ReportNodeBasicInfo(uint64_t timeStamp, uint64_t cmdItemId, uint32_t threadId, bool cachedReq)
{
    MsprofCompactInfo reporterData{};
    reporterData.level = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    reporterData.threadId = threadId;
    reporterData.dataLen = sizeof(MsprofNodeBasicInfo);
    reporterData.timeStamp = timeStamp;
    reporterData.data.nodeBasicInfo.opName = cmdItemId;
    reporterData.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_HCCL;
    reporterData.data.nodeBasicInfo.opType = cmdItemId;
    reporterData.data.nodeBasicInfo.opFlag = 0;
    HCCL_INFO(
        "[ProfilingHandler][ReportNodeBasicInfo], reporterData data is: level[%u], type[%u], threadId[%u], "
        "dataLen[%u], taskType[%u], opFlag[%u]",
        reporterData.level, reporterData.type, reporterData.threadId, reporterData.dataLen,
        reporterData.data.nodeBasicInfo.taskType, reporterData.data.nodeBasicInfo.opFlag);
    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cacheNodeBasicInfoMutex_);
        cacheNodeBasicInfo_.push(reporterData);
    }
    if (!enableHcclL1_) {
        return;
    }
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportCompactInfo(1, &reporterData, sizeof(MsprofCompactInfo));
    HCCL_INFO("Call MsprofReportCompactInfo, return value[%d]", ret);
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportCompactInfo failed, return[%d]", ret);
    }
}

void ProfilingHandler::ReportHcclOpInfo(uint64_t timeStamp, const DfxOpInfo& opInfo, uint32_t threadId, bool cachedReq)
{
    MsprofCompactInfo reporterData{};
    reporterData.level = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE;
    reporterData.threadId = threadId;
    reporterData.dataLen = sizeof(MsprofHCCLOPInfo);
    reporterData.timeStamp = timeStamp;
    reporterData.data.hcclopInfo.relay = 0;
    reporterData.data.hcclopInfo.retry = 0;
    reporterData.data.hcclopInfo.dataType = opInfo.op_.dataType;
    reporterData.data.hcclopInfo.algType = cachedAlgTypeHashId_.load();
    uint64_t groupName = GetProfHashId(opInfo.op_.opTag.c_str(), opInfo.op_.opTag.length());
    reporterData.data.hcclopInfo.groupName = groupName;
    u32 ranksize{0};
    if (opInfo.isIndop_ == true) {
        ranksize = opInfo.rankSize_;
        reporterData.data.hcclopInfo.count = opInfo.op_.dataCount;
    } else {
        CommunicatorImpl* commImp = static_cast<CommunicatorImpl*>(opInfo.comm_);
        ranksize = commImp->GetRankSize();
        if (opInfo.op_.opType == OpType::ALLTOALLV) {
            u64 sendCount = 0;
            for (u64 i = 0; i < ranksize; i++) {
                sendCount += *(static_cast<const u64*>(opInfo.op_.all2AllVDataDes.sendCounts) + i);
            }
            reporterData.data.hcclopInfo.count = sendCount;
        } else if (opInfo.op_.opType == OpType::ALLTOALL) {
            reporterData.data.hcclopInfo.count = opInfo.op_.all2AllDataDes.sendCount;
        } else {
            reporterData.data.hcclopInfo.count = opInfo.op_.dataCount;
        }
    }
    if (cachedReq) {
        std::lock_guard<std::mutex> lock(cacheHcclOpInfoMutex_);
        cacheHcclOpInfo_.push(reporterData);
    }
    HCCL_INFO(
        "[ProfilingHandler][ReportHcclOpInfo], data is: level[%u], type[%u], threadId[%u], dataLen[%u], "
        "timeStamp[%llu], relay [%u], retry[%u], dataType[%s], algType[%llu], groupName[%llu], count[%llu]",
        reporterData.level, reporterData.type, reporterData.threadId, reporterData.dataLen, reporterData.timeStamp,
        reporterData.data.hcclopInfo.relay, reporterData.data.hcclopInfo.retry,
        DataTypeToSerialString(reporterData.data.hcclopInfo.dataType).c_str(), reporterData.data.hcclopInfo.algType,
        reporterData.data.hcclopInfo.groupName, reporterData.data.hcclopInfo.count);
    if (!enableHostApi_) {
        return;
    }
    s32 ret = DlProfFunction::GetInstance().dlMsprofReportCompactInfo(1, &reporterData, sizeof(MsprofCompactInfo));
    if (ret != 0) {
        THROW<InternalException>("[ProfilingHandler] Call dlMsprofReportCompactInfo failed, return[%d]", ret);
    }
}

void ProfilingHandler::ReportAdditionInfo(MsprofAdditionalInfo& reporterData) const
{
    s32 ret
        = DlProfFunction::GetInstance().dlMsprofReportAdditionalInfo(0, &reporterData, sizeof(MsprofAdditionalInfo));
    HCCL_INFO(
        "[ProfilingHandler][ReportAdditionInfo], level[%u], type[%u], threadId[%u], dataLen[%u], "
        "timeStamp[%llu], return value[%d]",
        reporterData.level, reporterData.type, reporterData.threadId, reporterData.dataLen, reporterData.timeStamp,
        ret);
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportAdditionalInfo failed, return[%d]", ret);
    }
}

int32_t ProfilingHandler::CommandHandle(uint32_t rtType, void* data, uint32_t len) const
{
    (void)len;
    if (data == nullptr || rtType != rtProfCtrlType_t::RT_PROF_CTRL_SWITCH) {
        HCCL_ERROR("[ProfilingHandler][CommandHandle] data is nullptr or rtType is invalid, rtType[%u]", rtType);
        return HCCL_E_PARA;
    }
    rtProfCommandHandle_t* profConfigParam = static_cast<rtProfCommandHandle_t*>(data);
    auto type = profConfigParam->type;
    auto profconfig = profConfigParam->profSwitch;
    HCCL_RUN_INFO(
        "[Profiling][CommandHandle] CommandHandle's rtType is %u. CommandHandle_switch type[%u], "
        "profconfig[%llu], deviceLogicId[%u]",
        rtType, type, profconfig, profConfigParam->devIdList[0]);
    switch (type) {
        case PROF_COMMANDHANDLE_TYPE_START:
            instance_.StartSubscribe(profconfig);
            break;
        case PROF_COMMANDHANDLE_TYPE_STOP:
            instance_.StopSubscribe();
            break;
        default:
            HCCL_RUN_INFO("[Profiling][CommandHandle] Unexpected behavior.");
    }
    return HCCL_SUCCESS;
}

void ProfilingHandler::StartSubscribe(uint64_t profconfig)
{
    HCCL_RUN_INFO("[Profiling][CommandHandle] profSwitch is[%llu]", profconfig);
    SetCachedCclTag();
    cachedAlgTypeHashId_ = GetProfHashId("AlgType::NHR", strlen("AlgType::NHR"));

    StartHostApiSubscribe();    // 上报AclApi和Mc2CommInfo的缓存数据
    StartTaskApiSubscribe();    // 上报TaskApi和NodeApi的缓存数据
    StartHostHcclOpSubscribe(); // 上报HcclOpInfo的缓存数据

    if ((profconfig & PROF_TASK_TIME_L1_MASK) != 0) {
        StartAdditionInfoSubscribe();
        StartCcuSubscribe();
    }
    HCCL_RUN_INFO("[Profiling][CommandHandle] profSwitch is[%llu]", profconfig);
}

void ProfilingHandler::StartHostApiSubscribe()
{
    enableHostApi_ = true;
    CallProfRegHostApi();
    ReportMc2AdditionInfo();
    ReportStoragedAclApi();
    HCCL_RUN_INFO("SetHostApiSubscribe:[%d]", enableHostApi_);
}

void ProfilingHandler::CallProfRegHostApi() const
{
    if (!enableHostApi_) {
        return;
    }
    auto& profFunction = DlProfFunction::GetInstance();
    for (auto i = 0; i < OpType::__COUNT__; ++i) {
        OpType type(static_cast<OpType::Value>(i));
        s32 ret = profFunction.dlMsprofRegTypeInfo(
            MSPROF_REPORT_ACL_LEVEL, static_cast<uint32_t>(type) + MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE,
            type.Describe().c_str());
        if (ret != 0) {
            THROW<InternalException>("Call MsprofRegTypeInfo fail, return[%d]", ret);
        }
    }
    for (auto i = 0; i < OpType::__COUNT__; ++i) {
        OpType type(static_cast<OpType::Value>(i));
        s32 ret = profFunction.dlMsprofRegTypeInfo(
            MSPROF_REPORT_NODE_LEVEL, static_cast<uint32_t>(type) + MSPROF_REPORT_NODE_HCCL_BASE_TYPE,
            type.Describe().c_str());
        if (ret != 0) {
            THROW<InternalException>("Call MsprofRegTypeInfo fail, return[%d]", ret);
        }
    }
    const std::string hcclType("hccl_op_info");
    s32 ret = profFunction.dlMsprofRegTypeInfo(
        MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE, hcclType.c_str());
    if (ret != 0) {
        THROW<InternalException>("Call MsprofRegTypeInfo fail, return[%d]", ret);
    }

    ret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(
        MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE, "mc2_comm_info");
    if (ret != 0) {
        THROW<InternalException>("[ProfilingHandler]Call MsprofRegTypeInfo fail, return[%d]", ret);
    }
}

void ProfilingHandler::ReportStoragedCompactInfo()
{
    std::lock_guard<std::mutex> lock(cacheHcclOpInfoMutex_);
    HCCL_INFO("[ReportStoragedCompactInfo] The size of the storageCompactInfo_ is [%zu]", cacheHcclOpInfo_.size());
    std::queue<MsprofCompactInfo> tempCompactInfo = cacheHcclOpInfo_;
    while (!tempCompactInfo.empty()) {
        MsprofCompactInfo reportData = tempCompactInfo.front();
        tempCompactInfo.pop();
        s32 ret = DlProfFunction::GetInstance().dlMsprofReportCompactInfo(0, &reportData, sizeof(MsprofCompactInfo));
        if (ret != 0) {
            THROW<InternalException>("Call MsprofReportCompactInfo failed, return[%d]", ret);
        }
    }
}

void ProfilingHandler::ReportStoragedNodeBasicInfo()
{
    std::lock_guard<std::mutex> lock(cacheNodeBasicInfoMutex_);
    HCCL_INFO("[ReportStoragedNodeBasicInfo] The size of the cacheNodeBasicInfo_ is [%zu]", cacheNodeBasicInfo_.size());
    std::queue<MsprofCompactInfo> tempNodeBasicInfo = cacheNodeBasicInfo_;
    while (!tempNodeBasicInfo.empty()) {
        MsprofCompactInfo reportData = tempNodeBasicInfo.front();
        tempNodeBasicInfo.pop();
        s32 ret = DlProfFunction::GetInstance().dlMsprofReportCompactInfo(0, &reportData, sizeof(MsprofCompactInfo));
        if (ret != 0) {
            THROW<InternalException>("Call MsprofReportCompactInfo failed, return[%d]", ret);
        }
    }
}

void ProfilingHandler::ReportMc2AdditionInfo()
{
    std::lock_guard<std::mutex> lock(cacheHcclAdditionInfoMutex_);
    HCCL_INFO("[ReportMc2AdditionInfo] The size of the storageCompactInfo_ is [%zu]", cacheHcclAdditionInfo_.size());
    std::queue<MsprofAdditionalInfo> tempCompactInfo = cacheHcclAdditionInfo_;
    while (!tempCompactInfo.empty()) {
        MsprofAdditionalInfo reportData = tempCompactInfo.front();
        tempCompactInfo.pop();
        s32 ret
            = DlProfFunction::GetInstance().dlMsprofReportAdditionalInfo(1, &reportData, sizeof(MsprofAdditionalInfo));
        if (ret != 0) {
            THROW<InternalException>("Call MsprofReportAdditionalInfo failed, return[%d]", ret);
        }
    }
}

void ProfilingHandler::StartTaskApiSubscribe()
{
    enableHcclNode_ = true;
    CallProfRegTaskTypeApi();
    ReportStoragedTaskApi();
    HCCL_INFO("SetTaskApiSubscribe:[%d]", enableHcclNode_);
}

void ProfilingHandler::CallProfRegTaskTypeApi() const
{
    if (!enableHcclNode_) {
        HCCL_INFO("[ProfilingHandler] enableHostApi_ is false.");
        return;
    }
    const std::string hcclType("hccl_info");
    s32 sret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(
        MSPROF_REPORT_HCCL_NODE_LEVEL, static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO), hcclType.c_str());
    if (sret != 0) {
        THROW<InternalException>("Call MsprofRegTypeInfo fail, return[%d]", sret);
    }

    const std::string dpuhcclType("dpu_hccl_info");
    sret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(
        MSPROF_REPORT_HCCL_NODE_LEVEL, static_cast<uint32_t>(ProfTaskType::TASK_DPU_HCCL_INFO), dpuhcclType.c_str());
    if (sret != 0) {
        THROW<InternalException>("Call MsprofRegTypeInfo fail, return[%d]", sret);
    }
    const std::vector<std::pair<uint32_t, std::string>> taskTypes
        = {{MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE, "context_id_info"}};
    const std::vector<std::pair<uint32_t, std::string>> taskOtherTypes
        = {{MSPROF_REPORT_NODE_BASIC_INFO_TYPE, "node_basic_info"}};

    for (auto& it : taskTypes) {
        s32 ret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(
            MSPROF_REPORT_HCCL_NODE_LEVEL, it.first, it.second.c_str());
        if (ret != 0) {
            THROW<InternalException>("Call dlMsprofRegTypeInfo failed, return[%d]", ret);
        }
    }
    for (auto& it : taskOtherTypes) {
        s32 ret
            = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, it.first, it.second.c_str());
        if (ret != 0) {
            THROW<InternalException>("Call dlMsprofRegTypeInfo failed, return[%d]", ret);
        }
    }
}

void ProfilingHandler::ReportStoragedTaskApi()
{
    std::lock_guard<std::mutex> lock(cachedTaskApiInfoMutex_);
    HCCL_INFO("[ReportStoragedTaskApi] taskApiQueueSize is [%zu]", cachedTaskApiInfo_.size());
    if (!cachedTaskApiInfo_.empty()) {
        std::queue<MsprofApi> tempTaskApi = cachedTaskApiInfo_;
        while (!tempTaskApi.empty()) {
            MsprofApi reportData = tempTaskApi.front();
            tempTaskApi.pop();
            s32 ret = DlProfFunction::GetInstance().dlMsprofReportApi(0, &reportData);
            if (ret != 0) {
                THROW<InternalException>("Call dlMsprofReportApi failed, return[%d]", ret);
            }
        }
    }
}

void ProfilingHandler::ReportStoragedAclApi()
{
    std::lock_guard<std::mutex> lock(cachedAclApiInfoMutex_);
    HCCL_INFO("[ReportStoragedAclApi] aclApiQueueSize is [%u]", cachedAclApiInfo_.size());
    if (!cachedAclApiInfo_.empty()) {
        std::queue<MsprofApi> tempAclApi = cachedAclApiInfo_;
        while (!tempAclApi.empty()) {
            MsprofApi reportData = tempAclApi.front();
            tempAclApi.pop();
            s32 ret = DlProfFunction::GetInstance().dlMsprofReportApi(0, &reportData);
            if (ret != 0) {
                THROW<InternalException>("Call dlMsprofReportApi failed, return[%d]", ret);
            }
        }
    }
}

void ProfilingHandler::StartHostHcclOpSubscribe()
{
    enableHcclNode_ = true;
    enableHcclL0_ = true;
    CallProfRegHcclOpApi();
    ReportStoragedCompactInfo();
    HCCL_RUN_INFO("StartHostHcclOpSubscribe:enableHcclNode_[%d],enableHcclL0_[%d]", enableHcclNode_, enableHcclL0_);
}

void ProfilingHandler::CallProfRegHcclOpApi() const
{
    if (enableHcclL0_ == false) {
        HCCL_INFO("[ProfilingHandler] enableHcclNode_ is false.");
        return;
    }
    for (auto i = 0; i < OpType::__COUNT__; ++i) {
        OpType type(static_cast<OpType::Value>(i));
        s32 ret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(
            MSPROF_REPORT_HCCL_NODE_LEVEL, static_cast<uint32_t>(type) + MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE,
            type.Describe().c_str());
        if (ret != 0) {
            THROW<InternalException>("[ProfilingHandler]Call MsprofReportApi fail, return[%d]", ret);
        }
    }
}

void ProfilingHandler::StartAdditionInfoSubscribe()
{
    enableHcclL1_ = true;
    ReportStoragedNodeBasicInfo();
    ReportStoragedAdditionInfo();
    HCCL_RUN_INFO("StartAdditionInfoSubscribe:enableHcclL1_[%d]", enableHcclL1_);
}

void ProfilingHandler::ReportStoragedAdditionInfoLog() const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    if (cacheTaskInfos_.empty()) {
        return;
    }
    for (const auto& taskInfo : cacheTaskInfos_) {
        HCCL_INFO("ReportStoragedAdditionInfoLog, taskType[%u]", GetTaskTypeValue(taskInfo.taskParam_.taskType));
    }
}

void ProfilingHandler::ReportStoragedAdditionInfo()
{
    std::lock_guard<std::mutex> lock(cacheTaskInfosMutex_);
    ReportStoragedAdditionInfoLog();
    if (cacheTaskInfos_.empty()) {
        HCCL_INFO("[ProfilingHandler]ReportStoragedAdditionInfo cacheTaskInfos_ is empty.");
        return;
    }
    for (auto& taskInfo : cacheTaskInfos_) {
        MsprofAdditionalInfo reporterData{};
        FillTaskAdditionInfo(taskInfo, reporterData);
        CallAdditionInfo(reporterData);
    }
}

void ProfilingHandler::StartCcuSubscribe()
{
    enableHcclNode_ = true;
    enableHcclL1_ = true;
    HCCL_INFO("ProfilingHandler StartCcuSubscribe");
    const std::vector<std::pair<uint32_t, std::string>> ccuInfoTypes
        = {{MSPROF_REPORT_CCU_TASK_INFO, "ccu_task_info"},
           {MSPROF_REPORT_CCU_WAIT_SIGNAL_INFO, "ccu_wait_signal_info"},
           {MSPROF_REPORT_CCU_GROUP_INFO, "ccu_group_info"}};
    for (auto& it : ccuInfoTypes) {
        s32 ret = DlProfFunction::GetInstance().dlMsprofRegTypeInfo(
            MSPROF_REPORT_HCCL_NODE_LEVEL, it.first, it.second.c_str());
        if (ret != 0) {
            THROW<InternalException>("Call dlMsprofRegTypeInfo failed, return[%d]", ret);
        }
    }
    std::lock_guard<std::mutex> lock(cacheTaskInfosMutex_);
    if (cacheTaskInfos_.empty()) {
        HCCL_INFO("[ProfilingHandler]StartL2Subscribe cacheTaskInfos_ is empty.");
        return;
    }
    for (auto& taskInfo : cacheTaskInfos_) {
        ReportCcuInfo(taskInfo);
    }
}

void ProfilingHandler::ProfilingHandler::StopSubscribe()
{
    enableHostApi_ = false;
    enableHcclNode_ = false;
    enableHcclL0_ = false;
    enableHcclL1_ = false;
    HCCL_RUN_INFO("[ProfilingHandler]StopSubscribe.");
}

bool ProfilingHandler::GetHostApiState() const { return enableHostApi_; }
bool ProfilingHandler::GetHcclNodeState() const { return enableHcclNode_; }
bool ProfilingHandler::GetHcclL0State() const { return enableHcclL0_; }

bool ProfilingHandler::GetHcclL1State() const { return enableHcclL1_; }

uint64_t ProfilingHandler::GetProfHashId(const char* name, uint32_t len) const
{
    if (name == nullptr || len == 0) {
        HCCL_WARNING("HashData is empty. name:%s, len:%u", name, len);
        return INVALID_U64;
    }
    if (DlProfFunction::GetInstance().dlMsprofStr2Id == nullptr) {
        HCCL_WARNING("HashFunc is nullPtr");
        return INVALID_U64;
    }
    return DlProfFunction::GetInstance().dlMsprofStr2Id(name, len);
}

void ProfilingHandler::ReportHcclMC2CommInfoLog(
    [[maybe_unused]] const Stream& kfcStream, const Stream& stream, const std::vector<Stream*>& aicpuStreams,
    [[maybe_unused]] const std::string& id, [[maybe_unused]] RankId myRank, [[maybe_unused]] u32 rankSize,
    [[maybe_unused]] RankId rankInParentComm) const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    if (aicpuStreams.empty()) {
        HCCL_INFO("only exist main stream, streamId(sqId):%u", stream.GetSqId());
        return;
    }
    uint32_t reportId = 0;
    for (uint32_t streamIndex = 0; streamIndex < aicpuStreams.size(); streamIndex++) {
        HCCL_INFO(
            "streamIndex:%u, reportId:%u, streamId(sqId):%u", streamIndex, reportId,
            aicpuStreams[streamIndex]->GetSqId());
        reportId++;
    }
}

void ProfilingHandler::ReportHcclMC2CommInfo(
    const Stream& kfcStream, const Stream& stream, const std::vector<Stream*>& aicpuStreams, const std::string& id,
    RankId myRank, u32 rankSize, RankId rankInParentComm)
{
    ReportHcclMC2CommInfoLog(kfcStream, stream, aicpuStreams, id, myRank, rankSize, rankInParentComm);
    ProfilingDeviceCommResInfo hcclMc2Info;
    hcclMc2Info.groupName = GetProfHashId(id.c_str(), id.length());
    hcclMc2Info.rankSize = rankSize;
    hcclMc2Info.rankId = myRank;
    hcclMc2Info.usrRankId = rankInParentComm;
    hcclMc2Info.aicpuKfcStreamId = static_cast<uint32_t>(kfcStream.GetSqId());
    hcclMc2Info.reserve = 0;
    const uint32_t ONCE_REPORT_STREAM_NUM_MAX = 8;
    for (uint32_t streamIndex = 0, reportId = 0; streamIndex < aicpuStreams.size(); streamIndex++) {
        hcclMc2Info.commStreamIds[reportId++] = aicpuStreams[streamIndex]->GetSqId();
        if (reportId == ONCE_REPORT_STREAM_NUM_MAX) {
            hcclMc2Info.commStreamSize = reportId;
            ReportMc2AdditionInfo(
                DlProfFunction::GetInstance().dlMsprofSysCycleTime(), &hcclMc2Info, sizeof(hcclMc2Info));
            reportId = 0;
        }
        if (streamIndex == (aicpuStreams.size() - 1)) {
            hcclMc2Info.commStreamIds[reportId++] = stream.GetSqId();
            hcclMc2Info.commStreamSize = reportId;
            ReportMc2AdditionInfo(
                DlProfFunction::GetInstance().dlMsprofSysCycleTime(), &hcclMc2Info, sizeof(hcclMc2Info));
            reportId = 0;
        }
    }
    if (aicpuStreams.empty()) {
        hcclMc2Info.commStreamIds[0] = stream.GetSqId();
        hcclMc2Info.commStreamSize = 1;
        ReportMc2AdditionInfo(DlProfFunction::GetInstance().dlMsprofSysCycleTime(), &hcclMc2Info, sizeof(hcclMc2Info));
    }
}

void ProfilingHandler::ReportHcclMC2CommInfoLog(
    [[maybe_unused]] const u32 kfcStreamId, const std::vector<u32>& aicpuStreamsId, const std::string& id,
    [[maybe_unused]] RankId myRank, [[maybe_unused]] u32 rankSize, [[maybe_unused]] RankId rankInParentComm) const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    uint64_t groupName = GetProfHashId(id.c_str(), id.length());
    uint32_t reportId = 0;
    for (uint32_t streamIndex = 0; streamIndex < aicpuStreamsId.size(); streamIndex++) {
        HCCL_INFO(
            "streamIndex:[%u], reportId:[%u], streamId:[%u] id [%s] hcclMC2Info.groupName:[%llu]", streamIndex,
            reportId, aicpuStreamsId[streamIndex], id.c_str(), groupName);
        reportId++;
    }
}

void ProfilingHandler::ReportHcclMC2CommInfo(
    const u32 kfcStreamId, const std::vector<u32>& aicpuStreamsId, const std::string& id, RankId myRank, u32 rankSize,
    RankId rankInParentComm)
{
    ReportHcclMC2CommInfoLog(kfcStreamId, aicpuStreamsId, id, myRank, rankSize, rankInParentComm);
    ProfilingDeviceCommResInfo hcclMc2Info;
    hcclMc2Info.groupName = GetProfHashId(id.c_str(), id.length());
    hcclMc2Info.rankSize = rankSize;
    hcclMc2Info.rankId = myRank;
    hcclMc2Info.usrRankId = rankInParentComm;
    hcclMc2Info.aicpuKfcStreamId = static_cast<uint32_t>(kfcStreamId);
    hcclMc2Info.reserve = 0;

    const uint32_t ONCE_REPORT_STREAM_NUM_MAX = 8;
    uint32_t reportId = 0;
    for (uint32_t streamIndex = 0; streamIndex < aicpuStreamsId.size(); streamIndex++) {
        hcclMc2Info.commStreamIds[reportId++] = aicpuStreamsId[streamIndex];
        if (reportId == ONCE_REPORT_STREAM_NUM_MAX) {
            hcclMc2Info.commStreamSize = reportId;
            ReportMc2AdditionInfo(
                DlProfFunction::GetInstance().dlMsprofSysCycleTime(), &hcclMc2Info, sizeof(hcclMc2Info));
            reportId = 0;
        }
    }
    if (reportId > 0) {
        hcclMc2Info.commStreamSize = reportId;
        ReportMc2AdditionInfo(DlProfFunction::GetInstance().dlMsprofSysCycleTime(), &hcclMc2Info, sizeof(hcclMc2Info));
        reportId = 0;
    }
}
void ProfilingHandler::ReportMc2AdditionInfo(uint64_t timeStamp, const void* data, int len)
{
    MsprofAdditionalInfo reporterData{};
    reporterData.level = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE;
    reporterData.threadId = SalGetTid();
    reporterData.dataLen = len;
    reporterData.timeStamp = timeStamp;
    s32 sret = memcpy_s(reporterData.data, sizeof(reporterData.data), data, len);
    if (sret != EOK) {
        THROW<InternalException>("Call memcpy_s failed, errorno[%d]", sret);
    }
    HCCL_INFO(
        "[ProfilingHandler][ReportMc2CommInfo], level [%u], type[%u], threadId[%u], dataLen[%u], timeStamp[%llu]",
        reporterData.level, reporterData.type, reporterData.threadId, reporterData.dataLen, reporterData.timeStamp);
    if (!enableHostApi_) {
        std::lock_guard<std::mutex> lock(cacheHcclAdditionInfoMutex_);
        cacheHcclAdditionInfo_.push(reporterData);
        return;
    }
    s32 ret
        = DlProfFunction::GetInstance().dlMsprofReportAdditionalInfo(1, &reporterData, sizeof(MsprofAdditionalInfo));
    HCCL_INFO("Call MsprofReportAdditionalInfo, return value[%d]", ret);
    if (ret != 0) {
        THROW<InternalException>("Call MsprofReportAdditionalInfo failed, return[%d]", ret);
    }
}

} // namespace Hccl
