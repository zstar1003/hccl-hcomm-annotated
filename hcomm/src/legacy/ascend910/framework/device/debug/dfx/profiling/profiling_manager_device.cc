/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_manager_device.h"
#include "aicpu_schedule/aicpu_context.h"
#include "prof_common.h"
#include "log.h"
#include "sal_pub.h"
#include "common/aicpu_sqe_context.h"
#include "common/aicpu_hccl_common.h"
#include "dlprof_function.h"

namespace {
static constexpr u32 aging = 1;
}
namespace dfx {

std::mutex ProfilingManager::streamMutex_;
std::unordered_map<std::string, ProfCommInfo> ProfilingManager::tagOpInfoMap_;
std::unordered_map<s32, std::string> ProfilingManager::streamToTagMap_;

std::mutex ProfilingManager::startReportSqeIdxMutex_;
std::unordered_map<s32, u32> ProfilingManager::streamToSqeIdxMap_;

bool ProfilingManager::isL0Open_ = false;
bool ProfilingManager::isL1Open_ = false;

constexpr std::uint32_t HCCLINFO_REPORT_BATCH_NUM = 2;

bool ProfilingManager::IsProfOn(uint64_t feature)
{
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        if (AdprofCheckFeatureIsOn == nullptr) {
            return false;
        }
        return AdprofCheckFeatureIsOn(feature) > 0;
    } else {
        if (feature == ADPROF_TASK_TIME_L1) {
            return isL1Open_;
        } else if (feature == ADPROF_TASK_TIME_L0) {
            return isL0Open_;
        }
    }
    return false;
}

bool ProfilingManager::IsL1fromOffToOn()
{
    if (((!GetProfL1State()) && ProfilingManager::IsProfL1On())) {
        HCCL_INFO("Profiling L1 switch from off to on.");
        return true;
    }
    return false;
}

bool ProfilingManager::IsProfL1On()
{
    if (IsProfOn(ADPROF_TASK_TIME_L1)) {
        isL1Open_ = true;
        return true;
    }
    isL1Open_ = false;
    return false;
}

bool ProfilingManager::IsProfL0On()
{
    if (IsProfOn(ADPROF_TASK_TIME_L0)) {
        isL0Open_ = true;
        return true;
    }
    isL0Open_ = false;
    return false;
}

void ProfilingManager::SetProL1On(bool val)
{
    HCCL_INFO("[%s] val = [%d]", __func__, val);
    isL1Open_ = val;
}

void ProfilingManager::SetProL0On(bool val)
{
    HCCL_INFO("[%s] val = [%d]", __func__, val);
    isL0Open_ = val;
}

bool ProfilingManager::GetProfL0State()
{
    if (!isL0Open_) {
        return false;
    }
    return true;
}

bool ProfilingManager::GetProfL1State()
{
    if (!isL1Open_) {
        return false;
    }
    return true;
}

HcclResult ProfilingManager::CallMsprofReportAdditionInfo(uint32_t type, uint64_t timeStamp, const void* data, int len)
{
    MsprofAdditionalInfo reporterData{};
    reporterData.level = MSPROF_REPORT_AICPU_LEVEL;
    reporterData.type = type;
    reporterData.threadId = SalGetTid();
    reporterData.dataLen = len;
    reporterData.timeStamp = timeStamp;
    s32 sret = memcpy_s(reporterData.data, sizeof(reporterData.data), data, len);
    CHK_PRT_RET(
        sret != EOK,
        HCCL_ERROR(
            "memcpy failed. errorno:[%d] level:[%hu] type:[%u] threadId:[%u] sizeof_data[%zu] len:[%u] "
            "timeStamp:[%llu]",
            sret, reporterData.level, type, reporterData.threadId, sizeof(reporterData.data), len, timeStamp),
        HCCL_E_MEMORY);
    HCCL_DEBUG("CallMsprofReportAdditionInfo, AdditionInfoType[%u]", type);
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        CHK_PTR_NULL(AdprofReportAdditionalInfo);
        int32_t ret = AdprofReportAdditionalInfo(aging, &reporterData, sizeof(MsprofAdditionalInfo));
        CHK_PRT_RET(ret != 0, HCCL_ERROR("AdprofReportAdditionalInfo failed. ret = [%d]", ret), HCCL_E_INTERNAL);
    } else {
        CHK_PTR_NULL(MsprofReportAdditionalInfo);
        int32_t ret = MsprofReportAdditionalInfo(aging, &reporterData, sizeof(MsprofAdditionalInfo));
        CHK_PRT_RET(ret != 0, HCCL_ERROR("MsprofReportAdditionalInfo failed. ret = [%d]", ret), HCCL_E_INTERNAL);
    }

    HCCL_DEBUG("CallMsprofReportAdditionInfo with additionInfoType[%u] successfully", type);
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::TaskInfo2Addition(const void* data, int len, MsprofAdditionalInfo& reporterData)
{
    reporterData.level = MSPROF_REPORT_AICPU_LEVEL;
    reporterData.type = MSPROF_REPORT_AICPU_MC2_BATCH_HCCL_INFO;
    reporterData.threadId = SalGetTid();
    reporterData.dataLen = len;
    reporterData.timeStamp = 0;
    s32 sret = memcpy_s(reporterData.data, sizeof(reporterData.data), data, len);
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("memcpy failed. errorno[%d]:", sret), HCCL_E_MEMORY);
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::ReportTaskInfo(s32 streamId, void* ctxPtr)
{
    if (!GetProfL1State()) {
        return HCCL_SUCCESS;
    }
    ProfCommInfo profInfo;
    // 通信域中获取rankId, groupHashId等信息
    CHK_RET(GetProfInfoByStreamId(streamId, profInfo));
    CHK_PTR_NULL(ctxPtr);
    hccl::HcclSqeContext* sqeContext = reinterpret_cast<hccl::HcclSqeContext*>(ctxPtr);
    hccl::SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    CHK_PTR_NULL(sqeContextBuffer);
    u32 startSqeIdx = GetStartReportSqeIdx(streamId);
    HCCL_INFO(
        "[ReportTaskInfo] Rank:%u, stream:%d, sqeNum:%u, startSqeIdx:%u, curSqeTailIdx: %u", profInfo.rankId, streamId,
        sqeContextBuffer->tailSqeIdx - startSqeIdx, startSqeIdx, sqeContextBuffer->tailSqeIdx);
    MsprofAicpuHcclTaskInfo taskInfos[HCCLINFO_REPORT_BATCH_NUM] = {};
    auto endIdx = static_cast<uint32_t>(sqeContextBuffer->tailSqeIdx);
    bool isSupportBatchReport
        = (AdprofReportBatchAdditionalInfo != nullptr || MsprofReportBatchAdditionalInfo != nullptr);
    HCCL_INFO(
        "AdprofReportBatchAdditionalInfo != nullptr || MsprofReportBatchAdditionalInfo != nullptr: %s",
        isSupportBatchReport ? "true" : "false");
    constexpr int32_t MAX_BATCH_REPORT_NUM
        = 512; // 最大支持批量上报的MsprofAdditionalInfo个数, 需要与接口实现侧保持一致
    MsprofAdditionalInfo addInfoVec[MAX_BATCH_REPORT_NUM] = {};
    uint32_t addInfoIndx = 0;
    for (uint32_t idx = startSqeIdx, batchId = 0; idx < endIdx; ++idx) {
        // 获取SqeInfo
        SqeInfo sqeInfo{};
        SqeContextUtils::QuerySqeInfo(
            sqeContextBuffer->localBuff + idx * hccl::HCCL_SQE_SIZE, sqeContextBuffer->sqeType[idx],
            sqeContextBuffer->addInfo[idx], &sqeInfo);
        sqeInfo.remoteRank = sqeContextBuffer->dfxInfo[idx].remoteRank;
        // 转换为MsprofAicpuHcclTaskInfo
        auto& taskInfo = taskInfos[batchId++];
        dfx::ProfilingExtendInfoHelper::InitHcclInfo(taskInfo);
        CommInfo2HcclInfo(profInfo, taskInfo);
        dfx::ProfilingExtendInfoHelper::SqeInfo2MsprofAicpuMC2HcclInfo(sqeInfo, taskInfo);
        taskInfo.timeStamp = sqeContextBuffer->profTimestap[idx]; // 时间戳
        DumpHcclInfo(taskInfo, batchId, idx);
        // 上报信息
        if (batchId == HCCLINFO_REPORT_BATCH_NUM || idx == (endIdx - 1)) {
            if (!isSupportBatchReport) {
                CHK_PRT(dfx::ProfilingManager::CallMsprofReportAdditionInfo(
                    MSPROF_REPORT_AICPU_MC2_BATCH_HCCL_INFO, 0, taskInfos, sizeof(MsprofAicpuHcclTaskInfo) * batchId));
            } else {
                CHK_PRT(
                    TaskInfo2Addition(taskInfos, sizeof(MsprofAicpuHcclTaskInfo) * batchId, addInfoVec[addInfoIndx++]));
                if (addInfoIndx == MAX_BATCH_REPORT_NUM || idx == (endIdx - 1)) {
                    if (MsprofReportBatchAdditionalInfo == nullptr) {
                        CHK_PRT_RET(
                            AdprofReportBatchAdditionalInfo(
                                aging, addInfoVec, addInfoIndx * sizeof(MsprofAdditionalInfo)),
                            HCCL_ERROR("AdprofReportBatchAdditionalInfo failed"), HCCL_E_INTERNAL);
                        addInfoIndx = 0; // 后面直接覆盖就行不需要清零;
                    } else {
                        CHK_PRT_RET(
                            MsprofReportBatchAdditionalInfo(
                                aging, addInfoVec, addInfoIndx * sizeof(MsprofAdditionalInfo)),
                            HCCL_ERROR("MsprofReportBatchAdditionalInfo failed"), HCCL_E_INTERNAL);
                        addInfoIndx = 0; // 后面直接覆盖就行不需要清零;
                    }
                }
            }
            batchId = 0;
            CHK_SAFETY_FUNC_RET(memset_s(taskInfos, sizeof(taskInfos), 0, sizeof(taskInfos)));
        }
    }
    CHK_RET(UpdateStartReportSqeIdx(streamId, sqeContextBuffer->tailSqeIdx));
    return HCCL_SUCCESS;
}

void ProfilingManager::DumpHcclInfo(const MsprofAicpuHcclTaskInfo& taskInfo, u32 batchId, u32 idx)
{
    HCCL_DEBUG(
        "[ReportTaskInfo] batchId:%u, idx:%u, itemId:%llu, groupName:%llu, localRank:%u, remoteRank:%u, "
        "rankSize:%u, timeStamp:%llu, srcAddr:0x%llx, dstAddr:0x%llx, dataSize:%llu, taskId:%u, streamId:%u, "
        "planeID:%u,"
        "opType:%u, dataType:%u, linkType:%u, transportType:%u, rdmaType:%u, role:%u",
        batchId, idx, taskInfo.itemId, taskInfo.groupName, taskInfo.localRank, taskInfo.remoteRank, taskInfo.rankSize,
        taskInfo.timeStamp, taskInfo.srcAddr, taskInfo.dstAddr, taskInfo.dataSize, taskInfo.taskId, taskInfo.streamId,
        taskInfo.planeID, taskInfo.opType, taskInfo.dataType, taskInfo.linkType, taskInfo.transportType,
        taskInfo.rdmaType, taskInfo.role);
}

void ProfilingManager::CommInfo2HcclInfo(const dfx::ProfCommInfo& profInfo, MsprofAicpuHcclTaskInfo& taskInfo)
{
    taskInfo.groupName = profInfo.groupNameHashId;
    taskInfo.localRank = profInfo.rankId;
    taskInfo.rankSize = profInfo.rankNum;
}

HcclResult ProfilingManager::ReportHcclOpInfo(MsprofAicpuHCCLOPInfo& hcclOpInfo, std::string& algTypeStr)
{
    if (!GetProfL0State()) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(aicpu::GetTaskAndStreamId);
    uint64_t taskId = 0U;
    uint32_t streamId = 0;
    if (AicpuGetStreamId == nullptr || AicpuGetTaskId == nullptr) {
        CHK_PTR_NULL(aicpu::GetTaskAndStreamId);
        CHK_PRT_RET(
            aicpu::GetTaskAndStreamId(taskId, streamId) != aicpu::status_t::AICPU_ERROR_NONE,
            HCCL_ERROR("Failed to get task id and stream id."), HCCL_E_PARA);
    } else {
        streamId = AicpuGetStreamId();
        taskId = AicpuGetTaskId();
    }

    HCCL_INFO("[ProfilingManager] ReportHcclOpInfo streamId = %u, taskId = %llu", streamId, taskId);

    hcclOpInfo.algType = GetProfHashId(algTypeStr.c_str(), algTypeStr.length());
    hcclOpInfo.taskId = taskId;
    hcclOpInfo.streamId = streamId;
    HCCL_INFO(
        "[ReportHcclOpInfo] relay:%u, retry:%u, dataType:%u, algType:%llu, count:%llu, groupHashId:%llu",
        hcclOpInfo.relay, hcclOpInfo.retry, hcclOpInfo.dataType, hcclOpInfo.algType, hcclOpInfo.count,
        hcclOpInfo.groupName);
    CHK_PRT(dfx::ProfilingManager::CallMsprofReportAdditionInfo(
        MSPROF_REPORT_AICPU_HCCL_OP_INFO, ProfGetCurCpuTimestamp(), &hcclOpInfo, sizeof(MsprofAicpuHCCLOPInfo)));
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::ReportMainStreamTask(hccl::Stream& stream, uint16_t taskId, uint16_t type)
{
    if (!GetProfL0State()) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(aicpu::GetTaskAndStreamId);
    uint64_t aicpuKernelTaskId = 0U;
    uint32_t aicpuKernelStreamId = 0;
    if (AicpuGetStreamId == nullptr || AicpuGetTaskId == nullptr) {
        CHK_PTR_NULL(aicpu::GetTaskAndStreamId);
        CHK_PRT_RET(
            aicpu::GetTaskAndStreamId(aicpuKernelTaskId, aicpuKernelStreamId) != aicpu::status_t::AICPU_ERROR_NONE,
            HCCL_ERROR("Failed to get task id and stream id."), HCCL_E_PARA);
    } else {
        aicpuKernelTaskId = AicpuGetTaskId();
        aicpuKernelStreamId = AicpuGetStreamId();
    }
    HCCL_INFO(
        "[ReportMainStreamTask] aicpuKernelStreamId = %u, aicpuKernelTaskId = %llu", aicpuKernelStreamId,
        aicpuKernelTaskId);
    MsprofAicpuHcclMainStreamTask flagtask{};
    flagtask.streamId = stream.id();
    flagtask.taskId = taskId;
    flagtask.type = type;
    flagtask.aicpuStreamId = aicpuKernelStreamId;
    flagtask.aicpuTaskId = aicpuKernelTaskId;
    HCCL_INFO(
        "[ReportMainStreamTask] streamId:%u, taskId:%u, type:%u", flagtask.streamId, flagtask.taskId, flagtask.type);

    CHK_PRT(dfx::ProfilingManager::CallMsprofReportAdditionInfo(
        MSPROF_REPORT_AICPU_HCCL_FLAG_TASK, ProfGetCurCpuTimestamp(), &flagtask,
        sizeof(MsprofAicpuHcclMainStreamTask)));

    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::ReportFilpTask(s32 streamId, uint16_t taskId, uint32_t flipNum)
{
    if (!GetProfL0State()) {
        return HCCL_SUCCESS;
    }
    MsporfAicpuFlipTask flipTaskInfo{};
    flipTaskInfo.streamId = streamId;
    flipTaskInfo.taskId = taskId;
    flipTaskInfo.flipNum = flipNum;
    HCCL_INFO(
        "[ReportFlipTask] streamId:%u, taskId:%u, flipNum:%u", flipTaskInfo.streamId, flipTaskInfo.taskId,
        flipTaskInfo.flipNum);
    CHK_PRT(dfx::ProfilingManager::CallMsprofReportAdditionInfo(
        MSPROF_REPORT_AICPU_FILP_TASK, ProfGetCurCpuTimestamp(), &flipTaskInfo, sizeof(MsporfAicpuFlipTask)));

    return HCCL_SUCCESS;
}

uint64_t ProfilingManager::GetProfHashId(const char* name, uint32_t len)
{
    if (name == nullptr || len == 0) {
        HCCL_WARNING("HashData is empty.");
        return INVALID_U64;
    }
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        CHK_PRT_RET((AdprofGetHashId == nullptr), HCCL_WARNING("AdprofGetHashId is null, just return"), INVALID_U64);
        return AdprofGetHashId(name, len);
    } else {
        CHK_PRT_RET((MsprofStr2Id == nullptr), HCCL_WARNING("MsprofStr2Id is null, just return"), INVALID_U64);
        return MsprofStr2Id(name, len);
    }
}

uint32_t ProfilingManager::GetStartReportSqeIdx(s32 streamId)
{
    std::unique_lock<std::mutex> lock(startReportSqeIdxMutex_);
    u32 lastSqeTailIdx = 0;
    auto iter = streamToSqeIdxMap_.find(streamId);
    if (iter == streamToSqeIdxMap_.end()) {
        HCCL_INFO("[GetProfInfoByStreamId]streamId:%d is not found", streamId);
        streamToSqeIdxMap_.insert({streamId, 0});
    } else {
        lastSqeTailIdx = iter->second;
    }
    return lastSqeTailIdx;
}

HcclResult ProfilingManager::UpdateStartReportSqeIdx(s32 streamId, u32 newSqeTailIdx)
{
    std::unique_lock<std::mutex> lock(startReportSqeIdxMutex_);
    auto iter = streamToSqeIdxMap_.find(streamId);
    if (iter == streamToSqeIdxMap_.end()) {
        streamToSqeIdxMap_.insert({streamId, newSqeTailIdx});
        HCCL_INFO("[UpdateStartReportSqeIdx]streamId:%d is not found, newSqeTailIdx:%u", streamId, newSqeTailIdx);
    } else {
        // 到2048时，更新成0;
        newSqeTailIdx = (newSqeTailIdx == hccl::HCCL_SQE_MAX_CNT) ? 0 : newSqeTailIdx;
        HCCL_INFO(
            "[UpdateStartReportSqeIdx] streamId:%d, lastSqeTailIdx:%u, newSqeTailIdx:%u", streamId, iter->second,
            newSqeTailIdx);
        iter->second = newSqeTailIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::GetProfInfoByStreamId(s32 streamId, ProfCommInfo& profInfo)
{
    std::string tag = "unknown";
    std::unique_lock<std::mutex> lock(streamMutex_);
    auto iter = streamToTagMap_.find(streamId);
    if (iter == streamToTagMap_.end()) {
        HCCL_INFO("[GetProfInfoByStreamId]streamId:%d is not found", streamId);
    } else {
        tag = iter->second;
        auto opInfoIter = tagOpInfoMap_.find(tag);
        if (opInfoIter == tagOpInfoMap_.end()) {
            HCCL_INFO("[GetProfInfoByStreamId]tag:%s is not found", tag.c_str());
        } else {
            profInfo = opInfoIter->second;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::AddProfInfoByStreamId(s32 streamId, const std::string& tag, const ProfCommInfo& profInfo)
{
    std::unique_lock<std::mutex> lock(streamMutex_);
    auto tagMapIter = streamToTagMap_.find(streamId);
    if (tagMapIter == streamToTagMap_.end()) {
        streamToTagMap_.insert({streamId, tag});
    } else {
        tagMapIter->second = tag;
        // streamId之前存在说明流被销毁了，被其它通信域复用了, 上报一条信息告知profiling taskid发生翻转
        CHK_RET(ReportFilpTask(streamId, UINT16_MAX, UINT16_MAX));
        HCCL_INFO("[AddProfInfoByStreamId] streamId:%d content:%s update", streamId, tag.c_str());
    }

    auto opInfoMapIter = tagOpInfoMap_.find(tag);
    if (opInfoMapIter == tagOpInfoMap_.end()) {
        tagOpInfoMap_.insert(std::make_pair(tag, profInfo));
    } else {
        opInfoMapIter->second = profInfo;
    }
    return HCCL_SUCCESS;
}

void TaskProfilingCallBack([[maybe_unused]] void* userPtr, void* param, u32 length)
{
    if (UNLIKELY(param == nullptr)) {
        HCCL_ERROR("[ProfilingManager][%s]param is nullptr.", __func__);
        return;
    }
    struct hccl::TaskPara* taskPara = (struct hccl::TaskPara*)param;

    if (UNLIKELY(sizeof(hccl::TaskPara) < length)) {
        return;
    }
    HCCL_INFO(
        "[ProfilingManager][%s]Start handle task profiler, taskType[%d], profilerType[%d]", __func__,
        static_cast<int>(taskPara->type), static_cast<int>(taskPara->profilerType));
    switch (taskPara->type) {
        case hccl::TaskType::TASK_BATCH_REPORT:
            dfx::ProfilingManager::ReportTaskInfo(taskPara->streamTasks.streamID, taskPara->streamTasks.ctxPtr);
            break;

        case hccl::TaskType::TASK_FLIP:
            dfx::ProfilingManager::ReportFilpTask(
                taskPara->flipTask.streamID, taskPara->flipTask.taskID, taskPara->flipTask.flipNum);
            break;

        default:
            return;
    }
    return;
}
} // namespace dfx
