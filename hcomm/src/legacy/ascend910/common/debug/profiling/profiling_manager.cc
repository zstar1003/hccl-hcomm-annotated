/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_manager.h"
#include <string>

#include "adapter_prof.h"
#include "adapter_rts_common.h"
#include "workflow_pub.h"
#include "sal_pub.h"

namespace hccl {
std::queue<MsprofApi> ProfilingManager::storageTaskApi_;
std::queue<MsprofApi> ProfilingManager::storageOpApi_;
std::array<std::queue<MsprofAdditionalInfo>, MAX_MODULE_DEVICE_NUM> ProfilingManager::storageAdditionInfo_;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> ProfilingManager::reportAddInfoMutex_;
std::array<std::queue<MsprofCompactInfo>, MAX_MODULE_DEVICE_NUM> ProfilingManager::storageCompactInfo_;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> ProfilingManager::reportCompactInfoMutex_;
std::array<std::queue<MsprofAdditionalInfo>, MAX_MODULE_DEVICE_NUM> ProfilingManager::storageAdditionInfoFftsCapture_;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> ProfilingManager::reportAddInfoFftsCaptureMutex_;
std::mutex ProfilingManager::reportDataQueueMutex_;
std::unordered_map<s32, bool> ProfilingManager::captureStatusThreadIDMap_;
std::mutex ProfilingManager::captureStatusMapMutex_;
std::mutex ProfilingManager::reportDataOpQueueMutex_;

ProfilingManager::ProfilingManager()
    : reporterCallback_(nullptr),
      isHostApiSubscribe_(false),
      isTaskApiSubscribe_(false),
      isAdditionInfoSubscribe_(false)
{}

ProfilingManager::~ProfilingManager() {}

ProfilingManager& ProfilingManager::Instance()
{
    static ProfilingManager profilingManager;
    return profilingManager;
}

Prof_Status ProfilingManager::CallMsprofReport(ReporterData& reporterData) const
{
    CHK_PRT_RET(
        (reporterCallback_ == nullptr),
        HCCL_ERROR("[ProfilingManager][CallMsprofReport] MsprofReporterCallback callback is nullptr."), FAILED);
    return reporterCallback_(
        static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_HCCL),
        static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_REPORT), static_cast<void*>(&reporterData),
        sizeof(ReporterData));
}

HcclResult ProfilingManager::CallMsprofRegFftsLaunch() const
{
    if (!isFftsLaunchSubscribe_) {
        return HCCL_SUCCESS;
    }

    CHK_RET(hrtMsprofRegTypeInfo(
        MSPROF_REPORT_HCCL_NODE_LEVEL, MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE, "context_id_info"));

    CHK_RET(hrtMsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE, "mc2_comm_info"));

    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofRegHcclOpApi() const
{
    if (!isHostHcclOpSubscribe_ || GetExternalInputHcclAicpuUnfold() || GetExternalInputHcclEnableFfts()) {
        return HCCL_SUCCESS;
    }

    for (const auto& name_to_type : PROF_OP_NAME) {
        CHK_RET(hrtMsprofRegTypeInfo(
            MSPROF_REPORT_HCCL_NODE_LEVEL,
            static_cast<uint32_t>(name_to_type.first) + MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE,
            name_to_type.second.c_str()));
    }

    CHK_RET(hrtMsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE, "mc2_comm_info"));

    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofRegHostApi() const
{
    if (!isHostApiSubscribe_) {
        return HCCL_SUCCESS;
    }
    for (const auto& name_to_type : PROF_OP_NAME) {
        CHK_RET(hrtMsprofRegTypeInfo(
            MSPROF_REPORT_ACL_LEVEL, static_cast<uint32_t>(name_to_type.first) + MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE,
            name_to_type.second.c_str()));
    }

    for (const auto& name_to_type : PROF_OP_NAME) {
        CHK_RET(hrtMsprofRegTypeInfo(
            MSPROF_REPORT_NODE_LEVEL, static_cast<uint32_t>(name_to_type.first) + MSPROF_REPORT_NODE_HCCL_BASE_TYPE,
            name_to_type.second.c_str()));
    }

    const std::string hcclType("hccl_op_info");
    CHK_RET(hrtMsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE, hcclType.c_str()));
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportHostNodeApi(
    uint64_t beginTime, uint64_t endTime, uint64_t itemId, uint32_t threadId) const
{
    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    reporterData.threadId = threadId;
    reporterData.beginTime = beginTime;
    reporterData.endTime = endTime;
    reporterData.itemId = itemId;

    // 静态图场景或者acl graph场景, 一次下发，多次执行; 缓存对应数据
    auto mode = GetWorkflowMode();
    if ((mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) || GetThreadCaptureStatus()) {
        HCCL_INFO("CallMsprofReportTaskApi, storageTaskApi");
        std::unique_lock<std::mutex> lock(reportDataQueueMutex_);
        storageTaskApi_.push(reporterData);
        return HCCL_SUCCESS;
    }

    HCCL_INFO("CallMsprofReportHostNodeApi, HostNodeApiType[%u]", MSPROF_REPORT_NODE_LAUNCH_TYPE);
    CHK_RET(hrtMsprofReportApi(1, &reporterData));
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportNodeInfo(
    uint64_t beginTime, uint64_t endTime, const std::string profName, uint32_t threadId)
{
    uint64_t itemId = hrtMsprofGetHashId(profName.c_str(), profName.length());
    auto mode = GetWorkflowMode();
    // hostapi开关 1) 开启: 单算子、静态图模式均上报; 关闭: 静态图模式或者acl graph场景缓存, 单算子不上报
    if (isHostApiSubscribe_ || mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB
        || GetThreadCaptureStatus()) {
        CHK_RET(CallMsprofReportHostNodeApi(beginTime, endTime, itemId, threadId));
    }
    // additionInfo开关 1) 开启: 单算子、静态图模式均上报; 关闭: 静态图或者acl graph场景模式缓存, 单算子不上报
    if (isAdditionInfoSubscribe_ || mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB
        || GetThreadCaptureStatus()) {
        CHK_RET(CallMsprofReportHostNodeBasicInfo(endTime, itemId, threadId));
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportHostApi(
    HcclCMDType cmdType, uint64_t beginTime, u64 count, HcclDataType dataType, AlgType algType, uint64_t groupName,
    u32 numBlocks) const
{
    if (!isHostApiSubscribe_ && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && !GetThreadCaptureStatus()) {
        return HCCL_SUCCESS;
    }
    uint64_t endTime = hrtMsprofSysCycleTime();
    uint32_t threadId = SalGetTid();
    uint32_t type = static_cast<int32_t>(cmdType);
    const std::string profName(GetProfOpName(cmdType));
    uint64_t itemId = hrtMsprofGetHashId(profName.c_str(), profName.length());
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && IsLaunchKernelMode() != true) {
        CHK_RET(CallMsprofReportHostAclApi(type, beginTime, endTime, itemId, threadId));
        CHK_RET(CallMsprofReportHostNodeApi(beginTime, endTime, itemId, threadId));
        if (isAdditionInfoSubscribe_ || GetThreadCaptureStatus()) {
            CHK_RET(CallMsprofReportHostNodeBasicInfo(endTime, itemId, threadId, numBlocks));
        }
    }
    std::string algTypeStr = TransferAlgType(algType);
    CHK_RET(CallMsprofReportHostHcclOpInfo(endTime, threadId, count, dataType, algTypeStr, groupName));
    CHK_RET(CallMsprofReportHostHcclOpApi(beginTime, endTime, itemId, threadId));
    return HCCL_SUCCESS;
}

inline HcclResult ProfilingManager::RegEsTaskType(ProfTaskType taskType) const
{
    const std::string str(GetProfTaskOpName(taskType));
    CHK_RET(hrtMsprofRegTypeInfo(MSPROF_REPORT_HCCL_NODE_LEVEL, static_cast<uint32_t>(taskType), str.c_str()));
    return HCCL_SUCCESS;
}

HcclResult
ProfilingManager::ReportTaskApi(bool isMainStrem, uint64_t beginTime, ProfTaskType taskType, uint32_t agingFlag) const
{
    HcclWorkflowMode mode = GetWorkflowMode();
    // 1、单算子场景，如果订阅开关没有开，直接退出
    // 2、l0 l1 级别时, 子图 launch都需要上报
    if ((!isTaskApiSubscribe_) && (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
        && (taskType != ProfTaskType::TASK_LAUNCH_FFTS_TASK) && (taskType != ProfTaskType::TASK_AIV)
        && !GetThreadCaptureStatus()) {
        return HCCL_SUCCESS;
    }
    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type = (isMainStrem == true) ? MSPROF_REPORT_HCCL_MASTER_TYPE : MSPROF_REPORT_HCCL_SLAVE_TYPE;
    reporterData.threadId = SalGetTid();
    reporterData.beginTime = beginTime;
    reporterData.endTime = hrtMsprofSysCycleTime();
    const std::string taskName(GetProfTaskOpName(taskType));
    reporterData.itemId = hrtMsprofGetHashId(taskName.c_str(), taskName.length());

    // 2、图下沉场景或者acl graph场景，缓存对应数据
    if ((mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) || GetThreadCaptureStatus()) {
        // 缓存对应数据
        HCCL_INFO("CallMsprofReportTaskApi, storageTaskApi");
        std::unique_lock<std::mutex> lock(reportDataQueueMutex_);
        storageTaskApi_.push(reporterData);
        return HCCL_SUCCESS;
    }

    HCCL_INFO(
        "CallMsprofReportTaskApi, isMainStream[%u], taskType[%d], taskName[%s]", isMainStrem,
        static_cast<int32_t>(taskType), taskName.c_str());
    CHK_RET(hrtMsprofReportApi(agingFlag, &reporterData));
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportHostAclApi(
    uint32_t type, uint64_t beginTime, uint64_t endTime, uint64_t itemId, uint32_t threadId) const
{
    if (!isHostApiSubscribe_) {
        return HCCL_SUCCESS;
    }
    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_ACL_LEVEL;
    reporterData.type = static_cast<int32_t>(type) + MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE;
    reporterData.threadId = threadId;
    reporterData.beginTime = beginTime;
    reporterData.endTime = endTime;
    reporterData.itemId = itemId;

    HCCL_INFO("CallMsprofReportHostHcclOpApi, HcclOpApiType[%u]", reporterData.type);
    CHK_RET(hrtMsprofReportApi(1, &reporterData));
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportHostNodeBasicInfo(
    uint64_t timeStamp, uint64_t itemId, uint32_t threadId, u32 numBlocks) const
{
    MsprofCompactInfo reporterData{};

    reporterData.level = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    reporterData.threadId = threadId;
    reporterData.dataLen = sizeof(MsprofNodeBasicInfo);
    reporterData.timeStamp = timeStamp;

    reporterData.data.nodeBasicInfo.opName = itemId;
    reporterData.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_HCCL;
    reporterData.data.nodeBasicInfo.opType = itemId;
    reporterData.data.nodeBasicInfo.blockDim = numBlocks;
    reporterData.data.nodeBasicInfo.opFlag = 0;

    // 图下沉场景或者acl graph场景，缓存对应数据
    auto mode = GetWorkflowMode();
    if ((mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) || GetThreadCaptureStatus()) {
        s32 deviceLogicId = -1;
        CHK_RET(hrtGetDevice(&deviceLogicId));
        HCCL_INFO(
            "CallMsprofReportHostNodeBasicInfo, storageCompactInfo, The used deviceLogicId is [%d]", deviceLogicId);
        u32 maxDeviceNum;
        CHK_RET(GetMaxDevNum(maxDeviceNum));
        CHK_PRT_RET(
            static_cast<u32>(deviceLogicId) >= maxDeviceNum,
            HCCL_ERROR(
                "[ReportHostNodeBasicInfo]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]",
                static_cast<u32>(deviceLogicId), maxDeviceNum),
            HCCL_E_INTERNAL);
        std::unique_lock<std::mutex> lock(reportCompactInfoMutex_[deviceLogicId]);
        storageCompactInfo_[deviceLogicId].push(reporterData);
        if (!isAdditionInfoSubscribe_) {
            return HCCL_SUCCESS;
        }
    }
    HCCL_INFO("CallMsprofReportHostNodeBasicInfo, HostNodeBasicInfoType[%u]", MSPROF_REPORT_NODE_BASIC_INFO_TYPE);
    CHK_RET(hrtMsprofReportCompactInfo(1, &reporterData, sizeof(MsprofCompactInfo)));
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportHostHcclOpApi(
    uint64_t beginTime, uint64_t endTime, uint64_t itemId, uint32_t threadId) const
{
    auto mode = GetWorkflowMode();
    if ((!isHostHcclOpSubscribe_ || GetExternalInputHcclAicpuUnfold() || GetExternalInputHcclEnableFfts())
        && (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && !GetThreadCaptureStatus()) {
        return HCCL_SUCCESS;
    }

    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    // 集合通信算子粒度的都是主流
    reporterData.type = MSPROF_REPORT_HCCL_MASTER_TYPE;
    reporterData.threadId = threadId;
    reporterData.beginTime = beginTime;
    reporterData.endTime = endTime;
    reporterData.itemId = itemId;

    // 静态图场景或者acl graph场景, 一次下发，多次执行; 缓存对应数据
    if ((mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) || GetThreadCaptureStatus()) {
        HCCL_INFO("CallMsprofReportOpApi, storageOpApi");
        std::unique_lock<std::mutex> lock(reportDataOpQueueMutex_);
        storageOpApi_.push(reporterData);
        return HCCL_SUCCESS;
    }

    HCCL_INFO("CallMsprofReportHostHcclOpApi, HcclOpApiType[%u]", MSPROF_REPORT_HCCL_MASTER_TYPE);
    CHK_RET(hrtMsprofReportApi(1, &reporterData));
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportTaskApi(bool isMainStrem, uint64_t beginTime, ProfTaskType taskType) const
{
    uint32_t agingFlag = 0;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        agingFlag = aging;
    }
    CHK_RET(ReportTaskApi(isMainStrem, beginTime, taskType, agingFlag));

    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportHostHcclOpInfo(
    uint64_t timeStamp, uint32_t threadId, u64 count, HcclDataType dataType, std::string& algTypeStr,
    uint64_t groupName) const
{
    MsprofCompactInfo reporterData{};

    reporterData.level = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE;
    reporterData.threadId = threadId;
    reporterData.dataLen = sizeof(MsprofHCCLOPInfo);
    reporterData.timeStamp = timeStamp;

    reporterData.data.hcclopInfo.relay = 0;
    reporterData.data.hcclopInfo.retry = 0;
    reporterData.data.hcclopInfo.dataType = dataType;
    reporterData.data.hcclopInfo.algType = hrtMsprofGetHashId(algTypeStr.c_str(), algTypeStr.length());
    reporterData.data.hcclopInfo.count = count;
    reporterData.data.hcclopInfo.groupName = groupName;

    // 图下沉场景或者acl graph场景，缓存对应数据
    if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) || GetThreadCaptureStatus()) {
        // 缓存对应数据
        s32 deviceLogicId = -1;
        CHK_RET(hrtGetDevice(&deviceLogicId));
        HCCL_INFO("CallMsprofReportHostHcclOpInfo, storageCompactInfo, The used deviceLogicId is [%d]", deviceLogicId);
        u32 maxDeviceNum;
        CHK_RET(GetMaxDevNum(maxDeviceNum));
        CHK_PRT_RET(
            static_cast<u32>(deviceLogicId) >= maxDeviceNum,
            HCCL_ERROR(
                "[ReportAdditionInfo]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]",
                static_cast<u32>(deviceLogicId), maxDeviceNum),
            HCCL_E_INTERNAL);
        std::unique_lock<std::mutex> lock(reportCompactInfoMutex_[deviceLogicId]);
        storageCompactInfo_[deviceLogicId].push(reporterData);
        if (!isHostApiSubscribe_) {
            return HCCL_SUCCESS;
        }
    }

    HCCL_INFO("CallMsprofReportHostHcclOpInfo, hcclopInfoType[%u]", MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE);
    CHK_RET(hrtMsprofReportCompactInfo(1, &reporterData, sizeof(MsprofCompactInfo)));
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportMc2CommInfo(uint64_t timeStamp, const void* data, int len)
{
    uint32_t agingFlag = 0;
    auto mode = GetWorkflowMode();
    if (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        agingFlag = 1;
    }

    MsprofAdditionalInfo reporterData{};
    reporterData.level = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE;
    reporterData.threadId = SalGetTid();
    reporterData.dataLen = len;
    reporterData.timeStamp = timeStamp;

    s32 sret = memcpy_s(reporterData.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, data, len);
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("memcpy failed. errorno[%d]:", sret), HCCL_E_MEMORY);

    if (!isHostApiSubscribe_ || GetThreadCaptureStatus()) {
        // 缓存对应数据
        s32 deviceLogicId = -1;
        CHK_RET(hrtGetDevice(&deviceLogicId));
        HCCL_INFO("CallMsprofReportAdditionInfo, storageAdditionInfo, The used deviceLogicId is [%d]", deviceLogicId);
        u32 maxDeviceNum;
        CHK_RET(GetMaxDevNum(maxDeviceNum));
        CHK_PRT_RET(
            static_cast<u32>(deviceLogicId) >= maxDeviceNum,
            HCCL_ERROR(
                "[ReportAdditionInfo]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]",
                static_cast<u32>(deviceLogicId), maxDeviceNum),
            HCCL_E_INTERNAL);
        std::unique_lock<std::mutex> lock(reportAddInfoMutex_[deviceLogicId]);
        storageAdditionInfo_[deviceLogicId].push(reporterData);
        return HCCL_SUCCESS;
    }
    HCCL_INFO("CallMsprofReportMc2CommInfo, Mc2CommInfoType[%u]", MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE);
    CHK_RET(hrtMsprofReportAdditionalInfo(agingFlag, &reporterData, sizeof(MsprofAdditionalInfo)));
    return HCCL_SUCCESS;
}

HcclResult
ProfilingManager::CallEsMsprofReportTaskApi(bool isMainStrem, uint64_t beginTime, ProfTaskType taskType) const
{
    if (!isTaskApiSubscribe_) {
        return HCCL_SUCCESS;
    }
    MsprofApi reporterData{};
    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type = (isMainStrem == true) ? MSPROF_REPORT_HCCL_MASTER_TYPE : MSPROF_REPORT_HCCL_SLAVE_TYPE;
    reporterData.threadId = SalGetTid();
    reporterData.beginTime = beginTime;
    reporterData.endTime = hrtMsprofSysCycleTime();
    const std::string taskName(GetProfTaskOpName(taskType));
    reporterData.itemId = hrtMsprofGetHashId(taskName.c_str(), taskName.length());
    HCCL_INFO(
        "ReportTaskApi, isMainStream[%u], taskType[%d], taskName[%s]", isMainStrem, static_cast<int32_t>(taskType),
        taskName.c_str());
    CHK_RET(hrtMsprofReportApi(aging, &reporterData));
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportMultiThreadInfo(const std::vector<uint32_t>& tidInfo) const
{
    if (!isTaskApiSubscribe_ && !GetThreadCaptureStatus()) {
        return HCCL_SUCCESS;
    }

    struct MsprofMultiThread threadInfo;
    uint64_t timeStamp = hrtMsprofSysCycleTime();
    uint32_t totalSize = tidInfo.size();
    uint32_t currentSize;
    uint32_t sendNum = totalSize / MSPROF_MULTI_THREAD_MAX_NUM + 1;

    for (uint32_t j = 0; j < sendNum; j++) {
        currentSize = totalSize - j * MSPROF_MULTI_THREAD_MAX_NUM;
        threadInfo.threadNum = currentSize > MSPROF_MULTI_THREAD_MAX_NUM ? MSPROF_MULTI_THREAD_MAX_NUM : currentSize;
        for (uint32_t i = 0; i < threadInfo.threadNum; i++) {
            threadInfo.threadId[i] = tidInfo[i + j * MSPROF_MULTI_THREAD_MAX_NUM];
        }
        HCCL_INFO("CallMsprofReportMultiThreadInfo");
        CHK_RET(CallMsprofReportAdditionInfo(
            static_cast<int32_t>(ProfTaskType::TASK_MULTI_THREAD), timeStamp, &threadInfo,
            sizeof(struct MsprofMultiThread)));
    }

    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportContextIdInfo(u32 ctxIdMax) const
{
    struct MsprofContextIdInfo ctxInfo;
    ctxInfo.ctxIdNum = 2; // 因HCCL ctxId连续，固定上报2个：开始：0； 结束：ctxIdMax
    ctxInfo.ctxIds[0] = 0;
    ctxInfo.ctxIds[1] = ctxIdMax;

    uint64_t timeStamp = hrtMsprofSysCycleTime();
    HCCL_INFO("CallMsprofReportContextIdInfo, ctxIdNum[%u]", ctxInfo.ctxIdNum);
    CHK_RET(CallMsprofReportAdditionInfo(
        MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE, timeStamp, &ctxInfo, sizeof(struct MsprofContextIdInfo)));
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::ClearStoragedProfilingInfo()
{
    std::unique_lock<std::mutex> lockTaskApi(reportDataQueueMutex_);
    HCCL_INFO("[ClearStoragedProfilingInfo] taskApiQueueSize is [%u]", storageTaskApi_.size());

    std::queue<MsprofApi> emptyTaskApi;
    std::swap(storageTaskApi_, emptyTaskApi);

    std::unique_lock<std::mutex> lockOpApi(reportDataOpQueueMutex_);
    HCCL_INFO("[ClearStoragedProfilingInfo] opApiQueueSize is [%u]", storageOpApi_.size());

    std::queue<MsprofApi> emptyOpApi;
    std::swap(storageOpApi_, emptyOpApi);

    s32 deviceLogicId = -1;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    HCCL_INFO(
        "[ClearStoragedAdditionInfo] The size of the storageAdditionInfo_[%d] is [%u]", deviceLogicId,
        storageAdditionInfo_[deviceLogicId].size());
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        static_cast<u32>(deviceLogicId) >= maxDeviceNum,
        HCCL_ERROR(
            "[ReportStoragedAdditionInfo]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]",
            static_cast<u32>(deviceLogicId), maxDeviceNum),
        HCCL_E_INTERNAL);
    std::unique_lock<std::mutex> lockAddInfo(reportAddInfoMutex_[deviceLogicId]);
    std::queue<MsprofAdditionalInfo> emptyAddition;
    std::swap(storageAdditionInfo_[deviceLogicId], emptyAddition);

    HCCL_INFO(
        "[ClearStoragedCompactInfo] The size of the storageCompactInfo_[%d] is [%u]", deviceLogicId,
        storageCompactInfo_[deviceLogicId].size());
    std::unique_lock<std::mutex> lockCompactInfo(reportCompactInfoMutex_[deviceLogicId]);
    std::queue<MsprofCompactInfo> emptyCompactInfo;
    std::swap(storageCompactInfo_[deviceLogicId], emptyCompactInfo);

    HCCL_INFO(
        "[ClearStorageAdditionInfoFftsCapture_] The size of the storageAdditionInfoFftsCapture_[%d] is [%u]",
        deviceLogicId, storageAdditionInfoFftsCapture_[deviceLogicId].size());
    std::unique_lock<std::mutex> lockAddInfoCapture(reportAddInfoFftsCaptureMutex_[deviceLogicId]);
    std::queue<MsprofAdditionalInfo> emptyAdditionCapture;
    std::swap(storageAdditionInfoFftsCapture_[deviceLogicId], emptyAdditionCapture);
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::ReportAdditionInfo(
    uint32_t type, uint64_t timeStamp, const void* data, int len, uint32_t agingFlag) const
{
    HcclWorkflowMode mode = GetWorkflowMode();
    // 1、单算子场景，如果订阅开关没有开, 且上报的不是contextID,直接退出;
    if ((!isAdditionInfoSubscribe_) && (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
        && (type != MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE) && !GetThreadCaptureStatus()) {
        return HCCL_SUCCESS;
    }
    MsprofAdditionalInfo reporterData{};

    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type = type;
    reporterData.threadId = SalGetTid();
    reporterData.dataLen = len;
    reporterData.timeStamp = timeStamp;
    s32 sret = memcpy_s(reporterData.data, sizeof(reporterData.data), data, len);
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("memcpy failed. errorno[%d]:", sret), HCCL_E_MEMORY);

    // 2、图下沉场景或者acl graph场景，缓存对应数据
    if ((mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) || GetThreadCaptureStatus()
        || (isFftsDispatcher_.load() && // 3、FFTS+下发场景，addition开关打开，缓存对应数据
            (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && (isAdditionInfoSubscribe_))) {
        // 缓存对应数据
        s32 deviceLogicId = -1;
        CHK_RET(hrtGetDevice(&deviceLogicId));
        HCCL_INFO("CallMsprofReportAdditionInfo, storageAdditionInfo, The used deviceLogicId is [%d]", deviceLogicId);
        u32 maxDeviceNum;
        CHK_RET(GetMaxDevNum(maxDeviceNum));
        CHK_PRT_RET(
            static_cast<u32>(deviceLogicId) >= maxDeviceNum,
            HCCL_ERROR(
                "[ReportAdditionInfo]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]",
                static_cast<u32>(deviceLogicId), maxDeviceNum),
            HCCL_E_INTERNAL);
        std::unique_lock<std::mutex> lock(reportAddInfoMutex_[deviceLogicId]);
        storageAdditionInfo_[deviceLogicId].push(reporterData);
        if (isFftsDispatcher_ || !isAdditionInfoSubscribe_) {
            return HCCL_SUCCESS;
        }
    }

    // 4、开关开启，非子图下发场景，直接上报对应数据
    HCCL_INFO("CallMsprofReportAdditionInfo, AdditionInfoType[%u]", type);
    CHK_RET(hrtMsprofReportAdditionalInfo(agingFlag, &reporterData, sizeof(MsprofAdditionalInfo)));
    return HCCL_SUCCESS;
}

HcclResult
ProfilingManager::CallMsprofReportAdditionInfo(uint32_t type, uint64_t timeStamp, const void* data, int len) const
{
    uint32_t agingFlag = 0;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        agingFlag = 1;
    }
    CHK_RET(ReportAdditionInfo(type, timeStamp, data, len, agingFlag));
    return HCCL_SUCCESS;
}

HcclResult
ProfilingManager::CallMsprofReportEsAdditionInfo(uint32_t type, uint64_t timeStamp, const void* data, int len) const
{
    if (!isAdditionInfoSubscribe_) {
        return HCCL_SUCCESS;
    }
    MsprofAdditionalInfo reporterData{};

    reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
    reporterData.type = type;
    reporterData.threadId = SalGetTid();
    reporterData.dataLen = len;
    reporterData.timeStamp = timeStamp;

    s32 sret = memcpy_s(reporterData.data, sizeof(reporterData.data), data, len);
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("memcpy failed. errorno[%d]:", sret), HCCL_E_MEMORY);
    HCCL_INFO("CallMsprofReportAdditionInfo, AdditionInfoType[%u]", type);
    CHK_RET(hrtMsprofReportAdditionalInfo(aging, &reporterData, sizeof(MsprofAdditionalInfo)));
    return HCCL_SUCCESS;
}

Prof_Status ProfilingManager::PluginInit() const
{
    CHK_PRT_RET(
        (reporterCallback_ == nullptr),
        HCCL_ERROR("[ProfilingManager][PluginInit] MsprofReporterCallback callback is nullptr."), FAILED);

    int32_t cb_ret = reporterCallback_(
        static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_HCCL),
        static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_INIT), nullptr, 0);
    CHK_PRT_RET(
        (cb_ret != MSPROF_ERROR_NONE), HCCL_ERROR("[ProfilingManager][PluginInit] Reporter init failed."), FAILED);

    HCCL_INFO("[ProfilingManager][PluginInit] Reporter init success.");

    return SUCCESS;
}

Prof_Status ProfilingManager::PluginUnInit() const
{
    CHK_PRT_RET(
        (reporterCallback_ == nullptr),
        HCCL_ERROR("[ProfilingManager][PluginUnInit] MsprofReporterCallback callback is nullptr."), FAILED);

    int32_t cb_ret = reporterCallback_(
        static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_HCCL),
        static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_UNINIT), nullptr, 0);
    CHK_PRT_RET(
        (cb_ret != MSPROF_ERROR_NONE), HCCL_ERROR("[ProfilingManager][PluginUnInit] Profiling reporter uninit failed."),
        FAILED);

    HCCL_INFO("[ProfilingManager][PluginUnInit] Profiling reporter uninit success.");

    return SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportAdditionInfoForEsLookup(EsLoopUpPara& para, ProfTaskType type)
{
    HCCL_INFO("Entry CallMsprofReportAdditionInfoForEsLookup");
    HCCLReportData hcclReportData{};
    hcclReportData.ts = hrtMsprofSysCycleTime();
    std::string nameInfo = GetProfTaskOpName(type);
    hcclReportData.profInfo.itemId = hrtMsprofGetHashId(nameInfo.c_str(), nameInfo.length());
    std::string cclTag = std::to_string(para.tag);
    hcclReportData.profInfo.cclTag = hrtMsprofGetHashId(cclTag.c_str(), cclTag.length());
    hcclReportData.profInfo.groupName = static_cast<u64>(reinterpret_cast<uintptr_t>("unknown"));
    hcclReportData.profInfo.rankSize = 0;
    hcclReportData.profInfo.workFlowMode = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    hcclReportData.profInfo.planeID = 0;
    hcclReportData.profInfo.notifyID = 0;
    hcclReportData.profInfo.stage = 0;
    hcclReportData.profInfo.role = static_cast<uint32_t>(TaskRole::DST);
    hcclReportData.profInfo.durationEstimated = 0;
    hcclReportData.profInfo.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(para.srcAddr));
    hcclReportData.profInfo.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(para.dstAddr));
    hcclReportData.profInfo.dataSize = static_cast<u32>(para.dataSize);
    hcclReportData.profInfo.opType = 0;
    hcclReportData.profInfo.dataType = HCCL_DATA_TYPE_FP32;
    hcclReportData.profInfo.linkType = static_cast<u32>(LinkType::LINK_ONCHIP);
    hcclReportData.profInfo.transportType = static_cast<int32_t>(SimpleTaskType::RDMA);

    int32_t ret = CallMsprofReportEsAdditionInfo(
        static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO), hcclReportData.ts, &hcclReportData.profInfo,
        sizeof(hcclReportData.profInfo));
    CHK_PRT_RET((ret != 0), HCCL_ERROR("[TaskProfiling] CallMsprofReportAdditionInfoForEsLookup failed."), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

Prof_Status ProfilingManager::GetHashKey(MsprofHashData& data) const
{
    CHK_PRT_RET(
        (reporterCallback_ == nullptr),
        HCCL_ERROR("[ProfilingManager][GetHashKey] MsprofReporterCallback callback is nullptr."), FAILED);
    if (data.dataLen == 0) {
        data.hashId = 0;
        HCCL_INFO("[Check][Param]PluginUnInit MsprofReporterCallback GetHashKey in default.");
    } else {
        int32_t cb_ret = reporterCallback_(
            static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_HCCL),
            static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_HASH), static_cast<void*>(&data),
            sizeof(MsprofHashData));

        CHK_PRT_RET(
            (cb_ret != MSPROF_ERROR_NONE),
            HCCL_ERROR("[ProfilingManager][GetHashKey] Profiling reporter GetHashKey failed."), FAILED);

        HCCL_INFO("[ProfilingManager][GetHashKey] Profiling reporter GetHashKey success.");
    }

    return SUCCESS;
}

HcclResult ProfilingManager::ReportStoragedTaskApi()
{
    if (!isTaskApiSubscribe_ && !isHostApiSubscribe_) {
        return HCCL_SUCCESS;
    }
    std::unique_lock<std::mutex> lock(reportDataQueueMutex_);
    HCCL_INFO("[ReportStoragedTaskApi] taskApiQueueSize is [%u]", storageTaskApi_.size());
    if (!storageTaskApi_.empty()) {
        std::queue<MsprofApi> tempTaskApi = storageTaskApi_;
        lock.unlock();
        while (!tempTaskApi.empty()) {
            MsprofApi reportData = tempTaskApi.front();
            tempTaskApi.pop();
            CHK_RET(hrtMsprofReportApi(0, &reportData));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::ReportStoragedOpApi()
{
    if (!isHostHcclOpSubscribe_ || GetExternalInputHcclAicpuUnfold() || GetExternalInputHcclEnableFfts()) {
        return HCCL_SUCCESS;
    }

    std::unique_lock<std::mutex> lock(reportDataOpQueueMutex_);
    HCCL_INFO("[ReportStoragedOpApi] opApiQueueSize is [%u]", storageOpApi_.size());
    if (!storageOpApi_.empty()) {
        std::queue<MsprofApi> tempOpApi = storageOpApi_;
        lock.unlock();
        while (!tempOpApi.empty()) {
            MsprofApi reportData = tempOpApi.front();
            tempOpApi.pop();
            CHK_RET(hrtMsprofReportApi(0, &reportData));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::ReportStoragedAdditionInfo()
{
    for (u32 i = 0; i < MAX_MODULE_DEVICE_NUM; i++) {
        std::unique_lock<std::mutex> lock(reportAddInfoMutex_[i]);
        HCCL_INFO(
            "[ReportStoragedAdditionInfo] The size of the storageAdditionInfo_[%u] is [%u]", i,
            storageAdditionInfo_[i].size());
        if (!storageAdditionInfo_[i].empty()) {
            std::queue<MsprofAdditionalInfo> tempTaskAdditionalInfo = storageAdditionInfo_[i];
            lock.unlock();
            while (!tempTaskAdditionalInfo.empty()) {
                MsprofAdditionalInfo reportData = tempTaskAdditionalInfo.front();
                tempTaskAdditionalInfo.pop();
                CHK_RET(hrtMsprofReportAdditionalInfo(0, &reportData, sizeof(MsprofAdditionalInfo)));
            }
        }
        // acl graph ffts+场景下， 一次下发多次执行， 执行时上报保存的task信息
        std::unique_lock<std::mutex> lockCapture(reportAddInfoFftsCaptureMutex_[i]);
        HCCL_INFO(
            "[ReportStoragedAdditionInfo] The size of the storageAdditionInfoFftsCapture_[%u] is [%u]", i,
            storageAdditionInfoFftsCapture_[i].size());
        if (!storageAdditionInfoFftsCapture_[i].empty()) {
            std::queue<MsprofAdditionalInfo> tempTaskAdditionalInfo = storageAdditionInfoFftsCapture_[i];
            lockCapture.unlock();
            while (!tempTaskAdditionalInfo.empty()) {
                MsprofAdditionalInfo reportData = tempTaskAdditionalInfo.front();
                tempTaskAdditionalInfo.pop();
                CHK_RET(hrtMsprofReportAdditionalInfo(0, &reportData, sizeof(MsprofAdditionalInfo)));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::ReportStoragedCompactInfo()
{
    for (u32 i = 0; i < MAX_MODULE_DEVICE_NUM; i++) {
        std::unique_lock<std::mutex> lock(reportCompactInfoMutex_[i]);
        HCCL_INFO(
            "[ReportStoragedCompactInfo] The size of the storageCompactInfo_[%u] is [%u]", i,
            storageCompactInfo_[i].size());
        if (!storageCompactInfo_[i].empty()) {
            std::queue<MsprofCompactInfo> tempCompactInfo = storageCompactInfo_[i];
            lock.unlock();
            while (!tempCompactInfo.empty()) {
                MsprofCompactInfo reportData = tempCompactInfo.front();
                tempCompactInfo.pop();
                CHK_RET(hrtMsprofReportCompactInfo(0, &reportData, sizeof(MsprofCompactInfo)));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::ReportStoragedFftsInfo()
{
    uint64_t ts = hrtMsprofSysCycleTime();

    s32 deviceLogicId = -1;
    if (hrtGetDevice(&deviceLogicId) != HCCL_SUCCESS) {
        deviceLogicId = 0;
        HCCL_WARNING("[ReportStoragedAdditionInfo]deviceLogicId[%d]", deviceLogicId);
    }
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        static_cast<u32>(deviceLogicId) >= maxDeviceNum,
        HCCL_ERROR(
            "[ReportStoragedAdditionInfo]deviceLogicId_[%u] is bigger than maxDeviceNum[%u]",
            static_cast<u32>(deviceLogicId), maxDeviceNum),
        HCCL_E_INTERNAL);
    std::unique_lock<std::mutex> lock(reportAddInfoMutex_[deviceLogicId]);
    HCCL_INFO(
        "[ReportStoragedFftsInfo] The size of the storageAdditionInfo_[%d] is [%u] ", deviceLogicId,
        storageAdditionInfo_[deviceLogicId].size());

    while (!storageAdditionInfo_[deviceLogicId].empty()) {
        MsprofAdditionalInfo reportData = storageAdditionInfo_[deviceLogicId].front();
        storageAdditionInfo_[deviceLogicId].pop();
        reportData.timeStamp = ts;
        if (GetThreadCaptureStatus()) {
            // acl graph ffts+ 场景下， 下发的task信息进行保存以便后续多次使用
            std::unique_lock<std::mutex> lockCapture(reportAddInfoFftsCaptureMutex_[deviceLogicId]);
            storageAdditionInfoFftsCapture_[deviceLogicId].push(reportData);
            lockCapture.unlock();
        }
        if (isFftsLaunchSubscribe_ && isTaskApiSubscribe_) {
            CHK_RET(hrtMsprofReportAdditionalInfo(0, &reportData, sizeof(MsprofAdditionalInfo)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofReportAdditionInfoForEsUpdate(const EsUpdatePara& para, ProfTaskType type)
{
    HCCLReportData hcclReportData{};
    hcclReportData.ts = hrtMsprofSysCycleTime();
    std::string nameInfo = GetProfTaskOpName(type);
    hcclReportData.profInfo.itemId = hrtMsprofGetHashId(nameInfo.c_str(), nameInfo.length());
    std::string cclTag = std::to_string(para.tag);
    hcclReportData.profInfo.cclTag = hrtMsprofGetHashId(cclTag.c_str(), cclTag.length());
    hcclReportData.profInfo.groupName = static_cast<u64>(reinterpret_cast<uintptr_t>(para.groupName));
    hcclReportData.profInfo.localRank = INVALID_VALUE_RANKID;
    hcclReportData.profInfo.remoteRank = INVALID_VALUE_RANKID;
    hcclReportData.profInfo.rankSize = PARSE_RANK_SIZE(0);
    hcclReportData.profInfo.workFlowMode = static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    hcclReportData.profInfo.planeID = 0;
    hcclReportData.profInfo.notifyID = INVALID_U64;
    hcclReportData.profInfo.stage = 0;
    hcclReportData.profInfo.role = static_cast<uint32_t>(TaskRole::DST);
    hcclReportData.profInfo.durationEstimated = 0;
    hcclReportData.profInfo.srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(para.srcAddr));
    hcclReportData.profInfo.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(para.dstAddr));
    hcclReportData.profInfo.dataSize = static_cast<u32>(para.dataSize);
    hcclReportData.profInfo.opType = 0;
    hcclReportData.profInfo.dataType = HCCL_DATA_TYPE_FP32;
    hcclReportData.profInfo.linkType = static_cast<u32>(LinkType::LINK_ONCHIP);
    hcclReportData.profInfo.transportType = static_cast<int32_t>(SimpleTaskType::RDMA);
    hcclReportData.profInfo.rdmaType = static_cast<u32>(RdmaType::RDMA_SEND_PAYLOAD);

    int32_t ret = CallMsprofReportEsAdditionInfo(
        static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO), hcclReportData.ts, &hcclReportData.profInfo,
        sizeof(hcclReportData.profInfo));
    CHK_PRT_RET((ret != 0), HCCL_ERROR("[TaskProfiling] CallMsprofReportAdditionInfoForEsUpdate failed."), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofRegEsTaskTypeApi() const
{
    if (!isTaskApiSubscribe_) {
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[ProfilingManager][CallMsprofRegEsTaskTypeApi] ready to register task types");

    // new
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_LOOKUP_RESPONSE_MEMCPY));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_LOOKUP_RESPONSE_ISEND));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_SHARE_MEMORY_ISEND_RECORD));

    CHK_RET(RegEsTaskType(ProfTaskType::TASK_ABORT_SELF));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_SERVICE_CANCEL));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_DESTROY_RESOURCE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_EVENT_WAIT));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_ISET_LOOKUP_RESPONSE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_WAIT_SOME));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_GET_LOOKUP_REQUEST));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_ISEND_UPDATE_RESPONSE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_ISEND_LOOKUP_RESPONSE));

    CHK_RET(RegEsTaskType(ProfTaskType::TASK_HCCL_INFO));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_COLL_RECV_LOOKUP_REQUEST));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_COLL_RECV_UPDATE_REQUEST));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_REMOTE_UPDATE_SEND_REQUEST));

    CHK_RET(RegEsTaskType(ProfTaskType::TASK_KEY_DROP_DUPLICATES));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_SEND_KEYS));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_SEND_KEYS_RECORD));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_EVENT_WAIT_RECV_DONE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_RESET_UNIQUE_HANDLE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_EVENT_WAIT_SEND_DONE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_RECV_VALUES));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_RECOVER_VALUE_AICORE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_GATHER_FINISH));

    CHK_RET(RegEsTaskType(ProfTaskType::TASK_REMOTE_UPDATE_KEY_REDUCE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_VALUE_CLEAR_AICORE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_VALUE_REDUCE_SUM_AICORE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_UPDATE_RESET_UNIQUE_HANDLE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_NOTIFY_REMOTE_IMRECV_DONE_SIGNAL_KEY));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_NOTIFY_REMOTE_IMRECV_DONE_SIGNAL_VALUE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_REMOTE_UPDATE_RECV_RESPONSE));

    CHK_RET(RegEsTaskType(ProfTaskType::TASK_BUILD_CS_TRANSPORT));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_UPDATE_ALG_GLOBAL_REDUCE));
    CHK_RET(RegEsTaskType(ProfTaskType::TASK_AIV));

    return HCCL_SUCCESS;
}

HcclResult ProfilingManager::CallMsprofRegTaskTypeApi() const
{
    if (!isTaskApiSubscribe_) {
        return HCCL_SUCCESS;
    }

    const std::string taskType(GetProfTaskOpName(ProfTaskType::TASK_HCCL_INFO));
    CHK_RET(hrtMsprofRegTypeInfo(
        MSPROF_REPORT_HCCL_NODE_LEVEL, static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO), taskType.c_str()));

    const std::string multiThreadType(GetProfTaskOpName(ProfTaskType::TASK_MULTI_THREAD));
    CHK_RET(hrtMsprofRegTypeInfo(
        MSPROF_REPORT_HCCL_NODE_LEVEL, static_cast<uint32_t>(ProfTaskType::TASK_MULTI_THREAD),
        multiThreadType.c_str()));

    const std::string ctxIdInfo("context_id_info");
    CHK_RET(hrtMsprofRegTypeInfo(
        MSPROF_REPORT_HCCL_NODE_LEVEL, MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE, ctxIdInfo.c_str()));

    const std::string type("node_basic_info");
    CHK_RET(hrtMsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_BASIC_INFO_TYPE, type.c_str()));

    const std::string mc2Type("mc2_comm_info");
    CHK_RET(hrtMsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE, mc2Type.c_str()));
    return HCCL_SUCCESS;
}

void ProfilingManager::SetFftsDispatcherMode() { isFftsDispatcher_.store(true); }

void ProfilingManager::ReSetFftsDispatcherMode() { isFftsDispatcher_.store(false); }

void ProfilingManager::SetThreadCaptureStatus(s32 threadID, bool isCapture)
{
    HCCL_DEBUG("[SetThreadCaptureStatus] threadID[%d], captureStatus[%d]", threadID, isCapture);
    std::unique_lock<std::mutex> lockMap(captureStatusMapMutex_);
    captureStatusThreadIDMap_.insert(std::make_pair(threadID, isCapture));
}

bool ProfilingManager::GetThreadCaptureStatus()
{
    // 返回当前线程的capture状态
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_310P1 || devType == DevType::DEV_TYPE_310P3) {
        return false;
    }

    s32 threadID = SalGetTid();
    std::unique_lock<std::mutex> lockMap(captureStatusMapMutex_);
    if (captureStatusThreadIDMap_.count(threadID) == 0) {
        HCCL_DEBUG("[GetThreadCaptureStatus] threadID[%d] not in map", threadID);
        return false;
    } else {
        return captureStatusThreadIDMap_[threadID];
    }
}

void ProfilingManager::DeleteThreadCaptureStatus(s32 threadID)
{
    std::unique_lock<std::mutex> lockMap(captureStatusMapMutex_);
    captureStatusThreadIDMap_.erase(threadID);
}

} // namespace hccl
