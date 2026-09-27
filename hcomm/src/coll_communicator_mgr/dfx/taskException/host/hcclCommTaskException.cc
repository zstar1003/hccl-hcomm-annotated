/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcclCommTaskException.h"
#include <memory>
#include "log.h"
#include "coll_comm.h"
#include "acl/acl_rt.h"
#include "orion_adapter_hccp.h"
#include "aubdfx_api.h"
#include "hcomm_adapter_hccp.h"
#include <adapter_error_manager_pub.h>
#include "op_type.h"
#include "task_exception_handler.h"
#include "ccuTaskException.h"
#include "hccl_types.h"
#include "dpu_kernel_entrance.h"

namespace hcomm {

using namespace std;

constexpr u32 MAX_MODULE_DEVICE_NUM_V2 = 65;
constexpr uint32_t TASK_CONTEXT_SIZE = 50;
constexpr uint32_t TASK_CONTEXT_INFO_SIZE = LOG_TMPBUF_SIZE - 50; // task 执行失败时打印前序task信息的长度限制

// Leaky Singleton: 故意不释放，规避 Static Destruction Order Fiasco，
// 确保 RTS 回调访问 handlers_ 时内存仍有效
static TaskExceptionHost* handlers_ = new TaskExceptionHost[MAX_MODULE_DEVICE_NUM_V2];
GetAicpuCqeErrInfoCallBackHcomm g_getAicpuCqeErrInfoCallBack = nullptr;
AicpuGetErrStatusVecCallBack g_AicpuGetErrStatusVecCallBack = nullptr;

void RegisterGetAicpuCqeErrInfoCallBackHcomm(GetAicpuCqeErrInfoCallBackHcomm p1)
{
    g_getAicpuCqeErrInfoCallBack = p1;
    return;
}

void TaskExceptionHost::ClusterMoniterGetAicpuCqeErrInfo(
    u32 remoteLocalId, u32 locDeviceId, uint16_t status, string localEid, string remoteEid, string remoteInsId) const
{
    if (g_getAicpuCqeErrInfoCallBack != nullptr) {
        g_getAicpuCqeErrInfoCallBack(remoteLocalId, locDeviceId, status, localEid, remoteEid, remoteInsId);
    } else {
        HCCL_RUN_WARNING("[%s]g_getAicpuCqeErrInfoCallBack is nullptr.", __func__);
    }
    return;
}

void RegisterAicpuGetErrStatusVecCallBack(AicpuGetErrStatusVecCallBack p1)
{
    g_AicpuGetErrStatusVecCallBack = p1;
    return;
}

std::vector<std::string> AicpuGetErrStatusVec(s32 deviceLogicID)
{
    if (g_AicpuGetErrStatusVecCallBack != nullptr) {
        return g_AicpuGetErrStatusVecCallBack(deviceLogicID);
    } else {
        HCCL_RUN_WARNING("[AicpuGetErrStatusVec]g_AicpuGetErrStatusVecCallBack is nullptr.");
    }
    return std::vector<std::string>();
}

std::string AicpuGetAndPrintClusterMonitorErr(const rtExceptionInfo* exceptionInfo)
{
    auto errStatusVec = AicpuGetErrStatusVec(exceptionInfo->deviceid);
    std::string errMsg = "";
    int errSize = errStatusVec.size();
    if (errSize > 0) {
        int maxListSize = 3; // 放入errMsg中的异常事件最多只有3个
        if (errSize <= maxListSize) {
            errMsg = "\nthere are(is) " + std::to_string(errSize) + " abnormal device(s):\n";
        } else {
            errMsg = "\nthere are " + std::to_string(errSize) + " abnormal device(s), "
                     + "only the first 3 devices are listed:\n";
        }

        for (int i = 0; i < errSize; i++) {
            HCCL_ERROR(
                "[%s][%s][%s] %s", LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_RUN_FAILED.c_str(),
                LOG_KEYWORDS_AICPU.c_str(), errStatusVec[i].c_str());
            if (i < maxListSize) {
                errMsg += ("\t" + errStatusVec[i] + "\n");
            }
        }
    }
    return errMsg;
}

TaskExceptionHost* TaskExceptionHost::GetInstance(s32 deviceLogicID)
{
    if (deviceLogicID < 0 || static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM_V2) {
        HCCL_ERROR("[TaskExceptionHost][GetInstance] deviceLogicID[%d] is invalid", deviceLogicID);
        return nullptr;
    }
    return &handlers_[deviceLogicID];
}

TaskExceptionHost::~TaskExceptionHost()
{
    std::unique_lock<std::mutex> lock(taskExceptionMutex_);
    if (!CommRegisterMap_.empty()) {
        CommRegisterMap_.clear();
        aclError ret = aclrtExceptionInfoCallbackUnregister(ProcessCallback); // 注销注册给rts的TaskException回调
        HCCL_RUN_INFO("[%s]aclrtExceptionInfoCallbackUnregister, ret[%d]", __func__, ret);
    }
}

void TaskExceptionHost::ProcessCallback(rtExceptionInfo_t* exceptionInfo)
{
    CHK_PRT_RET(exceptionInfo == nullptr, HCCL_ERROR("[%s]fail, exceptionInfo is nullptr", __func__), );

    TaskExceptionHost* handler = TaskExceptionHost::GetInstance(exceptionInfo->deviceid);
    CHK_PRT_RET(handler == nullptr, HCCL_ERROR("[%s]fail, TaskExceptionHost is nullptr", __func__), );
    handler->Process(exceptionInfo);
}

HcclResult TaskExceptionHost::Register(u64 commHandle)
{
    std::unique_lock<std::mutex> lock(taskExceptionMutex_);
    if (CommRegisterMap_.empty()) {
        aclError ret = aclrtExceptionInfoCallbackRegister(ProcessCallback);
        CHK_PRT_RET(
            ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtExceptionInfoCallbackRegister failed, ret[%d]", __func__, ret),
            HCCL_E_RUNTIME);
        Hccl::HrtUnregTaskExceptionCallbackByModule(Hccl::TaskExceptionHandler::Process);
        HCCL_RUN_INFO(
            "[%s]aclrtExceptionInfoCallbackRegister set ProcessCallback success and unregister legacy "
            "TaskExceptionHandler::Process callback",
            __func__);
    }

    CommRegisterMap_.insert(commHandle);
    HCCL_INFO("[%s]success, commHandle[0x%llx]", __func__, commHandle);
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHost::UnRegister(u64 commHandle)
{
    std::unique_lock<std::mutex> lock(taskExceptionMutex_);
    if (CommRegisterMap_.find(commHandle) == CommRegisterMap_.end()) {
        HCCL_INFO("[%s] commHandle[0x%llx] has not registered, skip", __func__, commHandle);
        return HCCL_SUCCESS;
    }

    CommRegisterMap_.erase(commHandle);
    if (CommRegisterMap_.empty()) {
        aclError ret = aclrtExceptionInfoCallbackUnregister(ProcessCallback); // 注销注册给rts的TaskException回调
        CHK_PRT_RET(
            ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtExceptionInfoCallbackUnregister failed, ret[%d]", __func__, ret),
            HCCL_E_RUNTIME);
        HCCL_RUN_INFO("[%s]aclrtExceptionInfoCallbackUnregister success", __func__);
    }

    HCCL_INFO("[%s]success, commHandle[0x%llx]", __func__, commHandle);
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHost::PrintUbRegisters(s32 devLogicId, const RdmaHandle rdmaHandle) const
{
    HCCL_INFO("[PrintUbRegister] start, devLogicId[%d], rdmaHandle[%p]", devLogicId, rdmaHandle);
    Hccl::AuxInfoIn in;
    in.cqe.status = 0xffffffff; // 0xffffffff代表查询所有寄存器
    in.auxInfoInType = Hccl::AuxInfoInType::AUX_INFO_IN_TYPE_CQE;
    in.cqe.sR = 0;
    Hccl::AuxInfoOut auxInfo;
    auto ret = Hccl::RaGetAuxInfo(rdmaHandle, in, auxInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[PrintUbRegister]GetUbRegisterInfo failed, devLogicId[%d], rdmaHandle[%p]", devLogicId, rdmaHandle);
        return ret;
    }

    bool isAuxInfoExisted{false};
    for (u32 i = 0; i < auxInfo.auxInfoNum; i++) {
        if (auxInfo.auxInfoValues[i] != 0) { // 非零进行打印
            isAuxInfoExisted = true;
            HCCL_ERROR(
                "devLogicId[%d], cqe_aux_info_type[%u], cqe_aux_info_value[0x%x]", devLogicId, auxInfo.auxInfoTypes[i],
                auxInfo.auxInfoValues[i]);
        } else {
            HCCL_INFO(
                "devLogicId[%d], cqe_aux_info_type[%u], cqe_aux_info_value[0x%x]", devLogicId, auxInfo.auxInfoTypes[i],
                auxInfo.auxInfoValues[i]);
        }
    }
    if (!isAuxInfoExisted) {
        HCCL_ERROR("devLogicId[%d], all aux_info values are zero.", devLogicId);
    }
    return HCCL_SUCCESS;
}

bool IsMC2Exception(const rtExceptionInfo_t* exceptionInfo)
{
    return exceptionInfo->expandInfo.type == RT_EXCEPTION_FUSION
           && exceptionInfo->expandInfo.u.fusionInfo.type == RT_FUSION_AICORE_CCU;
}

bool TaskExceptionHost::ProcessDpuException(const rtExceptionInfo_t* exceptionInfo) const
{
    bool isDpuErr = false;
    errno_t ret = EOK;
    uint16_t hcclRet = 0;
    std::lock_guard<std::mutex> lock(g_serMapMutex);
    for (const auto& pairMap : g_taskExpMemMap) {
        auto innerIt = pairMap.second.find(exceptionInfo->deviceid);
        if (innerIt == pairMap.second.end() || innerIt->second == nullptr) {
            continue;
        }
        auto taskExpPtr = reinterpret_cast<uint8_t*>(innerIt->second);
        // 读取共享内存内容并打印
        ret = memcpy_s(&hcclRet, sizeof(uint16_t), taskExpPtr + sizeof(uint8_t) + sizeof(uint16_t), sizeof(uint16_t));
        if (ret != EOK) {
            HCCL_ERROR("memcpy_s get dpu taskexception failed: %d", ret);
            return isDpuErr;
        }
        if (hcclRet != 0) { // 有dpu任务出错
            HCCL_ERROR("[TaskExceptionHost][ProcessDpuException] Task from HCCL run failed.");
            HCCL_ERROR(
                "[TaskExceptionHost][ProcessDpuException] errorCode[%d], devId[%u], commId[%s]", hcclRet,
                exceptionInfo->deviceid, pairMap.first.c_str());
            ret = memset_s(
                taskExpPtr + sizeof(uint8_t) + sizeof(uint16_t), sizeof(uint16_t), 0,
                sizeof(uint16_t)); // 清空 dpu taskexception共享内存内容
            if (ret != EOK) {
                HCCL_ERROR("memset_s clean dpu taskexception failed: %d", ret);
                return isDpuErr;
            }
            isDpuErr = true;
            break;
        }
    }
    return isDpuErr;
}

void TaskExceptionHost::Process(rtExceptionInfo_t* exceptionInfo)
{
    HCCL_RUN_INFO(
        "[TaskExceptionHost][%s], taskid[%u], streamid[%u], tid[%u], deviceid[%u], retcode[%u], type[%d]", __func__,
        exceptionInfo->taskid, exceptionInfo->streamid, exceptionInfo->tid, exceptionInfo->deviceid,
        exceptionInfo->retcode, exceptionInfo->expandInfo.type);

    if (IsMC2Exception(exceptionInfo)) { // MC2 taskException 新流程暂未支持，回退到老流程
        Hccl::TaskExceptionHandler::Process(exceptionInfo);
        return;
    }

    // dpu taskexception
    if (ProcessDpuException(exceptionInfo)) {
        HCCL_ERROR("[TaskExceptionHost][ProcessDpuException] end.");
        return;
    };

    ProcessHcclTaskException(exceptionInfo);
}

void TaskExceptionHost::ProcessHcclTaskException(rtExceptionInfo_t* exceptionInfo)
{
    Hccl::TaskInfo* curTask = nullptr;
    HcclResult ret = Hccl::GlobalMirrorTasks::Instance().FindTaskInfo(
        exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid, curTask);
    CHK_PRT_RET(
        ret == HCCL_E_NOT_FOUND,
        HCCL_RUN_WARNING(
            "[%s]FindTaskInfo not found, deviceid[%u] streamid[%u] taskid[%u].", __func__, exceptionInfo->deviceid,
            exceptionInfo->streamid, exceptionInfo->taskid), );

    CHK_PRT_RET(
        ret != HCCL_SUCCESS || curTask == nullptr,
        HCCL_ERROR(
            "[%s]FindTaskInfo fail, ret[%d], deviceid[%u], streamid[%u], taskid[%u].", __func__, ret,
            exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid), );

    if (curTask->taskParam_.taskType == Hccl::TaskParamType::TASK_AIV) {
        HCCL_RUN_INFO(
            "[TaskExceptionHost][%s] AIV taskType, taskid[%u] streamid[%u] deviceid[%u].", __func__,
            exceptionInfo->taskid, exceptionInfo->streamid, exceptionInfo->deviceid);
        return;
    }

    CHK_PRT_RET(curTask->dfxOpInfo_ == nullptr, HCCL_ERROR("[%s]fail, dfxOpInfo is nullptr", __func__), );
    bool isIndop_ = curTask->dfxOpInfo_->isIndop_;
    HCCL_INFO("[%s]isIndop_[%d], taskType[%s]", __func__, isIndop_, curTask->taskParam_.taskType.Describe().c_str());

    // 老流程TaskException打印
    if (!isIndop_) {
        Hccl::TaskExceptionHandler::Process(exceptionInfo);
        return;
    }

    // 新流程支持的TaskException打印
    std::unique_lock<std::mutex> lock(taskExceptionMutex_);
    u64 commHandle = reinterpret_cast<u64>(curTask->dfxOpInfo_->comm_);
    if (curTask->dfxOpInfo_->comm_ == nullptr || CommRegisterMap_.find(commHandle) == CommRegisterMap_.end()) {
        HCCL_ERROR(
            "[TaskExceptionHost][%s] commHandle[0x%llx] not exist, "
            "the comm may have been destroyed or was never registered",
            __func__, commHandle);
        return;
    }

    if (curTask->taskParam_.taskType == Hccl::TaskParamType::TASK_CCU) {
        CcuTaskException::ProcessCcuException(exceptionInfo, *curTask);
    } else {
        ProcessException(exceptionInfo, *curTask);
    }
}

std::string TaskExceptionHost::GetGroupRankInfo(const Hccl::TaskInfo& taskInfo) const
{
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]TaskInfo communicator is nullptr.", __func__);
        return "";
    }

    hccl::CollComm* communicator = static_cast<hccl::CollComm*>(taskInfo.dfxOpInfo_->comm_);
    return Hccl::StringFormat(
        "group:[%s], rankSize[%u], rankId[%u]", communicator->GetCommId().c_str(), communicator->GetRankSize(),
        communicator->GetMyRankId());
}

void TaskExceptionHost::GetAicpuCqeErrRemoteLocalIdByRankId(
    hccl::CollComm* collComm, uint32_t rankid, u32& remoteLocalId) const
{
    if (collComm == nullptr || rankid == Hccl::DFX_INVALID_RANKID) {
        HCCL_ERROR("[GetAicpuCqeErrRemoteLocalIdByRankId]collComm is nullptr or rankId is invalid, rankId[%u]", rankid);
        remoteLocalId = Hccl::DFX_INVALID_RANKID;
        return;
    }

    Hccl::HcclCommunicator* commV2 = static_cast<Hccl::HcclCommunicator*>(collComm->GetCommunicatorV2());
    if (commV2 == nullptr) {
        HCCL_ERROR("[GetAicpuCqeErrRemoteLocalIdByRankId]commV2 is nullptr, rankId[%u]", rankid);
        remoteLocalId = Hccl::DFX_INVALID_RANKID;
        return;
    }
    void* rankGraph = nullptr;
    HcclResult ret = commV2->GetRankGraphV2(rankGraph);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[GetAicpuCqeErrRemoteLocalIdByRankId]GetRankGraphV2 failed, rankId[%u], ret[%d]", rankid, ret);
        remoteLocalId = Hccl::DFX_INVALID_RANKID;
        return;
    }
    Hccl::RankGraph* rankGraphv2 = static_cast<Hccl::RankGraph*>(rankGraph);
    u32 LocalId = rankGraphv2->GetLocalId(rankid);
    remoteLocalId = LocalId;
    return;
}

void TaskExceptionHost::GetAicpuCqeErrNetInstanceByRankId(
    hccl::CollComm* collComm, uint32_t rankid, std::string& netInstanceId) const
{
    if (collComm == nullptr || rankid == Hccl::DFX_INVALID_RANKID) {
        HCCL_ERROR("[GetAicpuCqeErrNetInstanceByRankId]collComm is nullptr or rankId is invalid, rankId[%u]", rankid);
        netInstanceId = "";
        return;
    }

    Hccl::HcclCommunicator* commV2 = static_cast<Hccl::HcclCommunicator*>(collComm->GetCommunicatorV2());
    if (commV2 == nullptr) {
        HCCL_ERROR("[GetAicpuCqeErrNetInstanceByRankId]commV2 is nullptr, rankId[%u]", rankid);
        netInstanceId = "";
        return;
    }
    void* rankGraph = nullptr;
    HcclResult ret = commV2->GetRankGraphV2(rankGraph);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[GetAicpuCqeErrNetInstanceByRankId]GetRankGraphV2 failed, rankId[%u], ret[%d]", rankid, ret);
        netInstanceId = "";
        return;
    }
    Hccl::RankGraph* rankGraphv2 = static_cast<Hccl::RankGraph*>(rankGraph);
    const Hccl::NetInstance* netInstance = rankGraphv2->GetNetInstanceByRankId(0, rankid);
    if (netInstance == nullptr) {
        HCCL_ERROR("[GetAicpuCqeErrNetInstanceByRankId]netInstance is nullptr, rankId[%u]", rankid);
        netInstanceId = "";
        return;
    }
    std::string netInsId = netInstance->GetNetInstId();
    netInstanceId = netInsId;
    return;
}

void TaskExceptionHost::GetAicpuCqeErrInfo(
    rtExceptionInfo_t* exceptionInfo, const Hccl::ErrorMessageReport& errorMessage,
    const Hccl::TaskInfo& taskInfo) const
{
    hccl::CollComm* collComm = static_cast<hccl::CollComm*>(taskInfo.dfxOpInfo_->comm_);
    u32 remoteLocalId = Hccl::DFX_INVALID_RANKID;
    GetAicpuCqeErrRemoteLocalIdByRankId(collComm, errorMessage.remoteUserRank, remoteLocalId);
    std::string netInstanceId = "";
    GetAicpuCqeErrNetInstanceByRankId(collComm, errorMessage.remoteUserRank, netInstanceId);
    ClusterMoniterGetAicpuCqeErrInfo(
        remoteLocalId, exceptionInfo->deviceid, errorMessage.ubCqeStatus, errorMessage.locEid.Describe(),
        errorMessage.rmtEid.Describe(), netInstanceId); // 上报AICPU CQE错误信息到集群监控
    return;
}

void TaskExceptionHost::NotifyControlPlaneOnUbError(
    u32 devPhyId, RdmaHandle rdmaHandle, const Hccl::ErrorMessageReport& errorMessage) const
{
    if (errorMessage.ubCqeStatus == 0) {
        return;
    }
    struct RaInfo raInfo = {};
    raInfo.mode = NETWORK_OFFLINE;
    raInfo.phyId = devPhyId;

    const bool supported = RaHasCapability(&raInfo, RA_CAP_UDMA_NOTIFY_EVENT);
    if (!supported) {
        HCCL_WARNING("[%s]RaHasCapability returned false, skip notify control plane, devPhyId[%u]", __func__, devPhyId);
        return;
    }

    if (rdmaHandle == nullptr) {
        HCCL_ERROR(
            "[%s]rdmaHandle is nullptr, skip notify control plane, devPhyId[%u], locEid[%s]", __func__, devPhyId,
            errorMessage.locEid.Describe().c_str());
        return;
    }

    struct CtxNotifyEvent event = {};
    event.serviceType = URMA_TYPE;
    event.errorType = static_cast<uint8_t>(errorMessage.ubCqeStatus);
    s32 sRet = memcpy_s(
        event.srcEid.raw, sizeof(event.srcEid.raw), errorMessage.locEid.raw, sizeof(errorMessage.locEid.raw));
    if (sRet != EOK) {
        HCCL_ERROR("[%s]memcpy_s srcEid failed, ret[%d]", __func__, sRet);
        return;
    }
    sRet = memcpy_s(
        event.dstEid.raw, sizeof(event.dstEid.raw), errorMessage.rmtEid.raw, sizeof(errorMessage.rmtEid.raw));
    if (sRet != EOK) {
        HCCL_ERROR("[%s]memcpy_s dstEid failed, ret[%d]", __func__, sRet);
        return;
    }
    event.errorInfo.tpn = errorMessage.tpn;

    int32_t retCode = RaCtxNotifyEvent(rdmaHandle, &event);
    std::string eventInfo = Hccl::StringFormat(
        "devPhyId[%u], rdmaHandle[%p], serviceType[%u], errorType[%u], tpn[%u], "
        "srcEid[%s], dstEid[%s]",
        devPhyId, static_cast<const void*>(rdmaHandle), event.serviceType, event.errorType, errorMessage.tpn,
        errorMessage.locEid.Describe().c_str(), errorMessage.rmtEid.Describe().c_str());
    if (retCode != 0) {
        HCCL_ERROR("[%s]RaCtxNotifyEvent failed, ret[%d], %s", __func__, retCode, eventInfo.c_str());
        return;
    }
    HCCL_ERROR("[%s]notify control plane finish, %s", __func__, eventInfo.c_str());
}

void TaskExceptionHost::ProcessException(rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& taskInfo)
{
    HCCL_RUN_INFO("[TaskExceptionHost][%s]begin to execute hccl task exception callback function.", __func__);
    Hccl::ErrorMessageReport errorMessage;
    if (!hasAicpuReport_) { // 防止aicpu task exception重复上报
        hccl::CollComm* communicator = static_cast<hccl::CollComm*>(taskInfo.dfxOpInfo_->comm_);
        errorMessage = communicator->GetAicpuTaskException();
    }

    if (strlen(errorMessage.tag) > 0) {
        hasAicpuReport_ = true;
        HCCL_RUN_INFO("[TaskExceptionHost]ErrorMessageReport: %s", errorMessage.Describe().c_str());
        HandleAicpuErrorReport(exceptionInfo, errorMessage, taskInfo);
    } else {
        HandleHostErrorReport(exceptionInfo, taskInfo);
    }
}

void TaskExceptionHost::HandleHostErrorReport(rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& taskInfo) const
{
    const auto& logKeywordL2 = taskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_NOTIFY_WAIT ?
                                   LOG_KEYWORDS_TIMEOUT :
                                   LOG_KEYWORDS_RUN_FAILED;
    std::string logKeywordL3 = LOG_KEYWORDS_HOST;
    if (taskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_AIV) {
        logKeywordL3 = LOG_KEYWORDS_AIV;
    } else if (taskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_CCU) {
        logKeywordL3 = LOG_KEYWORDS_CCU;
    }
    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + logKeywordL2 + "][" + logKeywordL3 + "]";
    HCCL_ERROR("%s Task from HCCL run failed.", stageErrInfo.c_str());
    if (taskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_NOTIFY_WAIT) {
        if (ShouldReportError()) {
            PrintTaskContextInfo(exceptionInfo->deviceid, exceptionInfo->streamid, exceptionInfo->taskid, stageErrInfo);
            RPT_INPUT_ERR(
                true, "EI0002",
                std::vector<std::string>(
                    {"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                std::vector<std::string>(
                    {std::to_string(taskInfo.remoteRank_), taskInfo.GetBaseInfo(), (taskInfo.GetParaInfo()), ""}));
        } else {
            HCCL_WARNING(
                "[TaskExceptionHost] EI0002 already reported on device[%u], skip duplicate.", exceptionInfo->deviceid);
        }
    }
    std::string clusterMonitorErrMsg = AicpuGetAndPrintClusterMonitorErr(exceptionInfo);
    HCCL_ERROR(
        "%s Task run failed, base information is deviceID:[%u], %s.", stageErrInfo.c_str(), exceptionInfo->deviceid,
        taskInfo.GetIndopBaseInfo().c_str());
    HCCL_ERROR("%s Task run failed, para information is %s.", stageErrInfo.c_str(), taskInfo.GetParaInfo().c_str());
    HCCL_ERROR(
        "%s Task run failed, groupRank information is %s.", stageErrInfo.c_str(), GetGroupRankInfo(taskInfo).c_str());
    HCCL_ERROR(
        "%s Task run failed, opData information is %s.", stageErrInfo.c_str(), taskInfo.GetIndopDataInfo().c_str());
}

void TaskExceptionHost::PrintTaskContextInfo(
    uint32_t deviceId, uint32_t streamId, uint32_t taskId, const std::string& stageErrInfo) const
{
    Hccl::TaskInfoQueue* queue = nullptr;
    try {
        queue = Hccl::GlobalMirrorTasks::Instance().GetQueue(deviceId, streamId);
    } catch (Hccl::HcclException& e) {
        HCCL_ERROR("Exception task queue not found. deviceId[%u], streamId[%u].", deviceId, streamId);
        return;
    }

    if (queue == nullptr) {
        // 未找到异常对应的TaskQueue
        HCCL_ERROR("Exception task queue not found. deviceId[%u], streamId[%u].", deviceId, streamId);
        return;
    }

    auto func = [taskId](const unique_ptr<Hccl::TaskInfo>& task) {
        return task->taskId_ == taskId;
    };
    auto taskIterPtr = queue->Find(func);
    if (taskIterPtr == nullptr || *taskIterPtr == *queue->End()) {
        // 在队列中未找到异常对应的TaskInfo
        HCCL_ERROR("Exception task not found. deviceId[%u], streamId[%u], taskId[%u].", deviceId, streamId, taskId);
        return;
    }

    // 找到当前异常task的前50个task(至多)
    vector<Hccl::TaskInfo*> taskContext{};
    for (uint32_t i = 0; i < TASK_CONTEXT_SIZE && *taskIterPtr != *queue->Begin(); ++i, --(*taskIterPtr)) {
        if ((**taskIterPtr)->taskId_ > taskId) {
            HCCL_ERROR(
                "[%s]prev taskId[%u] is bigger than err taskId[%u], traversal end.", __func__, (**taskIterPtr)->taskId_,
                taskId);
            break;
        }
        taskContext.emplace_back((**taskIterPtr).get());
    }

    HCCL_ERROR(
        "%s Task run failed, context sequence before error task is "
        "[SDMA:M(rank), RDMA:RS(rank,id), SendPayload:SP(rank), InlineReduce:IR(rank), Reduce:R(rank), "
        "NotifyRecord:NR(rank,id), NotifyWait:NW(rank,id), SendNotify:SN(rank,id), "
        "WriteWithNotify:WN(rank,id), WriteReduceWithNotify:WRN(rank,id)]:",
        stageErrInfo.c_str());

    std::string taskContextInfo = "";
    for (auto it = taskContext.rbegin(); it != taskContext.rend(); ++it) {
        std::string conciseInfo = (*it)->GetConciseBaseInfo();
        conciseInfo += ",";

        if (taskContextInfo.size() + conciseInfo.size() >= TASK_CONTEXT_INFO_SIZE) {
            HCCL_ERROR("%s %s", stageErrInfo.c_str(), taskContextInfo.c_str());
            taskContextInfo = "";
        }

        taskContextInfo += conciseInfo;
    }
    HCCL_ERROR("%s %s end.", stageErrInfo.c_str(), taskContextInfo.c_str());
}

inline void PrintBaseErrorLog(const std::string& stageErrInfo, const std::string& baseInfo)
{
    HCCL_ERROR("%s Task run failed, base information is %s", stageErrInfo.c_str(), baseInfo.c_str());
}

inline void PrintParaErrorLog(const std::string& stageErrInfo, const std::string& paraInfoStr)
{
    HCCL_ERROR("%s Task run failed, para information is %s.", stageErrInfo.c_str(), paraInfoStr.c_str());
}

inline void PrintOpDataErrorLog(const std::string& stageErrInfo, const std::string& opDataContent)
{
    HCCL_ERROR("%s Task run failed, opData information is %s", stageErrInfo.c_str(), opDataContent.c_str());
}

inline void PrintGroupErrorLog(const std::string& stageErrInfo, const std::string& groupRankContent)
{
    HCCL_ERROR("%s Task run failed, groupRank information is %s.", stageErrInfo.c_str(), groupRankContent.c_str());
}

void TaskExceptionHost::PrintGroupErrorMessage(
    const Hccl::ErrorMessageReport& errorMessage, [[maybe_unused]] const Hccl::TaskInfo& exceptionTaskInfo,
    std::string& groupRankContent, std::string& stageErrInfo) const
{
    groupRankContent += "group:[";
    groupRankContent += std::string(errorMessage.group);
    groupRankContent += "], rankSize[";
    groupRankContent += std::to_string(errorMessage.rankSize);
    groupRankContent += "], localRank[";
    groupRankContent += std::to_string(errorMessage.rankId);
    groupRankContent += "], remoteRank[";
    groupRankContent += errorMessage.remoteUserRank == Hccl::DFX_INVALID_RANKID ?
                            "local" :
                            std::to_string(errorMessage.remoteUserRank);
    groupRankContent += "]";

    PrintGroupErrorLog(stageErrInfo, groupRankContent);
    return;
}

const std::map<HcclReduceOp, std::string> HCOM_REDUCE_OP_STR_MAP{
    {HcclReduceOp::HCCL_REDUCE_SUM, "sum"},
    {HcclReduceOp::HCCL_REDUCE_PROD, "prod"},
    {HcclReduceOp::HCCL_REDUCE_MAX, "max"},
    {HcclReduceOp::HCCL_REDUCE_MIN, "min"},
    {HcclReduceOp::HCCL_REDUCE_RESERVED, "invalid"}};

inline std::string GetReduceOpEnumStr2(HcclReduceOp reduceOp)
{
    auto iter = HCOM_REDUCE_OP_STR_MAP.find(reduceOp);
    if (iter == HCOM_REDUCE_OP_STR_MAP.end()) {
        return "HcclReduceOp(" + std::to_string(reduceOp) + ")";
    } else {
        return iter->second;
    }
}

const std::map<HcclDataType, std::string> HCOM_DATA_TYPE_STR_MAP{
    {HcclDataType::HCCL_DATA_TYPE_INT8, "int8"},       {HcclDataType::HCCL_DATA_TYPE_INT16, "int16"},
    {HcclDataType::HCCL_DATA_TYPE_INT32, "int32"},     {HcclDataType::HCCL_DATA_TYPE_INT64, "int64"},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, "uint64"},   {HcclDataType::HCCL_DATA_TYPE_FP16, "float16"},
    {HcclDataType::HCCL_DATA_TYPE_FP32, "float32"},    {HcclDataType::HCCL_DATA_TYPE_UINT8, "uint8"},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, "uint16"},   {HcclDataType::HCCL_DATA_TYPE_UINT32, "uint32"},
    {HcclDataType::HCCL_DATA_TYPE_FP64, "float64"},    {HcclDataType::HCCL_DATA_TYPE_BFP16, "bfloat16"},
    {HcclDataType::HCCL_DATA_TYPE_INT128, "int128"},   {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, "fp8e4m3"},
    {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, "fp8e5m2"}, {HcclDataType::HCCL_DATA_TYPE_RESERVED, "reserved"}};

inline std::string GetDataTypeEnumStr2(HcclDataType dataType)
{
    auto iter = HCOM_DATA_TYPE_STR_MAP.find(dataType);
    if (iter == HCOM_DATA_TYPE_STR_MAP.end()) {
        return "HcclDataType(" + std::to_string(dataType) + ")";
    } else {
        return iter->second;
    }
}

inline std::string GetDataTypeEnumStr(u32 dataType)
{
    auto hcclDataType = static_cast<HcclDataType>(dataType);
    return GetDataTypeEnumStr2(hcclDataType);
}
inline std::string GetOpTypeEnumStr(u32 opType)
{
    Hccl::OpType hcclOpType = static_cast<Hccl::OpType::Value>(opType);
    return hcclOpType.Describe();
}

void TaskExceptionHost::PrintOpDataErrorMessage(
    u32 deviceId, const Hccl::ErrorMessageReport& errorMessage, std::string& stageErrInfo) const
{
    std::stringstream opDataStr;
    opDataStr << "src" << "[0x" << std::hex << errorMessage.srcAddr << "], dst[0x" << std::hex << errorMessage.dstAddr
              << "], ";

    std::string opStr;
    if (errorMessage.reduceType != HcclReduceOp::HCCL_REDUCE_RESERVED) {
        opStr += "reduceType[";
        opStr += GetReduceOpEnumStr2(static_cast<HcclReduceOp>(errorMessage.reduceType));
        opStr += "], ";
    }

    std::string opDataContent;
    opDataContent += "deviceId:[";
    opDataContent += std::to_string(deviceId);
    opDataContent += "], index[";
    opDataContent += std::to_string(errorMessage.opIndex);
    opDataContent += "], count[";
    opDataContent += std::to_string(errorMessage.count);
    opDataContent += "], ";
    opDataContent += opStr;
    opDataContent += opDataStr.str();
    opDataContent += "dataType[";
    opDataContent += GetDataTypeEnumStr(errorMessage.dataType);
    opDataContent += "], tag[";
    opDataContent += errorMessage.tag;
    opDataContent += "].";

    PrintOpDataErrorLog(stageErrInfo, opDataContent);
    return;
}

void TaskExceptionHost::ReportErrorMsg(
    const Hccl::TaskInfo& exceptionTaskInfo, [[maybe_unused]] const std::string& groupRankContent,
    const Hccl::ErrorMessageReport& errorMessage, rtExceptionInfo_t* exceptionInfo,
    const std::string& stageErrInfo) const
{
    HCCL_RUN_INFO(
        "[ReportErrorMsg] start, taskType[%s]",
        Hccl::TaskParamTypeValToStr(static_cast<u8>(exceptionTaskInfo.taskParam_.taskType)).c_str());

    std::string clusterMonitorErrMsg = AicpuGetAndPrintClusterMonitorErr(exceptionInfo);

    if (exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_NOTIFY_WAIT) {
        if (ShouldReportError()) {
            HCCL_ERROR("[ReportErrorMsg] EI0002");
            RPT_INPUT_ERR(
                true, "EI0002",
                std::vector<std::string>(
                    {"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                std::vector<std::string>(
                    {std::to_string(exceptionTaskInfo.remoteRank_), exceptionTaskInfo.GetIndopBaseInfo().c_str(),
                     (exceptionTaskInfo.GetParaInfo() + clusterMonitorErrMsg).c_str(), ""}));
        } else {
            HCCL_WARNING(
                "[ReportErrorMsg] EI0002 already reported on device[%u], skip duplicate.", exceptionInfo->deviceid);
        }
    } else if (
        exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY
        || exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_WRITE_WITH_NOTIFY
        || exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_UB_INLINE_WRITE
        || exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_UB_REDUCE_INLINE
        || exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_UB) {
        ReportEI0018Error(exceptionTaskInfo, errorMessage, stageErrInfo);
    }
}

bool TaskExceptionHost::ShouldReportError() const
{
    std::lock_guard<std::mutex> lock(ei0002ReportedMutex_);
    if (!ei0002Reported_) {
        ei0002Reported_ = true;
        return true;
    }
    return false;
}

void TaskExceptionHost::ReportEI0018Error(
    const Hccl::TaskInfo& exceptionTaskInfo, const Hccl::ErrorMessageReport& errorMessage,
    const std::string& stageErrInfo) const
{
    hccl::CollComm* collComm = static_cast<hccl::CollComm*>(exceptionTaskInfo.dfxOpInfo_->comm_);
    std::string localServerId = "";
    GetAicpuCqeErrNetInstanceByRankId(collComm, errorMessage.rankId, localServerId);
    u32 localDeviceId = Hccl::DFX_INVALID_RANKID;
    GetAicpuCqeErrRemoteLocalIdByRankId(collComm, errorMessage.rankId, localDeviceId);
    std::string remoteServerId = "";
    GetAicpuCqeErrNetInstanceByRankId(collComm, errorMessage.remoteUserRank, remoteServerId);
    u32 remoteDeviceId = Hccl::DFX_INVALID_RANKID;
    GetAicpuCqeErrRemoteLocalIdByRankId(collComm, errorMessage.remoteUserRank, remoteDeviceId);
    Hccl::IpAddress localAddr(errorMessage.locEid);
    Hccl::IpAddress remoteAddr(errorMessage.rmtEid);
    RPT_INPUT_ERR(
        true, "EI0018",
        std::vector<std::string>(
            {"localServerId", "localDeviceId", "localDeviceIp", "remoteServerId", "remoteDeviceId", "remoteDeviceIp"}),
        std::vector<std::string>(
            {localServerId, std::to_string(localDeviceId), localAddr.GetReverseEid().Describe().c_str(), remoteServerId,
             std::to_string(remoteDeviceId), remoteAddr.GetReverseEid().Describe().c_str()}));
    HCCL_ERROR(
        "%s UB CQE error, localServerId[%s] deviceId[%u] ip[%s], "
        "remoteServerId[%s] deviceId[%u] ip[%s]",
        stageErrInfo.c_str(), localServerId.c_str(), localDeviceId, localAddr.GetReverseEid().Describe().c_str(),
        remoteServerId.c_str(), remoteDeviceId, remoteAddr.GetReverseEid().Describe().c_str());
}

void GetTaskParam(Hccl::TaskParam& taskParam, const Hccl::ErrorMessageReport& errMsgInfo)
{
    switch (errMsgInfo.taskType) {
        case Hccl::TaskParamType::TASK_NOTIFY_WAIT:
        case Hccl::TaskParamType::TASK_NOTIFY_RECORD:
            taskParam.taskPara.Notify.notifyID = errMsgInfo.notifyId;
            taskParam.taskPara.Notify.value = errMsgInfo.notifyValue;
            break;
        case Hccl::TaskParamType::TASK_UB_REDUCE_INLINE:
        case Hccl::TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY:
            taskParam.taskPara.Reduce.notifyID = errMsgInfo.notifyId;
            taskParam.taskPara.Reduce.notifyValue = errMsgInfo.notifyValue;
            taskParam.taskPara.Reduce.locEid = errMsgInfo.locEid;
            taskParam.taskPara.Reduce.rmtEid = errMsgInfo.rmtEid;
            taskParam.taskPara.Reduce.linkType = errMsgInfo.linkType;
            taskParam.taskPara.Reduce.size = errMsgInfo.size;
            taskParam.taskPara.Reduce.src = reinterpret_cast<void*>(errMsgInfo.taskSrcAddr);
            taskParam.taskPara.Reduce.dst = reinterpret_cast<void*>(errMsgInfo.taskDstAddr);
            break;
        case Hccl::TaskParamType::TASK_REDUCE_INLINE:
            taskParam.taskPara.Reduce.reduceOp = static_cast<HcclReduceOp>(errMsgInfo.reduceType);
            break;
        case Hccl::TaskParamType::TASK_UB_INLINE_WRITE:
        case Hccl::TaskParamType::TASK_WRITE_WITH_NOTIFY:
            taskParam.taskPara.DMA.notifyID = errMsgInfo.notifyId;
            taskParam.taskPara.DMA.notifyValue = errMsgInfo.notifyValue;
            taskParam.taskPara.DMA.locEid = errMsgInfo.locEid;
            taskParam.taskPara.DMA.rmtEid = errMsgInfo.rmtEid;
            taskParam.taskPara.DMA.linkType = errMsgInfo.linkType;
            taskParam.taskPara.DMA.size = errMsgInfo.size;
            taskParam.taskPara.DMA.src = reinterpret_cast<void*>(errMsgInfo.taskSrcAddr);
            taskParam.taskPara.DMA.dst = reinterpret_cast<void*>(errMsgInfo.taskDstAddr);
            break;
        case Hccl::TaskParamType::TASK_UB:
            taskParam.taskPara.DMA.locEid = errMsgInfo.locEid;
            taskParam.taskPara.DMA.rmtEid = errMsgInfo.rmtEid;
            taskParam.taskPara.DMA.linkType = errMsgInfo.linkType;
            taskParam.taskPara.DMA.size = errMsgInfo.size;
            taskParam.taskPara.DMA.src = reinterpret_cast<void*>(errMsgInfo.taskSrcAddr);
            taskParam.taskPara.DMA.dst = reinterpret_cast<void*>(errMsgInfo.taskDstAddr);
            break;
        case Hccl::TaskParamType::TASK_SDMA:
            taskParam.taskPara.DMA.linkType = errMsgInfo.linkType;
            taskParam.taskPara.DMA.size = errMsgInfo.size;
            taskParam.taskPara.DMA.src = reinterpret_cast<void*>(errMsgInfo.taskSrcAddr);
            taskParam.taskPara.DMA.dst = reinterpret_cast<void*>(errMsgInfo.taskDstAddr);
            break;
        default:
            HCCL_ERROR(
                "[TaskException][HOST]%s taskType[%s] is not supported", __func__,
                Hccl::TaskParamTypeValToStr(static_cast<u8>(taskParam.taskType)).c_str());
            return;
    }
}

void TaskExceptionHost::HandleAicpuErrorReport(
    rtExceptionInfo_t* exceptionInfo, const Hccl::ErrorMessageReport& errorMessage,
    const Hccl::TaskInfo& taskInfo) const
{
    std::string groupRankContent;
    u32 streamId = static_cast<u32>(errorMessage.streamId);
    std::string tag = std::string(errorMessage.tag);
    Hccl::TaskParam taskParam{};
    taskParam.taskType = errorMessage.taskType;
    GetTaskParam(taskParam, errorMessage);
    std::shared_ptr<Hccl::DfxOpInfo> dfxOpInfo = std::make_shared<Hccl::DfxOpInfo>();
    dfxOpInfo->tag_ = tag;
    dfxOpInfo->comm_ = taskInfo.dfxOpInfo_->comm_;
    Hccl::TaskInfo exceptionTaskInfo(streamId, errorMessage.taskId, errorMessage.remoteUserRank, taskParam, dfxOpInfo);
    const auto& logKeywordL2 = exceptionTaskInfo.taskParam_.taskType == Hccl::TaskParamType::TASK_NOTIFY_WAIT ?
                                   LOG_KEYWORDS_TIMEOUT :
                                   LOG_KEYWORDS_RUN_FAILED;
    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + logKeywordL2 + "][" + LOG_KEYWORDS_AICPU + "]";
    HCCL_ERROR("%s Task from HCCL run failed.", stageErrInfo.c_str());
    // 防止tag字符串过长， 信息分开打印
    PrintBaseErrorLog(stageErrInfo, exceptionTaskInfo.GetIndopBaseInfo());
    PrintParaErrorLog(stageErrInfo, exceptionTaskInfo.GetParaInfo());
    PrintGroupErrorMessage(errorMessage, exceptionTaskInfo, groupRankContent, stageErrInfo);
    PrintOpDataErrorMessage(exceptionInfo->deviceid, errorMessage, stageErrInfo);
    HCCL_ERROR(
        "errorMessage taskType[%s], rtCqErrorType[%u], rtCqErrorCode[%u]. ", errorMessage.taskType.Describe().c_str(),
        static_cast<u32>(errorMessage.rtCqErrorType), errorMessage.rtCqErrorCode);

    // 打印UB DFX寄存器信息
    PrintUbDfxInfo(exceptionInfo, errorMessage);
    ReportErrorMsg(exceptionTaskInfo, groupRankContent, errorMessage, exceptionInfo, stageErrInfo);
    if (errorMessage.ubCqeStatus != 0) {
        GetAicpuCqeErrInfo(exceptionInfo, errorMessage, taskInfo);
    }
}

void TaskExceptionHost::PrintUbDfxInfo(
    rtExceptionInfo_t* exceptionInfo, const Hccl::ErrorMessageReport& errorMessage) const
{
    if (errorMessage.taskType == Hccl::TaskParamType::TASK_WRITE_WITH_NOTIFY
        || errorMessage.taskType == Hccl::TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY
        || errorMessage.taskType == Hccl::TaskParamType::TASK_UB_INLINE_WRITE
        || errorMessage.taskType == Hccl::TaskParamType::TASK_UB_REDUCE_INLINE
        || errorMessage.taskType == Hccl::TaskParamType::TASK_UB) {
        HCCL_ERROR(
            "errorMessage ubCqeStatus[%u], localEid[%s], remoteEid[%s]. ", static_cast<u32>(errorMessage.ubCqeStatus),
            errorMessage.locEid.Describe().c_str(), errorMessage.rmtEid.Describe().c_str());
        auto addr = Hccl::IpAddress(errorMessage.locEid);
        u32 devPhyId = Hccl::HrtGetDevicePhyIdByUserDevId(exceptionInfo->deviceid);
        auto rdmaHandle = Hccl::RdmaHandleManager::GetInstance().GetByIp(devPhyId, addr);
        HrtRaDumpJettyContext(reinterpret_cast<void*>(errorMessage.jettyHandle), errorMessage.jettyId);
        PrintUbRegisters(static_cast<s32>(exceptionInfo->deviceid), rdmaHandle);
        NotifyControlPlaneOnUbError(devPhyId, rdmaHandle, errorMessage);
    }
}
} // namespace hcomm
