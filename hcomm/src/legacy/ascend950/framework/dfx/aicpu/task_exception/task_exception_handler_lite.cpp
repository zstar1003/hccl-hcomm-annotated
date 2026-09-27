/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_exception_handler_lite.h"
#include "task_exception_func.h"
#include "communicator_impl_lite.h"
#include "error_message_v2.h"
#include "dlhal_function_v2.h"
#include "task_param.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

#include "task_struct_v2.h"
#include "hcomm_task_scheduler_error.h"
#include "orion_adapter_hal.h"
#include "internal_exception.h"

namespace Hccl {

using namespace std;

constexpr uint32_t TASK_CONTEXT_SIZE = 50;
constexpr uint32_t TASK_CONTEXT_INFO_SIZE = LOG_TMPBUF_SIZE - 50; // task 执行失败时打印前序task信息的长度限制
constexpr uint32_t UB_CQE_STATUS_MASK = 0xFF;                     // ubCqeStatus取低8位

TaskExceptionHandlerLite& TaskExceptionHandlerLite::GetInstance()
{
    static TaskExceptionHandlerLite instance;
    return instance;
}

TaskExceptionHandlerLite::TaskExceptionHandlerLite() { Register(); }

TaskExceptionHandlerLite::~TaskExceptionHandlerLite() {}

void TaskExceptionHandlerLite::Register() const { TaskExceptionFunc::GetInstance().RegisterCallback(Process); }

void GetUbErrMsgInfo(TaskInfo* taskInfo, ErrorMessageReport& errMsgInfo, const rtLogicCqReport_t* exceptionInfo)
{
    if (taskInfo->taskParam_.taskType == TaskParamType::TASK_WRITE_WITH_NOTIFY
        || taskInfo->taskParam_.taskType == TaskParamType::TASK_UB_INLINE_WRITE
        || taskInfo->taskParam_.taskType == TaskParamType::TASK_UB) {
        errMsgInfo.locEid = taskInfo->taskParam_.taskPara.DMA.locEid;
        errMsgInfo.rmtEid = taskInfo->taskParam_.taskPara.DMA.rmtEid;
        errMsgInfo.ubCqeStatus = exceptionInfo->errorCode & UB_CQE_STATUS_MASK;
        errMsgInfo.linkType = taskInfo->taskParam_.taskPara.DMA.linkType;
        errMsgInfo.size = taskInfo->taskParam_.taskPara.DMA.size;
        errMsgInfo.taskSrcAddr = reinterpret_cast<u64>(taskInfo->taskParam_.taskPara.DMA.src);
        errMsgInfo.taskDstAddr = reinterpret_cast<u64>(taskInfo->taskParam_.taskPara.DMA.dst);
    } else if (
        taskInfo->taskParam_.taskType == TaskParamType::TASK_UB_REDUCE_INLINE
        || taskInfo->taskParam_.taskType == TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY) {
        errMsgInfo.locEid = taskInfo->taskParam_.taskPara.Reduce.locEid;
        errMsgInfo.rmtEid = taskInfo->taskParam_.taskPara.Reduce.rmtEid;
        errMsgInfo.ubCqeStatus = exceptionInfo->errorCode & UB_CQE_STATUS_MASK;
        errMsgInfo.linkType = taskInfo->taskParam_.taskPara.Reduce.linkType;
        errMsgInfo.size = taskInfo->taskParam_.taskPara.Reduce.size;
        errMsgInfo.taskSrcAddr = reinterpret_cast<u64>(taskInfo->taskParam_.taskPara.Reduce.src);
        errMsgInfo.taskDstAddr = reinterpret_cast<u64>(taskInfo->taskParam_.taskPara.Reduce.dst);
    }
    errMsgInfo.rtCqErrorType = exceptionInfo->errorType;
    errMsgInfo.rtCqErrorCode = exceptionInfo->errorCode;
}

HcclResult GenerateErrorMessageReport(
    const CommunicatorImplLite* aicpuComm, TaskInfo* taskInfo, ErrorMessageReport& errMsgInfo,
    const rtLogicCqReport_t* exceptionInfo)
{
    // 获取需要上报的关键信息
    errMsgInfo.remoteUserRank = taskInfo->remoteRank_;
    errMsgInfo.streamId = taskInfo->streamId_;
    errMsgInfo.taskId = taskInfo->taskId_;
    errMsgInfo.rankId = aicpuComm->GetMyRank();
    errMsgInfo.rankSize = aicpuComm->GetRankSize();
    errMsgInfo.opIndex = taskInfo->dfxOpInfo_ == nullptr ? 0 : taskInfo->dfxOpInfo_->commIndex_;
    errMsgInfo.opType = taskInfo->dfxOpInfo_->op_.opType;
    errMsgInfo.count = taskInfo->dfxOpInfo_->op_.dataCount;
    errMsgInfo.dataType = taskInfo->dfxOpInfo_->op_.dataType;
    errMsgInfo.srcAddr = static_cast<u64>(taskInfo->dfxOpInfo_->op_.inputMem->GetAddr());
    errMsgInfo.dstAddr = static_cast<u64>(taskInfo->dfxOpInfo_->op_.outputMem->GetAddr());
    errMsgInfo.taskType = taskInfo->taskParam_.taskType;

    if (taskInfo->taskParam_.taskType == TaskParamType::TASK_NOTIFY_WAIT) {
        errMsgInfo.notifyId = taskInfo->taskParam_.taskPara.Notify.notifyID;
        errMsgInfo.notifyValue = taskInfo->taskParam_.taskPara.Notify.value;
    } else if (
        taskInfo->taskParam_.taskType == TaskParamType::TASK_UB_REDUCE_INLINE
        || taskInfo->taskParam_.taskType == TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY) {
        errMsgInfo.notifyId = taskInfo->taskParam_.taskPara.Reduce.notifyID;
        errMsgInfo.notifyValue = taskInfo->taskParam_.taskPara.Reduce.notifyValue;
    } else if (
        taskInfo->taskParam_.taskType == TaskParamType::TASK_UB_INLINE_WRITE
        || taskInfo->taskParam_.taskType == TaskParamType::TASK_WRITE_WITH_NOTIFY) {
        errMsgInfo.notifyId = taskInfo->taskParam_.taskPara.DMA.notifyID;
        errMsgInfo.notifyValue = taskInfo->taskParam_.taskPara.DMA.notifyValue;
    }

    if (taskInfo->taskParam_.taskType == TaskParamType::TASK_UB_REDUCE_INLINE
        || taskInfo->taskParam_.taskType == TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY
        || taskInfo->taskParam_.taskType == TaskParamType::TASK_REDUCE_INLINE) {
        errMsgInfo.reduceType = taskInfo->taskParam_.taskPara.Reduce.reduceOp;
    }

    memcpy_s(errMsgInfo.tag, sizeof(errMsgInfo.tag), aicpuComm->GetId().c_str(), aicpuComm->GetId().size());
    memcpy_s(errMsgInfo.group, sizeof(errMsgInfo.group), aicpuComm->GetId().c_str(), aicpuComm->GetId().size());

    GetUbErrMsgInfo(taskInfo, errMsgInfo, exceptionInfo);

    return HCCL_SUCCESS;
}

HcclResult SendErrorMessageReportToHost(CommunicatorImplLite* aicpuComm, ErrorMessageReport& errMsgInfo)
{
    CHK_RET(aicpuComm->SendErrorMessageReportToHost(errMsgInfo));
    return HCCL_SUCCESS;
}

constexpr u32 RT_SDMA_COMPERR = 0x9; // A3 sdma error类型为0x9时，表示写拷贝发生超时代答，或者数据搬移时地址译码错误
constexpr u32 RT_SDMA_COMPDATAERR = 0xa; // A3 sdma error类型为0xa时，表示读拷贝发生超时代答，或者读HBM返回ERROR
constexpr u32 RT_SDMA_DATAERR = 0x8;            // A3 sdma error类型为0x8时，表示读HBM返回ERROR
constexpr u32 RT_UB_LOCAL_OPERATIOINERR = 0x2;  // A5 ub error类型为0x2时，表示UB本端返回ERROR
constexpr u32 RT_UB_REMOTE_OPERATIOINERR = 0x3; // A5 ub error类型为0x3时，表示UB远端返回ERROR
constexpr u32 RT_UB_LINK_FAILEDERR = 0x5;       // A5 ub error类型为0x5时，表示网络异常，taack超时

// 把SDMA类错误码转换成Ts对应的错误码
uint16_t SwitchSdmaCqeErrCodeToTsErrCode(u32 cqeErrCode)
{
    switch (cqeErrCode) {
        case RT_SDMA_COMPERR:
            return TS_ERROR_SDMA_LINK_ERROR;
        case RT_SDMA_COMPDATAERR:
            return TS_ERROR_SDMA_POISON_ERROR;
        case RT_SDMA_DATAERR:
            return TS_ERROR_SDMA_DDRC_ERROR;
        default:
            return TS_ERROR_HCCL_OTHER_ERROR;
    }
}

// 把UB类错误码转换成Ts对应的错误码
uint16_t SwitchUBCqeErrCodeToTsErrCode(u32 cqeErrCode)
{
    switch (cqeErrCode) {
        case RT_UB_LOCAL_OPERATIOINERR:
            return TS_ERROR_HCCL_OP_UB_DDRC_FAILED;
        case RT_UB_REMOTE_OPERATIOINERR:
            return TS_ERROR_HCCL_OP_UB_POISON_FAILED;
        case RT_UB_LINK_FAILEDERR:
            return TS_ERROR_HCCL_OP_UB_LINK_FAILED;
        default:
            return TS_ERROR_HCCL_OTHER_ERROR;
    }
}

static std::string GetSdmaErrorDesc(u32 errorCode)
{
    switch (errorCode) {
        case RT_SDMA_DATAERR:
            return "read HBM return ERROR";
        case RT_SDMA_COMPERR:
            return "write copy timeout or address decode ERROR";
        case RT_SDMA_COMPDATAERR:
            return "read copy timeout or read HBM return ERROR";
        default:
            return "unknown SDMA ERROR";
    }
}

static void ReportSdmaError(
    u32 localDeviceId, u32 notifyId, u32 tsId, s32 userStreamId, u32 errorCode, const std::string& remoteRankId,
    const std::string& groupRankContent)
{
    std::string baseInfo = "localDeviceId=" + std::to_string(localDeviceId);
    std::string taskInfo = "notifyId= " + std::to_string(notifyId) + ", tsId=" + std::to_string(tsId)
                           + ", streamId=" + std::to_string(userStreamId) + ", error_code=" + std::to_string(errorCode)
                           + ", description=" + GetSdmaErrorDesc(errorCode);
    RPT_INPUT_ERR(
        true, "EI0012",
        std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
        std::vector<std::string>({remoteRankId, baseInfo, taskInfo, groupRankContent}));
    HCCL_ERROR(
        "[%s][%s][%s] sdma error, base information is %s, remote rank[%s].", LOG_KEYWORDS_TASK_EXEC.c_str(),
        LOG_KEYWORDS_RUN_FAILED.c_str(), LOG_KEYWORDS_AICPU.c_str(), baseInfo.c_str(), remoteRankId.c_str());
    HCCL_ERROR(
        "[%s][%s][%s] sdma error, task information is %s.", LOG_KEYWORDS_TASK_EXEC.c_str(),
        LOG_KEYWORDS_RUN_FAILED.c_str(), LOG_KEYWORDS_AICPU.c_str(), taskInfo.c_str());
    HCCL_ERROR(
        "[%s][%s][%s] sdma error, groupRank information is %s.", LOG_KEYWORDS_TASK_EXEC.c_str(),
        LOG_KEYWORDS_RUN_FAILED.c_str(), LOG_KEYWORDS_AICPU.c_str(), groupRankContent.c_str());
}

HcclResult SendTaskExceptionByMBox(
    const u32 localDeviceId, const u32 notifyId, const u32 tsId, const s32 userStreamId,
    const rtLogicCqReport_t* exceptionInfo)
{
    HCCL_INFO("[SendTaskExceptionByMBox] SendTaskExceptionByMBox start.");
    ts_aicpu_msg_info_t aicpuSqe = {};
    u32 hostpid = 0;
    u32 vf_id = 0;
    int pid = getpid();
    HCCL_INFO("SendTaskExceptionByMBox getpid[%d]", pid);
    // 调整drvQueryProcessHostPid获取pid和vf_id的值
    CHK_RET(HrtHalDrvQueryProcessHostPid(pid, nullptr, &vf_id, &hostpid, nullptr));

    aicpuSqe.pid = hostpid;
    aicpuSqe.cmd_type = TS_AICPU_RECORD;
    aicpuSqe.vf_id = vf_id;
    aicpuSqe.tid = 0U; // notify is no need tid
    aicpuSqe.u.aicpu_record.record_type = AICPU_MSG_NOTIFY_RECORD_V2;
    aicpuSqe.u.aicpu_record.record_id = notifyId;

    aicpuSqe.ts_id = static_cast<uint8_t>(tsId);

    aicpuSqe.u.aicpu_record.fault_task_id = 0xffffffff;

    std::string errorTypeStr = TaskExceptionFunc::GetInstance().ErrorType2Str(exceptionInfo->errorType);
    if (errorTypeStr.empty()) {
        errorTypeStr = "unknown";
    }
    HCCL_ERROR(
        "[SendTaskExceptionByMBox] exceptionInfo errorType[%u][%s], errorCode[0x%x]",
        static_cast<u32>(exceptionInfo->errorType), errorTypeStr.c_str(), exceptionInfo->errorCode);
    if (exceptionInfo->errorType == 1) { // ub类型为1
        aicpuSqe.u.aicpu_record.ret_code = SwitchUBCqeErrCodeToTsErrCode(exceptionInfo->errorCode & UB_CQE_STATUS_MASK);
    } else {
        aicpuSqe.u.aicpu_record.ret_code = SwitchSdmaCqeErrCodeToTsErrCode(exceptionInfo->errorCode);
    }

    struct event_summary event;
    event.dst_engine = TS_CPU;
    event.policy = ONLY;
    event.pid = 0;
    event.grp_id = 0;
    event.event_id = EVENT_TS_CTRL_MSG;
    event.subevent_id = 0U;
    event.msg_len = static_cast<uint32_t>(sizeof(ts_aicpu_msg_info_t));
    event.msg = reinterpret_cast<char_t*>(&aicpuSqe);
    auto ret = DlHalFunctionV2::GetInstance().dlHalEschedSubmitEvent(localDeviceId, &event);
    if (ret != static_cast<int32_t>(DRV_ERROR_NONE)) {
        HCCL_ERROR(
            "[SendTaskExceptionByMBox]Send msg async to ts failed. ret=%d, streamId=%d, "
            "notifyId=%u.",
            ret, userStreamId, notifyId);
        return HCCL_E_DRV;
    }
    HCCL_RUN_INFO(
        "[SendTaskExceptionByMBox]Send msg async to ts finished. streamId=%d, notifyId=%u, msg_size=%u, "
        "hostpid=%u, vf_id=%u, errCode=%u.",
        userStreamId, notifyId, static_cast<uint32_t>(sizeof(ts_aicpu_msg_info_t)), hostpid, vf_id,
        exceptionInfo->errorCode);
    return HCCL_SUCCESS;
}

HcclResult SendTaskExceptionByMBox(CommunicatorImplLite* aicpuComm, const rtLogicCqReport_t* exceptionInfo)
{
    u32 localDeviceId = 0;
    u32 notifyId = aicpuComm->GetHostDeviceSyncNotifyLiteMgr()->GetHostWaitNotify()->GetId();
    u32 devPhyId = aicpuComm->GetHostDeviceSyncNotifyLiteMgr()->GetHostWaitNotify()->GetDevPhyId();

    HCCL_INFO("[TaskExceptionHandlerLite][SendTaskExceptionByMBox] HostToDeviceLogicId[%u]", devPhyId);
    auto ret = drvGetLocalDevIDByHostDevID(devPhyId, &localDeviceId);
    if (ret != 0) {
        HCCL_ERROR("[TaskExceptionHandlerLite][SendTaskExceptionByMBox] HostToDeviceLogicId[%u] failed.", devPhyId);
        return HCCL_E_DRV;
    }
    HCCL_INFO(
        "[TaskExceptionHandlerLite][SendTaskExceptionByMBox] HostToDeviceLogicId[%u], localDeviceid[%u]", devPhyId,
        localDeviceId);

    CHK_RET(SendTaskExceptionByMBox(localDeviceId, notifyId, 0, aicpuComm->GetUserStreamId(), exceptionInfo));

    if (exceptionInfo->errorType != 1) {
        std::string remoteRankId = std::to_string(aicpuComm->GetMyRank());
        std::string groupRankContent = "group:[" + aicpuComm->GetId() + "], rankSize["
                                       + std::to_string(aicpuComm->GetRankSize()) + "], rankId["
                                       + std::to_string(aicpuComm->GetMyRank()) + "]";
        ReportSdmaError(
            localDeviceId, notifyId, 0, aicpuComm->GetUserStreamId(), exceptionInfo->errorCode, remoteRankId,
            groupRankContent);
    }
    return HCCL_SUCCESS;
}

string GetOpDataInfo(const TaskInfo& taskInfo)
{
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[GetOpDataInfo]TaskInfo communicator is nullptr.");
        return "";
    }
    CommunicatorImplLite* aicpuComm = static_cast<CommunicatorImplLite*>(taskInfo.dfxOpInfo_->comm_);
    u32 localDeviceId = 0;
    u32 devPhyId = aicpuComm->GetDevPhyId();

    HCCL_INFO("[GetOpDataInfo] HostToDeviceLogicId[%u]", devPhyId);
    auto ret = drvGetLocalDevIDByHostDevID(devPhyId, &localDeviceId);
    if (ret != 0) {
        HCCL_ERROR("[GetOpDataInfo] HostToDeviceLogicId[%u] failed.", devPhyId);
        return "";
    }
    return StringFormat("deviceId[%u], %s", localDeviceId, taskInfo.GetOpInfo().c_str());
}

void PrintEid(TaskInfo* taskInfo)
{
    if (taskInfo->taskParam_.taskType == TaskParamType::TASK_UB_REDUCE_INLINE
        || taskInfo->taskParam_.taskType == TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY) {
        HCCL_ERROR(
            "[PrintEid] Error UB link info: localEid[%s], remoteEid[%s]. ",
            taskInfo->taskParam_.taskPara.Reduce.locEid.Describe().c_str(),
            taskInfo->taskParam_.taskPara.Reduce.rmtEid.Describe().c_str());
    } else if (
        taskInfo->taskParam_.taskType == TaskParamType::TASK_UB_INLINE_WRITE
        || taskInfo->taskParam_.taskType == TaskParamType::TASK_WRITE_WITH_NOTIFY
        || taskInfo->taskParam_.taskType == TaskParamType::TASK_UB) {
        HCCL_ERROR(
            "[PrintEid] Error UB link info: localEid[%s], remoteEid[%s]. ",
            taskInfo->taskParam_.taskPara.DMA.locEid.Describe().c_str(),
            taskInfo->taskParam_.taskPara.DMA.rmtEid.Describe().c_str());
    }
}

void TaskExceptionHandlerLite::Process(CommunicatorImplLite* aicpuComm, rtLogicCqReport_t* exceptionInfo)
{
    if (exceptionInfo == nullptr) {
        HCCL_ERROR("Exception process failed, rtExceptionInfo is nullptr.");
        return;
    }
    // exceptionInfo->taskId和exceptionInfo->streamId拼成sqeId
    const u32 sqeId
        = static_cast<uint32_t>(exceptionInfo->taskId << 16) | static_cast<uint32_t>(exceptionInfo->streamId);
    const auto curTask = aicpuComm->GetMirrorTaskMgrLite()->GetTaskInfo(exceptionInfo->sqId, sqeId);
    if (curTask == nullptr) {
        // 未找到异常对应的TaskInfo
        HCCL_ERROR("Exception task not found. streamId(sqId)[%u], taskId(sqeId)[%u].", exceptionInfo->sqId, sqeId);
        return;
    }

    // 每个通信域仅首次上报（N秒快恢时重置）
    if (!aicpuComm->IsErrorReported()) {
        // 1) errorMessage上报
        ErrorMessageReport errMsgInfo{};
        auto ret = GenerateErrorMessageReport(aicpuComm, curTask, errMsgInfo, exceptionInfo);
        if (ret != HCCL_SUCCESS) {
            THROW<InvalidParamsException>("GenerateErrorMessageReport failed.");
        }
        ret = SendErrorMessageReportToHost(aicpuComm, errMsgInfo);
        if (ret != HCCL_SUCCESS) {
            THROW<InvalidParamsException>("SendErrorMessageReportToHost Failed.");
        }

        // 2) send mbox to tsfw
        CHK_RET_THROW(
            InternalException,
            StringFormat(
                "[TaskExceptionHandlerLite][SendTaskExceptionByMBox]group[%s]:"
                "Send task exception by mailBox failed, streamId[%d]",
                aicpuComm->GetId().c_str(), exceptionInfo->streamId),
            SendTaskExceptionByMBox(aicpuComm, exceptionInfo));

        aicpuComm->SetErrorReported();
    }

    const auto& logKeywordL2 = curTask->taskParam_.taskType == TaskParamType::TASK_NOTIFY_WAIT ?
                                   LOG_KEYWORDS_TIMEOUT :
                                   LOG_KEYWORDS_RUN_FAILED;
    auto stageErrInfo = "[" + LOG_KEYWORDS_TASK_EXEC + "][" + logKeywordL2 + "][" + LOG_KEYWORDS_AICPU + "]";
    HCCL_ERROR("%s Task from HCCL run failed.", stageErrInfo.c_str());
    if (curTask->taskParam_.taskType == TaskParamType::TASK_NOTIFY_WAIT) {
        PrintTaskContextInfo(aicpuComm, exceptionInfo->sqId, sqeId);
    }
    if (curTask->taskParam_.taskType == TaskParamType::TASK_WRITE_WITH_NOTIFY
        || curTask->taskParam_.taskType == TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY
        || curTask->taskParam_.taskType == TaskParamType::TASK_UB_INLINE_WRITE
        || curTask->taskParam_.taskType == TaskParamType::TASK_UB_REDUCE_INLINE
        || curTask->taskParam_.taskType == TaskParamType::TASK_UB) {
        HCCL_ERROR(
            "%s ubCqeStatus[%u]", stageErrInfo.c_str(),
            static_cast<u32>(exceptionInfo->errorCode & UB_CQE_STATUS_MASK));
    }
    PrintEid(curTask);
    HCCL_ERROR("%s Task run failed, base information is %s.", stageErrInfo.c_str(), curTask->GetBaseInfo().c_str());
    HCCL_ERROR("%s Task run failed, para information is %s.", stageErrInfo.c_str(), curTask->GetParaInfo().c_str());
    HCCL_ERROR(
        "%s Task run failed, groupRank information is %s.", stageErrInfo.c_str(), GetGroupRankInfo(*curTask).c_str());
    if (curTask->dfxOpInfo_ != nullptr && curTask->dfxOpInfo_->headOpCounterAddr_ != 0
        && curTask->dfxOpInfo_->tailOpCounterAddr_ != 0) {
        HCCL_ERROR(
            "%s Task run failed, headOpCounter[%u] tailOpCounter[%u] opIndex[%u].", stageErrInfo.c_str(),
            static_cast<u32>(*(reinterpret_cast<float*>(curTask->dfxOpInfo_->headOpCounterAddr_))),
            static_cast<u32>(*(reinterpret_cast<float*>(curTask->dfxOpInfo_->tailOpCounterAddr_))),
            curTask->dfxOpInfo_->opIndex_);
    }
    HCCL_ERROR("%s Task run failed, opData information is %s.", stageErrInfo.c_str(), GetOpDataInfo(*curTask).c_str());
}

string TaskExceptionHandlerLite::GetGroupRankInfo(const TaskInfo& taskInfo)
{
    if (taskInfo.dfxOpInfo_ == nullptr || taskInfo.dfxOpInfo_->comm_ == nullptr) {
        HCCL_ERROR("[TaskInfo][%s]TaskInfo communicator is nullptr.", __func__);
        return "";
    }
    const CommunicatorImplLite* commImplLite = static_cast<CommunicatorImplLite*>(taskInfo.dfxOpInfo_->comm_);
    return StringFormat(
        "group:[%s], rankSize[%u], localRank[%d], remoteRank[%u]", commImplLite->GetId().c_str(),
        commImplLite->GetRankSize(), commImplLite->GetMyRank(), taskInfo.remoteRank_);
}

void TaskExceptionHandlerLite::PrintTaskContextInfo(CommunicatorImplLite* aicpuComm, uint32_t sqId, uint32_t taskId)
{
    auto queue = aicpuComm->GetMirrorTaskMgrLite()->GetQueue(sqId);
    if (queue == nullptr) {
        // 未找到异常对应的TaskQueue
        HCCL_ERROR("Exception task queue not found. streamId(sqId)[%u].", sqId);
        return;
    }

    auto func = [taskId](const unique_ptr<TaskInfo>& task) {
        return task->taskId_ == taskId;
    };
    auto taskItorPtr = queue->Find(func);
    if (taskItorPtr == nullptr || *taskItorPtr == *queue->End()) {
        // 在队列中未找到异常对应的TaskInfo
        HCCL_ERROR("Exception task not found. streamId(sqId)[%u], taskId(sqeId)[%u].", sqId, taskId);
        return;
    }

    // 找到当前异常task的前50个task(至多)
    vector<TaskInfo*> taskContext{};
    for (uint32_t i = 0; i < TASK_CONTEXT_SIZE && *taskItorPtr != *queue->Begin(); ++i, --(*taskItorPtr)) {
        if ((**taskItorPtr)->taskId_ > taskId) {
            break;
        }
        if ((**taskItorPtr)->taskId_ != taskId) {
            taskContext.emplace_back((**taskItorPtr).get());
        }
    }

    if (taskContext.empty()) {
        return;
    }

    HCCL_ERROR("[TaskExceptionHandlerLite]Task run failed, context sequence before error task is "
               "[SDMA:M(rank), RDMA:RS(rank,id), SendPayload:SP(rank), InlineReduce:IR(rank), Reduce:R(rank), "
               "NotifyRecord:NR(rank,id), NotifyWait:NW(rank,id), SendNotify:SN(rank,id), "
               "WriteWithNotify:WN(rank,id), WriteReduceWithNotify:WRN(rank,id)]:");

    string taskContextInfo = "";
    for (auto it = taskContext.rbegin(); it != taskContext.rend(); ++it) {
        string conciseInfo = (*it)->GetConciseBaseInfo();
        conciseInfo += ",";

        if (taskContextInfo.size() + conciseInfo.size() >= TASK_CONTEXT_INFO_SIZE) {
            HCCL_ERROR("[TaskExceptionHandlerLite]%s", taskContextInfo.c_str());
            taskContextInfo = "";
        }

        taskContextInfo += conciseInfo;
    }
    HCCL_ERROR("[TaskExceptionHandlerLite]%s end.", taskContextInfo.c_str());
}

} // namespace Hccl
