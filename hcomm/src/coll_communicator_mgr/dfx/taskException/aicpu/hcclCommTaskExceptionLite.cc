/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcclCommTaskExceptionLite.h"
#include "stream_lite.h"
#include "hcomm_task_scheduler_error.h"
#include "task_struct_v2.h"
#include "dlhal_function_v2.h"
#include <shared_mutex>
#include "aicpu_indop_env.h"
#include "kernel_entrance.h"
#include "ub_transport_lite_impl.h"
#include "res_TE.h"

namespace hcomm {
constexpr u32 RT_SDMA_COMPERR = 0x9; // A3 sdma error类型为0x9时，表示写拷贝发生超时代答，或者数据搬移时地址译码错误
constexpr u32 RT_SDMA_COMPDATAERR = 0xa; // A3 sdma error类型为0xa时，表示读拷贝发生超时代答，或者读HBM返回ERROR
constexpr u32 RT_SDMA_DATAERR = 0x8;            // A3 sdma error类型为0x8时，表示读HBM返回ERROR
constexpr u32 RT_UB_LOCAL_OPERATIOINERR = 0x2;  // A5 ub error类型为0x2时，表示UB本端返回ERROR
constexpr u32 RT_UB_REMOTE_OPERATIOINERR = 0x3; // A5 ub error类型为0x3时，表示UB远端返回ERROR
constexpr u32 RT_UB_LINK_FAILEDERR = 0x5;       // A5 ub error类型为0x5时，表示网络异常，taack超时
constexpr uint8_t ubSqeType = 9;                // A5 sqeType为9表示UBDMA任务
constexpr uint8_t sdmaSqeType = 11;             // A5 sqeType为11表示SDMA任务

constexpr uint32_t TASK_CONTEXT_SIZE = 50; // task 执行失败时打印前序task信息的数量
constexpr uint32_t TASK_CONTEXT_INFO_SIZE
    = LOG_TMPBUF_SIZE - TASK_CONTEXT_SIZE; // task 执行失败时打印前序task信息的长度限制
constexpr u32 MAX_NAME_LEN = 64;
constexpr u32 TASK_ID_SHIFT_BITS = 16;
constexpr u32 ADDR_WORD_SHIFT_BITS = 32;      // 将32位高位地址与低地址拼接为64位地址的移位数
constexpr uint32_t UB_CQE_STATUS_MASK = 0xFF; // ubCqeStatus取低8位

HcclCommTaskExceptionLite& HcclCommTaskExceptionLite::GetInstance()
{
    static HcclCommTaskExceptionLite instance; // aicpu侧一个dev一个进程，不需要按dev区分单例对象
    return instance;
}

void HcclCommTaskExceptionLite::Init(u32 devId)
{
    devId_ = devId;
    HCCL_INFO("[%s]success, devId_[%u]", __func__, devId_);
}

void HcclCommTaskExceptionLite::Call()
{
    if (stopCall_ == true) {
        return;
    }

    HcclResult ret = HandleExceptionCqe();
    if (ret != HCCL_SUCCESS) {
        stopCall_ = true;
        HCCL_ERROR(
            "[%s]HandleExceptionCqe fail, set stopCall_[%d]", __func__, stopCall_); // 函数调用失败，停止调用避免刷屏
    }
}

HcclResult HcclCommTaskExceptionLite::IsHandleDpuStop(uint8_t* taskexceptionVa, bool& isStop)
{
    uint8_t stopSignal = 0;
    errno_t ret = memcpy_s(
        &stopSignal, sizeof(stopSignal), taskexceptionVa,
        sizeof(stopSignal)); // 读标志位,第1字节，存放host侧发送是否停止的信号。
    if (ret != EOK) {
        HCCL_ERROR("[HcclCommTaskExceptionLite::%s] memcpy_s failed on flag, return[%d].", __func__, ret);
        return HCCL_E_MEMORY;
    }
    if (stopSignal == 1) {
        isStop = true;
        stopSignal = 0;
        ret = memcpy_s(
            taskexceptionVa, sizeof(stopSignal), &stopSignal,
            sizeof(stopSignal)); // 读标志位,第1字节，存放host侧发送是否停止的信号。
        if (ret != EOK) {
            HCCL_ERROR("[HcclCommTaskExceptionLite::%s] memcpy_s failed on flag, return[%d].", __func__, ret);
            return HCCL_E_MEMORY;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::HandleDpuTaskexception(CollCommAicpu* aicpuComm)
{
    // 轮询taskexception共享内存
    auto commId = aicpuComm->GetIdentifier();
    std::lock_guard<std::mutex> lock(g_taskExpDevMemMapMutex);
    auto it = g_taskExpDevMemMap.find(commId);
    if (it == g_taskExpDevMemMap.end()) {
        return HCCL_SUCCESS; // 非dpu场景，map为空
    }

    auto taskexceptionVa = reinterpret_cast<uint8_t*>(it->second);
    if (taskexceptionVa == nullptr) {
        return HCCL_SUCCESS;
    }
    // 查是否要停止
    bool isStop = false;
    CHK_RET(IsHandleDpuStop(taskexceptionVa, isStop));
    if (isStop) {
        it->second = nullptr;
        return HCCL_SUCCESS;
    }
    // 查是否有错误
    uint16_t errorCode = 0;
    errno_t ret = memcpy_s(
        &errorCode, sizeof(errorCode), taskexceptionVa + sizeof(uint8_t),
        sizeof(errorCode)); // 读标志位,第2-3字节，存放HcclResult。
    if (ret != EOK) {
        HCCL_ERROR("[HcclCommTaskExceptionLite::%s] memcpy_s failed on errorCode, return[%d].", __func__, ret);
        return HCCL_E_MEMORY;
    }
    if (errorCode != 0) {
        // 触发taskexception
        HCCL_ERROR(
            "[HcclCommTaskExceptionLite][DPU] taskexceptionVa[%p], errorCode[0x%x], devId[%u], commId[%s]",
            taskexceptionVa, errorCode, aicpuComm->GetDevId(), commId.c_str());
        // 1、取notify，并构造rtLogicCqReport_t
        auto* hcclCommDfxLite = aicpuComm->GetHcclCommDfxLite();
        CHK_PTR_NULL(hcclCommDfxLite);
        const auto curDfxOpInfo = static_cast<const Hccl::DfxDfxOpInfo*>(hcclCommDfxLite->GetLatestDfxOpInfo());
        CHK_PTR_NULL(curDfxOpInfo);
        u32 notifyId = curDfxOpInfo->cpuWaitAicpuNotifyId;
        rtLogicCqReport_t exceptionInfo{};
        // 2、调用SendTaskExceptionByMBox触发taskexception回调
        CHK_RET(SendTaskExceptionByMBox(notifyId, 0, exceptionInfo));
        // 3、标志位置0
        ret = memset_s(taskexceptionVa + sizeof(uint8_t), sizeof(uint16_t), 0, sizeof(uint16_t)); // 标志位置0
        if (ret != EOK) {
            HCCL_ERROR("[HcclCommTaskExceptionLite::%s] memset_s failed on flag, return[%d].", __func__, ret);
            return HCCL_E_MEMORY;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::HandleExceptionCqe()
{
    std::shared_lock<std::shared_mutex> rwlock(CollCommAicpuMgr::GetInstance().GetMutex());

    std::vector<std::pair<std::string, CollCommAicpu*>> aicpuCommInfo;
    CHK_RET(CollCommAicpuMgr::GetInstance().GetAllComms(aicpuCommInfo));

    for (auto& commInfo : aicpuCommInfo) {
        CollCommAicpu* aicpuComm = commInfo.second;
        CHK_PTR_NULL(aicpuComm);

        if ((aicpuComm->GetCommmStatus() == HcclCommStatus::HCCL_COMM_STATUS_INVALID)
            || (aicpuComm->GetCommmStatus() == HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING)) {
            continue;
        }
        CHK_RET(HandleDpuTaskexception(aicpuComm)); // dpu taskexception

        std::shared_lock<std::shared_mutex> threadRwlock(aicpuComm->GetCommEngineResMgr()->GetThreadMutex());
        const std::vector<std::shared_ptr<hccl::Thread>> threads = aicpuComm->GetCommEngineResMgr()->GetAllThread();
        for (auto thread : threads) {
            rtLogicCqReport_t cqeException;
            dfx::CqeStatus cqeStatus = dfx::CqeStatus::kDefault;
            Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(thread->GetStreamLitePtr());
            CHK_PTR_NULL(streamLite);

            HcclResult ret = GetThreadCqe(thread.get(), cqeException, cqeStatus);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[%s]GetThreadCqe fail, aicpuComm[%s], streamId[%u]", __func__, aicpuComm->GetIdentifier().c_str(),
                    streamLite->GetId()),
                ret);

            ret = ProcessCqe(aicpuComm, cqeException, cqeStatus, aicpuCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[%s]ProcessCqe fail, aicpuComm[%s], streamId[%u], "
                    "cqeStatus[%lld]",
                    __func__, aicpuComm->GetIdentifier().c_str(), streamLite->GetId(),
                    static_cast<long long>(cqeStatus)),
                ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::PrintAllCommTaskException()
{
    std::shared_lock<std::shared_mutex> rwlock(CollCommAicpuMgr::GetInstance().GetMutex());

    std::vector<std::pair<std::string, CollCommAicpu*>> aicpuCommInfo;
    CHK_RET(CollCommAicpuMgr::GetInstance().GetAllComms(aicpuCommInfo));

    HCCL_RUN_INFO("[TaskException][AICPU]%s start, comm size[%u]", __func__, aicpuCommInfo.size());
    HcclResult ret = HCCL_SUCCESS;
    for (auto& commInfo : aicpuCommInfo) {
        CollCommAicpu* aicpuComm = commInfo.second;
        HcclResult pRet = PrintCommTaskException(aicpuComm);
        CHK_PRT_CONT(
            pRet != HCCL_SUCCESS,
            HCCL_ERROR("PrintCommTaskException fail, comm[%s]", aicpuComm->GetIdentifier().c_str()));
        ret = (pRet != HCCL_SUCCESS) ? pRet : ret;
    }
    HCCL_RUN_INFO("[TaskException][AICPU]%s end, ret[%d]", __func__, ret);
    return ret;
}

HcclResult HcclCommTaskExceptionLite::PrintCommTaskException(CollCommAicpu* aicpuComm)
{
    CHK_PTR_NULL(aicpuComm);
    HcclResult ret = HCCL_SUCCESS;
    HCCL_RUN_INFO("[TaskException][AICPU]%s comm[%s] start", __func__, aicpuComm->GetIdentifier().c_str());
    std::shared_lock<std::shared_mutex> threadRwlock(aicpuComm->GetCommEngineResMgr()->GetThreadMutex());
    const std::vector<std::shared_ptr<hccl::Thread>> threads = aicpuComm->GetCommEngineResMgr()->GetAllThread();
    for (auto thread : threads) {
        CHK_SMART_PTR_NULL(thread);
        Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(thread->GetStreamLitePtr());
        CHK_PTR_NULL(streamLite);
        u32 sqHead = 0U;
        u32 sqTail = 0U;
        ret = QuerySqStatus(devId_, streamLite->GetSqId(), sqHead, sqTail);
        if (ret != HCCL_SUCCESS || sqHead == sqTail) { // 此流为空时，不打印
            HCCL_RUN_INFO(
                "[TaskException][AICPU]PrintTaskExceptionBySqeId skip, "
                "QuerySqStatus ret[%d], aicpuComm[%s], sqId[%u], sqHead[%u], sqTail[%u]",
                ret, aicpuComm->GetIdentifier().c_str(), streamLite->GetSqId(), sqHead, sqTail);
            continue;
        }
        uint16_t streamId = 0;
        uint16_t taskId = 0;
        streamLite->GetRtsq()->GetStreamIdAndTaskIdBySqIdx(sqHead, streamId, taskId);
        const u32 sqeId = GetSqeId(taskId, streamId);
        HcclResult pRet = PrintTaskExceptionBySqeId(aicpuComm, streamLite->GetSqId(), sqeId);
        CHK_PRT_CONT(
            pRet != HCCL_SUCCESS, HCCL_ERROR(
                                      "PrintTaskExceptionBySqeId fail, comm[%s], sqId[%u], sqeId[%u]",
                                      aicpuComm->GetIdentifier().c_str(), streamLite->GetSqId(), sqeId));
        ret = (pRet != HCCL_SUCCESS) ? pRet : ret;
    }
    HCCL_RUN_INFO("[TaskException][AICPU]%s comm[%s] end, ret[%d]", __func__, aicpuComm->GetIdentifier().c_str(), ret);
    return ret;
}

HcclResult HcclCommTaskExceptionLite::GetThreadCqe(
    hccl::Thread* thread, rtLogicCqReport_t& cqeException, dfx::CqeStatus& cqeStatus)
{
    CHK_SMART_PTR_NULL(thread);
    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(thread->GetStreamLitePtr());
    CHK_PTR_NULL(streamLite);

    constexpr u32 reportSize = MAX_REPORT_CNT;
    rtLogicCqReport_t streamReport[reportSize];

    CqeQueryInput cqeQueryInput;
    cqeQueryInput.devId = devId_;
    cqeQueryInput.streamId = streamLite->GetId();
    cqeQueryInput.sqId = streamLite->GetSqId();
    cqeQueryInput.cqId = streamLite->GetCqId();
    cqeQueryInput.type = static_cast<uint32_t>(DRV_LOGIC_TYPE);
    cqeQueryInput.cqeAddr = reinterpret_cast<uint8_t*>(streamReport);

    cqeStatus = CqReportRecv(cqeQueryInput, cqeException);
    if (cqeStatus == dfx::CqeStatus::kCqeInnerError) {
        HCCL_ERROR("[%s]CqReportRecv fail, CqeQueryInput:%s", __func__, cqeQueryInput.ToString().c_str());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::ProcessCqe(
    CollCommAicpu* aicpuComm, const rtLogicCqReport_t& exceptionInfo, const CqeStatus& cqeStatus,
    const std::vector<std::pair<std::string, CollCommAicpu*>>& aicpuCommInfo)
{
    if (cqeStatus == dfx::CqeStatus::kDefault) {
        return HCCL_SUCCESS;
    }

    if (hcomm::GetTaskExceptionEnable() == false) {
        HCCL_ERROR("[TaskException][AICPU]taskException enable is false, skip print taskException");
        return HCCL_SUCCESS;
    }

    HcclResult ret = HCCL_SUCCESS;
    const u32 sqeId = GetSqeId(exceptionInfo.taskId, exceptionInfo.streamId);
    ret = PrintTaskExceptionBySqeId(aicpuComm, exceptionInfo.sqId, sqeId);
    CHK_PRT_CONT(
        ret != HCCL_SUCCESS, HCCL_ERROR(
                                 "[PrintTaskExceptionBySqeId]fail, ret[%d], group[%s], sqId[%u], taskId[%u]", ret,
                                 aicpuComm->GetIdentifier().c_str(), exceptionInfo.sqId,
                                 exceptionInfo.taskId)); // 如果上报失败，继续打印taskException

    ret = ReportErrMsg(aicpuComm, exceptionInfo);
    CHK_PRT_CONT(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ReportErrMsg]fail, ret[%d], group[%s], sqId[%u], taskId[%u]", ret, aicpuComm->GetIdentifier().c_str(),
            exceptionInfo.sqId, exceptionInfo.taskId)); // 如果上报失败，继续打印taskException

    // notify超时场景：step1 打印当前流信息；step2 打印当前通信域信息；step3 打印其他通信域信息
    if (cqeStatus == dfx::CqeStatus::kCqeException && exceptionInfo.sqeType == RT_STARS_SQE_TYPE_PLACE_HOLDER) {
        CHK_RET(PrintCommTaskException(aicpuComm));
        for (auto& commInfo : aicpuCommInfo) {
            CollCommAicpu* comm = commInfo.second;
            if (comm != nullptr && comm->GetIdentifier() != aicpuComm->GetIdentifier()) {
                CHK_RET(PrintCommTaskException(comm));
            }
        }
    }
    return ret;
}

u32 HcclCommTaskExceptionLite::GetSqeId(uint16_t taskId, uint16_t streamId)
{
    return (static_cast<u32>(taskId) << TASK_ID_SHIFT_BITS) | static_cast<u32>(streamId);
}

HcclResult HcclCommTaskExceptionLite::ReportErrMsg(CollCommAicpu* aicpuComm, const rtLogicCqReport_t& exceptionInfo)
{
    CHK_PTR_NULL(aicpuComm);

    const u32 sqeId = GetSqeId(exceptionInfo.taskId, exceptionInfo.streamId);
    HCCL_INFO(
        "[%s]group[%s], sqeId[0x%x], taskId[%u], streamId[%u].", __func__, aicpuComm->GetIdentifier().c_str(), sqeId,
        exceptionInfo.taskId, exceptionInfo.streamId);

    Hccl::DfxTaskInfo* curTask = FindDfxTaskInfo(aicpuComm, exceptionInfo.sqId, sqeId);
    CHK_PTR_NULL(curTask);

    const Hccl::DfxDfxOpInfo* opInfo = (curTask->dfxOpInfo != DFX_INVALID_U64) ?
                                           reinterpret_cast<const Hccl::DfxDfxOpInfo*>(curTask->dfxOpInfo) :
                                           nullptr;
    CHK_PTR_NULL(opInfo);

    if (!aicpuComm->IsErrorReported()) {
        Hccl::ErrorMessageReport errMsgInfo{};
        CHK_RET(GenerateErrorMessageReport(aicpuComm, *curTask, exceptionInfo, errMsgInfo));
        CHK_RET(aicpuComm->SendErrorMessageReportToHost(errMsgInfo));

        u32 notifyId = opInfo->cpuWaitAicpuNotifyId;
        CHK_RET(SendTaskExceptionByMBox(notifyId, 0, exceptionInfo));
        aicpuComm->SetErrorReported(true);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::PrintTaskExceptionBySqeId(CollCommAicpu* aicpuComm, u32 sqId, u32 sqeId)
{
    CHK_PTR_NULL(aicpuComm);

    // 已经打印过的不再重复打印
    auto it = threadsPrinted_.find(sqId);
    if (it != threadsPrinted_.end() && it->second == sqeId) {
        HCCL_RUN_INFO("[TaskException][AICPU]sqId:%u, sqeId:%u has been printed, skip", sqId, sqeId);
        return HCCL_SUCCESS;
    }
    threadsPrinted_[sqId] = sqeId;

    Hccl::DfxTaskInfo* curTask = FindDfxTaskInfo(aicpuComm, sqId, sqeId);
    CHK_PTR_NULL(curTask);

    u32 sqHead = 0U;
    u32 sqTail = 0U;
    (void)QuerySqStatus(devId_, sqId, sqHead, sqTail);

    HCCL_ERROR(
        "[TaskException][AICPU]base information is streamID(sqId):[%u], taskID(sqeId):[%u], taskType:[%s], %s, "
        "localRank:[%u], sqHead:%u, sqTail:%u",
        curTask->sqId, curTask->taskId, Hccl::TaskParamTypeValToStr(curTask->taskType).c_str(),
        GetParaInfo(*curTask).c_str(), aicpuComm->GetTopoInfo().userRank, sqHead, sqTail);

    HCCL_ERROR("[TaskException][AICPU]group information is %s.", GetGroupInfo(aicpuComm).c_str());

    if (curTask->taskType != static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT)) {
        PrintOpDataInfo(curTask);
    } else {
        CHK_RET(PrintTaskContextInfo(aicpuComm, sqId, sqeId));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::GenerateErrorMessageReport(
    CollCommAicpu* aicpuComm, const Hccl::DfxTaskInfo& taskInfo, const rtLogicCqReport_t& exceptionInfo,
    Hccl::ErrorMessageReport& errMsgInfo)
{
    const Hccl::DfxDfxOpInfo* opInfo = (taskInfo.dfxOpInfo != DFX_INVALID_U64) ?
                                           reinterpret_cast<const Hccl::DfxDfxOpInfo*>(taskInfo.dfxOpInfo) :
                                           nullptr;
    CHK_PTR_NULL(opInfo);

    errMsgInfo.remoteUserRank = GetRemoteRankId(taskInfo);
    errMsgInfo.streamId = taskInfo.sqId;
    errMsgInfo.taskId = taskInfo.taskId;
    errMsgInfo.rankId = aicpuComm->GetTopoInfo().userRank;
    errMsgInfo.rankSize = aicpuComm->GetTopoInfo().userRankSize;

    errMsgInfo.opIndex = opInfo->opIndex;
    errMsgInfo.opType = opInfo->opType;
    errMsgInfo.count = opInfo->count;
    errMsgInfo.dataType = opInfo->dataType;
    errMsgInfo.srcAddr = opInfo->srcAddr;
    errMsgInfo.dstAddr = opInfo->dstAddr;
    CHK_SAFETY_FUNC_RET(memcpy_s(
        errMsgInfo.tag, sizeof(errMsgInfo.tag), opInfo->algTag, strnlen(opInfo->algTag, sizeof(opInfo->algTag)) + 1));

    errMsgInfo.taskType = Hccl::TaskParamType(static_cast<Hccl::TaskParamType::Value>(taskInfo.taskType));
    errMsgInfo.rtCqErrorType = exceptionInfo.errorType;
    errMsgInfo.rtCqErrorCode = exceptionInfo.errorCode;

    errMsgInfo.jettyHandle = taskInfo.taskPara.ubDma.jettyHandle;
    errMsgInfo.jettyId = taskInfo.taskPara.ubDma.jettyId;
    errMsgInfo.tpn = taskInfo.taskPara.ubDma.tpn;

    CHK_SAFETY_FUNC_RET(memcpy_s(
        errMsgInfo.group, sizeof(errMsgInfo.group), aicpuComm->GetIdentifier().c_str(),
        aicpuComm->GetIdentifier().size() + 1));

    GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);

    HCCL_RUN_INFO("[TaskException][AICPU]ErrorMessageReport: %s", errMsgInfo.Describe().c_str());

    return HCCL_SUCCESS;
}

void HcclCommTaskExceptionLite::GenerateTaskErrMsg(
    const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo, const rtLogicCqReport_t& exceptionInfo)
{
    auto taskType = static_cast<Hccl::TaskParamTypeVal>(taskInfo.taskType);
    switch (taskType) {
        case Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT:
        case Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD:
            FillNotifyErrMsg(taskInfo, errMsgInfo);
            break;
        case Hccl::TaskParamTypeVal::TASK_UB_REDUCE_INLINE:
        case Hccl::TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY:
            FillReduceErrMsg(taskInfo, errMsgInfo, exceptionInfo);
            break;
        case Hccl::TaskParamTypeVal::TASK_REDUCE_INLINE:
            FillReduceInlineErrMsg(taskInfo, errMsgInfo);
            break;
        case Hccl::TaskParamTypeVal::TASK_UB_INLINE_WRITE:
        case Hccl::TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY:
            FillDmaErrMsg(taskInfo, errMsgInfo, exceptionInfo);
            break;
        case Hccl::TaskParamTypeVal::TASK_UB:
            FillUbErrMsg(taskInfo, errMsgInfo, exceptionInfo);
            break;
        case Hccl::TaskParamTypeVal::TASK_SDMA:
            FillSdmaErrMsg(taskInfo, errMsgInfo);
            break;
        default:
            HCCL_ERROR(
                "[TaskException][AICPU]%s taskType[%s] is not support", __func__,
                Hccl::TaskParamTypeValToStr(taskInfo.taskType).c_str());
            return;
    }
}

void HcclCommTaskExceptionLite::FillNotifyErrMsg(
    const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo)
{
    u32 notifyId = GetNotifyIdFromSqe(taskInfo.taskPara.Notify.sqeAddr);
    errMsgInfo.notifyId = notifyId;
    errMsgInfo.notifyValue = 1; // 非cnt notify统一赋值为1，cnt notify当前未使用
}

void HcclCommTaskExceptionLite::FillReduceErrMsg(
    const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo, const rtLogicCqReport_t& exceptionInfo)
{
    errMsgInfo.reduceType = taskInfo.taskPara.Reduce.reduceOp;
    errMsgInfo.notifyId = taskInfo.taskPara.Reduce.notifyId;
    errMsgInfo.notifyValue = INVALID_U32;
    GetEidFromChannelHandle(taskInfo, errMsgInfo.locEid, errMsgInfo.rmtEid);
    errMsgInfo.ubCqeStatus = exceptionInfo.errorCode & UB_CQE_STATUS_MASK;
    errMsgInfo.linkType = Hccl::DfxLinkType(static_cast<Hccl::DfxLinkType::Value>(taskInfo.linkType));
    errMsgInfo.size = taskInfo.taskPara.Reduce.size;
    errMsgInfo.taskSrcAddr = taskInfo.taskPara.Reduce.srcAddr;
    errMsgInfo.taskDstAddr = taskInfo.taskPara.Reduce.dstAddr;
    HCCL_ERROR(
        "[TaskException][AICPU]Error UB link info: ubCqeStatus[%u], localEid[%s], remoteEid[%s]. ",
        errMsgInfo.ubCqeStatus, errMsgInfo.locEid.Describe().c_str(), errMsgInfo.rmtEid.Describe().c_str());
}

void HcclCommTaskExceptionLite::FillDmaErrMsg(
    const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo, const rtLogicCqReport_t& exceptionInfo)
{
    errMsgInfo.notifyId = taskInfo.taskPara.ubDma.notifyId;
    errMsgInfo.notifyValue = 1; // 非cnt notify统一赋值为1，cnt notify当前未使用
    FillUbErrMsg(taskInfo, errMsgInfo, exceptionInfo);
}

void HcclCommTaskExceptionLite::FillSdmaErrMsg(const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo)
{
    errMsgInfo.linkType = Hccl::DfxLinkType(static_cast<Hccl::DfxLinkType::Value>(taskInfo.linkType));
    u64 src = 0;
    u64 dst = 0;
    u64 size = 0;
    GetSdmaSqeFields(taskInfo.taskPara.Dma.sqeAddr, src, dst, size);
    errMsgInfo.taskSrcAddr = src;
    errMsgInfo.taskDstAddr = dst;
    errMsgInfo.size = static_cast<std::size_t>(size);
}

void HcclCommTaskExceptionLite::FillUbErrMsg(
    const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo, const rtLogicCqReport_t& exceptionInfo)
{
    GetEidFromChannelHandle(taskInfo, errMsgInfo.locEid, errMsgInfo.rmtEid);
    errMsgInfo.ubCqeStatus = exceptionInfo.errorCode & UB_CQE_STATUS_MASK;
    errMsgInfo.linkType = Hccl::DfxLinkType(static_cast<Hccl::DfxLinkType::Value>(taskInfo.linkType));
    errMsgInfo.size = taskInfo.taskPara.ubDma.size;
    errMsgInfo.taskSrcAddr = taskInfo.taskPara.ubDma.srcAddr;
    errMsgInfo.taskDstAddr = taskInfo.taskPara.ubDma.dstAddr;
    HCCL_ERROR(
        "[TaskException][AICPU]Error UB link info: ubCqeStatus[%u], localEid[%s], remoteEid[%s]. ",
        errMsgInfo.ubCqeStatus, errMsgInfo.locEid.Describe().c_str(), errMsgInfo.rmtEid.Describe().c_str());
}

void HcclCommTaskExceptionLite::FillReduceInlineErrMsg(
    const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo)
{
    errMsgInfo.reduceType = taskInfo.taskPara.Reduce.reduceOp;
}

HcclResult HcclCommTaskExceptionLite::SendTaskExceptionByMBox(
    const u32 notifyId, const u32 tsId, const rtLogicCqReport_t& exceptionInfo)
{
    ts_aicpu_msg_info_t aicpuSqe = {};
    u32 hostpid = 0;
    u32 vfId = 0;
    int pid = getpid();
    HCCL_INFO("[%s]getpid[%d]", __func__, pid);
    // 调整drvQueryProcessHostPid获取pid和vf_id的值
    CHK_RET(HrtHalDrvQueryProcessHostPid(pid, nullptr, &vfId, &hostpid, nullptr));

    aicpuSqe.pid = hostpid;
    aicpuSqe.cmd_type = TS_AICPU_RECORD;
    aicpuSqe.vf_id = vfId;
    aicpuSqe.tid = 0U; // notify is no need tid
    aicpuSqe.u.aicpu_record.record_type = AICPU_MSG_NOTIFY_RECORD_V2;
    aicpuSqe.u.aicpu_record.record_id = notifyId;
    aicpuSqe.ts_id = static_cast<uint8_t>(tsId);
    aicpuSqe.u.aicpu_record.fault_task_id = 0xffffffff;

    if (exceptionInfo.sqeType == ubSqeType) {
        aicpuSqe.u.aicpu_record.ret_code = SwitchUBCqeErrCodeToTsErrCode(exceptionInfo.errorCode & UB_CQE_STATUS_MASK);
    } else if (exceptionInfo.sqeType == sdmaSqeType) {
        aicpuSqe.u.aicpu_record.ret_code = SwitchSdmaCqeErrCodeToTsErrCode(exceptionInfo.errorCode);
    } else {
        aicpuSqe.u.aicpu_record.ret_code = TS_ERROR_HCCL_OTHER_ERROR;
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
    drvError_t ret = Hccl::DlHalFunctionV2::GetInstance().dlHalEschedSubmitEvent(devId_, &event);
    if (ret != DRV_ERROR_NONE) {
        HCCL_ERROR(
            "[%s]dlHalEschedSubmitEvent failed, ret=%d, notifyId=%u, hostpid=%u, vfId=%u, tsId=%u", __func__, ret,
            notifyId, hostpid, vfId, tsId);
        return HCCL_E_DRV;
    }
    HCCL_RUN_INFO(
        "[%s]finished, notifyId=%u, hostpid=%u, vfId=%u, tsId=%u, errorType=%u, errorCode=0x%x, ret_code=%u", __func__,
        notifyId, hostpid, vfId, tsId, exceptionInfo.errorType, exceptionInfo.errorCode,
        aicpuSqe.u.aicpu_record.ret_code);
    return HCCL_SUCCESS;
}

// 把UB类错误码转换成Ts对应的错误码
uint16_t HcclCommTaskExceptionLite::SwitchUBCqeErrCodeToTsErrCode(u32 cqeErrCode)
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

// 把SDMA类错误码转换成Ts对应的错误码
uint16_t HcclCommTaskExceptionLite::SwitchSdmaCqeErrCodeToTsErrCode(u32 cqeErrCode)
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

HcclResult HcclCommTaskExceptionLite::CollectTaskContext(
    CollCommAicpu* aicpuComm, u32 sqId, u32 taskId, std::vector<Hccl::DfxTaskInfo*>& taskContext)
{
    Hccl::TaskInfoCircularQueue* queue = GetTaskQueueBySqId(aicpuComm, sqId);
    CHK_PRT_RET(
        queue == nullptr, HCCL_ERROR("[%s]GetTaskQueueBySqId nullptr, devId[%u], sqId[%u].", __func__, devId_, sqId),
        HCCL_E_PARA);

    u32 targetTaskId = taskId;
    u16 begin = queue->GetBegin();
    Hccl::DfxTaskInfo* found = nullptr;
    u16 foundIdx = 0;
    for (u16 idx = 0; idx < queue->GetCapacity(); idx++) {
        Hccl::DfxTaskInfo* slot = queue->GetSlot(idx);
        if (slot != nullptr && slot->taskId == targetTaskId) {
            found = slot;
            foundIdx = idx;
            break;
        }
    }
    CHK_PRT_RET(
        found == nullptr,
        HCCL_ERROR("[%s]exception task not found, devId[%u], sqId[%u], taskId[%u]", __func__, devId_, sqId, taskId),
        HCCL_E_PARA);

    for (u32 ctxCount = 0; ctxCount < TASK_CONTEXT_SIZE; ++ctxCount) {
        u32 idx = (foundIdx + queue->GetCapacity() - ctxCount) % queue->GetCapacity();
        CHK_PRT_BREAK(
            idx == begin,
            HCCL_ERROR("[%s]reach queue begin, taskNum[%u], stop traversal", __func__, taskContext.size()), );

        Hccl::DfxTaskInfo* slot = queue->GetSlot(idx);
        if (slot == nullptr || slot->taskId > targetTaskId) {
            HCCL_ERROR(
                "[%s]prev taskId[%u] is bigger than err taskId[%u], taskNum[%u], stop traversal", __func__,
                slot != nullptr ? slot->taskId : INVALID_U32, targetTaskId, taskContext.size());
            break;
        }
        taskContext.push_back(slot);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommTaskExceptionLite::PrintTaskContextInfo(CollCommAicpu* aicpuComm, u32 sqId, u32 taskId)
{
    std::vector<Hccl::DfxTaskInfo*> taskContext{};
    CHK_PRT_RET(
        CollectTaskContext(aicpuComm, sqId, taskId, taskContext) != HCCL_SUCCESS,
        HCCL_ERROR("[%s]CollectTaskContext failed, devId[%u], sqId[%u], taskId[%u]", __func__, devId_, sqId, taskId),
        HCCL_E_PARA);

    std::string taskContextInfo = "";
    Hccl::DfxTaskInfo* lastTask = nullptr;
    for (u32 i = 0; i < taskContext.size(); ++i) {
        if (taskContext[i] == nullptr) {
            continue;
        }
        if (lastTask == nullptr) {
            lastTask = taskContext[i];
        }
        std::string conciseInfo = GetConciseTaskName(*taskContext[i]) + ",";
        u32 lastOpIndex = GetOpIndex(lastTask);
        u32 curOpIndex = GetOpIndex(taskContext[i]);
        bool overSize = (taskContextInfo.size() + conciseInfo.size()) >= TASK_CONTEXT_INFO_SIZE;
        if (overSize || (lastOpIndex != curOpIndex)) {
            PrintOpDataInfo(lastTask);
            HCCL_ERROR("[TaskException][AICPU]task sequence is OP(%u): %s", lastOpIndex, taskContextInfo.c_str());
            taskContextInfo = "";
            lastTask = taskContext[i];
        }
        taskContextInfo += conciseInfo;
    }

    if (!taskContextInfo.empty() && lastTask != nullptr) {
        u32 lastOpIndex = GetOpIndex(lastTask);
        PrintOpDataInfo(lastTask);
        HCCL_ERROR("[TaskException][AICPU]task sequence is OP(%u): %s", lastOpIndex, taskContextInfo.c_str());
    }
    HCCL_ERROR("[TaskException][AICPU]task sequence end.");
    return HCCL_SUCCESS;
}

std::string HcclCommTaskExceptionLite::GetGroupInfo(CollCommAicpu* aicpuComm)
{
    if (aicpuComm == nullptr) {
        HCCL_ERROR("[%s]aicpuComm is nullptr, return empty string.", __func__);
        return "";
    }
    return Hccl::StringFormat(
        "group:[%s], rankSize:[%u], localRank:[%u]", aicpuComm->GetIdentifier().c_str(),
        aicpuComm->GetTopoInfo().userRankSize, aicpuComm->GetTopoInfo().userRank);
}

Hccl::DfxTaskInfo* HcclCommTaskExceptionLite::FindDfxTaskInfo(CollCommAicpu* aicpuComm, u32 sqId, u32 sqeId)
{
    Hccl::TaskInfoCircularQueue* queue = GetTaskQueueBySqId(aicpuComm, sqId);
    if (queue == nullptr || queue->IsEmpty()) {
        HCCL_ERROR("[%s]GetTaskQueueBySqId nullptr or queue is empty, devId[%u], sqId[%u].", __func__, devId_, sqId);
        return nullptr;
    }
    u32 targetTaskId = sqeId;
    for (u16 idx = 0; idx < queue->GetCapacity(); idx++) {
        Hccl::DfxTaskInfo* slot = queue->GetSlot(idx);
        if (slot != nullptr && slot->taskId == targetTaskId) {
            return slot;
        }
    }
    HCCL_ERROR("[%s]exception task not found, devId[%u], sqId[%u], sqeId[%u]", __func__, devId_, sqId, sqeId);
    return nullptr;
}

Hccl::TaskInfoCircularQueue* HcclCommTaskExceptionLite::GetTaskQueueBySqId(CollCommAicpu* aicpuComm, u32 sqId)
{
    std::shared_lock<std::shared_mutex> threadRwlock(aicpuComm->GetCommEngineResMgr()->GetThreadMutex());
    const std::vector<std::shared_ptr<hccl::Thread>> threads = aicpuComm->GetCommEngineResMgr()->GetAllThread();
    for (auto& thread : threads) {
        Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(thread->GetStreamLitePtr());
        if (streamLite != nullptr && streamLite->GetSqId() == sqId) {
            return streamLite->GetTaskInfos();
        }
    }
    return nullptr;
}

void HcclCommTaskExceptionLite::GetEidFromChannelHandle(
    const Hccl::DfxTaskInfo& taskInfo, Hccl::Eid& locEid, Hccl::Eid& rmtEid)
{
    if (taskInfo.channelHandle != DFX_INVALID_U64) {
        auto* transport = reinterpret_cast<Hccl::UbTransportLiteImpl*>(taskInfo.channelHandle);
        locEid = transport->GetLocEid();
        rmtEid = transport->GetRmtEid();
    } else {
        HCCL_ERROR(
            "[TaskException][AICPU][%s] channelHandle is invalid, sqId[%u], taskId[%u], taskType[%s]", __func__,
            taskInfo.sqId, taskInfo.taskId, Hccl::TaskParamTypeValToStr(taskInfo.taskType).c_str());
    }
}

u32 HcclCommTaskExceptionLite::GetRemoteRankId(const Hccl::DfxTaskInfo& taskInfo)
{
    CHK_PRT_RET(
        taskInfo.dfxOpInfo == 0 || taskInfo.dfxOpInfo == DFX_INVALID_U64,
        HCCL_ERROR(
            "[TaskException][AICPU][%s] fail, dfxOpInfo[%llu] is invalid, sqId[%u], taskId[%u], taskType[%s]", __func__,
            taskInfo.dfxOpInfo, taskInfo.sqId, taskInfo.taskId, Hccl::TaskParamTypeValToStr(taskInfo.taskType).c_str()),
        Hccl::DFX_INVALID_RANKID);

    auto* opInfo = reinterpret_cast<const Hccl::DfxDfxOpInfo*>(taskInfo.dfxOpInfo);
    CHK_PRT_RET(
        opInfo->hcclCommDfxLite == nullptr,
        HCCL_ERROR(
            "[TaskException][AICPU][%s] fail, hcclCommDfxLite is nullptr, sqId[%u], taskId[%u], algTag[%s]", __func__,
            taskInfo.sqId, taskInfo.taskId, opInfo->algTag),
        Hccl::DFX_INVALID_RANKID);

    return static_cast<hccl::HcclCommDfxLite*>(opInfo->hcclCommDfxLite)->GetChannelRemoteRankId(taskInfo.channelHandle);
}

u32 HcclCommTaskExceptionLite::GetNotifyIdFromSqe(u64 sqeAddr)
{
    u32 notifyId = INVALID_U32;
    void* sqePtr = reinterpret_cast<void*>(sqeAddr);
    if (sqePtr == nullptr) {
        HCCL_ERROR("[TaskException][AICPU][%s] sqeAddr is nullptr", __func__);
        return notifyId;
    }
    auto* header = reinterpret_cast<Hccl::Rt91095StarsSqeHeader*>(sqePtr);
    auto sqeType = static_cast<Hccl::Rt91095StarsSqeType>(header->type);
    if (sqeType == Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD
        || sqeType == Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT
        || sqeType == Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_PLACE_HOLDER) {
        auto* notifySqe = reinterpret_cast<Hccl::Rt91095StarsNotifySqe*>(sqePtr);
        notifyId = notifySqe->notifyId;
    } else {
        HCCL_ERROR(
            "[TaskException][AICPU][%s] sqeType[%u] is not NOTIFY_RECORD or NOTIFY_WAIT", __func__, header->type);
    }
    return notifyId;
}

void HcclCommTaskExceptionLite::GetSdmaSqeFields(u64 sqeAddr, u64& src, u64& dst, u64& size)
{
    constexpr u32 ADDR_WORD_SHIFT_BITS = 32;
    void* sqePtr = reinterpret_cast<void*>(sqeAddr);
    CHK_PRT_RET(sqePtr == nullptr, HCCL_ERROR("[TaskException][AICPU][%s] sqeAddr is nullptr", __func__), );
    auto* header = reinterpret_cast<Hccl::Rt91095StarsSqeHeader*>(sqePtr);
    if (static_cast<Hccl::Rt91095StarsSqeType>(header->type) == Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA) {
        auto* dmaSqe = reinterpret_cast<Hccl::Rt91095StarsMemcpySqe*>(sqePtr);
        src = (static_cast<u64>(dmaSqe->u.strideMode0.srcAddrHigh) << ADDR_WORD_SHIFT_BITS)
              | dmaSqe->u.strideMode0.srcAddrLow;
        dst = (static_cast<u64>(dmaSqe->u.strideMode0.dstAddrHigh) << ADDR_WORD_SHIFT_BITS)
              | dmaSqe->u.strideMode0.dstAddrLow;
        size = dmaSqe->u.strideMode0.lengthMove;
    } else {
        HCCL_ERROR("[TaskException][AICPU][%s] sqeType[%u] is not SDMA", __func__, header->type);
    }
}

std::string HcclCommTaskExceptionLite::GetNotifyInfo(const Hccl::DfxTaskInfo& taskInfo)
{
    auto taskType = static_cast<Hccl::TaskParamTypeVal>(taskInfo.taskType);
    u32 notifyId = INVALID_U32;
    switch (taskType) {
        case Hccl::TaskParamTypeVal::TASK_UB_INLINE_WRITE:
        case Hccl::TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY:
            notifyId = taskInfo.taskPara.ubDma.notifyId;
            break;
        case Hccl::TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY:
            notifyId = taskInfo.taskPara.Reduce.notifyId;
            break;
        case Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD:
        case Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT:
        case Hccl::TaskParamTypeVal::TASK_SEND_NOTIFY: {
            notifyId = GetNotifyIdFromSqe(taskInfo.taskPara.Notify.sqeAddr);
            break;
        }
        case Hccl::TaskParamTypeVal::TASK_RDMA: {
            notifyId = GetNotifyIdFromSqe(taskInfo.taskPara.Dma.sqeAddr);
            break;
        }
        default:
            return "/";
    }
    return (notifyId == INVALID_U32) ? "/" : std::to_string(notifyId);
}

std::string HcclCommTaskExceptionLite::GetConciseTaskName(const Hccl::DfxTaskInfo& taskInfo)
{
    const auto& taskConciseNameMap = Hccl::GetTaskConciseNameMap();
    auto it = taskConciseNameMap.find(taskInfo.taskType);
    std::string name = (it != taskConciseNameMap.end()) ? it->second : "UNKNOWN";
    u32 remoteRank = GetRemoteRankId(taskInfo);
    std::string rankStr = (remoteRank == Hccl::DFX_INVALID_RANKID) ? "/" : std::to_string(remoteRank);
    auto taskType = static_cast<Hccl::TaskParamTypeVal>(taskInfo.taskType);
    if (taskType == Hccl::TaskParamTypeVal::TASK_RDMA || taskType == Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD
        || taskType == Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT || taskType == Hccl::TaskParamTypeVal::TASK_SEND_NOTIFY
        || taskType == Hccl::TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY
        || taskType == Hccl::TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY
        || taskType == Hccl::TaskParamTypeVal::TASK_UB_INLINE_WRITE) {
        return name + "(" + rankStr + "," + GetNotifyInfo(taskInfo) + ")";
    }
    return name + "(" + rankStr + ")";
}

u32 HcclCommTaskExceptionLite::GetOpIndex(const Hccl::DfxTaskInfo* taskInfo)
{
    if (taskInfo == nullptr) {
        HCCL_ERROR("[TaskException][AICPU][%s] taskInfo is nullptr", __func__);
        return UINT32_MAX;
    }
    if (taskInfo->dfxOpInfo == DFX_INVALID_U64) {
        HCCL_ERROR(
            "[TaskException][AICPU][%s] dfxOpInfo is invalid, sqId[%u], taskId[%u], taskType[%s]", __func__,
            taskInfo->sqId, taskInfo->taskId, Hccl::TaskParamTypeValToStr(taskInfo->taskType).c_str());
        return UINT32_MAX;
    }
    return reinterpret_cast<const Hccl::DfxDfxOpInfo*>(taskInfo->dfxOpInfo)->opIndex;
}

void HcclCommTaskExceptionLite::PrintOpDataInfo(const Hccl::DfxTaskInfo* taskInfo)
{
    if (taskInfo == nullptr || taskInfo->dfxOpInfo == DFX_INVALID_U64) {
        HCCL_ERROR("[TaskException][AICPU]opData information is (dfxOpInfo unavailable).");
        return;
    }
    const Hccl::DfxDfxOpInfo* opInfo = reinterpret_cast<const Hccl::DfxDfxOpInfo*>(taskInfo->dfxOpInfo);
    HCCL_ERROR(
        "[TaskException][AICPU]opData information is opIndex[%u], algTag[%s], count[%llu], "
        "dataType[%u], input: ptr[0x%llx] size[%llu], output: ptr[0x%llx] size[%llu].",
        opInfo->opIndex, opInfo->algTag, opInfo->count, opInfo->dataType, opInfo->srcAddr, opInfo->srcSize,
        opInfo->dstAddr, opInfo->dstSize);
}

std::string HcclCommTaskExceptionLite::GetParaInfo(const Hccl::DfxTaskInfo& taskInfo)
{
    auto taskType = static_cast<Hccl::TaskParamTypeVal>(taskInfo.taskType);
    u32 remoteRank = GetRemoteRankId(taskInfo);
    std::string remoteRankStr = (remoteRank == Hccl::DFX_INVALID_RANKID) ? "local" : std::to_string(remoteRank);
    std::string linkTypeStr = Hccl::DfxLinkTypeValToStr(taskInfo.linkType);

    switch (taskType) {
        case Hccl::TaskParamTypeVal::TASK_UB:
        case Hccl::TaskParamTypeVal::TASK_UB_INLINE_WRITE:
        case Hccl::TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY:
            return Hccl::StringFormat(
                "src:[0x%llx], dst:[0x%llx], size:[0x%llx], notify id:[%u], link type:[%s], remote rank:[%s]",
                taskInfo.taskPara.ubDma.srcAddr, taskInfo.taskPara.ubDma.dstAddr, taskInfo.taskPara.ubDma.size,
                taskInfo.taskPara.ubDma.notifyId, linkTypeStr.c_str(), remoteRankStr.c_str());
        case Hccl::TaskParamTypeVal::TASK_UB_REDUCE_INLINE:
        case Hccl::TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY:
        case Hccl::TaskParamTypeVal::TASK_REDUCE_INLINE:
            return Hccl::StringFormat(
                "src:[0x%llx], dst:[0x%llx], size:[0x%llx], notify id:[%u], op:[%u], link type:[%s], remote "
                "rank:[%s]",
                taskInfo.taskPara.Reduce.srcAddr, taskInfo.taskPara.Reduce.dstAddr, taskInfo.taskPara.Reduce.size,
                taskInfo.taskPara.Reduce.notifyId, static_cast<u32>(taskInfo.taskPara.Reduce.reduceOp),
                linkTypeStr.c_str(), remoteRankStr.c_str());
        case Hccl::TaskParamTypeVal::TASK_SDMA: {
            u64 src = 0;
            u64 dst = 0;
            u64 size = 0;
            GetSdmaSqeFields(taskInfo.taskPara.Dma.sqeAddr, src, dst, size);
            return Hccl::StringFormat(
                "src:[0x%llx], dst:[0x%llx], size:[0x%llx], link type:[%s], remote rank:[%s]", src, dst, size,
                linkTypeStr.c_str(), remoteRankStr.c_str());
        }
        case Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD:
        case Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT:
            return Hccl::StringFormat(
                "notify id:[%u], value:[%u], remote rank:[%s]", GetNotifyIdFromSqe(taskInfo.taskPara.Notify.sqeAddr), 1,
                remoteRankStr.c_str());
        default:
            return Hccl::StringFormat("taskType[%s] not match", Hccl::TaskParamTypeValToStr(taskInfo.taskType).c_str());
    }
    return "";
}
} // namespace hcomm
