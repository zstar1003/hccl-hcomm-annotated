/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "exception_handle.h"
#include "log.h"
#include "thread.h"
#include "stream_lite.h"
#include "aicpu_thread_process.h"
#include "exception_callback_mgr.h"
#include "exception_util.h"
#include "ascend_hal.h"

using Hccl::HcclException;
using std::exception;
using std::string;

namespace hcomm {
constexpr u32 URMA_STARTS_CQE_FAILEDERR = 0x5;

ExceptionHandle& ExceptionHandle::GetInstance()
{
    static ExceptionHandle instance;
    return instance;
}

uint32_t ExceptionHandle::GetSqeId(uint16_t taskId, uint16_t streamId)
{
    constexpr uint32_t TASK_ID_SHIFT_BITS = 16;
    return (static_cast<uint32_t>(taskId) << TASK_ID_SHIFT_BITS) | static_cast<uint32_t>(streamId);
}

HcclResult ExceptionHandle::CheckRepeatBySqeId(
    Hccl::StreamLite* streamLite, [[maybe_unused]] uint32_t devId, uint16_t taskId, uint16_t streamId)
{
    CHK_PTR_NULL(streamLite);

    const uint32_t sqId = streamLite->GetSqId();
    const uint32_t sqeId = GetSqeId(taskId, streamId);

    auto it = threadsPrinted_.find(sqId);
    if (it != threadsPrinted_.end() && it->second == sqeId) {
        return HCCL_E_AGAIN;
    }
    threadsPrinted_[sqId] = sqeId;
    return HCCL_SUCCESS;
}

dfx::CqeStatus
ExceptionHandle::ReceiveCqeReport(uint32_t devId, Hccl::StreamLite* streamLite, rtLogicCqReport_t& cqeException)
{
    constexpr uint32_t REPORT_SIZE = 1;
    rtLogicCqReport_t streamReport[REPORT_SIZE] = {};

    CqeQueryInput cqeQueryInput{};
    cqeQueryInput.devId = devId;
    cqeQueryInput.streamId = streamLite->GetId();
    cqeQueryInput.sqId = streamLite->GetSqId();
    cqeQueryInput.cqId = streamLite->GetCqId();
    cqeQueryInput.type = static_cast<uint32_t>(DRV_LOGIC_TYPE);
    cqeQueryInput.cqeAddr = ReinterpretAs<uint8_t*>(streamReport);

    return CqReportRecv(cqeQueryInput, cqeException);
}

HcclResult ExceptionHandle::CheckExceptionCqe(hccl::Thread* thread, uint32_t devId)
{
    CHK_PTR_NULL(thread);

    Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(thread->GetStreamLitePtr());
    CHK_PTR_NULL(streamLite);

    uint32_t sqId = streamLite->GetSqId();
    if (sqCqeErrorSet_.count(sqId) > 0) {
        return HCCL_SUCCESS;
    }

    rtLogicCqReport_t cqeException{};
    dfx::CqeStatus cqeStatus = ReceiveCqeReport(devId, streamLite, cqeException);
    if (cqeStatus == dfx::CqeStatus::kCqeTimeOut) {
        cqeException.taskId = 0xFFFF;
    }

    if (cqeStatus == dfx::CqeStatus::kCqeInnerError) {
        HCCL_ERROR(
            "[ExceptionHandle][CheckExceptionCqe] CqReportRecv internal error, "
            "devId[%u], streamId[%u], sqId[%u], skip this sq in future polls",
            devId, streamLite->GetId(), sqId);
        sqCqeErrorSet_.insert(sqId);
        return HCCL_E_INTERNAL;
    }

    if (cqeStatus == dfx::CqeStatus::kDefault) {
        return HCCL_SUCCESS;
    }

    if (cqeException.sqeType != DFX_SQE_TYPE_UDMA) {
        return HCCL_SUCCESS;
    }

    if (CheckRepeatBySqeId(streamLite, devId, cqeException.taskId, cqeException.streamId) != HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    // 前6位为0，则表示无错误
    if ((cqeException.errorType & RT_STARS_EXIST_ERROR) == 0U) {
        return HCCL_SUCCESS;
    }

    HCCL_ERROR(
        "[ExceptionHandle][CheckExceptionCqe] CQE exception detected, "
        "devId[%u], streamId[%u], taskId[%u], errorCode[0x%x], errorType[0x%x]",
        devId, cqeException.streamId, cqeException.taskId, cqeException.errorCode, cqeException.errorType);
    HcommExceptionInfo exceptionInfo{};
    CHK_RET(FillExceptionInfo(exceptionInfo, thread, cqeException.taskId, cqeException));
    ExceptionCallbackMgr::GetInstance().NotifyAll(exceptionInfo);

    return HCCL_E_ROCE_TRANSFER;
}

// 把UB类错误码转换成hccl result对应的错误码
uint32_t ExceptionHandle::SwitchCqeErrCodeToHcclErrCode(uint32_t cqeErrCode)
{
    switch (cqeErrCode) {
        case URMA_STARTS_CQE_FAILEDERR:
            return HCCL_E_INTERNAL;
        default:
            return HCCL_E_ROCE_TRANSFER;
    }
}

HcclResult ExceptionHandle::FillExceptionInfo(
    HcommExceptionInfo& exceptionInfo, hccl::Thread* thread, uint32_t taskId, const rtLogicCqReport_t& cqeException)
{
    CHK_PTR_NULL(thread);

    exceptionInfo.thread = ReinterpretAs<uint64_t>(thread);
    exceptionInfo.channel = 0;
    exceptionInfo.taskId = taskId;
    exceptionInfo.retCode = SwitchCqeErrCodeToHcclErrCode(cqeException.errorCode & 0xFF);

    exceptionInfo.expandInfo.type = HCOMM_EXCEPTION_STARS;
    exceptionInfo.expandInfo.detail.starsInfo.starsErrcode = cqeException.errorType;
    exceptionInfo.expandInfo.detail.starsInfo.sqeType = cqeException.sqeType;
    exceptionInfo.expandInfo.detail.starsInfo.statusMerged = cqeException.errorCode & 0xFF;

    return HCCL_SUCCESS;
}

HcclResult ExceptionHandle::HandleExceptionCqe()
{
    std::shared_lock<std::shared_mutex> rwlock(AicpuThreadProcess::GetMutex());
    std::vector<std::shared_ptr<hccl::Thread>> threads = AicpuThreadProcess::GetThreads();

    for (auto& thread : threads) {
        if (thread == nullptr) {
            continue;
        }
        Hccl::StreamLite* streamLite = static_cast<Hccl::StreamLite*>(thread->GetStreamLitePtr());
        if (streamLite == nullptr) {
            continue;
        }

        uint32_t phyId = streamLite->GetDevPhyId();
        uint32_t localDevId = 0;
        drvError_t drvRet = drvGetLocalDevIDByHostDevID(phyId, &localDevId);
        if (drvRet != DRV_ERROR_NONE) {
            HCCL_ERROR(
                "[ExceptionHandle][HandleExceptionCqe] drvGetLocalDevIDByHostDevID failed, "
                "phyId=%u, ret=%d",
                phyId, drvRet);
            continue;
        }

        CheckExceptionCqe(thread.get(), localDevId);
    }
    return HCCL_SUCCESS;
}

void ExceptionHandle::Call()
{
    if (ExceptionCallbackMgr::GetInstance().IsEmpty()) {
        return;
    }
    TRY_CATCH_PRINT_ERROR(HandleExceptionCqe());
}

void ExceptionHandle::ClearStreamState(uint32_t sqId)
{
    threadsPrinted_.erase(sqId);
    sqCqeErrorSet_.erase(sqId);
    HCCL_INFO("[ExceptionHandle][%s] sqId[%u] state cleared", __func__, sqId);
}

} // namespace hcomm
