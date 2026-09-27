/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_hdc_utils.h"

using namespace hccl;

HcclResult AicpuHdcUtils::InitOpExecStatus(std::shared_ptr<hccl::HDCommunicate> d2hTransfer, HcclOpIdentifier& opId)
{
    if (!d2hTransfer) {
        return HCCL_SUCCESS;
    }
    KfcExecStatus status;
    status.opId = opId;
    status.execStatus.kfcStatus = KfcStatus::kRuning;
    status.execStatus.kfcError = KfcError::kNone;
    status.execStatus.retryInfo.retryCount = 0;
    HCCL_INFO("InitOpExecStatus: id:%u", status.opId.index);
    return d2hTransfer->Put(0, sizeof(status), reinterpret_cast<uint8_t*>(&status));
}

HcclResult AicpuHdcUtils::SetErrorMessage(std::shared_ptr<hccl::HDCommunicate> d2hTransfer, ErrorMessageReport& emrInfo)
{
    if (!d2hTransfer) {
        return HCCL_SUCCESS;
    }

    return d2hTransfer->Put(
        sizeof(HcclOpIdentifier) + sizeof(ExecStatusDef), sizeof(emrInfo), reinterpret_cast<uint8_t*>(&emrInfo));
}

HcclResult AicpuHdcUtils::SetOpExecStatus(
    std::shared_ptr<HDCommunicate> d2hTransfer, KfcStatus state, KfcError errorCode, u32 retryCount)
{
    if (!d2hTransfer) {
        return HCCL_SUCCESS;
    }
    KfcExecStatus status;
    status.execStatus.kfcStatus = state;
    status.execStatus.kfcError = errorCode;
    status.execStatus.retryInfo.retryCount = retryCount;
    HCCL_INFO("SetOpExecStatus: state:%u", state);
    return d2hTransfer->Put(
        sizeof(status.opId), sizeof(status.execStatus), reinterpret_cast<uint8_t*>(&status.execStatus));
}

HcclResult AicpuHdcUtils::GetOpExecCtrlCmd(std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    if (!h2dTransfer) {
        return HCCL_SUCCESS;
    }
    static KfcCommand lastCmd = KfcCommand::kNone;
    KfcCommand newCmd = KfcCommand::kNone;
    CHK_RET(h2dTransfer->Get(0, sizeof(newCmd), reinterpret_cast<uint8_t*>(&newCmd)));

    if ((newCmd == KfcCommand::kExit) || (newCmd == KfcCommand::kStopLaunch) || (newCmd == KfcCommand::NsStopLaunch)) {
        cmd = newCmd;
    } else {
        cmd = (lastCmd != newCmd) ? newCmd : KfcCommand::kNone;
    }
    lastCmd = newCmd;
    if (cmd != KfcCommand::kNone) {
        HCCL_INFO("hccl aicpu get cmd:%u", cmd);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHdcUtils::GetSuspendingStatus(std::shared_ptr<HDCommunicate> h2dTransfer, HcclComSuspendingFlag& flag)
{
    if (!h2dTransfer) {
        return HCCL_SUCCESS;
    }
    u32 startOffset = sizeof(KfcCommand) + sizeof(BackgroundCommand);
    CHK_RET(h2dTransfer->Get(startOffset, sizeof(HcclComSuspendingFlag), reinterpret_cast<uint8_t*>(&flag)));
    return HCCL_SUCCESS;
}

HcclResult AicpuHdcUtils::GetBackGroundCommand(std::shared_ptr<HDCommunicate> h2dTransfer, BackgroundCommand& bgCmd)
{
    if (!h2dTransfer) {
        return HCCL_SUCCESS;
    }
    u32 startOffset = sizeof(KfcCommand);
    CHK_RET(h2dTransfer->Get(startOffset, sizeof(BackgroundCommand), reinterpret_cast<uint8_t*>(&bgCmd)));
    return HCCL_SUCCESS;
}

HcclResult AicpuHdcUtils::ResponseBackGroundStatus(std::shared_ptr<HDCommunicate> d2hTransfer, KfcExecStatus& kfcStatus)
{
    if (!d2hTransfer) {
        return HCCL_SUCCESS;
    }
    CHK_RET(d2hTransfer->Put(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&kfcStatus)));
    return HCCL_SUCCESS;
}

HcclResult AicpuHdcUtils::GetKfcCommand(const std::shared_ptr<HDCommunicate>& h2dTransfer, KfcCommand& cmd)
{
    if (!h2dTransfer) {
        return HCCL_SUCCESS;
    }
    CHK_RET(h2dTransfer->Get(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&cmd)));
    return HCCL_SUCCESS;
}
