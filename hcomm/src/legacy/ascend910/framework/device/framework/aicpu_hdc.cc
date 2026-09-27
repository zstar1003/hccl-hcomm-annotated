/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_hdc.h"
#include "utils/aicpu_hdc_utils.h"

using namespace hccl;

HcclResult AicpuHdc::InitOpExecStatus(std::shared_ptr<hccl::HDCommunicate> d2hTransfer, HcclOpIdentifier& opId)
{
    return AicpuHdcUtils::InitOpExecStatus(d2hTransfer, opId);
}

HcclResult AicpuHdc::SetOpExecStatus(
    std::shared_ptr<HDCommunicate> d2hTransfer, KfcStatus state, KfcError errorCode, u32 retryCount)
{
    return AicpuHdcUtils::SetOpExecStatus(d2hTransfer, state, errorCode, retryCount);
}

HcclResult AicpuHdc::SetErrorMessage(std::shared_ptr<hccl::HDCommunicate> d2hTransfer, ErrorMessageReport& emrInfo)
{
    return AicpuHdcUtils::SetErrorMessage(d2hTransfer, emrInfo);
}

HcclResult AicpuHdc::GetOpExecCtrlCmd(std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    if (!h2dTransfer) {
        return HCCL_SUCCESS;
    }
    CHK_RET(h2dTransfer->Get(0, sizeof(cmd), reinterpret_cast<uint8_t*>(&cmd)));

    if (lastCmd_ != cmd) {
        HCCL_RUN_INFO("hccl aicpu get cmd:%u", cmd);
        lastCmd_ = cmd;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHdc::SetOpExecStatus(
    std::shared_ptr<HDCommunicate> d2hTransfer, HcclOpIdentifier& opId, KfcStatus state, KfcError errorCode,
    u32 retryCount)
{
    if (!d2hTransfer) {
        return HCCL_SUCCESS;
    }
    KfcExecStatus status;
    status.opId = opId;
    status.execStatus.kfcStatus = state;
    status.execStatus.kfcError = errorCode;
    status.execStatus.retryInfo.retryCount = retryCount;
    HCCL_DEBUG("SetOpExecStatus: state:%u", state);
    return d2hTransfer->Put(0, sizeof(status.opId) + sizeof(status.execStatus), reinterpret_cast<uint8_t*>(&status));
}

HcclResult AicpuHdc::GetOpExecCtrlTargetOp(std::shared_ptr<HDCommunicate> h2dTransfer, HcclOpIdentifier& opId)
{
    if (!h2dTransfer) {
        return HCCL_SUCCESS;
    }

    KfcExecControl ctrlCmd;
    CHK_RET(h2dTransfer->Get(0, sizeof(KfcExecControl), reinterpret_cast<uint8_t*>(&ctrlCmd)));

    opId = ctrlCmd.targetOp;
    HCCL_DEBUG(
        "[OpRetry][GetOpExecCtrlTargetOp]tag[%s], index[%u], srcRank[%u], detRank[%u], isSendRecv[%d], "
        "streamid[%u]",
        opId.tag, opId.index, opId.srcRank, opId.detRank, opId.isSendRecv, opId.streamId);
    return HCCL_SUCCESS;
}

HcclResult AicpuHdc::GetOpExecChangeLink(std::shared_ptr<HDCommunicate> h2dTransfer, ChangeLinkInfo& changeLinkInfo)
{
    if (!h2dTransfer) {
        return HCCL_SUCCESS;
    }

    KfcExecControl kfcExecControl;
    CHK_RET(h2dTransfer->Get(0, sizeof(KfcExecControl), reinterpret_cast<uint8_t*>(&kfcExecControl)));
    changeLinkInfo = kfcExecControl.changeLinkInfo;
    return HCCL_SUCCESS;
}
