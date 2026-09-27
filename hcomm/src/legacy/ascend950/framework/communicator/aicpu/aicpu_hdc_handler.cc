/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_hdc_handler.h"
#include "log.h"
#include "internal_exception.h"
#include "exception_util.h"

namespace Hccl {

AicpuHdcHandler::AicpuHdcHandler(HDCommunicateLite& h2dTransfer, HDCommunicateLite& d2hTransfer)
    : h2dTransfer_(&h2dTransfer),
      d2hTransfer_(&d2hTransfer)
{}

KfcCommand AicpuHdcHandler::GetKfcCommand()
{
    KfcCommand cmd;
    auto ret = h2dTransfer_->Get(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&cmd));
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<InternalException>(StringFormat("[AicpuHdcHandler] h2dTransfer Get fail, ret[%d]", ret));
    }

    if (lastCmd_ != cmd) {
        HCCL_INFO("[AicpuHdcHandler] Get new KfcCommand[%u], last KfcCommand[%u]", cmd, lastCmd_);
        lastCmd_ = cmd;
    }
    return cmd;
}

void AicpuHdcHandler::SetKfcExecStatus(KfcStatus state, KfcErrType errorCode) const
{
    KfcExecStatus status;
    status.kfcStatus = state;
    status.kfcError = errorCode;
    HCCL_INFO("[AicpuHdcHandler] SetKfcExecStatus: state[%u], code[%u]", state, errorCode);
    auto ret = d2hTransfer_->Put(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&status));
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<InternalException>(StringFormat("[AicpuHdcHandler] d2hTransfer Put fail, ret[%d]", ret));
    }
}

} // namespace Hccl
