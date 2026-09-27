/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_aicpu_hdc_handler.h"
#include "log.h"

namespace hccl {

HcclAicpuHdcHandler::HcclAicpuHdcHandler(
    const std::shared_ptr<HDCommunicate>& h2dTransfer, const std::shared_ptr<HDCommunicate>& d2hTransfer)
    : h2dTransfer_(h2dTransfer),
      d2hTransfer_(d2hTransfer)
{}

HcclResult HcclAicpuHdcHandler::GetKfcCommand(Hccl::KfcCommand& cmd)
{
    auto ret = h2dTransfer_->Get(0, sizeof(Hccl::KfcCommand), reinterpret_cast<uint8_t*>(&cmd));
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[HcclAicpuHdcHandler][GetKfcCommand] h2dTransfer Get fail, ret[%d]", ret);
        return ret;
    }
    if (lastCmd_ != cmd) {
        HCCL_INFO(
            "[HcclAicpuHdcHandler][GetKfcCommand] Get new KfcCommand[%d], last KfcCommand[%d]", static_cast<int>(cmd),
            static_cast<int>(lastCmd_));
        lastCmd_ = cmd;
    }
    return HCCL_SUCCESS;
}

void HcclAicpuHdcHandler::SetKfcExecStatus(Hccl::KfcStatus state, Hccl::KfcErrType errorCode)
{
    Hccl::KfcExecStatus status;
    status.kfcStatus = state;
    status.kfcError = errorCode;
    HCCL_INFO(
        "[HcclAicpuHdcHandler][SetKfcExecStatus] SetKfcExecStatus: state[%d], errorCode[%d]", static_cast<int>(state),
        static_cast<int>(errorCode));
    auto ret = d2hTransfer_->Put(0, sizeof(Hccl::KfcExecStatus), reinterpret_cast<uint8_t*>(&status));
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[HcclAicpuHdcHandler][SetKfcExecStatus] d2hTransfer Put fail, ret[%d]", ret);
    }
}

} // namespace hccl
