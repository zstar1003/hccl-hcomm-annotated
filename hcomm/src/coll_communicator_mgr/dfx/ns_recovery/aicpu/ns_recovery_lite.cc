/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ns_recovery_lite.h"

namespace hccl {

NsRecoveryLite::NsRecoveryLite() {}

void NsRecoveryLite::Init(
    const std::shared_ptr<HDCommunicate>& kfcControlTransferH2D,
    const std::shared_ptr<HDCommunicate>& kfcStatusTransferD2H)
{
    hdcHandler_ = std::make_unique<HcclAicpuHdcHandler>(kfcControlTransferH2D, kfcStatusTransferD2H);
}

Hccl::KfcCommand NsRecoveryLite::BackGroundGetCmd()
{
    std::unique_lock<std::mutex> lock(hdcShmLock_);
    Hccl::KfcCommand cmd{Hccl::KfcCommand::NONE};
    auto ret = hdcHandler_->GetKfcCommand(cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("BackGroundGetCmd failed, ret = 0x%016llx, cmd = %s", HCCL_ERROR_CODE(ret), cmd.Describe().c_str());
    }
    return cmd;
}

void NsRecoveryLite::BackGroundSetStatus(Hccl::KfcStatus status, Hccl::KfcErrType errorCode)
{
    std::unique_lock<std::mutex> lock(hdcShmLock_);
    hdcHandler_->SetKfcExecStatus(status, errorCode);
}

void NsRecoveryLite::SetNeedClean(bool flag) { needClean_ = flag; }
bool NsRecoveryLite::IsNeedClean() const { return needClean_; }

} // namespace hccl
