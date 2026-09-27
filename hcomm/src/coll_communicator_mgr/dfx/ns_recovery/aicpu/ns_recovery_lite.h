/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NS_RECOVERY_LITE_H
#define NS_RECOVERY_LITE_H

#include "hdc_pub.h"
#include "kfc.h"
#include "hccl_aicpu_hdc_handler.h"
#include <memory>

namespace hccl {

class NsRecoveryLite {
public:
    NsRecoveryLite();
    void Init(
        const std::shared_ptr<HDCommunicate>& kfcControlTransferH2D,
        const std::shared_ptr<HDCommunicate>& kfcStatusTransferD2H);

    Hccl::KfcCommand BackGroundGetCmd();
    void BackGroundSetStatus(Hccl::KfcStatus status, Hccl::KfcErrType errorCode = Hccl::KfcErrType::NONE);
    void SetNeedClean(bool flag);
    bool IsNeedClean() const;

private:
    bool needClean_{false};

    std::unique_ptr<HcclAicpuHdcHandler> hdcHandler_{nullptr};
    std::mutex hdcShmLock_;
};

using NsRecoveryLitePtr = std::shared_ptr<NsRecoveryLite>;

} // namespace hccl

#endif
