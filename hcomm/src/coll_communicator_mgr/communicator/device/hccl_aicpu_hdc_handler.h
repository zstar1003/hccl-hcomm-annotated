/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AICPU_HDC_HANDLER_H
#define HCCL_AICPU_HDC_HANDLER_H

#include "hdc_pub.h"
#include "kfc.h"

namespace hccl {

class HcclAicpuHdcHandler {
public:
    HcclAicpuHdcHandler(
        const std::shared_ptr<HDCommunicate>& h2dTransfer, const std::shared_ptr<HDCommunicate>& d2hTransfer);
    ~HcclAicpuHdcHandler() = default;

    HcclResult GetKfcCommand(Hccl::KfcCommand& cmd);
    void SetKfcExecStatus(Hccl::KfcStatus state, Hccl::KfcErrType errorCode);

private:
    std::shared_ptr<HDCommunicate> h2dTransfer_{nullptr};
    std::shared_ptr<HDCommunicate> d2hTransfer_{nullptr};
    Hccl::KfcCommand lastCmd_{Hccl::KfcCommand::NONE};
};

} // namespace hccl

#endif // HCCL_AICPU_HDC_HANDLER_H
