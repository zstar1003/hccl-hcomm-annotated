/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_AICPU_HDC_HANDLER_H
#define HCCLV2_AICPU_HDC_HANDLER_H

#include "hdc_lite.h"
#include "kfc.h"

namespace Hccl {

class AicpuHdcHandler {
public:
    AicpuHdcHandler(HDCommunicateLite& h2dTransfer, HDCommunicateLite& d2hTransfer);
    ~AicpuHdcHandler() = default;

    KfcCommand GetKfcCommand();
    void SetKfcExecStatus(KfcStatus state, KfcErrType errorCode) const;

private:
    HDCommunicateLite* h2dTransfer_{nullptr};
    HDCommunicateLite* d2hTransfer_{nullptr};
    KfcCommand lastCmd_{KfcCommand::NONE};
};

} // namespace Hccl

#endif // HCCLV2_AICPU_HDC_HANDLER_H
