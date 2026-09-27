/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AICPU_HDC_H
#define HCCL_AICPU_HDC_H

#include <memory>
#include "aicpu_operator_pub.h"
#include "hdc_pub.h"

class AicpuHdc {
public:
    AicpuHdc() {};
    ~AicpuHdc() {};

    HcclResult InitOpExecStatus(std::shared_ptr<hccl::HDCommunicate> d2hTransfer, HcclOpIdentifier& opId);
    HcclResult GetOpExecCtrlCmd(std::shared_ptr<hccl::HDCommunicate> h2dTransfer, KfcCommand& cmd);
    HcclResult SetOpExecStatus(
        std::shared_ptr<hccl::HDCommunicate> d2hTransfer, KfcStatus state, KfcError errorCode, u32 retryCount);
    HcclResult SetOpExecStatus(
        std::shared_ptr<hccl::HDCommunicate> d2hTransfer, HcclOpIdentifier& opId, KfcStatus state, KfcError errorCode,
        u32 retryCount);
    HcclResult GetOpExecCtrlTargetOp(std::shared_ptr<hccl::HDCommunicate> h2dTransfer, HcclOpIdentifier& opId);
    HcclResult GetOpExecChangeLink(std::shared_ptr<hccl::HDCommunicate> h2dTransfer, ChangeLinkInfo& changeLinkInfo);
    HcclResult SetErrorMessage(std::shared_ptr<hccl::HDCommunicate> d2hTransfer, ErrorMessageReport& emrInfo);

private:
    KfcCommand lastCmd_ = KfcCommand::kNone;
};

#endif // HCCL_AICPU_HDC_H
