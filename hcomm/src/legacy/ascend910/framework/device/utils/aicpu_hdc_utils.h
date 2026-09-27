/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AICPU_HDC_UTILS_H
#define HCCL_AICPU_HDC_UTILS_H

#include <memory>
#include "aicpu_operator_pub.h"
#include "hdc_pub.h"

class AicpuHdcUtils {
public:
    static HcclResult InitOpExecStatus(std::shared_ptr<hccl::HDCommunicate> d2hTransfer, HcclOpIdentifier& opId);
    static HcclResult GetOpExecCtrlCmd(std::shared_ptr<hccl::HDCommunicate> h2dTransfer, KfcCommand& cmd);
    static HcclResult SetOpExecStatus(
        std::shared_ptr<hccl::HDCommunicate> d2hTransfer, KfcStatus state, KfcError errorCode, u32 retryCount);
    static HcclResult
    GetSuspendingStatus(std::shared_ptr<hccl::HDCommunicate> h2dTransfer, HcclComSuspendingFlag& flag);
    static HcclResult GetBackGroundCommand(std::shared_ptr<hccl::HDCommunicate> h2dTransfer, BackgroundCommand& bgCmd);
    static HcclResult
    ResponseBackGroundStatus(std::shared_ptr<hccl::HDCommunicate> d2hTransfer, KfcExecStatus& kfcStatus);
    static HcclResult SetErrorMessage(std::shared_ptr<hccl::HDCommunicate> d2hTransfer, ErrorMessageReport& emrInfo);
    static HcclResult GetKfcCommand(const std::shared_ptr<hccl::HDCommunicate>& h2dTransfer, KfcCommand& cmd);
};
#endif // HCCL_AICPU_HDC_UTILS_H
