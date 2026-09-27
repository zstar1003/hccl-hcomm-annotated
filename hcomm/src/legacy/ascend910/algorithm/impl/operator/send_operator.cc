/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_operator.h"

namespace hccl {
SendOperator::SendOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_SEND)
{}

SendOperator::~SendOperator() {}

HcclResult
SendOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag)
{
    algName = "SendExecutor";
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = tag;
    } else {
        newTag = tag + algName;
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_INFO("[SelectAlg] Send newTag is [%s]", newTag.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_SEND, Send, SendOperator);
} // namespace hccl
