/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batchsendrecv_operator.h"
namespace hccl {

BatchSendRecvOperator::BatchSendRecvOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV)
{}

BatchSendRecvOperator::~BatchSendRecvOperator() {}

HcclResult BatchSendRecvOperator::SelectAlg(
    const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag)
{
    if (retryEnable_ && param.aicpuUnfoldMode) {
        algName = "BatchSendRecvRetry";
    } else if (param.isGroupMode) {
        algName = "BatchSendRecvGroup";
    } else {
        algName = "BatchSendRecv";
    }
    newTag = tag;
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_INFO("[BatchSendRecvOperator][SelectAlg] algName %s newTag %s", algName.c_str(), newTag.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, BatchSendRecv, BatchSendRecvOperator);
} // namespace hccl
