/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_batch_write_executor.h"

namespace hccl {
HcclResult CollBatchWriteExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    (void)param;
    (void)algRes;
    return HCCL_SUCCESS;
}

HcclResult CollBatchWriteExecutor::CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    std::vector<LevelNSubCommTransport> opTransport{
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))};
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    commCombinePara.meshSinglePlane = true;
    CHK_RET(topoMatcher_->CalcCommPlaneInfo(
        param.tag, commCombinePara, opTransport[COMM_COMBINE_ORDER], TransportMemType::CCL_INPUT,
        TransportMemType::CCL_OUTPUT));

    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto& transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = false;
        }
    }
    resourceRequest.opTransport = opTransport;
    resourceRequest.streamNum = param.BatchWriteDataDes.queueNum;
    HCCL_INFO(
        "[Sdma-BatchWrite]Calc resource request for tag %s, stream number %u.", param.tag.c_str(),
        resourceRequest.streamNum);
    return HCCL_SUCCESS;
}

REGISTER_EXEC(BATCH_WRITE_ALG_NAME, BatchWrite, CollBatchWriteExecutor);
} // namespace hccl
