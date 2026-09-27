/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "one_sided_component_lite.h"
#include <iostream>
#include <string>
#include <map>
#include "log.h"

namespace Hccl {
HcclResult OneSidedComponentLite::Orchestrate(const HcclAicpuOpLite& op, InsQuePtr queue)
{
    HCCL_INFO("[%s] Orchestrate Mode: Instruction.", __func__);
    bool isOneSidedComm = (op.algOperator.opType == OpType::BATCHPUT) || (op.algOperator.opType == OpType::BATCHGET);

    if (!isOneSidedComm) {
        HCCL_ERROR("[%s] OneSidedComm not support opType[%s].", __func__, op.algOperator.opType.Describe().c_str());
        return HCCL_E_PARA;
    }

    vector<RmaBufSliceLite> usrInSlice;
    vector<RmtRmaBufSliceLite> usrOutSlice;

    for (uint32_t i = 0; i < op.batchPutGetDescNum; i++) {
        HcclAicpuLocBufLite* localBuf = static_cast<HcclAicpuLocBufLite*>(op.batchPutGetLocalAddr) + i;
        usrInSlice.push_back(RmaBufSliceLite(localBuf->addr, localBuf->size, localBuf->tokenValue, localBuf->tokenId));

        HcclAicpuLocBufLite* rmtBuf = static_cast<HcclAicpuLocBufLite*>(op.batchPutGetRemoteAddr) + i;
        usrOutSlice.push_back(
            RmtRmaBufSliceLite(rmtBuf->addr, rmtBuf->size, 0, rmtBuf->tokenId, rmtBuf->tokenValue, UINT32_MAX));
    }

    RankId rmtRankId = op.sendRecvRemoteRank;
    vector<LinkData> link = linkMgr_->GetLinks(0, rmtRankId);
    HCCL_INFO("[%s] Orchestrate Mode: Instruction %d.", __func__, rmtRankId);
    if (op.algOperator.opType == OpType::BATCHGET) {
        std::unique_ptr<Instruction> ins
            = std::make_unique<InsBatchOneSidedRead>(rmtRankId, link[0], usrInSlice, usrOutSlice);
        queue->Append(std::move(ins));
    } else {
        std::unique_ptr<Instruction> ins
            = std::make_unique<InsBatchOneSidedWrite>(rmtRankId, link[0], usrInSlice, usrOutSlice);
        queue->Append(std::move(ins));
    }

    HCCL_INFO("[%s] finish orchestrate opType[%s].", __func__, op.algOperator.opType.Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
