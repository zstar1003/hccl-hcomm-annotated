/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <map>

#include "log.h"
#include "coll_alg_component_lite.h"

namespace Hccl {

void CollAlgComponentLite::EnableDetour(bool enableDetour)
{
    enableDetour_ = enableDetour;
    return;
}

void CollAlgComponentLite::EnableDataAllign(bool enableAllign)
{
    enableAllign_ = enableAllign;
    return;
}

void CollAlgComponentLite::SetAllignSize(u64 allignSize)
{
    allignSize_ = allignSize;
    return;
}

void CollAlgComponentLite::SetDmaMode(const DmaMode dmaMode)
{
    dmaMode_ = dmaMode;
    return;
}

HcclResult CollAlgComponentLite::ParsePackedData(std::vector<char> packedData)
{
    BinaryStream binaryStream(packedData);
    DmaMode dmaMode;
    binaryStream >> dmaMode;
    SetDmaMode(dmaMode);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollAlgComponentLite::Orchestrate(
    const CollAlgOperator& op, const std::string& algName, const AlgTopoInfo& algTopoInfo, PrimQuePtr queue)
{
    HCCL_DEBUG("[CollAlgComponentLite] Orchestrate Mode: Primitive.");
    if (rankSize_ == 1) {
        u64 dataSize = op.dataCount * DataTypeSizeGet(op.dataType);
        DataSlice usrInSlice = DataSlice(BufferType::INPUT, 0, dataSize);
        DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, 0, dataSize);
        std::unique_ptr<Primitive> primLocalCopy = std::make_unique<PrimLocalCopy>(usrInSlice, usrOutSlice);
        if (primLocalCopy == nullptr) {
            HCCL_ERROR("[CollAlgComponentLite] primLocalCopy is nullptr");
            return HcclResult::HCCL_E_PARA;
        }
        queue->Append(std::move(primLocalCopy));

        HCCL_DEBUG("[CollAlgComponentLite] rankSize = 1.");
        return HcclResult::HCCL_SUCCESS;
    }

    std::shared_ptr<CollAlgBase> primGenFunc = CollAlgRegistry::Global()->GetAlgImpl(op.opType, algName);
    CHK_PRT_RET(
        primGenFunc == nullptr, HCCL_ERROR("[CollAlgComponentLite] can not find collAlgName: [%s]", algName.c_str()),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        enableDetour_
            && ((algName != "AllGatherMesh") && (algName != "ReduceScatterMesh") && (algName != "AllReduceMesh")),
        HCCL_ERROR("[CollAlgComponentLite] Current algorithm can not support detouring, please check!"),
        HcclResult::HCCL_E_NOT_SUPPORT);

    primGenFunc->SetMyRank(myRank_);
    primGenFunc->SetRankSize(rankSize_);
    primGenFunc->SetDevType(devType_);
    primGenFunc->EnableDataAllign(enableAllign_);
    primGenFunc->SetAllignSize(allignSize_);
    primGenFunc->EnableDetour(enableDetour_);
    primGenFunc->SetDmaMode(dmaMode_);

    CollAlgParams params;
    params.maxTmpMemSize = scratchBufferSize_;
    params.opMode = OpMode::OPBASE;
    primGenFunc->GenPrimQuesAIC(algTopoInfo, op, params, linkMgr_, queue);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollAlgComponentLite::Orchestrate(
    const CollAlgOperator& op, const std::string& algName, const AlgTopoInfo& algTopoInfo, InsQuePtr queue)
{
    HCCL_DEBUG("[CollAlgComponentLite] Orchestrate Mode: Instruction.");
    bool isAlltoAll
        = (op.opType == OpType::ALLTOALL) || (op.opType == OpType::ALLTOALLV) || (op.opType == OpType::ALLTOALLVC);
    if ((rankSize_ == 1) && (!isAlltoAll)) {
        u64 dataSize = op.dataCount * DataTypeSizeGet(op.dataType);
        DataSlice usrInSlice = DataSlice(BufferType::INPUT, 0, dataSize);
        DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, 0, dataSize);
        std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(usrInSlice, usrOutSlice);
        if (insLocalCopy == nullptr) {
            HCCL_ERROR("[CollAlgComponentLite] insLocalCopy is nullptr");
            return HcclResult::HCCL_E_PARA;
        }
        queue->Append(std::move(insLocalCopy));

        HCCL_DEBUG("[CollAlgComponentLite] rankSize = 1.");
        HCCL_DEBUG("[CollAlgComponentLite] finish CollAlgComponentLite::Orchestrate.");
        return HcclResult::HCCL_SUCCESS;
    }

    std::shared_ptr<InsCollAlgBase> insGenFunc = InsCollAlgRegistry::Global()->GetAlgImpl(op.opType, algName);
    CHK_PRT_RET(
        insGenFunc == nullptr, HCCL_ERROR("[CollAlgComponentLite] can not find insCollAlgName: [%s]", algName.c_str()),
        HcclResult::HCCL_E_PARA);

    insGenFunc->SetMyRank(myRank_);
    insGenFunc->SetSendRecvRemoteRank(op.sendRecvRemoteRank);
    insGenFunc->SetRankSize(rankSize_);
    insGenFunc->SetDevType(devType_);
    insGenFunc->EnableDataAllign(enableAllign_);
    insGenFunc->SetAllignSize(allignSize_);
    insGenFunc->EnableDetour(enableDetour_);
    insGenFunc->SetDmaMode(dmaMode_);
    insGenFunc->SetRmaDataBufferMgr(rmaDataBufferMgr_);

    CollAlgParams params;
    params.maxTmpMemSize = scratchBufferSize_;
    params.opMode = op.opMode;
    insGenFunc->Orchestrate(algTopoInfo, op, params, linkMgr_, queue);

    HCCL_DEBUG("finish CollAlgComponentLite Orchestrate");
    return HcclResult::HCCL_SUCCESS;
}

void CollAlgComponentLite::UpdateScratchBufferSize(u64 bufferSize) { scratchBufferSize_ = bufferSize; }

} // namespace Hccl
