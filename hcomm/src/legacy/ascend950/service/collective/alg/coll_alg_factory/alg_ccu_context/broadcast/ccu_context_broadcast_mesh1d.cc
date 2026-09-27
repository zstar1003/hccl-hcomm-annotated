/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_broadcast_mesh1d.h"
#include "ccu_instruction_broadcast_mesh1d.h"

namespace Hccl {

constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;

CcuContextBroadcastMesh1D::CcuContextBroadcastMesh1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgBroadcastMesh1D* ctxArg = dynamic_cast<const CcuCtxArgBroadcastMesh1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextBroadcastMesh1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rootId_ = ctxArg->rootId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextBroadcastMesh1D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    HCCL_INFO(
        "[CcuContextBroadcastMesh1D] init end, ctxArg->dimSize size[%zu] rankSize[%llu]", ctxArg->dimSize_.size(),
        rankSize_);
}

void CcuContextBroadcastMesh1D::CreateAllVariables()
{
    HCCL_INFO("[CcuContextBroadcastMesh1D]transports.size: [%zu]", transports.size());
    uint16_t transportIdx = 0;
    input_ = CreateVariable();
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextBroadcastMesh1D transports is empty"));
    }
    HCCL_DEBUG("[CcuContextBroadcastMesh1D]transports.size: [%zu]", transports.size());
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextBroadcastMesh1D] MyRank[%u], PeerId[%llu], TransportId[%hu]", rankId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr || transportIdx >= transports.size(),
                HCCL_ERROR(
                    "[CcuContextBroadcastMesh1D] Algorithm transport ptr is null or transportIdx is out of bounds"), );
            output_.push_back(
                CreateVariable((*transports[transportIdx]), CKE_IDX_1)); // 获取transport中id=2的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), CKE_IDX_2));
            transportIdx++;
        }
    }
    offset_ = CreateVariable();
    slicesize_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    return;
}

void CcuContextBroadcastMesh1D::LoadAndExchangeData()
{
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));
    HCCL_DEBUG("[CcuContextBroadcastMesh1D] BroadcastMesh1D LoadAndExchangeData: rankId[%u]", rankId_);
    Load(input_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(offset_);
    Load(slicesize_);
    Load(groupOpSize_);
    for (auto t : transports) {
        WriteVariableWithSignal(*t, output_[rankId_], CKE_IDX_1, CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*t, token_[rankId_], CKE_IDX_2, CKE_IDX_2, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit);
    GroupWait(*transportGroup, CKE_IDX_2, allBit);
    return;
}

void CcuContextBroadcastMesh1D::BroadcastFromRootToAll()
{
    std::vector<CcuRep::Memory> dst;
    for (uint32_t index = 0; index < rankSize_; index++) {
        dst.emplace_back(CreateMemory());
    }
    CcuRep::Memory src = CreateMemory();
    src.addr = input_;
    src.token = token_[rankId_];
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rootId_) {
            dst[curId].addr = output_[rankIdx];
            dst[curId].token = token_[rankIdx];
            curId++;
        }
    }
    dst[rankSize_ - 1].addr = output_[rankId_];
    dst[rankSize_ - 1].token = token_[rankId_];
    GroupBroadcast(transports, dst, src, groupOpSize_);
    HCCL_INFO("[CcuContextBroadcastMesh1D] BroadcastMesh1D GroupBroadcast end");
    return;
}

void CcuContextBroadcastMesh1D::Algorithm()
{
    HCCL_INFO("[CcuContextBroadcastMesh1D] BroadcastMesh1D run");
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    CreateAllVariables();

    LoadAndExchangeData();

    if (rankId_ == rootId_) {
        BroadcastFromRootToAll();
    }

    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit);
    }

    GroupWait(*transportGroup, CKE_IDX_0, allBit);
    HCCL_INFO("[CcuContextBroadcastMesh1D] BroadcastMesh1D Broadcast groupwait end");
    return;
}

std::vector<uint64_t> CcuContextBroadcastMesh1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgBroadcastMesh1D* taskArg = dynamic_cast<const CcuTaskArgBroadcastMesh1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextBroadcastMesh1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t offset = taskArg->offset_;
    uint64_t sliceSize = taskArg->sliceSize_;
    auto goSize = CalGoSize(sliceSize);

    return {inputAddr, outputAddr, tokenInfo, offset, sliceSize, goSize[0], goSize[1], goSize[2], goSize[3]};
}
} // namespace Hccl
