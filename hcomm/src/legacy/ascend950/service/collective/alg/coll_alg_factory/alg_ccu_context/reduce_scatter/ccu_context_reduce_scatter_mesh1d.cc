/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_scatter_mesh1d.h"
#include "ccu_instruction_reduce_scatter_mesh1d.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;

CcuContextReduceScatterMesh1D::CcuContextReduceScatterMesh1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceScatterMesh1D* ctxArg = dynamic_cast<const CcuCtxArgReduceScatterMesh1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMesh1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType = ctxArg->op_.dataType;
    outputDataType = ctxArg->op_.outputDataType;
    if (outputDataType == DataType::INVALID) {
        outputDataType = dataType;
        HCCL_INFO(
            "[CcuContextReduceScatterMesh1D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType.Describe().c_str());
    }
    reduceOp_ = ctxArg->op_.reduceOp;
}

void CcuContextReduceScatterMesh1D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceScatterMesh1D] ReduceScatterMesh1D run");
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));
    output_.push_back(CreateVariable());
    uint16_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMesh1D transports is empty"));
    }
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextReduceScatterMesh1D] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextReduceScatterMesh1D] Algorithm transport ptr is null"), );
            input_.push_back(
                CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=1的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    offset_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();

    Load(input_[rankId_]);
    Load(output_[0]);
    Load(token_[rankId_]);
    Load(offset_);
    Load(groupOpSize_);
    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_2, selfBit); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit); // index = 2，传递token信息

    std::vector<CcuRep::Memory> src;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        src.push_back(CreateMemory());
    }
    CcuRep::Memory dst = CreateMemory();
    dst.addr = output_[0];
    dst.token = token_[rankId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        src[curId].addr = input_[rankIdx];
        src[curId].addr += offset_;
        src[curId].token = token_[rankIdx];
    }

    GroupReduce(transports, dst, src, groupOpSize_, dataType, outputDataType, reduceOp_);

    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit);
    HCCL_INFO("[CcuContextReduceScatterMesh1D] ReduceScatterMesh1D end");
    return;
}

std::vector<uint64_t> CcuContextReduceScatterMesh1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceScatterMesh1D* taskArg = dynamic_cast<const CcuTaskArgReduceScatterMesh1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMesh1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t offset = taskArg->offset_;
    uint64_t sliceSize = taskArg->sliceSize_;
    auto goSize = CalGoSize(sliceSize);
    return {inputAddr, outputAddr, tokenInfo, offset, goSize[0], goSize[1], goSize[2], goSize[3]};
}
} // namespace Hccl
