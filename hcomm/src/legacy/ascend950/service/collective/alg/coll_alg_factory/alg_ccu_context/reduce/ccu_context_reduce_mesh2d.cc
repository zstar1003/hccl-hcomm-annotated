/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_mesh2d.h"
#include "ccu_instruction_reduce_mesh2d.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;

CcuContextReduceMesh2D::CcuContextReduceMesh2D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceMesh2D* ctxArg = dynamic_cast<const CcuCtxArgReduceMesh2D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceMesh2D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    dimSize_ = ctxArg->dimSize_;
    axisId_ = ctxArg->axisId_; // 要进行操作的是 行或列

    if (dimSize_.size() != 2 || axisId_ > 1 || dimSize_[0] == 0) { // 2D 拓扑校验
        THROW<NullPtrException>(StringFormat(
            "[CcuContextReduceMesh2D] dimSize[%u] or axisId[%u] or dimSize[0] [%u] is invalid", dimSize_.size(),
            axisId_, dimSize_[0]));
    }
    dimId_.emplace_back(rankId_ % dimSize_[0]);
    dimId_.emplace_back(rankId_ / dimSize_[0]);
    localId_ = dimId_[axisId_];     // 本rank所在的行/列
    localSize_ = dimSize_[axisId_]; // 本rank所在的行/列的总数

    HCCL_INFO(
        "[CcuContextReduceMesh2D] RankId[%u], DimSize0[%llu], DimSize1[%llu], localId[%llu], localSize[%llu]", rankId_,
        dimSize_[0], dimSize_[1], localId_, localSize_);

    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceMesh2D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    reduceOp_ = ctxArg->op_.reduceOp;
    rootId_ = ctxArg->rootId_;
    rootDimId_.emplace_back(rootId_ % dimSize_[0]); // root的x
    rootDimId_.emplace_back(rootId_ / dimSize_[0]); // root的y
    rootLocalId_ = rootDimId_[axisId_];             // 未用
    HCCL_INFO(
        "[CcuContextReduceMesh2D] init end, ctxArg->dimSize size[%zu] localSize_[%u]", ctxArg->dimSize_.size(),
        localSize_);

    localAxisSignalName_ = "CcuContextReduceMesh2DAxisSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextReduceMesh2DAxisSync_" + std::to_string(1 - axisId_);
}

void CcuContextReduceMesh2D::InitResources()
{
    localAxisSignal_ = CreateMaskSignal();
    anotherAxisSignal_ = CreateMaskSignal();
    ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
    anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);
    offset_ = CreateVariable();

    output_.push_back(CreateVariable());
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceMesh2D transports is empty"));
    }
    uint32_t transportIdx = 0;
    for (uint32_t peerId = 0; peerId < localSize_; peerId++) {
        if (peerId == localId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextReduceMesh2D] MyRank[%u], PeerId[%u], TransportId[%u]", localId_, peerId, transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextReduceMesh2D] Algorithm transport ptr is null"), );
            input_.push_back(
                CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=1的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }

    xAxisGroupOpSize_ = CreateGroupOpSize();
    yAxisGroupOpSize_ = CreateGroupOpSize();
    HCCL_INFO("[CcuContextReduceMesh2D] InitResources finished");
}

void CcuContextReduceMesh2D::PreSync() // 前同步
{
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[localId_], INPUT_XN_ID, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[localId_], TOKEN_XN_ID, CKE_IDX_2, selfBit); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit); // index = 2，传递token信息
    HCCL_INFO("[CcuContextReduceMesh2D] PreSync run finished");
}

void CcuContextReduceMesh2D::PostSync(uint32_t signalIndex)
{
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        RemotePost(*t, signalIndex, selfBit);
    }
    GroupWait(*transportGroup, signalIndex, allBit);
    HCCL_INFO("[CcuContextReduceMesh2D] PostSync run finished");
}

void CcuContextReduceMesh2D::AxisSync(uint32_t signalIndex) // 轴间同步
{
    const uint32_t DIE_NUM = 2;
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_INFO("[CcuContextReduceMesh2D] AxisSync run finished");
    return;
}

void CcuContextReduceMesh2D::LoadArgs()
{
    Load(input_[localId_]);
    Load(output_[0]);
    Load(token_[localId_]);
    Load(offset_);
    Load(xAxisGroupOpSize_);
    Load(yAxisGroupOpSize_);
    HCCL_INFO("[CcuContextReduceMesh2D] LoadArgs run finished");
}

void CcuContextReduceMesh2D::Step1Reduce()
{
    // 只有与 root 同列的 rank 的 die0 进行第一步 reduce
    if (dimId_[0] != rootDimId_[0] || axisId_ != 0) {
        HCCL_INFO("[CcuContextReduceMesh2D] RankId [%u], axisId [%u], skip Step1Reduce", rankId_, axisId_);
        return;
    }
    HCCL_INFO("[CcuContextReduceMesh2D] RankId [%u], axisId [%u], run Step1Reduce", rankId_, axisId_);

    CcuRep::Memory dst = CreateMemory();
    dst.addr = input_[localId_]; // 第一步reduce都是从input reduce到input
    dst.token = token_[localId_];

    std::vector<CcuRep::Memory> src;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        src.push_back(CreateMemory());
    }
    uint32_t curId = 0;
    uint32_t dstId = 0;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        if (rankIdx != localId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = localSize_ - 1;
        }
        src[curId].addr = input_[rankIdx];
        src[curId].token = token_[rankIdx];
    }

    GroupReduce(transports, dst, src, xAxisGroupOpSize_, dataType_, outputDataType_, reduceOp_);
}

void CcuContextReduceMesh2D::Step2ReduceForRoot()
{
    // 只有与 root 的 die0 进行第一步 reduce
    if (rankId_ != rootId_ || axisId_ != 1) {
        HCCL_INFO("[CcuContextReduceMesh2D] RankId [%u], axisId [%u], skip Step2Reduce", rankId_, axisId_);
        return;
    }
    HCCL_INFO("[CcuContextReduceMesh2D] RankId [%u], axisId [%u], run Step2Reduce", rankId_, axisId_);

    std::vector<CcuRep::Memory> src;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        src.push_back(CreateMemory());
    }
    CcuRep::Memory dst = CreateMemory();
    dst.addr = output_[0]; // 第二步reduce是从input reduce到root的output
    dst.token = token_[localId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        if (rankIdx != localId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = localSize_ - 1; // 最后一个位置放root的input
        }
        src[curId].addr = input_[rankIdx];
        src[curId].token = token_[rankIdx];
    }
    GroupReduce(transports, dst, src, xAxisGroupOpSize_, dataType_, outputDataType_, reduceOp_);
}

void CcuContextReduceMesh2D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceMesh2D] ReduceMesh2D run");
    InitResources();
    LoadArgs();
    HCCL_INFO("[CcuContextReduceMesh2D] Algorithm first step begins.");
    PreSync(); // 前同步
    Step1Reduce();
    AxisSync(0);
    PostSync(CKE_IDX_3);
    AxisSync(1);
    Step2ReduceForRoot();
    AxisSync(0);
    PostSync(CKE_IDX_0);
    AxisSync(1);
}

std::vector<uint64_t> CcuContextReduceMesh2D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceMesh2D* taskArg = dynamic_cast<const CcuTaskArgReduceMesh2D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuTaskArgReduceMesh2D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t offset = taskArg->offset_;
    uint64_t xAxisSize = taskArg->xAxisSize_;
    uint64_t yAxisSize = taskArg->yAxisSize_;
    auto xAxisGoSize = CalGoSize(xAxisSize);
    auto yAxisGoSize = CalGoSize(yAxisSize);

    HCCL_INFO(
        "[CcuContextReduceMesh2D] ReduceMesh2D inputAddr [%llu] outputAddr [%llu] offset [%llu], "
        "xAxisSize [%llu] yAxisSize [%llu]",
        inputAddr, outputAddr, offset, xAxisSize, yAxisSize);

    return {inputAddr,      outputAddr,     tokenInfo,      offset,         xAxisGoSize[0], xAxisGoSize[1],
            xAxisGoSize[2], xAxisGoSize[3], yAxisGoSize[0], yAxisGoSize[1], yAxisGoSize[2], yAxisGoSize[3]};
}
} // namespace Hccl
