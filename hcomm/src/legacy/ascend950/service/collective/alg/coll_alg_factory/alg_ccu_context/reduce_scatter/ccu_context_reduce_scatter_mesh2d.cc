/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_scatter_mesh2d.h"
#include "ccu_instruction_reduce_scatter_mesh2d.h"

namespace Hccl {

constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;
constexpr int FST_AXIS_ID = 0;
constexpr int SEC_AXIS_ID = 1;
constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;

CcuContextReduceScatterMesh2D::CcuContextReduceScatterMesh2D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceScatterMesh2D* ctxArg = dynamic_cast<const CcuCtxArgReduceScatterMesh2D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMesh2D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    dimSize_ = ctxArg->dimSize_;
    axisId_ = ctxArg->axisId_;
    if (dimSize_[0] == 0) {
        THROW<InvalidParamsException>(StringFormat("Invalid dimSize[0][%u]", dimSize_[0]));
    }
    dimId_.emplace_back(rankId_ % dimSize_[0]);
    dimId_.emplace_back(rankId_ / dimSize_[0]);
    localId_ = dimId_[axisId_];
    localSize_ = dimSize_[axisId_];
    oppsiteSize_ = dimSize_[1 - axisId_];
    HCCL_INFO(
        "[CcuContextReduceScatterMesh2D] RankId[%u], DimSize0[%u], DimSize1[%u], localId[%u], localSize[%u], "
        "oppositeSize[%u]",
        rankId_, dimSize_[0], dimSize_[1], localId_, localSize_, oppsiteSize_);
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceScatterMesh2D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    reduceOp_ = ctxArg->op_.reduceOp;
    localAxisSignalName_ = "CcuContextReduceScatterMesh2DAxisSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextReduceScatterMesh2DAxisSync_" + std::to_string(1 - axisId_);
}

void CcuContextReduceScatterMesh2D::InitResources()
{
    step0BaseOffset_ = CreateVariable();
    step0AddOffset_ = CreateVariable();
    step1AddOffset_ = CreateVariable();
    xAxisGroupOpSize_ = CreateGroupOpSize();
    yAxisGroupOpSize_ = CreateGroupOpSize();
    localAxisSignal_ = CreateMaskSignal();
    anotherAxisSignal_ = CreateMaskSignal();
    yAxisOffset_ = CreateVariable();

    ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
    anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);

    output_.push_back(CreateVariable());
    uint32_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMesh2D transports is empty"));
    }
    for (uint64_t peerId = 0; peerId < localSize_; peerId++) {
        if (peerId == localId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextReduceScatterMesh2D] MyRank[%u], PeerId[%llu], TransportId[%u]", localId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr || transportIdx >= transports.size(),
                HCCL_ERROR("[CcuContextReduceScatterMesh2D] Algorithm transport ptr is null or transportIdx is out of "
                           "bounds"), );
            input_.push_back(
                CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=1的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    HCCL_INFO("[CcuContextReduceScatterMesh2D] InitResources finished");
}

void CcuContextReduceScatterMesh2D::PreSync()
{
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[localId_], INPUT_XN_ID, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[localId_], TOKEN_XN_ID, CKE_IDX_2, selfBit); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit); // index = 2，传递token信息
    HCCL_INFO("[CcuContextReduceScatterMesh2D] PreSync run finished");
}

void CcuContextReduceScatterMesh2D::PostSync(uint32_t signalIndex)
{
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        RemotePost(*t, signalIndex, selfBit);
    }
    GroupWait(*transportGroup, signalIndex, allBit);
    HCCL_INFO("[CcuContextReduceScatterMesh2D] PostSync run finished");
}

void CcuContextReduceScatterMesh2D::AxisSync(uint32_t signalIndex)
{
    const uint32_t DIE_NUM = 2;
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextReduceScatterMesh2D] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_INFO("[CcuContextReduceScatterMesh2D] AxisSync run finished");
    return;
}

void CcuContextReduceScatterMesh2D::LoadArgs()
{
    Load(input_[localId_]);
    Load(output_[0]);
    Load(token_[localId_]);
    Load(step0BaseOffset_);
    Load(step0AddOffset_);
    Load(step1AddOffset_);
    Load(yAxisOffset_);
    Load(xAxisGroupOpSize_);
    Load(yAxisGroupOpSize_);
    HCCL_INFO("[CcuContextReduceScatterMesh2D] LoadArgs run finished");
}

void CcuContextReduceScatterMesh2D::Step1Reduce()
{
    std::vector<CcuRep::Memory> src;
    std::vector<CcuRep::Memory> tempSrc;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        src.push_back(CreateMemory());
        tempSrc.push_back(CreateMemory());
    }
    CcuRep::Memory dst = CreateMemory();
    CcuRep::Memory tempDst = CreateMemory();
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint32_t localIdx = 0; localIdx < localSize_; localIdx++) {
        if (localIdx != localId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = localSize_ - 1;
        }
        src[curId].addr = input_[localIdx];
        src[curId].token = token_[localIdx];
    }
    dst.addr = input_[localId_];
    dst.token = token_[localId_];
    for (uint32_t oppositeIdx = 0; oppositeIdx < oppsiteSize_; oppositeIdx++) {
        for (uint32_t localIdx = 0; localIdx < localSize_; localIdx++) {
            if (oppositeIdx == 0) {
                src[localIdx].addr += step0BaseOffset_;
            } else {
                src[localIdx].addr += step0AddOffset_;
            }
        }
        if (oppositeIdx == 0) {
            dst.addr += step0BaseOffset_;
        } else {
            dst.addr += step0AddOffset_;
        }
        tempDst.addr = dst.addr;
        tempDst.token = dst.token;
        for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
            tempSrc[rankIdx].addr = src[rankIdx].addr;
            tempSrc[rankIdx].token = src[rankIdx].token;
        }
        if (axisId_ == 0) {
            GroupReduce(transports, tempDst, tempSrc, xAxisGroupOpSize_, dataType_, outputDataType_, reduceOp_);
        } else {
            GroupReduce(transports, tempDst, tempSrc, yAxisGroupOpSize_, dataType_, outputDataType_, reduceOp_);
        }
    }
    HCCL_INFO("[CcuContextReduceScatterMesh2D] Step1Reduce run finished");
}

void CcuContextReduceScatterMesh2D::Step2Reduce()
{
    std::vector<CcuRep::Memory> src;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        src.push_back(CreateMemory());
    }
    CcuRep::Memory dst = CreateMemory();
    dst.addr = output_[0];
    dst.token = token_[localId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint16_t localIdx = 0; localIdx < localSize_; localIdx++) {
        if (localIdx != localId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = localSize_ - 1;
        }
        src[curId].addr = input_[localIdx];
        src[curId].addr += step1AddOffset_;
        src[curId].token = token_[localIdx];
    }
    if (axisId_ == 0) {
        dst.addr += yAxisOffset_;
        GroupReduce(transports, dst, src, yAxisGroupOpSize_, dataType_, outputDataType_, reduceOp_);
    } else {
        GroupReduce(transports, dst, src, xAxisGroupOpSize_, dataType_, outputDataType_, reduceOp_);
    }
    HCCL_INFO("[CcuContextReduceScatterMesh2D] Step2Reduce run finished");
}

void CcuContextReduceScatterMesh2D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceScatterMesh2D] ReduceScatterMesh2D run");

    InitResources();
    LoadArgs();
    HCCL_INFO("[CcuContextReduceScatterMesh2D] Algorithm first step begins.");
    PreSync();

    Step1Reduce();
    PostSync(CKE_IDX_3);
    AxisSync(FST_AXIS_ID);

    HCCL_INFO("[CcuContextReduceScatterMesh2D] Algorithm second step begins.");
    PostSync(CKE_IDX_4);

    Step2Reduce();
    PostSync(CKE_IDX_0);
    AxisSync(SEC_AXIS_ID);

    HCCL_INFO("[CcuContextReduceScatterMesh2D] ReduceScatterMesh2D end");
    return;
}

std::vector<uint64_t> CcuContextReduceScatterMesh2D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceScatterMesh2D* taskArg = dynamic_cast<const CcuTaskArgReduceScatterMesh2D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMesh2D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t outputSize = taskArg->outputSize_;
    uint64_t offset = taskArg->offset_;
    uint64_t yAxisOffset = taskArg->xAxisSize_;
    uint64_t xAxisSize = taskArg->xAxisSize_;
    uint64_t yAxisSize = taskArg->yAxisSize_;

    // 计算不同die的数据
    uint64_t step0BaseOffset
        = axisId_ == 0 ? dimId_[0] * outputSize + offset : dimId_[1] * dimSize_[0] * outputSize + offset + xAxisSize;
    uint64_t step0AddOffset = axisId_ == 0 ? dimSize_[0] * outputSize : outputSize;
    uint64_t step1AddOffset = rankId_ * outputSize + offset + (axisId_ == 0 ? xAxisSize : 0);
    auto xAxisGoSize = CalGoSize(xAxisSize);
    auto yAxisGoSize = CalGoSize(yAxisSize);
    HCCL_INFO(
        "[CcuContextReduceScatterMesh2D] GeneArgs: inputAddr[%llu], outputAddr[%llu],"
        "step0BaseOffset[%llu], step0AddOffset[%llu], step1AddOffset[%llu]",
        inputAddr, outputAddr, step0BaseOffset, step0AddOffset, step1AddOffset);
    return {inputAddr,      outputAddr,     tokenInfo,      step0BaseOffset, step0AddOffset,
            step1AddOffset, yAxisOffset,    xAxisGoSize[0], xAxisGoSize[1],  xAxisGoSize[2],
            xAxisGoSize[3], yAxisGoSize[0], yAxisGoSize[1], yAxisGoSize[2],  yAxisGoSize[3]};
}
} // namespace Hccl
