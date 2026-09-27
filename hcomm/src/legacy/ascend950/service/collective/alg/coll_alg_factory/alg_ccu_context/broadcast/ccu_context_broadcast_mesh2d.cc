/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_broadcast_mesh2d.h"
#include "ccu_instruction_broadcast_mesh2d.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;
constexpr int X_AXIS_ID = 0;
constexpr int Y_AXIS_ID = 1;
constexpr int FST_AXIS_ID = 0;
constexpr int SEC_AXIS_ID = 1;

CcuContextBroadcastMesh2D::CcuContextBroadcastMesh2D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgBroadcastMesh2D* ctxArg = dynamic_cast<const CcuCtxArgBroadcastMesh2D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextBroadcastMesh2D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    dimSize_ = ctxArg->dimSize_;
    axisId_ = ctxArg->axisId_;
    if (dimSize_.size() != 2 || axisId_ > 1 || dimSize_[0] == 0) { // 2D 拓扑校验
        THROW<NullPtrException>(StringFormat(
            "[CcuContextBroadcastMesh2D] dimSize[%zu] or axisId[%u] or dimSize[0] [%u] is invalid.", dimSize_.size(),
            axisId_, dimSize_[0]));
    }
    dimId_.emplace_back(rankId_ % dimSize_[0]);
    dimId_.emplace_back(rankId_ / dimSize_[0]);
    localId_ = dimId_[axisId_];
    localSize_ = dimSize_[axisId_];
    HCCL_INFO(
        "[CcuContextBroadcastMesh2D] RankId[%u], DimSize0[%llu], DimSize1[%llu], localId[%llu], localSize[%llu].",
        rankId_, dimSize_[0], dimSize_[1], localId_, localSize_);
    rootId_ = ctxArg->op_.root;
    dataType_ = ctxArg->op_.dataType;
    rootDimId_.emplace_back(rootId_ % dimSize_[0]);
    rootDimId_.emplace_back(rootId_ / dimSize_[0]);
    rootLocalId_ = rootDimId_[axisId_];

    localAxisSignalName_ = "CcuContextBroadcastMesh2DAxisSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextBroadcastMesh2DAxisSync_" + std::to_string(1 - axisId_);
}

void CcuContextBroadcastMesh2D::InitResources()
{
    localAxisSignal_ = CreateMaskSignal();
    anotherAxisSignal_ = CreateMaskSignal();
    yAxisOffset_ = CreateVariable();

    xAxisGroupOpSize_ = CreateGroupOpSize();
    yAxisGroupOpSize_ = CreateGroupOpSize();

    ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
    anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);

    uint32_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextBroadcastMesh2D transports is empty"));
    }
    for (uint32_t peerId = 0; peerId < localSize_; peerId++) {
        if (peerId == localId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextBroadcastMesh2D] MyRank[%u], PeerId[%u], TransportId[%u]", localId_, peerId, transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr || transportIdx >= transports.size(),
                HCCL_ERROR(
                    "[CcuContextBroadcastMesh2D] Algorithm transport ptr is null or transportIdx is out of bounds"), );
            input_.push_back(
                CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=1的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    HCCL_INFO("[CcuContextBroadcastMesh2D] InitResources finished");
}

void CcuContextBroadcastMesh2D::PreSync()
{
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[localId_], INPUT_XN_ID, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[localId_], TOKEN_XN_ID, CKE_IDX_2, selfBit); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit); // index = 2，传递token信息
    HCCL_INFO("[CcuContextBroadcastMesh2D] PreSync run finished");
}

void CcuContextBroadcastMesh2D::PostSync(uint32_t signalIndex)
{
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        RemotePost(*t, signalIndex, selfBit);
    }
    GroupWait(*transportGroup, signalIndex, allBit);
    HCCL_INFO("[CcuContextBroadcastMesh2D] PostSync run finished");
}

void CcuContextBroadcastMesh2D::AxisSync(uint32_t signalIndex)
{
    const uint32_t DIE_NUM = 2;
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextBroadcastMesh2D] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_INFO("[CcuContextBroadcastMesh2D] AxisSync run finished");
    return;
}

void CcuContextBroadcastMesh2D::LoadArgs()
{
    Load(input_[localId_]);
    Load(yAxisOffset_);
    Load(token_[localId_]);
    Load(xAxisGroupOpSize_);
    Load(yAxisGroupOpSize_);
    HCCL_INFO("[CcuContextBroadcastMesh2D] LoadArgs run finished");
}

void CcuContextBroadcastMesh2D::Step1BroadcastForRoot()
{
    std::vector<CcuRep::Memory> dst;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        dst.push_back(CreateMemory());
    }
    CcuRep::Memory src = CreateMemory();
    src.token = token_[localId_];
    src.addr = input_[localId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        if (rankIdx != localId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = localSize_ - 1;
        }
        dst[curId].addr = input_[rankIdx];
        dst[curId].token = token_[rankIdx];
        if (axisId_ == Y_AXIS_ID) {
            dst[curId].addr += yAxisOffset_;
        }
    }
    if (axisId_ == X_AXIS_ID) {
        GroupBroadcast(transports, dst, src, xAxisGroupOpSize_);
    } else {
        src.addr += yAxisOffset_;
        GroupBroadcast(transports, dst, src, yAxisGroupOpSize_);
    }
    HCCL_INFO("[CcuContextBroadcastMesh2D] Step1Broadcast run finished");
}

void CcuContextBroadcastMesh2D::Step2BroadcastForRoot()
{
    std::vector<CcuRep::Memory> dst;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        dst.push_back(CreateMemory());
    }
    CcuRep::Memory src = CreateMemory();
    src.addr = input_[localId_];
    src.token = token_[localId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        if (rankIdx != localId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = localSize_ - 1;
        }
        dst[curId].addr = input_[rankIdx];
        dst[curId].token = token_[rankIdx];
        if (axisId_ == X_AXIS_ID) {
            dst[curId].addr += yAxisOffset_;
        }
    }
    if (axisId_ == X_AXIS_ID) {
        src.addr += yAxisOffset_;
        GroupBroadcast(transports, dst, src, yAxisGroupOpSize_);
    } else {
        GroupBroadcast(transports, dst, src, xAxisGroupOpSize_);
    }
    HCCL_INFO("[CcuContextBroadcastMesh2D] Step2Broadcast run finished");
}

void CcuContextBroadcastMesh2D::Step2Broadcast()
{
    std::vector<CcuRep::Memory> dst;
    CcuRep::Memory src = CreateMemory();
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        dst.push_back(CreateMemory());
    }
    src.addr = input_[localId_];
    src.token = token_[localId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    // 当前rank和root在同一行, 只有y轴需要再执行一次bcast; 当前rank和root在同列, 只有x轴需要再执行一次bcast
    if (dimId_[1] == rootDimId_[1] && axisId_ == Y_AXIS_ID) {
        HCCL_INFO("[CcuContextBroadcastMesh2D] rankId[%u] is on the same row as rootId[%u]", rankId_, rootId_);
        for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
            if (rankIdx != localId_) {
                curId = dstId;
                dstId++;
            } else {
                curId = localSize_ - 1;
            }
            dst[curId].addr = input_[rankIdx];
            dst[curId].token = token_[rankIdx];
        }
        GroupBroadcast(transports, dst, src, xAxisGroupOpSize_);
    } else if (dimId_[0] == rootDimId_[0] && axisId_ == X_AXIS_ID) {
        HCCL_INFO("[CcuContextBroadcastMesh2D] rankId[%u] is on the same column as rootId[%u]", rankId_, rootId_);
        for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
            if (rankIdx != localId_) {
                curId = dstId;
                dstId++;
            } else {
                curId = localSize_ - 1;
            }
            dst[curId].addr = input_[rankIdx];
            dst[curId].addr += yAxisOffset_;
            dst[curId].token = token_[rankIdx];
        }
        src.addr += yAxisOffset_;
        GroupBroadcast(transports, dst, src, yAxisGroupOpSize_);
    } else {
        HCCL_INFO("[CcuContextBroadcastMesh2D] rankId[%u] do nothing", rankId_);
    }
    HCCL_INFO("[CcuContextBroadcastMesh2D] Step2Broadcast run finished");
}

void CcuContextBroadcastMesh2D::Algorithm()
{
    HCCL_INFO("[CcuContextBroadcastMesh2D] BroadcastMesh2D run");
    InitResources();
    LoadArgs();

    PreSync();

    if (rankId_ == rootId_) {
        HCCL_INFO("[CcuContextBroadcastMesh2D] Algorithm root run begins.");
        Step1BroadcastForRoot();
        PostSync(CKE_IDX_3);
        AxisSync(FST_AXIS_ID);
        HCCL_INFO("[CcuContextBroadcastMesh2D] Algorithm second step begins.");
        PostSync(CKE_IDX_4);
        Step2BroadcastForRoot();
    } else if (dimId_[0] == rootDimId_[0] || dimId_[1] == rootDimId_[1]) { // 当前rank和root在同一行或同一列
        if (dimId_[axisId_] != rootDimId_[axisId_]) {
            PostSync(CKE_IDX_3);
        }
        AxisSync(FST_AXIS_ID);
        if (dimId_[axisId_] != rootDimId_[axisId_]) {
            PostSync(CKE_IDX_4);
        }
        Step2Broadcast();
    } else {
        AxisSync(FST_AXIS_ID);
    }
    PostSync(CKE_IDX_0);
    AxisSync(SEC_AXIS_ID);

    HCCL_INFO("[CcuContextBroadcastMesh2D] BroadcastMesh2D end");
    return;
}

std::vector<uint64_t> CcuContextBroadcastMesh2D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgBroadcastMesh2D* taskArg = dynamic_cast<const CcuTaskArgBroadcastMesh2D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextBroadcastMesh2D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t yAxisOffset = taskArg->xAxisSize_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t xAxisSize = taskArg->xAxisSize_;
    uint64_t yAxisSize = taskArg->yAxisSize_;
    auto xAxisGoSize = CalGoSize(xAxisSize);
    auto yAxisGoSize = CalGoSize(yAxisSize);
    return {inputAddr,      yAxisOffset,    tokenInfo,      xAxisGoSize[0], xAxisGoSize[1], xAxisGoSize[2],
            xAxisGoSize[3], yAxisGoSize[0], yAxisGoSize[1], yAxisGoSize[2], yAxisGoSize[3]};
}
} // namespace Hccl
