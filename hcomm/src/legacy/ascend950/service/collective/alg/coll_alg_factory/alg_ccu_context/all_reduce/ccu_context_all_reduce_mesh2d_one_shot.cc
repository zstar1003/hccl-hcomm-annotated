/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_reduce_mesh2d_one_shot.h"
#include "ccu_instruction_all_reduce_mesh2d_one_shot.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;
constexpr uint32_t AXIS_NUM = 2;

CcuContextAllReduceMesh2DOneShot::CcuContextAllReduceMesh2DOneShot(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllReduceMesh2DOneShot* ctxArg = dynamic_cast<const CcuCtxArgAllReduceMesh2DOneShot*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh2DOneShot::ctxArg ptr is null"));
    }
    dimSize_ = ctxArg->dimSize_;
    axisId_ = ctxArg->axisId_;
    rankId_ = ctxArg->rankId_;
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    reduceOp_ = ctxArg->op_.reduceOp;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextAllReduceMesh2DOneShot] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }

    HCCL_INFO(
        "[CcuContextAllReduceMesh2DOneShot] Init, CtxArgs are dimSize0[%llu], dimSize1[%llu], axisId[%u], "
        "rankId[%llu], dataType[%s], outputDataType[%s], reduceOp[%s]",
        dimSize_[0], dimSize_[1], axisId_, rankId_, dataType_.Describe().c_str(), outputDataType_.Describe().c_str(),
        reduceOp_.Describe().c_str());
    uint32_t max_dimSize = 2;
    if (dimSize_.size() != max_dimSize or axisId_ > 1) {
        THROW<NullPtrException>(StringFormat(
            "[CcuContextAllReduceMesh2DOneShot] dimSize[%u] or axisId[%u] is invalid", dimSize_.size(), axisId_));
    }
    CHK_PRT_THROW(
        dimSize_[0] == 0 || dimSize_[1] == 0,
        HCCL_ERROR(
            "[CcuContextAllReduceMesh2DOneShot] dimSize0[%llu] or dimSize1[%llu] is zero", dimSize_[0], dimSize_[1]),
        InvalidParamsException, "dimSize[0] or dimSize[1] is invalid");

    myRankIdxInAxis_.push_back(rankId_ % dimSize_[0]); // 本 rank 在第 0 维上的 index
    myRankIdxInAxis_.push_back(rankId_ / dimSize_[0]); // 本 rank 在第 1 维上的 index

    myRankIdxInCurrentAxis_ = myRankIdxInAxis_[axisId_];
    currentAxisRankSize_ = dimSize_[axisId_];

    // 同步信号初始化
    currAxisSignalName_ = "CcuContextAllReduceMesh2DOneShotAxisSync_" + std::to_string(axisId_);
    otherAxisSignalName_ = "CcuContextAllReduceMesh2DOneShotAxisSync_" + std::to_string(1 - axisId_);
    currAxisSignal_ = CreateMaskSignal();
    ExportMaskSignal(currAxisSignal_, currAxisSignalName_);
    otherAxisSignal_ = ImportMaskSignal(otherAxisSignalName_);

    HCCL_INFO(
        "[CcuContextAllReduceMesh2DOneShot] Init, myRankIdx0[%llu], myRankIdx1[%llu], "
        "myRankIdxInCurrentAxis[%llu], currentAxisRankSize[%llu]",
        myRankIdxInAxis_[0], myRankIdxInAxis_[1], myRankIdxInCurrentAxis_, currentAxisRankSize_);
}

void CcuContextAllReduceMesh2DOneShot::Algorithm()
{
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] AllReduceMesh2DOneShot run");
    uint16_t selfBit = 1 << myRankIdxInCurrentAxis_;
    uint16_t allBit = ((1 << currentAxisRankSize_) - 1) & (~(1 << myRankIdxInCurrentAxis_));

    InitVariables();

    LoadArgs();

    // 前同步
    for (auto t : transports) {
        WriteVariableWithSignal(*t, inputAddr_[myRankIdxInCurrentAxis_], INPUT_XN_ID, CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*t, scratchAddr_[myRankIdxInCurrentAxis_], SCRATCH_XN_ID, CKE_IDX_2, selfBit);
        WriteVariableWithSignal(*t, token_[myRankIdxInCurrentAxis_], TOKEN_XN_ID, CKE_IDX_3, selfBit);
    }

    GroupWait(*transportGroup, CKE_IDX_1, allBit);
    GroupWait(*transportGroup, CKE_IDX_2, allBit);
    GroupWait(*transportGroup, CKE_IDX_3, allBit);

    // OneShot Step1
    CcuRep::Variable& Step1Offset = (axisId_ == 0) ? xSliceOffset_ : ySliceOffset_;
    GroupOpSize& Step1GoSize = (axisId_ == 0) ? xGoSize_ : yGoSize_;

    DoGroupReduce(inputAddr_, scratchAddr_[myRankIdxInCurrentAxis_], Step1Offset, Step1GoSize);

    DoAxisSync(0);
    DoGroupSync(CKE_IDX_4, selfBit, allBit);
    DoAxisSync(1);

    // OneShot Step2
    CcuRep::Variable& Step2Offset = (axisId_ == 0) ? ySliceOffset_ : xSliceOffset_;
    GroupOpSize& Step2GoSize = (axisId_ == 0) ? yGoSize_ : xGoSize_;

    DoGroupReduce(scratchAddr_, outputAddr_[myRankIdxInCurrentAxis_], Step2Offset, Step2GoSize);

    DoAxisSync(0);
    DoGroupSync(CKE_IDX_0, selfBit, allBit);
    DoAxisSync(1);
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] AllReduceMesh2DOneShot end");
    return;
}

void CcuContextAllReduceMesh2DOneShot::DoGroupSync(int ckeIdx, uint16_t selfBit, uint16_t allBit)
{
    HCCL_INFO(
        "[CcuContextAllReduceMesh2DOneShot] DoGroupSync Starts, ckeIdx[%d], selfBit[%u], allBit[%u]", ckeIdx, selfBit,
        allBit);
    for (auto t : transports) {
        RemotePost(*t, ckeIdx, selfBit);
    }
    GroupWait(*transportGroup, ckeIdx, allBit);
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] DoGroupSync Ends");
    return;
}

void CcuContextAllReduceMesh2DOneShot::DoGroupReduce(
    std::vector<CcuRep::Variable>& srcBase, CcuRep::Variable& dstBase, CcuRep::Variable& offset, GroupOpSize& goSize)
{
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] DoGroupReduce Starts");
    // 从轴上所有的对端的对应位置读取数据
    std::vector<CcuRep::Memory> srcAddrs;
    for (uint64_t rankIdx = 0; rankIdx < currentAxisRankSize_; rankIdx++) {
        srcAddrs.push_back(CreateMemory());
    }
    uint32_t rmtId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < currentAxisRankSize_; rankIdx++) {
        if (rankIdx != myRankIdxInCurrentAxis_) {
            curId = rmtId;
            rmtId++;
        } else {
            curId = currentAxisRankSize_ - 1;
        }
        srcAddrs[curId].addr = srcBase[rankIdx];
        srcAddrs[curId].addr += offset;
        srcAddrs[curId].token = token_[rankIdx];
    }
    // Reduce 到本端
    CcuRep::Memory dstAddr = CreateMemory();
    dstAddr.addr = dstBase;
    dstAddr.addr += offset;
    dstAddr.token = token_[myRankIdxInCurrentAxis_];
    GroupReduce(transports, dstAddr, srcAddrs, goSize, dataType_, outputDataType_, reduceOp_);
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] DoGroupReduce Ends");
    return;
}

void CcuContextAllReduceMesh2DOneShot::DoAxisSync(uint32_t signalIdx)
{
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] DoAxisSync Starts, signalIdx[%u]", signalIdx);
    uint32_t sendBit = 1 << axisId_;
    uint32_t waitBit = 1 << (1 - axisId_);
    sendBit = sendBit << (AXIS_NUM * signalIdx);
    waitBit = waitBit << (AXIS_NUM * signalIdx);
    LocalCtxPost(otherAxisSignal_, sendBit);
    LocalWait(currAxisSignal_, waitBit);
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] DoAxisSync Ends");
    return;
}

void CcuContextAllReduceMesh2DOneShot::InitVariables()
{
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] InitVariables Starts");
    // 初始化资源
    uint16_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh2DOneShot transports is empty"));
    }
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < currentAxisRankSize_; peerId++) {
        if (peerId == myRankIdxInCurrentAxis_) {
            inputAddr_.push_back(CreateVariable());
            scratchAddr_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextAllReduceMesh2DOneShot] MyRank[%u], PeerId[%llu], TransportId[%u]", myRankIdxInCurrentAxis_,
                peerId, transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr || transportIdx >= transports.size(),
                HCCL_ERROR("[CcuContextAllReduceMesh2DOneShot] Algorithm transport ptr is null or transportIdx is out "
                           "of bounds"), );
            inputAddr_.push_back(CreateVariable((*transports[transportIdx]), INPUT_XN_ID));
            scratchAddr_.push_back(CreateVariable((*transports[transportIdx]), SCRATCH_XN_ID));
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
        outputAddr_.push_back(CreateVariable());
    }

    xSliceOffset_ = CreateVariable();
    ySliceOffset_ = CreateVariable();
    xGoSize_ = CreateGroupOpSize();
    yGoSize_ = CreateGroupOpSize();
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] InitVariables Ends");
    return;
}

void CcuContextAllReduceMesh2DOneShot::LoadArgs()
{
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] LoadArgs Starts");
    Load(inputAddr_[myRankIdxInCurrentAxis_]);
    Load(outputAddr_[myRankIdxInCurrentAxis_]);
    Load(token_[myRankIdxInCurrentAxis_]);
    Load(scratchAddr_[myRankIdxInCurrentAxis_]);
    Load(xSliceOffset_);
    Load(ySliceOffset_);
    Load(xGoSize_);
    Load(yGoSize_);
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] LoadArgs Ends");
    return;
}

std::vector<uint64_t> CcuContextAllReduceMesh2DOneShot::GeneArgs(const CcuTaskArg& arg)
{
    HCCL_INFO("[CcuContextAllReduceMesh2DOneShot] GeneArgs Starts");
    const CcuTaskArgAllReduceMesh2DOneShot* taskArg = dynamic_cast<const CcuTaskArgAllReduceMesh2DOneShot*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh2DOneShot::taskArg ptr is null"));
    }
    uint64_t tokenInfo = taskArg->token_;

    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t scratchAddr = taskArg->scratchAddr_;

    uint64_t xSliceOffset = taskArg->xSliceOffset_;
    uint64_t ySliceOffset = taskArg->ySliceOffset_;

    auto xGoSize = CalGoSize(taskArg->xSliceSize_);
    auto yGoSize = CalGoSize(taskArg->ySliceSize_);

    HCCL_INFO(
        "[CcuContextAllReduceMesh2DOneShot] GeneArgs, TaskArgs are inputAddr[%llu], "
        "outputAddr[%llu], scratchAddr[%llu], xSliceSize[%llu], ySliceSize[%llu], xSliceOffset[%llu], "
        "ySliceOffset[%llu]",
        inputAddr, outputAddr, scratchAddr, taskArg->xSliceSize_, taskArg->ySliceSize_, xSliceOffset, ySliceOffset);

    std::vector<uint64_t> taskArgList = {inputAddr, outputAddr, tokenInfo, scratchAddr, xSliceOffset, ySliceOffset};
    // push goSize
    for (auto goSize : {xGoSize, yGoSize}) {
        for (auto val : goSize) {
            taskArgList.push_back(val);
        }
    }
    return taskArgList;
}
} // namespace Hccl
