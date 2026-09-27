/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_reduce_mesh2d_two_shot.h"
#include "ccu_instruction_all_reduce_mesh2d_two_shot.h"

namespace Hccl {
constexpr uint32_t AXIS_NUM = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;
constexpr int CKE_IDX_5 = 5;
constexpr int CKE_IDX_6 = 6;
constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;

CcuContextAllReduceMesh2DTwoShot::CcuContextAllReduceMesh2DTwoShot(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllReduceMesh2DTwoShot* ctxArg = dynamic_cast<const CcuCtxArgAllReduceMesh2DTwoShot*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh2DTwoShot::ctxArg ptr is null"));
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
            "[CcuContextAllReduceMesh2DTwoShot] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }

    uint32_t max_dimSize = 2;
    if (dimSize_.size() != max_dimSize or axisId_ > 1) {
        THROW<NullPtrException>(StringFormat(
            "[CcuContextAllReduceMesh2DTwoShot] dimSize[%u] or axisId[%u] is invalid", dimSize_.size(), axisId_));
    }
    CHK_PRT_THROW(
        dimSize_[0] == 0 || dimSize_[1] == 0,
        HCCL_ERROR(
            "[CcuContextAllReduceMesh2DTwoShot] dimSize0[%llu] or dimSize1[%llu] is zero", dimSize_[0], dimSize_[1]),
        InvalidParamsException, "dimSize[0] or dimSize[1] is invalid");

    rankSize_ = dimSize_[0] * dimSize_[1];
    HCCL_INFO(
        "[CcuContextAllReduceMesh2DTwoShot] Init, CtxArgs are rankSize[%llu], dimSize0[%llu], dimSize1[%llu], "
        "axisId[%u], "
        "rankId[%llu], dataType[%s], outputDataType[%s], reduceOp[%s]",
        rankSize_, dimSize_[0], dimSize_[1], axisId_, rankId_, dataType_.Describe().c_str(),
        outputDataType_.Describe().c_str(), reduceOp_.Describe().c_str());

    CHK_PRT_THROW(
        dimSize_[0] == 0 || dimSize_[1] == 0,
        HCCL_ERROR(
            "[CcuContextAllReduceMesh2DTwoShot] dimSize0[%llu] or dimSize1[%llu] is zero", dimSize_[0], dimSize_[1]),
        InvalidParamsException, "dimSize[0] or dimSize[1] is invalid");

    myRankIdxInAxis_.push_back(rankId_ % dimSize_[0]); // 本 rank 在第 0 维上的 index
    myRankIdxInAxis_.push_back(rankId_ / dimSize_[0]); // 本 rank 在第 1 维上的 index

    myRankIdxInCurrentAxis_ = myRankIdxInAxis_[axisId_];
    currentAxisRankSize_ = dimSize_[axisId_];

    otherAxisId_ = 1 - axisId_;
    myRankIdxInOtherAxis_ = myRankIdxInAxis_[otherAxisId_];
    otherAxisRankSize_ = dimSize_[otherAxisId_];

    // 同步信号初始化
    currAxisSignalName_ = "CcuContextAllReduceMesh2DTwoShotAxisSync_" + std::to_string(axisId_);
    otherAxisSignalName_ = "CcuContextAllReduceMesh2DTwoShotAxisSync_" + std::to_string(otherAxisId_);
    currAxisSignal_ = CreateMaskSignal();
    ExportMaskSignal(currAxisSignal_, currAxisSignalName_);
    otherAxisSignal_ = ImportMaskSignal(otherAxisSignalName_);

    HCCL_INFO(
        "[CcuContextAllReduceMesh2DTwoShot] Init, myRankIdx0[%llu], myRankIdx1[%llu], "
        "myRankIdxInCurrentAxis[%llu], currentAxisRankSize[%llu]",
        myRankIdxInAxis_[0], myRankIdxInAxis_[1], myRankIdxInCurrentAxis_, currentAxisRankSize_);
}

void CcuContextAllReduceMesh2DTwoShot::Algorithm()
{
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] AllReduceMesh2DTwoShot run.");
    selfBit_ = 1 << myRankIdxInCurrentAxis_;
    allBit_ = ((1 << currentAxisRankSize_) - 1) & (~(1 << myRankIdxInCurrentAxis_));

    InitVariables();
    LoadArgs();
    PreSync();

    CcuRep::Variable currOffset = CreateVariable();
    GroupOpSize currGoSize = CreateGroupOpSize();

    // ==== TwoShot Step1 Reduce Scatter (GroupReduce) ====
    // 第1步reduce的第一个数据片：本轴 MyRank * 对轴 RankSize
    uint64_t currStepStartingSliceRankIdx = myRankIdxInCurrentAxis_ * otherAxisRankSize_;
    uint64_t currStepSliceNumber = otherAxisRankSize_; // 总片数为：对轴 RankSize
    uint64_t currStepSliceType = axisId_;              // 数据片为：本轴数据片
    HCCL_INFO(
        "[Algorithm] Step1: currStepStartingSliceRankIdx[%llu], currStepSliceNumber[%llu], "
        "currStepSliceType[%llu]",
        currStepStartingSliceRankIdx, currStepSliceNumber, currStepSliceType);

    for (uint64_t currentSliceRankIdx = currStepStartingSliceRankIdx;
         currentSliceRankIdx < currStepStartingSliceRankIdx + currStepSliceNumber; currentSliceRankIdx++) {
        GetSliceOffsetAndGoSize(currentSliceRankIdx, currStepSliceType, currOffset, currGoSize);
        DoGroupReduce(inputAddr_, inputAddr_[myRankIdxInCurrentAxis_], currOffset, currGoSize);
    }
    SyncAll(CKE_IDX_4);

    // ==== TwoShot Step2 Reduce Scatter (GroupReduce) ====
    // reduce数据片：对轴 MyRank * 本轴 RankSize + 对轴 MyRank
    currStepStartingSliceRankIdx = myRankIdxInOtherAxis_ * currentAxisRankSize_ + myRankIdxInCurrentAxis_;
    currStepSliceNumber = 1;          // 总片数为：1
    currStepSliceType = otherAxisId_; // 数据片为：对轴数据片
    HCCL_INFO(
        "[Algorithm] Step2: currStepStartingSliceRankIdx[%llu], currStepSliceNumber[%llu], "
        "currStepSliceType[%llu]",
        currStepStartingSliceRankIdx, currStepSliceNumber, currStepSliceType);

    for (uint64_t currentSliceRankIdx = currStepStartingSliceRankIdx;
         currentSliceRankIdx < currStepStartingSliceRankIdx + currStepSliceNumber; currentSliceRankIdx++) {
        GetSliceOffsetAndGoSize(currentSliceRankIdx, currStepSliceType, currOffset, currGoSize);
        DoGroupReduce(inputAddr_, inputAddr_[myRankIdxInCurrentAxis_], currOffset, currGoSize);
    }
    SyncAll(CKE_IDX_5);

    // ==== TwoShot Step3 All Gather (GroupBroadcast) ====
    // Broadcast 的第一个数据片：对轴 MyRank * 本轴 RankSize + 对轴 MyRank
    currStepStartingSliceRankIdx = myRankIdxInOtherAxis_ * currentAxisRankSize_ + myRankIdxInCurrentAxis_;
    currStepSliceNumber = 1;          // 总片数为：1
    currStepSliceType = otherAxisId_; // 数据片为：对轴数据片
    HCCL_INFO(
        "[Algorithm] Step3: currStepStartingSliceRankIdx[%llu], currStepSliceNumber[%llu], "
        "currStepSliceType[%llu]",
        currStepStartingSliceRankIdx, currStepSliceNumber, currStepSliceType);

    for (uint64_t currentSliceRankIdx = currStepStartingSliceRankIdx;
         currentSliceRankIdx < currStepStartingSliceRankIdx + currStepSliceNumber; currentSliceRankIdx++) {
        GetSliceOffsetAndGoSize(currentSliceRankIdx, currStepSliceType, currOffset, currGoSize);
        DoGroupBroadcast(inputAddr_[myRankIdxInCurrentAxis_], outputAddr_, currOffset, currGoSize);
    }
    SyncAll(CKE_IDX_6);

    // ==== TwoShot Step4 All Gather (GroupBroadcast) ====
    // Broadcast 的第一个数据片：本轴 MyRank * 对轴 RankSize
    currStepStartingSliceRankIdx = myRankIdxInCurrentAxis_ * otherAxisRankSize_;
    currStepSliceNumber = otherAxisRankSize_; // 总片数为：对轴 RankSize
    currStepSliceType = axisId_;              // 数据片为：本轴数据片
    HCCL_INFO(
        "[Algorithm] Step4: currStepStartingSliceRankIdx[%llu], currStepSliceNumber[%llu], "
        "currStepSliceType[%llu]",
        currStepStartingSliceRankIdx, currStepSliceNumber, currStepSliceType);

    for (uint64_t currentSliceRankIdx = currStepStartingSliceRankIdx;
         currentSliceRankIdx < currStepStartingSliceRankIdx + currStepSliceNumber; currentSliceRankIdx++) {
        GetSliceOffsetAndGoSize(currentSliceRankIdx, currStepSliceType, currOffset, currGoSize);
        DoGroupBroadcast(outputAddr_[myRankIdxInCurrentAxis_], outputAddr_, currOffset, currGoSize);
    }
    SyncAll(CKE_IDX_0);

    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] AllReduceMesh2DTwoShot end.");
    return;
}

void CcuContextAllReduceMesh2DTwoShot::GetSliceOffsetAndGoSize(
    uint64_t currentSliceRankIdx, uint64_t currStepSliceType, CcuRep::Variable& currOffset, GroupOpSize& currGoSize)
{
    HCCL_INFO(
        "[CcuContextAllReduceMesh2DTwoShot] GetSliceOffsetAndGoSize Starts, currentSliceRankIdx[%llu], "
        "currStepSliceType[%llu]",
        currentSliceRankIdx, currStepSliceType);
    currOffset = 0;

    CcuRep::Variable normalSliceSize = CreateVariable();
    normalSliceSize = normalRankXSliceSize_;
    normalSliceSize += normalRankYSliceSize_;
    // currentSliceRankIdx * normalSliceSize 是每个 rank 的 slice 的起始位置
    for (uint64_t i = 0; i < currentSliceRankIdx; i++) {
        currOffset += normalSliceSize;
    }

    if (currentSliceRankIdx == rankSize_ - 1) {
        // 最后一个rank的数据量可能会大过 normalSliceSize，因为要额外处理尾块
        if (currStepSliceType == 0) {
            HCCL_INFO("[GetSliceOffsetAndGoSize] Last Rank X Slice");
            currGoSize = lastRankXGoSize_;
        } else {
            HCCL_INFO("[GetSliceOffsetAndGoSize] Last Rank Y Slice");
            // Y 轴上需要额外添加 X 轴数据块大小的偏移
            currOffset += lastRankXSliceSize_;
            currGoSize = lastRankYGoSize_;
        }
    } else {
        if (currStepSliceType == 0) {
            HCCL_INFO("[GetSliceOffsetAndGoSize] Normal Rank X Slice");
            currGoSize = normalRankXGoSize_;
        } else {
            HCCL_INFO("[GetSliceOffsetAndGoSize] Normal Rank Y Slice");
            // Y 轴上需要额外添加 X 轴数据块大小的偏移
            currOffset += normalRankXSliceSize_;
            currGoSize = normalRankYGoSize_;
        }
    }

    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] GetSliceOffsetAndGoSize Ends");
    return;
}

void CcuContextAllReduceMesh2DTwoShot::InitVariables()
{
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] InitVariables Starts");
    // 初始化资源
    uint16_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh2DTwoShot transports is empty"));
    }
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < currentAxisRankSize_; peerId++) {
        if (peerId == myRankIdxInCurrentAxis_) {
            inputAddr_.push_back(CreateVariable());
            outputAddr_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextAllReduceMesh2DTwoShot] MyRank[%u], PeerId[%llu], TransportId[%u]", myRankIdxInCurrentAxis_,
                peerId, transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr || transportIdx >= transports.size(),
                HCCL_ERROR("[CcuContextAllReduceMesh2DTwoShot] Algorithm transport ptr is null or transportIdx is out "
                           "of bounds"), );
            inputAddr_.push_back(CreateVariable((*transports[transportIdx]), INPUT_XN_ID));
            outputAddr_.push_back(CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID));
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }

    lastRankXSliceSize_ = CreateVariable();
    lastRankYSliceSize_ = CreateVariable();
    normalRankXSliceSize_ = CreateVariable();
    normalRankYSliceSize_ = CreateVariable();

    normalRankXGoSize_ = CreateGroupOpSize();
    normalRankYGoSize_ = CreateGroupOpSize();
    lastRankXGoSize_ = CreateGroupOpSize();
    lastRankYGoSize_ = CreateGroupOpSize();

    for (uint64_t rankIdx = 0; rankIdx < currentAxisRankSize_; rankIdx++) {
        tmpAddrList_.push_back(CreateMemory());
    }
    tmpAddr_ = CreateMemory();

    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] InitVariables Ends");
    return;
}

void CcuContextAllReduceMesh2DTwoShot::LoadArgs()
{
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] LoadArgs Starts");
    Load(inputAddr_[myRankIdxInCurrentAxis_]);
    Load(outputAddr_[myRankIdxInCurrentAxis_]);
    Load(token_[myRankIdxInCurrentAxis_]);
    Load(normalRankXSliceSize_);
    Load(normalRankYSliceSize_);
    Load(lastRankXSliceSize_);
    Load(lastRankYSliceSize_);
    Load(normalRankXGoSize_);
    Load(normalRankYGoSize_);
    Load(lastRankXGoSize_);
    Load(lastRankYGoSize_);
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] LoadArgs Ends");
    return;
}

void CcuContextAllReduceMesh2DTwoShot::PreSync()
{
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] PreSync Starts");
    // 前同步
    for (auto t : transports) {
        WriteVariableWithSignal(*t, inputAddr_[myRankIdxInCurrentAxis_], INPUT_XN_ID, CKE_IDX_1, selfBit_);
        WriteVariableWithSignal(*t, outputAddr_[myRankIdxInCurrentAxis_], OUTPUT_XN_ID, CKE_IDX_2, selfBit_);
        WriteVariableWithSignal(*t, token_[myRankIdxInCurrentAxis_], TOKEN_XN_ID, CKE_IDX_3, selfBit_);
    }

    GroupWait(*transportGroup, CKE_IDX_1, allBit_);
    GroupWait(*transportGroup, CKE_IDX_2, allBit_);
    GroupWait(*transportGroup, CKE_IDX_3, allBit_);
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] PreSync Ends");
}

void CcuContextAllReduceMesh2DTwoShot::SyncAll(int ckeIdx)
{
    DoAxisSync(0);
    DoGroupSync(ckeIdx, selfBit_, allBit_);
    DoAxisSync(1);
}

void CcuContextAllReduceMesh2DTwoShot::DoAxisSync(uint32_t signalIdx)
{
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] DoAxisSync Starts, signalIdx[%u]", signalIdx);
    uint32_t sendBit = 1 << axisId_;
    uint32_t waitBit = 1 << (1 - axisId_);
    sendBit = sendBit << (AXIS_NUM * signalIdx);
    waitBit = waitBit << (AXIS_NUM * signalIdx);
    LocalCtxPost(otherAxisSignal_, sendBit);
    LocalWait(currAxisSignal_, waitBit);
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] DoAxisSync Ends");
    return;
}

void CcuContextAllReduceMesh2DTwoShot::DoGroupSync(int ckeIdx, uint16_t selfBit, uint16_t allBit)
{
    HCCL_INFO(
        "[CcuContextAllReduceMesh2DTwoShot] DoGroupSync Starts, ckeIdx[%d], selfBit[%u], allBit[%u]", ckeIdx, selfBit,
        allBit);
    for (auto t : transports) {
        RemotePost(*t, ckeIdx, selfBit);
    }
    GroupWait(*transportGroup, ckeIdx, allBit);
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] DoGroupSync Ends");
    return;
}

void CcuContextAllReduceMesh2DTwoShot::DoGroupReduce(
    std::vector<CcuRep::Variable>& srcBase, CcuRep::Variable& dstBase, CcuRep::Variable& offset, GroupOpSize& goSize)
{
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] DoGroupReduce Starts");
    // 从轴上所有的对端读取数据
    std::vector<CcuRep::Memory>& srcAddrs = tmpAddrList_;
    uint32_t curId = 0;
    uint32_t rmtId = 0;
    for (uint64_t rankIdx = 0; rankIdx < currentAxisRankSize_; rankIdx++) {
        if (rankIdx != myRankIdxInCurrentAxis_) {
            curId = rmtId;
            rmtId++;
        } else {
            curId = currentAxisRankSize_ - 1;
        }
        srcAddrs[curId].addr = srcBase[rankIdx];
        srcAddrs[curId].token = token_[rankIdx];
        srcAddrs[curId].addr += offset;
    }
    // Reduce 到本端
    CcuRep::Memory& dstAddr = tmpAddr_;
    dstAddr.addr = dstBase;
    dstAddr.addr += offset;
    dstAddr.token = token_[myRankIdxInCurrentAxis_];
    GroupReduce(transports, dstAddr, srcAddrs, goSize, dataType_, outputDataType_, reduceOp_);
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] DoGroupReduce Ends");
    return;
}

void CcuContextAllReduceMesh2DTwoShot::DoGroupBroadcast(
    CcuRep::Variable& srcBase, std::vector<CcuRep::Variable>& dstBase, CcuRep::Variable& offset, GroupOpSize& goSize)
{
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] DoGroupBroadcast Starts");
    // 从轴上所有的对端读取数据
    std::vector<CcuRep::Memory>& dstAddrs = tmpAddrList_;
    uint32_t rmtId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < currentAxisRankSize_; rankIdx++) {
        if (rankIdx != myRankIdxInCurrentAxis_) {
            curId = rmtId;
            rmtId++;
        } else {
            curId = currentAxisRankSize_ - 1;
        }
        dstAddrs[curId].addr = dstBase[rankIdx];
        dstAddrs[curId].addr += offset;
        dstAddrs[curId].token = token_[rankIdx];
    }
    // Reduce 到本端
    CcuRep::Memory& srcAddr = tmpAddr_;
    srcAddr.addr = srcBase;
    srcAddr.addr += offset;
    srcAddr.token = token_[myRankIdxInCurrentAxis_];
    GroupBroadcast(transports, dstAddrs, srcAddr, goSize);

    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] DoGroupBroadcast Ends");
    return;
}

std::vector<uint64_t> CcuContextAllReduceMesh2DTwoShot::GeneArgs(const CcuTaskArg& arg)
{
    HCCL_INFO("[CcuContextAllReduceMesh2DTwoShot] GeneArgs Starts");
    const CcuTaskArgAllReduceMesh2DTwoShot* taskArg = dynamic_cast<const CcuTaskArgAllReduceMesh2DTwoShot*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh2DTwoShot::taskArg ptr is null"));
    }
    uint64_t tokenInfo = taskArg->token_;
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;

    uint64_t normalRankXSliceSize = taskArg->normalRankXSliceSize_;
    uint64_t normalRankYSliceSize = taskArg->normalRankYSliceSize_;
    uint64_t lastRankXSliceSize = taskArg->lastRankXSliceSize_;
    uint64_t lastRankYSliceSize = taskArg->lastRankYSliceSize_;

    auto normalRankXGoSize = CalGoSize(normalRankXSliceSize);
    auto normalRankYGoSize = CalGoSize(normalRankYSliceSize);
    auto lastRankXGoSize = CalGoSize(lastRankXSliceSize);
    auto lastRankYGoSize = CalGoSize(lastRankYSliceSize);

    HCCL_INFO(
        "[CcuContextAllReduceMesh2DTwoShot] GeneArgs, TaskArgs are inputAddr[%llu], "
        "outputAddr[%llu], normalRankXSliceSize[%llu], normalRankYSliceSize[%llu], lastRankXSliceSize[%llu], "
        "lastRankYSliceSize[%llu]",
        inputAddr, outputAddr, normalRankXSliceSize, normalRankYSliceSize, lastRankXSliceSize, lastRankYSliceSize);

    std::vector<uint64_t> taskArgList{
        inputAddr,          outputAddr,        tokenInfo, normalRankXSliceSize, normalRankYSliceSize,
        lastRankXSliceSize, lastRankYSliceSize};

    // push goSize
    for (auto goSize : {normalRankXGoSize, normalRankYGoSize, lastRankXGoSize, lastRankYGoSize}) {
        for (auto val : goSize) {
            taskArgList.push_back(val);
        }
    }
    return taskArgList;
}
} // namespace Hccl
