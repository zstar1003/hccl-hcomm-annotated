/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_reduce_mesh2d_two_shot_mem2mem.h"
#include "ccu_instruction_all_reduce_mesh2d_two_shot_mem2mem.h"

namespace Hccl {
constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;
constexpr int CKE_IDX_5 = 5;
constexpr int CKE_IDX_6 = 6;
constexpr uint32_t AXIS_NUM = 2;
CcuContextAllReduceMeshTwoShotMem2Mem2D::CcuContextAllReduceMeshTwoShotMem2Mem2D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllReduceMeshTwoShotMem2Mem2D* ctxArg
        = dynamic_cast<const CcuCtxArgAllReduceMeshTwoShotMem2Mem2D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMeshTwoShotMem2Mem2D::ctxArg ptr is null"));
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
            "[CcuContextAllReduceMeshTwoShotMem2Mem2D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }

    uint32_t max_dimSize = 2;
    if (dimSize_.size() != max_dimSize or axisId_ > 1) {
        THROW<NullPtrException>(StringFormat(
            "[CcuContextAllReduceMeshTwoShotMem2Mem2D] dimSize[%u] or axisId[%u] is invalid", dimSize_.size(),
            axisId_));
    }
    CHK_PRT_THROW(
        dimSize_[0] == 0 || dimSize_[1] == 0,
        HCCL_ERROR(
            "[CcuContextAllReduceMeshTwoShotMem2Mem2D] dimSize0[%llu] or dimSize1[%llu] is zero", dimSize_[0],
            dimSize_[1]),
        InvalidParamsException, "dimSize[0] or dimSize[1] is invalid");

    rankSize_ = dimSize_[0] * dimSize_[1];
    myRankIdxInAxis_.push_back(rankId_ % dimSize_[0]); // 本 rank 在第 0 维上的 index
    myRankIdxInAxis_.push_back(rankId_ / dimSize_[0]); // 本 rank 在第 1 维上的 index

    myRankIdxInCurrentAxis_ = myRankIdxInAxis_[axisId_];
    currentAxisRankSize_ = dimSize_[axisId_];

    otherAxisId_ = 1 - axisId_;
    myRankIdxInOtherAxis_ = myRankIdxInAxis_[otherAxisId_];
    otherAxisRankSize_ = dimSize_[otherAxisId_];

    // 同步信号初始化
    currAxisSignalName_ = "CcuContextAllReduceMeshTwoShotMem2Mem2DAxisSync_" + std::to_string(axisId_);
    otherAxisSignalName_ = "CcuContextAllReduceMeshTwoShotMem2Mem2DAxisSync_" + std::to_string(otherAxisId_);
    currAxisSignal_ = CreateMaskSignal();
    ExportMaskSignal(currAxisSignal_, currAxisSignalName_);
    otherAxisSignal_ = ImportMaskSignal(otherAxisSignalName_);
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::Algorithm()
{
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] AllReduceMeshMem2Mem2D run.");
    selfBit_ = 1 << myRankIdxInCurrentAxis_;
    allBit_ = ((1 << currentAxisRankSize_) - 1) & (~(1 << myRankIdxInCurrentAxis_));

    InitVariables();
    LoadArgs();
    PreSync();

    // ==== TwoShot Step1 Reduce Scatter (GroupReduce) ====
    uint64_t currStepStartingSliceRankIdx = myRankIdxInCurrentAxis_ * otherAxisRankSize_;
    uint64_t currStepSliceNumber = otherAxisRankSize_; // 总片数为：对轴 RankSize
    uint64_t currStepSliceType = axisId_;              // 数据片为：本轴数据片
    for (uint64_t currentSliceRankIdx = currStepStartingSliceRankIdx;
         currentSliceRankIdx < currStepStartingSliceRankIdx + currStepSliceNumber; currentSliceRankIdx++) {
        GetSliceOffsetAndGoSize(currentSliceRankIdx, currStepSliceType);
        DoGroupReduce(inputAddr_, inputAddr_[myRankIdxInCurrentAxis_]);
    }
    SyncAll(CKE_IDX_4);

    // ==== TwoShot Step2 Reduce Scatter (GroupReduce) ====
    currStepStartingSliceRankIdx = myRankIdxInOtherAxis_ * currentAxisRankSize_ + myRankIdxInCurrentAxis_;
    currStepSliceNumber = 1;          // 总片数为：1
    currStepSliceType = otherAxisId_; // 数据片为：对轴数据片
    for (uint64_t currentSliceRankIdx = currStepStartingSliceRankIdx;
         currentSliceRankIdx < currStepStartingSliceRankIdx + currStepSliceNumber; currentSliceRankIdx++) {
        GetSliceOffsetAndGoSize(currentSliceRankIdx, currStepSliceType);
        DoGroupReduce(inputAddr_, inputAddr_[myRankIdxInCurrentAxis_]);
    }
    SyncAll(CKE_IDX_5);

    // ==== TwoShot Step3 All Gather (allGatherStep) ====
    currStepStartingSliceRankIdx = myRankIdxInOtherAxis_ * currentAxisRankSize_ + myRankIdxInCurrentAxis_;
    currStepSliceNumber = 1;          // 总片数为：1
    currStepSliceType = otherAxisId_; // 数据片为：对轴数据片
    for (uint64_t currentSliceRankIdx = currStepStartingSliceRankIdx;
         currentSliceRankIdx < currStepStartingSliceRankIdx + currStepSliceNumber; currentSliceRankIdx++) {
        GetSliceOffsetAndGoSize(currentSliceRankIdx, currStepSliceType);
        AllGatherStep(inputAddr_[myRankIdxInCurrentAxis_], outputAddr_);
    }
    SyncAll(CKE_IDX_6);

    // ==== TwoShot Step4 All Gather (allGatherStep) ====
    currStepStartingSliceRankIdx = myRankIdxInCurrentAxis_ * otherAxisRankSize_;
    currStepSliceNumber = otherAxisRankSize_; // 总片数为：对轴 RankSize
    currStepSliceType = axisId_;              // 数据片为：本轴数据片
    for (uint64_t currentSliceRankIdx = currStepStartingSliceRankIdx;
         currentSliceRankIdx < currStepStartingSliceRankIdx + currStepSliceNumber; currentSliceRankIdx++) {
        GetSliceOffsetAndGoSize(currentSliceRankIdx, currStepSliceType);
        AllGatherStep(outputAddr_[myRankIdxInCurrentAxis_], outputAddr_);
    }
    SyncAll(CKE_IDX_0);

    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] AllReduceMeshMem2Mem2D end.");
    return;
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::GetSliceOffsetAndGoSize(
    uint64_t currentSliceRankIdx, uint64_t currStepSliceType)
{
    HCCL_INFO(
        "[CcuContextAllReduceMeshTwoShotMem2Mem2D] GetSliceOffsetAndGoSize Starts, currentSliceRankIdx[%llu], "
        "currStepSliceType[%llu], myRankIdxInAxisX[%llu], myRankIdxInAxisY[%llu], axisId[%u]",
        currentSliceRankIdx, currStepSliceType, myRankIdxInAxis_[0], myRankIdxInAxis_[1], axisId_);
    curOffset_ = 0;
    CcuRep::Variable normalSliceSize = CreateVariable();
    normalSliceSize = normalRankXSliceSize_;
    normalSliceSize += normalRankYSliceSize_;
    // currentSliceRankIdx * normalSliceSize 是每个 rank 的 slice 的起始位置
    for (uint64_t i = 0; i < currentSliceRankIdx; i++) {
        curOffset_ += normalSliceSize;
    }

    if (currentSliceRankIdx == rankSize_ - 1) {
        // 最后一个rank的数据量可能会大过 normalSliceSize，因为要额外处理尾块
        if (currStepSliceType == 0) {
            HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D][GetSliceOffsetAndGoSize] Last Rank X Slice");
            currGoSize_ = lastRankXGoSize_;
            curSliceVec_ = lastXSlices_;
            curOffsetVec_ = lastXOffsets_;
            curSliceSize_ = lastRankXSliceSize_;
        } else {
            HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D][GetSliceOffsetAndGoSize] Last Rank Y Slice");
            // Y 轴上需要额外添加 X 轴数据块大小的偏移
            curOffset_ += lastRankXSliceSize_;
            currGoSize_ = lastRankYGoSize_;
            curSliceVec_ = lastYSlices_;
            curOffsetVec_ = lastYOffsets_;
            curSliceSize_ = lastRankYSliceSize_;
        }
    } else {
        if (currStepSliceType == 0) {
            HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D][GetSliceOffsetAndGoSize] Normal Rank X Slice");
            currGoSize_ = normalRankXGoSize_;
            curSliceVec_ = normalXSlices_;
            curOffsetVec_ = normalXOffsets_;
            curSliceSize_ = normalRankXSliceSize_;
        } else {
            HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D][GetSliceOffsetAndGoSize] Normal Rank Y Slice");
            // Y 轴上需要额外添加 X 轴数据块大小的偏移
            curOffset_ += normalRankXSliceSize_;
            currGoSize_ = normalRankYGoSize_;
            curSliceVec_ = normalYSlices_;
            curOffsetVec_ = normalYOffsets_;
            curSliceSize_ = normalRankYSliceSize_;
        }
    }

    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] GetSliceOffsetAndGoSize Ends");
    return;
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::InitVariables()
{
    uint16_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMeshTwoShotMem2Mem2D transports is empty"));
    }

    for (uint64_t peerId = 0; peerId < currentAxisRankSize_; peerId++) {
        if (peerId == myRankIdxInCurrentAxis_) {
            inputAddr_.push_back(CreateVariable());
            outputAddr_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            CHK_PRT_RET(
                transports[transportIdx] == nullptr || transportIdx >= transports.size(),
                HCCL_ERROR("[CcuContextAllReduceMeshTwoShotMem2Mem2D] Algorithm transport ptr is null or transportIdx "
                           "is out of bounds"), );
            inputAddr_.push_back(CreateVariable((*transports[transportIdx]), INPUT_XN_ID));
            outputAddr_.push_back(CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID));
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    normalRankXSliceSize_ = CreateVariable();
    normalRankYSliceSize_ = CreateVariable();
    lastRankXSliceSize_ = CreateVariable();
    lastRankYSliceSize_ = CreateVariable();
    curOffset_ = CreateVariable();
    curSliceSize_ = CreateVariable();
    for (uint64_t i = 0; i < currentAxisRankSize_ - 1; i++) {
        normalXSlices_.push_back(CreateVariable());
        normalXOffsets_.push_back(CreateVariable());
        lastXSlices_.push_back(CreateVariable());
        lastXOffsets_.push_back(CreateVariable());
        normalYSlices_.push_back(CreateVariable());
        normalYOffsets_.push_back(CreateVariable());
        lastYSlices_.push_back(CreateVariable());
        lastYOffsets_.push_back(CreateVariable());
    }

    for (uint64_t i = 0; i < currentAxisRankSize_ - 1; i++) {
        curOffsetVec_.push_back(CreateVariable());
        curSliceVec_.push_back(CreateVariable());
    }
    normalRankXGoSize_ = CreateGroupOpSize();
    normalRankYGoSize_ = CreateGroupOpSize();
    lastRankXGoSize_ = CreateGroupOpSize();
    lastRankYGoSize_ = CreateGroupOpSize();
    currGoSize_ = CreateGroupOpSize();
    for (uint64_t rankIdx = 0; rankIdx < currentAxisRankSize_; rankIdx++) {
        tmpAddrList_.push_back(CreateMemory());
    }
    tmpAddr_ = CreateMemory();
    return;
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::LoadArgs()
{
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] LoadArgs Starts");
    Load(inputAddr_[myRankIdxInCurrentAxis_]);
    Load(outputAddr_[myRankIdxInCurrentAxis_]);
    Load(token_[myRankIdxInCurrentAxis_]);
    Load(normalRankXSliceSize_);
    Load(normalRankYSliceSize_);
    Load(lastRankXSliceSize_);
    Load(lastRankYSliceSize_);

    for (uint64_t i = 0; i < currentAxisRankSize_ - 1; i++)
        Load(normalXSlices_[i]);
    for (uint64_t i = 0; i < currentAxisRankSize_ - 1; i++)
        Load(normalXOffsets_[i]);
    for (uint64_t i = 0; i < currentAxisRankSize_ - 1; i++)
        Load(normalYSlices_[i]);
    for (uint64_t i = 0; i < currentAxisRankSize_ - 1; i++)
        Load(normalYOffsets_[i]);

    for (uint64_t i = 0; i < currentAxisRankSize_ - 1; i++)
        Load(lastXSlices_[i]);
    for (uint64_t i = 0; i < currentAxisRankSize_ - 1; i++)
        Load(lastXOffsets_[i]);
    for (uint64_t i = 0; i < currentAxisRankSize_ - 1; i++)
        Load(lastYSlices_[i]);
    for (uint64_t i = 0; i < currentAxisRankSize_ - 1; i++)
        Load(lastYOffsets_[i]);

    Load(normalRankXGoSize_);
    Load(normalRankYGoSize_);
    Load(lastRankXGoSize_);
    Load(lastRankYGoSize_);
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] LoadArgs Ends");
    return;
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::PreSync()
{
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] PreSync Starts");
    // 前同步
    for (auto t : transports) {
        WriteVariableWithSignal(*t, inputAddr_[myRankIdxInCurrentAxis_], INPUT_XN_ID, CKE_IDX_1, selfBit_);
        WriteVariableWithSignal(*t, outputAddr_[myRankIdxInCurrentAxis_], OUTPUT_XN_ID, CKE_IDX_2, selfBit_);
        WriteVariableWithSignal(*t, token_[myRankIdxInCurrentAxis_], TOKEN_XN_ID, CKE_IDX_3, selfBit_);
    }

    GroupWait(*transportGroup, CKE_IDX_1, allBit_);
    GroupWait(*transportGroup, CKE_IDX_2, allBit_);
    GroupWait(*transportGroup, CKE_IDX_3, allBit_);
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] PreSync Ends");
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::SyncAll(int ckeIdx)
{
    DoAxisSync(0);
    DoGroupSync(ckeIdx, selfBit_, allBit_);
    DoAxisSync(1);
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::DoAxisSync(uint32_t signalIdx)
{
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] DoAxisSync Starts, signalIdx[%u]", signalIdx);
    uint32_t sendBit = 1 << axisId_;
    uint32_t waitBit = 1 << (1 - axisId_);
    sendBit = sendBit << (AXIS_NUM * signalIdx);
    waitBit = waitBit << (AXIS_NUM * signalIdx);
    LocalCtxPost(otherAxisSignal_, sendBit);
    LocalWait(currAxisSignal_, waitBit);
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] DoAxisSync Ends");
    return;
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::DoGroupSync(int ckeIdx, uint16_t selfBit, uint16_t allBit)
{
    HCCL_INFO(
        "[CcuContextAllReduceMeshTwoShotMem2Mem2D] DoGroupSync Starts, ckeIdx[%d], selfBit[%u], allBit[%u]", ckeIdx,
        selfBit, allBit);
    for (auto t : transports) {
        RemotePost(*t, ckeIdx, selfBit);
    }
    GroupWait(*transportGroup, ckeIdx, allBit);
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] DoGroupSync Ends");
    return;
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::DoGroupReduce(
    std::vector<CcuRep::Variable>& srcAddr, CcuRep::Variable& dstAddr)
{
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] DoGroupReduce starts");
    uint16_t allBit = ((1 << currentAxisRankSize_) - 1) & (~(1 << myRankIdxInCurrentAxis_));
    std::vector<CcuRep::Memory>& src = tmpAddrList_;
    CcuRep::Memory& dst = tmpAddr_;

    dst.token = token_[myRankIdxInCurrentAxis_];
    for (uint64_t rankIdx = 0; rankIdx < currentAxisRankSize_; rankIdx++) {
        src[rankIdx].token = token_[rankIdx];
    }

    CcuRep::MaskSignal locMask = CreateMaskSignal();
    for (uint64_t i = 0; i < (currentAxisRankSize_ - 1); i++) {
        for (uint64_t j = 0; j < (currentAxisRankSize_ - 1); j++) {
            uint16_t nextNum = i + j + 1;
            if (nextNum >= currentAxisRankSize_) {
                nextNum += 1;
            }
            uint16_t rmtRank = (myRankIdxInCurrentAxis_ + nextNum) % currentAxisRankSize_;
            uint16_t rmtTransport;
            if (rmtRank < myRankIdxInCurrentAxis_) {
                rmtTransport = rmtRank;
            } else {
                rmtTransport = rmtRank - 1;
            }

            dst.addr = dstAddr;
            src[rmtRank].addr = srcAddr[rmtRank];
            dst.addr += curOffset_;
            src[rmtRank].addr += curOffset_;
            dst.addr += curOffsetVec_[j];
            src[rmtRank].addr += curOffsetVec_[j];
            CCU_IF(curSliceVec_[j] == 0) { LocalPost(locMask, (1 << rmtRank)); }
            CCU_IF(curSliceVec_[j] != 0)
            {
                ReadReduce(
                    *transports[rmtTransport], dst, src[rmtRank], curSliceVec_[j], dataType_, reduceOp_, locMask,
                    1 << rmtRank);
            }
        }
        LocalWait(locMask, allBit);
    }
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] DoGroupReduce end");
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::AllGatherStep(
    CcuRep::Variable& srcAddr, std::vector<CcuRep::Variable>& dstAddr)
{
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] AllGatherStep Starts");
    CcuRep::Memory& src = tmpAddr_;
    std::vector<CcuRep::Memory>& dst = tmpAddrList_;
    src.addr = srcAddr;
    src.addr += curOffset_;
    src.token = token_[myRankIdxInCurrentAxis_];
    CCU_IF(curSliceSize_ != 0)
    {
        uint32_t transportId = 0;
        CcuRep::MaskSignal locMask = CreateMaskSignal();
        for (uint64_t rankIdx = 0; rankIdx < currentAxisRankSize_; rankIdx++) {
            dst[rankIdx].addr = dstAddr[rankIdx];
            dst[rankIdx].addr += curOffset_;
            dst[rankIdx].token = token_[rankIdx];

            if (rankIdx == myRankIdxInCurrentAxis_) {
                LocalPost(locMask, (1 << rankIdx));
            } else {
                Write(*transports[transportId], dst[rankIdx], src, curSliceSize_, locMask, 1 << rankIdx);
                transportId++;
            }
        }
        GroupCopy(dst[myRankIdxInCurrentAxis_], src, currGoSize_);
        LocalWait(locMask, (1 << currentAxisRankSize_) - 1);
    }
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] AllGatherStep end");
}

void CcuContextAllReduceMeshTwoShotMem2Mem2D::CalMeshChunkSlices(
    uint64_t totalSize, uint64_t sliceNum, std::vector<uint64_t>& slices, std::vector<uint64_t>& offsets)
{
    if (sliceNum == 0) {
        THROW<InvalidParamsException>(StringFormat(
            "[CcuContextAllReduceMeshTwoShotMem2Mem2D][CalMeshChunkSlices] Invalid sliceNum [%u] .", sliceNum));
    }
    uint64_t totalCount = totalSize / DataTypeSizeGet(dataType_);
    uint64_t bigNum = totalCount % sliceNum;
    uint64_t bigSize = (totalCount / sliceNum + 1) * DataTypeSizeGet(dataType_);
    uint64_t smallSize = (totalCount / sliceNum) * DataTypeSizeGet(dataType_);

    // 计算每个分片的大小和偏移量
    uint64_t currentOffset = 0;
    for (uint64_t i = 0; i < sliceNum; ++i) {
        uint64_t chunkSize = 0;
        if (i < bigNum) {
            chunkSize = bigSize;
        } else {
            chunkSize = smallSize;
        }
        slices.push_back(chunkSize);
        offsets.push_back(currentOffset);
        currentOffset += chunkSize;
    }
}

std::vector<uint64_t> CcuContextAllReduceMeshTwoShotMem2Mem2D::GeneArgs(const CcuTaskArg& arg)
{
    HCCL_INFO("[CcuContextAllReduceMeshTwoShotMem2Mem2D] GeneArgs Starts");
    const CcuTaskArgAllReduceMeshTwoShotMem2Mem2D* taskArg
        = dynamic_cast<const CcuTaskArgAllReduceMeshTwoShotMem2Mem2D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMeshTwoShotMem2Mem2D::taskArg ptr is null"));
    }
    uint64_t tokenInfo = taskArg->token_;
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;

    uint64_t normalRankXSliceSize = taskArg->normalRankXSliceSize_;
    uint64_t normalRankYSliceSize = taskArg->normalRankYSliceSize_;
    std::vector<uint64_t> normalXSlices{};
    std::vector<uint64_t> normalXOffsets{};
    std::vector<uint64_t> normalYSlices{};
    std::vector<uint64_t> normalYOffsets{};
    CalMeshChunkSlices(normalRankXSliceSize, currentAxisRankSize_ - 1, normalXSlices, normalXOffsets);
    CalMeshChunkSlices(normalRankYSliceSize, currentAxisRankSize_ - 1, normalYSlices, normalYOffsets);

    uint64_t lastRankXSliceSize = taskArg->lastRankXSliceSize_;
    uint64_t lastRankYSliceSize = taskArg->lastRankYSliceSize_;
    std::vector<uint64_t> lastXSlices{};
    std::vector<uint64_t> lastXOffsets{};
    std::vector<uint64_t> lastYSlices{};
    std::vector<uint64_t> lastYOffsets{};
    CalMeshChunkSlices(lastRankXSliceSize, currentAxisRankSize_ - 1, lastXSlices, lastXOffsets);
    CalMeshChunkSlices(lastRankYSliceSize, currentAxisRankSize_ - 1, lastYSlices, lastYOffsets);

    auto normalRankXGoSize = CalGoSize(normalRankXSliceSize);
    auto normalRankYGoSize = CalGoSize(normalRankYSliceSize);
    auto lastRankXGoSize = CalGoSize(lastRankXSliceSize);
    auto lastRankYGoSize = CalGoSize(lastRankYSliceSize);

    HCCL_INFO(
        "[CcuContextAllReduceMeshTwoShotMem2Mem2D] GeneArgs, TaskArgs are inputAddr[%llu], "
        "outputAddr[%llu], normalRankXSliceSize[%llu], normalRankYSliceSize[%llu], lastRankXSliceSize[%llu], "
        "lastRankYSliceSize[%llu]",
        inputAddr, outputAddr, normalRankXSliceSize, normalRankYSliceSize, lastRankXSliceSize, lastRankYSliceSize);

    std::vector<uint64_t> taskArgList{
        inputAddr,          outputAddr,        tokenInfo, normalRankXSliceSize, normalRankYSliceSize,
        lastRankXSliceSize, lastRankYSliceSize};

    for (const auto& vec :
         {normalXSlices, normalXOffsets, normalYSlices, normalYOffsets, lastXSlices, lastXOffsets, lastYSlices,
          lastYOffsets}) {
        for (auto val : vec) {
            taskArgList.push_back(val);
        }
    }

    // push goSize
    for (auto goSize : {normalRankXGoSize, normalRankYGoSize, lastRankXGoSize, lastRankYGoSize}) {
        for (auto val : goSize) {
            taskArgList.push_back(val);
        }
    }
    return taskArgList;
}
} // namespace Hccl
