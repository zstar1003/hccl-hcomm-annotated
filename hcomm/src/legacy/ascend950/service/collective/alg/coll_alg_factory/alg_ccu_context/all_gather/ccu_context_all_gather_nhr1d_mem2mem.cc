/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_gather_nhr1d_mem2mem.h"

namespace Hccl {

constexpr uint16_t OUTPUT_XN_ID = 1;
constexpr uint16_t TOKEN_XN_ID = 2;
constexpr uint16_t FST_AXIS_ID = 0;
constexpr uint16_t SEC_AXIS_ID = 1;
constexpr uint16_t CKE_IDX_0 = 0;
constexpr uint16_t CKE_IDX_1 = 1;
constexpr uint16_t CKE_IDX_2 = 2;
constexpr uint16_t CKE_IDX_3 = 3;
constexpr uint16_t CKE_IDX_4 = 4;
constexpr uint16_t BIT_NUM_PER_CKE = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE

CcuContextAllGatherNHR1D::CcuContextAllGatherNHR1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllGatherNHR1D* ctxArg = dynamic_cast<const CcuCtxArgAllGatherNHR1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllGatherNHR1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    axisId_ = ctxArg->axisId_;
    axisSize_ = ctxArg->axisSize_;
    dimSize_ = ctxArg->dimSize_[0];
    localAxisSignalName_ = "CcuContextAllGatherNHR1DDieSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextAllGatherNHR1DDieSync_" + std::to_string(1 - axisId_);
    stepInfoVector_ = ctxArg->stepInfoVector_;
    indexMap_ = ctxArg->indexMap_;
    localSize_ = indexMap_.size();
    myRankIdx_ = indexMap_.size();
    signalNum_ = (dimSize_ + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE; // 每个CKE有16个bit
    HCCL_INFO(
        "[CcuContextAllGatherNHR1D] CtxArg: rankId_[%u], axisId_[%u], axisSize_[%u], dimSize_[%u], localSize_[%u], "
        "signalNum_[%u]",
        rankId_, axisId_, axisSize_, dimSize_, localSize_, signalNum_);
}

void CcuContextAllGatherNHR1D::LoadArgs()
{
    Load(input_);
    Load(output_[myRankIdx_]);
    Load(token_[myRankIdx_]);
    Load(die0Size_);
    Load(die1Size_);
    Load(repeatNum_);
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(isInputOutputEqual_);

    HCCL_DEBUG("[CcuContextAllGatherNHR1D] LoadArgs run finished");
}

void CcuContextAllGatherNHR1D::InitResources()
{
    die0Size_ = CreateVariable();
    die1Size_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    outputSliceStride_ = CreateVariable();
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    repeatNum_ = CreateVariable();
    tmpCopyRepeatNum_ = CreateVariable();
    repeatTimeflag_ = CreateVariable();
    isInputOutputEqual_ = CreateVariable();
    myrankInputSliceOffset_ = CreateVariable();
    tmpSliceOffset_ = CreateVariable();
    for (u64 i = 0; i < dimSize_; i++) {
        outputSliceOffset_.push_back(CreateVariable());
    }
    constVar1_ = CreateVariable();
    constVar1_ = 1;

    localSignal_ = CreateMaskSignal();
    localAxisSignal_ = CreateMaskSignal();

    if (axisSize_ > 1) {
        ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
        anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);
    }

    input_ = CreateVariable();
    for (uint32_t transportIdx = 0; transportIdx < localSize_; transportIdx++) {
        HCCL_DEBUG("[CcuContextAllGatherNHR1D] MyRank[%u], TransportId[%u]", rankId_, transportIdx);
        CHK_PRT_RET(
            transports[transportIdx] == nullptr,
            HCCL_ERROR("[CcuContextAllGatherNHR1D] Algorithm transport ptr is null"), );
        output_.push_back(
            CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID)); // 获取transport中id=1的Var来传递output
        token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
    }
    output_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    srcMem_ = CreateMemory();
    dstMem_ = CreateMemory();
    HCCL_DEBUG("[CcuContextAllGatherNHR1D] InitResources finished");
}

void CcuContextAllGatherNHR1D::PreSync()
{
    HCCL_DEBUG("[CcuContextAllGatherNHR1D] PreSync start");
    uint16_t selfSignalId = rankId_ / BIT_NUM_PER_CKE;
    uint16_t selfBit = 1 << (rankId_ % BIT_NUM_PER_CKE);
    for (auto t : transports) {
        WriteVariableWithSignal(*t, output_[localSize_], OUTPUT_XN_ID, selfSignalId + signalNum_ * CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*t, token_[localSize_], TOKEN_XN_ID, selfSignalId + signalNum_ * CKE_IDX_2, selfBit);
    }
    std::vector<uint16_t> waitBitVector(signalNum_, 0);
    for (auto& pair : indexMap_) {
        uint16_t pairSignalId = pair.first / BIT_NUM_PER_CKE;
        uint16_t pairBit = 1 << (pair.first % BIT_NUM_PER_CKE);
        waitBitVector[pairSignalId] = waitBitVector[pairSignalId] | pairBit;
    }
    for (uint16_t sId = 0; sId < waitBitVector.size(); sId++) {
        GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_1, waitBitVector[sId]);
        GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_2, waitBitVector[sId]);
    }
    HCCL_DEBUG("[CcuContextAllGatherNHR1D] PreSync end");
}

void CcuContextAllGatherNHR1D::PostSync()
{
    uint16_t selfSignalId = rankId_ / BIT_NUM_PER_CKE;
    uint16_t selfBit = 1 << (rankId_ % BIT_NUM_PER_CKE);
    for (auto& t : transports) {
        RemotePost(*t, selfSignalId + signalNum_ * CKE_IDX_0, selfBit);
    }
    std::vector<uint16_t> waitBitVector(signalNum_, 0);
    for (auto& pair : indexMap_) {
        uint16_t pairSignalId = pair.first / BIT_NUM_PER_CKE;
        uint16_t pairBit = 1 << (pair.first % BIT_NUM_PER_CKE);
        waitBitVector[pairSignalId] = waitBitVector[pairSignalId] | pairBit;
    }
    for (uint32_t sId = 0; sId < signalNum_; sId++) {
        GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_0, waitBitVector[sId]);
    }
    HCCL_DEBUG("[CcuContextAllGatherNHR1D] PostSync run finished");
}

void CcuContextAllGatherNHR1D::AxisSync(uint32_t signalIndex)
{
    const uint32_t DIE_NUM = 2;
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextAllGatherNHR1D] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_DEBUG("[CcuContextAllGatherNHR1D] AxisSync run finished");
    return;
}

void CcuContextAllGatherNHR1D::DoRepeatAllGatherNHR()
{
    tmpSliceOffset_ = 0;
    myrankInputSliceOffset_ = 0;
    for (u64 i = 0; i < rankId_; i++) {
        myrankInputSliceOffset_ += inputSliceStride_;
    }
    for (u64 i = 0; i < dimSize_; i++) {
        outputSliceOffset_[i] = tmpSliceOffset_;
        tmpSliceOffset_ += outputSliceStride_;
    }
    srcMem_.addr = input_;
    srcMem_.addr += myrankInputSliceOffset_;
    dstMem_.addr = output_[myRankIdx_];
    dstMem_.addr += outputSliceOffset_[rankId_];
    srcMem_.token = token_[myRankIdx_];
    dstMem_.token = token_[myRankIdx_];
    tmpCopyRepeatNum_ = repeatNum_;
    repeatTimeflag_ = 0;
    CCU_WHILE(tmpCopyRepeatNum_ != UINT64_MAX)
    {
        tmpCopyRepeatNum_ += constVar1_;
        CCU_IF(repeatTimeflag_ != 0)
        {
            srcMem_.addr += inputRepeatStride_;
            dstMem_.addr += outputRepeatStride_;
        }
        CCU_IF(repeatTimeflag_ == 0)
        {
            if (axisId_ == 1) {
                srcMem_.addr += die0Size_;
                dstMem_.addr += die0Size_;
            }
        }
        CCU_IF(isInputOutputEqual_ == 0)
        {
            LocalCopy(dstMem_, srcMem_, axisId_ == 0 ? die0Size_ : die1Size_, localSignal_, 1 << rankId_);
        }
        CCU_IF(isInputOutputEqual_ != 0) { LocalPost(localSignal_, 1 << rankId_); }
        LocalWait(localSignal_, 1 << rankId_);
        repeatTimeflag_ = 1;
    }

    for (auto& nhrStepInfo : stepInfoVector_) {
        DoRepeatAllGatherNHRSingleStep(nhrStepInfo);
    }
}

void CcuContextAllGatherNHR1D::DoRepeatAllGatherNHRSingleStep(const NHRStepInfo& nhrStepInfo)
{
    u32& toRankIdx = indexMap_[nhrStepInfo.toRank];
    u32& fromRankIdx = indexMap_[nhrStepInfo.fromRank];
    u32 sendSliceIdx = 0;
    CcuTransport* sendTransport = transports[toRankIdx];
    CcuTransport* recvTransport = transports[fromRankIdx];
    const std::vector<u32>& sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    srcMem_.token = token_[myRankIdx_];
    dstMem_.token = token_[toRankIdx];
    for (u32 i = 0; i < sendSliceIdxList.size(); i++) { ////这里写的可能有问题
        sendSliceIdx = sendSliceIdxList[i];
        if (i != 0) {
            if (i % BIT_NUM_PER_CKE == 0) {
                LocalWait(localSignal_, (1 << BIT_NUM_PER_CKE) - 1);
            }
        }
        if (nhrStepInfo.step == 0) {
            srcMem_.addr = input_;
            srcMem_.addr += myrankInputSliceOffset_;
        } else {
            srcMem_.addr = output_[myRankIdx_];
            srcMem_.addr += outputSliceOffset_[sendSliceIdx];
        }
        dstMem_.addr = output_[toRankIdx];
        dstMem_.addr += outputSliceOffset_[sendSliceIdx];
        DoRepeatSendRecvSlices(nhrStepInfo.toRank, srcMem_, dstMem_, i % BIT_NUM_PER_CKE);
    }

    if (nhrStepInfo.step + 1 != stepInfoVector_.size()) {
        uint16_t selfSignalId = rankId_ / BIT_NUM_PER_CKE;
        uint16_t selfBit = 1 << (rankId_ % BIT_NUM_PER_CKE);
        RemotePost(*sendTransport, selfSignalId + signalNum_ * CKE_IDX_3, selfBit, true);
        uint16_t recvSignalId = nhrStepInfo.fromRank / BIT_NUM_PER_CKE;
        uint16_t recvBit = 1 << (nhrStepInfo.fromRank % BIT_NUM_PER_CKE);
        RemoteWait(*recvTransport, recvSignalId + signalNum_ * CKE_IDX_3, recvBit);
    }
}

void CcuContextAllGatherNHR1D::DoRepeatSendRecvSlices(
    const u32& toRank, CcuRep::Memory& src, CcuRep::Memory& dst, u32 signalIndex)
{
    CcuTransport* sendTransport = transports[indexMap_[toRank]];
    const CcuRep::Variable& sliceSize = axisId_ == 0 ? die0Size_ : die1Size_;
    repeatTimeflag_ = 0;
    tmpCopyRepeatNum_ = repeatNum_;
    CCU_WHILE(tmpCopyRepeatNum_ != UINT64_MAX)
    {
        tmpCopyRepeatNum_ += constVar1_;
        CCU_IF(repeatTimeflag_ == 1)
        {
            src.addr += inputRepeatStride_;
            dst.addr += outputRepeatStride_;
        }
        CCU_IF(repeatTimeflag_ == 0)
        {
            if (axisId_ == 1) {
                src.addr += die0Size_;
                dst.addr += die0Size_;
            }
        }
        Write(*sendTransport, dst, src, sliceSize, localSignal_, 1 << signalIndex);
        LocalWait(localSignal_, 1 << signalIndex);
        repeatTimeflag_ = 1;
    }
}

void CcuContextAllGatherNHR1D::Algorithm()
{
    HCCL_DEBUG("[CcuContextAllGatherNHR1D] AllgatherNHR1D run");
    InitResources();
    LoadArgs();
    if (axisSize_ > 1) {
        AxisSync(FST_AXIS_ID);
    }
    PreSync();
    DoRepeatAllGatherNHR();
    PostSync();
    if (axisSize_ > 1) {
        AxisSync(SEC_AXIS_ID);
    }
    HCCL_DEBUG("[CcuContextAllGatherNHR1D] AllgatherNHR1D end");
    return;
}

std::vector<uint64_t> CcuContextAllGatherNHR1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllGatherNHR1D* taskArg = dynamic_cast<const CcuTaskArgAllGatherNHR1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllGatherNHR1D::taskArg ptr is null"));
    }
    // input&output&buffer地址
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t token = taskArg->token_;
    uint64_t die0Size = taskArg->die0Size_;
    uint64_t die1Size = taskArg->die1Size_;
    uint64_t repeatNum = UINT64_MAX - taskArg->repeatNum_;
    uint64_t inputSliceStride = taskArg->inputSliceStride_;
    uint64_t outputSliceStride = taskArg->outputSliceStride_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t isInputOutputEqual = taskArg->isInputOutputEqual_;

    HCCL_INFO(
        "[CcuContextAllGatherNHR1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "die0Size[%llu], die1Size[%llu], repeatNum[%llu] "
        "inputSliceStride[%llu], outputSliceStride[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu]",
        inputAddr, outputAddr, die0Size, die1Size, repeatNum, inputSliceStride, outputSliceStride, inputRepeatStride,
        outputRepeatStride);

    return {inputAddr,          outputAddr,        token,
            die0Size,           die1Size,          repeatNum,
            inputSliceStride,   outputSliceStride, inputRepeatStride,
            outputRepeatStride, isInputOutputEqual};
}
} // namespace Hccl
