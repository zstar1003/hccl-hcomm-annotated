/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_scatter_nhr1d_mem2mem.h"

namespace Hccl {

// 注意型号量变化
constexpr uint16_t INPUT_XN_ID = 0;
constexpr uint16_t TOKEN_XN_ID = 2;
constexpr uint16_t CKE_IDX_0 = 0;
constexpr uint16_t CKE_IDX_1 = 1;
constexpr uint16_t CKE_IDX_2 = 2;
constexpr uint16_t CKE_IDX_3 = 3;
constexpr uint16_t CKE_IDX_4 = 4;
constexpr uint16_t FST_AXIS_ID = 0;
constexpr uint16_t SEC_AXIS_ID = 1;
constexpr uint16_t RANK_NUM_PER_CKE = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE
constexpr uint16_t LINK_SIZE = 2;

CcuContextReduceScatterNHR1DMem2Mem::CcuContextReduceScatterNHR1DMem2Mem(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceScatterNHR1D* ctxArg = dynamic_cast<const CcuCtxArgReduceScatterNHR1D*>(&arg);
    rankId_ = ctxArg->rankId_;
    axisId_ = ctxArg->axisId_;
    dimSize_ = ctxArg->dimSize_[0];
    localAxisSignalName_ = "CcuContextReduceScatterNHR1DDieSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextReduceScatterNHR1DDieSync_" + std::to_string(1 - axisId_);
    stepInfoVector_ = ctxArg->stepInfoVector_;
    indexMap_ = ctxArg->indexMap_;
    localSize_ = indexMap_.size();
    myRankIdx_ = indexMap_.size();
    reduceOp_ = ctxArg->op_.reduceOp;
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    linkNum_ = ctxArg->linkNum_;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceScatterNHR1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    signalNum_ = (dimSize_ + RANK_NUM_PER_CKE - 1) / RANK_NUM_PER_CKE; // 每个CKE有16个bit
    HCCL_INFO(
        "[CcuContextReduceScatterNHR1DMem2Mem] CtxArg: rankId_[%u], axisId_[%u], dimSize_[%u], localSize_[%u], "
        "dataType[%s], outputDataType[%s], reduceOp[%s], signalNum_[%u]",
        rankId_, axisId_, dimSize_, localSize_, dataType_.Describe().c_str(), outputDataType_.Describe().c_str(),
        reduceOp_.Describe().c_str(), signalNum_);
}

void CcuContextReduceScatterNHR1DMem2Mem::LoadArgs()
{
    Load(input_[myRankIdx_]);
    Load(output_);
    Load(token_[myRankIdx_]);
    Load(die0Size_);
    Load(die1Size_);
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(repeatNumVar_);
    Load(isBottom_);
    repeatNumVarTemp_ = repeatNumVar_;
    HCCL_INFO("[CcuContextReduceScatterNHR1DMem2Mem] LoadArgs run finished");
}

void CcuContextReduceScatterNHR1DMem2Mem::InitResources()
{
    die0Size_ = CreateVariable();
    die1Size_ = CreateVariable();
    sliceSize_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    outputSliceStride_ = CreateVariable();
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    localAxisSignal_ = CreateMaskSignal();
    anotherAxisSignal_ = CreateMaskSignal();
    localSignal_ = CreateMaskSignal();
    repeatNumVar_ = CreateVariable();
    repeatNumVarTemp_ = CreateVariable();
    isBottom_ = CreateVariable();

    if (linkNum_ == LINK_SIZE) {
        ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
        anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);
    }

    output_ = CreateVariable();
    for (uint32_t transportIdx = 0; transportIdx < localSize_; transportIdx++) {
        HCCL_INFO("[CcuContextReduceScatterNHR1DMem2Mem] MyRank[%u], TransportId[%u]", rankId_, transportIdx);
        CHK_PRT_RET(
            transports[transportIdx] == nullptr,
            HCCL_ERROR("[CcuContextReduceScatterNHR1DMem2Mem] Algorithm transport ptr is null"), );
        input_.push_back(
            CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=0的Var来传递input

        token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
    }
    input_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    repeatInputOffset_ = CreateVariable();
    repeatOutputOffset_ = CreateVariable();
    myrankInputSliceOffset_ = CreateVariable();

    srcMem_ = CreateMemory();
    dstMem_ = CreateMemory();
    flag_ = CreateVariable();
    HCCL_INFO("[CcuContextReduceScatterNHR1DMem2Mem] InitResources finished");
}

void CcuContextReduceScatterNHR1DMem2Mem::PreSync()
{
    HCCL_INFO("[CcuContextReduceScatterNHR1DMem2Mem] PreSync start");
    // 本rank用哪一个CKE
    uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
    // 本rank用CKE的哪一位
    uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[localSize_], INPUT_XN_ID, selfSignalId + signalNum_ * CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*t, token_[localSize_], TOKEN_XN_ID, selfSignalId + signalNum_ * CKE_IDX_2, selfBit);
    }
    std::vector<uint16_t> waitBitVector(signalNum_, 0);
    for (auto& pair : indexMap_) {
        uint16_t pairSignalId = pair.first / RANK_NUM_PER_CKE;
        uint16_t pairBit = 1 << (pair.first % RANK_NUM_PER_CKE);
        waitBitVector[pairSignalId] = waitBitVector[pairSignalId] | pairBit;
    }
    for (uint16_t sId = 0; sId < waitBitVector.size(); sId++) {
        GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_1, waitBitVector[sId]);
        GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_2, waitBitVector[sId]);
    }
    HCCL_INFO("[CcuContextReduceScatterNHR1DMem2Mem] PreSync end");
}

void CcuContextReduceScatterNHR1DMem2Mem::PostSync()
{
    uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
    uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
    for (auto& t : transports) {
        RemotePost(*t, selfSignalId + signalNum_ * CKE_IDX_0, selfBit);
    }
    std::vector<uint16_t> waitBitVector(signalNum_, 0);
    for (auto& pair : indexMap_) {
        uint16_t pairSignalId = pair.first / RANK_NUM_PER_CKE;
        uint16_t pairBit = 1 << (pair.first % RANK_NUM_PER_CKE);
        waitBitVector[pairSignalId] = waitBitVector[pairSignalId] | pairBit;
    }
    for (uint32_t sId = 0; sId < waitBitVector.size(); sId++) {
        GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_0, waitBitVector[sId]);
    }
    HCCL_INFO("[CcuContextReduceScatterNHR1DMem2Mem] PostSync run finished");
}

void CcuContextReduceScatterNHR1DMem2Mem::AxisSync(uint32_t signalIndex)
{
    const uint32_t DIE_NUM = 2;
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextReduceScatterNHR1DMem2Mem] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_INFO("[CcuContextReduceScatterNHR1DMem2Mem] AxisSync run finished");
    return;
}

void CcuContextReduceScatterNHR1DMem2Mem::DoRepeatReduceScatterNHR()
{
    CcuRep::Variable tmpSliceOffset = CreateVariable();
    tmpSliceOffset = 0;
    // 用来记录每个rank要读取的rank的sliceIdx的偏移
    // 后面会用inputAddr来加上这个偏移获取sliceIdx的地址
    std::vector<CcuRep::Variable> inputSliceOffset;
    CCU_IF(isBottom_ == 1)
    {
        for (u64 i = 0; i < dimSize_; i++) {
            inputSliceOffset.push_back(CreateVariable());
            inputSliceOffset[i] = tmpSliceOffset;
            tmpSliceOffset += inputSliceStride_;
        }
    }
    CCU_IF(isBottom_ == 0)
    {
        for (u64 i = 0; i < dimSize_; i++) {
            inputSliceOffset.push_back(CreateVariable());
            inputSliceOffset[i] = tmpSliceOffset;
            tmpSliceOffset += inputRepeatStride_;
        }
    }

    for (auto& nhrStepInfo : stepInfoVector_) {
        DoRepeatReduceScatterNHRSingleStep(nhrStepInfo, inputSliceOffset);
    }
    // 因为所有的修改都是在input上进行的，所以最后需要把input上的数据搬到output上
    dstMem_.addr = output_;
    dstMem_.token = token_[myRankIdx_];
    srcMem_.addr = input_[myRankIdx_];
    srcMem_.addr += inputSliceOffset[rankId_];
    srcMem_.token = token_[myRankIdx_];

    CcuRep::Variable repeatNumAdd2 = CreateVariable();
    repeatNumAdd2 = 1;
    CCU_WHILE(repeatNumVar_ != UINT64_MAX)
    {
        repeatNumVar_ += repeatNumAdd2;
        CCU_IF(flag_ == 1)
        {
            CCU_IF(isBottom_ == 0)
            {
                srcMem_.addr += inputSliceStride_;
                dstMem_.addr += outputRepeatStride_;
            }
            CCU_IF(isBottom_ == 1)
            {
                srcMem_.addr += inputRepeatStride_;
                dstMem_.addr += outputRepeatStride_;
            }
        }
        CCU_IF(flag_ == 0)
        {
            if (axisId_ == 1) {
                srcMem_.addr += die0Size_;
                dstMem_.addr += die0Size_;
            }
        }
        CcuRep::Variable& localSliceSize = (axisId_ == 0) ? die0Size_ : die1Size_;
        LocalCopy(dstMem_, srcMem_, localSliceSize, localSignal_, 1);
        LocalWait(localSignal_, 1);
        flag_ = 1;
    }
}

void CcuContextReduceScatterNHR1DMem2Mem::DoRepeatReduceScatterNHRSingleStep(
    const NHRStepInfo& nhrStepInfo, const std::vector<CcuRep::Variable>& inputSliceOffset)
{
    u32& toRankIdx = indexMap_[nhrStepInfo.toRank];
    u32& fromRankIdx = indexMap_[nhrStepInfo.fromRank];
    CcuTransport* sendTransport = transports[toRankIdx];
    CcuTransport* recvTransport = transports[fromRankIdx];
    const std::vector<u32>& sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    dstMem_.token = token_[toRankIdx];
    srcMem_.token = token_[myRankIdx_];

    // 被写之前告诉写自己的rank自己准备好了-前同步
    uint16_t recvSignalIdPrev = nhrStepInfo.fromRank / RANK_NUM_PER_CKE;
    uint16_t recvBitPrev = 1 << (nhrStepInfo.fromRank % RANK_NUM_PER_CKE);
    RemotePost(*recvTransport, recvSignalIdPrev + signalNum_ * CKE_IDX_3, recvBitPrev, true);

    uint16_t selfSignalIdPrev = rankId_ / RANK_NUM_PER_CKE;
    uint16_t selfBitPrev = 1 << (rankId_ % RANK_NUM_PER_CKE);
    RemoteWait(*sendTransport, selfSignalIdPrev + signalNum_ * CKE_IDX_3, selfBitPrev);

    for (const u32& sendSliceIdx : sendSliceIdxList) {
        dstMem_.addr = input_[toRankIdx];
        dstMem_.addr += inputSliceOffset[sendSliceIdx];
        srcMem_.addr = input_[myRankIdx_];
        srcMem_.addr += inputSliceOffset[sendSliceIdx];
        DoRepeatSendRecvSlices(nhrStepInfo.toRank, srcMem_, dstMem_);
    }

    // 写之后告诉对面写完了-后同步
    uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
    uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
    RemotePost(*sendTransport, selfSignalId + signalNum_ * CKE_IDX_4, selfBit, true);

    uint16_t recvSignalId = nhrStepInfo.fromRank / RANK_NUM_PER_CKE;
    uint16_t recvBit = 1 << (nhrStepInfo.fromRank % RANK_NUM_PER_CKE);
    RemoteWait(*recvTransport, recvSignalId + signalNum_ * CKE_IDX_4, recvBit);
}

void CcuContextReduceScatterNHR1DMem2Mem::DoRepeatSendRecvSlices(
    const u32& toRank, CcuRep::Memory& src, CcuRep::Memory& dst)
{
    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd = 1;
    flag_ = 0;
    CcuTransport* sendTransport = transports[indexMap_[toRank]];
    repeatNumVarTemp_ = repeatNumVar_;
    CCU_WHILE(repeatNumVarTemp_ != UINT64_MAX)
    {
        CCU_IF(repeatNumVarTemp_ != UINT64_MAX) { repeatNumVarTemp_ += repeatNumAdd; }

        CCU_IF(flag_ == 1)
        {
            CCU_IF(isBottom_ == 0)
            {
                src.addr += inputSliceStride_;
                dst.addr += inputSliceStride_;
            }
            CCU_IF(isBottom_ == 1)
            {
                src.addr += inputRepeatStride_;
                dst.addr += inputRepeatStride_;
            }
        }
        CCU_IF(flag_ == 0)
        {
            if (axisId_ == 1) {
                src.addr += die0Size_;
                dst.addr += die0Size_;
            }
        }
        sliceSize_ = (axisId_ == 0) ? die0Size_ : die1Size_;
        WriteReduce(*sendTransport, dst, src, sliceSize_, dataType_, reduceOp_, localSignal_, 1);
        LocalWait(localSignal_, (1 << 1) - 1);
        flag_ = 1;
    }
    flag_ = 0;
}

void CcuContextReduceScatterNHR1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuContextReduceScatterNHR1DMem2Mem] CcuContextReduceScatterNHR1DMem2Mem run.");
    InitResources();
    LoadArgs();
    if (linkNum_ == LINK_SIZE) {
        AxisSync(FST_AXIS_ID);
    }
    PreSync();
    DoRepeatReduceScatterNHR();
    PostSync();
    if (linkNum_ == LINK_SIZE) {
        AxisSync(SEC_AXIS_ID);
    }

    HCCL_INFO("[CcuContextReduceScatterNHR1DMem2Mem] CcuContextReduceScatterNHR1DMem2Mem end.");
    return;
}

std::vector<uint64_t> CcuContextReduceScatterNHR1DMem2Mem::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceScatterNHR1D* taskArg = dynamic_cast<const CcuTaskArgReduceScatterNHR1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterNHR1DMem2Mem::taskArg ptr is null"));
    }
    // input & output & buffer地址
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t token = taskArg->token_;
    uint64_t die0Size = taskArg->die0Size_;
    uint64_t die1Size = taskArg->die1Size_;
    uint64_t inputSliceStride = taskArg->inputSliceStride_;
    uint64_t outputSliceStride = taskArg->outputSliceStride_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t repeatNumVar = taskArg->repeatNum_;
    uint64_t isBottom = taskArg->isBottom_;

    HCCL_INFO(
        "[CcuContextReduceScatterNHR1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu],"
        "die0Size[%llu], die1Size[%llu],"
        "inputSliceStride[%llu], outputSliceStride[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu]",
        inputAddr, outputAddr, die0Size, die1Size, inputSliceStride, outputSliceStride, inputRepeatStride,
        outputRepeatStride);

    return {
        inputAddr,         outputAddr,         token,        die0Size, die1Size, inputSliceStride, outputSliceStride,
        inputRepeatStride, outputRepeatStride, repeatNumVar, isBottom};
}
} // namespace Hccl
