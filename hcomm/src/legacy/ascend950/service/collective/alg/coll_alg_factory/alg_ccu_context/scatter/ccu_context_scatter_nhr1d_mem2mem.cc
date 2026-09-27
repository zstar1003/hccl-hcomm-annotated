/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_scatter_nhr1d_mem2mem.h"
namespace Hccl {
constexpr uint16_t SCRATCH_XN_ID = 1;
constexpr uint16_t TOKEN_XN_ID = 2;
constexpr uint16_t CKE_IDX_0 = 0; // 后同步
constexpr uint16_t CKE_IDX_1 = 1; // 前同步addr
constexpr uint16_t CKE_IDX_2 = 2; // 前同步token
constexpr uint16_t CKE_IDX_3 = 3; // NHR step同步信号，用于scatter后同步
constexpr uint16_t FST_AXIS_ID = 0;
constexpr uint16_t SEC_AXIS_ID = 1;
constexpr uint16_t RANK_NUM_PER_CKE = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE

CcuContextScatterNHR1DMem2Mem::CcuContextScatterNHR1DMem2Mem(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgScatterNHR1D* ctxArg = dynamic_cast<const CcuCtxArgScatterNHR1D*>(&arg);
    rankId_ = ctxArg->rankId_;
    rootId_ = ctxArg->rootId_;
    axisId_ = ctxArg->axisId_;
    axisSize_ = ctxArg->axisSize_;
    dimSize_ = ctxArg->dimSize_[0];
    localAxisSignalName_ = "CcuContextScatterNHR1DMem2MemDieSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextScatterNHR1DMem2MemDieSync_" + std::to_string(1 - axisId_);
    stepInfoVector_ = ctxArg->stepInfoVector_;
    indexMap_ = ctxArg->indexMap_;
    localSize_ = indexMap_.size();
    myRankIdx_ = indexMap_.size();
    dataType_ = ctxArg->op_.dataType;
    signalNum_ = (dimSize_ + RANK_NUM_PER_CKE - 1) / RANK_NUM_PER_CKE; // 每个CKE有16个bit
    HCCL_INFO(
        "[CcuContextScatterNHR1DMem2Mem] CtxArg: rankId_[%u], rootId_[%u], axisId_[%u], axisSize_[%u], dimSize_[%u], "
        "localSize_[%u], "
        "signalNum_[%u], dataType[%s]",
        rankId_, rootId_, axisId_, axisSize_, dimSize_, localSize_, signalNum_, dataType_.Describe().c_str());
}

void CcuContextScatterNHR1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_);
    Load(token_[myRankIdx_]);
    Load(scratch_[myRankIdx_]);
    Load(die0Size_);
    Load(die1Size_);
    Load(inputSliceStride_);
    Load(curScratchStride_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(repeatNumVar_);
    Load(isOutputScratch_);
    HCCL_INFO("[CcuContextScatterNHR1DMem2Mem] LoadArgs run finished");
}

void CcuContextScatterNHR1DMem2Mem::InitResources()
{
    die0Size_ = CreateVariable();
    die1Size_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    curScratchStride_ = CreateVariable();
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    repeatNumVar_ = CreateVariable();
    repeatNumVarTemp_ = CreateVariable();
    repeatTimeflag_ = CreateVariable();
    curInputOffset_ = CreateVariable();
    curScratchOffset_ = CreateVariable();
    cursliceSize_ = CreateVariable();
    isOutputScratch_ = CreateVariable();
    localSignal_ = CreateMaskSignal();
    if (axisSize_ > 1) {
        localAxisSignal_ = CreateMaskSignal();
        ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
        anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);
    }

    input_ = CreateVariable();
    for (uint32_t transportIdx = 0; transportIdx < localSize_; transportIdx++) {
        HCCL_INFO("[CcuContextScatterNHR1DMem2Mem] MyRank[%u], TransportId[%u]", rankId_, transportIdx);
        CHK_PRT_RET(
            transports[transportIdx] == nullptr,
            HCCL_INFO("[CcuContextScatterNHR1DMem2Mem] Algorithm transport ptr is null"), );
        scratch_.push_back(CreateVariable((*transports[transportIdx]), SCRATCH_XN_ID)); // 存放
        token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
    }
    scratch_.push_back(CreateVariable()); // 本端放最后
    token_.push_back(CreateVariable());
    output_ = CreateVariable();
    srcMem_ = CreateMemory();
    dstMem_ = CreateMemory();
    HCCL_INFO("[CcuContextScatterNHR1DMem2Mem] InitResources finished");
}

void CcuContextScatterNHR1DMem2Mem::PreSync()
{
    HCCL_INFO("[CcuContextScatterNHR1DMem2Mem] PreSync start");
    uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
    uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
    for (auto t : transports) {
        WriteVariableWithSignal(
            *t, scratch_[localSize_], SCRATCH_XN_ID, selfSignalId + signalNum_ * CKE_IDX_1, selfBit);
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
    HCCL_INFO("[CcuContextScatterNHR1DMem2Mem] PreSync end");
}

void CcuContextScatterNHR1DMem2Mem::PostSync()
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
    HCCL_INFO("[CcuContextScatterNHR1DMem2Mem] PostSync run finished");
}

void CcuContextScatterNHR1DMem2Mem::AxisSync(uint32_t signalIndex)
{
    const uint32_t DIE_NUM = 2;
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextScatterNHR1DMem2Mem] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_INFO("[CcuContextScatterNHR1DMem2Mem] AxisSync run finished");
    return;
}

void CcuContextScatterNHR1DMem2Mem::DoScatterNHR()
{
    curInputOffset_ = 0;   // input偏移
    curScratchOffset_ = 0; // scratch偏移
    for (u64 i = 0; i < dimSize_; i++) {
        inputOffset_.push_back(CreateVariable());
        inputOffset_[i] = curInputOffset_;
        curInputOffset_ += inputSliceStride_;
    }
    for (u64 i = 0; i < dimSize_; i++) {
        ScratchOffset_.push_back(CreateVariable());
        ScratchOffset_[i] = curScratchOffset_;
        curScratchOffset_ += curScratchStride_;
    }
    // NHR
    for (u64 i = 0; i < stepInfoVector_.size(); i++) {
        const NHRStepInfo& nhrStepInfo = stepInfoVector_[i];
        DoScatterNHRSingleStep(nhrStepInfo);
    }
    // scratch->output
    if (rankId_ == rootId_) {
        srcMem_.addr = input_;
        srcMem_.addr += inputOffset_[rankId_];
    } else {
        srcMem_.addr = scratch_[myRankIdx_];
        srcMem_.addr += ScratchOffset_[rankId_];
    }
    dstMem_.addr = output_;
    srcMem_.token = token_[myRankIdx_];
    dstMem_.token = token_[myRankIdx_];
    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd = 1;
    repeatTimeflag_ = 0;
    CCU_WHILE(repeatNumVar_ != UINT64_MAX)
    {
        repeatNumVar_ += repeatNumAdd;
        CCU_IF(repeatTimeflag_ != 0)
        {
            if (rankId_ == rootId_) {
                srcMem_.addr += inputRepeatStride_;
            } else {
                srcMem_.addr += outputRepeatStride_;
            }
            dstMem_.addr += outputRepeatStride_;
        }
        CCU_IF(repeatTimeflag_ == 0)
        {
            if (axisId_ == 1) {
                srcMem_.addr += die0Size_;
                dstMem_.addr += die0Size_;
            }
        }
        cursliceSize_ = (axisId_ == 0) ? die0Size_ : die1Size_;
        {
            CCU_IF(isOutputScratch_ == 1)
            {
                if (rootId_ != 0 && rankId_ == 0) {
                    LocalPost(localSignal_, 1 << rankId_);
                } else {
                    LocalCopy(dstMem_, srcMem_, cursliceSize_, localSignal_, 1 << rankId_);
                }
            }
            CCU_IF(isOutputScratch_ != 1) { LocalCopy(dstMem_, srcMem_, cursliceSize_, localSignal_, 1 << rankId_); }
            LocalWait(localSignal_, 1 << rankId_);
        }
        repeatTimeflag_ = 1;
    }
}

void CcuContextScatterNHR1DMem2Mem::DoScatterNHRSingleStep(const NHRStepInfo& nhrStepInfo)
{
    const std::vector<u32>& sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const std::vector<u32>& recvSliceIdxList = nhrStepInfo.rxSliceIdxs;
    if (recvSliceIdxList.size() != 0) {
        u32& fromRankIdx = indexMap_[nhrStepInfo.fromRank];
        CcuTransport* recvTransport = transports[fromRankIdx];
        uint16_t recvSignalId = nhrStepInfo.fromRank / RANK_NUM_PER_CKE;
        uint16_t recvBit = 1 << (nhrStepInfo.fromRank % RANK_NUM_PER_CKE);
        RemoteWait(*recvTransport, recvSignalId + signalNum_ * CKE_IDX_3,
                   recvBit); // 后同步，等待通知写入完毕
    }
    if (sendSliceIdxList.size() != 0) {
        u32& toRankIdx = indexMap_[nhrStepInfo.toRank];
        u32 sendSliceIdx = 0;
        uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
        uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
        CcuTransport* sendTransport = transports[toRankIdx];
        for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
            sendSliceIdx = sendSliceIdxList[i];
            if (i != 0) {
                if (i % RANK_NUM_PER_CKE == 0) {
                    LocalWait(localSignal_, (1 << RANK_NUM_PER_CKE) - 1);
                }
            }
            if (rankId_ == rootId_) { // root节点的源数据从input中取
                srcMem_.addr = input_;
                srcMem_.addr += inputOffset_[sendSliceIdx];
            } else {
                srcMem_.addr = scratch_[myRankIdx_];
                srcMem_.addr += ScratchOffset_[sendSliceIdx];
            }
            srcMem_.token = token_[myRankIdx_];
            dstMem_.token = token_[toRankIdx];
            dstMem_.addr = scratch_[toRankIdx];
            dstMem_.addr += ScratchOffset_[sendSliceIdx];
            DoSendRecvSlice(nhrStepInfo.toRank, srcMem_, dstMem_, i % RANK_NUM_PER_CKE);
        }
        RemotePost(*sendTransport, selfSignalId + signalNum_ * CKE_IDX_3, selfBit, true); // 后同步,通知写入完毕
    }
    HCCL_INFO(
        "[DoScatterNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu", rankId_, nhrStepInfo.step,
        nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
}

void CcuContextScatterNHR1DMem2Mem::DoSendRecvSlice(
    const u32& toRank, CcuRep::Memory& src, CcuRep::Memory& dst, u32 signalIndex)
{
    CcuTransport* sendTransport = transports[indexMap_[toRank]];
    CcuRep::Variable repeatNumAdd2 = CreateVariable();
    repeatNumAdd2 = 1;
    repeatTimeflag_ = 0;
    repeatNumVarTemp_ = repeatNumVar_;
    CCU_WHILE(repeatNumVarTemp_ != UINT64_MAX)
    {
        repeatNumVarTemp_ += repeatNumAdd2;
        CCU_IF(repeatTimeflag_ == 1)
        {
            if (rankId_ == rootId_) {
                src.addr += inputRepeatStride_;
            } else {
                src.addr += outputRepeatStride_;
            }
            dst.addr += outputRepeatStride_;
        }
        CCU_IF(repeatTimeflag_ == 0)
        {
            if (axisId_ == 1) {
                src.addr += die0Size_;
                dst.addr += die0Size_;
            }
        }
        cursliceSize_ = (axisId_ == 0) ? die0Size_ : die1Size_;
        Write(*sendTransport, dst, src, cursliceSize_, localSignal_, 1 << signalIndex);
        LocalWait(localSignal_, 1 << signalIndex);
        repeatTimeflag_ = 1;
    }
}

void CcuContextScatterNHR1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuContextScatterNHR1DMem2Mem] ScatterNHR1D run");
    InitResources();
    LoadArgs();
    if (axisSize_ > 1)
        AxisSync(FST_AXIS_ID);
    PreSync();
    DoScatterNHR();
    PostSync();

    if (axisSize_ > 1)
        AxisSync(SEC_AXIS_ID);

    HCCL_INFO("[CcuContextScatterNHR1DMem2Mem] ScatterNHR1D end");
    return;
}

std::vector<uint64_t> CcuContextScatterNHR1DMem2Mem::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgScatterNHR1D* taskArg = dynamic_cast<const CcuTaskArgScatterNHR1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextScatterNHR1DMem2Mem::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t token = taskArg->token_;
    uint64_t scratchAddr = taskArg->scratchAddr_;
    uint64_t die0Size = taskArg->die0Size_;
    uint64_t die1Size = taskArg->die1Size_;
    uint64_t inputSliceStride = taskArg->inputSliceStride_;
    uint64_t curScratchStride = taskArg->sliceSize_ * taskArg->repeatNum_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t repeatNumVar = taskArg->repeatNumVar_;
    uint64_t isOutputScratch = taskArg->isOutputScratch_;

    HCCL_INFO(
        "[CcuContextScatterNHR1DMem2Mem] TaskArgs: rankId_[%llu], inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu],"
        "die0Size[%llu], die1Size[%llu], inputSliceStride[%llu], curScratchStride[%llu],"
        "inputRepeatStride[%llu], outputRepeatStride[%llu],repeatNumVar[%llu]",
        rankId_, inputAddr, outputAddr, scratchAddr, die0Size, die1Size, inputSliceStride, curScratchStride,
        inputRepeatStride, outputRepeatStride, repeatNumVar);
    return {inputAddr,          outputAddr,       token,
            scratchAddr,        die0Size,         die1Size,
            inputSliceStride,   curScratchStride, inputRepeatStride,
            outputRepeatStride, repeatNumVar,     isOutputScratch};
}
} // namespace Hccl
