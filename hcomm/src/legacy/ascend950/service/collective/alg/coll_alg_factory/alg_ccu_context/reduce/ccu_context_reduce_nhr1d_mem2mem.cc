/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_nhr1d_mem2mem.h"

namespace Hccl {

constexpr uint16_t CKE_IDX_0 = 0; // 后同步
constexpr uint16_t CKE_IDX_1 = 1; // 前同步addr
constexpr uint16_t CKE_IDX_2 = 2; // 前同步token
constexpr uint16_t CKE_IDX_3 = 3; // NHR step同步信号0，用于RS前同步，AG后同步
constexpr uint16_t CKE_IDX_4 = 4; // NHR step同步信号1，用于RS后同步
constexpr uint16_t FST_AXIS_ID = 0;
constexpr uint16_t SEC_AXIS_ID = 1;
constexpr uint16_t RANK_NUM_PER_CKE = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE
constexpr uint16_t OUTPUT_XN_ID = 1;
constexpr uint16_t TOKEN_XN_ID = 2;

CcuContextReduceNHR1DMem2mem::CcuContextReduceNHR1DMem2mem(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceNHR1D* ctxArg = dynamic_cast<const CcuCtxArgReduceNHR1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceNHR1DMem2mem::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rootId_ = ctxArg->rootId_;
    axisId_ = ctxArg->axisId_;
    axisSize_ = ctxArg->axisSize_;
    dimSize_ = ctxArg->dimSize_[0];
    localAxisSignalName_ = "CcuContextAllReduceNHR1DDieSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextAllReduceNHR1DDieSync_" + std::to_string(1 - axisId_);
    stepInfoVector_ = ctxArg->stepInfoVector_;
    indexMap_ = ctxArg->indexMap_;
    localSize_ = indexMap_.size();
    myRankIdx_ = indexMap_.size();
    dataType_ = ctxArg->op_.dataType;
    reduceOp_ = ctxArg->op_.reduceOp;
    signalNum_ = (dimSize_ + RANK_NUM_PER_CKE - 1) / RANK_NUM_PER_CKE; // 每个CKE有16个bit
    HCCL_INFO(
        "[CcuContextReduceNHR1DMem2mem] CtxArg: rankId_[%u], axisId_[%u], axisSize_[%u], dimSize_[%u], localSize_[%u], "
        "signalNum_[%u], dataType[%s], reduceOp[%s]",
        rankId_, axisId_, axisSize_, dimSize_, localSize_, signalNum_, dataType_.Describe().c_str(),
        reduceOp_.Describe().c_str());
}

void CcuContextReduceNHR1DMem2mem::LoadArgs()
{
    Load(input_);
    Load(output_[myRankIdx_]);
    Load(token_[myRankIdx_]);
    Load(isInputOutputEqual_);
    Load(die0Size_);
    Load(die1Size_);
    Load(die0SliceSize_);
    Load(die1SliceSize_);
    Load(die0LastSliceSize_);
    Load(die1LastSliceSize_);
    HCCL_DEBUG("[CcuContextReduceNHR1DMem2mem] LoadArgs run finished");
}

void CcuContextReduceNHR1DMem2mem::InitResources()
{
    die0Size_ = CreateVariable();
    die1Size_ = CreateVariable();
    die0LastSliceSize_ = CreateVariable();
    die1LastSliceSize_ = CreateVariable();
    die0SliceSize_ = CreateVariable();
    die1SliceSize_ = CreateVariable();
    localAxisSignal_ = CreateMaskSignal();
    localSignal_ = CreateMaskSignal();
    if (axisSize_ > 1) {
        ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
        anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);
    }

    input_ = CreateVariable();
    isInputOutputEqual_ = CreateVariable();
    for (uint32_t transportIdx = 0; transportIdx < localSize_; transportIdx++) {
        HCCL_DEBUG("[CcuContextReduceNHR1DMem2mem] MyRank[%u], TransportId[%u]", rankId_, transportIdx);
        CHK_PRT_RET(
            transports[transportIdx] == nullptr,
            HCCL_ERROR("[CcuContextReduceNHR1DMem2mem] Algorithm transport ptr is null"), );
        output_.push_back(
            CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID)); // 获取transport中id=1的Var来传递output
        token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
    }
    output_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    srcMem_ = CreateMemory();
    dstMem_ = CreateMemory();
    HCCL_DEBUG("[CcuContextReduceNHR1DMem2mem] InitResources finished");
}

void CcuContextReduceNHR1DMem2mem::PreSync()
{
    HCCL_DEBUG("[CcuContextReduceNHR1DMem2mem] PreSync start");
    uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
    uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
    for (auto t : transports) {
        WriteVariableWithSignal(*t, output_[localSize_], OUTPUT_XN_ID, selfSignalId + signalNum_ * CKE_IDX_1, selfBit);
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
    HCCL_DEBUG("[CcuContextReduceNHR1DMem2mem] PreSync end");
}

void CcuContextReduceNHR1DMem2mem::PostSync()
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
    for (uint32_t sId = 0; sId < signalNum_; sId++) {
        GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_0, waitBitVector[sId]);
    }
    HCCL_DEBUG("[CcuContextReduceNHR1DMem2mem] PostSync run finished");
}

void CcuContextReduceNHR1DMem2mem::AxisSync(uint32_t signalIndex)
{
    const uint32_t DIE_NUM = 2;
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextReduceNHR1DMem2mem] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_DEBUG("[CcuContextReduceNHR1DMem2mem] AxisSync run finished");
    return;
}

void CcuContextReduceNHR1DMem2mem::DoReduceScatterNHR()
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = 0; i < stepInfoVector_.size() / NHR_NUM; i++) {
        const NHRStepInfo& nhrStepInfo = stepInfoVector_[i];
        DoReduceScatterNHRSingleStep(nhrStepInfo);
    }
}

void CcuContextReduceNHR1DMem2mem::DoReduceScatterNHRSingleStep(const NHRStepInfo& nhrStepInfo)
{
    u32& toRankIdx = indexMap_[nhrStepInfo.toRank];
    u32& fromRankIdx = indexMap_[nhrStepInfo.fromRank];
    u32 sendSliceIdx = 0;
    CcuTransport* sendTransport = transports[toRankIdx];
    CcuTransport* recvTransport = transports[fromRankIdx];
    const std::vector<u32>& sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    srcMem_.token = token_[myRankIdx_];
    dstMem_.token = token_[toRankIdx];
    uint16_t sendSignalId = nhrStepInfo.toRank / RANK_NUM_PER_CKE;
    uint16_t sendBit = 1 << (nhrStepInfo.toRank % RANK_NUM_PER_CKE);
    uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
    uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
    if (nhrStepInfo.step != 0) {
        // 通知fromRank，可以写入
        RemotePost(*recvTransport, selfSignalId + signalNum_ * CKE_IDX_3, selfBit, true);

        // 等待toRank通知其可以写入
        RemoteWait(*sendTransport, sendSignalId + signalNum_ * CKE_IDX_3, sendBit);
    }

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % RANK_NUM_PER_CKE == 0) {
                LocalWait(localSignal_, (1 << RANK_NUM_PER_CKE) - 1);
            }
        }

        if (nhrStepInfo.step == 0) {
            // 只有第0步的源数据从input中取
            srcMem_.addr = input_;
            srcMem_.addr += sliceOffset_[sendSliceIdx];
        } else {
            srcMem_.addr = output_[myRankIdx_];
            srcMem_.addr += sliceOffset_[sendSliceIdx];
        }

        dstMem_.addr = output_[toRankIdx];
        dstMem_.addr += sliceOffset_[sendSliceIdx];

        DoWriteReduceSlice(nhrStepInfo.toRank, srcMem_, dstMem_, sendSliceIdx, i % RANK_NUM_PER_CKE);
    }
    LocalWait(localSignal_, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
    // 通知toRank数据写入完毕
    RemotePost(*sendTransport, selfSignalId + signalNum_ * CKE_IDX_4, selfBit, true);
    // 等待fromRank通知数据写入完毕
    uint16_t recvSignalId = nhrStepInfo.fromRank / RANK_NUM_PER_CKE;
    uint16_t recvBit = 1 << (nhrStepInfo.fromRank % RANK_NUM_PER_CKE);
    RemoteWait(*recvTransport, recvSignalId + signalNum_ * CKE_IDX_4, recvBit);

    HCCL_DEBUG(
        "[DoReduceScatterNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu", rankId_, nhrStepInfo.step,
        nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
}

void CcuContextReduceNHR1DMem2mem::DoWriteReduceSlice(
    const u32& toRank, CcuRep::Memory& src, CcuRep::Memory& dst, const u32& sendSliceIdx, u32 signalIndex)
{
    CcuTransport* sendTransport = transports[indexMap_[toRank]];
    bool islastSlice;

    // 添加 die1 偏移
    if (axisId_ == 1) {
        dst.addr += die0Size_;
        src.addr += die0Size_;
    }

    // allreduce切片的最后一块slice，大小可能不一致
    islastSlice = (sendSliceIdx + 1 == dimSize_);
    const CcuRep::Variable& sliceSize = axisId_ == 0 ? (islastSlice ? die0LastSliceSize_ : die0SliceSize_) :
                                                       (islastSlice ? die1LastSliceSize_ : die1SliceSize_);

    CCU_IF(sliceSize != 0)
    {
        WriteReduce(*sendTransport, dst, src, sliceSize, dataType_, reduceOp_, localSignal_, 1 << signalIndex);
    }
    CCU_IF(sliceSize == 0) { LocalPost(localSignal_, 1 << signalIndex); }
}

void CcuContextReduceNHR1DMem2mem::DoGatherNHR()
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = stepInfoVector_.size() / NHR_NUM; i < stepInfoVector_.size(); i++) {
        const NHRStepInfo& nhrStepInfo = stepInfoVector_[i];
        DoGatherNHRSingleStep(nhrStepInfo);
    }
}

void CcuContextReduceNHR1DMem2mem::DoGatherNHRSingleStep(const NHRStepInfo& nhrStepInfo)
{
    u32& toRankIdx = indexMap_[nhrStepInfo.toRank];
    u32& fromRankIdx = indexMap_[nhrStepInfo.fromRank];
    CcuTransport* recvTransport = transports[fromRankIdx];
    CcuTransport* sendTransport = transports[toRankIdx];
    u32 sendSliceIdx = 0;
    const std::vector<u32>& sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    srcMem_.token = token_[myRankIdx_];
    dstMem_.token = token_[toRankIdx];

    uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
    uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % RANK_NUM_PER_CKE == 0) {
                LocalWait(localSignal_, (1 << RANK_NUM_PER_CKE) - 1);
            }
        }

        srcMem_.addr = output_[myRankIdx_];
        srcMem_.addr += sliceOffset_[sendSliceIdx];

        dstMem_.addr = output_[toRankIdx];
        dstMem_.addr += sliceOffset_[sendSliceIdx];
        DoSendRecvSlice(nhrStepInfo.toRank, srcMem_, dstMem_, sendSliceIdx, i % RANK_NUM_PER_CKE);
    }
    LocalWait(localSignal_, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);

    if (nhrStepInfo.step + 1 != stepInfoVector_.size()) { // 最后一步不需要同步
        // 通知toRank，写入完毕
        RemotePost(*sendTransport, selfSignalId + signalNum_ * CKE_IDX_3, selfBit, true);
        // 等待fromRank通知写入完毕
        uint16_t recvSignalId = nhrStepInfo.fromRank / RANK_NUM_PER_CKE;
        uint16_t recvBit = 1 << (nhrStepInfo.fromRank % RANK_NUM_PER_CKE);
        RemoteWait(*recvTransport, recvSignalId + signalNum_ * CKE_IDX_3, recvBit);
    }

    HCCL_DEBUG(
        "[DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu", rankId_, nhrStepInfo.step,
        nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
}

void CcuContextReduceNHR1DMem2mem::DoSendRecvSlice(
    const u32& toRank, CcuRep::Memory& src, CcuRep::Memory& dst, const u32& sendSliceIdx, u32 signalIndex)
{
    CcuTransport* sendTransport = transports[indexMap_[toRank]];
    bool islastSlice;

    // 添加 die1 偏移
    if (axisId_ == 1) {
        src.addr += die0Size_;
        dst.addr += die0Size_;
    }

    islastSlice = (sendSliceIdx + 1 == dimSize_);
    const CcuRep::Variable& sliceSize = axisId_ == 0 ? (islastSlice ? die0LastSliceSize_ : die0SliceSize_) :
                                                       (islastSlice ? die1LastSliceSize_ : die1SliceSize_);

    CCU_IF(sliceSize != 0) { Write(*sendTransport, dst, src, sliceSize, localSignal_, 1 << signalIndex); }
    CCU_IF(sliceSize == 0) { LocalPost(localSignal_, 1 << signalIndex); }
}

void CcuContextReduceNHR1DMem2mem::LocalCopySlices()
{
    u32 nonTxSliceIdx = 0;
    CcuRep::Variable tmpSliceOffset = CreateVariable();
    tmpSliceOffset = 0;

    for (u64 i = 0; i < dimSize_; i++) {
        sliceOffset_.push_back(CreateVariable());
        sliceOffset_[i] = tmpSliceOffset;
        tmpSliceOffset += axisId_ == 0 ? die0SliceSize_ : die1SliceSize_;
    }

    // 当input == output时，不需要拷贝
    CCU_IF(isInputOutputEqual_ == 0)
    {
        // 将step0中不需要写的slice，拷贝到本rank的output中
        const NHRStepInfo& nhrStepInfo = stepInfoVector_[0];
        const std::vector<u32>& nonTxSliceIdxList = GetNonTxSliceIdxs(nhrStepInfo.txSliceIdxs);
        for (u32 i = 0; i < nonTxSliceIdxList.size(); i++) {
            nonTxSliceIdx = nonTxSliceIdxList[i];

            if (i != 0) {
                if (i % RANK_NUM_PER_CKE == 0) {
                    LocalWait(localSignal_, (1 << RANK_NUM_PER_CKE) - 1);
                }
            }

            srcMem_.addr = input_;
            srcMem_.addr += sliceOffset_[nonTxSliceIdx];
            srcMem_.token = token_[myRankIdx_];

            dstMem_.addr = output_[myRankIdx_];
            dstMem_.addr += sliceOffset_[nonTxSliceIdx];
            dstMem_.token = token_[myRankIdx_];
            DoLocalCopySlice(srcMem_, dstMem_, nonTxSliceIdx, i);
        }
        LocalWait(localSignal_, (1 << (nonTxSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
    }
}

std::vector<u32> CcuContextReduceNHR1DMem2mem::GetNonTxSliceIdxs(const std::vector<u32>& txSliceIdxs) const
{
    std::vector<bool> isTx(dimSize_, false);
    for (u32 idx : txSliceIdxs) {
        if (idx < dimSize_) {
            isTx[idx] = true;
        }
    }

    std::vector<u32> nonTxSliceIdxs;
    for (u32 idx = 0; idx < dimSize_; ++idx) {
        if (!isTx[idx]) {
            nonTxSliceIdxs.push_back(idx);
        }
    }

    return nonTxSliceIdxs;
}

void CcuContextReduceNHR1DMem2mem::DoLocalCopySlice(
    CcuRep::Memory& src, CcuRep::Memory& dst, const u32& copySliceIdx, u32 signalIndex)
{
    bool islastSlice;
    // 添加 die1 偏移
    if (axisId_ == 1) {
        dst.addr += die0Size_;
        src.addr += die0Size_;
    }

    islastSlice = (copySliceIdx + 1 == dimSize_);
    const CcuRep::Variable& sliceSize = axisId_ == 0 ? (islastSlice ? die0LastSliceSize_ : die0SliceSize_) :
                                                       (islastSlice ? die1LastSliceSize_ : die1SliceSize_);

    CCU_IF(sliceSize != 0) { LocalCopy(dst, src, sliceSize, localSignal_, 1 << signalIndex); }
    CCU_IF(sliceSize == 0) { LocalPost(localSignal_, 1 << signalIndex); }
}

void CcuContextReduceNHR1DMem2mem::Algorithm()
{
    HCCL_DEBUG("[CcuContextReduceNHR1DMem2mem] AllReduceNHR1D run");

    InitResources();
    LoadArgs();
    if (axisSize_ > 1) {
        AxisSync(FST_AXIS_ID);
    }
    LocalCopySlices();
    PreSync();
    DoReduceScatterNHR();
    DoGatherNHR();
    PostSync();
    if (axisSize_ > 1) {
        AxisSync(SEC_AXIS_ID);
    }

    HCCL_DEBUG("[CcuContextReduceNHR1DMem2mem] AllReduceNHR1D end");
    return;
}

std::vector<uint64_t> CcuContextReduceNHR1DMem2mem::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceNHR1D* taskArg = dynamic_cast<const CcuTaskArgReduceNHR1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceNHR1DMem2mem::taskArg ptr is null"));
    }
    // input&output&buffer地址
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t token = taskArg->token_;
    uint64_t isInputOutputEqual = taskArg->isInputOutputEqual_;
    uint64_t die0Size = taskArg->die0Size_;
    uint64_t die1Size = taskArg->die1Size_;
    uint64_t die0SliceSize = taskArg->die0SliceSize_;
    uint64_t die1SliceSize = taskArg->die1SliceSize_;
    uint64_t die0LastSliceSize = taskArg->die0LastSliceSize_;
    uint64_t die1LastSliceSize = taskArg->die1LastSliceSize_;

    HCCL_INFO(
        "[CcuContextReduceNHR1DMem2mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "die0Size[%llu], die1Size[%llu], die0SliceSize[%llu], die1SliceSize[%llu],"
        "die0LastSliceSize[%llu], die1LastSliceSize[%llu], isInputOutputEqual is [%lu]",
        inputAddr, outputAddr, die0Size, die1Size, die0SliceSize, die1SliceSize, die0LastSliceSize, die1LastSliceSize,
        isInputOutputEqual);

    return {inputAddr, outputAddr,    token,         isInputOutputEqual, die0Size,
            die1Size,  die0SliceSize, die1SliceSize, die0LastSliceSize,  die1LastSliceSize};
}
} // namespace Hccl
