/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_reduce_nhr1d_mem2mem.h"

namespace Hccl {

constexpr uint16_t RANK_NUM_PER_CKE = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE
constexpr uint16_t OUTPUT_XN_ID = 1;
constexpr uint16_t TOKEN_XN_ID = 2;
constexpr uint16_t CKE_IDX_0 = 0; // 后同步
constexpr uint16_t CKE_IDX_1 = 1; // 前同步addr
constexpr uint16_t CKE_IDX_2 = 2; // 前同步token
constexpr uint16_t CKE_IDX_3 = 3; // NHR step同步信号0，用于RS前同步，AG后同步
constexpr uint16_t CKE_IDX_4 = 4; // NHR step同步信号1，用于RS后同步
constexpr uint16_t FST_AXIS_ID = 0;
constexpr uint16_t SEC_AXIS_ID = 1;

CcuContextAllReduceNHR1D::CcuContextAllReduceNHR1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllReduceNHR1D* ctxArg = dynamic_cast<const CcuCtxArgAllReduceNHR1D*>(&arg);
    rankId_ = ctxArg->rankId_;
    axisId_ = ctxArg->axisId_;
    axisSize_ = ctxArg->axisSize_;
    dimSize_ = ctxArg->dimSize_[0];
    stepInfoVector_ = ctxArg->stepInfoVector_;
    indexMap_ = ctxArg->indexMap_;
    localSize_ = indexMap_.size();
    myRankIdx_ = indexMap_.size();
    dataType_ = ctxArg->op_.dataType;
    reduceOp_ = ctxArg->op_.reduceOp;
    signalNum_ = (dimSize_ + RANK_NUM_PER_CKE - 1) / RANK_NUM_PER_CKE; // 每个CKE有16个bit
    HCCL_INFO(
        "[CcuContextAllReduceNHR1D] CtxArg: rankId_[%u], axisId_[%u], axisSize_[%u], dimSize_[%u], localSize_[%u], "
        "signalNum_[%u], dataType[%s], reduceOp[%s]",
        rankId_, axisId_, axisSize_, dimSize_, localSize_, signalNum_, dataType_.Describe().c_str(),
        reduceOp_.Describe().c_str());
}

void CcuContextAllReduceNHR1D::LoadArgs()
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
    HCCL_DEBUG("[CcuContextAllReduceNHR1D] LoadArgs run finished");
}

void CcuContextAllReduceNHR1D::InitResources()
{
    isInputOutputEqual_ = CreateVariable();
    die0SliceSize_ = CreateVariable();
    die1SliceSize_ = CreateVariable();
    die0LastSliceSize_ = CreateVariable();
    die1LastSliceSize_ = CreateVariable();
    die0Size_ = CreateVariable();
    die1Size_ = CreateVariable();
    localSignal_ = CreateMaskSignal();

    input_ = CreateVariable();
    for (uint32_t transportIdx = 0; transportIdx < localSize_; transportIdx++) {
        HCCL_DEBUG("[CcuContextAllReduceNHR1D] MyRank[%u], TransportId[%u]", rankId_, transportIdx);
        CHK_PRT_RET(
            transports[transportIdx] == nullptr,
            HCCL_ERROR("[CcuContextAllReduceNHR1D] Algorithm transport ptr is null"), );
        output_.push_back(
            CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID)); // 获取transport中id=1的Var来传递output
        token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
    }
    output_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    srcMem_ = CreateMemory();
    dstMem_ = CreateMemory();
    HCCL_DEBUG("[CcuContextAllReduceNHR1D] InitResources finished");
}

void CcuContextAllReduceNHR1D::PreSync()
{
    HCCL_DEBUG("[CcuContextAllReduceNHR1D] PreSync start");
    uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
    uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
    std::vector<uint16_t> waitBitVector(signalNum_, 0);
    for (auto t : transports) {
        WriteVariableWithSignal(*t, output_[localSize_], OUTPUT_XN_ID, selfSignalId + signalNum_ * CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*t, token_[localSize_], TOKEN_XN_ID, selfSignalId + signalNum_ * CKE_IDX_2, selfBit);
    }
    for (auto& pair : indexMap_) {
        uint16_t pairBit = 1 << (pair.first % RANK_NUM_PER_CKE);
        uint16_t pairSignalId = pair.first / RANK_NUM_PER_CKE;
        waitBitVector[pairSignalId] = waitBitVector[pairSignalId] | pairBit;
    }
    for (uint16_t sId = 0; sId < waitBitVector.size(); sId++) {
        GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_1, waitBitVector[sId]);
        GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_2, waitBitVector[sId]);
    }
    HCCL_DEBUG("[CcuContextAllReduceNHR1D] PreSync end");
}

void CcuContextAllReduceNHR1D::PostSync()
{
    uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
    uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
    std::vector<uint16_t> waitBitVector(signalNum_, 0);
    for (auto& t : transports) {
        RemotePost(*t, selfSignalId + signalNum_ * CKE_IDX_0, selfBit);
    }
    for (auto& pair : indexMap_) {
        uint16_t pairSignalId = pair.first / RANK_NUM_PER_CKE;
        uint16_t pairBit = 1 << (pair.first % RANK_NUM_PER_CKE);
        waitBitVector[pairSignalId] = waitBitVector[pairSignalId] | pairBit;
    }
    for (uint32_t sId = 0; sId < signalNum_; sId++) {
        GroupWait(*transportGroup, sId + signalNum_ * CKE_IDX_0, waitBitVector[sId]);
    }
    HCCL_DEBUG("[CcuContextAllReduceNHR1D] PostSync run finished");
}

void CcuContextAllReduceNHR1D::AxisSync(uint32_t signalIndex)
{
    const uint32_t DIE_NUM = 2;
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextAllReduceNHR1D] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_DEBUG("[CcuContextAllReduceNHR1D] AxisSync run finished");
    return;
}

void CcuContextAllReduceNHR1D::DoReduceScatterNHR()
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = 0; i < stepInfoVector_.size() / NHR_NUM; i++) {
        const NHRStepInfo& nhrStepInfo = stepInfoVector_[i];
        DoReduceScatterNHRSingleStep(nhrStepInfo);
    }
}

void CcuContextAllReduceNHR1D::DoReduceScatterNHRSingleStep(const NHRStepInfo& nhrStepInfo)
{
    u32 sendSliceIdx = 0;
    u32& fromRankIdx = indexMap_[nhrStepInfo.fromRank];
    u32& toRankIdx = indexMap_[nhrStepInfo.toRank];
    const std::vector<u32>& sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    srcMem_.token = token_[myRankIdx_];
    dstMem_.token = token_[toRankIdx];
    CcuTransport* sendTransport = transports[toRankIdx];
    CcuTransport* recvTransport = transports[fromRankIdx];

    uint16_t selfSignalId = rankId_ / RANK_NUM_PER_CKE;
    uint16_t selfBit = 1 << (rankId_ % RANK_NUM_PER_CKE);
    uint16_t sendSignalId = nhrStepInfo.toRank / RANK_NUM_PER_CKE;
    uint16_t sendBit = 1 << (nhrStepInfo.toRank % RANK_NUM_PER_CKE);
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
            HCCL_INFO("[CcuContextAllReduceNHR1D] nhrStepInfo 0.");
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
    uint16_t recvBit = 1 << (nhrStepInfo.fromRank % RANK_NUM_PER_CKE);
    uint16_t recvSignalId = nhrStepInfo.fromRank / RANK_NUM_PER_CKE;
    RemoteWait(*recvTransport, recvSignalId + signalNum_ * CKE_IDX_4, recvBit);
    HCCL_DEBUG(
        "[DoReduceScatterNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu", rankId_, nhrStepInfo.step,
        nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
}

void CcuContextAllReduceNHR1D::DoWriteReduceSlice(
    const u32& toRank, CcuRep::Memory& src, CcuRep::Memory& dst, const u32& sendSliceIdx, u32 signalIndex)
{
    bool islastSlice;
    // 添加 die1 偏移
    if (axisId_ == 1) {
        dst.addr += die0Size_;
        src.addr += die0Size_;
    }
    CcuTransport* sendTransport = transports[indexMap_[toRank]];
    islastSlice = (sendSliceIdx + 1 == dimSize_);

    // allreduce切片的最后一块slice，大小可能不一致
    const CcuRep::Variable& sliceSize = axisId_ == 0 ? (islastSlice ? die0LastSliceSize_ : die0SliceSize_) :
                                                       (islastSlice ? die1LastSliceSize_ : die1SliceSize_);
    CCU_IF(sliceSize != 0)
    {
        WriteReduce(*sendTransport, dst, src, sliceSize, dataType_, reduceOp_, localSignal_, 1 << signalIndex);
    }
    CCU_IF(sliceSize == 0) { LocalPost(localSignal_, 1 << signalIndex); }
}

void CcuContextAllReduceNHR1D::DoAllGatherNHR()
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = stepInfoVector_.size() / NHR_NUM; i < stepInfoVector_.size(); i++) {
        const NHRStepInfo& nhrStepInfo = stepInfoVector_[i];
        DoAllGatherNHRSingleStep(nhrStepInfo);
    }
}

void CcuContextAllReduceNHR1D::DoAllGatherNHRSingleStep(const NHRStepInfo& nhrStepInfo)
{
    u32 sendSliceIdx = 0;
    u32& toRankIdx = indexMap_[nhrStepInfo.toRank];
    u32& fromRankIdx = indexMap_[nhrStepInfo.fromRank];
    CcuTransport* sendTransport = transports[toRankIdx];
    CcuTransport* recvTransport = transports[fromRankIdx];
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
        dstMem_.addr = output_[toRankIdx];
        dstMem_.addr += sliceOffset_[sendSliceIdx];
        srcMem_.addr += sliceOffset_[sendSliceIdx];
        DoSendRecvSlice(nhrStepInfo.toRank, srcMem_, dstMem_, sendSliceIdx, i % RANK_NUM_PER_CKE);
    }
    LocalWait(localSignal_, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);

    if (nhrStepInfo.step + 1 != stepInfoVector_.size()) { // 最后一步不需要同步
        // 通知toRank，写入完毕
        RemotePost(*sendTransport, selfSignalId + signalNum_ * CKE_IDX_3, selfBit, true);
        // 等待fromRank通知写入完毕
        uint16_t recvBit = 1 << (nhrStepInfo.fromRank % RANK_NUM_PER_CKE);
        uint16_t recvSignalId = nhrStepInfo.fromRank / RANK_NUM_PER_CKE;
        RemoteWait(*recvTransport, recvSignalId + signalNum_ * CKE_IDX_3, recvBit);
    }

    HCCL_DEBUG(
        "[DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu", rankId_, nhrStepInfo.step,
        nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
}

void CcuContextAllReduceNHR1D::DoSendRecvSlice(
    const u32& toRank, CcuRep::Memory& src, CcuRep::Memory& dst, const u32& sendSliceIdx, u32 signalIndex)
{
    bool islastSlice;
    CcuTransport* sendTransport = transports[indexMap_[toRank]];
    islastSlice = (sendSliceIdx + 1 == dimSize_);
    // 添加 die1 偏移
    if (axisId_ == 1) {
        dst.addr += die0Size_;
        src.addr += die0Size_;
    }
    const CcuRep::Variable& sliceSize = axisId_ == 0 ? (islastSlice ? die0LastSliceSize_ : die0SliceSize_) :
                                                       (islastSlice ? die1LastSliceSize_ : die1SliceSize_);
    CCU_IF(sliceSize != 0) { Write(*sendTransport, dst, src, sliceSize, localSignal_, 1 << signalIndex); }
    CCU_IF(sliceSize == 0) { LocalPost(localSignal_, 1 << signalIndex); }
}

void CcuContextAllReduceNHR1D::LocalCopySlices()
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
            dstMem_.addr = output_[myRankIdx_];
            srcMem_.addr += sliceOffset_[nonTxSliceIdx];
            dstMem_.addr += sliceOffset_[nonTxSliceIdx];
            srcMem_.token = token_[myRankIdx_];
            dstMem_.token = token_[myRankIdx_];
            DoLocalCopySlice(srcMem_, dstMem_, nonTxSliceIdx, i);
        }
        LocalWait(localSignal_, (1 << (nonTxSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
    }
}

std::vector<u32> CcuContextAllReduceNHR1D::GetNonTxSliceIdxs(const std::vector<u32>& txSliceIdxs) const
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

void CcuContextAllReduceNHR1D::DoLocalCopySlice(
    CcuRep::Memory& src, CcuRep::Memory& dst, const u32& copySliceIdx, u32 signalIndex)
{
    bool islastSlice;
    // 添加 die1 偏移
    if (axisId_ == 1) {
        src.addr += die0Size_;
        dst.addr += die0Size_;
    }

    islastSlice = (copySliceIdx + 1 == dimSize_);
    const CcuRep::Variable& sliceSize = axisId_ == 0 ? (islastSlice ? die0LastSliceSize_ : die0SliceSize_) :
                                                       (islastSlice ? die1LastSliceSize_ : die1SliceSize_);
    CCU_IF(sliceSize == 0) { LocalPost(localSignal_, 1 << signalIndex); }

    CCU_IF(sliceSize != 0) { LocalCopy(dst, src, sliceSize, localSignal_, 1 << signalIndex); }
}

void CcuContextAllReduceNHR1D::Algorithm()
{
    HCCL_DEBUG("[CcuContextAllReduceNHR1D] AllReduceNHR1D run");

    InitResources();
    LoadArgs();
    LocalCopySlices();
    PreSync();
    DoReduceScatterNHR();
    DoAllGatherNHR();
    PostSync();

    HCCL_DEBUG("[CcuContextAllReduceNHR1D] AllReduceNHR1D end");
    return;
}

std::vector<uint64_t> CcuContextAllReduceNHR1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllReduceNHR1D* taskArg = dynamic_cast<const CcuTaskArgAllReduceNHR1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceNHR1D::taskArg ptr is null"));
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
        "[CcuContextAllReduceNHR1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "die0Size[%llu], die1Size[%llu], die0SliceSize[%llu], die1SliceSize[%llu],"
        "die0LastSliceSize[%llu], die1LastSliceSize[%llu]",
        inputAddr, outputAddr, die0Size, die1Size, die0SliceSize, die1SliceSize, die0LastSliceSize, die1LastSliceSize);

    return {inputAddr, outputAddr,    token,         isInputOutputEqual, die0Size,
            die1Size,  die0SliceSize, die1SliceSize, die0LastSliceSize,  die1LastSliceSize};
}
} // namespace Hccl
