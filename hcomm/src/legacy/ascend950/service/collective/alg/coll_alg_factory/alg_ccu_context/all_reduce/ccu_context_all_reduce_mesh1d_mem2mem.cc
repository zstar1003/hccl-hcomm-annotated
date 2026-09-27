/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_reduce_mesh1d_mem2mem.h"
#include "ccu_instruction_all_reduce_mesh1d_mem2mem.h"
#include "ccu_assist.h"
namespace Hccl {
constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int SCRATCH_XN_ID = 2;
constexpr int TOKEN_XN_ID = 3;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

using CurrentCtxArg = CcuCtxArgAllReduceMeshMem2Mem1D;
using CurrentTaskArg = CcuTaskArgAllReduceMeshMem2Mem1D;

CcuContextAllReduceMeshMem2Mem1D::CcuContextAllReduceMeshMem2Mem1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    HCCL_DEBUG("[CcuContextAllReduceMeshMem2Mem1D] Enter Constructor.");
    const CurrentCtxArg* ctxArg = dynamic_cast<const CurrentCtxArg*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMeshMem2Mem1D::ctxArg ptr is null"));
    }
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    reduceOp_ = ctxArg->op_.reduceOp;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextAllReduceMeshMem2Mem1D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    CHK_PRT_THROW(
        ctxArg->dimSize_[0] == 0, HCCL_ERROR("[CcuContextAllReduceMeshMem2Mem1D] ctxArg->dimSize_[0] is zero"),
        InvalidParamsException, "ctxArg->dimSize_[0] is invalid");
    if (ctxArg->dimSize_.size() > 0) {
        rankSize_ = ctxArg->dimSize_[0];
    }
    HCCL_INFO(
        "[CcuContextAllReduceMeshMem2Mem1D] Init, CtxArgs are rankId[%u], rankSize_[%llu], dataType[%s], "
        "outputDataType[%s], reduceOp[%s]",
        rankId_, rankSize_, dataType_.Describe().c_str(), outputDataType_.Describe().c_str(),
        reduceOp_.Describe().c_str());
}

void CcuContextAllReduceMeshMem2Mem1D::InitResource()
{
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMeshMem2Mem1D transports is empty"));
    }
    HCCL_INFO("[CcuContextAllReduceMeshMem2Mem1D]transports.size: [%zu]", transports.size());
    uint16_t transportIdx = 0;
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            scratch_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG(
                "[CcuContextAllReduceMeshMem2Mem1D] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_THROW(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextAllReduceMeshMem2Mem1D] [InitResource] transports[%u] is nullptr", transportIdx),
                NullPtrException, "transport is null");
            input_.push_back(CreateVariable((*transports[transportIdx]), INPUT_XN_ID));
            output_.push_back(CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID));
            scratch_.push_back(CreateVariable((*transports[transportIdx]), SCRATCH_XN_ID));
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    currentRankSliceInputOffset_ = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    normalSliceSize_ = CreateVariable();
    lastSliceSize_ = CreateVariable();
    mySliceSize_ = CreateVariable();
    sliceOffset_ = CreateVariable();
    isInputOutputEqual_ = CreateVariable();
    locMask_ = CreateMaskSignal();
    srcMem_ = CreateMemory();
    dstMem_ = CreateMemory();
    reduceScatterSrc_.reserve(rankSize_);
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterSrc_.push_back(CreateMemory());
    }
    reduceScatterDst_.reserve(rankSize_);
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterDst_.push_back(CreateMemory());
    }
    sliceSize_ = CreateVariable();
    selfBit_ = 1 << rankId_;
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_)); // rankId_位为0，其他位都为1
    localGoSize_ = CreateGroupOpSize();
    return;
}

std::string CcuContextAllReduceMeshMem2Mem1D::GetLoopBlockTag(std::string loopType, int32_t index)
{
    return loopType + LOOP_BLOCK_TAG + std::to_string(index);
}

void CcuContextAllReduceMeshMem2Mem1D::CreateReduceLoop(
    uint32_t size, DataType dataType, DataType outputDataType, ReduceOp opType)
{
    constexpr uint32_t LOOP_NUM = 16;
    AllocGoResource(LOOP_NUM);

    std::string loopType = CcuRep::GetReduceTypeStr(dataType, opType);
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    uint32_t expansionNum = CcuRep::GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
        CcuRep::Memory src = CreateMemory();
        CcuRep::Memory dst = CreateMemory();
        std::vector<CcuRep::Memory> scratch;
        for (uint32_t i = 0; i < size; i++) {
            scratch.emplace_back(CreateMemory());
        }
        CcuRep::Variable lenForExpansion = CreateVariable();
        CcuRep::Variable len = CreateVariable();
        CcuRep::LoopBlock lb(this, GetLoopBlockTag(loopType, index));
        lb(dst, src, scratch, len, lenForExpansion);

        std::vector<CcuRep::CcuBuffer> bufs
            = {moRes.ccuBuffer.begin() + index * moConfig.msInterleave,
               moRes.ccuBuffer.begin() + index * moConfig.msInterleave + usedBufNum};
        CcuRep::MaskSignal sem = moRes.maskSignal[index];

        for (uint32_t i = 0; i < size; i++) {
            if (i == rankId_) {
                LocalCopy(bufs[i], src, len, sem, 1 << i);
            } else {
                LocalCopy(bufs[i], scratch[i], len, sem, 1 << i);
            }
        }
        LocalWait(sem, (1 << size) - 1);

        if (size > 1) {
            LocalReduce(bufs, size, dataType, outputDataType, opType, sem, len);
            LocalWait(sem);
        }

        LocalCopy(dst, bufs[0], lenForExpansion, sem);
        LocalWait(sem);
    }

    registeredLoop.insert(loopType);
}

void CcuContextAllReduceMeshMem2Mem1D::ReduceLoopGroup(
    CcuRep::Memory outDstOrg, CcuRep::Memory srcOrg, std::vector<CcuRep::Memory>& scratchOrg, GroupOpSize goSize,
    DataType dataType, DataType outputDataType, ReduceOp opType)
{
    const uint32_t size = scratchOrg.size();

    CcuRep::Memory dst = CreateMemory();
    CcuRep::Memory src = CreateMemory();
    std::vector<CcuRep::Memory> scratch;
    dst = outDstOrg;
    src = srcOrg;
    for (uint32_t idx = 0; idx < size; idx++) {
        scratch.push_back(CreateMemory());
        scratch[idx] = scratchOrg[idx];
    }

    CreateReduceLoop(size, dataType, outputDataType, opType);

    std::string loopType = CcuRep::GetReduceTypeStr(dataType, opType);
    uint32_t expansionNum = CcuRep::GetReduceExpansionNum(opType, dataType, outputDataType);
    CcuRep::Variable sliceSizeExpansion = CreateVariable();

    if (expansionNum != 1) {
        CcuRep::Variable tmp = CreateVariable();
        tmp = CcuRep::GetExpansionParam(expansionNum);
        dst.token += tmp;
    }

    // m部分
    CCU_IF(goSize.loopParam != 0) // goSize1
    {
        CcuRep::Variable loopParam = CreateVariable();
        CcuRep::Variable sliceSize = CreateVariable();
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;
        sliceSize = moConfig.memSlice;

        auto lc = Loop(GetLoopBlockTag(loopType, 0))(dst, src, scratch, sliceSize, sliceSizeExpansion);

        CcuRep::Variable paraCfg = CreateVariable();
        CcuRep::Variable offsetCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }

    CCU_IF(goSize.parallelParam != 0) // goSize2
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.addrOffset;
        }
        src.addr += goSize.addrOffset; // goSize0
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual; // goSize3
        }

        auto lc0 = Loop(GetLoopBlockTag(loopType, 0))(dst, src, scratch, goSize.residual, sliceSizeExpansion);

        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.residual;
        }
        src.addr += goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc1 = Loop(GetLoopBlockTag(loopType, 1))(dst, src, scratch, sliceSize, sliceSizeExpansion);

        CcuRep::Variable loopCfg0 = CreateVariable();
        CcuRep::Variable loopCfg1 = CreateVariable();
        CcuRep::Variable offsetCfg = CreateVariable();
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
    }
}

void CcuContextAllReduceMeshMem2Mem1D::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(scratch_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(mySliceSize_);
    Load(sliceOffset_);
    Load(isInputOutputEqual_);
    Load(localGoSize_);
    return;
}

void CcuContextAllReduceMeshMem2Mem1D::PreSync()
{
    // 互换内存信息
    for (auto t : transports) {
        HCCL_INFO("[CcuContextAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D LocalPost begin");
        // 交换起始地址
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit_);
        WriteVariableWithSignal(*t, output_[rankId_], OUTPUT_XN_ID, CKE_IDX_2, selfBit_);
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_3, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit_);
    GroupWait(*transportGroup, CKE_IDX_2, allBit_);
    GroupWait(*transportGroup, CKE_IDX_3, allBit_);
    HCCL_INFO("[CcuContextAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D wait all end");
    return;
}

void CcuContextAllReduceMeshMem2Mem1D::PostSync()
{
    for (auto& t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit_);
    HCCL_INFO("[CcuContextAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D AllReduce groupwait end");
}

void CcuContextAllReduceMeshMem2Mem1D::BcastLocToRmt(
    const CcuRep::Variable& srcAddr, const std::vector<CcuRep::Variable>& dstAddr)
{
    CHK_PRT_THROW(
        dstAddr.size() != transports.size() + 1,
        HCCL_ERROR("[ReduceRmtToLoc] dstAddr.size[%zu] != transports size[%zu] + 1", dstAddr.size(), transports.size()),
        InvalidParamsException, "Invalid srcAddr size");

    srcMem_.addr = srcAddr;
    srcMem_.addr += sliceOffset_;
    srcMem_.token = token_[rankId_];

    uint32_t transportIdx = 0;
    for (uint32_t rmtId = 0; rmtId < dstAddr.size(); rmtId++) {
        if (rmtId == rankId_) {
            continue;
        }
        dstMem_.addr = dstAddr[rmtId];
        dstMem_.addr += sliceOffset_;
        dstMem_.token = token_[rmtId];

        Write(*transports[transportIdx], dstMem_, srcMem_, sliceSize_, locMask_, 1 << rmtId);
        transportIdx++;
    }
    LocalWait(locMask_, allBit_);
}

void CcuContextAllReduceMeshMem2Mem1D::ReduceRmtToLoc(
    const std::vector<CcuRep::Variable>& srcAddr, const CcuRep::Variable& dstAddr)
{
    CHK_PRT_THROW(
        srcAddr.size() != transports.size() + 1,
        HCCL_ERROR("[ReduceRmtToLoc] srcAddr.size[%zu] != transports size[%zu] +1", srcAddr.size(), transports.size()),
        InvalidParamsException, "Invalid srcAddr size");

    dstMem_.addr = dstAddr;
    dstMem_.addr += sliceOffset_;
    dstMem_.token = token_[rankId_];

    CcuRep::Variable scratchOffset = CreateVariable();
    scratchOffset = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterSrc_[rankIdx].addr = srcAddr[rankIdx];
        reduceScatterSrc_[rankIdx].addr += sliceOffset_;
        reduceScatterSrc_[rankIdx].token = token_[rankIdx];

        reduceScatterDst_[rankIdx].addr = scratch_[rankId_];
        reduceScatterDst_[rankIdx].addr += scratchOffset;
        scratchOffset += normalSliceSize_;
        reduceScatterDst_[rankIdx].token = token_[rankId_];
    }

    uint32_t transportId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            LocalPost(locMask_, 1 << rankIdx);
        } else {
            Read(
                *transports[transportId], reduceScatterDst_[rankIdx], reduceScatterSrc_[rankIdx], sliceSize_, locMask_,
                1 << rankIdx);
            transportId++;
        }
    }
    LocalWait(locMask_, (1 << rankSize_) - 1);
    ReduceLoopGroup(
        dstMem_, reduceScatterSrc_[rankId_], reduceScatterDst_, localGoSize_, dataType_, outputDataType_, reduceOp_);
}

void CcuContextAllReduceMeshMem2Mem1D::DoRepeatAllReduce()
{
    if (rankId_ != rankSize_ - 1) {
        sliceSize_ = normalSliceSize_;
    } else {
        sliceSize_ = lastSliceSize_;
    }
    ReduceRmtToLoc(input_, output_[rankId_]);
    BcastLocToRmt(output_[rankId_], output_);
}

void CcuContextAllReduceMeshMem2Mem1D::Algorithm()
{
    HCCL_INFO("[CcuContextAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D run");
    InitResource();
    LoadArgs();
    PreSync();

    CCU_IF(mySliceSize_ != 0) { DoRepeatAllReduce(); }
    PostSync();
    HCCL_INFO("[CcuContextAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D end");
    return;
}

std::vector<uint64_t> CcuContextAllReduceMeshMem2Mem1D::GeneArgs(const CcuTaskArg& arg)
{
    const CurrentTaskArg* taskArg = dynamic_cast<const CurrentTaskArg*>(&arg);
    // 空指针校验
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMeshMem2Mem1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t scratchAddr = taskArg->scratchAddr_;
    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset = taskArg->outputSliceStride_ * rankId_;
    uint64_t normalSliceSize = taskArg->normalSliceSize_;
    uint64_t lastSliceSize = taskArg->lastSliceSize_;
    uint64_t mySliceSize = taskArg->mySliceSize_;
    uint64_t sliceOffset = taskArg->normalSliceSize_ * rankId_;
    uint64_t isInputOutputEqual = taskArg->isInputOutputEqual_;

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        tokenInfo,
        scratchAddr,
        currentRankSliceInputOffset,
        currentRankSliceOutputOffset,
        normalSliceSize,
        lastSliceSize,
        mySliceSize,
        sliceOffset,
        isInputOutputEqual,
    };

    auto normalGoSize = CalGoSize(normalSliceSize);
    auto lastGoSize = CalGoSize(lastSliceSize);

    if (rankId_ != rankSize_ - 1) {
        taskArgs.insert(taskArgs.end(), normalGoSize.begin(), normalGoSize.end());
    } else {
        taskArgs.insert(taskArgs.end(), lastGoSize.begin(), lastGoSize.end());
    }

    HCCL_INFO(
        "[CcuContextAllReduce1DMesh] TaskArgs: inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu], "
        "currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], normalSliceSize[%llu], "
        "lastSliceSize[%llu], mySliceSize[%llu], sliceOffset[%llu], isInputOutputEqual[%llu]",
        inputAddr, outputAddr, scratchAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, normalSliceSize,
        lastSliceSize, mySliceSize, sliceOffset, isInputOutputEqual);

    return taskArgs;
}
} // namespace Hccl
