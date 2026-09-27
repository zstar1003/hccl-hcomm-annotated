/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_scatter_mesh1d_mem2mem.h"
#include "ccu_instruction_reduce_scatter_mesh1d_mem2mem.h"
#include "ccu_assist.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

CcuContextReduceScatterMeshMem2Mem1D::CcuContextReduceScatterMeshMem2Mem1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceScatterMeshMem2Mem1D* ctxArg = dynamic_cast<const CcuCtxArgReduceScatterMeshMem2Mem1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMeshMem2Mem1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuContextReduceScatterMeshMem2Mem1D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    reduceOp_ = ctxArg->op_.reduceOp;
    HCCL_INFO(
        "[CcuContextReduceScatterMeshMem2Mem1D] Init, CtxArgs are rankId[%u], rankSize_[%u], dataType[%s], "
        "outputDataType[%s], reduceOp[%s]",
        rankId_, rankSize_, dataType_.Describe().c_str(), outputDataType_.Describe().c_str(),
        reduceOp_.Describe().c_str());
}

void CcuContextReduceScatterMeshMem2Mem1D::InitResource()
{
    uint16_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMeshMem2Mem1D transports is empty"));
    }

    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            scratch_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG(
                "[CcuContextReduceScatterMeshMem2Mem1D] MyRank[%u], PeerId[%u], TransportId[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_THROW(
                transports[transportIdx] == nullptr,
                HCCL_ERROR(
                    "[CcuContextReduceScatterMeshMem2Mem1D][InitResource] transports[%u] is nullptr", transportIdx),
                NullPtrException, "transport is null");

            input_.push_back(
                CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=0的Var来传递output
            scratch_.push_back(CreateVariable((*transports[transportIdx]), SCRATCH_XN_ID));
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    output_ = CreateVariable();
    currentRankSliceInputOffset_ = CreateVariable();
    normalSliceSize_ = CreateVariable();
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    repeatNum_ = CreateVariable();
    flag_ = CreateVariable();

    normalGoSize_ = CreateGroupOpSize();

    selfBit_ = 1 << rankId_;                              // 仅rankid位为1，其他位为0，代表本端准备好了
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_)); // 仅rankid位为0，其他位为1，代表远端准备好了
    localMem_.reserve(rankSize_);
    remoteMem_.reserve(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        remoteMem_.push_back(CreateMemory());
        localMem_.push_back(CreateMemory());
    }

    localSignal_ = CreateMaskSignal();
    return;
}

void CcuContextReduceScatterMeshMem2Mem1D::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_);
    Load(token_[rankId_]);
    Load(scratch_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(repeatNum_);
    Load(normalGoSize_);
    return;
}

void CcuContextReduceScatterMeshMem2Mem1D::PreSync()
{
    for (auto& t : transports) {
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit_); // index = 1，传递input信息
        WriteVariableWithSignal(*t, scratch_[rankId_], SCRATCH_XN_ID, CKE_IDX_2, selfBit_);
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_3, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit_);
    GroupWait(*transportGroup, CKE_IDX_2, allBit_);
    GroupWait(*transportGroup, CKE_IDX_3, allBit_);
    return;
}

void CcuContextReduceScatterMeshMem2Mem1D::PostSync()
{
    for (auto& t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit_);
}

void CcuContextReduceScatterMeshMem2Mem1D::DoReduceScatter()
{
    u32 transportId = 0;

    CcuRep::Memory outDst = CreateMemory();
    outDst.addr = output_;
    outDst.token = token_[rankId_];

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            LocalPost(localSignal_, 1 << rankIdx);
        } else {
            Read(
                *transports[transportId], remoteMem_[rankIdx], localMem_[rankIdx], normalSliceSize_, localSignal_,
                1 << rankIdx);
            transportId++;
        }
    }
    // 等读完所有对端
    LocalWait(localSignal_, (1 << rankSize_) - 1);

    ReduceLoopGroup(outDst, localMem_[rankId_], remoteMem_, normalGoSize_, dataType_, outputDataType_, reduceOp_);
}

void CcuContextReduceScatterMeshMem2Mem1D::DoRepeatReduceScatter()
{
    CcuRep::Variable scratchOffset = CreateVariable();
    scratchOffset = 0;

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        localMem_[rankIdx].addr = input_[rankIdx];
        localMem_[rankIdx].addr += currentRankSliceInputOffset_;
        localMem_[rankIdx].token = token_[rankIdx];

        remoteMem_[rankIdx].addr = scratch_[rankId_];
        remoteMem_[rankIdx].addr += scratchOffset;
        scratchOffset += normalSliceSize_;
        remoteMem_[rankIdx].token = token_[rankId_];
    }

    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd = 1;
    flag_ = 0;
    CCU_WHILE(repeatNum_ != UINT64_MAX)
    {
        repeatNum_ += repeatNumAdd;
        CCU_IF(flag_ == 1)
        {
            // 非第一轮执行时，src和dst已经初始化，需要添加偏移量
            for (auto& s : localMem_) {
                s.addr += inputRepeatStride_;
            }
            output_ += outputRepeatStride_;
        }
        CCU_IF(normalSliceSize_ != 0) { DoReduceScatter(); }
        flag_ = 1;
    }
}

std::string CcuContextReduceScatterMeshMem2Mem1D::GetLoopBlockTag(std::string loopType, int32_t index)
{
    return loopType + LOOP_BLOCK_TAG + std::to_string(index);
}

void CcuContextReduceScatterMeshMem2Mem1D::CreateReduceLoop(
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
        CcuRep::Memory dst = CreateMemory();
        CcuRep::Memory src = CreateMemory();
        std::vector<CcuRep::Memory> scratch;
        for (uint32_t i = 0; i < size; i++) {
            scratch.emplace_back(CreateMemory());
        }
        CcuRep::Variable len = CreateVariable();
        CcuRep::Variable lenForExpansion = CreateVariable();
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

void CcuContextReduceScatterMeshMem2Mem1D::ReduceLoopGroup(
    CcuRep::Memory outDstOrg, CcuRep::Memory srcOrg, std::vector<CcuRep::Memory>& scratchOrg, GroupOpSize goSize,
    DataType dataType, DataType outputDataType, ReduceOp opType)
{
    const uint32_t size = scratchOrg.size();

    CcuRep::Memory dst = CreateMemory();
    dst = outDstOrg;

    CcuRep::Memory src = CreateMemory();
    src = srcOrg;

    std::vector<CcuRep::Memory> scratch;
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
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc = Loop(GetLoopBlockTag(loopType, 0))(dst, src, scratch, sliceSize, sliceSizeExpansion);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
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
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
    }
}

void CcuContextReduceScatterMeshMem2Mem1D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem1D] ReduceScatterMesh1DMem2Mem run");

    InitResource();

    LoadArgs();

    PreSync();

    DoRepeatReduceScatter();

    PostSync();

    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem1D] ReduceScatterMesh1DMem2Mem end");
    return;
}

std::vector<uint64_t> CcuContextReduceScatterMeshMem2Mem1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceScatterMeshMem2Mem1D* taskArg
        = dynamic_cast<const CcuTaskArgReduceScatterMeshMem2Mem1D*>(&arg);
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t scratchAddr = taskArg->scratchAddr_;
    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_ * rankId_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize = taskArg->normalSliceSize_;
    uint64_t repeatNum = taskArg->repeatNum_;
    auto normalGoSize = CalGoSize(normalSliceSize);

    std::vector<uint64_t> taskArgs
        = {inputAddr,         outputAddr,         tokenInfo,       scratchAddr, currentRankSliceInputOffset,
           inputRepeatStride, outputRepeatStride, normalSliceSize, repeatNum};

    HCCL_INFO(
        "[CcuContextReduceScatterMeshMem2Mem1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "scratchAddr[%llu], currentRankSliceInputOffset[%llu], inputRepeatStride[%llu],"
        "outputRepeatStride[%llu], normalSliceSize[%llu], repeatNum[%llu]",
        inputAddr, outputAddr, scratchAddr, currentRankSliceInputOffset, inputRepeatStride, outputRepeatStride,
        normalSliceSize, repeatNum);

    taskArgs.insert(taskArgs.cend(), normalGoSize.cbegin(), normalGoSize.cend());

    return taskArgs;
}

} // namespace Hccl
