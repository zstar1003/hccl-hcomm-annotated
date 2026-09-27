/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_scatter_v_mesh1d_mem2mem.h"
#include "ccu_instruction_reduce_scatter_v_mesh1d_mem2mem.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

CcuContextReduceScatterVMeshMem2Mem1D::CcuContextReduceScatterVMeshMem2Mem1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceScatterVMeshMem2Mem1D* ctxArg
        = dynamic_cast<const CcuCtxArgReduceScatterVMeshMem2Mem1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterVMeshMem2Mem1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceScatterVMeshMem2Mem1D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    reduceOp_ = ctxArg->op_.reduceOp;
    HCCL_INFO(
        "[CcuContextReduceScatterVMeshMem2Mem1D] Init, CtxArgs are rankId[%u], rankSize_[%llu], dataType[%s], "
        "outputDataType[%s], reduceOp[%s]",
        rankId_, rankSize_, dataType_.Describe().c_str(), outputDataType_.Describe().c_str(),
        reduceOp_.Describe().c_str());
}

void CcuContextReduceScatterVMeshMem2Mem1D::InitResources()
{
    uint16_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterVMeshMem2Mem1D transports is empty"));
    }
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            scratch_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextReduceScatterVMeshMem2Mem1D] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextReduceScatterVMeshMem2Mem1D] Algorithm transport ptr is null"), );
            input_.push_back(
                CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=1的Var来传递output
            scratch_.push_back(CreateVariable((*transports[transportIdx]), SCRATCH_XN_ID));
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    output_ = CreateVariable();
    scratchInterval_ = CreateVariable();
    sliceSize_ = CreateVariable();
    offset_ = CreateVariable();
    return;
}

void CcuContextReduceScatterVMeshMem2Mem1D::CollectAllRanksSlice(
    std::vector<CcuRep::Memory>& tmpSrc, std::vector<CcuRep::Memory>& tmpDst, const CcuRep::MaskSignal& locMask)
{
    uint16_t allBit = (1 << rankSize_) - 1; // 等待包含自身的全部对端
    u32 transportId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            LocalCopy(tmpDst[rankIdx], tmpSrc[rankIdx], sliceSize_, locMask, 1 << rankIdx);
        } else {
            Read(*transports[transportId], tmpDst[rankIdx], tmpSrc[rankIdx], sliceSize_, locMask, 1 << rankIdx);
            transportId++;
        }
    }
    // 等读完所有对端
    LocalWait(locMask, allBit);
}

void CcuContextReduceScatterVMeshMem2Mem1D::PrepareReduceScatterVData(
    std::vector<CcuRep::Memory>& reduceScatterVSrc, std::vector<CcuRep::Memory>& reduceScatterVDst)
{
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterVSrc.push_back(CreateMemory());
        reduceScatterVDst.push_back(CreateMemory());
    }

    CcuRep::Variable scratchOffset = CreateVariable();
    scratchOffset = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterVSrc[rankIdx].addr = input_[rankIdx];
        reduceScatterVSrc[rankIdx].addr += offset_;
        reduceScatterVSrc[rankIdx].token = token_[rankIdx];

        reduceScatterVDst[rankIdx].addr = scratch_[rankId_];
        reduceScatterVDst[rankIdx].addr += scratchOffset;
        scratchOffset += scratchInterval_;
        reduceScatterVDst[rankIdx].token = token_[rankId_];
    }
    return;
}

void CcuContextReduceScatterVMeshMem2Mem1D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceScatterVMeshMem2Mem1D] ReduceScatterVMeshMem2Mem1D run");
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    InitResources();

    Load(input_[rankId_]);
    Load(output_);
    Load(token_[rankId_]);
    Load(scratch_[rankId_]);
    Load(scratchInterval_);
    Load(sliceSize_);
    Load(offset_);

    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*t, scratch_[rankId_], SCRATCH_XN_ID, CKE_IDX_2, selfBit);
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_3, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit);
    GroupWait(*transportGroup, CKE_IDX_2, allBit);
    GroupWait(*transportGroup, CKE_IDX_3, allBit);

    CCU_IF(sliceSize_ != 0)
    {
        std::vector<CcuRep::Memory> reduceScatterVSrc;
        std::vector<CcuRep::Memory> reduceScatterVDst;

        CcuRep::MaskSignal locMask = CreateMaskSignal();
        PrepareReduceScatterVData(reduceScatterVSrc, reduceScatterVDst);
        CollectAllRanksSlice(reduceScatterVSrc, reduceScatterVDst, locMask);

        for (uint32_t rankIdx = 1; rankIdx < rankSize_; rankIdx++) {
            LocalReduce(reduceScatterVDst[0], reduceScatterVDst[rankIdx], sliceSize_, dataType_, reduceOp_, locMask, 1);
            LocalWait(locMask, 1);
        }

        CcuRep::Memory outDst = CreateMemory();
        outDst.addr = output_;
        outDst.token = token_[rankId_];
        LocalCopy(outDst, reduceScatterVDst[0], sliceSize_, locMask, 1 << rankId_);
        LocalWait(locMask, 1 << rankId_);
    }
    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit);
    HCCL_INFO("[CcuContextReduceScatterVMeshMem2Mem1D] ReduceScatterVMeshMem2Mem1D end");
    return;
}

std::vector<uint64_t> CcuContextReduceScatterVMeshMem2Mem1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceScatterVMeshMem2Mem1D* taskArg
        = dynamic_cast<const CcuTaskArgReduceScatterVMeshMem2Mem1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterVMeshMem2Mem1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t scratchAddr = taskArg->scratchAddr_;
    uint64_t scratchInterval = taskArg->scratchInterval_;
    uint64_t sliceSize = taskArg->sliceSize_;
    uint64_t offset = taskArg->offset_;
    return {inputAddr, outputAddr, tokenInfo, scratchAddr, scratchInterval, sliceSize, offset};
}

} // namespace Hccl
