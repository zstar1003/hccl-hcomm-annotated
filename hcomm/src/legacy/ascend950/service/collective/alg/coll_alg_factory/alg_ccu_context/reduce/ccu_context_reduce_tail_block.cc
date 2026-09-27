/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_tail_block.h"
#include "ccu_instruction_reduce_tail_block.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;

constexpr uint64_t TAIL_BLOCK_LOOP_NUM = 64;
constexpr uint64_t MISSION_NUM_2 = 2;

CcuContextReduceTailBlock::CcuContextReduceTailBlock(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceTailBlock* ctxArg = dynamic_cast<const CcuCtxArgReduceTailBlock*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceTailBlock::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    notifySignal_ = ctxArg->notifySignal_;
    reduceOp_ = ctxArg->op_.reduceOp;
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceTailBlock] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }

    HCCL_INFO(
        "[CcuContextReduceTailBlock] Init, CtxArgs are rankId[%u], rankSize[%llu], notifySignal[%s], "
        "reduceOp[%s], dataType[%s], outputDataType[%s]",
        rankId_, rankSize_, notifySignal_.c_str(), reduceOp_.Describe().c_str(), dataType_.Describe().c_str(),
        outputDataType_.Describe().c_str());
}

void CcuContextReduceTailBlock::Algorithm()
{
    HCCL_INFO("[CcuContextReduceTailBlock] Algorithm start");
    InitResource();
    ExportVariables();
    SyncMainBlock(0); // 与住任务进行前同步， 主 mission 激活尾块 mission

    DoGroupReduce();

    SyncMainBlock(1); // 与住任务进行后同步， 通知主 mission 尾块处理完成
    HCCL_INFO("[CcuContextReduceTailBlock] Algorithm end");
    return;
}

void CcuContextReduceTailBlock::SyncMainBlock(uint32_t ctxSignalIndex)
{
    HCCL_INFO("[CcuContextReduceTailBlock] SyncMainBlock start, ctxSignalIndex[%u]", ctxSignalIndex);
    LocalCtxPost(mainBlockCtxSignal_, 1 << (1 + ctxSignalIndex * MISSION_NUM_2));
    LocalWait(tailBlockCtxSignal_, 1 << (0 + ctxSignalIndex * MISSION_NUM_2));
    HCCL_INFO("[CcuContextReduceTailBlock] SyncMainBlock end");
}

void CcuContextReduceTailBlock::InitResource()
{
    HCCL_INFO("[CcuContextReduceTailBlock] InitResource start");
    // init resource
    output_ = CreateVariable();
    uint16_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceTailBlock transports is empty"));
    }
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextReduceTailBlock] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId, transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextReduceTailBlock] Algorithm transport ptr is null"), );
            input_.push_back(CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=1的Var
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }

    inputOffset_ = CreateVariable();
    outputOffset_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    tailBlockCtxSignal_ = CreateMaskSignal();

    // init loop param
    AllocGoResource(TAIL_BLOCK_LOOP_NUM);
    HCCL_INFO("[CcuContextReduceTailBlock] InitResource end");
}

void CcuContextReduceTailBlock::DoGroupReduce()
{
    HCCL_INFO("[CcuContextReduceTailBlock] DoGroupReduce start");
    // 初始化地址寄存器
    std::vector<CcuRep::Memory> reduceSrc;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceSrc.push_back(CreateMemory());
    }
    CcuRep::Memory reduceDst = CreateMemory();

    // 填充地址
    uint32_t dstId = 0;
    uint32_t curId = 0;
    // SRC
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        reduceSrc[curId].addr = input_[rankIdx];
        reduceSrc[curId].addr += inputOffset_;
        reduceSrc[curId].token = token_[rankIdx];
    }

    // DST
    reduceDst.addr = output_;
    reduceDst.addr += outputOffset_;
    reduceDst.token = token_[rankId_];

    // 执行 reduce 操作
    GroupReduce(transports, reduceDst, reduceSrc, groupOpSize_, outputDataType_, dataType_, reduceOp_);
    HCCL_INFO("[CcuContextReduceTailBlock] DoGroupReduce end");
    return;
}

void CcuContextReduceTailBlock::ExportVariables()
{
    HCCL_INFO("[CcuContextReduceTailBlock] ExportVariables start");
    ExportVariable(input_[rankId_], notifySignal_ + "_Input_Reduce_Tail_Block");
    ExportVariable(output_, notifySignal_ + "_Output_Reduce_Tail_Block");
    ExportVariable(token_[rankId_], notifySignal_ + "_Token_Reduce_Tail_Block");
    ExportVariable(inputOffset_, notifySignal_ + "_Input_Offset_Reduce_Tail_Block");
    ExportVariable(outputOffset_, notifySignal_ + "_Output_Offset_Reduce_Tail_Block");
    ExportVariable(groupOpSize_.addrOffset, notifySignal_ + "_AddrOffset_Reduce_Tail_Block");
    ExportVariable(groupOpSize_.loopParam, notifySignal_ + "_LoopParam_Reduce_Tail_Block");
    ExportVariable(groupOpSize_.parallelParam, notifySignal_ + "_ParallelParam_Reduce_Tail_Block");
    ExportVariable(groupOpSize_.residual, notifySignal_ + "_Residual_Reduce_Tail_Block");

    ExportMaskSignal(tailBlockCtxSignal_, notifySignal_ + "_CtxSync_Reduce_Tail_Block");
    mainBlockCtxSignal_ = ImportMaskSignal(notifySignal_ + "_CtxSync_Main_Block");
    HCCL_INFO("[CcuContextReduceTailBlock] ExportVariables end");
}

std::vector<uint64_t> CcuContextReduceTailBlock::GeneArgs(const CcuTaskArg& arg)
{
    (void)arg; // MC2 要求尾块处理不能使用 CTX_ARG，参数需要通过主任务传入
    HCCL_INFO("[CcuContextReduceTailBlock] GeneArgs, TailBlock, no need to gene args");
    return {};
}
} // namespace Hccl
