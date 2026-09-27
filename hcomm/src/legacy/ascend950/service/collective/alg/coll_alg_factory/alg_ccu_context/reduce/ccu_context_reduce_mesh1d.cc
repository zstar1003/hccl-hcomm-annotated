/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_mesh1d.h"
#include "ccu_instruction_reduce_mesh1d.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

using CurrentCtxArg = CcuCtxArgReduceMesh1D;
using CurrentTaskArg = CcuTaskArgReduceMesh1D;

CcuContextReduceMesh1D::CcuContextReduceMesh1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    HCCL_DEBUG("[CcuContextReduceMesh1D] Enter Constructor");
    const CurrentCtxArg* ctxArg = dynamic_cast<const CurrentCtxArg*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceMesh1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;

    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceMesh1D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }

    if (ctxArg->dimSize_.size() > 0) {
        rankSize_ = ctxArg->dimSize_[0];
    }

    HCCL_INFO("[CcuContextReduceMesh1D] CtxArg: rankId[%u] rankSize[%llu]", rankId_, rankSize_);

    reduceOp_ = ctxArg->op_.reduceOp;
    rootId_ = ctxArg->rootId_;
    HCCL_INFO(
        "[CcuContextReduceMesh1D] init end, ctxArg->dimSize size[%zu] rankSize[%llu]", ctxArg->dimSize_.size(),
        rankSize_);
}

void CcuContextReduceMesh1D::InitResource()
{
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceMesh1D transports is empty"));
    }
    HCCL_INFO("[CcuContextReduceMesh1D]transports.size: [%zu]", transports.size());
    uint16_t transportIdx = 0;
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint16_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG(
                "[CcuContextReduceMesh1D] MyRank[%u], PeerId[%hu], TransportId[%hu]", rankId_, peerId, transportIdx);
            // 判断transport是否为空，为空直接报错
            CHK_PRT_THROW(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextReduceMesh1D] [InitResource] transports[%u] is nullptr", transportIdx),
                NullPtrException, "transport is null");
            input_.push_back(
                CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=1的Var来传递input
            output_.push_back(
                CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID)); // 获取transport中id=2的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }

    groupOpSize_ = CreateGroupOpSize();

    currentRankSliceInputOffset_ = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    repeatNum_ = CreateVariable();
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();

    normalSliceSize_ = CreateVariable();
    lastSliceSize_ = CreateVariable();
    repeatNumVar_ = CreateVariable();
    flag_ = CreateVariable();

    selfBit_ = 1 << rankId_;
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    localMem_ = CreateMemory();
    reomteMem_.reserve(rankSize_);
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reomteMem_.push_back(CreateMemory());
    }

    localSignal_ = CreateMaskSignal();
    return;
}

void CcuContextReduceMesh1D::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(repeatNum_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(repeatNumVar_);
    Load(groupOpSize_);
    return;
}

void CcuContextReduceMesh1D::PreSync()
{
    for (auto t : transports) {
        HCCL_INFO("[CcuContextReduceMesh1D] ReduceMesh1D LocalPost begin");
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit_); // index = 1，传递input信息
        WriteVariableWithSignal(*t, output_[rankId_], OUTPUT_XN_ID, CKE_IDX_2, selfBit_); // index = 0，传递output信息
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_3, selfBit_); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit_);
    GroupWait(*transportGroup, CKE_IDX_2, allBit_);
    GroupWait(*transportGroup, CKE_IDX_3, allBit_);
    HCCL_INFO("[CcuContextReduceMesh1D] ReduceMesh1D wait all end");
    return;
}

void CcuContextReduceMesh1D::PostSync()
{
    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit_);
    HCCL_INFO("[CcuContextReduceMesh1D] ReduceMesh1D Reduce groupwait end");
}

void CcuContextReduceMesh1D::DoRepeatReduce()
{
    std::vector<CcuRep::Memory>& src = reomteMem_;
    CcuRep::Memory& dst = localMem_;

    dst.addr = output_[rankId_];
    dst.token = token_[rankId_];
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rootId_) {
            src[curId].addr = input_[rankIdx];
            src[curId].token = token_[rankIdx];
            curId++;
        } else {
            continue;
        }
    }
    src[rankSize_ - 1].addr = input_[rankId_];
    src[rankSize_ - 1].token = token_[rankId_];

    CCU_IF(flag_ != 0)
    {
        // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        dst.addr += outputRepeatStride_;
        for (auto& s : src) {
            s.addr += inputRepeatStride_;
        }
    }
    GroupReduce(transports, dst, src, groupOpSize_, dataType_, outputDataType_, reduceOp_);
}

void CcuContextReduceMesh1D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceMesh1D] ReduceMesh1D run");
    InitResource();
    LoadArgs();
    PreSync();
    if (rankId_ == rootId_) {
        CcuRep::Variable repeatNumAdd = CreateVariable();
        repeatNumAdd = 1;
        flag_ = 0;
        CCU_WHILE(repeatNumVar_ != UINT64_MAX)
        { // 循环repeatNum_次
            DoRepeatReduce();
            repeatNumVar_ += repeatNumAdd;
            flag_ = 1;
        }
    }
    PostSync();
    HCCL_INFO("[CcuContextReduceMesh1D] ReduceMesh1D end");
    return;
}

std::vector<uint64_t> CcuContextReduceMesh1D::GeneArgs(const CcuTaskArg& arg)
{
    const CurrentTaskArg* taskArg = dynamic_cast<const CurrentTaskArg*>(&arg);
    // 空指针校验
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceMesh1D::taskArg ptr is null"));
    }
    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset = taskArg->outputSliceStride_ * rankId_;
    uint64_t repeatNum = taskArg->repeatNum_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize = taskArg->normalSliceSize_;
    uint64_t lastSliceSize = taskArg->lastSliceSize_;
    uint64_t repeatNumVar = taskArg->repeatNumVar_;
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;

    auto goSize = CalGoSize(normalSliceSize);

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        tokenInfo,
        currentRankSliceInputOffset,
        currentRankSliceOutputOffset,
        repeatNum,
        inputRepeatStride,
        outputRepeatStride,
        normalSliceSize,
        lastSliceSize,
        repeatNumVar,
        goSize[0],
        goSize[1],
        goSize[2],
        goSize[3],
    };

    HCCL_INFO(
        "[CcuContextReduceMesh1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], currentRankSliceInputOffset[%llu], "
        "currentRankSliceOutputOffset[%llu], repeatNum[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], "
        "normalSliceSize[%llu], lastSliceSize[%llu], repeatNumVar[%llu], goSize[0][%llu], goSize[1][%llu], "
        "goSize[2][%llu], goSize[3][%llu], ",
        inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, repeatNum, inputRepeatStride,
        outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar, goSize[0], goSize[1], goSize[2], goSize[3]);

    return taskArgs;
}
} // namespace Hccl
