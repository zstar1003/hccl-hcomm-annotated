/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_scatter_mesh1d.h"
#include "ccu_instruction_scatter_mesh1d.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;

using CurrentCtxArg = CcuCtxArgScatterMesh1D;
using CurrentTaskArg = CcuTaskArgScatterMesh1D;

CcuContextScatterMesh1D::CcuContextScatterMesh1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    HCCL_INFO("[CcuContextScatterMesh1D] Enter Constructor");
    const CurrentCtxArg* ctxArg = dynamic_cast<const CurrentCtxArg*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextScatterMesh1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    rootId_ = ctxArg->rootId_;

    HCCL_INFO("[CcuContextScatterMesh1D] CtxArg: rankId[%u] rankSize[%llu]", rankId_, rankSize_);
}

void CcuContextScatterMesh1D::InitResource()
{
    selfBit_ = 1 << rankId_;
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextScatterMesh1D transports is empty"));
    }
    HCCL_INFO("[CcuContextScatterMesh1D]transports.size: [%zu]", transports.size());
    uint16_t transportIdx = 0;
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextScatterMesh1D] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId, transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr || transportIdx >= transports.size(),
                HCCL_ERROR(
                    "[CcuContextScatterMesh1D] Algorithm transport ptr is null or transportIdx is out of bounds"), );
            output_.push_back(
                CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID)); // 获取transport中id=1的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    input_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    currentRankSliceInputOffset_ = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    normalSliceSize_ = CreateVariable();
    lastSliceSize_ = CreateVariable();
    repeatNumVar_ = CreateVariable();
    flag_ = CreateVariable();
    flag_ = 0;
    localMem_.reserve(rankSize_);
    remoteMem_.reserve(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        remoteMem_.push_back(CreateMemory());
        localMem_.push_back(CreateMemory());
    }

    localSignal_ = CreateMaskSignal();
    return;
}

void CcuContextScatterMesh1D::LoadArgs()
{
    Load(input_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(repeatNumVar_);
    Load(groupOpSize_);
    return;
}

void CcuContextScatterMesh1D::Algorithm()
{
    HCCL_INFO("[CcuContextScatterMesh1D] ScatterMesh1D run");
    InitResource();
    LoadArgs();
    PreSync();
    if (rankId_ == rootId_) {
        RunSendScatter();
    } else {
        HCCL_INFO(
            "[CcuContextScatterMesh1D] RunRecvScatter local rank[%u], root rank[%u], do nothing", rankId_, rootId_);
    }
    PostSync();
    HCCL_INFO("[CcuContextScatterMesh1D] ScatterMesh1D end");
    return;
}

void CcuContextScatterMesh1D::PostSync()
{
    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit_);
    HCCL_INFO("[CcuContextScatterMesh1D] ScatterMesh1D groupwait end");
}

void CcuContextScatterMesh1D::PreSync()
{
    for (auto t : transports) {
        HCCL_INFO("[CcuContextScatterMesh1D] ScatterMesh1D LocalPost begin");
        WriteVariableWithSignal(*t, output_[rankId_], CKE_IDX_1, CKE_IDX_1, selfBit_); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[rankId_], CKE_IDX_2, CKE_IDX_2, selfBit_);  // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit_); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit_); // index = 2，传递token信息
    HCCL_INFO("[CcuContextScatterMesh1D] ScatterMesh1D wait all end");
    return;
}

void CcuContextScatterMesh1D::RunSendScatter()
{
    uint16_t fullBit = ((1 << rankSize_) - 1);
    HCCL_INFO(
        "[CcuContextScatterMesh1D] RunSendScatter local rank[%u], root rank[%u], start send data", rankId_, rootId_);
    std::vector<CcuRep::Memory>& dst = remoteMem_; // 在initresource里面push_back
    std::vector<CcuRep::Memory>& src = localMem_;

    for (uint64_t curId = 0; curId < rankSize_; curId++) {
        src[curId].token = token_[curId];
        dst[curId].token = token_[curId];

        src[curId].addr = input_;
        for (uint64_t i = 0; i < curId; i++) {
            src[curId].addr += currentRankSliceInputOffset_;
        }
        dst[curId].addr = output_[curId];
    }
    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd = 1;
    CCU_WHILE(repeatNumVar_ != UINT64_MAX)
    { // 循环repeatNum_次
        repeatNumVar_ += repeatNumAdd;
        CCU_IF(flag_ != 0)
        {
            // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
            for (auto& s : src) {
                s.addr += inputRepeatStride_;
            }
            for (auto& d : dst) {
                d.addr += outputRepeatStride_;
            }
        }
        flag_ = 1;
        CcuRep::MaskSignal locSig = CreateMaskSignal();
        uint16_t transportIdx = 0;
        for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            if (rankIdx == rankId_) {
                LocalCopy(dst[rankIdx], src[rankIdx], normalSliceSize_, locSig, 1 << rankIdx);
            } else {
                Write(*transports[transportIdx], dst[rankIdx], src[rankIdx], normalSliceSize_, locSig, 1 << rankIdx);
                transportIdx++;
            }
        }
        LocalWait(locSig, fullBit);
    }
    return;
}

std::vector<uint64_t> CcuContextScatterMesh1D::GeneArgs(const CcuTaskArg& arg)
{
    const CurrentTaskArg* taskArg = dynamic_cast<const CurrentTaskArg*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextScatterMesh1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_;
    uint64_t currentRankSliceOutputOffset = taskArg->outputSliceStride_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize = taskArg->normalSliceSize_;
    uint64_t lastSliceSize = taskArg->lastSliceSize_;
    uint64_t repeatNumVar = taskArg->repeatNumVar_;
    auto goSize = CalGoSize(normalSliceSize);

    std::vector<uint64_t> taskArgs
        = {inputAddr,
           outputAddr,
           tokenInfo,
           currentRankSliceInputOffset,
           currentRankSliceOutputOffset,
           inputRepeatStride,
           outputRepeatStride,
           normalSliceSize,
           lastSliceSize,
           repeatNumVar,
           goSize[0],
           goSize[1],
           goSize[2],
           goSize[3]};

    HCCL_INFO(
        "[CcuContextScatterMesh1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], "
        "inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu],"
        "repeatNumVar[%llu], goSize[0][%llu], goSize[1][%llu], goSize[2][%llu], goSize[3][%llu]",
        inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, inputRepeatStride,
        outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar, goSize[0], goSize[1], goSize[2], goSize[3]);
    return taskArgs;
}
} // namespace Hccl
