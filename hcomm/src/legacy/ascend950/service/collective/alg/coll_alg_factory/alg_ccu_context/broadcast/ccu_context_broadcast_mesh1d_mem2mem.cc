/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_broadcast_mesh1d_mem2mem.h"
#include "ccu_instruction_broadcast_mesh1d_mem2mem.h"

namespace Hccl {
constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;

using CurrentCtxArg = CcuCtxArgBroadcastMesh1DMem2Mem;
using CurrentTaskArg = CcuTaskArgBroadcastMesh1DMem2Mem;

CcuContextBroadcastMesh1DMem2Mem::CcuContextBroadcastMesh1DMem2Mem(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    HCCL_DEBUG("[CcuContextBroadcastMesh1DMem2Mem] Enter Constructor");
    const CurrentCtxArg* ctxArg = dynamic_cast<const CurrentCtxArg*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextBroadcastMesh1DMem2Mem::ctxArg ptr is null"));
    }
    if (ctxArg->dimSize_.size() != 1) {
        THROW<NullPtrException>(StringFormat("CcuContextBroadcastMesh1DMem2Mem::dimSize is not 1"));
    }
    rankId_ = ctxArg->rankId_;
    rootId_ = ctxArg->rootId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;

    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextBroadcastMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }

    HCCL_INFO(
        "[CcuContextBroadcastMesh1DMem2Mem] init end, ctxArg->dimSize size[%zu] rankSize[%llu].",
        ctxArg->dimSize_.size(), rankSize_);
}

void CcuContextBroadcastMesh1DMem2Mem::InitResource()
{
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextBroadcastMesh1DMem2Mem transports is empty"));
    }
    HCCL_INFO("[CcuContextBroadcastMesh1DMem2Mem]transports.size: [%zu]", transports.size());
    uint16_t transportIdx = 0;
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint16_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG(
                "[CcuContextBroadcastMesh1DMem2Mem] MyRank[%u], PeerId[%hu], TransportId[%hu]", rankId_, peerId,
                transportIdx);
            // 判断transport是否为空，为空直接报错
            CHK_PRT_THROW(
                transports[transportIdx] == nullptr || transportIdx >= transports.size(),
                HCCL_ERROR(
                    "[CcuContextBroadcastMesh1DMem2Mem] [InitResource] transports[%u] is nullptr or out of bounds",
                    transportIdx),
                NullPtrException, "transport is null");
            input_.push_back(
                CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=1的Var来传递input
            output_.push_back(
                CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID)); // 获取transport中id=2的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++; // transport对应远端通信数
        }
    }
    currentRankSliceInputOffset_ = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    normalSliceSize_ = CreateVariable();
    lastSliceSize_ = CreateVariable();
    allgatherOffset_ = CreateVariable();
    repeatNumVar_ = CreateVariable();
    flag_ = CreateVariable();
    SliceOffset_ = CreateVariable();

    selfBit_ = 1 << rankId_;
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    scattersrcMem_.reserve(rankSize_);
    scatterdstMem_.reserve(rankSize_);
    allgatherdstMem_.reserve(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        scattersrcMem_.push_back(CreateMemory());
        scatterdstMem_.push_back(CreateMemory());
        allgatherdstMem_.push_back(CreateMemory());
    }
    localSignal_ = CreateMaskSignal();
    return;
}

void CcuContextBroadcastMesh1DMem2Mem::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(allgatherOffset_);
    Load(repeatNumVar_);
    return;
}

void CcuContextBroadcastMesh1DMem2Mem::PreSync() // 前同步
{
    for (auto t : transports) {
        HCCL_INFO("[CcuContextBroadcastMesh1DMem2Mem] BroadcastMesh1D LocalPost begin");
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit_); // index = 1，传递input信息
        WriteVariableWithSignal(*t, output_[rankId_], OUTPUT_XN_ID, CKE_IDX_2, selfBit_); // index = 0，传递output信息
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_3, selfBit_); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit_);
    GroupWait(*transportGroup, CKE_IDX_2, allBit_);
    GroupWait(*transportGroup, CKE_IDX_3, allBit_);
    HCCL_INFO("[CcuContextBroadcastMesh1DMem2Mem] BroadcastMesh1D wait all end");
    return;
}

void CcuContextBroadcastMesh1DMem2Mem::PostSync(int CKE_id) // 后同步
{
    for (auto t : transports) {
        RemotePost(*t, CKE_id, selfBit_);
    }
    GroupWait(*transportGroup, CKE_id, allBit_);
    HCCL_INFO("[CcuContextBroadcastMesh1DMem2Mem] BroadcastMesh1D groupwait end");
}

void CcuContextBroadcastMesh1DMem2Mem::DoRepeaScatterMem2Mem()
{
    if (rankId_ != rootId_) {
        return;
    }
    std::vector<CcuRep::Memory>& src = scattersrcMem_;
    std::vector<CcuRep::Memory>& dst = scatterdstMem_;

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == 0) {
            SliceOffset_ = 0;
        } else {
            SliceOffset_ += normalSliceSize_;
        }

        src[rankIdx].addr = input_[rankId_];
        src[rankIdx].addr += currentRankSliceInputOffset_;
        src[rankIdx].addr += SliceOffset_;
        src[rankIdx].token = token_[rankIdx];

        dst[rankIdx].addr = output_[rankIdx];
        dst[rankIdx].addr += currentRankSliceOutputOffset_;
        dst[rankIdx].addr += SliceOffset_;
        dst[rankIdx].token = token_[rankIdx];

        CCU_IF(flag_ != 0)
        {
            // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
            for (uint32_t curId = 0; curId < rankSize_; curId++) {
                src[curId].addr += inputRepeatStride_;
                dst[curId].addr += outputRepeatStride_;
            }
        }
    }
    DoScatter(src, dst);
}
void CcuContextBroadcastMesh1DMem2Mem::DoRepeatAllGatherMem2Mem()
{
    CcuRep::Memory& src = scatterdstMem_[rankId_];
    std::vector<CcuRep::Memory>& dst = allgatherdstMem_;
    src.addr = output_[rankId_];
    src.addr += currentRankSliceOutputOffset_;
    src.addr += allgatherOffset_;
    src.token = token_[rankId_];

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        dst[rankIdx].addr = output_[rankIdx];
        dst[rankIdx].addr += currentRankSliceOutputOffset_;
        dst[rankIdx].addr += allgatherOffset_;
        dst[rankIdx].token = token_[rankIdx];
    }
    CCU_IF(flag_ != 0)
    {
        //  非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        src.addr += inputRepeatStride_;
        for (auto& d : dst) {
            d.addr += outputRepeatStride_;
        }
    }

    DoAllGather(src, dst);
}

void CcuContextBroadcastMesh1DMem2Mem::DoScatter(
    const std::vector<CcuRep::Memory>& src, const std::vector<CcuRep::Memory>& dst)
{
    uint64_t transportId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        auto& sliceSize = (rankIdx + 1 == rankSize_) ? lastSliceSize_ : normalSliceSize_;
        CCU_IF(sliceSize != 0)
        {
            if (rankIdx == rankId_) {
                LocalPost(localSignal_, 1 << rankIdx);
            } else {
                Write(*transports[transportId], dst[rankIdx], src[rankIdx], sliceSize, localSignal_, 1 << rankIdx);
                transportId++;
            }
        }
        CCU_IF(sliceSize == 0) { LocalPost(localSignal_, 1 << rankIdx); }
    }

    LocalWait(localSignal_, (1 << rankSize_) - 1);
}

void CcuContextBroadcastMesh1DMem2Mem::DoAllGather(const CcuRep::Memory& src, const std::vector<CcuRep::Memory>& dst)
{
    uint64_t transportId = 0;
    auto& sliceSize = (rankId_ + 1 == rankSize_) ? lastSliceSize_ : normalSliceSize_;
    CCU_IF(sliceSize != 0)
    {
        for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            if (rankIdx == rankId_) {
                LocalPost(localSignal_, (1 << rankIdx));
            } else {
                Write(*transports[transportId], dst[rankIdx], src, sliceSize, localSignal_, 1 << rankIdx);
                transportId++;
            }
        }
        LocalWait(localSignal_, (1 << rankSize_) - 1);
    }
}

void CcuContextBroadcastMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuContextBroadcastMesh1DMem2Mem] BroadcastMesh1D run");
    InitResource();
    LoadArgs();
    PreSync();

    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd = 1;

    flag_ = 0;
    CCU_WHILE(repeatNumVar_ != UINT64_MAX)
    { // 循环repeatNum_次
        DoRepeaScatterMem2Mem();
        PostSync(CKE_IDX_4);
        DoRepeatAllGatherMem2Mem();
        PostSync(CKE_IDX_0);
        repeatNumVar_ += repeatNumAdd;
        flag_ = 1;
    }

    HCCL_INFO("[CcuContextBroadcastMesh1DMem2Mem] BroadcastMesh1D end");
    return;
}

std::vector<uint64_t> CcuContextBroadcastMesh1DMem2Mem::GeneArgs(const CcuTaskArg& arg)
{
    const CurrentTaskArg* taskArg = dynamic_cast<const CurrentTaskArg*>(&arg);
    // 空指针校验
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextBroadcastMesh1DMem2Mem::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;

    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset = taskArg->outputSliceStride_ * rankId_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize = taskArg->normalSliceSize_;
    uint64_t lastSliceSize = taskArg->lastSliceSize_;
    uint64_t allgatherOffset = taskArg->normalSliceSize_ * rankId_;
    uint64_t repeatNumVar = taskArg->repeatNumVar_;
    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        tokenInfo,
        currentRankSliceInputOffset,
        currentRankSliceOutputOffset,
        inputRepeatStride,
        outputRepeatStride,
        normalSliceSize,
        lastSliceSize,
        allgatherOffset,
        repeatNumVar,
    };

    HCCL_INFO(
        "[CcuContextBroadcastMesh1DMem2Mem] TaskArgs: inputAddr[%llx], outputAddr[%llx], "
        "currentRankSliceInputOffset[%llu], "
        "currentRankSliceOutputOffset[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], "
        "normalSliceSize[%llu], lastSliceSize[%llu], allgatherOffset[%llu], repeatNumVar[%llu] ",
        inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, inputRepeatStride,
        outputRepeatStride, normalSliceSize, lastSliceSize, allgatherOffset, repeatNumVar);

    return taskArgs;
}
} // namespace Hccl
