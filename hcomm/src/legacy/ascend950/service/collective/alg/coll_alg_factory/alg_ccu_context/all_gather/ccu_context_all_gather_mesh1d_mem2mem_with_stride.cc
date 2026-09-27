/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_gather_mesh1d_mem2mem_with_stride.h"
#include "ccu_instruction_all_gather_mesh1d_mem2mem_with_stride.h"

namespace Hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

using CurrentCtxArg = CcuCtxArgAllGatherMesh1DMem2MemWithStride;
using CurrentTaskArg = CcuTaskArgAllGatherMesh1DMem2MemWithStride;

CcuContextAllGatherMesh1DMem2MemWithStride::CcuContextAllGatherMesh1DMem2MemWithStride(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    HCCL_DEBUG("[CcuContextAllGatherMesh1DMem2MemWithStride] Enter Constructor.");
    const CurrentCtxArg* ctxArg = dynamic_cast<const CurrentCtxArg*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllGatherMesh1DMem2MemWithStride::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    if (ctxArg->dimSize_.size() > 0) {
        rankSize_ = ctxArg->dimSize_[0];
    }
    HCCL_INFO("[CcuContextAllGatherMesh1DMem2MemWithStride] CtxArg: rankId[%u] rankSize[%u].", rankId_, rankSize_);
}

void CcuContextAllGatherMesh1DMem2MemWithStride::InitResource()
{
    localInput_ = CreateVariable();
    uint16_t transportIdx = 0;
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG(
                "[CcuContextAllGatherMesh1DMem2MemWithStride] MyRank[%u], PeerId[%u], TransportId[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_THROW(
                transports.at(transportIdx) == nullptr,
                HCCL_ERROR(
                    "[CcuContextAllGatherMesh1DMem2MemWithStride][InitResource] transports[%u] is nullptr",
                    transportIdx),
                NullPtrException, "transport is null");
            output_.push_back(
                CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID)); // 获取transport中id=1的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }

    currentRankSliceInputOffset_ = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    tmpRepeatNum_ = CreateVariable();
    normalSliceSize_ = CreateVariable();
    lastSliceSize_ = CreateVariable();
    isInputOutputEqual_ = CreateVariable();
    repeatTimeflag_ = CreateVariable();
    repeatTimeflag_ = 0;
    constVar1_ = CreateVariable();
    constVar1_ = 1;

    selfBit_ = 1 << rankId_;
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    localMem_ = CreateMemory();
    reomteMem_.reserve(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reomteMem_.push_back(CreateMemory());
    }

    localSignal_ = CreateMaskSignal();
    return;
}

void CcuContextAllGatherMesh1DMem2MemWithStride::LoadArgs()
{
    Load(localInput_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(tmpRepeatNum_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(isInputOutputEqual_);
    return;
}

void CcuContextAllGatherMesh1DMem2MemWithStride::PreSync()
{
    for (auto t : transports) {
        WriteVariableWithSignal(*t, output_[rankId_], OUTPUT_XN_ID, CKE_IDX_1, selfBit_); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_2, selfBit_); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit_); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit_); // index = 2，传递token信息
    return;
}

void CcuContextAllGatherMesh1DMem2MemWithStride::PostSync()
{
    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit_);
}

void CcuContextAllGatherMesh1DMem2MemWithStride::DoRepeatAllGather()
{
    CcuRep::Memory& src = localMem_;
    std::vector<CcuRep::Memory>& dst = reomteMem_;
    //  初始化 src 和 dst
    src.addr = localInput_;
    src.addr += currentRankSliceInputOffset_;
    src.token = token_[rankId_];
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        dst[rankIdx].addr = output_[rankIdx];
        dst[rankIdx].addr += currentRankSliceOutputOffset_;
        dst[rankIdx].token = token_[rankIdx];
    }
    CCU_WHILE(tmpRepeatNum_ != UINT64_MAX)
    {
        tmpRepeatNum_ += constVar1_;
        CCU_IF(repeatTimeflag_ != 0)
        {
            src.addr += inputRepeatStride_;
            for (auto& d : dst) {
                d.addr += outputRepeatStride_;
            }
        }
        CCU_IF(normalSliceSize_ != 0) { DoAllGather(src, dst, normalSliceSize_); }
        repeatTimeflag_ = 1;
    }
}

void CcuContextAllGatherMesh1DMem2MemWithStride::DoAllGather(
    const CcuRep::Memory& src, const std::vector<CcuRep::Memory>& dst, const CcuRep::Variable& sliceSize)
{
    uint32_t transportId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            CCU_IF(isInputOutputEqual_ != 0) { LocalPost(localSignal_, 1 << rankIdx); }
            CCU_IF(isInputOutputEqual_ == 0) { LocalCopy(dst[rankIdx], src, sliceSize, localSignal_, 1 << rankIdx); }
        } else {
            CCU_IF(normalSliceSize_ != 0)
            {
                Write(*transports[transportId], dst[rankIdx], src, sliceSize, localSignal_, 1 << rankIdx);
            }
            transportId++;
        }
    }
    LocalWait(localSignal_, (1 << rankSize_) - 1);
}

void CcuContextAllGatherMesh1DMem2MemWithStride::Algorithm()
{
    HCCL_INFO("[CcuContextAllGatherMesh1DMem2MemWithStride] AllgatherMesh1D run.");
    InitResource();
    LoadArgs();
    PreSync();
    DoRepeatAllGather();
    PostSync();
    HCCL_INFO("[CcuContextAllGatherMesh1DMem2MemWithStride] AllgatherMesh1D end.");
    return;
}

std::vector<uint64_t> CcuContextAllGatherMesh1DMem2MemWithStride::GeneArgs(const CcuTaskArg& arg)
{
    const CurrentTaskArg* taskArg = dynamic_cast<const CurrentTaskArg*>(&arg);
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;

    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset = taskArg->outputSliceStride_ * rankId_;
    uint64_t tmpRepeatNum = UINT64_MAX - taskArg->repeatNum_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize = taskArg->normalSliceSize_;
    uint64_t lastSliceSize = taskArg->lastSliceSize_;
    uint64_t isInputOutputEqual = taskArg->isInputOutputEqual_;

    std::vector<uint64_t> taskArgs
        = {inputAddr,
           outputAddr,
           tokenInfo,
           currentRankSliceInputOffset,
           currentRankSliceOutputOffset,
           tmpRepeatNum,
           inputRepeatStride,
           outputRepeatStride,
           normalSliceSize,
           lastSliceSize,
           isInputOutputEqual};

    HCCL_INFO(
        "[CcuContextAllGatherMesh1DMem2MemWithStride] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], "
        "repeatNum[%llu],inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu]",
        inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, tmpRepeatNum,
        inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize);
    return taskArgs;
}
} // namespace Hccl
