/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_alg_template_base.h"
#include "ccu_context_utils.h"
#include "ccu_assist.h"
#include "log.h"

namespace Hccl {
CcuAlgTemplateBase::CcuAlgTemplateBase(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : myRank_(virtualRank),
      tempRankSize_(tempRankSize),
      tempVTopo_(tempVTopo),
      tempVirtRankMap_(tempVirtRankMap)
{}

CcuAlgTemplateBase::~CcuAlgTemplateBase() {}

HcclResult CcuAlgTemplateBase::CalcRes(AlgTempResReq& tempResReq)
{
    (void)tempResReq;
    HCCL_ERROR("[CcuAlgTemplateBase] [CalcRes] Current alg do not support detour mode!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CcuAlgTemplateBase::CalcResDetour(const RankGraph* rankGraph, AlgTempResReq& tempResReq)
{
    (void)rankGraph;
    (void)tempResReq;
    HCCL_ERROR("[CcuAlgTemplateBase] [CalcRes] Current alg do not support detour mode!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CcuAlgTemplateBase::CalcResDetour(ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq)
{
    (void)linkMgr;
    (void)tempResReq;
    HCCL_ERROR("[CcuAlgTemplateBase] [CalcRes] Current alg do not support detour mode!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CcuAlgTemplateBase::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempFuncs;
    (void)sliceInfoVec;
    (void)buffInfo;
    (void)tempLinks;
    (void)tempInsQues;
    HCCL_ERROR("[CcuAlgTemplateBase] Unsupported interface of CcuAlgTemplateBase::Run!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CcuAlgTemplateBase::SetScratchBufferSize(uint64_t size)
{
    scratchBufferSize_ = size;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuAlgTemplateBase::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    (void)allignInfo;
    (void)dataSize;
    (void)sliceInfoVec;
    HCCL_WARNING("[CcuAlgTemplateBase] Interface of CcuAlgTemplateBase::CalcSliceInfo is not implemented!");
    return HcclResult::HCCL_SUCCESS;
}

void CcuAlgTemplateBase::SetDmaMode(const DmaMode dmaMode)
{
    dmaMode_ = dmaMode;
    return;
}

void CcuAlgTemplateBase::SetCollOp(const CollAlgOperator& op)
{
    op_ = op;
    return;
}

void CcuAlgTemplateBase::SetDataType(const DataType& dataType)
{
    dataType_ = dataType;
    return;
}

HcclResult CcuAlgTemplateBase::GetScratchBufferInfo(const uint64_t scratchBufferSize, DataType dataType)
{
    (void)scratchBufferSize;
    (void)dataType;
    return HcclResult::HCCL_SUCCESS;
}

void CcuAlgTemplateBase::SetRoot(const u32 root)
{
    rootId_ = root;
    return;
}

void CcuAlgTemplateBase::SetLoadInfo(const CollAlgParams& params)
{
    loadFromMem_ = params.isMc2; // 当前只有mc2场景会设置该标记，故暂作为mc2标记使用
    return;
}

u64 CcuAlgTemplateBase::CalcLoopMaxCount(ParamPool& paramPool)
{
    u64 loopMaxCount = 0;
    if (paramPool.params.opMode == OpMode::OPBASE) {
        u64 maxLoopSize
            = std::min(static_cast<u64>(paramPool.params.maxTmpMemSize), static_cast<u64>(UB_MAX_DATA_SIZE));
        loopMaxCount = maxLoopSize / (DataTypeSizeGet(paramPool.op.dataType) * tempRankSize_) * tempRankSize_;
    } else {
        loopMaxCount = paramPool.op.dataCount;
    }
    return loopMaxCount;
}

HcclResult CcuAlgTemplateBase::GetToken(const CollAlgOperator& op, uint64_t& token) const
{
    if (op.inputMem != nullptr && op.inputMem->GetAddr() != 0) {
        token = CcuRep::GetTokenInfo(
            static_cast<uint64_t>(op.inputMem->GetAddr()), static_cast<uint64_t>(op.inputMem->GetSize()));
        return HCCL_SUCCESS;
    } else if (op.outputMem != nullptr && op.outputMem->GetAddr() != 0) {
        token = CcuRep::GetTokenInfo(
            static_cast<uint64_t>(op.outputMem->GetAddr()), static_cast<uint64_t>(op.outputMem->GetSize()));
        return HCCL_SUCCESS;
    } else if (op.scratchMem != nullptr && op.scratchMem->GetAddr() != 0) {
        token = CcuRep::GetTokenInfo(
            static_cast<uint64_t>(op.scratchMem->GetAddr()), static_cast<uint64_t>(op.scratchMem->GetSize()));
        return HCCL_SUCCESS;
    }
    HCCL_WARNING("[GetToken] Both inputMem and outputMem are null");
    return HCCL_E_PTR;
}
u32 CcuAlgTemplateBase::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

HcclResult CcuAlgTemplateBase::GetMaxTransPortDataSize(u64& maxTransPortDataSize) const
{
    maxTransPortDataSize = MAX_LOOP_GROUP_TRANS_SIZE;
    return HCCL_SUCCESS;
}

uint64_t CcuAlgTemplateBase::BufferTypeToAddr(const BufferType bufferType)
{
    if (bufferType == BufferType::INPUT && op_.inputMem != nullptr) {
        return static_cast<uint64_t>(op_.inputMem->GetAddr());
    } else if (bufferType == BufferType::OUTPUT && op_.outputMem != nullptr) {
        return static_cast<uint64_t>(op_.outputMem->GetAddr());
    } else if (bufferType == BufferType::SCRATCH && op_.scratchMem != nullptr) {
        return static_cast<uint64_t>(op_.scratchMem->GetAddr());
    } else {
        return 0;
    }
}

HcclResult CcuAlgTemplateBase::AddRanksToGroup(
    const std::vector<std::vector<RankId>>& tempVTopo, RankGroup& rankGroupX, RankGroup& rankGroupY) const
{
    for (auto& peer : tempVTopo[0]) {
        rankGroupX.AddRank(peer);
    }

    for (auto& peer : tempVTopo[1]) {
        rankGroupY.AddRank(peer);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuAlgTemplateBase::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)numBlocks;
    (void)dataSize;
    (void)numBlocksLimit;
    HCCL_WARNING("CalNumBlocks does not support ccu template.");
    return HCCL_SUCCESS;
}

HcclResult CcuAlgTemplateBase::setPathNumMap(const std::map<u32, u32>& rank2PathNumMap) const
{
    (void)rank2PathNumMap;
    HCCL_WARNING("[CcuAlgTemplateBase] Unsupported interface of setPathNumMap!");
    return HCCL_SUCCESS;
}

} // namespace Hccl
