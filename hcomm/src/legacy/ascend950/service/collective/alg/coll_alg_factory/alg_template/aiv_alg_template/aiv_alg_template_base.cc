/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_alg_template_base.h"

namespace Hccl {

AivAlgTemplateBase::AivAlgTemplateBase(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : myRank_(virtualRank),
      tempRankSize_(tempRankSize),
      tempVTopo_(tempVTopo),
      tempVirtRankMap_(tempVirtRankMap)
{}

AivAlgTemplateBase::~AivAlgTemplateBase() {}

void AivAlgTemplateBase::SetCollOp(const CollAlgOperator& op)
{
    op_ = op;
    return;
}

void AivAlgTemplateBase::SetDmaMode(const DmaMode dmaMode)
{
    dmaMode_ = dmaMode;
    return;
}

void AivAlgTemplateBase::InitReduceInfo(const ReduceOp& redOp, const DataType& dataType)
{
    reduceOp_ = redOp;
    dataType_ = dataType;
    return;
}

void AivAlgTemplateBase::SetDataType(const DataType& dataType)
{
    dataType_ = dataType;
    return;
}

void AivAlgTemplateBase::SetRoot(const u32 root)
{
    root_ = root;
    return;
}

HcclResult AivAlgTemplateBase::CalcRes(AlgTempResReq& tempResReq)
{
    (void)tempResReq;
    HCCL_ERROR("[AivAlgTemplateBase] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult AivAlgTemplateBase::CalcResDetour(const RankGraph* rankGraph, AlgTempResReq& tempResReq)
{
    (void)rankGraph;
    (void)tempResReq;
    HCCL_ERROR("[AivAlgTemplateBase] Current alg do not support detour mode!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult AivAlgTemplateBase::CalcResDetour(ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq)
{
    (void)linkMgr;
    (void)tempResReq;
    HCCL_ERROR("[AivAlgTemplateBase] Current alg do not support detour mode!");
    return HcclResult::HCCL_E_INTERNAL;
}

u32 AivAlgTemplateBase::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // AIV模式默认返回一整块Scratch
    return 1;
}

HcclResult AivAlgTemplateBase::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempFuncs;
    (void)templateDataParams;
    (void)tempLinks;
    (void)tempInsQues;
    HCCL_ERROR("[AivAlgTemplateBase] Unsupported interface of instruction generation!");
    return HcclResult::HCCL_E_INTERNAL;
}

void AivAlgTemplateBase::IncSliceId()
{
    sliceId_++;
    return;
}

HcclResult AivAlgTemplateBase::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;
    if (numBlocksLimit >= tempRankSize_) {
        numBlocks = tempRankSize_;
    } else {
        numBlocks = numBlocksLimit;
    }
    HCCL_INFO("[AivAlgTemplateBase] Actually use core num[%u]", numBlocks);
    return HCCL_SUCCESS;
}

HcclResult AivAlgTemplateBase::setPathNumMap(const std::map<u32, u32>& rank2PathNumMap) const
{
    (void)rank2PathNumMap;
    HCCL_WARNING("[AivAlgTemplateBase] Unsupported interface of setPathNumMap!");
    return HCCL_SUCCESS;
}

} // namespace Hccl
