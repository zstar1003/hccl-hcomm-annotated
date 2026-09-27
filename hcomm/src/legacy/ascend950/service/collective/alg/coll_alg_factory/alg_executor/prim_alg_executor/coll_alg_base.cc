/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_alg_base.h"

namespace Hccl {

CollAlgBase::CollAlgBase() {}

CollAlgBase::~CollAlgBase() {}

void CollAlgBase::SetMyRank(RankId myRank)
{
    myRank_ = myRank;
    return;
}

void CollAlgBase::SetRankSize(u32 rankSize)
{
    rankSize_ = rankSize;
    return;
}

void CollAlgBase::SetDevType(DevType devType)
{
    devType_ = devType;
    return;
}

void CollAlgBase::SetAllignSize(u64 allignSize)
{
    allignSize_ = allignSize;
    return;
}

void CollAlgBase::EnableDataAllign(bool enableAllign)
{
    enableAllign_ = enableAllign;
    return;
}

void CollAlgBase::EnableDetour(bool enableDetour)
{
    enableDetour_ = enableDetour;
    return;
}

void CollAlgBase::SetDmaMode(const DmaMode dmaMode)
{
    dmaMode_ = dmaMode;
    return;
}

bool CollAlgBase::IsEnableCounterNotify() const { return IsEnableCounterNotifyByDevType(myRank_, devType_); }

HcclResult CollAlgBase::Init(const CollAlgOperator& op, const CollAlgParams& params, PrimQuePtr primQue)
{
    // init params
    CHK_PRT_RET(
        InitParams(op, params) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Fail to init params.", myRank_), HcclResult::HCCL_E_PARA);

    // init queMap
    CHK_PRT_RET(
        GenPrimQueMap(primQue) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Fail to init primQueMap.", myRank_), HcclResult::HCCL_E_PARA);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollAlgBase::InitParams(const CollAlgOperator& op, const CollAlgParams& params)
{
    opMode_ = params.opMode;
    maxTmpMemSize_ = (opMode_ == OpMode::OPBASE) ? params.maxTmpMemSize : 0;

    CHK_PRT_RET(
        (maxTmpMemSize_ == 0) && (opMode_ == OpMode::OPBASE),
        HCCL_ERROR("[CollAlgFactory] maxTmpMemSize equals to zero for OPBASE."), HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        InitDataInfo(op, dataType_, outputDataType_, dataCount_),
        HCCL_ERROR("[CollAlgFactory] unable to init DataInfo."), HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        InitOpInfo(op, opType_, redOp_, root_), HCCL_ERROR("[CollAlgFactory] unable to init OpInfo."),
        HcclResult::HCCL_E_PARA);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollAlgBase::GenPrimQueMap(PrimQuePtr primQue)
{
    CHK_PRT_RET(
        !primQue->IsMaster(),
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Input Primitive Queue is not a master queue.", myRank_),
        HcclResult::HCCL_E_PARA);
    queId2PrimQue_.insert(std::make_pair(primQue->GetId(), primQue));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollAlgBase::InitQueue(const u32& requiredQueNum, std::vector<PrimQuePtr>& requiredQue)
{
    CHK_PRT_RET(
        !static_cast<bool>(queId2PrimQue_.count(0)),
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid queId2PrimQue Map.", myRank_), HcclResult::HCCL_E_INTERNAL);
    PrimQuePtr primQue = queId2PrimQue_[0];

    for (u32 queIdx = 0; queIdx < requiredQueNum; queIdx++) {
        if (!static_cast<bool>(queId2PrimQue_.count(queIdx))) {
            queId2PrimQue_.insert(std::make_pair(queIdx, primQue->Fork()));
        }
        requiredQue.push_back(queId2PrimQue_[queIdx]);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollAlgBase::SetLinkPrty(const std::vector<BasePortType>& linkPriority)
{
    CHK_PRT_RET(
        linkPriority.size() == 0, HCCL_ERROR("[CollAlgFactory] Invalid given link priority."), HcclResult::HCCL_E_PARA);
    linkPriority_.assign(linkPriority.begin(), linkPriority.end());

    return HcclResult::HCCL_SUCCESS;
}

LinkReq CollAlgBase::GetSeqLinksUnion(const LinkReq& linkReq0, const LinkReq& linkReq1) const
{
    LinkReq retLinkReq = linkReq0;
    for (auto linkReqIter = linkReq1.begin(); linkReqIter != linkReq1.end(); linkReqIter++) {
        if (retLinkReq.find(linkReqIter->first) == retLinkReq.end()) {
            retLinkReq.insert(std::pair<RankId, u32>(linkReqIter->first, linkReqIter->second));
        } else {
            u32 tmpLinkReq = retLinkReq[linkReqIter->first];
            retLinkReq[linkReqIter->first] = std::max(tmpLinkReq, linkReqIter->second);
        }
    }

    return retLinkReq;
}

HcclResult
CollAlgBase::AllocTempResLinks(const ResLinks& execResLinks, const LinkReq& tempLinkReq, ResLinks& tempResLinks) const
{
    for (auto resLinkReqIter = tempLinkReq.begin(); resLinkReqIter != tempLinkReq.end(); resLinkReqIter++) {
        auto execResLinkIter = execResLinks.find(resLinkReqIter->first);
        CHK_PRT_RET(
            execResLinkIter == execResLinks.end(),
            HCCL_ERROR("[CollAlgFactory] Rank [%d], required link not in provided resLinks.", myRank_),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            execResLinkIter->second.size() < (resLinkReqIter->second),
            HCCL_ERROR("[CollAlgFactory] Rank [%d], provided linkNum smaller than required.", myRank_),
            HcclResult::HCCL_E_INTERNAL);
        std::vector<LinkData> resLinkVec(
            execResLinkIter->second.begin(), execResLinkIter->second.begin() + resLinkReqIter->second);
        tempResLinks.insert(std::pair<RankId, std::vector<LinkData>>(resLinkReqIter->first, resLinkVec));
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
