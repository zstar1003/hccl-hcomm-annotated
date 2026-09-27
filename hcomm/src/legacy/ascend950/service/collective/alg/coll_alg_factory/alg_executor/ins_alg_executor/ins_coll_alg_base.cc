/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"

#include "ins_coll_alg_base.h"

namespace Hccl {

InsCollAlgBase::InsCollAlgBase() {}

InsCollAlgBase::~InsCollAlgBase() {}

void InsCollAlgBase::SetMyRank(RankId myRank)
{
    myRank_ = myRank;
    return;
}

void InsCollAlgBase::SetRankSize(u32 rankSize)
{
    rankSize_ = rankSize;
    return;
}

void InsCollAlgBase::SetDevType(DevType devType)
{
    devType_ = devType;
    return;
}

void InsCollAlgBase::SetSendRecvRemoteRank(RankId sendRecvRemoteRank)
{
    sendRecvRemoteRank_ = sendRecvRemoteRank;
    return;
}

void InsCollAlgBase::SetOp(const CollAlgOperator& op)
{
    op_ = op;
    return;
}

void InsCollAlgBase::SetAllignSize(u64 allignSize)
{
    allignSize_ = allignSize;
    return;
}

void InsCollAlgBase::EnableDataAllign(bool enableAllign)
{
    enableAllign_ = enableAllign;
    return;
}

void InsCollAlgBase::EnableDetour(bool enableDetour)
{
    enableDetour_ = enableDetour;
    return;
}

void InsCollAlgBase::SetDmaMode(const DmaMode dmaMode)
{
    dmaMode_ = dmaMode;
    return;
}

void InsCollAlgBase::SetRmaDataBufferMgr(const RmtDataBufferMgr* rmaDataBufferMgr)
{
    (void)rmaDataBufferMgr;
    return;
}

bool InsCollAlgBase::IsEnableCounterNotify() const
{
    HCCL_DEBUG("[InsCollAlgBase][%s] start.", __func__);
    return true;
}

HcclResult InsCollAlgBase::Init(const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    // init params
    CHK_PRT_RET(
        InitParams(op, params) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[InsCollAlgFactory] Rank [%d], Fail to init params.", myRank_), HcclResult::HCCL_E_PARA);

    // init queMap
    CHK_PRT_RET(
        GenInsQueMap(insQue) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[InsCollAlgFactory] Rank [%d], Fail to init insQueMap.", myRank_), HcclResult::HCCL_E_PARA);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsCollAlgBase::InitParams(const CollAlgOperator& op, const CollAlgParams& params)
{
    op_ = op;
    opMode_ = params.opMode;
    maxTmpMemSize_ = params.maxTmpMemSize;

    CHK_PRT_RET(
        (maxTmpMemSize_ == 0) && (opMode_ == OpMode::OPBASE),
        HCCL_ERROR("[InsCollAlgFactory] maxTmpMemSize equals to zero for OPBASE."), HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        InitDataInfo(op, dataType_, outputDataType_, dataCount_),
        HCCL_ERROR("[InsCollAlgFactory] unable to init DataInfo."), HcclResult::HCCL_E_PARA);
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataTypeSize_;

    CHK_PRT_RET(
        InitOpInfo(op, opType_, redOp_, root_), HCCL_ERROR("[InsCollAlgFactory] unable to init OpInfo."),
        HcclResult::HCCL_E_PARA);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsCollAlgBase::GenInsQueMap(InsQuePtr insQue)
{
    if (insQue == nullptr) {
        HCCL_ERROR("[InsCollAlgBase] insQue is nullptr.");
        return HcclResult::HCCL_E_PTR;
    }
    CHK_PRT_RET(
        !insQue->IsMaster(),
        HCCL_ERROR("[InsCollAlgFactory] Rank [%d], Input Primitive Queue is not a master queue.", myRank_),
        HcclResult::HCCL_E_PARA);
    queId2InsQue_.insert(std::make_pair(insQue->GetId(), insQue));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsCollAlgBase::InitQueue(const u32& requiredQueNum, std::vector<InsQuePtr>& requiredQue)
{
    CHK_PRT_RET(
        !static_cast<bool>(queId2InsQue_.count(0)),
        HCCL_ERROR("[InsCollAlgFactory] Rank [%d], Invalid queId2InsQue Map.", myRank_), HcclResult::HCCL_E_INTERNAL);
    InsQuePtr insQue = queId2InsQue_[0];

    for (u32 queIdx = 0; queIdx < requiredQueNum; queIdx++) {
        if (!static_cast<bool>(queId2InsQue_.count(queIdx))) {
            queId2InsQue_.insert(std::make_pair(queIdx, insQue->Fork()));
        }
        requiredQue.push_back(queId2InsQue_[queIdx]);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsCollAlgBase::SetLinkPrty(const std::vector<BasePortType>& linkPriority)
{
    CHK_PRT_RET(
        linkPriority.size() == 0, HCCL_ERROR("[InsCollAlgFactory] Invalid given link priority."),
        HcclResult::HCCL_E_PARA);
    linkPriority_.assign(linkPriority.begin(), linkPriority.end());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsCollAlgBase::CalcParallelNotifyReq(
    const u32 primQueueNum, const u32 IntraqueNum, std::vector<std::tuple<QId, QId, u32>>& queueNotifys) const
{
    u32 slaveNum = primQueueNum - 1;
    std::vector<std::tuple<QId, QId, u32>> notifyRequests;

    notifyRequests.reserve(slaveNum); // 每个从流需要1个
    for (QId q = 1; q < primQueueNum; q++) {
        notifyRequests.emplace_back(std::make_tuple(0, q, 0));
        notifyRequests.emplace_back(std::make_tuple(q, 0, 0));
    }
    for (QId q = IntraqueNum; q < primQueueNum; q++) {
        if (IntraqueNum == q) {
            continue;
        }
        notifyRequests.emplace_back(std::make_tuple(IntraqueNum, q, 0));
        notifyRequests.emplace_back(std::make_tuple(q, IntraqueNum, 0));
    }
    queueNotifys = notifyRequests;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsCollAlgBase::CalcLocalRankSize(
    const RankId myRank, const std::vector<std::vector<RankId>>& virtRanks, u32& rankSizeLevel0,
    u32& rankSizeLevel1) const
{
    constexpr uint64_t virtRanks_2 = 2;
    CHK_PRT_RET(
        virtRanks.size() < virtRanks_2, HCCL_ERROR("[CalcLocalRankSize] virtRanks level num is smaller than 2."),
        HcclResult::HCCL_E_INTERNAL);

    rankSizeLevel0 = virtRanks.at(0).size();
    rankSizeLevel1 = virtRanks.at(1).size();

    HCCL_INFO(
        "[CalcLocalRankSize] localRankSize: myRank[%d] rankSizeLevel0_[%u] rankSizeLevel1_[%u]", myRank, rankSizeLevel0,
        rankSizeLevel1);
    return HcclResult::HCCL_SUCCESS;
};

LinkReq InsCollAlgBase::GetSeqLinksUnion(const LinkReq& linkReq0, const LinkReq& linkReq1) const
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

HcclResult InsCollAlgBase::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    numBlocks = 0;
    (void)dataSize;
    (void)numBlocksLimit;
    HCCL_INFO("[InsCollAlgFactory] current executor doesn't support controlling num of aiv cores.");
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
