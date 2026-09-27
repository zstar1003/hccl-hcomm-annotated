/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_alg_template_base.h"
#include "log.h"

namespace Hccl {

InsAlgTemplateBase::InsAlgTemplateBase(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : myRank_(virtualRank),
      tempRankSize_(tempRankSize),
      tempVTopo_(tempVTopo),
      tempVirtRankMap_(tempVirtRankMap)
{}

InsAlgTemplateBase::~InsAlgTemplateBase() {}

void InsAlgTemplateBase::SetCollOp(const CollAlgOperator& op)
{
    op_ = op;
    return;
}

void InsAlgTemplateBase::SetDmaMode(const DmaMode dmaMode)
{
    dmaMode_ = dmaMode;
    return;
}

void InsAlgTemplateBase::SetRoot(const u32 root)
{
    root_ = root;
    return;
}

u64 InsAlgTemplateBase::CalcLoopMaxCount(ParamPool& paramPool)
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

HcclResult InsAlgTemplateBase::PostCopyOpbase(const UsrData& usrData, std::vector<InsQuePtr>& tempInsQues) const
{
    for (size_t i = 0; i < usrData.scratchOutSlices.size(); i++) {
        std::unique_ptr<Instruction> insLocalCopy
            = std::make_unique<InsLocalCopy>(usrData.scratchOutSlices[i], usrData.usrOutSlices[i]);
        tempInsQues[0]->Append(std::move(insLocalCopy));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsAlgTemplateBase::PreCopyOpbase(const UsrData& usrData, std::vector<InsQuePtr>& tempInsQues) const
{
    for (size_t i = 0; i < usrData.usrInSlices.size(); i++) {
        std::unique_ptr<Instruction> insLocalCopy
            = std::make_unique<InsLocalCopy>(usrData.usrInSlices[i], usrData.scratchInSlices[i]);
        tempInsQues[0]->Append(std::move(insLocalCopy));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsAlgTemplateBase::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    (void)allignInfo;
    (void)dataSize;
    (void)sliceInfoVec;
    HCCL_ERROR("[InsCollAlgFactory] Unsupported interface of slice info calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::CalcRes(AlgTempResReq& tempResReq)
{
    (void)tempResReq;
    HCCL_ERROR("[InsCollAlgFactory] Unsupported interface of resource calculation!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::CalcResDetour(const RankGraph* rankGraph, AlgTempResReq& tempResReq)
{
    (void)rankGraph;
    (void)tempResReq;
    HCCL_ERROR("[InsCollAlgFactory] Current algorithm does not support detour mode!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult InsAlgTemplateBase::CalcResDetour(ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq)
{
    (void)linkMgr;
    (void)tempResReq;
    HCCL_ERROR("[InsCollAlgFactory] Current algorithm does not support detour mode!");
    return HcclResult::HCCL_E_INTERNAL;
}

uint64_t InsAlgTemplateBase::GetMaxSliceSize()
{
    return UB_MAX_DATA_SIZE; //  return max value
}

void InsAlgTemplateBase::InitReduceInfo(const ReduceOp& redOp, const DataType& dataType)
{
    redOp_ = redOp;
    dataType_ = dataType;
    return;
}

HcclResult InsAlgTemplateBase::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempFuncs;
    (void)sliceInfoVec;
    (void)buffInfo;
    (void)tempLinks;
    (void)tempInsQues;
    HCCL_ERROR("[InsAlgTemplateBase] Unsupported interface of GenInsQue!");
    return HcclResult::HCCL_E_INTERNAL;
}

void InsAlgTemplateBase::SetDataType(const DataType& dataType)
{
    dataType_ = dataType;
    return;
}

void InsAlgTemplateBase::SetReduceOp(const ReduceOp& redOp)
{
    redOp_ = redOp;
    return;
}

HcclResult InsAlgTemplateBase::PreSync(const u32 queIdx, std::vector<InsQuePtr>& syncInsQues) const
{
    InsQuePtr currInsQue = syncInsQues[queIdx];
    if (queIdx == 0) {
        // Semaphore Post
        if (enableCounterNotify_) {
            std::unique_ptr<InsLocalBcastPost> insLocalBcastPost = std::make_unique<InsLocalBcastPost>(0);
            for (size_t qidx = 1; qidx < syncInsQues.size(); qidx++) {
                insLocalBcastPost->Append(syncInsQues[qidx]->GetId());
            }
            CHK_PTR_NULL(insLocalBcastPost);
            currInsQue->Append(std::move(insLocalBcastPost));
        } else {
            for (size_t qidx = 1; qidx < syncInsQues.size(); qidx++) {
                std::unique_ptr<Instruction> insLocalPostTo
                    = std::make_unique<InsLocalPostTo>(syncInsQues[qidx]->GetId());
                CHK_PTR_NULL(insLocalPostTo);
                currInsQue->Append(std::move(insLocalPostTo));
            }
        }
    } else {
        // Semaphore Wait
        if (enableCounterNotify_) {
            std::unique_ptr<Instruction> insLocalWaitFrom
                = std::make_unique<InsLocalWaitFrom>(syncInsQues[0]->GetId(), NotifyType::COUNTER);
            CHK_PTR_NULL(insLocalWaitFrom);
            currInsQue->Append(std::move(insLocalWaitFrom));
        } else {
            std::unique_ptr<Instruction> insLocalWaitFrom = std::make_unique<InsLocalWaitFrom>(syncInsQues[0]->GetId());
            CHK_PTR_NULL(insLocalWaitFrom);
            currInsQue->Append(std::move(insLocalWaitFrom));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsAlgTemplateBase::PostSync(const u32 queIdx, std::vector<InsQuePtr>& syncInsQues) const
{
    InsQuePtr currInsQue = syncInsQues[queIdx];
    if (queIdx == 0) {
        // Semaphore Wait
        if (enableCounterNotify_) {
            std::unique_ptr<InsLocalWaitGroup> insLocalWaitGroup = std::make_unique<InsLocalWaitGroup>(0);
            for (size_t qidx = 1; qidx < syncInsQues.size(); qidx++) {
                insLocalWaitGroup->Append(syncInsQues[qidx]->GetId());
            }
            CHK_PTR_NULL(insLocalWaitGroup);
            currInsQue->Append(std::move(insLocalWaitGroup));
        } else {
            for (size_t qidx = 1; qidx < syncInsQues.size(); qidx++) {
                std::unique_ptr<Instruction> insLocalWaitFrom
                    = std::make_unique<InsLocalWaitFrom>(syncInsQues[qidx]->GetId());
                CHK_PTR_NULL(insLocalWaitFrom);
                currInsQue->Append(std::move(insLocalWaitFrom));
            }
        }
    } else {
        // Semaphore Post
        if (enableCounterNotify_) {
            std::unique_ptr<Instruction> insLocalPostTo
                = std::make_unique<InsLocalPostTo>(syncInsQues[0]->GetId(), NotifyType::COUNTER);
            CHK_PTR_NULL(insLocalPostTo);
            currInsQue->Append(std::move(insLocalPostTo));
        } else {
            std::unique_ptr<Instruction> insLocalPostTo = std::make_unique<InsLocalPostTo>(syncInsQues[0]->GetId());
            CHK_PTR_NULL(insLocalPostTo);
            currInsQue->Append(std::move(insLocalPostTo));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsAlgTemplateBase::PreSyncInterQueues(std::vector<InsQuePtr>& syncInsQues) const
{
    for (size_t queIdx = 0; queIdx < syncInsQues.size(); queIdx++) {
        CHK_PRT_RET(
            PreSync(queIdx, syncInsQues) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[InsCollAlgFactory] Rank [%d], Que [%u], Semaphore Synchronization Failed.", myRank_,
                syncInsQues[queIdx]->GetId()),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsAlgTemplateBase::PostSyncInterQueues(std::vector<InsQuePtr>& syncInsQues) const
{
    for (size_t queIdx = 0; queIdx < syncInsQues.size(); queIdx++) {
        CHK_PRT_RET(
            PostSync(queIdx, syncInsQues) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[InsCollAlgFactory] Rank [%d], Que [%u], Semaphore Synchronization Failed.", myRank_,
                syncInsQues[queIdx]->GetId()),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsAlgTemplateBase::PrepBitMask(const u32 queNumPerNeighbor)
{
    for (auto rankId : tempVTopo_[0]) {
        u32 algRank;
        CHK_RET(GetAlgRank(rankId, tempVTopo_[0], algRank));
        std::vector<u32> bitPosRank(queNumPerNeighbor);
        for (u32 posIdx = 0; posIdx < queNumPerNeighbor; posIdx++) {
            bitPosRank[posIdx] = algRank * queNumPerNeighbor + posIdx;
        }
        std::pair<RankId, std::vector<u32>> newPair(rankId, bitPosRank);
        rank2BitPos_.insert(newPair);
    }
    return HcclResult::HCCL_SUCCESS;
}

std::vector<std::tuple<QId, QId, u32>>
InsAlgTemplateBase::CreateMasterSlaveQueNotifiesRequest(u32 queueNum, u32 pairNum, QId masterId) const
{
    std::vector<std::tuple<QId, QId, u32>> notifyRequests;
    HCCL_DEBUG(
        "[Create][MasterSlaveQueNotifiesRequest] queueNum[%u], pairNum[%u], masterId[%u]", queueNum, pairNum, masterId);
    if (queueNum == 0 || pairNum == 0) {
        HCCL_INFO("[Create][MasterSlaveQueNotifiesRequest] queueNum or pairNum is zero, "
                  "return empty notifyRequests");
        return notifyRequests;
    };

    u32 slaveNum = queueNum - 1;
    HCCL_INFO("[Create][MasterSlaveQueNotifiesRequest] slaveNum[%u]", slaveNum);
    if (slaveNum < 1 || pairNum < 1) {
        return notifyRequests;
    }
    notifyRequests.reserve(slaveNum * pairNum);
    for (QId q = 0; q < queueNum; q++) {
        if (q == masterId) {
            continue;
        }
        for (u32 i = 0; i < pairNum; i++) {
            notifyRequests.emplace_back(std::make_tuple(masterId, q, i));
            notifyRequests.emplace_back(std::make_tuple(q, masterId, i));
        }
    }
    return notifyRequests;
}

std::vector<std::tuple<QId, QId, u32>>
InsAlgTemplateBase::CreateNotifiesRequestByMap(std::map<std::tuple<QId, QId>, u32>& notifyRequestMap) const
{
    std::vector<std::tuple<QId, QId, u32>> notifuRequests;

    for (auto iter = notifyRequestMap.begin(); iter != notifyRequestMap.end(); iter++) {
        u32 notifyNum = iter->second;
        for (u32 i = 0; i < notifyNum; i++) {
            notifuRequests.emplace_back(std::make_tuple(std::get<0>(iter->first), std::get<1>(iter->first), i));
        }
    }
    return notifuRequests;
}

std::vector<std::tuple<QId, QId, u32>> InsAlgTemplateBase::MergeNotifiesRequest(
    const std::vector<std::vector<std::tuple<QId, QId, u32>>>& notifiesRequests) const
{
    std::vector<std::tuple<QId, QId, u32>> ret;
    std::map<std::tuple<QId, QId>, u32> requestMap;
    for (auto& notifiesRequest : notifiesRequests) {
        for (auto& request : notifiesRequest) {
            QId fromQ = std::get<0>(request);
            QId toQ = std::get<1>(request);
            requestMap[std::make_tuple(fromQ, toQ)]++;
        }
    }
    return CreateNotifiesRequestByMap(requestMap);
}

void InsAlgTemplateBase::SetLoadInfo(const CollAlgParams& params) const
{
    (void)params;
    return;
}

HcclResult InsAlgTemplateBase::GetMaxTransPortDataSize(u64& maxTransPortDataSize) const
{
    maxTransPortDataSize = UB_MAX_DATA_SIZE; // 256M
    return HCCL_SUCCESS;
}

HcclResult InsAlgTemplateBase::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)numBlocks;
    (void)dataSize;
    (void)numBlocksLimit;
    HCCL_WARNING("CalNumBlocks not support ins template.");
    return HCCL_SUCCESS;
}

bool InsAlgTemplateBase::IsPcieLink(const ResLinks& tempLinks) const
{
    for (auto it = tempLinks.begin(); it != tempLinks.end(); it++) {
        const std::vector<LinkData>& linkVector = it->second;

        for (auto vecIt = linkVector.begin(); vecIt != linkVector.end(); vecIt++) {
            if (vecIt->GetType() == PortDeploymentType::P2P && vecIt->GetLinkProtocol() == LinkProtocol::PCIE) {
                HCCL_INFO("IsPcieLink[true]");
                return true;
            }
        }
    }
    HCCL_INFO("IsPcieLink[false]");
    return false;
}
HcclResult InsAlgTemplateBase::setPathNumMap(const std::map<u32, u32>& rank2PathNumMap)
{
    rank2PathNumMap_ = rank2PathNumMap;
    return HCCL_SUCCESS;
}

} // namespace Hccl
