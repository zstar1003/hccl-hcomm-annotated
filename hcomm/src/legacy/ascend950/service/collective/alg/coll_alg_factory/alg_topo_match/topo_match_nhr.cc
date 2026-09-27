/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_nhr.h"

namespace Hccl {
TopoMatchNHR::TopoMatchNHR(const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType)
    : TopoMatchBase(vRank, rankSize, rankGraph, devType)
{}

TopoMatchNHR::~TopoMatchNHR() {}

HcclResult TopoMatchNHR::MatchTopo(
    std::vector<std::vector<RankId>>& vTopo, std::vector<RankId>& virtRanks, std::map<RankId, u32>& virtRankMap)
{
    // 获取并校验通信层数
    std::set<u32> levelSet = rankGraph_->GetLevels(myRank_);
    CHK_PRT_RET(
        (levelSet.size() == COMM_LEVEL_SIZE_0),
        HCCL_ERROR("[CollAlgFactory] [TopoMatchNHR] Rank [%d], Invalid virtual topo.", myRank_),
        HcclResult::HCCL_E_PARA);
    CHK_RET(NHRTopoForAllRanks());
    virtRanks = rankIds_;
    vTopo.push_back(rankIds_);

    CHK_PRT_RET(
        GenVirtRankMapping(virtRanks, virtRankMap) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [TopoMatchNHR] Rank [%d], Fail to generate virtRankMapping.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchNHR::NHRTopoForAllRanks()
{
    RankId sendToRank;
    RankId recvFromRank;
    u32 nSteps = GetNHRStepNum(rankSize_);
    std::set<RankId> rankSet;
    for (u32 currentStep = 0; currentStep < nSteps; currentStep++) {
        u32 deltaRank = nSteps - 1 - currentStep;
        sendToRank = (myRank_ + (1 << deltaRank)) % rankSize_;
        recvFromRank = (myRank_ + rankSize_ - (1 << deltaRank)) % rankSize_;
        rankSet.insert(sendToRank);
        rankSet.insert(recvFromRank);
    }
    for (RankId rankId : rankSet) {
        if (GetPathNum(myRank_, rankId) == 0) {
            HCCL_ERROR("[CollAlgFactory] [TopoMatchNHR] Rank [%d], Invalid virtual topo for NHR.", myRank_);
            return HcclResult::HCCL_E_PARA;
        }
    }
    for (u32 rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        rankIds_.push_back(RankId(rankIdx));
    }
    return HcclResult::HCCL_SUCCESS;
}

// NHR的算法步数 = Ceil(log2(N))
u32 TopoMatchNHR::GetNHRStepNum(u32 rankSize) const
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[NHRBase][GetStepNumInterServer] rankSize[%u] nSteps[%u]", rankSize, nSteps);

    return nSteps;
}
} // namespace Hccl
