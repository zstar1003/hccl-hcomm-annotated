/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_concurr_mesh.h"

namespace Hccl {
TopoMatchConcurrMesh::TopoMatchConcurrMesh(
    const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType)
    : TopoMatchBase(vRank, rankSize, rankGraph, devType)
{}

TopoMatchConcurrMesh::~TopoMatchConcurrMesh() {}

HcclResult TopoMatchConcurrMesh::MatchTopo(
    std::vector<std::vector<RankId>>& vTopo, std::vector<RankId>& virtRanks, std::map<RankId, u32>& virtRankMap)
{
    // 获取并校验当前通信层数
    std::set<u32> levelSet = rankGraph_->GetLevels(myRank_);

    CHK_PRT_RET(
        (levelSet.size() == COMM_LEVEL_SIZE_0),
        HCCL_ERROR("[CollAlgFactory] [TopoMatchConcurrMesh] Rank [%d], Invalid virtual topo.", myRank_),
        HcclResult::HCCL_E_PARA);
    const NetInstance* netInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    std::set<RankId> rankSet = netInstance->GetRankIds();
    for (RankId rankId : rankSet) {
        rankIds_.push_back(rankId);
    }
    // 判断level0上的拓扑是否符合 m x n 要求
    CHK_RET(CalcRankOnSamePlaneOfR0(rankOnSameBoardVector_, rankOnSameSlotVector_, numRanksPerBoard_));

    const auto minmxPair = std::minmax_element(numRanksPerBoard_.begin(), numRanksPerBoard_.end());
    u32 minNumRankPerBoard = *minmxPair.first;
    u32 maxNumRankPerBoard = *minmxPair.second;
    CHK_PRT_RET(
        (minNumRankPerBoard != maxNumRankPerBoard),
        HCCL_ERROR(
            "[CollAlgFactory] [TopoMatchConcurrMesh] Rank [%d], Invalid virtual topo for "
            "multi-dimensional concurrent mesh, min numRanksPerBoard_[%u], max numRanksPerBoard_[%u].",
            myRank_, minNumRankPerBoard, maxNumRankPerBoard),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        ((rankSize_ == 1) || (numRanksPerBoard_[0] * numRanksPerBoard_.size() != rankSize_)),
        HCCL_ERROR(
            "[CollAlgFactory] [TopoMatchConcurrMesh] Rank [%d], Invalid virtual topo for "
            "multi-dimensional concurrent mesh "
            "algorithm with rankSize [%u], ranksPerBoard [%u], ranksPerSlot [%u].",
            myRank_, rankSize_, numRanksPerBoard_[0], numRanksPerBoard_.size()),
        HcclResult::HCCL_E_PARA);

    // 计算当前rankd的virtRanks, vTopo和virtRankMap
    u32 myLocalId = rankGraph_->GetReplacedLocalId(myRank_);
    rankOnSameBoard_ = rankOnSameBoardVector_[myLocalId / RANK_SIZE_EIGHT];
    rankOnSameSlot_ = rankOnSameSlotVector_[myLocalId % RANK_SIZE_EIGHT];

    if ((rankOnSameBoard_.size() == 1) || (rankOnSameSlot_.size() == 1)) {
        HCCL_DEBUG(
            "[CollAlgFactory] [TopoMatchConcurrMesh] Rank [%d],Virtual topo with rankSize [%u], ranksPerBoard "
            "[%u], ranksPerSlot [%u]. "
            "1-D Mesh algorithm is adopted.",
            myRank_, rankSize_, rankOnSameBoard_.size(), rankOnSameSlot_.size());
    }
    sort(rankOnSameBoard_.begin(), rankOnSameBoard_.end());
    sort(rankOnSameSlot_.begin(), rankOnSameSlot_.end());
    vTopo.push_back(rankOnSameBoard_);
    vTopo.push_back(rankOnSameSlot_);

    sort(rankIds_.begin(), rankIds_.end());
    virtRanks = rankIds_;

    CHK_PRT_RET(
        GenVirtRankMapping(virtRanks, virtRankMap) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [TopoMatchConcurrMesh] Rank [%d], Fail to generate virtRankMapping.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
