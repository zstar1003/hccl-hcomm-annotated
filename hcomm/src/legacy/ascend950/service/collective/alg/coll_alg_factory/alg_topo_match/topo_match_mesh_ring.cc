/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_mesh_ring.h"

namespace Hccl {
TopoMatchMeshRing::TopoMatchMeshRing(
    const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType)
    : TopoMatchBase(vRank, rankSize, rankGraph, devType)
{}

TopoMatchMeshRing::~TopoMatchMeshRing() {}

HcclResult TopoMatchMeshRing::MatchTopo(
    std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks,
    std::vector<std::map<RankId, u32>>& virtRankMap)
{
    // 获取并校验当前通信层数
    std::set<u32> levelSet = rankGraph_->GetLevels(myRank_);
    CHK_PRT_RET(
        (levelSet.size() == COMM_LEVEL_SIZE_0), // 获取当前rank通信层数
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMeshRing] Rank [%d], Invalid virtual topo.", myRank_),
        HcclResult::HCCL_E_PARA);

    // 获取 level0 Topo 信息
    const NetInstance* netInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (netInstance == nullptr) {
        HCCL_ERROR("TopoMatchMeshRing::MatchTopo netInstance is nullptr");
        return HcclResult::HCCL_E_PTR;
    }
    std::set<RankId> rankSetR0 = netInstance->GetRankIds();
    CHK_RET(CalcRankOnSamePlaneOfR0(rankOnSameBoardVector_, rankOnSameSlotVector_, numRanksPerBoard_));

    if (levelSet.size() == COMM_LEVEL_SIZE_1) {
        // 判断level0上的拓扑是否符合 m x n 要求
        const auto minmxPair = std::minmax_element(numRanksPerBoard_.begin(), numRanksPerBoard_.end());
        u32 minNumRankPerBoard = *minmxPair.first;
        u32 maxNumRankPerBoard = *minmxPair.second;
        CHK_PRT_RET(
            (minNumRankPerBoard != maxNumRankPerBoard),
            HCCL_ERROR(
                "[CollAlgFactory] [TopoMatchMeshRing] Rank [%d], Invalid virtual topo for "
                "mesh_ring, min numRanksPerBoard_[%u], max numRanksPerBoard_[%u].",
                myRank_, minNumRankPerBoard, maxNumRankPerBoard),
            HcclResult::HCCL_E_PARA);

        CHK_PRT_RET(
            ((rankSize_ == 1) || (numRanksPerBoard_[0] * numRanksPerBoard_.size() != rankSize_)),
            HCCL_ERROR(
                "[CollAlgFactory] [TopoMatchMeshRing] Rank [%d], Invalid virtual topo for "
                "mesh_ring algorithm with rankSize [%u], ranksPerBoard [%u], ranksPerSlot [%lu].",
                myRank_, rankSize_, numRanksPerBoard_[0], numRanksPerBoard_.size()),
            HcclResult::HCCL_E_PARA);

        // 计算R0的AlgTopoInfo
        u32 myLocalId = rankGraph_->GetReplacedLocalId(myRank_);
        virtRanks.push_back(rankOnSameBoardVector_[myLocalId / RANK_SIZE_EIGHT]);
        virtRanks.push_back(rankOnSameSlotVector_[myLocalId % RANK_SIZE_EIGHT]);
        for (auto vRankIter = virtRanks.begin(); vRankIter != virtRanks.end(); vRankIter++) {
            vTopo.push_back({*vRankIter});
        }
    } else {
        CHK_RET(MeshRingTopoForAllLevel(rankSetR0, vTopo, virtRanks));
    }
    // generate rank mapping
    CHK_PRT_RET(
        GenVirtRankMappingMultiLevel(virtRanks, virtRankMap) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMeshRing] Rank [%d], Fail to generate virtRankMapping.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchMeshRing::MeshRingTopoForAllLevel(
    std::set<RankId> rankSetR0, std::vector<std::vector<std::vector<RankId>>>& vTopo,
    std::vector<std::vector<RankId>>& virtRanks)
{
    // 计算R0的virtRanks
    if (numRanksPerBoard_.size() != 1 && numRanksPerBoard_[0] != 1) {
        if (!IsAllRanksFullMeshConnected(rankSetR0)) {
            HCCL_ERROR(
                "[CollAlgFactory] [TopoMatchMeshRing] Rank [%d], Invalid virtual topo for "
                "mesh_ring in level0.",
                myRank_);
            return HcclResult::HCCL_E_PARA;
        }
        std::vector<RankId> ranksPerRack(rankSetR0.size());
        for (RankId rankId : rankSetR0) {
            ranksPerRack.push_back(rankId);
        }
        virtRanks.push_back(ranksPerRack);
    }
    // 计算R1的virtRanks
    const NetInstance* fabGroupLevel1 = rankGraph_->GetNetInstanceByRankId(1, myRank_);
    if (fabGroupLevel1 == nullptr) {
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMeshRing] Rank [%d],fabGroupLevel1 is nullptr", myRank_);
        return HcclResult::HCCL_E_PTR;
    }
    std::set<RankId> rankSetLevel1 = fabGroupLevel1->GetRankIds();
    std::vector<RankId> rankOnSamePlaneVector(rankSetLevel1.size());
    // 获取所有rank在level1的平面
    for (RankId rankId : rankSetLevel1) {
        rankOnSamePlaneVector.push_back(rankId);
    }
    virtRanks.push_back(rankOnSamePlaneVector);
    // 计算vTopo
    for (u32 i = 0; i < virtRanks.size(); i++) {
        vTopo.push_back({virtRanks[i]});
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
