/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_mesh_nhr_pcie.h"

namespace Hccl {
TopoMatchMeshNHRPcie::TopoMatchMeshNHRPcie(
    const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType)
    : TopoMatchBase(vRank, rankSize, rankGraph, devType)
{}

TopoMatchMeshNHRPcie::~TopoMatchMeshNHRPcie() {}

HcclResult TopoMatchMeshNHRPcie::LoadTopoInstRanks(u32 topoInstId, std::vector<RankId>& ranksOfSameLinkType) const
{
    std::vector<u32> ranksInTopo;
    u32 rankNum;
    CHK_RET(rankGraph_->GetRanksByTopoInst(0, topoInstId, ranksInTopo, rankNum));
    HCCL_DEBUG(
        "[CollAlgFactory] [TopoMatchMeshNHRPcie] Rank [%d], in topoInst[%u], ranks=[%s]", myRank_, topoInstId,
        PrintVector<u32>(ranksInTopo).c_str());
    ranksOfSameLinkType.insert(ranksOfSameLinkType.end(), ranksInTopo.begin(), ranksInTopo.end());
    return HCCL_SUCCESS;
}

HcclResult
TopoMatchMeshNHRPcie::DeduplicateLevelRanks(std::vector<RankId>& level0Ranks, std::vector<RankId>& level1Ranks) const
{
    u32 level0RankSize = level0Ranks.size();
    auto level1End = std::remove_if(level1Ranks.begin(), level1Ranks.end(), [this, level0RankSize](int val) {
        return val % level0RankSize != myRank_ % level0RankSize;
    });
    level1Ranks.erase(level1End, level1Ranks.end());
    return HCCL_SUCCESS;
}

HcclResult TopoMatchMeshNHRPcie::MatchTopo(
    std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks,
    std::vector<std::map<RankId, u32>>& virtRankMap)
{
    // 获取并校验当前通信层数
    std::set<u32> levelSet = rankGraph_->GetLevels(myRank_);
    HCCL_DEBUG(
        "[CollAlgFactory] [TopoMatchMeshNHRPcie] Rank [%d], virtual topo levelSet[%u][%s]", myRank_, levelSet.size(),
        PrintSet<u32>(levelSet).c_str());

    u32 topoInstNum = 0;
    vector<u32> topoInsts = {};
    rankGraph_->GetTopoInstsByLayer(0, topoInsts, topoInstNum);
    HCCL_INFO("[CollAlgFactory] [TopoMatchMeshNHRPcie] Rank [%d], [%u] topo instances", myRank_, topoInsts.size());

    std::vector<RankId> ranksIn1DMeshTopo = {};
    std::vector<RankId> ranksInClosTopo = {};
    for (auto& topoInstId : topoInsts) {
        TopoType topoType = TopoType::TOPO_TYPE_RESERVED;
        CHK_RET(rankGraph_->GetTopoType(0, topoInstId, topoType));

        if (topoType == TopoType::MESH_1D) {
            CHK_RET(LoadTopoInstRanks(topoInstId, ranksIn1DMeshTopo));
        } else if (topoType == TopoType::CLOS) {
            CHK_RET(LoadTopoInstRanks(topoInstId, ranksInClosTopo));
        } else {
            HCCL_ERROR(
                "[CollAlgFactory][TopoMatchMeshNHRPcie] Rank[%d], topoInstId[%u], Invalid topo type[%u]", myRank_,
                topoInstId, topoType);
            return HCCL_E_PARA;
        }
    }

    CHK_RET(DeduplicateLevelRanks(ranksIn1DMeshTopo, ranksInClosTopo));

    virtRanks.emplace_back(ranksIn1DMeshTopo);
    virtRanks.emplace_back(ranksInClosTopo);

    vTopo.push_back({ranksIn1DMeshTopo});
    vTopo.push_back({ranksInClosTopo});

    CHK_PRT_RET(
        GenVirtRankMappingMultiLevel(virtRanks, virtRankMap) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMeshNHRPcie] Rank [%d], Fail to generate virtRankMapping.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    HCCL_INFO(
        "[CollAlgFactory] [TopoMatchMeshNHRPcie] Rank[%d], ranks in 1DMesh topo=[%s], ranks in Clos topo=[%s], "
        "virtRanks size=[%u], vTopo size=[%u], virtRankMap size=[%u]",
        myRank_, PrintVector<RankId>(ranksIn1DMeshTopo).c_str(), PrintVector<RankId>(ranksInClosTopo).c_str(),
        virtRanks.size(), vTopo.size(), virtRankMap.size());

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
