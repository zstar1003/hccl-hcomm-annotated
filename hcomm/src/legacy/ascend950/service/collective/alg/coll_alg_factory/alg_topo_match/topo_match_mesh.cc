/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_mesh.h"

namespace Hccl {
TopoMatchMesh::TopoMatchMesh(const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType)
    : TopoMatchBase(vRank, rankSize, rankGraph, devType)
{}

TopoMatchMesh::~TopoMatchMesh() {}

HcclResult TopoMatchMesh::MatchTopo(
    std::vector<std::vector<RankId>>& vTopo, std::vector<RankId>& virtRanks, std::map<RankId, u32>& virtRankMap)
{
    // 获取并校验通信层数
    std::set<u32> levelSet = rankGraph_->GetLevels(myRank_);
    CHK_PRT_RET(
        (levelSet.size() == COMM_LEVEL_SIZE_0),
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMesh] Rank [%d], Invalid virtual topo.", myRank_),
        HcclResult::HCCL_E_PARA);
    CHK_RET(MeshTopoForAllLevel());
    virtRanks = rankIds_;
    vTopo.push_back(rankIds_);

    CHK_PRT_RET(
        GenVirtRankMapping(virtRanks, virtRankMap) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMesh] Rank [%d], Fail to generate virtRankMapping.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchMesh::MeshTopoForAllLevel()
{
    const NetInstance* netInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (netInstance == nullptr) {
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMesh] Rank [%d], netInstance is nullptr.", myRank_);
        return HcclResult::HCCL_E_PTR;
    }
    std::set<RankId> rankSet = netInstance->GetRankIds();
    for (u32 rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        RankId rankId = static_cast<RankId>(rankIdx);
        auto rankInRankSet = std::find(rankSet.begin(), rankSet.end(), rankId);
        if (rankInRankSet != rankSet.end()) {
            u32 srcLocalId = rankGraph_->GetReplacedLocalId(myRank_);
            u32 dstLocalId = rankGraph_->GetReplacedLocalId(rankId);
            if ((srcLocalId / RANK_SIZE_EIGHT == dstLocalId / RANK_SIZE_EIGHT)
                || (srcLocalId % RANK_SIZE_EIGHT == dstLocalId % RANK_SIZE_EIGHT)) {
                rankIds_.push_back(rankId);
                continue;
            }
        }
        if (GetPathNum(myRank_, rankId) == 0) {
            HCCL_ERROR("[CollAlgFactory] [TopoMatchMesh] Rank [%d], Invalid virtual topo for full mesh.", myRank_);
            return HcclResult::HCCL_E_PARA;
        }
        rankIds_.push_back(rankId);
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
