/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_mesh_nhr.h"

namespace Hccl {
TopoMatchMeshNHR::TopoMatchMeshNHR(
    const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType)
    : TopoMatchBase(vRank, rankSize, rankGraph, devType)
{}

TopoMatchMeshNHR::~TopoMatchMeshNHR() {}

// 在 ranksOnDim[0] 或 ranksOnDim[1] 中找到 myRank 所属的 level 0 子通信域
HcclResult TopoMatchMeshNHR::GenerateLevel0(
    const std::set<RankId>& rankSet, u32 levelSize, RankId rankId, std::vector<std::vector<std::vector<RankId>>>& vTopo,
    std::vector<std::vector<RankId>>& virtRanks)
{
    // 计算pod size (m x n)以及level0形状
    u32 dim0Size = numRanksPerBoard_.at(0);
    u32 level0Dim0 = GcdTwo(levelSize, dim0Size);
    u32 level0Dim1 = levelSize / level0Dim0;
    CHK_PRT_RET(
        (level0Dim0 != 1 && level0Dim1 != 1), // 非1d level0
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMeshNHR] Rank [%d],Not 1D topo. Invalid level0 virtual topo.", myRank_),
        HcclResult::HCCL_E_PARA);
    // 查找 rankId 在原始向量中的索引
    auto it = rankSet.find(rankId);
    CHK_PRT_RET(
        (it == rankSet.end()),
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMeshNHR] Rank [%d], Invalid virtual topo.", myRank_),
        HcclResult::HCCL_E_PARA);
    u32 rankIndex = std::distance(rankSet.begin(), it);

    // 计算 rankId 所在的段编号
    // 因为每段长度为 levelSize，所以 (目标索引) / (段长) 就是段编号
    u32 segmentIndex = rankIndex / levelSize;
    // 计算该段的起始索引 (start_index)
    u32 startIndex = segmentIndex * levelSize;

    // 提取子段数据
    // 提取范围是 [start_index, start_index + levelSize)
    std::vector<RankId> level0Vec = {};
    auto startIter = rankSet.begin();
    std::advance(startIter, startIndex);

    u32 endIndex = startIndex + levelSize;
    auto endIter = rankSet.begin();
    std::advance(endIter, std::min(endIndex, static_cast<u32>(rankSet.size())));

    level0Vec.assign(startIter, endIter);
    HCCL_DEBUG(
        "[CollAlgFactory] [TopoMatchMeshNHR] Rank [%d], virtual topo level0 rankSetLevel0[%s]", myRank_,
        PrintVector<RankId>(level0Vec).c_str());
    virtRanks.push_back(level0Vec); // 所有level0子通信域的集合
    vTopo.push_back({level0Vec});
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchMeshNHR::MatchTopo(
    std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks,
    std::vector<std::map<RankId, u32>>& virtRankMap)
{
    // 获取并校验当前通信层数
    std::set<u32> levelSet = rankGraph_->GetLevels(myRank_);
    HCCL_DEBUG(
        "[CollAlgFactory] [TopoMatchMeshNHR] Rank [%d], virtual topo levelSet[%u][%s]", myRank_, levelSet.size(),
        PrintSet<u32>(levelSet).c_str());
    // 获取每个pod上rank数量以及pod数量
    u32 podNum = 0;
    vector<u32> instanceSizeVec = {};
    rankGraph_->GetNetInstanceList(0, instanceSizeVec, podNum);
    u32 rankSizeLevel0 = GcdMultiple(instanceSizeVec); // 作为不规则topo level0的大小
    HCCL_INFO(
        "[CollAlgFactory] [TopoMatchMeshNHR] Rank [%d], [%u] pods ,ranksize on each pod :[%s]", myRank_,
        instanceSizeVec.size(), PrintVector<u32>(instanceSizeVec).c_str());
    // 得到myrank所在pod的信息
    const NetInstance* netInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (netInstance == nullptr) {
        HCCL_ERROR("TopoMatchMeshNHR::MatchTopo netInstance is nullptr");
        return HcclResult::HCCL_E_PTR;
    }
    std::set<RankId> rankSetR0 = netInstance->GetRankIds(); // 得到此pod上所有rank
    rankOnSameBoardVector_.resize(RANK_SIZE_EIGHT, {});
    rankOnSameSlotVector_.resize(RANK_SIZE_EIGHT, {});
    CHK_RET(CalcRankOnSamePlaneOfR0(rankOnSameBoardVector_, rankOnSameSlotVector_, numRanksPerBoard_));

    // 计算R0的virtRanks 检查level0 mesh连通性 只有board + 每板board多个 rank的情况才需要检查
    if (numRanksPerBoard_.size() != 1 && numRanksPerBoard_[0] != 1) {
        if (!IsAllRanksFullMeshConnected(rankSetR0)) {
            HCCL_ERROR(
                "[CollAlgFactory] [TopoMatchMeshNHR] Rank [%d], Invalid virtual topo for "
                "mesh_ring in level0.",
                myRank_);
            return HcclResult::HCCL_E_PARA;
        }
    }
    // 计算level0 所有rank
    CHK_RET(GenerateLevel0(rankSetR0, rankSizeLevel0, myRank_, vTopo, virtRanks));

    // 计算R1的virtRanks
    const NetInstance* fabGroupLevel1 = rankGraph_->GetNetInstanceByRankId(1, myRank_);
    if (fabGroupLevel1 == nullptr) {
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMeshNHR] Rank [%d],fabGroupLevel1 is nullptr", myRank_);
        return HcclResult::HCCL_E_PTR;
    }
    std::set<RankId> rankSetLevel1 = fabGroupLevel1->GetRankIds(); // 所有pod的所有rank 顺序
    HCCL_DEBUG(
        "[CollAlgFactory] [TopoMatchMeshNHR] Rank [%d], all ranks [%s]", myRank_,
        PrintSet<RankId>(rankSetLevel1).c_str());
    std::vector<RankId> rankOnSamePlaneVector;
    CHK_RET(GenerateLevel1(rankSetLevel1, rankSizeLevel0, myRank_, vTopo, virtRanks));
    CHK_PRT_RET(
        GenVirtRankMappingMultiLevel(virtRanks, virtRankMap) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMeshNHR] Rank [%d], Fail to generate virtRankMapping.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
