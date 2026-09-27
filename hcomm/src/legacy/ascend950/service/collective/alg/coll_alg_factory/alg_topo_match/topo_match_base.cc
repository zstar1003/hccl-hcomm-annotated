/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_instance.h"
#include "rank_gph.h"
#include "topo_match_base.h"

namespace Hccl {
TopoMatchBase::TopoMatchBase(const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType)
    : myRank_(vRank),
      rankSize_(rankSize),
      rankGraph_(rankGraph),
      devType_(devType)
{}

TopoMatchBase::~TopoMatchBase() {}

HcclResult TopoMatchBase::MatchTopo(
    std::vector<std::vector<RankId>>& vTopo, std::vector<RankId>& virtRanks, std::map<RankId, u32>& virtRankMap)
{
    (void)vTopo;
    (void)virtRanks;
    (void)virtRankMap;
    HCCL_ERROR("[CollAlgFactory] Rank [%d], use proper multi-level interface to match topo.", myRank_);
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult TopoMatchBase::MatchTopo(
    std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks,
    std::vector<std::map<RankId, u32>>& virtRankMap)
{
    (void)vTopo;
    (void)virtRanks;
    (void)virtRankMap;
    HCCL_ERROR("[CollAlgFactory] Rank [%d], use proper 1-level interface to match topo.", myRank_);
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult TopoMatchBase::SetTargetRanks(std::set<u32>& targetRanks)
{
    (void)targetRanks;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchBase::GenVirtRankMapping(std::vector<RankId>& virtRanks, std::map<RankId, u32>& virtRankMap) const
{
    std::sort(virtRanks.begin(), virtRanks.end());
    for (u64 idx = 0; idx < virtRanks.size(); idx++) {
        virtRankMap.insert(std::make_pair(virtRanks[idx], idx));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchBase::GenVirtRankMappingMultiLevel(
    std::vector<std::vector<RankId>>& virtRanks, std::vector<std::map<RankId, u32>>& virtRankMap) const
{
    for (auto vRankIter = virtRanks.begin(); vRankIter != virtRanks.end(); vRankIter++) {
        std::map<RankId, u32> tmpVirtRankMap;
        CHK_PRT_RET(
            GenVirtRankMapping((*vRankIter), tmpVirtRankMap) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CollAlgFactory] [TopoMatchMeshRing] Rank [%d], Fail to generate virtRankMapping.", myRank_),
            HcclResult::HCCL_E_INTERNAL);
        virtRankMap.push_back(tmpVirtRankMap);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchBase::CalcRankOnSamePlaneOfR0(
    std::vector<std::vector<RankId>>& rankOnSameBoardVector, std::vector<std::vector<RankId>>& rankOnSameSlotVector,
    std::vector<u32>& numRanksPerBoard) const
{
    rankOnSameBoardVector.resize(RANK_SIZE_EIGHT, {});
    rankOnSameSlotVector.resize(RANK_SIZE_EIGHT, {});
    const NetInstance* netInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (netInstance == nullptr) {
        HCCL_ERROR("TopoMatchBase::CalcRankOnSamePlaneOfR0 netInstance is nullptr");
        return HcclResult::HCCL_E_PTR;
    }
    std::set<RankId> rankSet = netInstance->GetRankIds();
    CHK_PRT_RET(
        (rankSet.size() == 0), HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid virtual topo.", myRank_),
        HcclResult::HCCL_E_PARA);

    for (RankId rankId : rankSet) {
        u32 localId = rankGraph_->GetReplacedLocalId(rankId);
        CHK_PRT_RET(
            localId >= RANK_SIZE_EIGHT * RANK_SIZE_EIGHT,
            HCCL_ERROR("localId[%u] is bigger than 63 (RANK_SIZE_EIGHT * RANK_SIZE_EIGHT - 1).", localId),
            HcclResult::HCCL_E_PARA);
        rankOnSameBoardVector[localId / RANK_SIZE_EIGHT].push_back(rankId);
        rankOnSameSlotVector[localId % RANK_SIZE_EIGHT].push_back(rankId);
    }
    for (u32 i = 0; i < RANK_SIZE_EIGHT; i++) {
        if (rankOnSameBoardVector[i].size() != 0) {
            numRanksPerBoard.push_back(rankOnSameBoardVector[i].size());
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

bool TopoMatchBase::IsAllRanksFullMeshConnected(std::set<RankId> rankSet) const
{
    std::set<u32> levelSet = rankGraph_->GetLevels(myRank_);
    u32 pathNum = 0;
    for (auto it1 = rankSet.begin(); it1 != rankSet.end(); it1++) {
        for (auto it2 = std::next(it1, 1); it2 != rankSet.end(); it2++) {
            pathNum = 0;
            for (u32 levelIdx : levelSet) {
                std::vector<NetInstance::Path> paths = rankGraph_->GetPaths(levelIdx, *it1, *it2);
                pathNum += paths.size();
            }
            if (pathNum == 0) {
                return false;
            }
        }
    }
    return true;
}

u32 TopoMatchBase::GetPathNum(RankId srcRankId, RankId dstRankId) const
{
    std::set<u32> levelSet = rankGraph_->GetLevels(myRank_);
    u32 pathNum = 0;
    for (u32 levelIdx : levelSet) {
        std::vector<NetInstance::Path> paths = rankGraph_->GetPaths(levelIdx, srcRankId, dstRankId);
        pathNum += paths.size();
    }
    return pathNum;
}

u32 TopoMatchBase::GcdTwo(u32 a, u32 b) const
{
    while (0 != b) {
        a %= b;
        std::swap(a, b);
    }
    return a;
}

u32 TopoMatchBase::GcdMultiple(const std::vector<u32>& numbers) const
{
    if (numbers.empty()) {
        THROW<InvalidParamsException>(StringFormat("Input vector cannot be empty."));
    }
    uint32_t result = numbers[0];
    for (const auto num : numbers) {
        result = GcdTwo(result, num);
        if (result == 1) {
            return 1;
        }
    }
    return result;
}

HcclResult TopoMatchBase::GenerateLevel1(
    const std::set<RankId>& rankSetLevel1, u32 gcdInstSize, RankId rankId,
    std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks) const
{
    CHK_PRT_RET(
        (gcdInstSize == 0), HCCL_ERROR("[CollAlgFactory] [TopoMatchBase] Rank [%d], gcdInstSize = 0", myRank_),
        HcclResult::HCCL_E_PARA);

    auto rankIter = rankSetLevel1.find(rankId);
    CHK_PRT_RET(
        (rankIter == rankSetLevel1.end()),
        HCCL_ERROR(
            "[CollAlgFactory] [TopoMatchBase] Rank [%d], "
            "failed to find this rank in rankSetLevel1[%s].",
            myRank_, PrintSet<RankId>(rankSetLevel1).c_str()),
        HcclResult::HCCL_E_PARA);
    u64 globalIdx = static_cast<u64>(std::distance(rankSetLevel1.begin(), rankIter));
    u64 relativeIdx = globalIdx % gcdInstSize;

    std::vector<RankId> rankOnSamePlaneVector;
    for (u64 step = relativeIdx; step < rankSetLevel1.size(); step += gcdInstSize) {
        auto targetIt = rankSetLevel1.begin();
        std::advance(targetIt, step);
        rankOnSamePlaneVector.push_back(*targetIt);
    }

    vTopo.push_back({rankOnSamePlaneVector});
    virtRanks.push_back(rankOnSamePlaneVector);
    return HCCL_SUCCESS;
}
} // namespace Hccl
