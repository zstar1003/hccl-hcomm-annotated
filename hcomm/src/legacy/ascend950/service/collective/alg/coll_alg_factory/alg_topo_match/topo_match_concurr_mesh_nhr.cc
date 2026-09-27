/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_concurr_mesh_nhr.h"

namespace Hccl {
TopoMatchConcurrMeshNHR::TopoMatchConcurrMeshNHR(
    const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType)
    : TopoMatchBase(vRank, rankSize, rankGraph, devType)
{}

TopoMatchConcurrMeshNHR::~TopoMatchConcurrMeshNHR() {}

HcclResult TopoMatchConcurrMeshNHR::MatchTopo(
    std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks,
    std::vector<std::map<RankId, u32>>& virtRankMap)
{
    // 获取并校验当前通信层数
    std::set<u32> levelSet = rankGraph_->GetLevels(myRank_);
    CHK_PRT_RET(
        (levelSet.size() < COMM_LEVEL_SIZE_2),
        HCCL_ERROR(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], Invalid virtual topo. levelSet[%u]", myRank_,
            levelSet.size()),
        HcclResult::HCCL_E_PARA);

    rankGraph_->Dump();

    // 获取 level0 Pod Topo 信息
    const NetInstance* netInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (netInstance == nullptr) {
        HCCL_ERROR("TopoMatchConcurrMeshNHR::MatchTopo netInstance is nullptr");
        return HcclResult::HCCL_E_PTR;
    }
    const u32 rankSizeLevel0 = netInstance->GetRankSize();
    std::set<RankId> rankSetLevel0 = netInstance->GetRankIds();

    // 校验 level0 Pod Topo 是否符合 m x n 要求
    CHK_RET(CalcRankOnSamePlaneOfR0(rankOnSameBoardVector_, rankOnSameSlotVector_, numRanksPerBoard_));
    const auto minmaxPair = std::minmax_element(numRanksPerBoard_.begin(), numRanksPerBoard_.end());
    u32 minNumRankPerBoard = *minmaxPair.first;
    u32 maxNumRankPerBoard = *minmaxPair.second;
    CHK_PRT_RET(
        (minNumRankPerBoard != maxNumRankPerBoard),
        HCCL_ERROR(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], Invalid virtual topo for "
            "multi-dimensional concurrent mesh, min numRanksPerBoard_[%u], max numRanksPerBoard_[%u].",
            myRank_, minNumRankPerBoard, maxNumRankPerBoard),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        ((rankSize_ == 1) || (numRanksPerBoard_[0] * numRanksPerBoard_.size() != rankSizeLevel0)),
        HCCL_ERROR(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], Invalid virtual topo for "
            "multi-dimensional concurrent mesh algorithm with rankSize [%u], rankSizeLevel0 [%u], "
            "ranksPerBoard [%u], ranksPerSlot [%u].",
            myRank_, rankSize_, rankSizeLevel0, numRanksPerBoard_[0], numRanksPerBoard_.size()),
        HcclResult::HCCL_E_PARA);

    // 在全局视角下计算 level0 sub-communicator size
    std::vector<u32> instSizeList;                             // rank num on each pod
    u32 listSize;                                              // pod num
    rankGraph_->GetNetInstanceList(0, instSizeList, listSize); // global view
    HCCL_DEBUG(
        "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], instSizeList[%u]=[%s]", myRank_, instSizeList.size(),
        PrintVector<u32>(instSizeList).c_str());

    if (!CheckSymmetric(instSizeList)) {
        // 非对称情形
        HCCL_DEBUG("[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], Asymmetric MatchTopo.", myRank_);
        u32 gcdInstSize = GcdMultiple(instSizeList);

        // 计算 level 0 子通信域
        CHK_RET(GenerateLevel0(rankSetLevel0, gcdInstSize, myRank_, vTopo, virtRanks));

        // 计算 level 1 子通信域
        const NetInstance* netInstanceL1 = rankGraph_->GetNetInstanceByRankId(1, myRank_);
        if (netInstanceL1 == nullptr) {
            HCCL_ERROR("TopoMatchConcurrMeshNHR::MatchTopo netInstanceL1 is nullptr");
            return HcclResult::HCCL_E_PTR;
        }
        std::set<RankId> rankSetLevel1 = netInstanceL1->GetRankIds();

        CHK_RET(GenerateLevel1(rankSetLevel1, gcdInstSize, myRank_, vTopo, virtRanks));

        HCCL_DEBUG(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], level0 & level1 virtRanks=[%s] ", myRank_,
            PrintMatrix<RankId>(virtRanks).c_str());
        HCCL_DEBUG(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], level0 & level1 vTopo=[%s] ", myRank_,
            PrintTensor<RankId>(vTopo).c_str());
    } else {
        // 对称情形
        HCCL_DEBUG("[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], Symmetric MatchTopo.", myRank_);

        u32 myLocalId = rankGraph_->GetLocalId(myRank_);
        rankOnSameBoard_ = rankOnSameBoardVector_[myLocalId / RANK_SIZE_EIGHT];
        rankOnSameSlot_ = rankOnSameSlotVector_[myLocalId % RANK_SIZE_EIGHT];
        if ((rankOnSameBoard_.size() == 1) || (rankOnSameSlot_.size() == 1)) {
            HCCL_DEBUG(
                "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], virtual topo with rankSize [%u], ranksPerBoard "
                "[%u], ranksPerSlot [%u]. 1-D Mesh algorithm should be adopted.",
                myRank_, rankSize_, rankOnSameBoard_.size(), rankOnSameSlot_.size());
        }

        sort(rankOnSameBoard_.begin(), rankOnSameBoard_.end());
        sort(rankOnSameSlot_.begin(), rankOnSameSlot_.end());

        Matrix<RankId> rankOnSamePod;
        rankOnSamePod.push_back(rankOnSameBoard_);
        rankOnSamePod.push_back(rankOnSameSlot_);
        vTopo.push_back(rankOnSamePod);
        virtRanks.push_back({rankSetLevel0.cbegin(), rankSetLevel0.cend()});

        // 获取 level1 Topo 信息
        const NetInstance* netInstanceL1 = rankGraph_->GetNetInstanceByRankId(1, myRank_);
        if (netInstanceL1 == nullptr) {
            HCCL_ERROR("TopoMatchConcurrMeshNHR::MatchTopo netInstanceL1 is nullptr");
            return HcclResult::HCCL_E_PTR;
        }
        std::set<RankId> rankSetLevel1 = netInstanceL1->GetRankIds();

        // 它要求 rankId 是连续的, 每隔 rankSizeLevel0 取一个
        CHK_RET(GenerateLevel1(rankSetLevel1, rankSizeLevel0, myRank_, vTopo, virtRanks));

        HCCL_DEBUG(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], level0 & level1 virtRanks=[%s] ", myRank_,
            PrintMatrix<RankId>(virtRanks).c_str());
        HCCL_DEBUG(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], level0 & level1 vTopo=[%s] ", myRank_,
            PrintTensor<RankId>(vTopo).c_str());
    }

    // 子通信域计算完毕, 生成 virtRankMap
    CHK_PRT_RET(
        GenVirtRankMappingMultiLevel(virtRanks, virtRankMap) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], Fail to generate virtRankMapping.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchConcurrMeshNHR::GenerateLevel0(
    const std::set<RankId>& rankSetLevel0, u32 gcdInstSize, RankId rankId,
    std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks)
{
    // 获取 my pod size, 计算 level0 subcommunicator size
    (void)rankId;
    u32 dim0Size = numRanksPerBoard_.at(0);
    u32 gcdDim0Size = GcdTwo(gcdInstSize, dim0Size);
    u32 gcdDim1Size = gcdInstSize / gcdDim0Size;

    // 维数校验 (2D)
    CHK_PRT_RET(
        (gcdDim0Size == 1) || (gcdDim1Size == 1), // 1D case
        HCCL_ERROR(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], Pod Topo Shape != 2D."
            "gcdDim0Size [%u], gcdDim1Size [%u]",
            myRank_, gcdDim0Size, gcdDim1Size),
        HcclResult::HCCL_E_PARA);

    // 判断 level 0 topo 是否满足 rankId 连续限制
    bool isRankIdContinue = true;
    if (gcdDim0Size < dim0Size && gcdDim1Size > 1) {
        isRankIdContinue = false;
    }

    if (!isRankIdContinue) {
        HCCL_ERROR("RankId [%d]. Checker does not support nhr + nhr algorithm validation.", myRank_);
        RankId minRankId = *rankSetLevel0.cbegin();
        RankId relRankId = myRank_ - minRankId;
        u32 groupId = relRankId / gcdInstSize;

        const u32 startOffset = groupId * gcdInstSize;
        const u32 totalSize = rankSetLevel0.size();
        auto startIt = rankSetLevel0.cbegin();
        std::advance(startIt, startOffset);

        const u32 elementsToTake = std::min(gcdInstSize, totalSize - startOffset);
        auto endIt = startIt;
        std::advance(endIt, elementsToTake);

        std::vector<RankId> level0Ranks = std::vector<RankId>(startIt, endIt);
        vTopo.push_back({level0Ranks});
        virtRanks.push_back(level0Ranks);
        return HcclResult::HCCL_SUCCESS;
    }

    std::vector<RankId> rankVecLevel0 = std::vector<RankId>(rankSetLevel0.cbegin(), rankSetLevel0.cend());
    Matrix<RankId> rankMatLevel0 = {};
    auto startIt = rankVecLevel0.cbegin();
    auto endIt = rankVecLevel0.cend();
    while (startIt != endIt) {
        auto currentEndIt = startIt;
        std::advance(currentEndIt, std::min(dim0Size, static_cast<u32>(std::distance(startIt, endIt))));
        rankMatLevel0.emplace_back(startIt, currentEndIt);
        startIt = currentEndIt;
    }

    std::vector<RankId> level0Ranks;
    Matrix<RankId> level0Topo;
    CHK_RET(FindLevel0Block(rankMatLevel0, gcdDim1Size, gcdDim0Size, myRank_, level0Ranks, level0Topo));

    CHK_PRT_RET(
        (level0Topo[0].size() != gcdDim0Size || level0Topo[1].size() != gcdDim1Size),
        HCCL_ERROR(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], Invalid level0Topo size[%zu][%zu]", myRank_,
            level0Topo[0].size(), level0Topo[1].size()),
        HcclResult::HCCL_E_PARA);
    vTopo.push_back(level0Topo);
    virtRanks.push_back(level0Ranks);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchConcurrMeshNHR::FindLevel0Block(
    const Matrix<RankId>& podTopo, u32 subDim0Size, u32 subDim1Size, u32 myRank, std::vector<RankId>& subRankList,
    Matrix<RankId>& subRankTopo) const
{
    CHK_PRT_RET(
        (podTopo.empty() || podTopo[0].empty() || subDim0Size == 0 || subDim1Size == 0),
        HCCL_ERROR("[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], empty podTopo or invalid subDim.", myRank_),
        HcclResult::HCCL_E_PARA);
    const u32 totalRows = podTopo.size();
    const u32 totalCols = podTopo[0].size();
    CHK_PRT_RET(
        (totalRows % subDim0Size != 0 || totalCols % subDim1Size != 0),
        HCCL_ERROR("[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], invalid subDim.", myRank_),
        HcclResult::HCCL_E_PARA);

    u32 targetRow = -1;
    u32 targetCol = -1;
    CHK_RET(FindMyRankLocation(podTopo, myRank, targetRow, targetCol)); // 查找 myRank 在 podTopo 中坐标

    // 计算 podTopo 块信息
    const u32 blockRow = targetRow / subDim0Size;
    const u32 blockCol = targetCol / subDim1Size;

    // 计算目标子块编号和起始索引
    const u32 startRow = blockRow * subDim0Size;
    const u32 startCol = blockCol * subDim1Size;

    CHK_RET(ExtractLevel0Block(
        podTopo, startRow, startCol, targetRow, targetCol, subDim0Size, subDim1Size, subRankList, subRankTopo));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
TopoMatchConcurrMeshNHR::FindMyRankLocation(const Matrix<RankId>& podTopo, u32 myRank, u32& row, u32& col) const
{
    row = -1;
    col = -1; // 初始化为非法值
    for (u32 r = 0; r < podTopo.size(); r++) {
        auto it = std::find(podTopo[r].cbegin(), podTopo[r].cend(), myRank);
        if (it != podTopo[r].cend()) {
            row = r;
            col = std::distance(podTopo[r].cbegin(), it);
            return HcclResult::HCCL_SUCCESS;
        }
    }
    return HcclResult::HCCL_E_PARA;
}

HcclResult TopoMatchConcurrMeshNHR::ExtractLevel0Block(
    const Matrix<RankId>& podTopo, u32 startRow, u32 startCol, u32 targetRow, u32 targetCol, u32 subDim0Size,
    u32 subDim1Size, std::vector<RankId>& subRankList, Matrix<RankId>& subRankTopo) const
{
    subRankList.clear();
    subRankList.reserve(subDim0Size * subDim1Size);
    subRankTopo.clear();

    const u32 targetRelRow = targetRow - startRow; // myRank 在子块中的相对行 = 绝对行 - 子块起始行
    const u32 targetRelCol = targetCol - startCol; // myRank 在子块中的相对列 = 绝对列 - 子块起始列

    // 校验相对行&列在 podTopo 范围内
    const u32 totalRows = podTopo.size();
    const u32 totalCols = podTopo[0].size();
    CHK_PRT_RET(
        targetRelRow >= totalRows,
        HCCL_ERROR(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], Invalid targetRelRow[%u], "
            "totalRows[%u]",
            myRank_, targetRelRow, totalRows),
        HcclResult::HCCL_E_PARA);
    CHK_PRT_RET(
        targetRelCol >= totalCols,
        HCCL_ERROR(
            "[CollAlgFactory] [TopoMatchConcurrMeshNHR] Rank [%d], Invalid targetRelCol[%u], "
            "totalCols[%u]",
            myRank_, targetRelCol, totalCols),
        HcclResult::HCCL_E_PARA);

    std::vector<RankId> targetRowVec;
    std::vector<RankId> targetColVec;
    targetRowVec.reserve(subDim1Size);
    targetColVec.reserve(subDim0Size);

    auto matRowStartIt = podTopo.cbegin() + startRow; // 使用迭代器算术获取起始行迭代器

    // 循环 subDim0Size 次，每次处理一行
    for (u32 r = 0; r < subDim0Size; r++) {
        const auto& sourceRow = *(matRowStartIt + r); // 获取当前行的常量引用
        // 计算该行的列坐标起始
        auto startIt = sourceRow.cbegin() + startCol;
        auto endIt = sourceRow.cbegin() + startCol + subDim1Size;

        subRankList.insert(subRankList.end(), startIt, endIt);

        if (r == targetRelRow) {
            targetRowVec.insert(targetRowVec.end(), startIt, endIt);
        }

        auto colIt = sourceRow.cbegin() + startCol + targetRelCol;
        targetColVec.push_back(*colIt);
    }

    std::sort(targetRowVec.begin(), targetRowVec.end());
    std::sort(targetColVec.begin(), targetColVec.end());

    subRankTopo.emplace_back(std::move(targetRowVec));
    subRankTopo.emplace_back(std::move(targetColVec));

    return HcclResult::HCCL_SUCCESS;
}

bool TopoMatchConcurrMeshNHR::CheckSymmetric(std::vector<u32>& values) const
{
    const auto minmaxPair = std::minmax_element(values.begin(), values.end());
    u32 minValue = *minmaxPair.first;
    u32 maxValue = *minmaxPair.second;

    return (minValue == maxValue);
}
} // namespace Hccl
