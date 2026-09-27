/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "connected_link_mgr.h"
#include "binary_stream.h"

namespace Hccl {
const vector<LinkData>& ConnectedLinkMgr::GetLinks(RankId dstRank)
{
    for (auto levelMap : levelRankPairLinkDataMap) {
        if (levelRankPairLinkDataMap[levelMap.first].find(dstRank) != levelRankPairLinkDataMap[levelMap.first].end()
            && levelRankPairLinkDataMap[levelMap.first][dstRank].size() > 0) {
            HCCL_INFO(
                "[ConnectedLinkMgr][GetLinks] level[%u], dstRank[%d], links.size[%u]", levelMap.first, dstRank,
                levelRankPairLinkDataMap[levelMap.first][dstRank].size());
            return levelRankPairLinkDataMap[levelMap.first][dstRank];
        }
    }
    HCCL_WARNING("[ConnectedLinkMgr][GetLinks] links is empty, dstRank[%d]", dstRank);
    return levelRankPairLinkDataMap[0][dstRank];
}

const std::vector<LinkData>& ConnectedLinkMgr::GetLinks(u32 level, RankId dstRank)
{
    if (levelRankPairLinkDataMap.find(level) == levelRankPairLinkDataMap.end()
        || levelRankPairLinkDataMap[level].find(dstRank) == levelRankPairLinkDataMap[level].end()) {
        HCCL_WARNING("[ConnectedLinkMgr][GetLinks] links is empty, dstRank[%d]", dstRank);
    }
    return levelRankPairLinkDataMap[level][dstRank];
}

void ConnectedLinkMgr::Reset() { levelRankPairLinkDataMap.clear(); }

void ConnectedLinkMgr::ParsePackedData(std::vector<char>& data)
{
    u32 levelRankPairsNum;
    u32 linkSize;
    std::vector<std::pair<u32, RankId>> leveRankPairs;
    std::vector<u32> numVec;
    BinaryStream binaryStream(data);
    binaryStream >> levelRankPairsNum;
    binaryStream >> linkSize;
    HCCL_INFO("levelRankPairsNum=%u, linkSize=%u", levelRankPairsNum, linkSize);
    for (u32 idx = 0; idx < levelRankPairsNum; idx++) {
        u32 level;
        RankId rank;
        u32 num;
        binaryStream >> level;
        binaryStream >> rank;
        binaryStream >> num;
        leveRankPairs.emplace_back(level, rank);
        numVec.push_back(num);
        HCCL_INFO("level=%u, RankId=%d, num=%u", level, rank, numVec[idx]);
    }

    std::vector<LinkData> allLinkVec;
    for (u32 idx = 0; idx < linkSize; idx++) {
        std::vector<char> linkUniqueId;
        binaryStream >> linkUniqueId;
        allLinkVec.emplace_back(linkUniqueId);
    }

    u32 linkIdx = 0;
    for (u32 idx = 0; idx < levelRankPairsNum; idx++) {
        u32 level = leveRankPairs[idx].first;
        RankId dRank = leveRankPairs[idx].second;
        HCCL_INFO("level=%u, RankId=%d, num=%u", level, dRank, numVec[idx]);
        std::vector<LinkData> linkVec;
        for (u32 i = 0; i < numVec[idx]; i++) {
            HCCL_INFO("ConnectedLinkMgr::ParsePackedData: %s", allLinkVec[linkIdx].Describe().c_str());
            linkVec.push_back(allLinkVec[linkIdx++]);
        }

        if (levelRankPairLinkDataMap.find(level) == levelRankPairLinkDataMap.end()
            || levelRankPairLinkDataMap[level].find(dRank) == levelRankPairLinkDataMap[level].end()) {
            levelRankPairLinkDataMap[level][dRank] = linkVec;
        }
    }
}
} // namespace Hccl
