/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "diff_rank_updater.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <linux/limits.h>
#include "changed_rank_info.h"
#include "rank_table_info.h"
#include "json_parser.h"
#include "hccl_log_keywords.h"

using namespace Hccl;

// 64+1替换前校验
static HcclResult Check64Plus1Replace(const NewRankInfo& changeRank, const NewRankInfo& snapshotRank)
{
    if (changeRank.replacedLocalId == snapshotRank.localId && changeRank.rankId == snapshotRank.rankId) {
        if ((changeRank.localId == BACKUP_LOCAL_ID && snapshotRank.localId != BACKUP_LOCAL_ID)
            || (changeRank.localId != BACKUP_LOCAL_ID && snapshotRank.localId == BACKUP_LOCAL_ID)) {
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("[%s] 64+1 replacement check failed.", __func__);
    return HCCL_E_PARA;
}

// 整框替换前校验,传入全量的Rank数组changeRanks和snapshotRanks和下标数组changeRankIndex和snapshotRankIndex
// 通过下标数组里的下标，可以找到对应R0Id的newRankInfo
static HcclResult CheckPodReplace(
    vector<NewRankInfo>& changeRanks, const vector<u32>& changeRankIndex, vector<NewRankInfo>& snapshotRanks,
    const vector<u32>& snapshotRankIndex)
{
    // 将changeRankId存入map，key为rankId，value为下标
    unordered_map<u32, u32> changeRankMap;
    for (u32 i = 0; i < changeRankIndex.size(); ++i) {
        u32 changeRankId = changeRanks[changeRankIndex[i]].rankId;
        changeRankMap[changeRankId] = changeRankIndex[i];
    }

    // 根据替换前后的rankId相同，校验rankId是否全部存在
    // 再将替换前后的下标对存入rankPair，校验是否满足整框替换的条件
    vector<pair<u32, u32>> rankPair;
    for (u32 i = 0; i < snapshotRankIndex.size(); ++i) {
        u32 snapshotRankId = snapshotRanks[snapshotRankIndex[i]].rankId;
        if (changeRankMap.find(snapshotRankId) == changeRankMap.end()) {
            HCCL_ERROR(
                "[%s] full frame replacement check failed, rankId[%d] not found.", __func__,
                snapshotRanks[snapshotRankIndex[i]].rankId);
            return HCCL_E_PARA;
        } else {
            rankPair.push_back({snapshotRankIndex[i], changeRankMap[snapshotRankId]});
        }
    }

    // 每组rank替换，需要校验以下四种情况
    // 1、正常场景 整框替换为 备份场景
    // 2、备份场景 整框替换为 正常场景
    // 3、正常场景 整框替换为 正常场景
    // 4、备份场景 整框替换为 备份场景
    for (u32 i = 0; i < rankPair.size(); ++i) {
        NewRankInfo& snapshotRank = snapshotRanks[rankPair[i].first];
        NewRankInfo& changeRank = changeRanks[rankPair[i].second];
        bool isChangeReplace = false;
        bool isSnapshotReplace = false;

        if (snapshotRank.localId == changeRank.localId) {
            continue;
        } else if (
            changeRank.localId == BACKUP_LOCAL_ID && changeRank.replacedLocalId == snapshotRank.localId
            && !isChangeReplace) {
            isChangeReplace = true;
            continue;
        } else if (
            snapshotRank.localId == BACKUP_LOCAL_ID && snapshotRank.replacedLocalId == changeRank.localId
            && !isSnapshotReplace) {
            isSnapshotReplace = true;
            continue;
        } else {
            HCCL_ERROR(
                "[%s] full frame replacement check failed, localId[%d] not match, rankId[%d].", __func__,
                snapshotRank.localId, snapshotRank.rankId);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

static HcclResult ParseChangeInfo(const char* changeInfo, ChangedRankInfo& changeTable)
{
    HCCL_INFO("[%s] start.", __func__);
    // 获取真实路径
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(changeInfo, resolvedPath) == nullptr) {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] RanktableRealPath: %s is not a valid real path", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA), changeInfo);
        return HCCL_E_PARA;
    }

    // 读取文件、校验文件是否存在
    HCCL_INFO("[%s] waiting for json file load complete", __func__);
    ifstream infoFile(resolvedPath, ifstream::in);
    if (!infoFile) {
        HCCL_ERROR("open file %s failed.", resolvedPath);
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    // 获取ranktableM
    stringstream rankTableStr;
    rankTableStr << infoFile.rdbuf();
    string ranktableM = rankTableStr.str();

    // 解析ranktable
    JsonParser rankTableParser;
    rankTableParser.ParseString(ranktableM, changeTable);

    HCCL_INFO("[%s] end.", __func__);
    return HCCL_SUCCESS;
}
static HcclResult GetRankMapAndChangeMap(
    unordered_map<string, vector<u32>>& rankTableMap, unordered_map<string, vector<u32>>& changedRankMap,
    RankTableInfo& rankTableInfo, ChangedRankInfo& changeTable)
{
    HCCL_INFO("[%s] start.", __func__);
    // 将changeTable和rankTable按R0 id分组
    for (u32 i = 0; i < changeTable.ranks.size(); ++i) {
        const auto& rank = changeTable.ranks[i];
        // 遍历rankLevelInfos中的每个RankLevelInfo
        for (const auto& levelInfo : rank.rankLevelInfos) {
            if (levelInfo.netLayer == 0) {
                changedRankMap[levelInfo.netInstId].push_back(i);
                break;
            }
        }
    }

    for (u32 i = 0; i < rankTableInfo.ranks.size(); ++i) {
        const auto& rank = rankTableInfo.ranks[i];
        // 遍历rankLevelInfos中的每个RankLevelInfo
        for (const auto& levelInfo : rank.rankLevelInfos) {
            if (levelInfo.netLayer == 0) {
                rankTableMap[levelInfo.netInstId].push_back(i);
                break;
            }
        }
    }

    HCCL_INFO("[%s] end.", __func__);
    return HCCL_SUCCESS;
}
HcclResult Hccl::DiffRankUpdater(const char* changeInfo, RankTableInfo& rankTableInfo)
{
    CHK_PTR_NULL(changeInfo);
    HCCL_INFO("[%s] Start to update rankTableInfo by changeInfo changeInfo[%s]", __func__, changeInfo);
    rankTableInfo.Dump();

    ChangedRankInfo changeTable;
    CHK_RET(ParseChangeInfo(changeInfo, changeTable));

    // 将changeTable和rankTable按R0 id分组
    unordered_map<string, vector<u32>> changedRankMap;
    unordered_map<string, vector<u32>> rankTableMap;
    CHK_RET(GetRankMapAndChangeMap(rankTableMap, changedRankMap, rankTableInfo, changeTable));

    vector<pair<u32, u32>> needChangeRank;
    for (auto& rankInfo : rankTableMap) {
        // 获取R0 id
        std::string levelZeroId = rankInfo.first;

        // 获取对应的changedRankInfo的Rank数量
        u32 changeCount = changedRankMap[levelZeroId].size();
        u32 snapshotCount = rankTableMap[levelZeroId].size();

        // 确定替换策略
        if (changeCount == 0) {
            HCCL_INFO("[%s] Level[%s] No changes to apply.", __func__, levelZeroId.c_str());
            continue;
        } else if (changeCount == snapshotCount) {
            // 整框替换
            HCCL_INFO("[%s] Level[%s] Performing full frame replacement.", __func__, levelZeroId.c_str());
            CHK_RET(CheckPodReplace(
                changeTable.ranks, changedRankMap[levelZeroId], rankTableInfo.ranks, rankTableMap[levelZeroId]));

            for (u32 i = 0; i < snapshotCount; ++i) {
                needChangeRank.push_back({changedRankMap[levelZeroId][i], rankTableMap[levelZeroId][i]});
            }
        } else if (changeCount == 1) {
            // 64+1替换
            HCCL_INFO("[%s] Level[%s] Performing 64+1 replacement.", __func__, levelZeroId.c_str());
            auto changeRankId = changeTable.ranks[changedRankMap[levelZeroId][0]].rankId;
            for (u32 i = 0; i < snapshotCount; ++i) {
                if (changeRankId == rankTableInfo.ranks[rankTableMap[levelZeroId][i]].rankId) {
                    CHK_RET(Check64Plus1Replace(
                        changeTable.ranks[changedRankMap[levelZeroId][0]],
                        rankTableInfo.ranks[rankTableMap[levelZeroId][i]]));

                    needChangeRank.push_back({changedRankMap[levelZeroId][0], rankTableMap[levelZeroId][i]});
                    break;
                }
            }
        } else {
            HCCL_ERROR(
                "[%s] Level[%s] Invalid number of changed ranks. Must be 0, 1, or equal to the snapshot count (%u).",
                __func__, levelZeroId.c_str(), snapshotCount);
            return HCCL_E_PARA;
        }
    }

    // 更新rankTableInfo
    for (auto& rankInfo : needChangeRank) {
        auto& rank = rankTableInfo.ranks[rankInfo.second];
        rank = changeTable.ranks[rankInfo.first];
    }

    rankTableInfo.Check();

    HCCL_INFO("Update rankTableInfo by changeInfo success");
    rankTableInfo.Dump();
    return HCCL_SUCCESS;
}
