/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_table_info.h"

#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <string>
#include "json_parser.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "rank_table_report_macro.h"
#include "hccl_log_keywords.h"

namespace Hccl {

void RankTableInfo::Check()
{
    if (version != "2.0") {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] version[%s] is not \"2.0\"", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), version.c_str());
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({version, "version", "2.0"}));
        THROW<InvalidParamsException>(
            StringFormat("[RankTableInfo::%s] failed with version is not \"2.0\" in ranktable file.", __func__));
    }

    if (rankCount > MAX_RANKCOUNT) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(rankCount), "rankCount", "lower than " + std::to_string(MAX_RANKCOUNT)}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rankCount[%u] exceeds maximum limit[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankCount, MAX_RANKCOUNT);
        THROW<InvalidParamsException>(StringFormat(
            "[RankTableInfo::%s] failed with rankCount [%u] exceeds maximum limit of [%u]", __func__, rankCount,
            MAX_RANKCOUNT));
    }

    if (rankCount == 0) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(rankCount), "rankCount", "should not be 0"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rankCount is 0", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
        THROW<InvalidParamsException>(StringFormat(
            "[RankTableInfo::%s] failed with rankCount [%u] exceeds minimum limit of [%u]", __func__, rankCount, 0));
    }

    if (rankCount != ranks.size()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(rankCount), "rankCount",
                 "rankCount is equal to rankSize[" + std::to_string(ranks.size()) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rankCount[%u] != ranks.size[%zu]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankCount, ranks.size());
        THROW<InvalidParamsException>(StringFormat(
            "[RankTableInfo::%s] failed with rankCount is not equal "
            "to rank_list size. version[%s], rankCount[%u], ranks.size[%u]",
            __func__, version.c_str(), rankCount, ranks.size()));
    }

    std::unordered_set<u32> rankIdSet;
    std::unordered_set<u32> localIdSet;
    u32 recordedReplaceLocalId{UNDEFIEND_LOCAL_ID};
    for (auto& rank : ranks) {
        if (static_cast<u32>(rank.rankId) >= rankCount) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>(
                    {std::to_string(rank.rankId), "rankId", "[0," + std::to_string(rankCount) + ")"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] rankId[%d] out of range [0,%u)", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rank.rankId, rankCount);
            THROW<InvalidParamsException>(StringFormat(
                "[Parse][ClusterInfo][RankTableInfo::%s] failed with rank_id is "
                "out of range. version[%s], rankCount[%u], rank_id[%d]",
                __func__, version.c_str(), rankCount, rank.rankId));
        }
        if (rankIdSet.count(rank.rankId) > 0) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(rank.rankId), "rankId", "rank_id is not repeat"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] rankId[%d] is repeat", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rank.rankId);
            THROW<InvalidParamsException>(StringFormat(
                "[Parse][ClusterInfo][RankTableInfo::%s] failed with rank_id is "
                "repeat. version[%s], rankCount[%u], rank_id[%d]",
                __func__, version.c_str(), rankCount, rank.rankId));
        }
        rankIdSet.insert(rank.rankId);

        if (rank.localId != BACKUP_LOCAL_ID && rank.localId != rank.replacedLocalId) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>(
                    {std::to_string(rank.replacedLocalId), "replacedLocalId",
                     "replacedLocalId equal to localId[" + std::to_string(rank.localId) + "]"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] replacedLocalId[%u] != localId[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rank.replacedLocalId,
                rank.localId);
            THROW<InvalidParamsException>(StringFormat(
                "[Parse][ClusterInfo][RankTableInfo::Check] "
                "failed with replacedLocalId[%u] not equal to localId[%u].",
                rank.replacedLocalId, rank.localId));
        } else if (rank.localId == BACKUP_LOCAL_ID) {
            if (recordedReplaceLocalId == UNDEFIEND_LOCAL_ID) {
                recordedReplaceLocalId = rank.replacedLocalId;
            } else {
                RPT_INPUT_ERR(
                    true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                    std::vector<std::string>({"NA", "NA", "multiple replaced rank is configured"}));
                HCCL_ERROR(
                    "[%s][%s] errNo[0x%016llx] multiple replaced rank is configured", LOG_KEYWORDS_INIT_GROUP.c_str(),
                    LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
                THROW<InvalidParamsException>(StringFormat("[Parse][ClusterInfo][RankTableInfo::Check] "
                                                           "multiple replaced rank is configured"));
            }
        } else {
            localIdSet.emplace(rank.localId);
        }
    }

    for (u32 rankRange = 0; rankRange < rankCount; rankRange++) {
        if (rankIdSet.find(rankRange) == rankIdSet.end()) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(rankRange), "rankId", "rank_id is continuous"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] rankId[%u] is not continuous", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankRange);
            THROW<InvalidParamsException>(StringFormat(
                "[Parse][ClusterInfo][RankTableInfo::%s] failed with rank_id is "
                "not continuous. version[%s], rankCount[%u], rankRange[%d]",
                __func__, version.c_str(), rankCount, rankRange));
        }
    }

    std::vector<std::unordered_map<std::string, u32>> verifyRankAddr;
    for (auto& rank : ranks) {
        for (auto& levelInfo : rank.rankLevelInfos) {
            InsertToRank(levelInfo.netInstId, levelInfo.rankAddrs.size(), verifyRankAddr, levelInfo.netLayer);
        }
    }

    if (localIdSet.find(recordedReplaceLocalId) != localIdSet.end()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(recordedReplaceLocalId), "recordedReplacedLocalId",
                 "failed with configuring same local_id with replaced one simultaneously"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] configuring same local_id[%u] with replaced one simultaneously",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), recordedReplaceLocalId);
        THROW<InvalidParamsException>(StringFormat(
            "[Parse][ClusterInfo][RankTableInfo::%s] failed with configuring "
            "same local_id[%u] with replaced one simultaneously",
            __func__, recordedReplaceLocalId));
    }
}

void RankTableInfo::Deserialize(const nlohmann::json& rankTableInfoJson, bool isCheck, RankTableSource source)
{
    source_ = source;
    std::string msgVersion = "error occurs when parser object of propName \"version\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgVersion, (version = GetJsonProperty(rankTableInfoJson, "version")),
        rankTableInfoJson, "version", "non-empty string", source_);
    std::string msgStatus = "error occurs when parser object of propName \"status\"";

    std::string detourStr;
    std::string msgDetour = "error occurs when parser object of propName \"detour\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgDetour, (detourStr = GetJsonProperty(rankTableInfoJson, "detour", false)),
        rankTableInfoJson, "detour", "non-empty string", source_);
    if (detourStr == "true") {
        detour = true;
    } else if (detourStr == "false" || detourStr == "") {
        detour = false;
    } else {
        THROW<InvalidParamsException>(StringFormat("Invalid detour value [%s]", detourStr.c_str()));
    }

    std::string msgRankcount = "error occurs when parser object of propName \"rank_count\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgRankcount, (rankCount = GetJsonPropertyUInt(rankTableInfoJson, "rank_count")),
        rankTableInfoJson, "rank_count", "0 ~ UINT32_MAX", source_);

    nlohmann::json rankJsons;
    std::string msgRanklist = "error occurs when parser object of propName \"rank_list\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgRanklist, (GetJsonPropertyList(rankTableInfoJson, "rank_list", rankJsons)),
        rankTableInfoJson, "rank_list", "array", source_);
    for (auto& rankJson : rankJsons) {
        NewRankInfo rankInfo;
        rankInfo.Deserialize(rankJson, source_);
        ranks.emplace_back(rankInfo);
    }

    // check
    if (isCheck) {
        Check();
    }
}

void RankTableInfo::CheckAndInsert(
    const std::string& levelId, u32 rankAddrSize, std::unordered_map<std::string, u32>& idRankSizeMap) const
{
    if (idRankSizeMap.find(levelId) != idRankSizeMap.end() && idRankSizeMap[levelId] != rankAddrSize) {
        THROW<InvalidParamsException>(StringFormat(
            "[RankTableInfo::%s] failed with the size of "
            "rank_addrs with the same id is different. leveId[%s],"
            "rankAddrSize[%u]",
            __func__, levelId.c_str(), rankAddrSize));
    }
    idRankSizeMap[levelId] = rankAddrSize;
}

void RankTableInfo::InsertToRank(
    const std::string& levelId, u32 rankAddrSize, std::vector<std::unordered_map<std::string, u32>>& rankLists,
    u32 levelNum) const
{
    if (rankLists.size() <= levelNum) {
        rankLists.resize(levelNum + 1);
    }
    CheckAndInsert(levelId, rankAddrSize, rankLists[levelNum]);
}

std::string RankTableInfo::Describe() const
{
    return StringFormat(
        "RankTableInfo[version=%s, rankCount=%u, ranks size=%d]", version.c_str(), rankCount, ranks.size());
}

void RankTableInfo::Dump() const
{
    HCCL_DEBUG("RankTableInfo Dump:");
    HCCL_DEBUG("%s", Describe().c_str());
    HCCL_DEBUG("ranks:");
    for (const auto& rank : ranks) {
        HCCL_DEBUG("%s", rank.Describe().c_str());
        for (const auto& levelInfo : rank.rankLevelInfos) {
            HCCL_DEBUG("- %s", levelInfo.Describe().c_str());
        }
    }
}

RankTableInfo::RankTableInfo(BinaryStream& binaryStream)
{
    binaryStream >> version >> rankCount;
    size_t ranksSize = 0;
    binaryStream >> ranksSize;
    HCCL_INFO("[%s] version[%s] rankCount[%u] ranks size[%u]", __func__, version.c_str(), rankCount, ranksSize);
    for (u32 i = 0; i < ranksSize; i++) {
        NewRankInfo rankInfo(binaryStream);
        ranks.emplace_back(rankInfo);
    }
    binaryStream >> detour;
}

void RankTableInfo::GetBinStream(bool isContainLocId, BinaryStream& binaryStream) const
{
    if (ranks.size() == 0) {
        std::string msg = StringFormat("ranks size is zero.");
        THROW<InvalidParamsException>(msg);
    }
    HCCL_INFO("[%s] version[%s] rankCount[%u] ranks size[%u]", __func__, version.c_str(), rankCount, ranks.size());

    binaryStream << version << rankCount;
    binaryStream << ranks.size();
    for (auto& it : ranks) {
        it.GetBinStream(isContainLocId, binaryStream);
    }
    binaryStream << detour;
}

vector<char> RankTableInfo::GetUniqueId(bool isContainLocId) const
{
    if (ranks.size() == 0) {
        std::string msg = StringFormat("ranks size is zero.");
        THROW<InvalidParamsException>(msg);
    }
    std::vector<char> result(0);

    BinaryStream binaryStream;
    binaryStream << version << rankCount;

    u32 ranksSize = ranks.size();
    binaryStream << ranksSize;
    for (auto& it : ranks) {
        it.GetBinStream(isContainLocId, binaryStream);
    }

    binaryStream.Dump(result);
    return result;
}

void RankTableInfo::UpdateRankTable(const RankTableInfo& localRankInfo)
{
    // version
    if (detour) {
        CHK_PRT_THROW(
            localRankInfo.detour != true, HCCL_ERROR("[%s] detour cfg is not same with other ranks.", __func__),
            InvalidParamsException, "updateRankTableInfo error");
    }
    detour = localRankInfo.detour;
    if (rankCount == 0) {
        version = localRankInfo.version;
    } else {
        CHK_PRT_THROW(
            version != localRankInfo.version,
            HCCL_ERROR(
                "[%s] version[%s] error, local version[%s] .", __func__, version.c_str(),
                localRankInfo.version.c_str()),
            InvalidParamsException, "updateRankTableInfo error");
    }

    // ranks size
    CHK_PRT_THROW(
        localRankInfo.ranks.size() == 0, HCCL_ERROR("[%s] ranks size is zero.", __func__), InvalidParamsException,
        "updateRankTableInfo error");

    ranks.insert(ranks.end(), localRankInfo.ranks.begin(), localRankInfo.ranks.end());
    rankCount++;

    HCCL_INFO("[%s] success, current rankTableInfo[%s]", __func__, Describe().c_str());
}

std::unordered_map<u32, std::unordered_map<IpAddress, u32>> RankTableInfo::GetRankDeviceListenPortMap()
{
    std::unordered_map<u32, std::unordered_map<IpAddress, u32>> ranklListenPortMap;
    for (auto& rankinfo : ranks) {
        std::unordered_map<IpAddress, u32> listenPortMap;
        for (auto& rankLevelInfo : rankinfo.rankLevelInfos) {
            for (auto& rankAddr : rankLevelInfo.rankAddrs) {
                listenPortMap.insert(std::make_pair(rankAddr.addr, rankAddr.socketPort_));
            }
        }
        listenPortMap.insert(std::make_pair(DEVICE_PORT_KEY_IPADDRESS, rankinfo.devicePort));
        ranklListenPortMap.insert(std::make_pair(rankinfo.rankId, listenPortMap));
    }
    return ranklListenPortMap;
}

} // namespace Hccl
