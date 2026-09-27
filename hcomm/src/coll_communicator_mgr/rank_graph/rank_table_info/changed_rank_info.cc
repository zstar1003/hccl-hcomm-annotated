/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "changed_rank_info.h"
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include "json_parser.h"
#include "exception_util.h"
#include "log.h"
#include "adapter_error_manager_pub.h"
#include "rank_table_report_macro.h"
namespace Hccl {

std::string ChangedRankInfo::Describe() const
{
    return StringFormat(
        "ChangedRankInfo[version=%s, rankCount=%u, ranks size=%d]", version.c_str(), rankCount, ranks.size());
}

void ChangedRankInfo::Dump() const
{
    HCCL_DEBUG("ChangedRankInfo Dump:");
    HCCL_DEBUG("%s", Describe().c_str());
    HCCL_DEBUG("ranks:");
    for (const auto& rank : ranks) {
        HCCL_DEBUG("%s", rank.Describe().c_str());
        for (const auto& levelInfo : rank.rankLevelInfos) {
            HCCL_DEBUG("    %s", levelInfo.Describe().c_str());
        }
    }
}

void ChangedRankInfo::Deserialize(const nlohmann::json& changedRankInfoJson, RankTableSource source)
{
    std::string msgVersion = "[ChangedRankInfo] error occurs when parser object of propName \"version\"";
    std::string msgRankcount = "[ChangedRankInfo] error occurs when parser object of propName \"rank_count\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgVersion, (version = GetJsonProperty(changedRankInfoJson, "version")),
        changedRankInfoJson, "version", "non-empty string", source);
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgRankcount, (rankCount = GetJsonPropertyUInt(changedRankInfoJson, "rank_count")),
        changedRankInfoJson, "rank_count", "0 ~ UINT32_MAX", source);

    nlohmann::json rankJsons;
    std::string msgRanklist = "error occurs when parser object of propName \"rank_list\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgRanklist, (GetJsonPropertyList(changedRankInfoJson, "rank_list", rankJsons)),
        changedRankInfoJson, "rank_list", "array", source);
    for (auto& rankJson : rankJsons) {
        NewRankInfo rankInfo;
        rankInfo.Deserialize(rankJson, source);
        ranks.emplace_back(rankInfo);
    }
}
} // namespace Hccl
