/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktable_partition.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace hccl {

TopoinfoRanktablePartition::TopoinfoRanktablePartition(
    hccl::HcclCommParams& globalParams, hccl::RankTable_t& globalRankTable)
    : globalParams_(globalParams),
      globalRankTable_(globalRankTable)
{}

TopoinfoRanktablePartition::~TopoinfoRanktablePartition() {}

HcclResult TopoinfoRanktablePartition::GenerateSubRankTable(
    [[maybe_unused]] const uint32_t rankNum, [[maybe_unused]] const uint32_t* rankIds,
    [[maybe_unused]] hccl::RankTable_t& subRankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktablePartition::GenerateSubSuperPodId([[maybe_unused]] hccl::RankTable_t& subRankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktablePartition::GenerateSubParams(
    [[maybe_unused]] const hccl::RankTable_t& subRankTable, [[maybe_unused]] const uint32_t subCommRankId,
    [[maybe_unused]] hccl::HcclCommParams& subParams)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktablePartition::GetRankTableStr(
    [[maybe_unused]] const hccl::RankTable_t& subRankTable, [[maybe_unused]] std::string& rankTableStr)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktablePartition::TransformRankInfo(
    [[maybe_unused]] const RankTable_t& clusterInfo, [[maybe_unused]] nlohmann::json& perRankJson,
    [[maybe_unused]] u32 rankIndex)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktablePartition::TransformServerList(
    [[maybe_unused]] const RankTable_t& clusterInfo, [[maybe_unused]] nlohmann::json& rankListJson)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktablePartition::Struct2JsonRankTable(
    [[maybe_unused]] const RankTable_t& clusterInfo, [[maybe_unused]] const DevType deviceType,
    [[maybe_unused]] nlohmann::json& ClusterJson)
{
    return HCCL_E_NOT_SUPPORT;
}
} // namespace hccl
