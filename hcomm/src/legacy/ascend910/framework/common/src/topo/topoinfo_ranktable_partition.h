/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_RANKTABLE_PARTITION_H
#define TOPOINFO_RANKTABLE_PARTITION_H

#include <hccl/base.h>
#include <nlohmann/json.hpp>
#include "topoinfo_struct.h"
#include "hccl_types.h"
#include "comm.h"

namespace hccl {
class TopoinfoRanktablePartition {
public:
    explicit TopoinfoRanktablePartition(hccl::HcclCommParams& globalParams, hccl::RankTable_t& globalRankTable);
    ~TopoinfoRanktablePartition();

    HcclResult GenerateSubRankTable(const uint32_t rankNum, const uint32_t* rankIds, hccl::RankTable_t& subRankTable);
    HcclResult GenerateSubParams(
        const hccl::RankTable_t& subRankTable, const uint32_t subCommRankId, hccl::HcclCommParams& subParams);
    HcclResult GetRankTableStr(const hccl::RankTable_t& subRankTable, std::string& rankTableStr);

private:
    hccl::HcclCommParams& globalParams_;
    hccl::RankTable_t& globalRankTable_;

    HcclResult TransformRankInfo(const RankTable_t& clusterInfo, nlohmann::json& perRankJson, u32 rankIndex);
    HcclResult TransformServerList(const RankTable_t& clusterInfo, nlohmann::json& rankListJson);
    HcclResult
    Struct2JsonRankTable(const RankTable_t& clusterInfo, const DevType deviceType, nlohmann::json& ClusterJson);
    HcclResult GenerateSubSuperPodId(hccl::RankTable_t& subRankTable);
};
} // namespace hccl
#endif // TOPOINFO_RANKTABLE_PARTITION_H
