/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TOPO_MATCH_MESH_NHR_PCIE
#define HCCLV2_TOPO_MATCH_MESH_NHR_PCIE
#include <string>
#include <vector>
#include <map>

#include "types.h"
#include "dev_type.h"
#include "topo_match_base.h"
#include "rank_gph.h"

namespace Hccl {

class TopoMatchMeshNHRPcie : public TopoMatchBase {
public:
    explicit TopoMatchMeshNHRPcie(
        const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType);
    ~TopoMatchMeshNHRPcie() override;

    std::string Describe() const override
    {
        return "Topo Match for combined Algorithm: level 0 Mesh, level 1 NHR, for PCIE only.";
    }
    using TopoMatchBase::MatchTopo;
    HcclResult MatchTopo(
        std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks,
        std::vector<std::map<RankId, u32>>& virtRankMap) override;

private:
    HcclResult LoadTopoInstRanks(u32 topoInstId, std::vector<RankId>& ranksOfSameLinkType) const;
    HcclResult DeduplicateLevelRanks(std::vector<RankId>& level0Ranks, std::vector<RankId>& level1Ranks) const;
};
} // namespace Hccl

#endif // !HCCLV2_TOPO_MATCH_MESH_NHR
