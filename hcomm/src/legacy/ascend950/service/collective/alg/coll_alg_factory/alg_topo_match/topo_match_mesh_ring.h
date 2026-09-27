/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TOPO_MATCH_MESH_RING
#define HCCLV2_TOPO_MATCH_MESH_RING
#include <string>
#include <vector>
#include <map>

#include "types.h"
#include "dev_type.h"
#include "topo_match_base.h"
#include "rank_gph.h"
#include "net_instance.h"

namespace Hccl {

class TopoMatchMeshRing : public TopoMatchBase {
public:
    explicit TopoMatchMeshRing(
        const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType);
    ~TopoMatchMeshRing() override;

    std::string Describe() const override { return "Topo Match for combined Algorithm: level 0 Mesh, level 1 Ring."; }
    using TopoMatchBase::MatchTopo;
    HcclResult MatchTopo(
        std::vector<std::vector<std::vector<RankId>>>& vTopo, std::vector<std::vector<RankId>>& virtRanks,
        std::vector<std::map<RankId, u32>>& virtRankMap) override;

private:
    HcclResult MeshRingTopoForAllLevel(
        std::set<RankId> rankSetR0, std::vector<std::vector<std::vector<RankId>>>& vTopo,
        std::vector<std::vector<RankId>>& virtRanks);
    std::vector<std::vector<RankId>> rankOnSameBoardVector_; // ranks vector with same boardIds in rack
    std::vector<std::vector<RankId>> rankOnSameSlotVector_;  // ranks vector with same slotIds in rack
    std::vector<u32> numRanksPerBoard_;                      // ranks num with same boardIds in rack
};
} // namespace Hccl

#endif // !HCCLV2_TOPO_MATCH_MESH_RING
