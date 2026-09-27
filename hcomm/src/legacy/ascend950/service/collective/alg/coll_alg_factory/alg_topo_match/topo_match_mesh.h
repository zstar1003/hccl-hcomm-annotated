/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TOPO_MATCH_MESH
#define HCCLV2_TOPO_MATCH_MESH
#include <string>
#include <vector>
#include <map>

#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "types/types.h"
#include "dev_type.h"
#include "topo_match_base.h"
#include "rank_gph.h"
#include "net_instance.h"

namespace Hccl {

class TopoMatchMesh : public TopoMatchBase {
public:
    explicit TopoMatchMesh(const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType);
    ~TopoMatchMesh() override;

    std::string Describe() const override { return "Topo Match for Mesh Algorithm (CURRENTLY only 950 is supported)."; }
    using TopoMatchBase::MatchTopo;
    HcclResult MatchTopo(
        std::vector<std::vector<RankId>>& vTopo, std::vector<RankId>& virtRanks,
        std::map<RankId, u32>& virtRankMap) override;

private:
    HcclResult MeshTopoForAllLevel();
    std::vector<RankId> rankIds_;                            // virtualTopoRank
    std::vector<std::vector<RankId>> rankOnSameBoardVector_; // ranks vector with same boardIds in rack
    std::vector<std::vector<RankId>> rankOnSameSlotVector_;  // ranks vector with same slotIds in rack
    std::vector<u32> numRanksPerBoard_;                      // ranks num with same boardIds in rack
};
} // namespace Hccl

#endif // !HCCLV2_TOPO_MATCH_MESH
