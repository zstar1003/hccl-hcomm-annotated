/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TOPO_MATCH_NHR
#define HCCLV2_TOPO_MATCH_NHR
#include <string>
#include <vector>
#include <map>

#include "types.h"
#include "dev_type.h"
#include "topo_match_base.h"
#include "rank_gph.h"

namespace Hccl {

class TopoMatchNHR : public TopoMatchBase {
public:
    explicit TopoMatchNHR(const RankId vRank, const u32 rankSize, const RankGraph* rankGraph, const DevType devType);
    ~TopoMatchNHR() override;

    std::string Describe() const override { return "Topo Match for NHR Algorithm (CURRENTLY only 950 is supported)."; }
    using TopoMatchBase::MatchTopo;
    HcclResult MatchTopo(
        std::vector<std::vector<RankId>>& vTopo, std::vector<RankId>& virtRanks,
        std::map<RankId, u32>& virtRankMap) override;

private:
    HcclResult NHRTopoForAllRanks();
    u32 GetNHRStepNum(u32 rankSize) const;
    std::vector<RankId> rankIds_;                            // virtualTopoRank
    std::vector<std::vector<RankId>> rankOnSameBoardVector_; // ranks vector with same boardIds in rack
    std::vector<std::vector<RankId>> rankOnSameSlotVector_;  // ranks vector with same slotIds in rack
    std::vector<u32> numRanksPerBoard_;                      // ranks num with same boardIds in rack
};
} // namespace Hccl

#endif // !HCCLV2_TOPO_MATCH_NHR
