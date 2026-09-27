/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ALL_REDUCE_COMB_EXECUTOR
#define HCCLV2_ALL_REDUCE_COMB_EXECUTOR

#include "temp_reduce_scatter_ring.h"
#include "temp_reduce_scatter_concurr_mesh.h"
#include "temp_reduce_scatter_mesh.h"
#include "temp_all_gather_ring.h"
#include "temp_all_gather_concurr_mesh.h"
#include "temp_all_gather_mesh.h"
#include "coll_alg_base.h"
#include "topo_match_concurr_mesh.h"
#include "topo_match_mesh.h"

namespace Hccl {

template <typename AlgTopoMatch, typename AlgTempRS, typename AlgTempAG>
class AllReduceCombExecutor : public CollAlgBase {
public:
    explicit AllReduceCombExecutor();
    ~AllReduceCombExecutor() override;

    std::string Describe() const override
    {
        return "All Reduce Comb Executor: sole TempReduceScatter + sole TempAllGather.";
    }

    HcclResult GenPrimQues(
        const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params,
        PrimQuePtr primQue) override;
    HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) override;
    HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) override;

    HcclResult GenPrimQuesAIC(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        PrimQuePtr primQue) override;

private:
    HcclResult GenPrimQues4Offload(AlgTemplateBase& tempRSAlg, AlgTemplateBase& tempAGAlg);
    HcclResult GenPrimQues4Opbase(
        const u32 requiredScratchMultiplier, const u32 dataSizePerVolume, AlgTemplateBase& tempRSAlg,
        AlgTemplateBase& tempAGAlg);

    std::vector<RankId> virtRanks_;
    std::map<RankId, u32> virtRankMap_; // map<virtRank, virtRankOrder>
    std::vector<std::vector<RankId>> vTopo_;

    std::vector<PrimQuePtr> requiredQue_;
    ResLinks tempResLinks_;
};

} // namespace Hccl

#endif // HCCLV2_ALL_REDUCE_COMB_EXECUTOR
