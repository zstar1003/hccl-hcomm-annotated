/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_REDUCE_EXECUTOR_H
#define HCCLV2_INS_REDUCE_EXECUTOR_H

#include "ins_coll_alg_base.h"
#include "topo_match_mesh.h"

namespace Hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsReduceSoleExecutor : public InsCollAlgBase {
public:
    explicit InsReduceSoleExecutor();
    ~InsReduceSoleExecutor() override;

    std::string Describe() const override { return "Instruction based Reduce Executor."; }

    // HOST 接口
    HcclResult Orchestrate(
        const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue) override;
    // AICPU 接口
    HcclResult Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue) override;

    HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) override;
    HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) override;

private:
    HcclResult OrchestrateOffload(InsAlgTemplate& tempAlg, const ParamPool& paramPool);
    HcclResult OrchestrateOpbase(InsAlgTemplate& tempAlg, ParamPool& paramPool);

    std::vector<RankId> virtRanks_;
    std::map<RankId, u32> virtRankMap_; // map<virtRank, virtRankOrder>
    std::vector<std::vector<RankId>> vTopo_;

    std::vector<InsQuePtr> requiredQue_;
    ResLinks tempResLinks_;
};

} // namespace Hccl

#endif // HCCLV2_INS_REDUCE_EXECUTOR_H
