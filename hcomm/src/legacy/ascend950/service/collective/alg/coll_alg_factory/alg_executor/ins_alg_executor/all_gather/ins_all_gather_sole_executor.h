/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_ALL_GATHER_SOLE_EXECUTOR
#define HCCLV2_INS_ALL_GATHER_SOLE_EXECUTOR
#include "ins_coll_alg_base.h"

namespace Hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsAllGatherSoleExecutor : public InsCollAlgBase {
public:
    explicit InsAllGatherSoleExecutor();
    ~InsAllGatherSoleExecutor() override;

    std::string Describe() const override { return "Instruction based All Gather Sole Executor."; }

    HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) override;
    HcclResult Orchestrate(
        const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue) override;
    HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) override;
    HcclResult Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue) override;

private:
    HcclResult GenInsQues4Offload(InsAlgTemplate& tempAlg);
    HcclResult GenInsQues4Opbase(InsAlgTemplate& tempAlg);

    std::vector<RankId> virtRanks_;
    std::map<RankId, u32> virtRankMap_; // map<virtRank, virtRankOrder>
    std::vector<std::vector<RankId>> vTopo_;

    std::vector<InsQuePtr> requiredQue_;
    ResLinks tempResLinks_;
};

} // namespace Hccl

#endif // HCCLV2_ALL_GATHER_SOLE_EXECUTOR
