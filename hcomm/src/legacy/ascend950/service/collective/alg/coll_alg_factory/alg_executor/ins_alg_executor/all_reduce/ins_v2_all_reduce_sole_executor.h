/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_REDUCE_SOLE_EXECUTOR_H
#define HCCLV2_INS_V2_ALL_REDUCE_SOLE_EXECUTOR_H
#include "ins_coll_alg_base.h"

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsV2AllReduceSoleExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllReduceSoleExecutor();
    ~InsV2AllReduceSoleExecutor() override;

    std::string Describe() const override { return "Instruction based All Reduce Sole Executor."; }

    // AICPU 接口
    HcclResult Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue) override;

    HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) override;

    HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) override;
    HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit) override;
    // HOST 接口
    HcclResult Orchestrate(
        const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue) override;

private:
    HcclResult CreateTemplates(std::shared_ptr<InsAlgTemplate>& algTemplatePtr);
    HcclResult GetTemplateResRequest(
        const RankGraph* rankGraph, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const;
    HcclResult GetTemplateResRequest(
        ConnectedLinkMgr* linkMgr, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const;
    HcclResult OrchestrateLoop(std::shared_ptr<InsAlgTemplate> algTemplate);
    HcclResult InitCommInfo(const RankGraph* rankGraph);
    HcclResult InitCommInfo(const AlgTopoInfo& topoInfo);

    std::vector<RankId> virtRanks_;
    std::vector<std::vector<RankId>> vTopo_;
    ResLinks tempResLinks_;
    std::vector<InsQuePtr> tempInsQue_;
    std::map<RankId, u32> virtRankMap_;
};
} // namespace Hccl

#endif // HCCLV2_INS_V2_ALL_REDUCE_SOLE_EXECUTOR_H
