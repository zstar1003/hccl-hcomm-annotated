/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_SCATTER_PARALLEL_EXECUTOR_H
#define HCCLV2_INS_SCATTER_PARALLEL_EXECUTOR_H
#include "ins_coll_alg_base.h"

namespace Hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsScatterParallelExecutor : public InsCollAlgBase {
public:
    explicit InsScatterParallelExecutor();
    ~InsScatterParallelExecutor() override;

    HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) override;

    // HOST 接口
    HcclResult Orchestrate(
        const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue) override;

    HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) override;
    // AICPU 接口
    HcclResult Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue) override;

    std::string Describe() const override { return "Instruction based Scatter Parallel Executor."; }

private:
    void GenTemplateAlgParamsIntra0(
        const u64 dataOffset, const u64 dataCountPerLoopAixs0, const u64 scratchOffset,
        TemplateDataParams& tempAlgParamsIntra0) const;
    void GenTemplateAlgParamsInter0(
        const u64 dataOffset, const u64 dataCountPerLoopAixs0, const u64 scratchOffset,
        TemplateDataParams& tempAlgParamsInter0) const;
    void GenTemplateAlgParamsIntra1(
        const u64 dataOffset, const u64 dataCountPerLoopAixs1, const u64 scratchOffset,
        TemplateDataParams& tempAlgParamsIntra1) const;
    void GenTemplateAlgParamsInter1(
        const u64 dataOffset, const u64 dataCountPerLoopAixs1, const u64 scratchOffset,
        TemplateDataParams& tempAlgParamsInter1) const;
    HcclResult GenInsQuesHost(InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter);
    HcclResult
    PrepareResForTemplate(const RankGraph* rankGraph, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter);
    HcclResult
    PrepareResForTemplate(ConnectedLinkMgr* linkMgr, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter);
    void GetParallelDataSplit(std::vector<double>& splitDataSize) const;
    u32 rankSizeLevel0_{0};
    u32 rankSizeLevel1_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    const RankGraph* rankGraph_ = nullptr;

    std::vector<std::vector<RankId>> virtRanks_;
    std::vector<std::map<RankId, u32>> virtRankMap_; // map<virtRank, virtRankOrder>
    std::vector<std::vector<std::vector<RankId>>> vTopo_;

    ResLinks interLinks_;
    ResLinks intraLinks_;
    std::vector<InsQuePtr> requiredQue_;
    std::vector<InsQuePtr> interQue_;
    std::vector<InsQuePtr> intraQue_;
    std::vector<InsQuePtr> syncQueues_;

    const RankGraph* rankGraphPtr_ = nullptr;
};

} // namespace Hccl

#endif // HCCLV2_INS_SCATTER_PARALLEL_EXECUTOR_H
