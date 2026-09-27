/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_ALL_REDUCE_PARALLEL_EXECUTOR_H
#define HCCLV2_INS_ALL_REDUCE_PARALLEL_EXECUTOR_H
#include "ins_coll_alg_base.h"

namespace Hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsAllReduceParallelExecutor : public InsCollAlgBase {
public:
    explicit InsAllReduceParallelExecutor();
    ~InsAllReduceParallelExecutor() override;

    std::string Describe() const override { return "Instruction based All Reduce Parallel Executor."; }

    HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) override;

    HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) override;
    // HOST 接口
    HcclResult Orchestrate(
        const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue) override;
    // AICPU 接口
    HcclResult Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue) override;

private:
    HcclResult GenInsQues(InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter);
    void GenTemplateAlgParams0(
        const u64 dataOffset, const u64 dataCount, const u64 scratchOffset, TemplateDataParams& tempAlgParams) const;
    void GenTemplateAlgParams1(
        const u64 dataOffset, const u64 dataCount, const u64 scratchOffset, TemplateDataParams& tempAlgParams) const;
    HcclResult CalcSendDataSize(u64& memBlockSize, float& SplitRate, u32& multipleIntra, u32& multipleInter);

    void GetParallelDataSplitRate(std::vector<float>& splitDataSize) const;
    HcclResult
    PrepareResForTemplate(const RankGraph* rankGraph, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter);
    HcclResult
    PrepareResForTemplate(ConnectedLinkMgr* linkMgr, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter);

    u32 rankSizeLevel0_{0};
    u32 rankSizeLevel1_{0};

    std::vector<std::vector<RankId>> virtRanks_;
    std::vector<std::map<RankId, u32>> virtRankMap_; // map<virtRank, virtRankOrder>
    std::vector<std::vector<std::vector<RankId>>> vTopo_;

    std::vector<InsQuePtr> requiredQue_;
    std::vector<InsQuePtr> intraQue_;
    std::vector<InsQuePtr> interQue_;
    std::vector<InsQuePtr> syncQueues_;
    ResLinks intraLinks_;
    ResLinks interLinks_;

    u64 interScratchOffset0{0};
    u64 interScratchOffset1{0};
    u64 Intra0ScratchSize{0};
    u64 Intra1ScratchSize{0};
    u64 Inter0ScratchSize{0};
    u64 Inter1ScratchSize{0};
    float dataSplitRate{0.5};
};

} // namespace Hccl

#endif // HCCLV2_INS_ALL_REDUCE_PARALLEL_EXECUTOR_H
