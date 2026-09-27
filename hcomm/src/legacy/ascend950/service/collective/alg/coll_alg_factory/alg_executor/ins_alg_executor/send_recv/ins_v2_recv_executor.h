/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_V2_INS_RECV_EXECUTOR_H
#define HCCLV2_V2_INS_RECV_EXECUTOR_H

#include "ins_coll_alg_base.h"

namespace Hccl {

class InsV2RecvExecutor : public InsCollAlgBase {
public:
    explicit InsV2RecvExecutor();
    ~InsV2RecvExecutor() override;

    std::string Describe() const override { return "Instruction based Recv Executor."; }

    HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) override;

    HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) override;

    HcclResult Orchestrate(
        const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue) override;

    HcclResult Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue) override;
    HcclResult
    ExecAiv(const CollAlgOperator& op, const CollAlgParams& params, LinkData& recvLinkData, InsQuePtr insQue);
    HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit) override;

private:
    std::vector<RankId> virtRanks_;
    std::map<RankId, u32> virtRankMap_; // map<virtRank, virtRankOrder>
    std::vector<std::vector<RankId>> vTopo_;
    u32 sliceId_{0}; // 用于组装aivTag
};

} // namespace Hccl
#endif // HCCLV2_V2_INS_RECV_EXECUTOR_H
