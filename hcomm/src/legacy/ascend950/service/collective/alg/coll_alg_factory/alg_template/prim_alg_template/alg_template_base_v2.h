/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ALG_TEMPLATE_BASE
#define HCCLV2_ALG_TEMPLATE_BASE

#include <memory>
#include <map>
#include <vector>
#include <queue>
#include <string>
#include "const_val.h"
#include "template_utils.h"
#include "primitive.h"
#include "prim_queue.h"

namespace Hccl {

using PrimQuePtr = std::shared_ptr<PrimQueue>;

class AlgTemplateBase {
public:
    explicit AlgTemplateBase(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    virtual ~AlgTemplateBase();

    virtual std::string Describe() const = 0;

    virtual HcclResult GenPrimQue(
        const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo,
        const ResLinks& tempLinks, std::vector<PrimQuePtr>& tempPrimQues)
        = 0;

    // init reduceInfo
    void InitReduceInfo(const ReduceOp& redOp, const DataType& dataType);
    void SetDataType(const DataType& dataType);

    void SetDmaMode(const DmaMode dmaMode);

    // calculate slices
    virtual HcclResult CalcSliceInfo(
        const AllignInfo& allignInfo, const bool forAllReduce, const u64 dataSize, RankSliceInfo& sliceInfoVec);
    virtual HcclResult CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec);

    // calculate resources
    virtual HcclResult CalcRes(AlgTempResReq& tempResReq);
    virtual HcclResult CalcResDetour(const RankGraph* rankGraph, AlgTempResReq& tempResReq);
    virtual HcclResult CalcResDetour(ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq);

    virtual HcclResult CalcRes(
        const bool forAllReduce, AlgTempResReq& tempResReq,
        u32& requiredScratchMultiplier); // reduction-involved operations
    virtual HcclResult CalcResDetour(
        const bool forAllReduce, const RankGraph* rankGraph, AlgTempResReq& tempResReq,
        u32& requiredScratchMultiplier); // reduction-involved
    virtual HcclResult CalcResDetour(
        const bool forAllReduce, ConnectedLinkMgr* linkMgr, AlgTempResReq& tempResReq, u32& requiredScratchMultiplier);

    // Sync
    HcclResult PreSync(const u32 queIdx, std::vector<PrimQuePtr>& syncPrimQues) const;
    HcclResult PostSync(const u32 queIdx, std::vector<PrimQuePtr>& syncPrimQues) const;
    HcclResult PreSyncInterQueues(std::vector<PrimQuePtr>& syncPrimQues) const;
    HcclResult PostSyncInterQueues(std::vector<PrimQuePtr>& syncPrimQues) const;

protected:
    HcclResult PostCopyOpbase(const UsrData& usrData, std::vector<PrimQuePtr>& tempPrimQues) const;
    HcclResult PreCopyOpbase(const UsrData& usrData, std::vector<PrimQuePtr>& tempPrimQues) const;

    OpMode opMode_;

    RankId myRank_ = INVALID_RANKID;
    u32 tempRankSize_ = 0;
    std::vector<std::vector<RankId>> tempVTopo_;
    std::map<RankId, u32> tempVirtRankMap_;

    BuffInfo buffInfo_;

    u32 queNum_ = 0;
    ReduceOp redOp_;
    DataType dataType_;

    bool enableCounterNotify_ = false;

    DmaMode dmaMode_ = DmaMode::DEFAULT;

    bool enableDetour_ = false;
    u32 linkNumBtwPeers_ = 1;
};

} // namespace Hccl

#endif // !HCCLV2_ALG_TEMPLATE_BASE
