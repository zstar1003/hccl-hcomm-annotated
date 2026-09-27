/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TEMP_ALL_GATHER_RING
#define HCCLV2_TEMP_ALL_GATHER_RING

#include "string_util.h"

#include "alg_template_base_v2.h"

namespace Hccl {

class TempAllGatherRing : public AlgTemplateBase {
public:
    explicit TempAllGatherRing(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~TempAllGatherRing() override;

    std::string Describe() const override
    {
        return StringFormat("Template of all gather ring with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenPrimQue(
        const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo,
        const ResLinks& tempLinks, std::vector<PrimQuePtr>& tempPrimQues) override;
    using AlgTemplateBase::CalcSliceInfo;
    HcclResult CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec) override;
    using AlgTemplateBase::CalcRes;
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;

private:
    HcclResult
    PreCopyOffload(const RankSliceInfo& sliceInfoVec, const bool forAllReduce, std::vector<PrimQuePtr>& tempPrimQues);
    HcclResult RunIndividualRing(
        const u32 queIdx, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, PrimQuePtr currPrimQue);

    u32 stepNum_ = 0;
};

} // namespace Hccl

#endif // !HCCLV2_TEMP_ALL_GATHER_RING
