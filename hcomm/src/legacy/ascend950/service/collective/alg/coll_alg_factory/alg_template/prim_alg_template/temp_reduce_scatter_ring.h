/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TEMP_REDUCE_SCATTER_RING
#define HCCLV2_TEMP_REDUCE_SCATTER_RING
#include <vector>
#include <map>
#include <string>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "types/types.h"
#include "string_util.h"
#include "template_utils.h"
#include "alg_template_base_v2.h"

namespace Hccl {

class TempReduceScatterRing : public AlgTemplateBase {
public:
    explicit TempReduceScatterRing(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~TempReduceScatterRing() override;

    std::string Describe() const override
    {
        return StringFormat("Template of reduce scatter ring with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenPrimQue(
        const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo,
        const ResLinks& tempLinks, std::vector<PrimQuePtr>& tempPrimQues) override;

    using AlgTemplateBase::CalcSliceInfo;
    HcclResult CalcSliceInfo(
        const AllignInfo& allignInfo, const bool forAllReduce, const u64 dataSize,
        RankSliceInfo& sliceInfoVec) override;

    using AlgTemplateBase::CalcRes;
    HcclResult CalcRes(const bool forAllReduce, AlgTempResReq& tempResReq, u32& requiredScratchMultiplier) override;

private:
    HcclResult CalcSliceInfoAllReduce(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec);

    HcclResult RunIndividualRing(
        const u32 queIdx, const bool& forAllReduce, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
        PrimQuePtr currPrimQue);

    HcclResult PostCopyOffload(const RankSliceInfo& sliceInfoVec, std::vector<PrimQuePtr>& tempPrimQues);

    u32 stepNum_ = 0;
};

} // namespace Hccl

#endif // !HCCLV2_TEMP_REDUCE_SCATTER_RING
