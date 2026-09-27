/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_REDUCE_MESH_1D_TWO_SHOT_H
#define HCCLV2_INS_TEMP_REDUCE_MESH_1D_TWO_SHOT_H

#include "string_util.h"
#include "ins_alg_template_base.h"
#include "executor_utils.h"

namespace Hccl {

class InsTempReduceMesh1DTwoShot : public InsAlgTemplateBase {
public:
    explicit InsTempReduceMesh1DTwoShot(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempReduceMesh1DTwoShot() override;

    std::string Describe() const override
    {
        return StringFormat("Template of reduce 1D mesh two shot with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const;
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);

private:
    HcclResult CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec);
    HcclResult RunReduceScatter(
        const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
        const TemplateDataParams& tempAlgParams);
    HcclResult RunGatherToRoot(
        const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
        const TemplateDataParams& tempAlgParams);
    RankId GetRankFromMap(const u32 rankIdx);

    u32 myIdx_ = INVALID_U32;

    std::vector<RankId> idxToRankMap_;
};

} // namespace Hccl

#endif // HCCLV2_INS_TEMP_REDUCE_MESH_1D_TWO_SHOT_H
