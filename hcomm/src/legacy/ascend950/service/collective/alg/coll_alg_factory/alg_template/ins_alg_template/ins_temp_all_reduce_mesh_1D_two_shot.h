/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_ALL_REDUCE_1D_MESH_TWO_SHOT
#define HCCLV2_INS_TEMP_ALL_REDUCE_1D_MESH_TWO_SHOT

#include "string_util.h"
#include "ins_alg_template_base.h"
#include "executor_utils.h"

namespace Hccl {

class InsTempAllReduceMesh1DTwoShot : public InsAlgTemplateBase {
public:
    explicit InsTempAllReduceMesh1DTwoShot(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempAllReduceMesh1DTwoShot() override;

    std::string Describe() const override
    {
        return StringFormat("Template of all reduce 1D mesh with tempRankSize [%u].", tempRankSize_);
    }

    u32 CalcScratchMultiple(BufferType input, BufferType output) const;
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcSlice(const u64 dataSize, const u64 baseOff, RankSliceInfo& sliceInfoVec);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;

private:
    HcclResult RunAllReduceScatter(
        const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
        const TemplateDataParams& tempAlgParams, u32 linkIdx);
    HcclResult RunAllReduceAllgather(
        const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
        const TemplateDataParams& tempAlgParams, u32 linkIdx);
    RankId GetRankFromMap(const u32 rankIdx);
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_ALL_REDUCE_1D_MESH_TWO_SHOT
