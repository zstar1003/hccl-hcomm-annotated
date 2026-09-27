/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_ALL_REDUCE_1D_MESH_TWO_SHOT_MESH_CHUNK
#define HCCLV2_INS_TEMP_ALL_REDUCE_1D_MESH_TWO_SHOT_MESH_CHUNK

#include "string_util.h"
#include "ins_alg_template_base.h"
#include "executor_utils.h"

namespace Hccl {

class InsTempAllReduceMesh1DTwoShotMeshChunk : public InsAlgTemplateBase {
public:
    explicit InsTempAllReduceMesh1DTwoShotMeshChunk(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempAllReduceMesh1DTwoShotMeshChunk() override;

    std::string Describe() const override
    {
        return StringFormat("Template of all reduce 1D mesh chunk with tempRankSize [%u].", tempRankSize_);
    }

    u32 CalcScratchMultiple(BufferType input, BufferType output) const;
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    RankId GetRankFromMap(const u32 rankIdx);
    HcclResult PreCopy(
        const TemplateDataParams& tempAlgParams, const RankSliceInfo& sliceInfoVec,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult ReduceScatterMeshChunk(
        const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
        const TemplateDataParams& tempAlgParams, const std::vector<vector<u64>>& sliceSize, const u32& stepIndex,
        const u32& myAlgRank);

private:
    HcclResult RunReduceScatter(
        const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
        const TemplateDataParams& tempAlgParams);
    HcclResult RunAllgather(
        const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
        const TemplateDataParams& tempAlgParams);
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_ALL_REDUCE_1D_MESH_TWO_SHOT_MESH_CHUNK
