/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_ALL_REDUCE_2D_MESH_TWO_SHOT
#define HCCLV2_INS_TEMP_ALL_REDUCE_2D_MESH_TWO_SHOT

#include "string_util.h"
#include "ins_alg_template_base.h"
#include "executor_utils.h"

namespace Hccl {

struct SubStageArgs {
    BufferType inType;
    BufferType outType;
    u64 inbaseOff;
    u64 outbaesOff;
};

class InsTempAllReduceMesh2DTwoShot : public InsAlgTemplateBase {
public:
    explicit InsTempAllReduceMesh2DTwoShot(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempAllReduceMesh2DTwoShot() override;

    std::string Describe() const override
    {
        return StringFormat("Template of all reduce 1D mesh with tempRankSize [%u].", tempRankSize_);
    }
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    u32 CalcScratchMultiple(BufferType input, BufferType output) const;

private:
    std::vector<std::tuple<QId, QId, u32>>
    CreateQueNotifiesRequest(u32 queueNum, u32 pairNum, QId masterIdX, QId masterIdY) const;
    HcclResult BuildSlice(
        const std::vector<RankId>& rankInfo, const u64 dataSize, const u64 chunkSize,
        RankSliceInfo& sliceInfoVec) const;
    HcclResult RunReduceScatter(
        SubStageArgs& subparams, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues, const std::vector<RankId>& rankInfo) const;
    HcclResult RunAllgather(
        SubStageArgs& subparams, const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues, const std::vector<RankId>& rankInfo) const;
    HcclResult InitInnerParams(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);

    RankSliceInfo XsliceInfoVec_;
    RankSliceInfo YsliceInfoVec_;
    RankSliceInfo XsliceInfoVecS2_;
    RankSliceInfo YsliceInfoVecS2_;
    u64 XDataSize_{0};
    u64 YDataSize_{0};
    u32 M_{0};
    u32 N_{0};
    u64 chunkSize_{0};
    u32 XAlgrankId_{0};
    u32 YAlgrankId_{0};
    std::vector<InsQuePtr> XtempInsQues_;
    std::vector<InsQuePtr> YtempInsQues_;
    u64 YDataSizeS2_{0};
    u64 XDataSizeS2_{0};
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_ALL_REDUCE_2D_MESH_TWO_SHOT
