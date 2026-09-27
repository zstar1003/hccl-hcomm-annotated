/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_ALL_REDUCE_NHR
#define HCCLV2_INS_TEMP_ALL_REDUCE_NHR

#include "string_util.h"
#include "ins_alg_template_base.h"
#include "executor_utils.h"
#include "ins_temp_all_gather_nhr.h"

namespace Hccl {

class InsTempAllReduceNHR : public InsAlgTemplateBase {
public:
    explicit InsTempAllReduceNHR(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempAllReduceNHR() override;

    std::string Describe() const override
    {
        return StringFormat("Template of reduce scatter NHR with tempRankSize [%u].", tempRankSize_);
    }

    u32 CalcScratchMultiple(BufferType input, BufferType output);
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcSlice(const u64 dataSize, const u64 baseOff, RankSliceInfo& sliceInfoVec);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;

private:
    HcclResult PreCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues);
    HcclResult
    PrepareDataForAllGather(const RankSliceInfo& sliceInfoVec, std::vector<InsQuePtr>& tempInsQues, u32 linkIdx);
    HcclResult PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues);
    HcclResult RunReduceScatter(
        const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, u32 linkIdx);
    HcclResult RunAllGather(
        const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, u32 linkIdx);
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo);
    HcclResult GetStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList);
    RankId GetRankFromMap(const u32 rankIdx);

    BufferType nhrInBuffType_ = BufferType::INPUT;
    BufferType nhrOutBuffType_ = BufferType::OUTPUT;
    u64 nhrInBuffBaseOff_ = 0;
    u64 nhrOutBuffBaseOff_ = 0;
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_ALL_REDUCE_NHR
