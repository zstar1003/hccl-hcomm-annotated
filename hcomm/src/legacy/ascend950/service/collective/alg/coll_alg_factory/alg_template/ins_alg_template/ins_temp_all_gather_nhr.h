/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_ALL_GATHER_NHR
#define HCCLV2_INS_TEMP_ALL_GATHER_NHR

#include "string_util.h"
#include "executor_utils.h"
#include "ins_alg_template_base.h"

namespace Hccl {

using AicpuNHRStepInfo = struct AicpuNHRStepInfo {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    AicpuNHRStepInfo() : nSlices(0) {}
};

class InsTempAllGatherNHR : public InsAlgTemplateBase {
public:
    explicit InsTempAllGatherNHR(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempAllGatherNHR() override;

    std::string Describe() const override
    {
        return StringFormat("Template of all gather NHR with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec) override;
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo);
    RankId GetRankFromMap(const u32 rankIdx);

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);

    u32 CalcScratchMultiple(const BufferType& inBufferTpye, const BufferType& outBufferTpye) const
    {
        (void)inBufferTpye;
        (void)outBufferTpye;
        HCCL_INFO("[InsTempAllGatherNHR][CalcScratchMultiple] templateScratchMultiplier[%llu]", tempRankSize_);
        return tempRankSize_;
    }

private:
    TemplateDataParams tempAlgParams_;
    ResLinks tempLinks_;

    HcclResult LocalDataCopy(std::vector<InsQuePtr>& tempInsQues);
    HcclResult PostLocalCopy(std::vector<InsQuePtr>& tempInsQues);
    HcclResult RunNHR(std::vector<InsQuePtr>& tempInsQues);
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_ALL_GATHER_NHR
