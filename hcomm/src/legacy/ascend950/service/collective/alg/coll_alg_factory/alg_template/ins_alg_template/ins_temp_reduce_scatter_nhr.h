/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_REDUCE_SCATTER_NHR
#define HCCLV2_INS_TEMP_REDUCE_SCATTER_NHR

#include "string_util.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_alg_template_base.h"
namespace Hccl {

class InsTempReduceScatterNHR : public InsAlgTemplateBase {
public:
    explicit InsTempReduceScatterNHR(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempReduceScatterNHR() override;

    std::string Describe() const override
    {
        return StringFormat("Template of reduce scatter NHR with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult
    Run(const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo,
        const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues) override;
    HcclResult CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec) override;
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    HcclResult
    PreCopy(const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, std::vector<InsQuePtr>& tempInsQues);
    HcclResult
    PostCopy(const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, std::vector<InsQuePtr>& tempInsQues);
    HcclResult GetScratchBufferInfo(const uint64_t scratchBufferSize, DataType dataType) const;
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    u32 CalcScratchMultiple(const BufferType& inBufferTpye, const BufferType& outBufferTpye) const
    {
        (void)inBufferTpye;
        (void)outBufferTpye;
        HCCL_INFO("[InsTempReduceScatterNHR][CalcScratchMultiple] templateScratchMultiplier[%llu]", tempRankSize_);
        return tempRankSize_;
    }

private:
    HcclResult MultiSliceLocalCopy(
        InsQuePtr& insQue, const std::vector<DataSlice>& srcList, const std::vector<DataSlice>& dstList) const;
    HcclResult
    RunReduceScatter(const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues);
    HcclResult GetStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList);
    RankId GetRankFromMap(const u32 rankIdx);
    HcclResult LocalDataCopy(std::vector<InsQuePtr>& tempInsQues, const TempFuncs& tempFuncs, u32 linkIdx);
    HcclResult RunNHR(std::vector<InsQuePtr>& tempInsQues, u32 linkIdx);
    HcclResult PostLocalCopy(std::vector<InsQuePtr>& tempInsQues, u32 linkIdx);
    TemplateDataParams tempAlgParams_;
    ResLinks tempLinks_;
    vector<u64> processSize_{0};
    vector<u64> linkSliceOffsetVec_{0};
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_REDUCE_SCATTER_NHR
