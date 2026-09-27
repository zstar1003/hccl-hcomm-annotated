/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_GATHER_NHR
#define INS_TEMP_GATHER_NHR

#include "string_util.h"

#include "ins_alg_template_base.h"
#include "ins_temp_all_gather_nhr.h"
namespace Hccl {

class InsTempGatherNHR : public InsAlgTemplateBase {
public:
    explicit InsTempGatherNHR(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempGatherNHR() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Gather NHR with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult
    Run(const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo,
        const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues) override;
    HcclResult CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec) override;
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    HcclResult GetScatterStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const;
    uint64_t GetExpandedMode() const;

private:
    HcclResult
    PreCopy(const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, std::vector<InsQuePtr>& tempInsQues);
    HcclResult
    PostCopy(const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, std::vector<InsQuePtr>& tempInsQues);

    HcclResult BatchTxRx(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec);
    HcclResult BatchSend(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
        BufferType memType, u32 memOffset) const;
    HcclResult BatchRecv(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
        BufferType memType, u32 memOffset) const;
    HcclResult BatchSR(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
        BufferType memType, u32 memOffset) const;

    HcclResult GetGatherStepInfo(std::vector<AicpuNHRStepInfo>& nhrSteps) const;

    // 主要搬运所用的Buffer
    BufferType mainBufferType_;
    u64 mainBufferBaseOffset_{0};
};

} // namespace Hccl

#endif /* INS_TEMP_GATHER_NHR */
