/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_BROADCAST_NHR_H
#define HCCLV2_INS_TEMP_BROADCAST_NHR_H

#include "string_util.h"

#include "ins_alg_template_base.h"
#include "executor_utils.h"
#include "ins_temp_all_gather_nhr.h"

namespace Hccl {

class InsTempBroadcastNHR : public InsAlgTemplateBase {
public:
    explicit InsTempBroadcastNHR(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempBroadcastNHR() override;

    std::string Describe() const override
    {
        return StringFormat("Template of broadcast NHR with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);

    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType);

private:
    HcclResult PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult PreCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult
    RunScatter(const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues);
    HcclResult
    RunAllGather(const RankSliceInfo& sliceInfoVec, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues);
    HcclResult GetScatterStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const;
    HcclResult GetAllGatherStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo);
    HcclResult BatchTxRx(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec);
    HcclResult BatchSend(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
        BufferType memType, u64 memOffset) const;
    HcclResult BatchRecv(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
        BufferType memType, u64 memOffset) const;
    HcclResult BatchSR(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, InsQuePtr& queue, const RankSliceInfo& sliceInfoVec,
        BufferType memType, u64 memOffset) const;
    RankId GetRankFromMap(const u32 rankIdx) const;
    HcclResult CalcDataSliceInfo(const u64 dataSize, RankSliceInfo& sliceInfoVec);
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_BROADCAST_NHR_H
