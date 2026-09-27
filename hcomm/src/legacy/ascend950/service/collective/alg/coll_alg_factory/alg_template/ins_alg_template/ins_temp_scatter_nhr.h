/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_SCATTER_NHR
#define HCCLV2_INS_TEMP_SCATTER_NHR

#include "string_util.h"

#include "ins_alg_template_base.h"
#include "ins_temp_all_gather_nhr.h"
namespace Hccl {

class InsTempScatterNHR : public InsAlgTemplateBase {
public:
    explicit InsTempScatterNHR(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempScatterNHR() override;

    std::string Describe() const override
    {
        return StringFormat("Template of scatter NHR with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const;
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    uint64_t GetExpandedMode() const;
    HcclResult GenExtIns(
        TempFuncs& tempFuncs, TemplateDataParams& templateDataParams, ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const;

private:
    HcclResult PreCopy(const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues);
    HcclResult PostCopy(const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues);
    HcclResult
    RunNHR(TemplateDataParams& templateDataParams, ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult BatchSend(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
        TemplateDataParams& templateDataParams, u32 repeat) const;
    HcclResult BatchRecv(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
        TemplateDataParams& templateDataParams, u32 repeat) const;
    HcclResult BatchSR(
        AicpuNHRStepInfo& stepInfo, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues,
        TemplateDataParams& templateDataParams, u32 repeat) const;
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_SCATTER_NHR
