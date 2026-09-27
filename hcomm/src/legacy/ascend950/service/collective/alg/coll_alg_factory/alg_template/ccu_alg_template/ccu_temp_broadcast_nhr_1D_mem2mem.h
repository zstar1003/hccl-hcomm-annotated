/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_BROADCAST_NHR_1D_H_
#define HCCLV2_CCU_TEMP_BROADCAST_NHR_1D_H_
#include "string_util.h"
#include "env_config_v2.h"
#include "ccu_alg_template_base.h"
#include "ccu_instruction_broadcast_nhr1d_mem2mem.h"
#include "executor_utils.h"

namespace Hccl {
class CcuTempBroadcastNHRMem2Mem1D : public CcuAlgTemplateBase {
public:
    explicit CcuTempBroadcastNHRMem2Mem1D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);

    ~CcuTempBroadcastNHRMem2Mem1D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Broadcast ccu nhr 1D mem2mem with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    uint64_t GetMaxSliceSize() const;
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

private:
    uint32_t virtRankId2RankId(const uint32_t virtRankId);
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo);
    HcclResult GetScatterStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo);
    HcclResult GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo);
};

} // namespace Hccl

#endif // HCCLV2_CCU_TEMP_BROADCAST_NHR_1D_MEM2MEM_H_
