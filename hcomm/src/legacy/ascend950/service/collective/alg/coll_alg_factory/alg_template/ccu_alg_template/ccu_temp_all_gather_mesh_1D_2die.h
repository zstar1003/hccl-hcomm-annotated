/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_ALL_MESH_1D_2DIE_H_
#define HCCLV2_CCU_TEMP_ALL_MESH_1D_2DIE_H_
#include "string_util.h"
#include "env_config_v2.h"
#include "ccu_alg_template_base.h"
#include "ccu_instruction_all_gather_mesh1d_2die.h"
#include "executor_utils.h"

namespace Hccl {
class CcuTempAllGatherMesh1D2Die : public CcuAlgTemplateBase {
public:
    explicit CcuTempAllGatherMesh1D2Die(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~CcuTempAllGatherMesh1D2Die() override;

    std::string Describe() const override
    {
        return StringFormat("Template of all gather ccu mesh 1D 2Die with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    uint64_t GetMaxSliceSize() const;
};

} // namespace Hccl

#endif // HCCLV2_CCU_TEMP_ALL_GATHER_NHR_MESH_1D_MEM2MEM_H_
