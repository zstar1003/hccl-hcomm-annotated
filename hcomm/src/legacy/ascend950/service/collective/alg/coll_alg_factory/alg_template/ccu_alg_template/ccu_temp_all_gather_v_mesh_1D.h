/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_ALL_GATHER_V_MESH_1D_H_
#define HCCLV2_CCU_TEMP_ALL_GATHER_V_MESH_1D_H_

#include "string_util.h"
#include "env_config_v2.h"
#include "executor_utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_instruction_all_gather_v_mesh1d.h"

namespace Hccl {

class CcuTempAllGatherVMesh1D : public CcuAlgTemplateBase {
public:
    explicit CcuTempAllGatherVMesh1D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~CcuTempAllGatherVMesh1D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of all gather V ccu mesh 1D with tempRankSize [%u].", tempRankSize_);
    }
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
};

} // namespace Hccl

#endif // HCCLV2_CCU_TEMP_ALL_GATHER_V_MESH_1D_H_
