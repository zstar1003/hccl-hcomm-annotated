/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_REDUCE_SCATTER_V_MESH_1D_H_
#define HCCLV2_CCU_TEMP_REDUCE_SCATTER_V_MESH_1D_H_

#include "string_util.h"
#include "env_config_v2.h"
#include "executor_utils.h"
#include "ccu_alg_template_base.h"

namespace Hccl {

class CcuTempReduceScatterVMesh1D : public CcuAlgTemplateBase {
public:
    explicit CcuTempReduceScatterVMesh1D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~CcuTempReduceScatterVMesh1D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Reduce Scatter V ccu mesh 1D with tempRankSize [%u].", tempRankSize_);
    }

    void InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);

private:
    ReduceOp reduceOp_;
    DataType dataType_;
};

} // namespace Hccl

#endif // HCCLV2_CCU_TEMP_REDUCE_SCATTER_V_MESH_1D_H_
