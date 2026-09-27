/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_REDUCE_MESH_2D_MEM2MEM_H_
#define HCCLV2_CCU_TEMP_REDUCE_MESH_2D_MEM2MEM_H_

#include "string_util.h"
#include "env_config_v2.h"
#include "ccu_alg_template_base.h"
#include "ccu_instruction_reduce_mesh2d_mem2mem.h"
#include "executor_utils.h"

namespace Hccl {

class CcuTempReduceMeshMem2Mem2D : public CcuAlgTemplateBase {
public:
    explicit CcuTempReduceMeshMem2Mem2D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~CcuTempReduceMeshMem2Mem2D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Reduce ccu mesh mem2mem 2D with tempRankSize [%u].", tempRankSize_);
    }

    void InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);

private:
    ReduceOp reduceOp_;
    DataType dataType_;
    std::vector<uint64_t> dimSize_;
    std::vector<LinkData> linksX_;
    std::vector<LinkData> linksY_;
};

} // namespace Hccl

#endif // HCCLV2_CCU_TEMP_REDUCE_MESH_2D_MEM2MEM_H_
