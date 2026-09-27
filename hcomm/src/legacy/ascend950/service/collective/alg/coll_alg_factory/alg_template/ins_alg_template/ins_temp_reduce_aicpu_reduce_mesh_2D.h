/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_REDUCE_AICPU_REDUCE_MESH_2D
#define HCCLV2_INS_TEMP_REDUCE_AICPU_REDUCE_MESH_2D

#include "string_util.h"

#include "ins_alg_template_base.h"
#include "executor_utils.h"

namespace Hccl {

class InsTempReduceAicpuReduceMesh2D : public InsAlgTemplateBase {
public:
    explicit InsTempReduceAicpuReduceMesh2D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempReduceAicpuReduceMesh2D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of aicpu reduce reduce 2D Mesh with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;

    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const;

private:
    HcclResult RunAicpuLocalReduce(const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues);
    HcclResult RunGatherToRootXY(
        const TemplateDataParams& templateDataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues);
    HcclResult RunGatherToRootX(
        const TemplateDataParams& templateDataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues);
    HcclResult RunGatherToRootY(
        const TemplateDataParams& templateDataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues);
    HcclResult RunXYGatherToRoot(
        const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult RunXGatherToRoot(
        const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult RunYGatherToRoot(
        const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues) const;
    u32 sizeX_{0};
    u32 sizeY_{0};
    u32 rootX_{0};
    u32 rootY_{0};
    u32 curX_{0};
    u32 curY_{0};
    u64 dataTypeSize_{0};
    u64 dataSizeX_{0};
    u64 dataSizeY_{0};
    u64 rankOffsetY_{0};
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_REDUCE_AICPU_REDUCE_MESH_2D
