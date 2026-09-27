/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_SCATTER_MESH_1D
#define HCCLV2_INS_TEMP_SCATTER_MESH_1D

#include "string_util.h"

#include "ins_alg_template_base.h"
#include "executor_utils.h"

namespace Hccl {

class InsTempScatterMesh1D : public InsAlgTemplateBase {
public:
    explicit InsTempScatterMesh1D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempScatterMesh1D() override;

    std::string Describe() const override
    {
        return StringFormat("Instruction based Template of scatter mesh with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, ResLinks& tempResLinks,
        std::vector<InsQuePtr>& tempInsQues);
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    uint64_t GetExpandedMode() const;

private:
    HcclResult RunMesh(TemplateDataParams& tempAlgParams, ResLinks& tempResLinks, std::vector<InsQuePtr>& tempInsQues);
    HcclResult RunMeshTx(
        u32 myAlgRank, u32 repeatTimes, const TemplateDataParams& tempAlgParams, ResLinks& tempResLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult RunMeshRx(
        u32 myAlgRank, u32 repeatTimes, const TemplateDataParams& tempAlgParams, ResLinks& tempResLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult PreCopy(TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues);
    HcclResult PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues);
    u32 majorQueNum_ = 0;
    u32 queNumPerNeighbor_ = 1;
    bool enableInterRankCounterNotify_ = false;
    bool isZeroCopy_ = false;
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_SCATTER_MESH
