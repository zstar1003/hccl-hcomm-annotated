/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_ALL_GATHER_MESH_2D
#define HCCLV2_INS_TEMP_ALL_GATHER_MESH_2D

#include "string_util.h"
#include "executor_utils.h"
#include "ins_alg_template_base.h"

namespace Hccl {

class InsTempAllGatherMesh2D : public InsAlgTemplateBase {
public:
    explicit InsTempAllGatherMesh2D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempAllGatherMesh2D() override;

    std::string Describe() const override
    {
        return StringFormat("Instruction based Template of all gather mesh 2d with tempRankSize [%u].", tempRankSize_);
    }
    u32 CalcScratchMultiple(const BufferType& inBufferTpye, const BufferType& outBufferTpye)
    {
        (void)inBufferTpye;
        (void)outBufferTpye;
        if (op_.opMode == OpMode::OPBASE) {
            HCCL_INFO("[InsTempAllGatherMesh2D][CalcScratchMultiple] templateScratchMultiplier[%llu]", tempRankSize_);
            return tempRankSize_;
        } else {
            HCCL_INFO("[InsTempAllGatherMesh2D][CalcScratchMultiple] templateScratchMultiplier[0](offload)");
            return 0;
        }
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult Run2DStep1(std::vector<InsQuePtr>& xInsQues, std::vector<InsQuePtr>& yInsQues);
    HcclResult Run2DStep2(std::vector<InsQuePtr>& xInsQues, std::vector<InsQuePtr>& yInsQues);

    HcclResult CalcRes(AlgTempResReq& tempResReq) override;

private:
    HcclResult LocalDataCopy(std::vector<InsQuePtr>& tempInsQues);
    HcclResult PostLocalCopy(std::vector<InsQuePtr>& tempInsQues);

    HcclResult RunMesh(
        const u32 myAlgRank, RankId globalSrcRank, int rankOffset, const std::vector<RankId>& vTopo,
        std::vector<InsQuePtr>& tempInsQues, u64 xyOffset, u64 size, DmaMode dmaMode);

    u32 majorQueNum_ = 0;
    u32 xQueNum_ = 0;
    u32 yQueNum_ = 0;
    TemplateDataParams tempAlgParams_;
    ResLinks tempLinks_;
    TempFuncs tempFuncs_;
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_ALL_GATHER_MESH
