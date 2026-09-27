/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMP_ALL_SCATTER_MESH_1D
#define AIV_TEMP_ALL_SCATTER_MESH_1D

#include "string_util.h"
#include "executor_utils.h"

#include "aiv_alg_template_base.h"

namespace Hccl {

class AivTempAllReduceMesh1DTwoShot : public AivAlgTemplateBase {
public:
    explicit AivTempAllReduceMesh1DTwoShot(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~AivTempAllReduceMesh1DTwoShot() override;

    std::string Describe() const override
    {
        return StringFormat("Instruction based Template of allreduce mesh 1D with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues) override;
    HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit) override;
};

} // namespace Hccl

#endif // AIV_TEMP_ALL_REDUCE_MESH_1D
