/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_REDUCE_SCATTER_AICPU_REDUCE
#define HCCLV2_INS_TEMP_REDUCE_SCATTER_AICPU_REDUCE

#include "string_util.h"

#include "ins_alg_template_base.h"
#include "executor_utils.h"

namespace Hccl {

class InsTempReduceScatterAicpuReduce : public InsAlgTemplateBase {
public:
    explicit InsTempReduceScatterAicpuReduce(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempReduceScatterAicpuReduce() override;

    std::string Describe() const override
    {
        return StringFormat("Template of aicpu reducescatter with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;

    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const;

private:
    HcclResult RunAlltoAllMesh(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult RunAicpuLocalReduce(const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues);
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_REDUCE_SCATTER_AICPU_REDUCE
