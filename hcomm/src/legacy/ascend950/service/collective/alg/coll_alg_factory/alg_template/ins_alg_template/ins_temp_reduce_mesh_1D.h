/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_MESH1D
#define INS_TEMP_REDUCE_MESH1D

#include "string_util.h"
#include "executor_utils.h"
#include "ins_alg_template_base.h"

namespace Hccl {

class InsTempReduceMesh1D : public InsAlgTemplateBase {
public:
    explicit InsTempReduceMesh1D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempReduceMesh1D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of reduce Mesh1D with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const;

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& dataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);

private:
    HcclResult
    RunReduce(const TemplateDataParams& dataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues);
    HcclResult
    SendData(const TemplateDataParams& dataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues);
    HcclResult
    GatherData(const TemplateDataParams& dataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues);
    HcclResult ReduceData(const TemplateDataParams& dataParams, std::vector<InsQuePtr>& tempInsQues);

    u32 myIdx_ = INVALID_U32; // 本rank在通信域内的索引
};

} // namespace Hccl

#endif // INS_TEMP_REDUCE_MESH1D
