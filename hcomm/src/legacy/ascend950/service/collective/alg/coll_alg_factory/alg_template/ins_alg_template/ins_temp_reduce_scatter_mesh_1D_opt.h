/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPEN_HCCL_INS_TEMP_REDUCE_SCATTER_MESH_H
#define OPEN_HCCL_INS_TEMP_REDUCE_SCATTER_MESH_H

#include "string_util.h"
#include "ins_alg_template_base.h"
#include "executor_utils.h"
namespace Hccl {

class InsTempReduceScatterMesh1DOpt : public InsAlgTemplateBase {
public:
    explicit InsTempReduceScatterMesh1DOpt(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempReduceScatterMesh1DOpt() override;

    std::string Describe() const override
    {
        return StringFormat("Template of reduce scatter NHR with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    HcclResult PostCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues);
    u64 CalcScratchMultiple(const BufferType& inBuffType, const BufferType& outBuffType) const;

private:
    HcclResult RunReduceScatter(
        const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams);
    RankId GetRankFromMap(const u32 rankIdx);
    u64 processSize_{0};
};

} // namespace Hccl

#endif // OPEN_HCCL_INS_TEMP_REDUCE_SCATTER_MESH_H
