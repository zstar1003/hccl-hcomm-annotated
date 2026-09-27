/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_MESH_2D_H
#define INS_TEMP_REDUCE_SCATTER_MESH_2D_H

#include "string_util.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_alg_template_base.h"
#include "executor_utils.h"

#define DATASLICE_ONE 1

namespace Hccl {

constexpr u32 PARALLEL_SIZE = 2;
class InsTempReduceScatterMesh2D : public InsAlgTemplateBase {
public:
    explicit InsTempReduceScatterMesh2D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempReduceScatterMesh2D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of reduce scatter Mesh2D with tempRankSize [%u].", tempRankSize_);
    }
    u64 CalcScratchMultiple(const BufferType& inBuffType, const BufferType& outBuffType);
    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);

private:
    HcclResult CalcResLinksMesh2D(const u32 linkNumBtwPeers, AlgTempResReq& tempResReq);
    HcclResult PreCopy(const TemplateDataParams& tempAlgParams, std::vector<InsQuePtr>& tempInsQues);
    HcclResult SendRecvProcess(
        const ResLinks& tempLinks, std::vector<std::vector<DataSlice>> allSliceVec, std::vector<InsQuePtr>& tempInsQues,
        u32 remoteRank, u32 queIdx) const;
    HcclResult RunFirstLevel(
        const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams);
    HcclResult RunFirstReduce(std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams);
    HcclResult RunSecondLevel(
        const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams);
    HcclResult RunSecondReduce(std::vector<InsQuePtr>& tempInsQues, const TemplateDataParams& tempAlgParams);
    RankId GetRankFromMap(const u32 rankIdx);
    u32 xQueNum_ = 0;
    u32 yQueNum_ = 0;
    u32 xRankSize_ = 0;
    u32 yRankSize_ = 0;
    u32 xRankId_ = 0;
    u32 yRankId_ = 0;
    u64 halfDataSize_ = 0;
};

} // namespace Hccl

#endif // INS_TEMP_REDUCE_SCATTER_MESH_2D_H
