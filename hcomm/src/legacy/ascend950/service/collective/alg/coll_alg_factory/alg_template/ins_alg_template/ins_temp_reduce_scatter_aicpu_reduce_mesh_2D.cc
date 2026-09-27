/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"

#include "alg_data_trans_wrapper.h"
#include "ins_temp_reduce_scatter_aicpu_reduce_mesh_2D.h"

namespace Hccl {
InsTempReduceScatterAicpuReduceMesh2D::InsTempReduceScatterAicpuReduceMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap),
      alltoallMesh2D_(myRank_, tempRankSize_, tempVTopo_, tempVirtRankMap_)
{}

InsTempReduceScatterAicpuReduceMesh2D::~InsTempReduceScatterAicpuReduceMesh2D() {}

HcclResult InsTempReduceScatterAicpuReduceMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    CHK_RET(alltoallMesh2D_.CalcRes(tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempReduceScatterAicpuReduceMesh2D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    const u32 executor2DTemp = 4;
    return tempRankSize_ * executor2DTemp;
}

HcclResult InsTempReduceScatterAicpuReduceMesh2D::RunAicpuLocalReduce(
    const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues)
{
    u64 baseOffset = templateDataParams.sliceSize * tempRankSize_;
    DataSlice dataSlice = DataSlice(BufferType::SCRATCH, baseOffset, templateDataParams.sliceSize);
    for (u32 rankId = 1; rankId < tempRankSize_; rankId++) {
        DataSlice addSlice = DataSlice(
            BufferType::SCRATCH, baseOffset + templateDataParams.sliceSize * rankId, templateDataParams.sliceSize);
        AicpuReduce(tempInsQues[0], addSlice, dataSlice, dataType_, redOp_);
    }
    DataSlice outputSlice
        = DataSlice(BufferType::OUTPUT, templateDataParams.buffInfo.inBuffBaseOff, templateDataParams.sliceSize);
    LocalCopy(tempInsQues[0], dataSlice, outputSlice);
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceMesh2D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempReduceScatterAicpuReduceMesh2D] Run start");
    if (tempVTopo_[0].size() == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    opMode_ = tempFuncs.opMode;
    queNum_ = tempVTopo_[0].size() + tempVTopo_[1].size();
    CHK_PRT_RET(
        queNum_ > tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempReduceScatterAicpuReduceMesh2D] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    TempFuncs alltoallFuncs = tempFuncs;
    TemplateDataParams alltoallParams = templateDataParams;
    alltoallFuncs.isBottom = false;
    alltoallParams.buffInfo.outBuffBaseOff = templateDataParams.sliceSize * tempRankSize_;
    alltoallParams.outputSliceStride = templateDataParams.sliceSize;
    alltoallMesh2D_.SetDataType(dataType_);
    alltoallMesh2D_.GenExtIns(alltoallFuncs, alltoallParams, tempLinks, tempInsQues);
    StreamSync(tempInsQues);
    RunAicpuLocalReduce(templateDataParams, tempInsQues);
    HCCL_INFO("[InsTempReduceScatterAicpuReduceMesh2D] Run finished");
    return HCCL_SUCCESS;
}

} // namespace Hccl
