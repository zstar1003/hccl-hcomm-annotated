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
#include "ins_temp_all_reduce_aicpu_reduce_mesh_2D.h"
#include "ins_temp_all_gather_mesh_2D.h"

namespace Hccl {
InsTempAllReduceAicpuReduceMesh2D::InsTempAllReduceAicpuReduceMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempAllReduceAicpuReduceMesh2D::~InsTempAllReduceAicpuReduceMesh2D() {}

HcclResult InsTempAllReduceAicpuReduceMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    const int TwoD = 2;
    CHK_PRT_RET(
        tempVTopo_.size() < TwoD,
        HCCL_ERROR("[InsTempAllReduceAicpuReduceMesh2D] tempVTopo_ mismatch size:%zu", tempVTopo_.size()),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        tempVTopo_[0].size() <= 1 || tempVTopo_[1].size() <= 1,
        HCCL_ERROR(
            "[InsTempAllReduceAicpuReduceMesh2D] tempVTopo_ size error, size:%zu %zu", tempVTopo_[0].size(),
            tempVTopo_[1].size()),
        HcclResult::HCCL_E_INTERNAL);
    tempResReq.queNum = tempVTopo_[0].size() - 1 + tempVTopo_[1].size() - 1;

    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    HCCL_DEBUG("InsTempAllReduceAicpuReduceMesh2D::CalcRes queNotifys size[%zu]", tempResReq.queNotifys.size());

    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    CHK_RET(CalcResLinksMesh2D(myRank_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempAllReduceAicpuReduceMesh2D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}

HcclResult InsTempAllReduceAicpuReduceMesh2D::RunAicpuLocalReduce(
    const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues)
{
    DataSlice dataSlice = DataSlice(BufferType::SCRATCH, 0, templateDataParams.sliceSize);
    for (u32 rankId = 1; rankId < tempRankSize_; rankId++) {
        DataSlice addSlice
            = DataSlice(BufferType::SCRATCH, templateDataParams.sliceSize * rankId, templateDataParams.sliceSize);
        AicpuReduce(tempInsQues[0], addSlice, dataSlice, dataType_, redOp_);
    }
    DataSlice outputSlice
        = DataSlice(BufferType::OUTPUT, templateDataParams.buffInfo.inBuffBaseOff, templateDataParams.sliceSize);
    LocalCopy(tempInsQues[0], dataSlice, outputSlice);
    return HCCL_SUCCESS;
}

HcclResult InsTempAllReduceAicpuReduceMesh2D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempAllReduceAicpuReduceMesh2D] Run start");
    if (tempVTopo_[0].size() == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    opMode_ = tempFuncs.opMode;
    queNum_ = tempVTopo_[0].size() - 1 + tempVTopo_[1].size() - 1;
    CHK_PRT_RET(
        queNum_ > tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempAllReduceAicpuReduceMesh2D] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    InsTempAllGatherMesh2D allgatherMesh2D(myRank_, tempRankSize_, tempVTopo_, tempVirtRankMap_);
    TempFuncs allgatherFuncs = tempFuncs;
    TemplateDataParams allgatherParams = templateDataParams;
    allgatherFuncs.isBottom = false;
    allgatherParams.buffInfo.outBuffBaseOff = 0;
    allgatherParams.outputSliceStride = templateDataParams.sliceSize;
    allgatherMesh2D.SetDataType(dataType_);
    allgatherMesh2D.GenExtIns(allgatherFuncs, allgatherParams, tempLinks, tempInsQues);
    StreamSync(tempInsQues);
    RunAicpuLocalReduce(templateDataParams, tempInsQues);
    HCCL_INFO("[InsTempAllReduceAicpuReduceMesh2D] Run finished");
    return HCCL_SUCCESS;
}

} // namespace Hccl
