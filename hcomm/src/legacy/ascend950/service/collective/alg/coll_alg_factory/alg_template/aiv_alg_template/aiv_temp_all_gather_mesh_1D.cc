/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_aiv_utils.h"
#include "aiv_ins.h"
#include "aiv_temp_all_gather_mesh_1D.h"
#include "executor_utils.h"

namespace Hccl {

AivTempAllGatherMesh1D::AivTempAllGatherMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AivAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

AivTempAllGatherMesh1D::~AivTempAllGatherMesh1D() {}

HcclResult AivTempAllGatherMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    HCCL_INFO("[AivTempAllGatherMesh1D] Calculate communication resources start");
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAllGatherMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;
    numBlocks = numBlocksLimit;
    HCCL_INFO("[AivTempAllGatherMesh1D] Actually use core num[%u]", numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAllGatherMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[AivTempAllGatherMesh1D] GenExtIns start");
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[AivTempAllGatherMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    std::vector<LinkData> allLinks;
    for (auto iter = tempLinks.begin(); iter != tempLinks.end(); ++iter) {
        allLinks.emplace_back(iter->second.at(0));
    }

    IncSliceId(); // 自动增长sliceId，传入aivTag

    AivOpArgs aivAllGatherArgs;
    aivAllGatherArgs.cmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    aivAllGatherArgs.input = templateDataParams.buffInfo.inBuffBaseOff;
    aivAllGatherArgs.output = templateDataParams.buffInfo.outBuffBaseOff;
    aivAllGatherArgs.rank = u32(myRank_);
    aivAllGatherArgs.rankSize = tempRankSize_;
    aivAllGatherArgs.count = templateDataParams.sliceSize / DataTypeSizeGet(dataType_);
    aivAllGatherArgs.dataType = dataType_;
    aivAllGatherArgs.op = reduceOp_;
    aivAllGatherArgs.root = root_;
    aivAllGatherArgs.aivTag = sliceId_; // 传入aivTag，Launch时重新组装为aivTag
    aivAllGatherArgs.isOpBase = (tempFuncs.opMode == OpMode::OPBASE);
    aivAllGatherArgs.xRankSize = tempVTopo_[0].size();
    aivAllGatherArgs.yRankSize = 0;
    aivAllGatherArgs.zRankSize = 0;
    u64 dataSize = op_.dataCount * DataTypeSizeGet(dataType_);
    CHK_RET(CalNumBlocks(aivAllGatherArgs.numBlocks, dataSize, op_.numBlocksLimit));
    for (u32 i = 0; i < tempVTopo_[0].size(); i++) {
        aivAllGatherArgs.topo_[i] = tempVTopo_[0][i];
    }
    if (tempVTopo_.size() > 1) {
        aivAllGatherArgs.yRankSize = tempVTopo_[1].size();
        for (u32 i = 0; i < tempVTopo_[1].size(); i++) {
            aivAllGatherArgs.topo_[TOPO_LEN_Y_OFFSET + i] = tempVTopo_[1][i];
        }
    }
    if (tempVTopo_.size() == MAX_DIM_NUM) {
        aivAllGatherArgs.zRankSize = tempVTopo_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < tempVTopo_[MAX_DIM_NUM - 1].size(); i++) {
            aivAllGatherArgs.topo_[TOPO_LEN_Z_OFFSET + i] = tempVTopo_[MAX_DIM_NUM - 1][i];
        }
    }

    aivAllGatherArgs.inputSliceStride = templateDataParams.inputSliceStride;
    aivAllGatherArgs.outputSliceStride = templateDataParams.outputSliceStride;
    aivAllGatherArgs.repeatNum = templateDataParams.repeatNum;
    aivAllGatherArgs.inputRepeatStride = templateDataParams.inputRepeatStride;
    aivAllGatherArgs.outputRepeatStride = templateDataParams.outputRepeatStride;

    std::unique_ptr<Instruction> aivInsAllGatherMesh1D = std::make_unique<AivInstruction>(allLinks, aivAllGatherArgs);

    tempInsQues[0]->Append(std::move(aivInsAllGatherMesh1D));

    HCCL_INFO("[AivTempAllGatherMesh1D] GenExtIns finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
