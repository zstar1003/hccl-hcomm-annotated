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
#include "aiv_temp_scatter_mesh_1D.h"
#include "executor_utils.h"

namespace Hccl {

AivTempScatterMesh1D::AivTempScatterMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AivAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

AivTempScatterMesh1D::~AivTempScatterMesh1D() {}

HcclResult AivTempScatterMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempScatterMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[AivTempScatterMesh1D] GenExtIns start");
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[AivTempScatterMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    std::vector<LinkData> allLinks;
    for (auto iter = tempLinks.begin(); iter != tempLinks.end(); ++iter) {
        allLinks.emplace_back(iter->second.at(0));
    }

    IncSliceId(); // 自动增长sliceId，传入aivTag

    AivOpArgs aivScatterArgs;
    aivScatterArgs.cmdType = HcclCMDType::HCCL_CMD_SCATTER;
    aivScatterArgs.input = templateDataParams.buffInfo.inBuffBaseOff;
    aivScatterArgs.output = templateDataParams.buffInfo.outBuffBaseOff;
    aivScatterArgs.rank = u32(myRank_);
    aivScatterArgs.rankSize = tempRankSize_;
    aivScatterArgs.count = templateDataParams.sliceSize / DataTypeSizeGet(dataType_);
    aivScatterArgs.dataType = dataType_;
    aivScatterArgs.op = reduceOp_;
    aivScatterArgs.root = root_;
    aivScatterArgs.aivTag = sliceId_; // 传入aivTag，Launch时重新组装为aivTag
    aivScatterArgs.isOpBase = (tempFuncs.opMode == OpMode::OPBASE);
    aivScatterArgs.xRankSize = tempVTopo_[0].size();
    aivScatterArgs.yRankSize = 0;
    aivScatterArgs.zRankSize = 0;
    for (u32 i = 0; i < tempVTopo_[0].size(); i++) {
        aivScatterArgs.topo_[i] = tempVTopo_[0][i];
    }
    if (tempVTopo_.size() > 1) {
        aivScatterArgs.yRankSize = tempVTopo_[1].size();
        for (u32 i = 0; i < tempVTopo_[1].size(); i++) {
            aivScatterArgs.topo_[TOPO_LEN_Y_OFFSET + i] = tempVTopo_[1][i];
        }
    }
    if (tempVTopo_.size() == MAX_DIM_NUM) {
        aivScatterArgs.zRankSize = tempVTopo_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < tempVTopo_[MAX_DIM_NUM - 1].size(); i++) {
            aivScatterArgs.topo_[TOPO_LEN_Z_OFFSET + i] = tempVTopo_[MAX_DIM_NUM - 1][i];
        }
    }

    aivScatterArgs.inputSliceStride = templateDataParams.inputSliceStride;
    aivScatterArgs.outputSliceStride = templateDataParams.outputSliceStride;
    aivScatterArgs.repeatNum = templateDataParams.repeatNum;
    aivScatterArgs.inputRepeatStride = templateDataParams.inputRepeatStride;
    aivScatterArgs.outputRepeatStride = templateDataParams.outputRepeatStride;

    std::unique_ptr<Instruction> aivInsScatterMesh1D = std::make_unique<AivInstruction>(allLinks, aivScatterArgs);

    tempInsQues[0]->Append(std::move(aivInsScatterMesh1D));

    HCCL_INFO("[AivTempScatterMesh1D] GenExtIns finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
