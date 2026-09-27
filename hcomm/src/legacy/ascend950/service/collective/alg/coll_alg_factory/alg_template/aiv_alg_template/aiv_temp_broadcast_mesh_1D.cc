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
#include "aiv_temp_broadcast_mesh_1D.h"

namespace Hccl {

AivTempBroadcastMesh1D::AivTempBroadcastMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AivAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

AivTempBroadcastMesh1D::~AivTempBroadcastMesh1D() {}

HcclResult AivTempBroadcastMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempBroadcastMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[AivTempBroadcastMesh1D] GenExtIns start");
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[AivTempBroadcastMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    std::vector<LinkData> allLinks;
    for (auto iter = tempLinks.begin(); iter != tempLinks.end(); ++iter) {
        allLinks.emplace_back(iter->second.at(0));
    }

    IncSliceId(); // 自动增长sliceId，传入aivTag

    AivOpArgs aivBroadcastArgs;
    aivBroadcastArgs.cmdType = HcclCMDType::HCCL_CMD_BROADCAST;
    aivBroadcastArgs.input = templateDataParams.buffInfo.inBuffBaseOff;
    aivBroadcastArgs.output = templateDataParams.buffInfo.outBuffBaseOff;
    aivBroadcastArgs.rank = myRank_;
    aivBroadcastArgs.rankSize = tempRankSize_;
    aivBroadcastArgs.count = templateDataParams.sliceSize / DataTypeSizeGet(dataType_);
    aivBroadcastArgs.dataType = dataType_;
    aivBroadcastArgs.root = root_;
    aivBroadcastArgs.aivTag = sliceId_; // 传入aivTag，Launch时重新组装为aivTag
    aivBroadcastArgs.isOpBase = (tempFuncs.opMode == OpMode::OPBASE);
    aivBroadcastArgs.xRankSize = tempVTopo_[0].size();
    aivBroadcastArgs.yRankSize = 0;
    aivBroadcastArgs.zRankSize = 0;
    for (u32 i = 0; i < tempVTopo_[0].size(); i++) {
        aivBroadcastArgs.topo_[i] = tempVTopo_[0][i];
    }
    if (tempVTopo_.size() > 1) {
        aivBroadcastArgs.yRankSize = tempVTopo_[1].size();
        for (u32 i = 0; i < tempVTopo_[1].size(); i++) {
            aivBroadcastArgs.topo_[TOPO_LEN_Y_OFFSET + i] = tempVTopo_[1][i];
        }
    }
    if (tempVTopo_.size() > MAX_DIM_NUM) {
        aivBroadcastArgs.zRankSize = tempVTopo_[MAX_DIM_NUM].size();
        for (u32 i = 0; i < tempVTopo_[MAX_DIM_NUM].size(); i++) {
            aivBroadcastArgs.topo_[TOPO_LEN_Z_OFFSET + i] = tempVTopo_[MAX_DIM_NUM][i];
        }
    }
    aivBroadcastArgs.inputSliceStride = templateDataParams.inputSliceStride;
    aivBroadcastArgs.outputSliceStride = templateDataParams.outputSliceStride;
    aivBroadcastArgs.repeatNum = templateDataParams.repeatNum;
    aivBroadcastArgs.inputRepeatStride = templateDataParams.inputRepeatStride;
    aivBroadcastArgs.outputRepeatStride = templateDataParams.outputRepeatStride;

    std::unique_ptr<Instruction> aivInsBroadcastMesh1D = std::make_unique<AivInstruction>(allLinks, aivBroadcastArgs);

    tempInsQues[0]->Append(std::move(aivInsBroadcastMesh1D));

    HCCL_INFO("[AivTempBroadcastMesh1D] GenExtIns finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
