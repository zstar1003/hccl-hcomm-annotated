/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor_utils.h"
#include "hccl_aiv_utils.h"
#include "aiv_ins.h"
#include "aiv_temp_reduce_mesh_1D.h"

namespace Hccl {

AivTempReduceMesh1D::AivTempReduceMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AivAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

AivTempReduceMesh1D::~AivTempReduceMesh1D() {}

HcclResult AivTempReduceMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    HCCL_INFO("[AivTempReduceMesh1D] Calculate communication resources start");
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    HCCL_INFO(
        "[AivTempReduceMesh1D] Calculate communication resources finished, queNum[%u], streamNum[%u], linkNum[%u]",
        tempResReq.queNum, tempResReq.streamNum, tempResReq.links.size());
    return HcclResult::HCCL_SUCCESS;
}

u32 AivTempReduceMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    HCCL_INFO("[AivTempReduceMesh1D] Scratch multiple is [%u]", (tempRankSize_ - 1));
    return tempRankSize_ - 1;
}

HcclResult AivTempReduceMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[AivTempReduceMesh1D] GenExtIns start rank[%u]", u32(myRank_));
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[AivTempReduceMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    std::vector<LinkData> allLinks;
    for (auto iter = tempLinks.begin(); iter != tempLinks.end(); ++iter) {
        allLinks.emplace_back(iter->second.at(0));
    }

    IncSliceId(); // 自动增长sliceId，传入aivTag

    AivOpArgs aivReduceArgs;
    aivReduceArgs.cmdType = HcclCMDType::HCCL_CMD_REDUCE;
    aivReduceArgs.input = templateDataParams.buffInfo.inBuffBaseOff;
    aivReduceArgs.output = templateDataParams.buffInfo.outBuffBaseOff;
    aivReduceArgs.rank = u32(myRank_);
    aivReduceArgs.rankSize = tempRankSize_;
    aivReduceArgs.count = templateDataParams.sliceSize / DataTypeSizeGet(dataType_);
    aivReduceArgs.dataType = dataType_;
    aivReduceArgs.op = reduceOp_;
    aivReduceArgs.root = root_;
    aivReduceArgs.aivTag = sliceId_; // 传入aivTag，Launch时重新组装为aivTag
    aivReduceArgs.isOpBase = (tempFuncs.opMode == OpMode::OPBASE);
    aivReduceArgs.xRankSize = tempVTopo_[0].size();
    aivReduceArgs.yRankSize = 0;
    aivReduceArgs.zRankSize = 0;

    constexpr u32 level0 = 0;
    constexpr u32 level1 = 1;
    constexpr u32 level2 = 2;
    for (u32 i = 0; i < tempVTopo_[level0].size(); i++) {
        aivReduceArgs.topo_[i] = tempVTopo_[level0][i];
    }
    if (tempVTopo_.size() > level1) {
        aivReduceArgs.yRankSize = tempVTopo_[level1].size();
        for (u32 i = 0; i < tempVTopo_[level1].size(); i++) {
            aivReduceArgs.topo_[TOPO_LEN_Y_OFFSET + i] = tempVTopo_[level1][i];
        }
    }
    if (tempVTopo_.size() > level2) {
        aivReduceArgs.zRankSize = tempVTopo_[level2].size();
        for (u32 i = 0; i < tempVTopo_[level2].size(); i++) {
            aivReduceArgs.topo_[TOPO_LEN_Z_OFFSET + i] = tempVTopo_[level2][i];
        }
    }

    aivReduceArgs.inputSliceStride = templateDataParams.inputSliceStride;
    aivReduceArgs.outputSliceStride = templateDataParams.outputSliceStride;
    aivReduceArgs.repeatNum = templateDataParams.repeatNum;
    aivReduceArgs.inputRepeatStride = templateDataParams.inputRepeatStride;
    aivReduceArgs.outputRepeatStride = templateDataParams.outputRepeatStride;

    std::unique_ptr<Instruction> aivInsReduceMesh1D = std::make_unique<AivInstruction>(allLinks, aivReduceArgs);

    tempInsQues[0]->Append(std::move(aivInsReduceMesh1D));
    HCCL_INFO("[AivTempReduceMesh1D] GenExtIns finished rank[%u]", u32(myRank_));
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
