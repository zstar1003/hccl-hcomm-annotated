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
#include "aiv_temp_reduce_scatter_mesh_1D.h"
#include "executor_utils.h"

namespace Hccl {

AivTempReduceScatterMesh1D::AivTempReduceScatterMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AivAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

AivTempReduceScatterMesh1D::~AivTempReduceScatterMesh1D() {}

u32 AivTempReduceScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // 小数据量下，这里是2*rankSize基本不影响，大数据量下，需要2倍的cclBuffer去并发读
    return 2 * tempRankSize_;
}

HcclResult AivTempReduceScatterMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempReduceScatterMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    numBlocks = numBlocksLimit;
    if (dataSize < REDUCE_SCATTER_SMALL_COUNT_512KB) { // 小数据量，走原来极致低时延的流程
        constexpr uint32_t stepNum = 2;
        if (numBlocks > stepNum * tempRankSize_) {
            numBlocks = stepNum * tempRankSize_;
        }
    }
    HCCL_INFO("[AivTempReduceScatterMesh1D] Actually use core num[%u]", numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempReduceScatterMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[AivTempReduceScatterMesh1D] GenExtIns start");
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[AivTempReduceScatterMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    std::vector<LinkData> allLinks;
    for (auto iter = tempLinks.begin(); iter != tempLinks.end(); ++iter) {
        allLinks.emplace_back(iter->second.at(0));
    }

    IncSliceId(); // 自动增长sliceId，传入aivTag

    AivOpArgs aivReduceScatterArgs;
    aivReduceScatterArgs.cmdType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    aivReduceScatterArgs.input = templateDataParams.buffInfo.inBuffBaseOff;
    aivReduceScatterArgs.output = templateDataParams.buffInfo.outBuffBaseOff;
    aivReduceScatterArgs.rank = u32(myRank_);
    aivReduceScatterArgs.rankSize = tempRankSize_;
    aivReduceScatterArgs.count = templateDataParams.sliceSize / DataTypeSizeGet(dataType_);
    aivReduceScatterArgs.dataType = dataType_;
    aivReduceScatterArgs.op = reduceOp_;
    aivReduceScatterArgs.root = root_;
    aivReduceScatterArgs.aivTag = sliceId_; // 传入aivTag，Launch时重新组装为aivTag
    aivReduceScatterArgs.isOpBase = (tempFuncs.opMode == OpMode::OPBASE);
    aivReduceScatterArgs.xRankSize = tempVTopo_[0].size();
    aivReduceScatterArgs.yRankSize = 0;
    aivReduceScatterArgs.zRankSize = 0;
    for (u32 i = 0; i < tempVTopo_[0].size(); i++) {
        aivReduceScatterArgs.topo_[i] = tempVTopo_[0][i];
    }
    if (tempVTopo_.size() > 1) {
        aivReduceScatterArgs.yRankSize = tempVTopo_[1].size();
        for (u32 i = 0; i < tempVTopo_[1].size(); i++) {
            aivReduceScatterArgs.topo_[TOPO_LEN_Y_OFFSET + i] = tempVTopo_[1][i];
        }
    }
    if (tempVTopo_.size() == MAX_DIM_NUM) {
        aivReduceScatterArgs.zRankSize = tempVTopo_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < tempVTopo_[MAX_DIM_NUM - 1].size(); i++) {
            aivReduceScatterArgs.topo_[TOPO_LEN_Z_OFFSET + i] = tempVTopo_[MAX_DIM_NUM - 1][i];
        }
    }

    u64 dataSize = op_.dataCount * DataTypeSizeGet(dataType_);
    CHK_RET(CalNumBlocks(aivReduceScatterArgs.numBlocks, dataSize, op_.numBlocksLimit));

    aivReduceScatterArgs.inputSliceStride = templateDataParams.inputSliceStride;
    aivReduceScatterArgs.outputSliceStride = templateDataParams.outputSliceStride;
    aivReduceScatterArgs.repeatNum = templateDataParams.repeatNum;
    aivReduceScatterArgs.inputRepeatStride = templateDataParams.inputRepeatStride;
    aivReduceScatterArgs.outputRepeatStride = templateDataParams.outputRepeatStride;

    std::unique_ptr<Instruction> aivInsReduceScatterMesh1D
        = std::make_unique<AivInstruction>(allLinks, aivReduceScatterArgs);

    tempInsQues[0]->Append(std::move(aivInsReduceScatterMesh1D));

    HCCL_INFO("[AivTempReduceScatterMesh1D] GenExtIns finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
