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
#include "aiv_temp_all_to_all_mesh_1D.h"
#include "executor_utils.h"

namespace Hccl {

AivTempAlltoAllMesh1D::AivTempAlltoAllMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AivAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

AivTempAlltoAllMesh1D::~AivTempAlltoAllMesh1D() {}

u32 AivTempAlltoAllMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // 单算子和图模式一致，AlltoAll的usrIn、scratchBuffer，usrOut大小一致
    return 1;
}

HcclResult AivTempAlltoAllMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;
    HCCL_INFO("[AivTempAlltoAllMesh1D] Limit core num[%u]", numBlocksLimit);

    // 小于1的场景
    if (numBlocksLimit < 1) {
        numBlocks = numBlocksLimit;
        return HcclResult::HCCL_SUCCESS;
    }

    if (numBlocksLimit >= tempRankSize_) {
        numBlocks = numBlocksLimit / tempRankSize_ * tempRankSize_;
    } else {
        u32 rankPerCore = (tempRankSize_ + numBlocksLimit - 1) / numBlocksLimit; // 向上取整
        numBlocks = (tempRankSize_ + rankPerCore - 1) / rankPerCore;             // 向上取整
    }

    HCCL_INFO("[AivTempAlltoAllMesh1D] Actually use core num[%u]", numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[AivTempAlltoAllMesh1D] Calculate resource, stream number is[%u],", tempResReq.streamNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[AivTempAlltoAllMesh1D] Run algorithm start: rank[%d]", myRank_);
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[AivTempAlltoAllMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    std::vector<LinkData> allLinks;
    for (auto iter = tempLinks.begin(); iter != tempLinks.end(); ++iter) {
        allLinks.emplace_back(iter->second.at(0));
    }

    IncSliceId(); // 自动增长sliceId，传入aivTag

    AivOpArgs aivAlltoAllArgs;
    aivAlltoAllArgs.cmdType = HcclCMDType::HCCL_CMD_ALLTOALL;
    aivAlltoAllArgs.input = templateDataParams.buffInfo.inBuffBaseOff;
    aivAlltoAllArgs.output = templateDataParams.buffInfo.outBuffBaseOff;
    aivAlltoAllArgs.rank = u32(myRank_);
    aivAlltoAllArgs.rankSize = tempRankSize_;
    aivAlltoAllArgs.count = templateDataParams.sliceSize / DataTypeSizeGet(dataType_);
    aivAlltoAllArgs.dataType = dataType_;
    aivAlltoAllArgs.op = reduceOp_;
    aivAlltoAllArgs.root = root_;
    aivAlltoAllArgs.aivTag = sliceId_; // 传入aivTag，Launch时重新组装为aivTag
    aivAlltoAllArgs.isOpBase = (tempFuncs.opMode == OpMode::OPBASE);
    aivAlltoAllArgs.xRankSize = tempVTopo_[0].size();
    aivAlltoAllArgs.yRankSize = 0;
    aivAlltoAllArgs.zRankSize = 0;
    u64 dataSize = op_.dataCount * DataTypeSizeGet(dataType_);
    CHK_RET(CalNumBlocks(aivAlltoAllArgs.numBlocks, dataSize, op_.numBlocksLimit));
    for (u32 i = 0; i < tempVTopo_[0].size(); i++) {
        aivAlltoAllArgs.topo_[i] = tempVTopo_[0][i];
    }
    if (tempVTopo_.size() > 1) {
        aivAlltoAllArgs.yRankSize = tempVTopo_[1].size();
        for (u32 i = 0; i < tempVTopo_[1].size(); i++) {
            aivAlltoAllArgs.topo_[TOPO_LEN_Y_OFFSET + i] = tempVTopo_[1][i];
        }
    }
    if (tempVTopo_.size() == MAX_DIM_NUM) {
        aivAlltoAllArgs.zRankSize = tempVTopo_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < tempVTopo_[MAX_DIM_NUM - 1].size(); i++) {
            aivAlltoAllArgs.topo_[TOPO_LEN_Z_OFFSET + i] = tempVTopo_[MAX_DIM_NUM - 1][i];
        }
    }

    aivAlltoAllArgs.inputSliceStride = templateDataParams.inputSliceStride;
    aivAlltoAllArgs.outputSliceStride = templateDataParams.outputSliceStride;
    aivAlltoAllArgs.repeatNum = templateDataParams.repeatNum;
    aivAlltoAllArgs.inputRepeatStride = templateDataParams.inputRepeatStride;
    aivAlltoAllArgs.outputRepeatStride = templateDataParams.outputRepeatStride;

    std::unique_ptr<Instruction> aivInsAlltoAllMesh1D = std::make_unique<AivInstruction>(allLinks, aivAlltoAllArgs);

    tempInsQues[0]->Append(std::move(aivInsAlltoAllMesh1D));

    HCCL_INFO("[AivTempAlltoAllMesh1D] Run algorithm end: rank[%d]", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
