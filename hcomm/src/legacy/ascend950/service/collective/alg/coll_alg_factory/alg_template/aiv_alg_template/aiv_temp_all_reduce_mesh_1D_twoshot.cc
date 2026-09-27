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
#include "aiv_temp_all_reduce_mesh_1D_twoshot.h"

namespace Hccl {

AivTempAllReduceMesh1DTwoShot::AivTempAllReduceMesh1DTwoShot(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AivAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    HCCL_INFO("[AivTempAllReduceMesh1DTwoShot] Init.");
}

AivTempAllReduceMesh1DTwoShot::~AivTempAllReduceMesh1DTwoShot() { HCCL_INFO("[AivTempAllReduceMesh1DTwoShot] exit."); }

HcclResult AivTempAllReduceMesh1DTwoShot::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

u32 AivTempAllReduceMesh1DTwoShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u32 multiplier = 4;
    return multiplier;
}

HcclResult AivTempAllReduceMesh1DTwoShot::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;

    if (numBlocksLimit >= (tempRankSize_ + 1)) {
        u32 coreNumPerRank = numBlocksLimit / (tempRankSize_ + 1);
        numBlocks = coreNumPerRank * (tempRankSize_ + 1);
    } else {
        // 如果要用更少的核心可以在这里折算，比如rankSize/2个核心
        numBlocks = numBlocksLimit;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAllReduceMesh1DTwoShot::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[AivTempAllReduceMesh1DTwoShot] start.");
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[AivTempAllReduceMesh1DTwoShot] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    std::vector<LinkData> allLinks;
    for (auto iter = tempLinks.begin(); iter != tempLinks.end(); ++iter) {
        allLinks.emplace_back(iter->second.at(0));
    }
    IncSliceId(); // 自动增长sliceId，传入aivTag

    AivOpArgs aivAllreduceArgs;
    aivAllreduceArgs.cmdType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    aivAllreduceArgs.argsType = KernelArgsType::ARGS_TYPE_TWO_SHOT;
    aivAllreduceArgs.input = templateDataParams.buffInfo.inBuffBaseOff;
    aivAllreduceArgs.output = templateDataParams.buffInfo.outBuffBaseOff;
    aivAllreduceArgs.rank = myRank_;
    aivAllreduceArgs.rankSize = tempRankSize_;
    aivAllreduceArgs.count = templateDataParams.sliceSize / DataTypeSizeGet(dataType_);
    aivAllreduceArgs.dataType = dataType_;
    aivAllreduceArgs.op = reduceOp_;
    aivAllreduceArgs.root = root_;
    aivAllreduceArgs.aivTag = sliceId_; // 传入aivTag，Launch时重新组装为aivTag
    aivAllreduceArgs.isOpBase = (tempFuncs.opMode == OpMode::OPBASE);
    aivAllreduceArgs.xRankSize = tempVTopo_[0].size();
    CalNumBlocks(aivAllreduceArgs.numBlocks, templateDataParams.sliceSize, op_.numBlocksLimit);
    HCCL_INFO("[AivTempAllReduceMesh1DTwoShot] Actually use core num[%u]", aivAllreduceArgs.numBlocks);

    for (u32 i = 0; i < tempVTopo_[0].size(); i++) {
        aivAllreduceArgs.topo_[i] = tempVTopo_[0][i];
    }

    u32 sizeOne = 1, sizeTwo = 1;
    if (tempVTopo_.size() > sizeOne) {
        aivAllreduceArgs.yRankSize = tempVTopo_[1].size();
        for (u32 i = 0; i < tempVTopo_[1].size(); i++) {
            aivAllreduceArgs.topo_[TOPO_LEN_Y_OFFSET + i] = tempVTopo_[1][i];
        }
    }
    if (tempVTopo_.size() > sizeTwo) {
        aivAllreduceArgs.zRankSize = tempVTopo_[2].size();
        for (u32 i = 0; i < tempVTopo_[2].size(); i++) {
            aivAllreduceArgs.topo_[TOPO_LEN_Z_OFFSET + i] = tempVTopo_[2][i];
        }
    }

    aivAllreduceArgs.inputSliceStride = templateDataParams.inputSliceStride;
    aivAllreduceArgs.outputSliceStride = templateDataParams.outputSliceStride;
    aivAllreduceArgs.repeatNum = templateDataParams.repeatNum;
    aivAllreduceArgs.inputRepeatStride = templateDataParams.inputRepeatStride;
    aivAllreduceArgs.outputRepeatStride = templateDataParams.outputRepeatStride;
    std::unique_ptr<Instruction> aivInsAllreduceMesh1D = std::make_unique<AivInstruction>(allLinks, aivAllreduceArgs);
    tempInsQues[0]->Append(std::move(aivInsAllreduceMesh1D));
    HCCL_INFO("[AivTempAllReduceMesh1DTwoShot] GenExtIns finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
