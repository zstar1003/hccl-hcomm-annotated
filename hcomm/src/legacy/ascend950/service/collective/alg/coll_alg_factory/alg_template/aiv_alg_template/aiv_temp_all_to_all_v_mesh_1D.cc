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
#include "aiv_temp_all_to_all_v_mesh_1D.h"
#include "executor_utils.h"

namespace Hccl {
constexpr u64 MAX_NUM_BLOCKS_ALL_TO_ALL_V = 48; // 算法不交付控核

AivTempAlltoAllVMesh1D::AivTempAlltoAllVMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : AivAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

AivTempAlltoAllVMesh1D::~AivTempAlltoAllVMesh1D() {}

u32 AivTempAlltoAllVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // 单算子和图模式一致，AlltoAllV的usrIn、scratchBuffer，usrOut大小一致
    return 1;
}

HcclResult AivTempAlltoAllVMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[AivTempAlltoAllVMesh1D] Calculate resource, stream number is[%u],", tempResReq.streamNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllVMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[AivTempAlltoAllVMesh1D] Run algorithm start: rank[%d]", myRank_);
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[AivTempAlltoAllVMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    std::vector<LinkData> allLinks;
    for (auto iter = tempLinks.begin(); iter != tempLinks.end(); ++iter) {
        allLinks.emplace_back(iter->second.at(0));
    }

    IncSliceId(); // 自动增长sliceId，传入aivTag

    AivOpArgs aivAlltoAllVArgs;
    aivAlltoAllVArgs.cmdType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    aivAlltoAllVArgs.input = 0; // ins_rules.cc里面，这里会和起始地址累加起来作为input
    aivAlltoAllVArgs.output = 0;
    aivAlltoAllVArgs.rank = u32(myRank_);
    aivAlltoAllVArgs.rankSize = tempRankSize_;
    aivAlltoAllVArgs.count = templateDataParams.sliceSize / DataTypeSizeGet(dataType_);
    aivAlltoAllVArgs.dataType = dataType_;
    aivAlltoAllVArgs.op = reduceOp_;
    aivAlltoAllVArgs.root = root_;
    aivAlltoAllVArgs.aivTag = sliceId_; // 传入aivTag，Launch时重新组装为aivTag
    aivAlltoAllVArgs.isOpBase = (tempFuncs.opMode == OpMode::OPBASE);
    aivAlltoAllVArgs.xRankSize = tempVTopo_[0].size();
    aivAlltoAllVArgs.yRankSize = 0;
    aivAlltoAllVArgs.zRankSize = 0;
    aivAlltoAllVArgs.numBlocks = MAX_NUM_BLOCKS_ALL_TO_ALL_V;

    for (u64 i = 0; i < tempVTopo_[0].size(); i++) {
        aivAlltoAllVArgs.extraArgs.sendCounts[i] = static_cast<u64*>(op_.all2AllVDataDes.sendCounts)[i];
        aivAlltoAllVArgs.extraArgs.sendDispls[i] = static_cast<u64*>(op_.all2AllVDataDes.sdispls)[i];
        aivAlltoAllVArgs.extraArgs.recvCounts[i] = static_cast<u64*>(op_.all2AllVDataDes.recvCounts)[i];
        aivAlltoAllVArgs.extraArgs.recvDispls[i] = static_cast<u64*>(op_.all2AllVDataDes.rdispls)[i];
        HCCL_INFO(
            "[AivTempAlltoAllVMesh1D] rank is [%llu], iter is [%llu], sendCounts is [%llu], sendDispls is [%llu], "
            "recvCounts is [%llu], recvDispls is [%llu]",
            u64(myRank_), i, aivAlltoAllVArgs.extraArgs.sendCounts[i], aivAlltoAllVArgs.extraArgs.sendDispls[i],
            aivAlltoAllVArgs.extraArgs.recvCounts[i], aivAlltoAllVArgs.extraArgs.recvDispls[i]);
    }

    for (u32 i = 0; i < tempVTopo_[0].size(); i++) {
        aivAlltoAllVArgs.topo_[i] = tempVTopo_[0][i];
    }
    if (tempVTopo_.size() > 1) {
        aivAlltoAllVArgs.yRankSize = tempVTopo_[1].size();
        for (u32 i = 0; i < tempVTopo_[1].size(); i++) {
            aivAlltoAllVArgs.topo_[TOPO_LEN_Y_OFFSET + i] = tempVTopo_[1][i];
        }
    }
    if (tempVTopo_.size() == MAX_DIM_NUM) {
        aivAlltoAllVArgs.zRankSize = tempVTopo_[MAX_DIM_NUM - 1].size();
        for (u32 i = 0; i < tempVTopo_[MAX_DIM_NUM - 1].size(); i++) {
            aivAlltoAllVArgs.topo_[TOPO_LEN_Z_OFFSET + i] = tempVTopo_[MAX_DIM_NUM - 1][i];
        }
    }

    aivAlltoAllVArgs.inputSliceStride = templateDataParams.inputSliceStride;
    aivAlltoAllVArgs.outputSliceStride = templateDataParams.outputSliceStride;
    aivAlltoAllVArgs.repeatNum = templateDataParams.repeatNum;
    aivAlltoAllVArgs.inputRepeatStride = templateDataParams.inputRepeatStride;
    aivAlltoAllVArgs.outputRepeatStride = templateDataParams.outputRepeatStride;

    std::unique_ptr<Instruction> aivInsAlltoAllVMesh1D = std::make_unique<AivInstruction>(allLinks, aivAlltoAllVArgs);

    tempInsQues[0]->Append(std::move(aivInsAlltoAllVMesh1D));

    HCCL_INFO("[AivTempAlltoAllVMesh1D] Run algorithm end: rank[%d]", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
