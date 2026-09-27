/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ios>
#include <iostream>

#include "ccu_temp_all_gather_v_mesh_1D.h"
#include "ccu_instruction_all_gather_v_mesh1d.h"
#include "ccu_context_all_gather_v_mesh1d.h"

#include "log.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {
static CcuInstRegister<CcuContextAllGatherVMesh1D> g_registrarAllGatherV(CcuInstType::CCU_ALL_GATHER_V_MESH_1D_DIRECT);

CcuTempAllGatherVMesh1D::CcuTempAllGatherVMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllGatherVMesh1D::~CcuTempAllGatherVMesh1D() {}

HcclResult CcuTempAllGatherVMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherVMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[CcuTempAllGatherVMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    CcuInstructionAllGatherVMesh1D ccuIns;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint64_t inputAddr
        = BufferTypeToAddr(templateDataParams.buffInfo.inBuffType) + templateDataParams.buffInfo.inBuffBaseOff;
    uint64_t outputAddr
        = BufferTypeToAddr(templateDataParams.buffInfo.outBuffType) + templateDataParams.buffInfo.outBuffBaseOff;

    uint64_t mySliceSize = templateDataParams.sliceSize;
    uint64_t mySliceOutputOffset = templateDataParams.outputSliceStride;

    uint64_t token;
    CHK_RET(GetToken(op_, token));

    ccuIns.Init(
        static_cast<uint32_t>(myRank_), inputAddr, outputAddr, mySliceSize, mySliceOutputOffset, token, op_,
        tempVTopo_);
    HCCL_INFO(
        "[CcuTempAllGatherVMesh1D] Run Init: myRank_[%d], dimSize[%llu], inputAddr[%llu],"
        "outputAddr[%llu], mySliceSize[%llu], mySliceOutputOffset[%llu]",
        myRank_, dimSize[0], inputAddr, outputAddr, mySliceSize, mySliceOutputOffset);

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_INFO("[CcuTempAllGatherVMesh1D] links.size[%zu]", links.size());
    ccuIns.SetLinks(links);

    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    ccuIns.SetRankGroup(rankGroup);

    u32 cntCkeNum = 3;
    ccuIns.SetCntCkeNum(cntCkeNum);
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionAllGatherVMesh1D>(ccuIns)));

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
