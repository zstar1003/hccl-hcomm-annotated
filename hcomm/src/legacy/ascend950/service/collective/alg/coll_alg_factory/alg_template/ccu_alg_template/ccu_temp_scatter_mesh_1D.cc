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

#include "log.h"

#include "ccu_instruction_scatter_mesh1d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_scatter_mesh1d.h"
#include "ccu_temp_scatter_mesh_1D.h"
#include "dev_mode.h"

namespace Hccl {

static CcuInstRegister<CcuContextScatterMesh1D> g_registrarScatter(CcuInstType::CCU_SCATTER_MESH_1D_DIRECT);

CcuTempScatterMesh1D::CcuTempScatterMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempScatterMesh1D::~CcuTempScatterMesh1D() {}

uint64_t CcuTempScatterMesh1D::GetExpandedMode() const { return DeviceMode::CCU; }

uint64_t CcuTempScatterMesh1D::GetMaxSliceSize() const { return UB_MAX_DATA_SIZE; }

HcclResult CcuTempScatterMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[CcuTempScatterMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    HCCL_INFO("[CcuTempScatterMesh1D] Run.");
    opMode_ = tempFuncs.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    CcuInstructionScatterMesh1D ccuIns;
    uint32_t virtRankId = tempVirtRankMap_[myRank_];

    const CollAlgOperator& op = op_;
    const std::vector<std::vector<RankId>>& tempVTopo = tempVTopo_;
    uint64_t rootId = tempVirtRankMap_[rootId_];
    uint64_t inputAddr
        = BufferTypeToAddr(templateDataParams.buffInfo.inBuffType) + templateDataParams.buffInfo.inBuffBaseOff;
    uint64_t outputAddr
        = BufferTypeToAddr(templateDataParams.buffInfo.outBuffType) + templateDataParams.buffInfo.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize = templateDataParams.sliceSize;
    uint64_t lastSliceSize = templateDataParams.tailSize;
    uint64_t repeatNum = templateDataParams.repeatNum;
    uint64_t repeatNumVar = UINT64_MAX - repeatNum; // 在context中让repeatNumVar累加到UINT64_MAX结束ccu while循环

    ccuIns.Init(
        virtRankId, rootId, op, tempVTopo, inputAddr, outputAddr, token, inputSliceStride, outputSliceStride,
        inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar);
    HCCL_INFO(
        "[CcuTempScatterMesh1D] Run Init: virtRankId[%u], rankId[%d], inputAddr[%llu], "
        "outputAddr[%llu], inputSliceStride[%llu], outputSliceStride[%llu], "
        "inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu], "
        "repeatNumVar[%llu]",
        virtRankId, myRank_, inputAddr, outputAddr, inputSliceStride, outputSliceStride, inputRepeatStride,
        outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar);

    if (normalSliceSize == 0) {
        HCCL_INFO("[CcuTempScatterMesh1D] DataCount == 0, Template Run Ends.");
        return HCCL_SUCCESS;
    }

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_INFO("[CcuTempScatterMesh1D] links.size[%zu]", links.size());
    ccuIns.SetLinks(links);

    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 3;
    ccuIns.SetCntCkeNum(cntCkeNum);
    ccuIns.SetRankGroup(rankGroup);
    HCCL_INFO("CcuInstructionScatterMesh1D is [%s]", ccuIns.Describe().c_str());
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionScatterMesh1D>(ccuIns)));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterMesh1D::GenExtIns(
    const RankGraph* rankGraph, const TemplateInfo& tmpInfo, const std::vector<InsQuePtr>& tempInsQues) const
{
    (void)rankGraph;
    (void)tmpInfo;
    (void)tempInsQues;
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
