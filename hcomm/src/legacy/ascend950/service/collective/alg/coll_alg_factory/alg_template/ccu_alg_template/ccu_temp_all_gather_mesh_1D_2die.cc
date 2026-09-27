/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_gather_mesh1d_2die.h"
#include "ccu_temp_all_gather_mesh_1D_2die.h"
#include "ccu_ins_group.h"

namespace Hccl {

static CcuInstRegister<CcuContextAllGatherMesh1D2Die> g_registrarAllGather(CcuInstType::CCU_ALLGATHER_MESH_1D_2DIE);

CcuTempAllGatherMesh1D2Die::CcuTempAllGatherMesh1D2Die(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllGatherMesh1D2Die::~CcuTempAllGatherMesh1D2Die() {}

HcclResult CcuTempAllGatherMesh1D2Die::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum + 1;
    HCCL_DEBUG("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);

    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempAllGatherMesh1D2Die::GetMaxSliceSize() const { return UB_MAX_DATA_SIZE; }

HcclResult CcuTempAllGatherMesh1D2Die::GenExtIns(
    const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempAllGatherMesh1D2Die] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    CcuInstructionAllGatherMesh1D2Die ccuInsAllGatherMesh1D2Die;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint64_t sliceSize = tempAlgParams.sliceSize;
    uint64_t inputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.inBuffType) + tempAlgParams.buffInfo.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.outBuffType) + tempAlgParams.buffInfo.outBuffBaseOff;
    uint64_t outputSliceStride = tempAlgParams.outputSliceStride;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    HCCL_INFO(
        "[CcuTempAllGatherMesh1D2Die] dimSize[%llu], sliceSize[%llu], inputAddr[%llu],"
        "outputAddr[%llu], outputSliceStride[%llu]",
        dimSize[0], sliceSize, inputAddr, outputAddr, outputSliceStride);

    // key表示为dieId
    std::map<uint32_t, std::vector<LinkData>> linksDie;
    std::map<uint32_t, RankGroup> rankGroup;

    for (auto link : tempLinks) {
        std::vector<LinkData> linkData = link.second;
        RankId peerRankId = link.first;
        if (linkData.size() == 0) {
            continue;
        }

        linksDie[linkData[0].GetLocalDieId()].push_back(linkData[0]);
        rankGroup[linkData[0].GetLocalDieId()].AddRank(peerRankId);
    }

    HCCL_INFO(
        "[CcuTempAllGatherMesh1D2Die] linksDie0Size[%zu], linksDie1Size[%zu]", linksDie[0].size(), linksDie[1].size());

    rankGroup[0].AddRank(myRank_);
    rankGroup[1].AddRank(myRank_);

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t dieId = 0; dieId < 2; dieId++) { // 2Die算法，需要下发 2 条通信指令
        CcuInstructionAllGatherMesh1D2Die ccuInstruction;
        bool withMyRank = linksDie[dieId].size() > linksDie[1 - dieId].size() ? false : true;
        ccuInstruction.Init(
            myRank_, inputAddr, outputAddr, sliceSize, outputSliceStride, token, withMyRank, op_, tempVTopo_);
        ccuInstruction.SetLinks(linksDie[dieId]);
        ccuInstruction.SetRankGroup(rankGroup[dieId]);
        ccuInstruction.SetCntCkeNum(5); // 每个transport用5个CKE
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D2Die>(ccuInstruction)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条que
    HCCL_INFO("[CcuTempAllGatherMesh1D2Die] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
