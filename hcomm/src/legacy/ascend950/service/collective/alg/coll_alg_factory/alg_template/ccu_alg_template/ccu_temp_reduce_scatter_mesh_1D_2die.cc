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
#include "ccu_ins_group.h"
#include "ccu_context_reduce_scatter_mesh1d_2die.h"
#include "ccu_temp_reduce_scatter_mesh_1D_2die.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {

static CcuInstRegister<CcuContextReduceScatterMesh1D2Die>
    g_registrarReduceScatter(CcuInstType::CCU_REDUCE_SCATTER_MESH_1D_2DIE);

CcuTempReduceScatterMesh1D2Die::CcuTempReduceScatterMesh1D2Die(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempReduceScatterMesh1D2Die::~CcuTempReduceScatterMesh1D2Die() {}

u32 CcuTempReduceScatterMesh1D2Die::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 2;
}

void CcuTempReduceScatterMesh1D2Die::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult CcuTempReduceScatterMesh1D2Die::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum + 1;
    HCCL_INFO("[CcuTempReduceScatterMesh1D2Die][CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempReduceScatterMesh1D2Die::GetMaxSliceSize() const
{
    u64 msSize = 4 * 1024;
    u64 loopNum = 64;
    u64 maxIterNum = 8192;
    return msSize * loopNum * maxIterNum; // 2G
}

HcclResult CcuTempReduceScatterMesh1D2Die::GenExtIns(
    const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempFuncs;
    HCCL_INFO(
        "[CcuTempReduceScatterMesh1D2Die] sliceSize[%llu], inputSliceStride[%llu], outputSliceStride[%llu], "
        "repeatNum[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], tailSize[%llu]",
        tempAlgParams.sliceSize, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride,
        tempAlgParams.repeatNum, tempAlgParams.inputRepeatStride, tempAlgParams.outputRepeatStride,
        tempAlgParams.tailSize);
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempReduceScatterMesh1D2Die] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    uint64_t inputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.inBuffType) + tempAlgParams.buffInfo.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.outBuffType) + tempAlgParams.buffInfo.outBuffBaseOff;
    uint64_t scratchAddr
        = BufferTypeToAddr(tempAlgParams.buffInfo.scratBuffType) + tempAlgParams.buffInfo.scratchBuffBaseOff;
    uint64_t inputSliceStride = tempAlgParams.inputSliceStride;
    uint64_t outputSliceStride = tempAlgParams.outputSliceStride;
    u64 sliceSize = tempAlgParams.sliceSize;

    uint64_t token;
    CHK_RET(GetToken(op_, token));
    HCCL_INFO(
        "[CcuTempReduceScatterMesh1D2Die] inputAddr[%llu], outputAddr[%llu], scratchAddr0[%llu], "
        "inputSliceStride[%llu], outputSliceStride[%llu]",
        inputAddr, outputAddr, scratchAddr, inputSliceStride, outputSliceStride);

    u32 dieNum = 2;
    std::vector<std::vector<LinkData>> linksForDie(dieNum);
    std::vector<RankGroup> rankGroupForDie(dieNum);
    for (auto& pair : tempLinks) {
        const RankId rank = pair.first;
        const std::vector<LinkData>& links = pair.second;
        if (links.size() == 0) {
            continue;
        }
        const LinkData& link = links[0];
        u32 dieIdx = link.GetLocalDieId();
        linksForDie[dieIdx].push_back(link);
        rankGroupForDie[dieIdx].AddRank(rank);
    }
    for (auto& rankGroup : rankGroupForDie) {
        rankGroup.AddRank(myRank_);
    }
    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (u32 die = 0; die < dieNum; die++) {
        bool rmtReduceWithMyRank = linksForDie[die].size() > linksForDie[1 - die].size() ? false : true;
        CcuInstructionReduceScatterMesh1D2Die ccuInstruction;
        ccuInstruction.Init(
            tempVirtRankMap_[myRank_], op_, rmtReduceWithMyRank, tempVTopo_, inputAddr, outputAddr, scratchAddr,
            sliceSize, token);
        ccuInstruction.SetLinks(linksForDie[die]);
        ccuInstruction.SetRankGroup(rankGroupForDie[die]);
        ccuInstruction.SetCntCkeNum(3);
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionReduceScatterMesh1D2Die>(ccuInstruction)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr));

    DataSlice srcSlice(tempAlgParams.buffInfo.scratBuffType, tempAlgParams.buffInfo.scratchBuffBaseOff, sliceSize);
    DataSlice dstSlice(tempAlgParams.buffInfo.outBuffType, tempAlgParams.buffInfo.outBuffBaseOff, sliceSize);
    LocalReduce(tempInsQues[0], srcSlice, dstSlice, op_.dataType, op_.reduceOp);

    HCCL_INFO("[CcuTempReduceScatterMesh1D2Die] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
