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

#include "ccu_instruction_all_gather_mesh1d_mem2mem_with_stride.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_gather_mesh1d_mem2mem_with_stride.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem_with_stride.h"

namespace Hccl {

static CcuInstRegister<CcuContextAllGatherMesh1DMem2MemWithStride>
    g_registrarAllGather(CcuInstType::CCU_ALLGATHER_MESH_1D_MEM2MEM_WITH_STRIDE_DIRECT);

CcuTempAllGatherMesh1DMem2MemWithStride::CcuTempAllGatherMesh1DMem2MemWithStride(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllGatherMesh1DMem2MemWithStride::~CcuTempAllGatherMesh1DMem2MemWithStride() {}

HcclResult CcuTempAllGatherMesh1DMem2MemWithStride::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_DEBUG("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempAllGatherMesh1DMem2MemWithStride::GetMaxSliceSize() const { return UB_MAX_DATA_SIZE; }

HcclResult CcuTempAllGatherMesh1DMem2MemWithStride::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempAllGatherMesh1DMem2MemWithStride] empty queue"),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);

    opMode_ = tempFuncs.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    CcuInstructionAllGatherMesh1DMem2MemWithStride ccuIns;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint32_t rankId = myRank_;
    uint32_t repeatNum = templateDataParams.repeatNum;
    const CollAlgOperator& op = op_;
    const std::vector<std::vector<RankId>>& tempVTopo = tempVTopo_;
    uint64_t inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize = templateDataParams.sliceSize;
    uint64_t lastSliceSize = templateDataParams.tailSize;
    uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;

    ccuIns.Init(
        tempVirtRankMap_[rankId], repeatNum, op, tempVTopo, inputAddr, outputAddr, token, inputSliceStride,
        outputSliceStride, inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, isInputOutputEqual);

    HCCL_DEBUG(
        "[CcuTempAllGatherMesh1DMem2MemWithStride] Run Init: rankId[%u], repeatNum[%u], inputAddr[%llu], "
        "outputAddr[%llu], inputSliceStride[%llu], outputSliceStride[%llu], "
        "inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu]",
        rankId, repeatNum, inputAddr, outputAddr, inputSliceStride, outputSliceStride, inputRepeatStride,
        outputRepeatStride, normalSliceSize, lastSliceSize);

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2MemWithStride] links.size[%zu]", links.size());
    ccuIns.SetLinks(links);

    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 3;
    ccuIns.SetCntCkeNum(cntCkeNum);
    ccuIns.SetRankGroup(rankGroup);
    HCCL_DEBUG("CcuInstructionAllGatherMesh1DMem2MemWithStride is [%s]", ccuIns.Describe().c_str());
    ccuIns.Describe();
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1DMem2MemWithStride>(ccuIns)));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh1DMem2MemWithStride::GenExtIns(
    const RankGraph* rankGraph, const TemplateInfo& tmpInfo, const std::vector<InsQuePtr>& tempInsQues) const
{
    (void)rankGraph;
    (void)tmpInfo;
    (void)tempInsQues;
    // 框架解析aicpuIns，算法的algCompnnetLite在device侧直接调用Run（）
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
