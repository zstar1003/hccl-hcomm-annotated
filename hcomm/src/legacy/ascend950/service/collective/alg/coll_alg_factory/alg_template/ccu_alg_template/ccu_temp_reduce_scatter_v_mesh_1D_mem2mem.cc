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

#include "ccu_temp_reduce_scatter_v_mesh_1D_mem2mem.h"
#include "ccu_instruction_reduce_scatter_v_mesh1d_mem2mem.h"
#include "ccu_context_reduce_scatter_v_mesh1d_mem2mem.h"

#include "log.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {
static CcuInstRegister<CcuContextReduceScatterVMeshMem2Mem1D>
    g_registrarReduceScatterVmem2mem(CcuInstType::CCU_REDUCE_SCATTER_V_MESH_1D_MEM2MEM_DIRECT);

CcuTempReduceScatterVMeshMem2Mem1D::CcuTempReduceScatterVMeshMem2Mem1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempReduceScatterVMeshMem2Mem1D::~CcuTempReduceScatterVMeshMem2Mem1D() {}

void CcuTempReduceScatterVMeshMem2Mem1D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult CcuTempReduceScatterVMeshMem2Mem1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterVMeshMem2Mem1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempReduceScatterVMeshMem2Mem1D] empty queue"),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    CcuInstructionReduceScatterVMeshMem2Mem1D ccuIns;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint64_t inputAddr
        = BufferTypeToAddr(templateDataParams.buffInfo.inBuffType) + templateDataParams.buffInfo.inBuffBaseOff;
    uint64_t outputAddr
        = BufferTypeToAddr(templateDataParams.buffInfo.outBuffType) + templateDataParams.buffInfo.outBuffBaseOff;
    uint64_t scratchAddr
        = BufferTypeToAddr(templateDataParams.buffInfo.scratBuffType) + templateDataParams.buffInfo.scratchBuffBaseOff;

    uint64_t mySliceSize = templateDataParams.sliceSize;
    uint64_t mySliceInputOffset = templateDataParams.inputSliceStride;

    uint64_t token;
    CHK_RET(GetToken(op_, token));

    ccuIns.Init(
        static_cast<uint32_t>(myRank_), inputAddr, outputAddr, scratchAddr, mySliceSize, mySliceSize,
        mySliceInputOffset, token, op_, tempVTopo_);
    HCCL_INFO(
        "[CcuTempReduceScatterVMeshMem2Mem1D] Run Init: myRank_[%d], dimSize[%llu], inputAddr[%llu],"
        "outputAddr[%llu], mySliceSize[%llu], mySliceInputOffset[%llu]",
        myRank_, dimSize[0], inputAddr, outputAddr, mySliceSize, mySliceInputOffset);

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_INFO("[CcuTempReduceScatterVMeshMem2Mem1D] links.size[%zu]", links.size());
    ccuIns.SetLinks(links);

    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    ccuIns.SetRankGroup(rankGroup);

    u32 cntCkeNum = 3;
    ccuIns.SetCntCkeNum(cntCkeNum);
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionReduceScatterVMeshMem2Mem1D>(ccuIns)));

    return HcclResult::HCCL_SUCCESS;
}

u32 CcuTempReduceScatterVMeshMem2Mem1D::CalcScratchMultiple(BufferType input, BufferType output)
{
    (void)input;
    (void)output;
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    return tempRankSize_;
}

} // namespace Hccl
