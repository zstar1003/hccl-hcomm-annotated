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

#include "ccu_instruction_reduce_scatter_mesh1d_mem2mem.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_reduce_scatter_mesh1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"

namespace Hccl {
static CcuInstRegister<CcuContextReduceScatterMeshMem2Mem1D>
    g_registrarReduceScatter(CcuInstType::CCU_REDUCE_SCATTER_MESH_1D_MEM2MEM);

CcuTempReduceScatterMeshMem2Mem1D::CcuTempReduceScatterMeshMem2Mem1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempReduceScatterMeshMem2Mem1D::~CcuTempReduceScatterMeshMem2Mem1D() {}

void CcuTempReduceScatterMeshMem2Mem1D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D::CalcRes(AlgTempResReq& tempResReq)
{
    // 按照IODienum来确定stream数量，支持1D和2D的template
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_DEBUG("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

u32 CcuTempReduceScatterMeshMem2Mem1D::CalcScratchMultiple(BufferType input, BufferType output)
{
    (void)input;
    (void)output;
    // one shot场景，scratch Buffer需要是usrIn的rankSize倍
    return tempRankSize_;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempReduceScatterMeshMem2Mem1D] empty queue"),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    CcuInstructionReduceScatterMeshMem2Mem1D ccuIns;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint32_t rankId = myRank_;
    uint64_t repeatNumTmp = templateDataParams.repeatNum;
    const CollAlgOperator& op = op_;
    const std::vector<std::vector<RankId>>& tempVTopo = tempVTopo_;
    uint64_t inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint64_t scratchAddr = BufferTypeToAddr(buffInfo_.scratBuffType) + buffInfo_.scratchBuffBaseOff;
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize = templateDataParams.sliceSize;
    uint64_t lastSliceSize = templateDataParams.tailSize;

    uint64_t repeatNum = UINT64_MAX - repeatNumTmp;
    ccuIns.Init(
        tempVirtRankMap_[rankId], repeatNum, op, tempVTopo, inputAddr, outputAddr, token, scratchAddr, inputSliceStride,
        inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize);

    HCCL_DEBUG(
        "[CcuTempReduceScatterMeshMem2Mem1D] Run Init: rankId[%u], repeatNum[%u], inputAddr[%llu], "
        "outputAddr[%llu], scratchAddr[%llu], inputSliceStride[%llu], "
        "inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu]",
        rankId, repeatNum, inputAddr, outputAddr, scratchAddr, inputSliceStride, inputRepeatStride, outputRepeatStride,
        normalSliceSize, lastSliceSize);

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_DEBUG("[CcuTempReduceScatterMeshMem2Mem1D] links.size[%zu]", links.size());
    ccuIns.SetLinks(links);

    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 4; // context里要用的前后同步的信号数
    ccuIns.SetCntCkeNum(cntCkeNum);
    ccuIns.SetRankGroup(rankGroup);
    HCCL_DEBUG("CcuTempReduceScatterMeshMem2Mem1D is [%s]", ccuIns.Describe().c_str());
    ccuIns.Describe();
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionReduceScatterMeshMem2Mem1D>(
        ccuIns))); // 将一个新创建的 CcuInstructionReduceScatterMesh1DMem2Mem 对象添加到 tempInsQues[0] 管理的指令队列

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMeshMem2Mem1D::GenExtIns(
    const RankGraph* rankGraph, const TemplateInfo& tmpInfo, const std::vector<InsQuePtr>& tempInsQues) const
{
    (void)rankGraph;
    (void)tmpInfo;
    (void)tempInsQues;
    // 框架解析aicpuIns，算法的algCompnnetLite在device侧直接调用Run（）
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
