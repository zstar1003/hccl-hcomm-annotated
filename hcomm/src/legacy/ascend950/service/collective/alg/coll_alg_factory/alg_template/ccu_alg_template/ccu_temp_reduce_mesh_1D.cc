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

#include "ccu_temp_reduce_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_instruction_reduce_mesh1d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_reduce_mesh1d.h"

namespace Hccl {

static CcuInstRegister<CcuContextReduceMesh1D> registrarReduce(CcuInstType::CCU_REDUCE_MESH_1D_DIRECT);

CcuTempReduceMesh1D::CcuTempReduceMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempReduceMesh1D::~CcuTempReduceMesh1D() {}

void CcuTempReduceMesh1D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult CcuTempReduceMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_DEBUG("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[CcuTempReduceMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    buffInfo_ = templateDataParams.buffInfo;
    opMode_ = tempFuncs.opMode;
    CcuInstructionReduceMesh1D ccuIns;
    rootId_ = op_.root;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint32_t rankId = myRank_;
    uint32_t rootId = rootId_;
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
    uint64_t repeatNum = templateDataParams.repeatNum;
    uint64_t normalSliceSize = templateDataParams.sliceSize;
    uint64_t lastSliceSize = templateDataParams.tailSize;
    uint64_t repeatNumVar = UINT64_MAX - repeatNum;

    ccuIns.Init(
        rankId, rootId, op, tempVTopo, inputAddr, outputAddr, token, inputSliceStride, outputSliceStride, repeatNum,
        inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar);

    HCCL_DEBUG(
        "[CcuTempReduceMesh1D] Run Init: rankId[%u], inputAddr[%llu], "
        "outputAddr[%llu], inputSliceStride[%llu], outputSliceStride[%llu], repeatNum[%llu], "
        "inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu], "
        "repeatNumVar[%llu]",
        rankId, inputAddr, outputAddr, inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride,
        outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar);

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_DEBUG("[CcuTempReduceMesh1D] links.size[%zu]", links.size());
    ccuIns.SetLinks(links);

    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 4;
    ccuIns.SetCntCkeNum(cntCkeNum);
    ccuIns.SetRankGroup(rankGroup);
    HCCL_DEBUG("CcuInstructionReduceMesh1D is [%s]", ccuIns.Describe().c_str());
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionReduceMesh1D>(ccuIns)));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMesh1D::GenExtIns(
    const RankGraph* rankGraph, const TemplateInfo& tmpInfo, const std::vector<InsQuePtr>& tempInsQues) const
{
    (void)rankGraph;
    (void)tmpInfo;
    (void)tempInsQues;
    // 框架解析aicpuIns，算法的algCompnnetLite在device侧直接调用Run（）
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
