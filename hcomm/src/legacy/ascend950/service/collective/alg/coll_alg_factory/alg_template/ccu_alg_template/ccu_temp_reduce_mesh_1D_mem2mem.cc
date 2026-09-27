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

#include "ccu_temp_reduce_mesh_1D_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_instruction_reduce_mesh1d_mem2mem.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_reduce_mesh1d_mem2mem.h"

namespace Hccl {

static CcuInstRegister<CcuContextReduceMeshMem2Mem1D> registrarReduce(CcuInstType::CCU_REDUCE_MESH_1D_MEM2MEM);

CcuTempReduceMeshMem2Mem1D::CcuTempReduceMeshMem2Mem1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempReduceMeshMem2Mem1D::~CcuTempReduceMeshMem2Mem1D() {}

void CcuTempReduceMeshMem2Mem1D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult CcuTempReduceMeshMem2Mem1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum + 1; // 多申请一个 stream 给 ccuInsGroup
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMeshMem2Mem1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempReduceMeshMem2Mem1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);

    buffInfo_ = templateDataParams.buffInfo;
    opMode_ = tempFuncs.opMode;
    CcuInstructionReduceMeshMem2Mem1D ccuIns;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint32_t rankId = myRank_;
    uint32_t rootId = tempVirtRankMap_[rootId_];

    const CollAlgOperator& op = op_;
    const std::vector<std::vector<RankId>>& tempVTopo = tempVTopo_;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint64_t inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
    uint64_t repeatNum = templateDataParams.repeatNum;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize = templateDataParams.sliceSize;
    uint64_t lastSliceSize = templateDataParams.tailSize;
    uint64_t repeatNumVar = UINT64_MAX - repeatNum;

    // 数据切分为sliceNum块，当数据量不能均匀切分时，后面smallDataSliceNum个数据块比前面bigDataSliceNum个数据块每块少1个数据
    uint64_t sliceNum = tempRankSize_ - 1;
    uint64_t sliceSize = templateDataParams.sliceSize; // 获取本rank需要处理的数据量
    uint64_t sliceCount = sliceSize / DataTypeSizeGet(op_.dataType);

    uint64_t bigDataSliceNum = sliceCount % sliceNum;
    uint64_t bigDataSliceSize = (sliceCount / sliceNum + 1) * DataTypeSizeGet(op_.dataType);
    uint64_t smallDataSliceNum = sliceNum - sliceCount % sliceNum;
    uint64_t smallDataSliceSize = sliceCount / sliceNum * DataTypeSizeGet(op_.dataType);

    ccuIns.Init(
        tempVirtRankMap_[myRank_], rootId, op, tempVTopo, inputAddr, outputAddr, token, bigDataSliceNum,
        bigDataSliceSize, smallDataSliceNum, smallDataSliceSize, inputRepeatStride, outputRepeatStride, normalSliceSize,
        lastSliceSize, repeatNumVar);

    HCCL_INFO(
        "[CcuTempReduceMeshMem2Mem1D] Run Init: rankId[%u], rootId[%u], inputAddr[%llu], outputAddr[%llu],"
        "bigDataSliceNum[%llu], bigDataSliceSize[%llu], smallDataSliceNum[%llu], smallDataSliceSize[%llu],"
        "inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu], "
        "repeatNumVar[%llu]",
        rankId, rootId, inputAddr, outputAddr, bigDataSliceNum, bigDataSliceSize, smallDataSliceNum, smallDataSliceSize,
        inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar);

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_DEBUG("[CcuTempReduceMeshMem2Mem1D] links.size[%zu]", links.size());
    ccuIns.SetLinks(links);

    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 4;
    ccuIns.SetCntCkeNum(cntCkeNum);
    ccuIns.SetRankGroup(rankGroup);
    HCCL_DEBUG("CcuTempReduceMeshMem2Mem1D is [%s]", ccuIns.Describe().c_str());
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionReduceMeshMem2Mem1D>(ccuIns)));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMeshMem2Mem1D::GenExtIns(
    const RankGraph* rankGraph, const TemplateInfo& tmpInfo, const std::vector<InsQuePtr>& tempInsQues) const
{
    (void)rankGraph;
    (void)tmpInfo;
    (void)tempInsQues;
    // 框架解析aicpuIns，算法的algCompnnetLite在device侧直接调用Run（）
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
