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

#include "ccu_temp_reduce_mesh_1D_two_shot_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_instruction_reduce_mesh1d_two_shot_mem2mem.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_reduce_mesh1d_two_shot_mem2mem.h"

namespace Hccl {

static CcuInstRegister<CcuContextReduceMeshTwoShotMem2Mem1D>
    registrarReduce(CcuInstType::CCU_REDUCE_MESH_1D_TWO_SHOT_MEM2MEM);

CcuTempReduceMeshTwoShotMem2Mem1D::CcuTempReduceMeshTwoShotMem2Mem1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempReduceMeshTwoShotMem2Mem1D::~CcuTempReduceMeshTwoShotMem2Mem1D() {}

void CcuTempReduceMeshTwoShotMem2Mem1D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult CcuTempReduceMeshTwoShotMem2Mem1D::CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    u64 unitAlignSize = DataTypeSizeGet(dataType_);
    u64 chunkSize = RoundUp(dataSize, (tempRankSize_ * unitAlignSize)) * unitAlignSize;
    HCCL_INFO(
        "chunkSize[%llu], dataSize[%llu], tempRankSize_[%u], unitAlignSize[%llu]", chunkSize, dataSize, tempRankSize_,
        unitAlignSize);
    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize_; rankIdx++) {
        u64 currChunkSize = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice = {accumOff, currChunkSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize_ - 1][0].offset + sliceInfoVec[tempRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR(
            "[CcuTempReduceMeshTwoShotMem2Mem1D] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!", chunkSize,
            myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMeshTwoShotMem2Mem1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum + 1; // 多申请一个 stream 给 ccuInsGroup
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMeshTwoShotMem2Mem1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    buffInfo_ = templateDataParams.buffInfo;
    opMode_ = tempFuncs.opMode;
    CcuInstructionReduceMeshTwoShotMem2Mem1D ccuIns;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);
    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSlice(templateDataParams.sliceSize, sliceInfoVec));

    uint32_t rankId = tempVirtRankMap_[myRank_];
    uint32_t rootId = tempVirtRankMap_[rootId_];

    const CollAlgOperator& op = op_;
    const std::vector<std::vector<RankId>>& tempVTopo = tempVTopo_;
    uint64_t inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
    uint64_t scratchAddr = BufferTypeToAddr(buffInfo_.scratBuffType) + buffInfo_.scratchBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint64_t normalSliceSize = sliceInfoVec[0][0].size;
    uint64_t lastSliceSize = sliceInfoVec[tempRankSize_ - 1][0].size;
    uint64_t mySliceSize = sliceInfoVec[rankId][0].size;

    ccuIns.Init(
        rankId, rootId, op, tempVTopo, inputAddr, outputAddr, scratchAddr, token, normalSliceSize, lastSliceSize,
        mySliceSize);

    HCCL_INFO(
        "[CcuTempReduceMeshTwoShotMem2Mem1D] Run Init: rankId[%u], rootId[%u], inputAddr[%llu], outputAddr[%llu],"
        "scratchAddr[%llu], normalSliceSize[%llu], lastSliceSize[%llu], mySliceSize[%llu]",
        rankId, rootId, inputAddr, outputAddr, scratchAddr, normalSliceSize, lastSliceSize, mySliceSize);

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_DEBUG("[CcuTempReduceMeshTwoShotMem2Mem1D] links.size[%zu]", links.size());
    ccuIns.SetLinks(links);

    RankGroup rankGroup;
    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 4;
    ccuIns.SetCntCkeNum(cntCkeNum);
    ccuIns.SetRankGroup(rankGroup);
    HCCL_DEBUG("CcuTempReduceMeshTwoShotMem2Mem1D is [%s]", ccuIns.Describe().c_str());
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionReduceMeshTwoShotMem2Mem1D>(ccuIns)));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceMeshTwoShotMem2Mem1D::GenExtIns(
    const RankGraph* rankGraph, const TemplateInfo& tmpInfo, const std::vector<InsQuePtr>& tempInsQues) const
{
    (void)rankGraph;
    (void)tmpInfo;
    (void)tempInsQues;
    // 框架解析aicpuIns，算法的algCompnnetLite在device侧直接调用Run（）
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
