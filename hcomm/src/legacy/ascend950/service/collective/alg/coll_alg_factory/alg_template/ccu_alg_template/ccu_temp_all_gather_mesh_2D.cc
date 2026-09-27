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

#include "ccu_temp_all_gather_mesh_2D.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_gather_mesh2d.h"
#include "ccu_ins_group.h"

namespace Hccl {
constexpr u32 MESH_2D_DIMENSION_NUM = 2;

static CcuInstRegister<CcuContextAllGatherMesh2D> registrarAllGather(CcuInstType::CCU_ALLGATHER_MESH_2D_DIRECT);

CcuTempAllGatherMesh2D::CcuTempAllGatherMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    // 填充框内的维度大小
    if (tempVTopo_.size() != MESH_2D_DIMENSION_NUM || tempVTopo_[0].size() <= 1
        || tempVTopo_[1].size() <= 1) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempAllGatherMesh2D] Rank[%d], Invalid tempVTopo "
            "Size[%u] or Invalid tempVTopo[0] size [%u] or tempVTopo[1] size [%u].",
            myRank_, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size()));
    }
    dimSize_.emplace_back(tempVTopo[0].size());
    dimSize_.emplace_back(tempVTopo[1].size());
}

CcuTempAllGatherMesh2D::~CcuTempAllGatherMesh2D() {}

HcclResult CcuTempAllGatherMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1; // 只申请一个insQue，填充一个insGroup，由框架将其中的ins放在多个stream上
    tempResReq.streamNum = tempResReq.queNum + 1; // 多申请一个 stream 给 ccuInsGroup
    uint32_t dieNum = tempVTopo_.size();
    if (dieNum != 2) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        THROW<InvalidParamsException>(
            StringFormat("[CcuTempAllGatherMesh2D] Rank[%d], Invalid IODieNum[%u].", myRank_, dieNum));
    }
    HCCL_INFO(
        "[CcuTempAllGatherMesh2D] Rank[%d] requiredQueNum[%u] VtopoSize[%u], VtopoSize0[%u] VtopoSize1[%u].", myRank_,
        tempResReq.queNum, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size());

    uint32_t myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            // find neighbors -> virtualRank
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
            RankId neighborRank = tempVTopo_[dim][neighborAlgRank];
            HCCL_INFO(
                "[CollAlgFactory] [CcuTempAllGatherMesh2D] Rank[%d], Dim[%u], NeighborRank[%d].", myRank_, dim,
                neighborRank);

            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
/*
dataSize / (rankSize) --> chunkSize
dataSize / (rankSize * queNum) --> sliceSize

SliceInfoVecforNHR: [1st chunk: [1st Slice, 2nd Slice, ...], 2nd chunk: [1st Slice, 2nd Slice, ...], ...]
*/
HcclResult
CcuTempAllGatherMesh2D::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherMesh2D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    opMode_ = tempFuncs.opMode;
    buffInfo_ = buffInfo;

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t offset;

    // 分别记录两个Die上的link，构造rankGroup
    for (auto pair : tempLinks) {
        if (pair.second.size() == 0 || pair.second[0].GetHop() != 1) { // ESL环境上暂只有直连链路
            THROW<InvalidParamsException>(StringFormat(
                "[CcuTempAllGatherMesh2D] Rank[%d]--Peer[%d], InvalidHop[%u].", myRank_, pair.first,
                pair.second[0].GetHop()));
        }
        if ((pair.first / dimSize_[0] == myRank_ / dimSize_[0]) && pair.second[0].GetHop() == 1) {
            HCCL_INFO("[CcuTempAllGatherMesh2D][Run] Rank[%d] insert link to Rank[%d] in linksX", myRank_, pair.first);
            linksX_.emplace_back(pair.second[0]);
        } else if ((pair.first % dimSize_[0] == myRank_ % dimSize_[0]) && pair.second[0].GetHop() == 1) {
            HCCL_INFO("[CcuTempAllGatherMesh2D][Run] Rank[%d] insert link to Rank[%d] in linksY", myRank_, pair.first);
            linksY_.emplace_back(pair.second[0]);
        } else {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuTempAllGatherMesh2D] Rank[%d], Unexpected peerRank[%d] in tempLinks.", myRank_, pair.first));
        }
    }

    RankGroup rankGroupX;
    RankGroup rankGroupY;
    AddRanksToGroup(tempVTopo_, rankGroupX, rankGroupY);

    if (opMode_ == OpMode::OPBASE) {
        if (tempFuncs.isForepart) {
            inputAddr = BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType())
                        + tempFuncs.usrData.usrInSlices[0].GetOffset();
        } else {
            inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
        }
        if (tempFuncs.isBottom) {
            outputAddr = BufferTypeToAddr(tempFuncs.usrData.usrOutSlices[0].GetType())
                         + tempFuncs.usrData.usrInSlices[0].GetOffset();
        } else {
            outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
        }
    } else {
        // 图模式没有 tempFuncs.usrData，直接通过 buffInfo_ 获取输入输出地址
        inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff
                    + tempFuncs.usrData.usrInSlices[0].GetOffset();
        outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff
                     + tempFuncs.usrData.usrInSlices[0].GetOffset();
    }

    offset = static_cast<uint64_t>(op_.inputMem->GetSize());
    uint64_t sliceSize = sliceInfoVec[myRank_][0].size;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    HCCL_INFO("[CcuTempAllGatherMesh2D] Rank[%d], input[%llu], output[%llu].", myRank_, inputAddr, outputAddr);

    uint64_t aSize = sliceSize * dimSize_[0] / (dimSize_[0] + dimSize_[1]);
    uint64_t bSize = sliceSize - aSize;
    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t axisId = 0; axisId < 2; axisId++) { // 2D算法，需要执行两次
        HCCL_INFO(
            "[CcuTempAllGatherMesh2D][Run] Rank[%d], axisId[%u], aSize[%llu], bSize[%llu].", myRank_, axisId, aSize,
            bSize);

        CcuInstructionAllGatherMesh2D ins = CcuInstructionAllGatherMesh2D();
        ins.Init(myRank_, axisId, inputAddr, outputAddr, aSize, bSize, sliceSize, offset, token, op_, tempVTopo_);
        ins.SetLinks(axisId == 0 ? linksX_ : linksY_);
        ins.SetRankGroup(axisId == 0 ? rankGroupX : rankGroupY);
        u32 cntCkeNum = 5; // 每个transport用5个CKE
        ins.SetCntCkeNum(cntCkeNum);
        ins.Describe();
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh2D>(ins)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
