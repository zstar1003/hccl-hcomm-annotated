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

#include "ccu_instruction_reduce_scatter_mesh2d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_ins_group.h"
#include "ccu_context_reduce_scatter_mesh2d.h"
#include "ccu_temp_reduce_scatter_mesh_2D.h"

namespace Hccl {
constexpr u32 MESH_2D_DIMENSION_NUM = 2;

static CcuInstRegister<CcuContextReduceScatterMesh2D>
    g_registrarReduceScatterMesh2D(CcuInstType::CCU_REDUCE_SCATTER_MESH_2D_DIRECT);

CcuTempReduceScatterMesh2D::CcuTempReduceScatterMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    if (tempVTopo_.size() != MESH_2D_DIMENSION_NUM || tempVTopo_[0].size() <= 1
        || tempVTopo_[1].size() <= 1) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempReducescatterMesh2D] Rank[%d], Invalid tempVTopo "
            "Size[%u] or Invalid tempVTopo[0] size [%u] or tempVTopo[1] size [%u].",
            myRank_, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size()));
    }
    dimSize_.emplace_back(tempVTopo[0].size());
    dimSize_.emplace_back(tempVTopo[1].size());
}

CcuTempReduceScatterMesh2D::~CcuTempReduceScatterMesh2D() {}

void CcuTempReduceScatterMesh2D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult
CcuTempReduceScatterMesh2D::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);
    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    // 按照IODienum来确定stream数量，支持2D和2D的template
    tempResReq.queNum = 1; // 只申请一个insQue，填充一个insGroup，由框架将其中的ins放在多个stream上
    tempResReq.streamNum = tempResReq.queNum + 1; // 多申请一个stream给ccuInsGroup
    uint32_t dieNum = tempVTopo_.size();
    if (dieNum != 2) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        HCCL_ERROR("[CcuTempReducescatterMesh2D] Rank[%d], Invalid IODieNum[%zu].", myRank_, tempVTopo_.size());
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_INFO(
        "[CcuTempReducescatterMesh2D] Rank[%d] requiredQueNum[%u] VtopoSize[%u], VtopoSize0[%u] VtopoSize1[%u].",
        myRank_, tempResReq.queNum, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size());

    uint32_t myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            // find neighbors -> virtualRank
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
            RankId neighborRank = tempVTopo_[dim][neighborAlgRank];
            HCCL_INFO(
                "[CollAlgFactory] [CcuTempReducescatterMesh2D] Rank[%d], Dim[%u], NeighborRank[%d].", myRank_, dim,
                neighborRank);

            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh2D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    opMode_ = tempFuncs.opMode;
    buffInfo_ = buffInfo;

    // 分别记录两个Die上的link，构造rankGroup
    for (auto& pair : tempLinks) {
        if (pair.second.size() == 0 || pair.second[0].GetHop() != 1) { // ESL环境上暂只有直连链路
            THROW<InvalidParamsException>(StringFormat(
                "[CcuTempAlltoAllMesh2D] Rank[%d]--Peer[%d], InvalidHop[%u].", myRank_, pair.first,
                pair.second[0].GetHop()));
        }
        if ((pair.first / dimSize_[0] == myRank_ / dimSize_[0]) && pair.second[0].GetHop() == 1) {
            HCCL_INFO("[CcuTempAlltoAllMesh2D][Run] Rank[%d] insert link to Rank[%d] in linksX", myRank_, pair.first);
            linksX_.emplace_back(pair.second[0]);
        } else if ((pair.first % dimSize_[0] == myRank_ % dimSize_[0]) && pair.second[0].GetHop() == 1) {
            HCCL_INFO("[CcuTempAlltoAllMesh2D][Run] Rank[%d] insert link to Rank[%d] in linksY", myRank_, pair.first);
            linksY_.emplace_back(pair.second[0]);
        } else {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuTempAlltoAllMesh2D] Rank[%d], Unexpected peerRank[%d] in tempLinks.", myRank_, pair.first));
        }
    }

    RankGroup rankGroupX;
    RankGroup rankGroupY;
    AddRanksToGroup(tempVTopo_, rankGroupX, rankGroupY);

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t offset;
    uint64_t outputSize = static_cast<uint64_t>(op_.outputMem->GetSize());
    if (opMode_ == OpMode::OPBASE) {
        if (tempFuncs.isForepart) {
            inputAddr = BufferTypeToAddr(tempFuncs.usrData.usrInSlices[myRank_].GetType());
            // 当前loop的size大小
            offset = tempFuncs.usrData.usrOutSlices[0].GetOffset();
        } else {
            inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
            // 从inBuff获取数据，只需要加上rank偏移
            offset = sliceInfoVec[myRank_][0].offset;
        }
        if (tempFuncs.isBottom) {
            outputAddr = BufferTypeToAddr(tempFuncs.usrData.usrOutSlices[0].GetType())
                         + tempFuncs.usrData.usrOutSlices[0].GetOffset();
        } else {
            outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
        }
    } else {
        offset = tempFuncs.usrData.usrOutSlices[0].GetOffset();
        inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
        outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff
                     + tempFuncs.usrData.usrOutSlices[0].GetOffset();
    }
    uint64_t sliceSize = sliceInfoVec[myRank_][0].size; // 获取本rank需要处理的数据量
    uint64_t token;
    CHK_RET(GetToken(op_, token));

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();

    for (uint32_t axisId = 0; axisId < 2; axisId++) { // 2D算法，需要执行两次
        // 计算每次编译的偏移量和数据量
        uint64_t sliceCount = sliceSize / DataTypeSizeGet(dataType_);
        uint64_t xAxisSize
            = (sliceCount * dimSize_[0] / (dimSize_[axisId] + dimSize_[1 - axisId])) * DataTypeSizeGet(dataType_);
        uint64_t yAxisSize = sliceSize - xAxisSize;

        CcuInstructionReduceScatterMesh2D ccuInsReduceScatterMesh2D;
        ccuInsReduceScatterMesh2D.Init(
            dimSize_, static_cast<uint32_t>(myRank_), inputAddr, outputAddr, axisId, outputSize, xAxisSize, yAxisSize,
            offset, token, op_, tempVTopo_);

        HCCL_INFO(
            "[CcuTempReduceScatterMesh2D] Init: dimSize0[%llu], dimSize1[%llu], myRank_[%d], inputAddr[%llu],"
            "outputAddr[%llu], sliceSize[%llu], xAxisSize[%llu], yAxisSize[%llu], offset[%llu]",
            dimSize_[0], dimSize_[1], myRank_, inputAddr, outputAddr, outputSize, xAxisSize, yAxisSize, offset);

        ccuInsReduceScatterMesh2D.SetLinks(axisId == 0 ? linksX_ : linksY_);
        ccuInsReduceScatterMesh2D.SetRankGroup(axisId == 0 ? rankGroupX : rankGroupY);
        u32 ckeNum = 5;
        ccuInsReduceScatterMesh2D.SetCntCkeNum(ckeNum); // 每个transport用4个CKE
        ccuInsReduceScatterMesh2D.Describe();
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionReduceScatterMesh2D>(ccuInsReduceScatterMesh2D)));
    }

    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有1条流

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
