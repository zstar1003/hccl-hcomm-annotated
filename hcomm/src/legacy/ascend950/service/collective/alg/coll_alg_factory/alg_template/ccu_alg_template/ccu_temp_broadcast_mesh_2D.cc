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

#include "ccu_instruction_broadcast_mesh2d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_ins_group.h"
#include "ccu_context_broadcast_mesh2d.h"
#include "ccu_temp_broadcast_mesh_2D.h"

namespace Hccl {
constexpr u32 MESH_2D_DIMENSION_NUM = 2;

static CcuInstRegister<CcuContextBroadcastMesh2D> g_registrarBroadcastMesh2D(CcuInstType::CCU_BROADCAST_MESH_2D_DIRECT);

CcuTempBroadcastMesh2D::CcuTempBroadcastMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    if (tempVTopo_.size() != MESH_2D_DIMENSION_NUM || tempVTopo_[0].size() <= 1
        || tempVTopo_[1].size() <= 1) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempBroadcastMesh2D] Rank[%d], Invalid tempVTopo "
            "Size[%u] or Invalid tempVTopo[0] size [%u] or tempVTopo[1] size [%u].",
            myRank_, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size()));
    }
    dimSize_.emplace_back(tempVTopo[0].size());
    dimSize_.emplace_back(tempVTopo[1].size());
}

CcuTempBroadcastMesh2D::~CcuTempBroadcastMesh2D() {}

HcclResult CcuTempBroadcastMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    // 按照IODienum来确定stream数量，支持2D和2D的template
    tempResReq.queNum = 1; // 只申请一个insQue，填充一个insGroup，由框架将其中的ins放在多个stream上
    tempResReq.streamNum = tempResReq.queNum + 1; // 多申请一个 stream 给 ccuInsGroup
    uint32_t dieNum = tempVTopo_.size();
    if (dieNum != 2) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        HCCL_ERROR("[CcuTempBroadcastMesh2D] Rank[%d], Invalid IODieNum[%zu].", myRank_, tempVTopo_.size());
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_INFO(
        "[CcuTempBroadcastMesh2D] Rank[%d] requiredQueNum[%u] VtopoSize[%u], VtopoSize0[%u] VtopoSize1[%u].", myRank_,
        tempResReq.queNum, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size());

    uint32_t myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            // find neighbors -> virtualRank
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
            RankId neighborRank = tempVTopo_[dim][neighborAlgRank];
            HCCL_INFO(
                "[CollAlgFactory] [CcuTempBroadcastMesh2D] Rank[%d], Dim[%u], NeighborRank[%d].", myRank_, dim,
                neighborRank);

            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuTempBroadcastMesh2D::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);

    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastMesh2D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)sliceInfoVec;
    opMode_ = tempFuncs.opMode;
    buffInfo_ = buffInfo;

    // 分别记录两个Die上的link，构造rankGroup
    for (auto pair : tempLinks) {
        if (pair.second.size() == 0 || pair.second[0].GetHop() != 1) { // ESL环境上暂只有直连链路
            THROW<InvalidParamsException>(StringFormat(
                "[CcuTempBroadcastMesh2D] Rank[%d]--Peer[%d], InvalidHop[%u].", myRank_, pair.first,
                pair.second[0].GetHop()));
        }
        if ((pair.first / dimSize_[0] == myRank_ / dimSize_[0]) && pair.second[0].GetHop() == 1) {
            HCCL_INFO("[CcuTempBroadcastMesh2D][Run] Rank[%d] insert link to Rank[%d] in linksX", myRank_, pair.first);
            linksX_.emplace_back(pair.second[0]);
        } else if ((pair.first % dimSize_[0] == myRank_ % dimSize_[0]) && pair.second[0].GetHop() == 1) {
            HCCL_INFO("[CcuTempBroadcastMesh2D][Run] Rank[%d] insert link to Rank[%d] in linksY", myRank_, pair.first);
            linksY_.emplace_back(pair.second[0]);
        } else {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuTempBroadcastMesh2D] Rank[%d], Unexpected peerRank[%d] in tempLinks.", myRank_, pair.first));
        }
    }

    RankGroup rankGroupX;
    RankGroup rankGroupY;
    AddRanksToGroup(tempVTopo_, rankGroupX, rankGroupY);

    uint64_t inputAddr;
    if (opMode_ == OpMode::OPBASE) {
        if (tempFuncs.isForepart) {
            // 从 UserIn 获取数据
            inputAddr = BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType())
                        + tempFuncs.usrData.usrInSlices[0].GetOffset();
        } else {
            // 从 inBuff 获取数据
            inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
        }
    } else {
        // 图模式没有 tempFuncs.usrData，直接通过 buffInfo_ 获取输入输出地址
        inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
    }
    uint64_t sliceSize = tempFuncs.usrData.usrInSlices[0].GetSize(); // 获取本rank需要处理的数据量

    uint64_t token;
    CHK_RET(GetToken(op_, token));

    std::vector<uint32_t> dimId;
    dimId.emplace_back(myRank_ % dimSize_[0]);
    dimId.emplace_back(myRank_ / dimSize_[0]);

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t axisId = 0; axisId < 2; axisId++) { // 2D算法，需要执行两次
        // 计算每次编译的偏移量和数据量
        uint64_t xAxisSize = sliceSize * dimSize_[0] / (dimSize_[axisId] + dimSize_[1 - axisId]);
        uint64_t yAxisSize = sliceSize - xAxisSize;

        CcuInstructionBroadcastMesh2D ccuInsBroadcastMesh2D;
        ccuInsBroadcastMesh2D.Init(
            dimSize_, static_cast<uint32_t>(myRank_), inputAddr, axisId, sliceSize, xAxisSize, yAxisSize, token, op_,
            tempVTopo_);

        HCCL_INFO(
            "[CcuTempBroadcastMesh2D] Run Init: dimSize0[%llu], dimSize1[%llu], myRank_[%d],"
            "inputAddr[%llu], sliceSize[%llu], xAxisSize[%llu], yAxisSize[%llu]",
            dimSize_[0], dimSize_[1], myRank_, inputAddr, sliceSize, xAxisSize, yAxisSize);

        ccuInsBroadcastMesh2D.SetLinks(axisId == 0 ? linksX_ : linksY_);
        ccuInsBroadcastMesh2D.SetRankGroup(axisId == 0 ? rankGroupX : rankGroupY);
        u32 ckeNum = 5;
        ccuInsBroadcastMesh2D.SetCntCkeNum(ckeNum); // 每个transport用4个CKE
        ccuInsBroadcastMesh2D.Describe();
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionBroadcastMesh2D>(ccuInsBroadcastMesh2D)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
