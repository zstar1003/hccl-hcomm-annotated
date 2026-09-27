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
#include "ccu_temp_all_to_all_mesh2d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_to_all_mesh2d.h"
#include "ccu_ins_group.h"

namespace Hccl {
constexpr u32 MESH_2D_DIMENSION_NUM = 2;

static CcuInstRegister<CcuContextAlltoAllMesh2D> registerAlltoAll(CcuInstType::CCU_ALLTOALL_MESH_2D_DIRECT);

CcuTempAlltoAllMesh2D::CcuTempAlltoAllMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    // 填充框内的维度大小
    if (tempVTopo_.size() != MESH_2D_DIMENSION_NUM || tempVTopo_[0].size() <= 1
        || tempVTopo_[1].size() <= 1) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempAlltoAllMesh2D] Rank[%d], Invalid tempVTopo "
            "Size[%u] or Invalid tempVTopo[0] size [%u] or tempVTopo[1] size [%u].",
            myRank_, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size()));
    }
    dimSize_.emplace_back(tempVTopo[0].size());
    dimSize_.emplace_back(tempVTopo[1].size());
}

CcuTempAlltoAllMesh2D::~CcuTempAlltoAllMesh2D() {}

void CcuTempAlltoAllMesh2D::SetA2ASendRecvInfo(const A2ASendRecvInfo& sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
    return;
}

HcclResult CcuTempAlltoAllMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1; // 只申请一个insQue，填充一个insGroup，由框架将其中的ins放在多个stream上
    tempResReq.streamNum = tempResReq.queNum + 1; // 多申请一个 stream 给 ccuInsGroup
    uint32_t dieNum = tempVTopo_.size();
    if (dieNum != 2) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        THROW<InvalidParamsException>(
            StringFormat("[CcuTempAlltoAllMesh2D] Rank[%d], Invalid IODieNum[%u].", myRank_, dieNum));
    }
    HCCL_INFO(
        "[CcuTempAlltoAllMesh2D] Rank[%d] requiredQueNum[%u] VtopoSize[%u], VtopoSize0[%u] VtopoSize1[%u].", myRank_,
        tempResReq.queNum, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size());

    uint32_t myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            // find neighbors -> virtualRank
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
            RankId neighborRank = tempVTopo_[dim][neighborAlgRank];
            HCCL_INFO(
                "[CollAlgFactory] [CcuTempAlltoAllMesh2D] Rank[%d], Dim[%u], NeighborRank[%d].", myRank_, dim,
                neighborRank);

            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllMesh2D::FillLinks(const ResLinks& tempLinks)
{
    for (auto pair : tempLinks) {
        if (pair.second.size() == 0) { // ESL环境上暂只有直连链路
            THROW<InvalidParamsException>(
                StringFormat("[CcuTempAlltoAllMesh2D] Rank[%d]--Peer[%d].", myRank_, pair.first));
        }
        if (pair.first / dimSize_[0] == myRank_ / dimSize_[0]) {
            HCCL_INFO("[CcuTempAlltoAllMesh2D][Run] Rank[%d] insert link to Rank[%d] in linksX", myRank_, pair.first);
            linksX_.emplace_back(pair.second[0]);
        } else if (pair.first % dimSize_[0] == myRank_ % dimSize_[0]) {
            HCCL_INFO("[CcuTempAlltoAllMesh2D][Run] Rank[%d] insert link to Rank[%d] in linksY", myRank_, pair.first);
            linksY_.emplace_back(pair.second[0]);
        } else {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuTempAlltoAllMesh2D] Rank[%d], Unexpected peerRank[%d] in tempLinks.", myRank_, pair.first));
        }
    }
    for (auto& peer : tempVTopo_[0]) {
        rankGroupX_.AddRank(peer);
    }
    for (auto& peer : tempVTopo_[1]) {
        rankGroupY_.AddRank(peer);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllMesh2D::RunOneStep(
    uint64_t sendRecvSize, uint64_t maxTransportSize, uint32_t sendRecvTimes, uint32_t step,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[CcuTempAlltoAllMesh2D][Run] Rank[%d], Step[%u], sendRecvTimes[%u].", myRank_, step, sendRecvTimes);

    uint64_t inputAddr = op_.inputMem->GetAddr();
    uint64_t outputAddr = op_.outputMem->GetAddr();
    uint64_t scratchAddr = op_.scratchMem->GetAddr();
    uint32_t typeSize = DataTypeSizeGet(op_.all2AllDataDes.sendType);
    uint64_t sendStrideSize = 0 * typeSize;
    uint64_t recvStrideSize = 0 * typeSize;
    uint64_t token;
    CHK_RET(GetToken(op_, token));

    uint64_t stepSize = (step == sendRecvTimes - 1) ? (sendRecvSize - step * maxTransportSize) : maxTransportSize;
    uint64_t aSize
        = static_cast<uint64_t>((stepSize / 2) / typeSize) * typeSize; // 暂定X和Y方向每轮传输大小一致，按照count对齐
    uint64_t bSize = stepSize - aSize;
    uint64_t baseOffset = step * maxTransportSize; // 已传输完成的数据量
    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t axisId = 0; axisId < 2; axisId++) { // 2D算法，需要执行两次
        HCCL_INFO(
            "[CcuTempAlltoAllMesh2D][Run] Rank[%d], Step[%u], axisId[%u], aSize[%llu], bSize[%llu], baseOffset[%llu].",
            myRank_, step, axisId, aSize, bSize, baseOffset);

        CcuInstructionAlltoAllMesh2D ins = CcuInstructionAlltoAllMesh2D(op_, dimSize_, tempVTopo_);
        ins.Init(
            myRank_, inputAddr, outputAddr, scratchAddr, axisId, sendStrideSize, recvStrideSize,
            localSendRecvInfo_.sendLength[0], aSize, bSize, baseOffset, token);
        ins.SetLinks(axisId == 0 ? linksX_ : linksY_);
        ins.SetRankGroup(axisId == 0 ? rankGroupX_ : rankGroupY_);
        ins.SetCntCkeNum(4); // 每个transport用4个CKE
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionAlltoAllMesh2D>(ins)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllMesh2D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    // 分别记录两个Die上的link，构造rankGroup
    (void)tempFuncs;
    (void)sliceInfoVec;
    (void)buffInfo;
    CHK_RET(FillLinks(tempLinks));

    // scratch分两组，每组rankSize份，放一个分片，按照传输大小限制与buffer大小限制分多轮执行算子
    HCCL_INFO(
        "[CcuTempAlltoAllMesh2D] dataType[%s] sendType[%s]", op_.dataType.Describe().c_str(),
        op_.all2AllDataDes.sendType.Describe().c_str());
    uint32_t typeSize = DataTypeSizeGet(op_.all2AllDataDes.sendType);
    uint64_t sendRecvSize = localSendRecvInfo_.sendLength[0];
    uint64_t blockBufferSize
        = static_cast<uint64_t>((scratchBufferSize_ / tempRankSize_ / 2) / typeSize) * typeSize; // 分2组buffer
    uint64_t maxTransportSize = min(min(CalcLGMaxTransSize(), UB_MAX_TRANS_SIZE), blockBufferSize);
    uint32_t sendRecvTimes = (sendRecvSize + maxTransportSize - 1) / maxTransportSize;
    HCCL_INFO(
        "[CollAlgFactory][Run] Rank[%d], blockBufferSize[%llu], sendRecvTimes[%u].", myRank_, blockBufferSize,
        sendRecvTimes);

    uint64_t token;
    CHK_RET(GetToken(op_, token));
    HCCL_INFO(
        "[CcuTempAlltoAllMesh2D] Rank[%d], input[%llu], output[%llu], scratch[%llu], sendStride[%llu], "
        "recvStride[%llu].",
        myRank_, op_.inputMem->GetAddr(), op_.outputMem->GetAddr(), op_.scratchMem->GetAddr(), 0, 0);

    if (tempInsQues.size() == 0) {
        HCCL_ERROR("[CcuTempAlltoAllMesh2D][Run] invalid tempInsQues size is [%zu].", tempInsQues.size());
        return HcclResult::HCCL_E_PARA;
    }
    for (uint32_t step = 0; step < sendRecvTimes; step++) { // 零数据量时会跳过
        if (tempRankSize_ == 1) {
            // alltoall算子的单P场景单独处理
            DataSlice usrInSlice = DataSlice(BufferType::INPUT, 0, sendRecvSize);
            DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, 0, sendRecvSize);
            std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(usrInSlice, usrOutSlice);
            tempInsQues[0]->Append(std::move(insLocalCopy));
            HCCL_INFO("[CcuTempAlltoAllMesh2D] rankSize = 1, use InsLocalCopy for sliceSize[%llu].", sendRecvSize);
            break;
        }
        CHK_RET(RunOneStep(sendRecvSize, maxTransportSize, sendRecvTimes, step, tempInsQues));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllMesh2D::GetScratchBufferInfo(const uint64_t scratchBufferSize, DataType dataType)
{
    scratchBufferSize_ = scratchBufferSize;
    dataType_ = dataType;
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
