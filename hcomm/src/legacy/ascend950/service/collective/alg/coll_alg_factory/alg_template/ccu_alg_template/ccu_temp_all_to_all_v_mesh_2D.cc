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
#include "ccu_temp_all_to_all_v_mesh_2D.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_to_all_v_mesh2d.h"
#include "ccu_ins_group.h"

namespace Hccl {
constexpr u32 MESH_2D_DIMENSION_NUM = 2;

static CcuInstRegister<CcuContextAllToAllVMesh2D> g_registerAlltoAllV(CcuInstType::CCU_ALLTOALLV_MESH_2D_DIRECT);

CcuTempAlltoAllVMesh2D::CcuTempAlltoAllVMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    // 填充框内的维度大小
    if (tempVTopo_.size() != MESH_2D_DIMENSION_NUM || tempVTopo_[0].size() <= 1
        || tempVTopo_[1].size() <= 1) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempAlltoAllVMesh2D] Rank[%d], Invalid tempVTopo "
            "Size[%u] or Invalid tempVTopo[0] size [%u] or tempVTopo[1] size [%u].",
            myRank_, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size()));
    }
    dimSize_.emplace_back(tempVTopo[0].size());
    dimSize_.emplace_back(tempVTopo[1].size());
}

CcuTempAlltoAllVMesh2D::~CcuTempAlltoAllVMesh2D() {}

void CcuTempAlltoAllVMesh2D::SetA2ASendRecvInfo(const A2ASendRecvInfo& sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
    return;
}

HcclResult CcuTempAlltoAllVMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1; // 只申请一个insQue，填充一个insGroup，由框架将其中的ins放在多个stream上
    tempResReq.streamNum = tempResReq.queNum + 1; // 多申请一个 stream 给 ccuInsGroup
    uint32_t dieNum = tempVTopo_.size();
    if (dieNum != 2) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        THROW<InvalidParamsException>(
            StringFormat("[CcuTempAlltoAllVMesh2D] Rank[%d], Invalid IODieNum[%u].", myRank_, dieNum));
    }
    HCCL_INFO(
        "[CcuTempAlltoAllVMesh2D] Rank[%d] requiredQueNum[%u] VtopoSize[%u], VtopoSize0[%u] VtopoSize1[%u].", myRank_,
        tempResReq.queNum, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size());

    uint32_t myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            // find neighbors -> virtualRank
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
            RankId neighborRank = tempVTopo_[dim][neighborAlgRank];
            HCCL_INFO(
                "[CollAlgFactory] [CcuTempAlltoAllVMesh2D] Rank[%d], Dim[%u], NeighborRank[%d].", myRank_, dim,
                neighborRank);

            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempAlltoAllVMesh2D::CalcSendRecvNumSubStep(uint64_t sliceSize)
{
    sendNumSubStep_.clear();
    recvNumSubStep_.clear();
    uint64_t numSubStep = 0;
    if (localSendRecvInfo_.sendLength.size() != localSendRecvInfo_.recvLength.size()) {
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempAlltoAllVMesh2D][CalcSendRecvNumSubStep] Rank[%d] sendLength size[%u] is not equal to"
            "recvLength size[%u]",
            myRank_, localSendRecvInfo_.sendLength.size(), localSendRecvInfo_.recvLength.size()));
    }
    u32 rankSize = localSendRecvInfo_.sendLength.size();
    if (rankSize == 0 || sliceSize == 0) {
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempAlltoAllVMesh2D][CalcSendRecvNumSubStep] Invalid rankSize [%u] or invalid slicesize[%u].",
            rankSize, sliceSize));
    }
    for (u32 destRank = 0; destRank < rankSize; destRank++) {
        uint64_t currRankSendSubStep = ((localSendRecvInfo_.sendLength[destRank] + sliceSize - 1) / sliceSize);
        sendNumSubStep_[destRank] = currRankSendSubStep;

        uint64_t currRankRecvSubStep = ((localSendRecvInfo_.recvLength[destRank] + sliceSize - 1) / sliceSize);
        recvNumSubStep_[destRank] = currRankRecvSubStep;
        HCCL_INFO(
            "[CcuTempAlltoAllVMesh2D][CalcNumSubStep] myRank [%d] currRankSendSubStep[%llu] "
            "currRankRecvSubStep[%llu]",
            myRank_, currRankSendSubStep, currRankRecvSubStep);
        numSubStep = std::max(numSubStep, std::max(currRankSendSubStep, currRankRecvSubStep));
    }
    HCCL_INFO("[CcuTempAlltoAllVMesh1D][CalcNumSubStep] myRank [%d] max communication step[%u]", myRank_, numSubStep);
    return numSubStep;
}

HcclResult CcuTempAlltoAllVMesh2D::FillLinks(const ResLinks& tempLinks)
{
    for (auto pair : tempLinks) {
        if (pair.second.size() == 0) { // ESL环境上暂只有直连链路
            THROW<InvalidParamsException>(
                StringFormat("[CcuTempAlltoAllVMesh2D] Rank[%d]--Peer[%d].", myRank_, pair.first));
        }
        if (pair.first / dimSize_[0] == myRank_ / dimSize_[0]) {
            HCCL_INFO("[CcuTempAlltoAllVMesh2D][Run] Rank[%d] insert link to Rank[%d] in linksX", myRank_, pair.first);
            linksX_.emplace_back(pair.second[0]);
        } else if (pair.first % dimSize_[0] == myRank_ % dimSize_[0]) {
            HCCL_INFO("[CcuTempAlltoAllVMesh2D][Run] Rank[%d] insert link to Rank[%d] in linksY", myRank_, pair.first);
            linksY_.emplace_back(pair.second[0]);
        } else {
            THROW<InvalidParamsException>(StringFormat(
                "[CcuTempAlltoAllVMesh2D] Rank[%d], Unexpected peerRank[%d] in tempLinks.", myRank_, pair.first));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh2D::FillRankGroup()
{
    for (auto& peer : tempVTopo_[0]) {
        rankGroupX_.AddRank(peer);
    }
    for (auto& peer : tempVTopo_[1]) {
        rankGroupY_.AddRank(peer);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh2D::CalcSliceSize(uint32_t sendRecvTime, uint64_t maxTransportSize)
{
    sendSliceSize_.resize(tempRankSize_);
    recvSliceSize_.resize(tempRankSize_);
    for (u32 j = 0; j < tempRankSize_; j++) {
        if ((sendRecvTime + 1) < sendNumSubStep_[j]) {
            sendSliceSize_[j] = maxTransportSize;
        } else if ((sendRecvTime + 1) == sendNumSubStep_[j]) {
            sendSliceSize_[j] = localSendRecvInfo_.sendLength[j] - sliceBias_;
        } else {
            sendSliceSize_[j] = 0;
        }
    }
    for (u32 j = 0; j < tempRankSize_; j++) {
        if ((sendRecvTime + 1) < recvNumSubStep_[j]) {
            recvSliceSize_[j] = maxTransportSize;
        } else if ((sendRecvTime + 1) == recvNumSubStep_[j]) {
            recvSliceSize_[j] = localSendRecvInfo_.recvLength[j] - sliceBias_;
        } else {
            recvSliceSize_[j] = 0;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh2D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    if (tempVTopo_.size() == 0 || tempInsQues.size() == 0) {
        THROW<NullPtrException>(StringFormat(
            "[CcuTempAlltoAllVMesh2D][Run] invalid tempVTopo size is [%u] or invalid tempInsQues size is [%u].",
            tempVTopo_.size(), tempInsQues.size()));
    }
    // 分别记录两个Die上的link，构造rankGroup
    (void)tempFuncs;
    (void)sliceInfoVec;
    (void)buffInfo;
    CHK_RET(FillLinks(tempLinks));
    CHK_RET(FillRankGroup());

    // scratch分两组，每组rankSize份，放一个分片，按照传输大小限制与buffer大小限制分多轮执行算子
    HCCL_INFO("[CcuTempAlltoAllVMesh2D] dataType[%d] sendType[%d]", op_.dataType, op_.all2AllVDataDes.sendType);
    if (!op_.scratchMem) {
        HCCL_ERROR("[CcuTempAlltoAllVMesh2D][Run] Rank[%d] inputmem or outputmem or scratchMem is null", myRank_);
        return HcclResult::HCCL_E_PTR;
    }

    uint64_t inputAddr = op_.inputMem == nullptr ? 0 : op_.inputMem->GetAddr();
    uint64_t outputAddr = op_.outputMem == nullptr ? 0 : op_.outputMem->GetAddr();
    uint64_t scratchAddr = op_.scratchMem->GetAddr();
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint32_t typeSize = DataTypeSizeGet(op_.all2AllVDataDes.sendType);
    // scratchmem需要切成blockSize大小的格子,发送的数据块blockBufferSize是scratchmem每一格大小的两倍
    uint32_t blockSize = (tempVTopo_[0].size() - 1) * (tempVTopo_[1].size() - 1) * 2;
    uint64_t blockBufferSize = static_cast<uint64_t>((scratchBufferSize_ / blockSize) / typeSize) * typeSize * 2;

    HCCL_INFO(
        "[CcuTempAlltoAllVMesh2D] Rank[%d], input[%llu], output[%llu], scratch[%llu], blockSize[%llu],"
        "blockBufferSize[%llu].",
        myRank_, inputAddr, outputAddr, scratchAddr, blockSize, blockBufferSize);

    if (tempRankSize_ == 1) {
        // alltoallv算子的单P场景单独处理
        DataSlice usrInSlice = DataSlice(BufferType::INPUT, 0, localSendRecvInfo_.sendLength[0]);
        DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, 0, localSendRecvInfo_.sendLength[0]);
        std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(usrInSlice, usrOutSlice);
        tempInsQues[0]->Append(std::move(insLocalCopy));
        HCCL_INFO(
            "[CcuTempAlltoAllVMesh2D] rankSize = 1, use InsLocalCopy for sliceSize[%llu].",
            localSendRecvInfo_.sendLength[0]);
    }

    uint64_t scratchSliceSize = blockBufferSize / 2;
    uint64_t scratchSliceBias = scratchSliceSize * (tempVTopo_[0].size() - 1) * (tempVTopo_[1].size() - 1);
    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t axisId = 0; axisId < 2; axisId++) { // 2D算法，需要执行两次
        CcuInstructionAllToAllVMesh2D ins = CcuInstructionAllToAllVMesh2D(op_, dimSize_, tempVTopo_);
        ins.Init(
            myRank_, axisId, inputAddr, outputAddr, scratchAddr, token, scratchSliceSize, scratchSliceBias,
            localSendRecvInfo_);
        ins.SetLinks(axisId == 0 ? linksX_ : linksY_);
        ins.SetRankGroup(axisId == 0 ? rankGroupX_ : rankGroupY_);
        u32 ckeNum = 5 + 2 * std::max(dimSize_[0], dimSize_[1]);
        ins.SetCntCkeNum(ckeNum);
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionAllToAllVMesh2D>(ins)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh2D::GetScratchBufferInfo(const uint64_t scratchBufferSize, DataType dataType)
{
    scratchBufferSize_ = scratchBufferSize;
    dataType_ = dataType;
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
