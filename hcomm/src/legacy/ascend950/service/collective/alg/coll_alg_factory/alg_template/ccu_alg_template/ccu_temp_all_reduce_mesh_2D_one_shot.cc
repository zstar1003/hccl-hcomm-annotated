/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_temp_all_reduce_mesh_2D_one_shot.h"

#include <ios>
#include <iostream>

#include "log.h"

#include "alg_data_trans_wrapper.h"

#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_ins_group.h"

#include "ccu_instruction_all_reduce_mesh2d_one_shot.h"
#include "ccu_context_all_reduce_mesh2d_one_shot.h"

namespace Hccl {
constexpr u32 MESH_2D_DIMENSION_NUM = 2;
static CcuInstRegister<CcuContextAllReduceMesh2DOneShot>
    g_registrarCcuAllReduce2DOneShot(CcuInstType::CCU_ALL_REDUCE_MESH_2D_ONE_SHOT_DIRECT);

CcuTempAllReduceMesh2DOneShot::CcuTempAllReduceMesh2DOneShot(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{
    if (tempVTopo_.size() != MESH_2D_DIMENSION_NUM || tempVTopo_[0].size() <= 1
        || tempVTopo_[1].size() <= 1) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTempAllReduceMesh2DOneShot] Rank[%d], Invalid tempVTopo "
            "Size[%u] or Invalid tempVTopo[0] size %u or tempVTopo[1] size %u.",
            myRank_, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size()));
    }
    dimSize_.emplace_back(tempVTopo[0].size());
    dimSize_.emplace_back(tempVTopo[1].size());
}

CcuTempAllReduceMesh2DOneShot::~CcuTempAllReduceMesh2DOneShot() {}

void CcuTempAllReduceMesh2DOneShot::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult CcuTempAllReduceMesh2DOneShot::CalcSliceInfo(
    const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    // OneShot 算法不需要对数据进行切分，每个rank上的数据量即为整个数据量
    (void)allignInfo;
    SliceInfo basicSlice;
    basicSlice.offset = 0;
    basicSlice.size = dataSize;
    HCCL_INFO("[CcuTempAllReduceMesh2DOneShot] basicSlice.size[%u].", basicSlice.size);
    std::vector<SliceInfo> singleRankSliceInfoVector{basicSlice};
    sliceInfoVec.resize(tempRankSize_, singleRankSliceInfoVector);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh2DOneShot::CalcRes(AlgTempResReq& tempResReq)
{
    // 按照IODienum来确定stream数量，支持2D和2D的template
    tempResReq.queNum = 1; // 只申请一个insQue，填充一个insGroup，由框架将其中的ins放在多个stream上
    tempResReq.streamNum = tempResReq.queNum + 1; // 多申请一个 stream 给 ccuInsGroup
    uint32_t dieNum = tempVTopo_.size();
    if (dieNum != 2) { // concurrmesh的topoMatch返回的vTopo大小应当为2，对应X轴和Y轴的大小
        HCCL_ERROR("[CcuTempAllReduceMesh2DOneShot] Rank[%d], Invalid IODieNum[%zu].", myRank_, tempVTopo_.size());
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_INFO(
        "[CcuTempAllReduceMesh2DOneShot] Rank[%d] requiredQueNum[%u] VtopoSize[%u], VtopoSize0[%u] VtopoSize1[%u].",
        myRank_, tempResReq.queNum, tempVTopo_.size(), tempVTopo_[0].size(), tempVTopo_[1].size());

    uint32_t myAlgRank;
    for (u32 dim = 0; dim < tempVTopo_.size(); dim++) {
        CHK_RET(GetAlgRank(myRank_, tempVTopo_[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo_[dim].size() - 1; queIdx++) {
            // find neighbors -> virtualRank
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo_[dim].size());
            RankId neighborRank = tempVTopo_[dim][neighborAlgRank];
            HCCL_INFO(
                "[CcuTempAllReduceMesh2DOneShot] Rank[%d], Dim[%u], NeighborRank[%d].", myRank_, dim, neighborRank);
            // LinkNum
            tempResReq.links[neighborRank] = 1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh2DOneShot::GetBufferAddr(
    const TempFuncs& tempFuncs, uint64_t& inputAddr, uint64_t& outputAddr, uint64_t& scratchAddr)
{
    uint64_t inputBaseAddr;
    uint64_t outputBaseAddr;
    uint64_t scratchBaseAddr;
    uint64_t inputOffSet;
    uint64_t scratchOffSet;
    uint64_t outputOffSet;

    if (opMode_ == OpMode::OPBASE) {
        if (tempFuncs.isForepart) {
            inputBaseAddr = BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType());
            inputOffSet = tempFuncs.usrData.usrInSlices[0].GetOffset();
        } else {
            inputBaseAddr = BufferTypeToAddr(buffInfo_.inBuffType);
            inputOffSet = buffInfo_.inBuffBaseOff;
        }
        if (tempFuncs.isBottom) {
            outputBaseAddr = BufferTypeToAddr(tempFuncs.usrData.usrOutSlices[0].GetType());
            outputOffSet = tempFuncs.usrData.usrOutSlices[0].GetOffset();
        } else {
            outputBaseAddr = BufferTypeToAddr(buffInfo_.outBuffType);
            outputOffSet = buffInfo_.outBuffBaseOff;
        }
    } else {
        outputBaseAddr = BufferTypeToAddr(buffInfo_.outBuffType);
        outputOffSet = buffInfo_.outBuffBaseOff + tempFuncs.usrData.usrOutSlices[0].GetOffset();
        inputBaseAddr = BufferTypeToAddr(buffInfo_.inBuffType);
        inputOffSet = buffInfo_.inBuffBaseOff + tempFuncs.usrData.usrInSlices[0].GetOffset();
    }

    scratchBaseAddr = BufferTypeToAddr(buffInfo_.scratBuffType);
    scratchOffSet = buffInfo_.scratchBuffBaseOff;

    HCCL_INFO(
        "[GetBufferAddr] inputBaseAddr[%llu], inputOffSet[%llu], outputBaseAddr[%llu], outputOffSet[%llu], "
        "scratchBaseAddr[%llu], scratchOffSet[%llu]",
        inputBaseAddr, inputOffSet, outputBaseAddr, outputOffSet, scratchBaseAddr, scratchOffSet);

    inputAddr = inputBaseAddr + inputOffSet;
    outputAddr = outputBaseAddr + outputOffSet;
    scratchAddr = scratchBaseAddr + scratchOffSet;

    HCCL_INFO(
        "[GetBufferAddr] inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu]", inputAddr, outputAddr, scratchAddr);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh2DOneShot::PrepareLinks(const ResLinks& tempLinks)
{
    HCCL_INFO("[CcuTempAllReduceMesh2DOneShot] PrepareLinks Starts.");
    // 分别记录两个Die上的link，构造rankGroup
    for (auto pair : tempLinks) {
        if (pair.second.size() == 0 || pair.second[0].GetHop() != 1) { // ESL环境上暂只有直连链路
            HCCL_ERROR(
                "[CcuTempAllReduceMesh2DOneShot] Rank[%d]--Peer[%d], InvalidHop[%u].", myRank_, pair.first,
                pair.second[0].GetHop());
            return HcclResult::HCCL_E_PARA;
        }
        if ((pair.first / dimSize_[0] == myRank_ / dimSize_[0]) && pair.second[0].GetHop() == 1) {
            HCCL_INFO(
                "[CcuTempAllReduceMesh2DOneShot][Run] Rank[%d] insert link to Rank[%d] in linksX", myRank_, pair.first);
            linksX_.emplace_back(pair.second[0]);
        } else if ((pair.first % dimSize_[0] == myRank_ % dimSize_[0]) && pair.second[0].GetHop() == 1) {
            HCCL_INFO(
                "[CcuTempAllReduceMesh2DOneShot][Run] Rank[%d] insert link to Rank[%d] in linksY", myRank_, pair.first);
            linksY_.emplace_back(pair.second[0]);
        } else {
            HCCL_ERROR(
                "[CcuTempAllReduceMesh2DOneShot] Rank[%d], Unexpected peerRank[%d] in tempLinks.", myRank_, pair.first);
            return HcclResult::HCCL_E_PARA;
        }
    }
    HCCL_INFO(
        "[CcuTempAllReduceMesh2DOneShot] PrepareLinks Ends. linksX Size[%u], linksY Size[%u]", linksX_.size(),
        linksY_.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh2DOneShot::PrepareRankGroups()
{
    HCCL_INFO("[CcuTempAllReduceMesh2DOneShot] PrepareRankGroups Starts.");
    for (auto& peer : tempVTopo_[0]) {
        rankGroupX_.AddRank(peer);
    }
    for (auto& peer : tempVTopo_[1]) {
        rankGroupY_.AddRank(peer);
    }
    CHK_PRT_RET(
        rankGroupX_.GetRanks().size() <= 1 || rankGroupY_.GetRanks().size() <= 1,
        HCCL_ERROR(
            "[CcuTempAllReduceMesh2DOneShot][PrepareRankGroups] Rank[%d] RankGroupX size[%u] or RankGroupY size[%u] is "
            "not greater than 1. ",
            myRank_, rankGroupX_.GetRanks().size(), rankGroupY_.GetRanks().size()),
        HcclResult::HCCL_E_PARA);
    HCCL_INFO(
        "[CcuTempAllReduceMesh2DOneShot][PrepareRankGroups] RankGroupX size[%u], RankGroupY size[%u].",
        rankGroupX_.GetRanks().size(), rankGroupY_.GetRanks().size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh2DOneShot::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[CcuTempAllReduceMesh2DOneShot][Run] Template Run Starts.");
    opMode_ = tempFuncs.opMode;
    buffInfo_ = buffInfo;

    u32 xDimSize = dimSize_[0];
    u32 yDimSize = dimSize_[1];
    HCCL_INFO("[Run] xDimSize[%u], yDimSize[%u]", xDimSize, yDimSize);

    CHK_RET(PrepareLinks(tempLinks));
    CHK_RET(PrepareRankGroups());
    // 计算 buffer 地址信息
    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t scratchAddr;
    CHK_RET(GetBufferAddr(tempFuncs, inputAddr, outputAddr, scratchAddr));
    HCCL_INFO("[Run] inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu]", inputAddr, outputAddr, scratchAddr);

    uint64_t totalSliceSize = sliceInfoVec[myRank_][0].size; // 获取本rank需要处理的数据总量
    uint64_t totalSliceCount = totalSliceSize / DataTypeSizeGet(dataType_);
    uint64_t xSliceSize = (totalSliceCount / (xDimSize + yDimSize)) * xDimSize * DataTypeSizeGet(dataType_);
    uint64_t ySliceSize = totalSliceSize - xSliceSize;
    uint64_t xSliceOffset = 0;
    uint64_t ySliceOffset = xSliceSize;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    HCCL_INFO(
        "[Run] totalSliceSize[%llu], xSliceSize[%llu], ySliceSize[%llu], xSliceOffset[%llu], "
        "ySliceOffset[%llu]",
        totalSliceSize, xSliceSize, ySliceSize, xSliceOffset, ySliceOffset);

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t axisId = 0; axisId < 2; axisId++) { // 2D算法，需要下发 2 条通信指令
        CcuInstructionAllReduceMesh2DOneShot ccuInstruction;
        ccuInstruction.Init(
            dimSize_, myRank_, inputAddr, outputAddr, scratchAddr, axisId, xSliceSize, ySliceSize, xSliceOffset,
            ySliceOffset, token, op_, tempVTopo_);
        ccuInstruction.SetLinks(axisId == 0 ? linksX_ : linksY_);
        ccuInstruction.SetRankGroup(axisId == 0 ? rankGroupX_ : rankGroupY_);
        ccuInstruction.SetCntCkeNum(5); // 每个transport用5个CKE
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionAllReduceMesh2DOneShot>(ccuInstruction)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流
    HCCL_INFO("[CcuTempAllReduceMesh2DOneShot][Run] Template Run Ends.");
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
