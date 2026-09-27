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

#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_reduce_nhr1d_mem2mem.h"
#include "ccu_temp_reduce_nhr_1D_mem2mem.h"
#include "ccu_ins_group.h"

namespace Hccl {

static CcuInstRegister<CcuContextReduceNHR1DMem2mem> g_registrarReduce(CcuInstType::CCU_REDUCE_NHR_1D_MEM2MEM);

CcuTempReduceNHRMem2Mem1D::CcuTempReduceNHRMem2Mem1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempReduceNHRMem2Mem1D::~CcuTempReduceNHRMem2Mem1D() {}

HcclResult CcuTempReduceNHRMem2Mem1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_DEBUG("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    u32 linkNum = 1;
    linkNumBtwPeers_ = linkNum;
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceNHRMem2Mem1D::CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    AllignInfo allignInfo;
    allignInfo.enableAllign = false;
    allignInfo.dataType = dataType_;
    CHK_RET(CalcSliceInfoAllReduce(allignInfo, tempRankSize_, dataSize, sliceInfoVec));
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempReduceNHRMem2Mem1D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

uint64_t CcuTempReduceNHRMem2Mem1D::GetMaxSliceSize() const { return UB_MAX_DATA_SIZE; }

uint32_t CcuTempReduceNHRMem2Mem1D::virtRankId2RankId(const uint32_t virtRankId)
{
    for (auto iter = tempVirtRankMap_.begin(); iter != tempVirtRankMap_.end(); iter++) {
        if (iter->second == virtRankId) {
            return iter->first;
        }
    }
    return 0;
}

HcclResult CcuTempReduceNHRMem2Mem1D::GenExtIns(
    const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[CcuTempReduceNHR][GenExtIns] ReduceNHR begin: rank[%d] start", myRank_);
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[CcuTempReduceNHR] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    rootId_ = op_.root;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint32_t axisSize = tempLinks.begin()->second.size();

    uint32_t myVirtRankId = tempVirtRankMap_[myRank_];
    uint64_t inputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.inBuffType) + tempAlgParams.buffInfo.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.outBuffType) + tempAlgParams.buffInfo.outBuffBaseOff;
    uint64_t DataCount = (tempAlgParams.sliceSize / DataTypeSizeGet(dataType_));
    uint64_t die0Size = DataCount / axisSize * DataTypeSizeGet(dataType_);
    uint64_t die1Size = tempAlgParams.sliceSize - die0Size;
    uint64_t repeatNum = tempAlgParams.repeatNum;
    uint64_t token;
    CHK_RET(GetToken(op_, token));

    if (DataCount == 0) {
        HCCL_INFO("[CcuTempReduceNHRMem2Mem1D] DataCount == 0, Template Run Ends.");
        return HCCL_SUCCESS;
    }
    if (axisSize > 1 && die1Size == 0) {
        axisSize = 1;
    }

    RankSliceInfo die0SliceInfoVec;
    CHK_RET(CalcSlice(die0Size, die0SliceInfoVec));
    RankSliceInfo die1SliceInfoVec;
    CHK_RET(CalcSlice(die1Size, die1SliceInfoVec));

    HCCL_INFO(
        "[CcuTempReduceNHRMem2Mem1D] dimSize[%llu], die0Size[%llu], die1Size[%llu], inputAddr[%llu],"
        "outputAddr[%llu], repeatNum[%llu], die0Slicesize[%llu], die1Slicesize[%llu], die0LastSlicesize[%llu],"
        "die1LastSlicesize[%llu]",
        dimSize[0], die0Size, die1Size, inputAddr, outputAddr, repeatNum, die0SliceInfoVec[0][0].size,
        die1SliceInfoVec[0][0].size, die0SliceInfoVec[tempRankSize_ - 1][0].size,
        die1SliceInfoVec[tempRankSize_ - 1][0].size);

    std::vector<LinkData> linksDie0;
    std::vector<LinkData> linksDie1;
    RankGroup reduceRankGroup;
    std::map<u32, u32> indexMap;
    std::vector<NHRStepInfo> stepInfoVector;
    u32 nSteps = GetNHRStepNum(tempRankSize_) * 2; // 分为RS和AG两次NHR

    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
        if (indexMap.count(stepInfo.fromRank) == 0) {
            u32 fromRankIdx = virtRankId2RankId(stepInfo.fromRank);
            indexMap[stepInfo.fromRank] = linksDie0.size();
            linksDie0.push_back(tempLinks.at(fromRankIdx)[0]);
            if (axisSize > 1) {
                linksDie1.push_back(tempLinks.at(fromRankIdx)[1]);
            }
            reduceRankGroup.AddRank(fromRankIdx);
        }
        if (indexMap.count(stepInfo.toRank) == 0) {
            u32 toRankIdx = virtRankId2RankId(stepInfo.toRank);
            indexMap[stepInfo.toRank] = linksDie0.size();
            linksDie0.push_back(tempLinks.at(toRankIdx)[0]);
            if (axisSize > 1) {
                linksDie1.push_back(tempLinks.at(toRankIdx)[1]);
            }
            reduceRankGroup.AddRank(toRankIdx);
        }
    }
    reduceRankGroup.AddRank(myRank_);

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t axisId = 0; axisId < axisSize; axisId++) { // 2个die上各一个mission
        CcuInstructionReduceNHR1D ccuInstruction;
        uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;
        ccuInstruction.Init(
            myVirtRankId, rootId_, inputAddr, outputAddr, axisId, axisSize, die0Size, die1Size,
            die0SliceInfoVec[0][0].size, die1SliceInfoVec[0][0].size, die0SliceInfoVec[tempRankSize_ - 1][0].size,
            die1SliceInfoVec[tempRankSize_ - 1][0].size, stepInfoVector, indexMap, token, isInputOutputEqual, op_,
            tempVTopo_);
        ccuInstruction.SetLinks(axisId == 0 ? linksDie0 : linksDie1);
        ccuInstruction.SetRankGroup(reduceRankGroup);
        ccuInstruction.SetCntCkeNum(5); // 每个transport用5个CKE
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionReduceNHR1D>(ccuInstruction)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流
    HCCL_INFO("[CcuTempReduceNHRMem2Mem1D] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceNHRMem2Mem1D::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo)
{
    u32 nStepsNHR = nSteps / 2;
    u32 realStep = step;
    if (realStep < nStepsNHR) {
        CHK_RET(GetReduceScatterStepInfo(realStep, stepInfo));
    } else {
        realStep = step % nStepsNHR;
        CHK_RET(GetAllGatherStepInfo(realStep, nStepsNHR, stepInfo));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceNHRMem2Mem1D::GetReduceScatterStepInfo(u32 step, NHRStepInfo& stepInfo)
{
    u32 virtRankIdx = tempVirtRankMap_[myRank_];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    // ReduceNHR计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (virtRankIdx + tempRankSize_ - deltaRank) % tempRankSize_;
    u32 recvFrom = (virtRankIdx + deltaRank) % tempRankSize_;

    // ReduceNHR数据份数和数据编号增量
    u32 nSlices = (tempRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 rxSliceIdx = virtRankIdx;
    u32 txSliceIdx = (virtRankIdx - (1 << step) + tempRankSize_) % tempRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG(
            "[ReduceNHR][GetReduceScatterStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceNHRMem2Mem1D::GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo)
{
    u32 virtRankIdx = tempVirtRankMap_[myRank_];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    // ReduceNHR计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (virtRankIdx + tempRankSize_ - deltaRank) % tempRankSize_;
    u32 sendTo = (virtRankIdx + deltaRank) % tempRankSize_;

    // ReduceNHR数据份数和数据编号增量
    u32 nSlices = (tempRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = virtRankIdx;
    u32 rxSliceIdx = (virtRankIdx - (1 << (nSteps - 1 - step)) + tempRankSize_) % tempRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[ReduceNHR][GetAllGatherStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
