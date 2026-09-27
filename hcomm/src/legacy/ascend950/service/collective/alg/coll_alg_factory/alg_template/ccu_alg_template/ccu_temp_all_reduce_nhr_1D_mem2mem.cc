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
#include "ccu_context_all_reduce_nhr1d_mem2mem.h"
#include "ccu_temp_all_reduce_nhr_1D_mem2mem.h"
#include "ccu_ins_group.h"

namespace Hccl {

static CcuInstRegister<CcuContextAllReduceNHR1D> g_registrarAllReduce(CcuInstType::CCU_ALLREDUCE_NHR_1D_MEM2MEM);

CcuTempAllReduceNHRMem2Mem1D::CcuTempAllReduceNHRMem2Mem1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllReduceNHRMem2Mem1D::~CcuTempAllReduceNHRMem2Mem1D() {}

HcclResult CcuTempAllReduceNHRMem2Mem1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_DEBUG("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    u32 linkNum = 1;
    linkNumBtwPeers_ = linkNum;
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    AllignInfo allignInfo;
    allignInfo.enableAllign = false;
    allignInfo.dataType = dataType_;
    CHK_RET(CalcSliceInfoAllReduce(allignInfo, tempRankSize_, dataSize, sliceInfoVec));
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAllReduceNHRMem2Mem1D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

uint64_t CcuTempAllReduceNHRMem2Mem1D::GetMaxSliceSize() const { return UB_MAX_DATA_SIZE; }

uint32_t CcuTempAllReduceNHRMem2Mem1D::virtRankId2RankId(const uint32_t virtRankId)
{
    for (auto iter = tempVirtRankMap_.begin(); iter != tempVirtRankMap_.end(); iter++) {
        if (iter->second == virtRankId) {
            return iter->first;
        }
    }
    return 0;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::SplitDataFor2Dies(
    uint64_t dataCount, const ResLinks& tempLinks, uint64_t& die0Size, uint64_t& die1Size) const
{
    constexpr uint64_t MULTIPLIER = 4;

    if (dataCount <= tempRankSize_ * MULTIPLIER) { // 数据量极小，不划分die
        die0Size = 0;
        die1Size = dataCount * DataTypeSizeGet(dataType_);
        return HcclResult::HCCL_SUCCESS;
    }
    u8 die0PortGroupSize = 1;
    u8 die1PortGroupSize = 1;
    for (LinkData linkData : tempLinks.begin()->second) {
        if (linkData.GetPortGroupSize() == 0)
            continue;
        if (linkData.GetLocalDieId() == 0) {
            die0PortGroupSize = linkData.GetPortGroupSize();
        } else {
            die1PortGroupSize = linkData.GetPortGroupSize();
        }
    }

    die0Size = (dataCount * die0PortGroupSize / (die0PortGroupSize + die1PortGroupSize)) * DataTypeSizeGet(dataType_);
    die1Size = dataCount * DataTypeSizeGet(dataType_) - die0Size;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::ProcessNHRStepInfo(
    std::vector<NHRStepInfo>& stepInfoVector, RankGroup& rankGroup, std::map<u32, u32>& indexMap,
    std::vector<LinkData>& linksDie0, std::vector<LinkData>& linksDie1, const ResLinks& tempLinks, uint32_t axisSize)
{
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
            rankGroup.AddRank(fromRankIdx);
        }
        if (indexMap.count(stepInfo.toRank) == 0) {
            u32 toRankIdx = virtRankId2RankId(stepInfo.toRank);
            indexMap[stepInfo.toRank] = linksDie0.size();
            linksDie0.push_back(tempLinks.at(toRankIdx)[0]);
            if (axisSize > 1) {
                linksDie1.push_back(tempLinks.at(toRankIdx)[1]);
            }
            rankGroup.AddRank(toRankIdx);
        }
    }
    rankGroup.AddRank(myRank_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::GenExtIns(
    const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempAllReduceNHRMem2Mem1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    CcuInstructionAllReduceNHR1D ccuInsAllReduceNHR1D;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint32_t myVirtRankId = tempVirtRankMap_[myRank_];
    uint64_t dataCount = (tempAlgParams.sliceSize / DataTypeSizeGet(dataType_));
    if (dataCount == 0) {
        HCCL_INFO("[CcuTempAllReduceNHRMem2Mem1D] dataCount == 0, Template Run Ends.");
        return HCCL_SUCCESS;
    }
    uint32_t axisSize = tempLinks.begin()->second.size();

    uint64_t die0Size = 0;
    uint64_t die1Size = 0;
    if (axisSize == 1) {
        die0Size = tempAlgParams.sliceSize;
    } else {
        SplitDataFor2Dies(dataCount, tempLinks, die0Size, die1Size);
    }
    RankSliceInfo die0SliceInfoVec;
    RankSliceInfo die1SliceInfoVec;
    CHK_RET(CalcSlice(die0Size, die0SliceInfoVec));
    CHK_RET(CalcSlice(die1Size, die1SliceInfoVec));
    uint64_t inputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.inBuffType) + tempAlgParams.buffInfo.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.outBuffType) + tempAlgParams.buffInfo.outBuffBaseOff;
    uint64_t repeatNum = tempAlgParams.repeatNum;
    uint64_t token;
    CHK_RET(GetToken(op_, token));

    HCCL_INFO(
        "[CcuTempAllReduceNHRMem2Mem1D] dimSize[%llu], die0Size[%llu], die1Size[%llu], inputAddr[%llu],"
        "outputAddr[%llu], repeatNum[%llu], die0Slicesize[%llu], die1Slicesize[%llu], die0LastSlicesize[%llu],"
        "die1LastSlicesize[%llu]",
        dimSize[0], die0Size, die1Size, inputAddr, outputAddr, repeatNum, die0SliceInfoVec[0][0].size,
        die1SliceInfoVec[0][0].size, die0SliceInfoVec[tempRankSize_ - 1][0].size,
        die1SliceInfoVec[tempRankSize_ - 1][0].size);

    std::vector<LinkData> linksDie0;
    std::vector<LinkData> linksDie1;
    RankGroup rankGroup;
    std::map<u32, u32> indexMap;
    std::vector<NHRStepInfo> stepInfoVector;

    CHK_RET(ProcessNHRStepInfo(stepInfoVector, rankGroup, indexMap, linksDie0, linksDie1, tempLinks, axisSize));

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();

    for (uint32_t axisId = 0; axisId < axisSize; axisId++) { // 2个die上各一个mission
        if ((axisId == 0 && die0Size == 0) || (axisId == 1 && die1Size == 0)) {
            continue;
        }

        CcuInstructionAllReduceNHR1D ccuInstruction;
        uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;

        ccuInstruction.Init(
            myVirtRankId, inputAddr, outputAddr, axisId, axisSize, die0Size, die1Size, die0SliceInfoVec[0][0].size,
            die1SliceInfoVec[0][0].size, die0SliceInfoVec[tempRankSize_ - 1][0].size,
            die1SliceInfoVec[tempRankSize_ - 1][0].size, stepInfoVector, indexMap, token, isInputOutputEqual, op_,
            tempVTopo_);
        ccuInstruction.SetLinks(axisId == 0 ? linksDie0 : linksDie1);
        ccuInstruction.SetRankGroup(rankGroup);
        ccuInstruction.SetCntCkeNum(5); // 每个transport用5个CKE
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionAllReduceNHR1D>(ccuInstruction)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流
    HCCL_INFO("[CcuTempAllReduceNHRMem2Mem1D] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo)
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

HcclResult CcuTempAllReduceNHRMem2Mem1D::GetReduceScatterStepInfo(u32 step, NHRStepInfo& stepInfo)
{
    u32 virtRankIdx = tempVirtRankMap_[myRank_];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    // AllReduceNHR计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (virtRankIdx + tempRankSize_ - deltaRank) % tempRankSize_;
    u32 recvFrom = (virtRankIdx + deltaRank) % tempRankSize_;

    // AllReduceNHR数据份数和数据编号增量
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
            "[AllReduceNHR][GetReduceScatterStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceNHRMem2Mem1D::GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo)
{
    u32 virtRankIdx = tempVirtRankMap_[myRank_];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    // AllReduceNHR计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (virtRankIdx + tempRankSize_ - deltaRank) % tempRankSize_;
    u32 sendTo = (virtRankIdx + deltaRank) % tempRankSize_;

    // AllReduceNHR数据份数和数据编号增量
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

        HCCL_DEBUG(
            "[AllReduceNHR][GetAllGatherStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
