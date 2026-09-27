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
#include "ccu_ins_group.h"
#include "ccu_context_broadcast_nhr1d_mem2mem.h"
#include "ccu_temp_broadcast_nhr_1D_mem2mem.h"

namespace Hccl {

static CcuInstRegister<CcuContextBroadcastNHRMem2Mem1D>
    g_registrarReduceScatter(CcuInstType::CCU_BROADCAST_NHR_1D_MEM2MEM);

CcuTempBroadcastNHRMem2Mem1D::CcuTempBroadcastNHRMem2Mem1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempBroadcastNHRMem2Mem1D::~CcuTempBroadcastNHRMem2Mem1D() {}

u32 CcuTempBroadcastNHRMem2Mem1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

HcclResult CcuTempBroadcastNHRMem2Mem1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    u32 linkNum = 1;
    linkNumBtwPeers_ = linkNum;
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempBroadcastNHRMem2Mem1D::GetMaxSliceSize() const { return UB_MAX_DATA_SIZE; }

uint32_t CcuTempBroadcastNHRMem2Mem1D::virtRankId2RankId(const uint32_t virtRankId)
{
    for (auto iter = tempVirtRankMap_.begin(); iter != tempVirtRankMap_.end(); iter++) {
        if (iter->second == virtRankId) {
            return iter->first;
        }
    }
    return 0;
}

HcclResult CcuTempBroadcastNHRMem2Mem1D::GenExtIns(
    const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempBroadcastNHRMem2Mem1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint32_t axisSize = tempLinks.begin()->second.size();
    uint32_t myVirtRankId = tempVirtRankMap_[myRank_];
    uint64_t DataCount = (tempAlgParams.sliceSize / DataTypeSizeGet(dataType_));
    uint64_t die0Size = DataCount / axisSize * DataTypeSizeGet(dataType_);
    uint64_t die1Size = tempAlgParams.sliceSize - die0Size;
    uint64_t inputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.inBuffType) + tempAlgParams.buffInfo.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.outBuffType) + tempAlgParams.buffInfo.outBuffBaseOff;
    uint64_t repeatNum = tempAlgParams.repeatNum;
    uint64_t die0SliceSize = die0Size / tempRankSize_;
    uint64_t die0LastSliceSize = die0Size % tempRankSize_ + die0SliceSize;
    uint64_t die1SliceSize = die1Size / tempRankSize_;
    uint64_t die1LastSliceSize = die1Size % tempRankSize_ + die1SliceSize;
    uint64_t token;
    CHK_RET(GetToken(op_, token));

    if (DataCount == 0) {
        HCCL_INFO("[CcuTempBroadcastNHRMem2Mem1D] DataCount == 0, Template Run Ends.");
        return HCCL_SUCCESS;
    }
    if (axisSize > 1 && die1Size == 0) {
        axisSize = 1;
    }

    HCCL_INFO(
        "[CcuTempBroadcastNHRMem2Mem1D] dimSize[%llu], die0Size[%llu], die1Size[%llu], inputAddr[%llu],"
        "outputAddr[%llu], repeatNum[%llu], die0SliceSize[%llu], die0LastSliceSize[%llu], die1SliceSize[%llu],"
        "die1LastSliceSize[%llu]",
        dimSize[0], die0Size, die1Size, inputAddr, outputAddr, repeatNum, die0SliceSize, die0LastSliceSize,
        die1SliceSize, die1LastSliceSize);

    std::vector<LinkData> linksDie0;
    std::vector<LinkData> linksDie1;
    RankGroup broadcastRankGroup;
    std::map<u32, u32> indexMap;
    std::vector<NHRStepInfo> stepInfoVector;
    u32 nSteps = GetNHRStepNum(tempRankSize_) * 2; // 分为Scatter和AG两次NHR

    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
        if (indexMap.count(stepInfo.fromRank) == 0 && stepInfo.rxSliceIdxs.size() != 0) {
            u32 fromRankIdx = virtRankId2RankId(stepInfo.fromRank);
            indexMap[stepInfo.fromRank] = linksDie0.size();
            linksDie0.push_back(tempLinks.at(fromRankIdx)[0]);
            if (axisSize > 1) {
                linksDie1.push_back(tempLinks.at(fromRankIdx)[1]);
            }
            broadcastRankGroup.AddRank(fromRankIdx);
        }
        if (indexMap.count(stepInfo.toRank) == 0 && stepInfo.txSliceIdxs.size() != 0) {
            u32 toRankIdx = virtRankId2RankId(stepInfo.toRank);
            indexMap[stepInfo.toRank] = linksDie0.size();
            linksDie0.push_back(tempLinks.at(toRankIdx)[0]);
            if (axisSize > 1) {
                linksDie1.push_back(tempLinks.at(toRankIdx)[1]);
            }
            broadcastRankGroup.AddRank(toRankIdx);
        }
    }
    broadcastRankGroup.AddRank(myRank_);

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t axisId = 0; axisId < axisSize; axisId++) { // 2个die上各一个mission
        CcuInstructionBroadcastNHRMem2Mem1D ccuInstruction;

        ccuInstruction.Init(
            myVirtRankId, inputAddr, outputAddr, axisId, axisSize, die0Size, die1Size, die0SliceSize, die1SliceSize,
            die0LastSliceSize, die1LastSliceSize, stepInfoVector, indexMap, token, op_, tempVTopo_);
        ccuInstruction.SetLinks(axisId == 0 ? linksDie0 : linksDie1);
        ccuInstruction.SetRankGroup(broadcastRankGroup);
        ccuInstruction.SetCntCkeNum(5); // 每个transport用5个CKE
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionBroadcastNHRMem2Mem1D>(ccuInstruction)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流
    HCCL_INFO("[CcuTempBroadcastNHRMem2Mem1D] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastNHRMem2Mem1D::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo)
{
    u32 nStepsNHR = nSteps / 2;
    u32 realStep = step;
    if (realStep < nStepsNHR) {
        CHK_RET(GetScatterStepInfo(realStep, nStepsNHR, stepInfo));
    } else {
        realStep = step % nStepsNHR;
        CHK_RET(GetAllGatherStepInfo(realStep, nStepsNHR, stepInfo));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastNHRMem2Mem1D::GetScatterStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo)
{
    u32 virtRankIdx = tempVirtRankMap_[myRank_];
    u32 rankSize = tempRankSize_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    uint32_t rootId = tempVirtRankMap_[rootId_];
    u32 deltaRoot = (rootId + rankSize - virtRankIdx) % rankSize;
    u32 deltaRankPair = 1 << step;

    // 数据份数和数据编号增量
    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    // 判断是否是2的幂
    u32 nRanks = 0; // 本步需要进行收/发的rank数
    bool isPerfect = (rankSize & (rankSize - 1)) == 0;
    if (!isPerfect && step == nSteps - 1) {
        nRanks = rankSize - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) { // 需要发
        u32 sendTo = (virtRankIdx + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.toRank = sendTo;
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom = (virtRankIdx + deltaRankPair) % rankSize;
        u32 rxSliceIdx = virtRankIdx;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = rxSliceIdx;
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.fromRank = recvFrom;
        stepInfo.nSlices = nSlices;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastNHRMem2Mem1D::GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo)
{
    u32 virtRankIdx = tempVirtRankMap_[myRank_];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    // BroadcastNHR计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (virtRankIdx + tempRankSize_ - deltaRank) % tempRankSize_;
    u32 sendTo = (virtRankIdx + deltaRank) % tempRankSize_;

    // BroadcastNHR数据份数和数据编号增量
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
            "[BroadcastNHR][GetAllGatherStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
