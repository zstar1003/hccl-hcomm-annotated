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
#include "ccu_context_scatter_nhr1d_mem2mem.h"
#include "ccu_temp_scatter_nhr_1D_mem2mem.h"
#include "ccu_ins_group.h"

namespace Hccl {

static CcuInstRegister<CcuContextScatterNHR1DMem2Mem> g_registrarScatter(CcuInstType::CCU_SCATTER_NHR_1D_MEM2MEM);

CcuTempScatterNHRMem2Mem1D::CcuTempScatterNHRMem2Mem1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempScatterNHRMem2Mem1D::~CcuTempScatterNHRMem2Mem1D() {}

HcclResult CcuTempScatterNHRMem2Mem1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    u32 linkNum = 1;
    linkNumBtwPeers_ = linkNum;
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempScatterNHRMem2Mem1D::GetMaxSliceSize() const { return UB_MAX_DATA_SIZE; }

uint32_t CcuTempScatterNHRMem2Mem1D::virtRankId2RankId(const uint32_t virtRankId)
{
    for (auto iter = tempVirtRankMap_.begin(); iter != tempVirtRankMap_.end(); iter++) {
        if (iter->second == virtRankId) {
            return iter->first;
        }
    }
    return 0;
}

u32 CcuTempScatterNHRMem2Mem1D::CalcScratchMultiple(BufferType input, BufferType output)
{
    (void)input;
    (void)output;
    return tempRankSize_;
}

HcclResult CcuTempScatterNHRMem2Mem1D::GenExtIns(
    const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuInstructionScatterNHR1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    uint64_t rootId = tempVirtRankMap_[rootId_];
    CcuInstructionScatterNHR1D ccuInsScatterNHR1D;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);
    DataType dataType_ = op_.dataType;
    uint32_t axisSize = tempLinks.begin()->second.size();
    uint32_t myVirtRankId = tempVirtRankMap_[myRank_];
    uint64_t sliceSize = tempAlgParams.sliceSize;
    uint64_t DataCount = (tempAlgParams.sliceSize / DataTypeSizeGet(dataType_));
    uint64_t die0Size = DataCount / axisSize * DataTypeSizeGet(dataType_);
    uint64_t die1Size = tempAlgParams.sliceSize - die0Size;
    uint64_t inputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.inBuffType) + tempAlgParams.buffInfo.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.outBuffType) + tempAlgParams.buffInfo.outBuffBaseOff;
    uint64_t isOutputScratch = (tempAlgParams.buffInfo.outBuffType == BufferType::SCRATCH) ? 1 : 0;
    uint64_t scratchAddr
        = BufferTypeToAddr(tempAlgParams.buffInfo.scratBuffType) + tempAlgParams.buffInfo.scratchBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint64_t inputSliceStride = tempAlgParams.inputSliceStride;
    uint64_t outputSliceStride = tempAlgParams.outputSliceStride;
    uint64_t inputRepeatStride = tempAlgParams.inputRepeatStride;
    uint64_t outputRepeatStride = tempAlgParams.outputRepeatStride;
    uint64_t repeatNum = tempAlgParams.repeatNum;
    uint64_t repeatNumVar = UINT64_MAX - repeatNum;

    HCCL_INFO(
        "[CcuInstructionScatterNHR1D] dimSize[%llu], die0Size[%llu], die1Size[%llu], inputAddr[%llu], outputAddr[%llu],"
        "scratchAddr[%llu], repeatNum[%llu]",
        dimSize[0], die0Size, die1Size, inputAddr, outputAddr, scratchAddr, repeatNum);

    if (DataCount == 0) {
        HCCL_INFO("[CcuTempScatterNHRMem2Mem1D] DataCount == 0, Template Run Ends.");
        return HCCL_SUCCESS;
    }

    if (axisSize > 1 || die1Size == 0) {
        axisSize = 1;
    }

    std::vector<LinkData> linksDie0;
    std::vector<LinkData> linksDie1;
    RankGroup scatterRankGroup;
    std::map<u32, u32> indexMap;
    std::vector<NHRStepInfo> stepInfoVector;
    u32 nSteps = GetNHRStepNum(tempRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetScatterStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
        if (indexMap.count(stepInfo.fromRank) == 0 && stepInfo.rxSliceIdxs.size() > 0) {
            u32 fromRankIdx = virtRankId2RankId(stepInfo.fromRank);
            indexMap[stepInfo.fromRank] = linksDie0.size();
            linksDie0.push_back(tempLinks.at(fromRankIdx)[0]);
            if (axisSize > 1) {
                linksDie1.push_back(tempLinks.at(fromRankIdx)[1]);
            }
            scatterRankGroup.AddRank(fromRankIdx);
        }
        if (indexMap.count(stepInfo.toRank) == 0 && stepInfo.txSliceIdxs.size() > 0) {
            u32 toRankIdx = virtRankId2RankId(stepInfo.toRank);
            indexMap[stepInfo.toRank] = linksDie0.size();
            linksDie0.push_back(tempLinks.at(toRankIdx)[0]);
            if (axisSize > 1) {
                linksDie1.push_back(tempLinks.at(toRankIdx)[1]);
            }
            scatterRankGroup.AddRank(toRankIdx);
        }
    }
    scatterRankGroup.AddRank(myRank_);

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t axisId = 0; axisId < axisSize; axisId++) { // 2个die上各一个mission
        CcuInstructionScatterNHR1D ccuInstruction;
        ccuInstruction.Init(
            myVirtRankId, rootId, inputAddr, outputAddr, scratchAddr, axisId, axisSize, sliceSize, die0Size, die1Size,
            inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride, repeatNum, repeatNumVar,
            stepInfoVector, indexMap, token, op_, tempVTopo_, isOutputScratch);
        HCCL_INFO(
            "[CcuTempScatterNHRMem2Mem1D] Run Init: myVirtRankId[%u], myRank_[%d], repeatNum[%u], inputAddr[%llu], "
            "outputAddr[%llu], inputSliceStride[%llu], outputSliceStride[%llu], "
            "inputRepeatStride[%llu], outputRepeatStride[%llu], die0Size[%llu], die1Size[%llu], "
            "repeatNumVar[%llu]",
            myVirtRankId, myRank_, repeatNum, inputAddr, outputAddr, inputSliceStride, outputSliceStride,
            inputRepeatStride, outputRepeatStride, die0Size, die1Size, repeatNumVar);
        ccuInstruction.SetLinks(axisId == 0 ? linksDie0 : linksDie1);
        ccuInstruction.SetRankGroup(scatterRankGroup);
        ccuInstruction.SetCntCkeNum(5); // 每个transport用5个CKE
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionScatterNHR1D>(ccuInstruction)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流
    HCCL_INFO("[CcuTempScatterNHRMem2Mem1D] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterNHRMem2Mem1D::GetScatterStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo)
{
    u32 virtRankIdx = tempVirtRankMap_[myRank_];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = tempRankSize_;
    stepInfo.fromRank = tempRankSize_;
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;
    uint32_t rootId = tempVirtRankMap_[rootId_];
    u32 deltaRoot = (rootId + tempRankSize_ - virtRankIdx) % tempRankSize_;
    u32 deltaRankPair = 1 << step;
    // 数据份数和数据编号增量
    u32 nSlices = (tempRankSize_ - 1 + (1 << step)) / (1 << (step + 1)); // 向上取整设计了下的
    u32 deltaSliceIndex = 1 << (step + 1);
    // 是否为2的幂
    u32 nRanks = 0;
    bool isPowerOfTwo = (tempRankSize_ & (tempRankSize_ - 1)) == 0;
    if (!isPowerOfTwo && step == nSteps - 1) {
        nRanks = tempRankSize_ - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) { // 需要发
        u32 sendTo = (virtRankIdx + tempRankSize_ - deltaRankPair) % tempRankSize_;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        }
        stepInfo.toRank = sendTo;
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom = (virtRankIdx + deltaRankPair) % tempRankSize_;
        u32 rxSliceIdx = virtRankIdx;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = rxSliceIdx;
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        }
        stepInfo.fromRank = recvFrom;
        stepInfo.nSlices = nSlices;
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
