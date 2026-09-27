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
#include "ccu_context_reduce_scatter_nhr1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_nhr_1D_mem2mem.h"

namespace Hccl {

static CcuInstRegister<CcuContextReduceScatterNHR1DMem2Mem>
    g_registrarReduceScatter(CcuInstType::CCU_REDUCE_SCATTER_NHR_1D_MEM2MEM);

CcuTempReduceScatterNHR1DMem2Mem::CcuTempReduceScatterNHR1DMem2Mem(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempReduceScatterNHR1DMem2Mem::~CcuTempReduceScatterNHR1DMem2Mem() {}

u32 CcuTempReduceScatterNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

void CcuTempReduceScatterNHR1DMem2Mem::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult CcuTempReduceScatterNHR1DMem2Mem::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    // 由于出框暂时只有一条Die，所以此处暂时实现单Die，改为1
    u32 linkNum = 1;
    linkNum_ = linkNum;
    u32 linkSize = 2;
    if (linkNum == linkSize) {
        isSipportTwoDie_ = true;
    }
    linkNumBtwPeers_ = linkNum;
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempReduceScatterNHR1DMem2Mem::GetMaxSliceSize() const { return UB_MAX_DATA_SIZE; }

HcclResult CcuTempReduceScatterNHR1DMem2Mem::GenExtIns(
    const TempFuncs& tempFuncs, TemplateDataParams& tempAlgParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[CcuTempReduceScatterNHRMem2Mem1D] Template Run start.");
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempReduceScatterNHRMem2Mem1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    uint64_t isBottom = tempFuncs.isBottom;
    opMode_ = tempFuncs.opMode;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);
    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("[CcuTempReduceScatterNHRMem2Mem1D] sliceSize is 0, nothing to do, just return success.");
        return HCCL_SUCCESS;
    }
    uint64_t die0Size = 0;
    uint64_t die1Size = 0;
    uint64_t dieNum = 2;
    if (isSipportTwoDie_) {
        die0Size = tempAlgParams.sliceSize / dieNum;
        die1Size = tempAlgParams.sliceSize - die0Size;
    } else {
        die0Size = tempAlgParams.sliceSize;
    }
    uint64_t inputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.inBuffType) + tempAlgParams.buffInfo.inBuffBaseOff;
    uint64_t outputAddr = BufferTypeToAddr(tempAlgParams.buffInfo.outBuffType) + tempAlgParams.buffInfo.outBuffBaseOff;
    uint64_t repeatNum = tempAlgParams.repeatNum;
    uint64_t inputSliceStride = tempAlgParams.inputSliceStride;
    uint64_t outputSliceStride = tempAlgParams.outputSliceStride;
    uint64_t inputRepeatStride = tempAlgParams.inputRepeatStride;
    uint64_t outputRepeatStride = tempAlgParams.outputRepeatStride;
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    uint64_t repeatNumVar = UINT64_MAX - repeatNum;
    HCCL_INFO(
        "[CcuTempReduceScatterNHR1D] dimSize[%llu], die0Size[%llu], die1Size[%llu], inputAddr[%llu],"
        "outputAddr[%llu], repeatNum[%llu], inputSliceStride[%llu], outputSliceStride[%llu],"
        "inputRepeatStride[%llu], outputRepeatStride[%llu]",
        dimSize[0], die0Size, die1Size, inputAddr, outputAddr, repeatNum, inputSliceStride, outputSliceStride,
        inputRepeatStride, outputRepeatStride);

    std::vector<LinkData> linksDie0;
    std::vector<LinkData> linksDie1;
    RankGroup rankGroup;
    std::map<u32, u32> indexMap;
    std::vector<NHRStepInfo> stepInfoVector;
    u32 nSteps = GetNHRStepNum(tempRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, stepInfo));
        stepInfoVector.push_back(stepInfo);
        if (indexMap.count(stepInfo.fromRank) == 0) {
            indexMap[stepInfo.fromRank] = linksDie0.size();
            linksDie0.push_back(tempLinks.at(GetRankFromMap(stepInfo.fromRank))[0]);
            if (isSipportTwoDie_) {
                linksDie1.push_back(tempLinks.at(GetRankFromMap(stepInfo.fromRank))[1]);
            }
            rankGroup.AddRank(GetRankFromMap(stepInfo.fromRank));
        }
        if (indexMap.count(stepInfo.toRank) == 0) {
            indexMap[stepInfo.toRank] = linksDie0.size();
            linksDie0.push_back(tempLinks.at(GetRankFromMap(stepInfo.toRank))[0]);
            if (isSipportTwoDie_) {
                linksDie1.push_back(tempLinks.at(GetRankFromMap(stepInfo.toRank))[1]);
            }
            rankGroup.AddRank(GetRankFromMap(stepInfo.toRank));
        }
    }
    rankGroup.AddRank(myRank_);

    std::unique_ptr<CcuInsGroup> insGroupPtr = std::make_unique<CcuInsGroup>();
    for (uint32_t axisId = 0; axisId < linkNum_; axisId++) { // 2D算法，需要下发2条通信指令
        CcuInstructionReduceScatterNHR1D ccuInstruction;
        ccuInstruction.Init(
            tempVirtRankMap_[myRank_], inputAddr, outputAddr, axisId, die0Size, die1Size, repeatNumVar,
            inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride, stepInfoVector, indexMap, token,
            op_, tempVTopo_, linkNum_, isBottom);
        ccuInstruction.SetLinks(axisId == 0 ? linksDie0 : linksDie1);
        ccuInstruction.SetRankGroup(rankGroup);
        ccuInstruction.SetCntCkeNum(5); // 每个transport用5个CKE
        insGroupPtr->Append(std::move(std::make_unique<CcuInstructionReduceScatterNHR1D>(ccuInstruction)));
    }
    tempInsQues[0]->Append(std::move(insGroupPtr)); // 只有一条流
    HCCL_INFO("[CcuTempReduceScatterNHRMem2Mem1D] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterNHR1DMem2Mem::GetStepInfo(u32 step, NHRStepInfo& stepInfo)
{
    // 将本rank号转换成算法使用的索引号
    u32 rankIdx = tempVirtRankMap_[myRank_];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;

    // 计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (rankIdx + tempRankSize_ - deltaRank) % tempRankSize_;
    u32 recvFrom = (rankIdx + deltaRank) % tempRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (tempRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 txSliceIdx = sendTo;
    u32 rxSliceIdx = rankIdx;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    // 计算本rank在本轮收/发中的slice编号
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        HCCL_INFO("[ReduceScatterNHR1D][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);
        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

RankId CcuTempReduceScatterNHR1DMem2Mem::GetRankFromMap(const u32 rankIdx)
{
    RankId rank = -1;
    for (auto& pair : tempVirtRankMap_) {
        if (pair.second == rankIdx) {
            rank = pair.first;
            break;
        }
    }
    return rank;
}
} // namespace Hccl
