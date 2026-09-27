/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>

#include "log.h"

#include "coll_alg_registry.h"
#include "temp_all_gather_mesh.h"
#include "topo_match_mesh.h"
#include "topo_match_concurr_mesh.h"
#include "temp_all_gather_concurr_mesh.h"
#include "all_gather_sole_executor.h"

namespace Hccl {
template <typename AlgTopoMatch, typename AlgTemplate>
AllGatherSoleExecutor<AlgTopoMatch, AlgTemplate>::AllGatherSoleExecutor() : CollAlgBase()
{}

template <typename AlgTopoMatch, typename AlgTemplate>
AllGatherSoleExecutor<AlgTopoMatch, AlgTemplate>::~AllGatherSoleExecutor()
{}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult AllGatherSoleExecutor<AlgTopoMatch, AlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    resReq.requiredScratchMemSize = 0;

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // instantiate a template
    AlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    // calculate required primQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    resReq.requiredSubQueNum = tempResReq.queNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

// dataSize_ as input
template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult AllGatherSoleExecutor<AlgTopoMatch, AlgTemplate>::GenPrimQues(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, PrimQuePtr primQue)
{
    // init and check params
    CHK_RET(Init(op, params, primQue));

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    HCCL_INFO("[CollAlgFactory] Rank[%d], [%s].", myRank_, topoMatch.Describe().c_str());

    // instantiate a template
    AlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);

    // calculate required primQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        tempAlg.SetDataType(dataType_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    HCCL_INFO(
        "[CollAlgFactory] Rank[%d], template [%s], requiredQue Num [%u].", myRank_, tempAlg.Describe().c_str(),
        tempResReq.queNum);

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));

    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;

    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OFFLOAD Mode for HOST.", myRank_);
        CHK_RET(GenPrimQues4Offload(tempAlg));
    } else { // OPBASE
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OPBASE Mode for HOST.", myRank_);
        CHK_RET(GenPrimQues4Opbase(dataSizePerVolume, tempAlg));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult
AllGatherSoleExecutor<AlgTopoMatch, AlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    // instantiate a template
    AlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    // calculate required primQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    algResReq.primQueueNum = tempResReq.queNum;
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult AllGatherSoleExecutor<AlgTopoMatch, AlgTemplate>::GenPrimQuesAIC(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    PrimQuePtr primQue)
{
    // init and check params
    CHK_RET(Init(op, params, primQue));

    // instantiate a template
    AlgTemplate tempAlg(myRank_, rankSize_, topoInfo.vTopo[0], topoInfo.virtRankMap[0]);
    tempAlg.SetDmaMode(dmaMode_);
    virtRankMap_ = topoInfo.virtRankMap[0];

    // calculate required primQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        tempAlg.SetDataType(dataType_);
        CHK_RET(tempAlg.CalcResDetour(linkMgr, tempResReq));
    } else {
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    HCCL_INFO(
        "[CollAlgFactory] Rank[%d], template [%s], requiredQue Num [%u].", myRank_, tempAlg.Describe().c_str(),
        tempResReq.queNum);

    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));

    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;

    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OFFLOAD Mode for AICPU.", myRank_);
        CHK_RET(GenPrimQues4Offload(tempAlg));
    } else { // OPBASE
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OPBASE Mode for AICPU.", myRank_);
        CHK_RET(GenPrimQues4Opbase(dataSizePerVolume, tempAlg));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult AllGatherSoleExecutor<AlgTopoMatch, AlgTemplate>::GenPrimQues4Offload(AlgTemplateBase& tempAlg)
{
    RankSliceInfo sliceInfoVec;
    AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
    CHK_RET(tempAlg.CalcSliceInfo(allignInfo, dataSize_, sliceInfoVec));

    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::INPUT;
    buffInfo.outBuffType = BufferType::OUTPUT;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();

    CHK_RET(tempAlg.GenPrimQue(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult AllGatherSoleExecutor<AlgTopoMatch, AlgTemplate>::GenPrimQues4Opbase(
    const u32 dataSizePerVolume, AlgTemplateBase& tempAlg)
{
    CHK_PRT_RET(
        dataSizePerVolume == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid dataSizePerVolume [%u].", myRank_, dataSizePerVolume),
        HcclResult::HCCL_E_INTERNAL);

    u32 scratchInputMemSize = static_cast<int>(
        (rankSize_ % dataSizePerVolume == 0) ?
            floor(maxTmpMemSize_ / rankSize_) :
            floor(maxTmpMemSize_ / (rankSize_ * dataSizePerVolume)) * dataSizePerVolume);

    CHK_PRT_RET(
        scratchInputMemSize == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid input maxTmpMemSize [%u].", myRank_, maxTmpMemSize_),
        HcclResult::HCCL_E_PARA);

    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::SCRATCH;
    buffInfo.outBuffType = BufferType::SCRATCH;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;

    u32 sendRecvTimes = (dataSize_ / scratchInputMemSize) + ((dataSize_ % scratchInputMemSize) == 0 ? 0 : 1);
    HCCL_INFO("[CollAlgFactory] Rank [%d], sendRecvTimes [%u].", myRank_, sendRecvTimes);

    for (u32 idx = 0; idx < sendRecvTimes; idx++) {
        u64 currDataSize = (idx == (sendRecvTimes - 1)) ? (dataSize_ - idx * scratchInputMemSize) : scratchInputMemSize;

        RankSliceInfo sliceInfoVec;
        AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
        CHK_RET(tempAlg.CalcSliceInfo(allignInfo, currDataSize, sliceInfoVec));

        TempFuncs tempFuncs;
        tempFuncs.opMode = opMode_;
        tempFuncs.enableCounterNotify = IsEnableCounterNotify();
        tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
        tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required

        UsrData usrData;
        DataSlice usrInSlice = DataSlice(BufferType::INPUT, idx * scratchInputMemSize, currDataSize);
        DataSlice scratchInSlice = DataSlice(BufferType::SCRATCH, virtRankMap_[myRank_] * currDataSize, currDataSize);
        usrData.usrInSlices.push_back(usrInSlice);
        usrData.scratchInSlices.push_back(scratchInSlice);

        for (u32 rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            DataSlice scratchOutSlice
                = DataSlice(BufferType::SCRATCH, virtRankMap_[rankIdx] * currDataSize, currDataSize);
            DataSlice usrOutSlice = DataSlice(
                BufferType::OUTPUT, virtRankMap_[rankIdx] * dataSize_ + idx * scratchInputMemSize, currDataSize);
            usrData.scratchOutSlices.push_back(scratchOutSlice);
            usrData.usrOutSlices.push_back(usrOutSlice);
        }

        tempFuncs.usrData = usrData;
        CHK_RET(tempAlg.GenPrimQue(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    }

    return HcclResult::HCCL_SUCCESS;
}

REGISTER_IMPL_BY_TEMP(OpType::ALLGATHER, AllGatherMesh, AllGatherSoleExecutor, TopoMatchMesh, TempAllGatherMesh);
REGISTER_IMPL_BY_TEMP(
    OpType::ALLGATHER, AllGatherConcurrMesh, AllGatherSoleExecutor, TopoMatchConcurrMesh, TempAllGatherConcurrMesh);
} // namespace Hccl
