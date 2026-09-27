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
#include "all_reduce_comb_executor.h"

namespace Hccl {
template <typename AlgTopoMatch, typename AlgTempRS, typename AlgTempAG>
AllReduceCombExecutor<AlgTopoMatch, AlgTempRS, AlgTempAG>::AllReduceCombExecutor() : CollAlgBase()
{}

template <typename AlgTopoMatch, typename AlgTempRS, typename AlgTempAG>
AllReduceCombExecutor<AlgTopoMatch, AlgTempRS, AlgTempAG>::~AllReduceCombExecutor()
{}

template <typename AlgTopoMatch, typename AlgTempRS, typename AlgTempAG>
HcclResult AllReduceCombExecutor<AlgTopoMatch, AlgTempRS, AlgTempAG>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    resReq.requiredScratchMemSize = 0;

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    HCCL_INFO("[CollAlgFactory] Rank[%d], [%s].", myRank_, topoMatch.Describe().c_str());

    // instantiate templates
    AlgTempRS tempRSAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    AlgTempAG tempAGAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    // calculate required primQues and prepare queue
    AlgTempResReq tempResReqRS;
    u32 requiredScratchMultiplier = 0;
    if (enableDetour_) {
        CHK_RET(tempRSAlg.CalcResDetour(true, rankGraph, tempResReqRS, requiredScratchMultiplier));
    } else {
        CHK_RET(tempRSAlg.CalcRes(true, tempResReqRS, requiredScratchMultiplier));
    }

    AlgTempResReq tempResReqAG;
    if (enableDetour_) {
        CHK_RET(tempAGAlg.CalcResDetour(rankGraph, tempResReqAG));
    } else {
        CHK_RET(tempAGAlg.CalcRes(tempResReqAG));
    }

    CHK_PRT_RET(
        tempResReqRS.queNum != tempResReqAG.queNum,
        HCCL_ERROR(
            "[CollAlgFactory] Rank [%d], required QueNum for RS template [%u] not equals to it for AG template [%u].",
            myRank_, tempResReqRS.queNum, tempResReqAG.queNum),
        HcclResult::HCCL_E_INTERNAL);

    resReq.requiredSubQueNum = tempResReqRS.queNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTempRS, typename AlgTempAG>
HcclResult AllReduceCombExecutor<AlgTopoMatch, AlgTempRS, AlgTempAG>::GenPrimQues(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, PrimQuePtr primQue)
{
    // init and check params
    CHK_RET(Init(op, params, primQue));

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // instantiate templates
    AlgTempRS tempRSAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempRSAlg.InitReduceInfo(redOp_, dataType_);
    tempRSAlg.SetDmaMode(dmaMode_);
    AlgTempAG tempAGAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAGAlg.SetDmaMode(dmaMode_);

    // calculate required primQues and prepare queue
    AlgTempResReq tempResReqRS;
    u32 requiredScratchMultiplier = 0;
    if (enableDetour_) {
        tempRSAlg.SetDataType(dataType_);
        CHK_RET(tempRSAlg.CalcResDetour(true, rankGraph, tempResReqRS, requiredScratchMultiplier));
    } else {
        CHK_RET(tempRSAlg.CalcRes(true, tempResReqRS, requiredScratchMultiplier));
    }

    CHK_RET(InitQueue(tempResReqRS.queNum, requiredQue_));

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReqRS.links, tempResLinks_));

    AlgTempResReq tempResReqAG;
    if (enableDetour_) {
        tempAGAlg.SetDataType(dataType_);
        CHK_RET(tempAGAlg.CalcResDetour(rankGraph, tempResReqAG));
    } else {
        CHK_RET(tempAGAlg.CalcRes(tempResReqAG));
    }

    CHK_PRT_RET(
        tempResReqAG.queNum != tempResReqRS.queNum,
        HCCL_ERROR(
            "[CollAlgFactory] Rank [%d], required QueNum for RS template [%u] not equals to it for AG template [%u].",
            myRank_, tempResReqRS.queNum, tempResReqAG.queNum),
        HcclResult::HCCL_E_INTERNAL);

    HCCL_INFO(
        "[CollAlgFactory] Rank[%d], reduce scatter template [%s], all gather template [%s]: requiredQue Num [%u].",
        myRank_, tempRSAlg.Describe().c_str(), tempAGAlg.Describe().c_str(), tempResReqRS.queNum);

    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume; // for allreduce, dataSize is the size of whole data

    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OFFLOAD Mode for Host.", myRank_);
        CHK_RET(GenPrimQues4Offload(tempRSAlg, tempAGAlg));
    } else { // OPBASE
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OPBASE Mode for Host.", myRank_);
        CHK_RET(GenPrimQues4Opbase(requiredScratchMultiplier, dataSizePerVolume, tempRSAlg, tempAGAlg));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTempRS, typename AlgTempAG>
HcclResult
AllReduceCombExecutor<AlgTopoMatch, AlgTempRS, AlgTempAG>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    // instantiate a template
    AlgTempRS tempRSAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempRSAlg.InitReduceInfo(redOp_, dataType_);

    // calculate required primQues and prepare queue
    AlgTempResReq tempResReqRS;
    u32 requiredScratchMultiplier = 0;
    if (enableDetour_) {
        tempRSAlg.SetDataType(dataType_);
        CHK_RET(tempRSAlg.CalcResDetour(true, rankGraph, tempResReqRS, requiredScratchMultiplier));
    } else {
        CHK_RET(tempRSAlg.CalcRes(true, tempResReqRS, requiredScratchMultiplier));
    }

    algResReq.primQueueNum = tempResReqRS.queNum;
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReqRS.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTempRS, typename AlgTempAG>
HcclResult AllReduceCombExecutor<AlgTopoMatch, AlgTempRS, AlgTempAG>::GenPrimQuesAIC(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    PrimQuePtr primQue)
{
    // init and check params
    CHK_RET(Init(op, params, primQue));

    // instantiate templates
    AlgTempRS tempRSAlg(myRank_, rankSize_, topoInfo.vTopo[0], topoInfo.virtRankMap[0]);
    tempRSAlg.InitReduceInfo(redOp_, dataType_);
    tempRSAlg.SetDmaMode(dmaMode_);
    AlgTempAG tempAGAlg(myRank_, rankSize_, topoInfo.vTopo[0], topoInfo.virtRankMap[0]);
    tempAGAlg.SetDmaMode(dmaMode_);

    // calculate required primQues and prepare queue
    AlgTempResReq tempResReqRS;
    u32 requiredScratchMultiplier = 0;
    if (enableDetour_) {
        tempRSAlg.SetDataType(dataType_);
        CHK_RET(tempRSAlg.CalcResDetour(true, linkMgr, tempResReqRS, requiredScratchMultiplier));
    } else {
        CHK_RET(tempRSAlg.CalcRes(true, tempResReqRS, requiredScratchMultiplier));
    }

    CHK_RET(InitQueue(tempResReqRS.queNum, requiredQue_));

    CHK_RET(PrepResLinks(myRank_, tempResReqRS.links, linkMgr, tempResLinks_));

    AlgTempResReq tempResReqAG;

    if (enableDetour_) {
        tempAGAlg.SetDataType(dataType_);
        CHK_RET(tempAGAlg.CalcResDetour(linkMgr, tempResReqAG));
    } else {
        CHK_RET(tempAGAlg.CalcRes(tempResReqAG));
    }

    HCCL_INFO(
        "[CollAlgFactory] Rank[%d], reduce scatter template [%s], all gather template [%s]: requiredQue Num [%u].",
        myRank_, tempRSAlg.Describe().c_str(), tempAGAlg.Describe().c_str(), tempResReqRS.queNum);

    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume; // for allreduce, dataSize is the size of whole data

    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OFFLOAD Mode for AICPU.", myRank_);
        CHK_RET(GenPrimQues4Offload(tempRSAlg, tempAGAlg));
    } else { // OPBASE
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OPBASE Mode for AICPU.", myRank_);
        CHK_RET(GenPrimQues4Opbase(requiredScratchMultiplier, dataSizePerVolume, tempRSAlg, tempAGAlg));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTempRS, typename AlgTempAG>
HcclResult AllReduceCombExecutor<AlgTopoMatch, AlgTempRS, AlgTempAG>::GenPrimQues4Offload(
    AlgTemplateBase& tempRSAlg, AlgTemplateBase& tempAGAlg)
{
    RankSliceInfo sliceInfoVec;
    AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
    CHK_RET(tempRSAlg.CalcSliceInfo(allignInfo, true, dataSize_, sliceInfoVec));

    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::INPUT;
    buffInfo.outBuffType = BufferType::OUTPUT;
    buffInfo.scratBuffType = BufferType::OUTPUT;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;
    buffInfo.scratchBuffBaseOff = 0;

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.forAllReduce = true;

    CHK_RET(tempRSAlg.GenPrimQue(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    CHK_RET(tempAGAlg.GenPrimQue(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTempRS, typename AlgTempAG>
HcclResult AllReduceCombExecutor<AlgTopoMatch, AlgTempRS, AlgTempAG>::GenPrimQues4Opbase(
    const u32 requiredScratchMultiplier, const u32 dataSizePerVolume, AlgTemplateBase& tempRSAlg,
    AlgTemplateBase& tempAGAlg)
{
    u64 scratchInputMemSize
        = (rankSize_ % dataSizePerVolume == 0) ?
              static_cast<int>(floor(maxTmpMemSize_ / (rankSize_ + requiredScratchMultiplier)) * rankSize_) :
              static_cast<int>(
                  floor(maxTmpMemSize_ / ((rankSize_ + requiredScratchMultiplier) * dataSizePerVolume)) * rankSize_
                  * dataSizePerVolume);

    CHK_PRT_RET(
        scratchInputMemSize == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid input maxTmpMemSize [%u].", myRank_, maxTmpMemSize_),
        HcclResult::HCCL_E_PARA);

    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::SCRATCH;
    buffInfo.outBuffType = BufferType::SCRATCH;
    buffInfo.scratBuffType = BufferType::SCRATCH;

    u32 sendRecvTimes = (dataSize_ / scratchInputMemSize) + ((dataSize_ % scratchInputMemSize) == 0 ? 0 : 1);
    HCCL_INFO("[CollAlgFactory] Rank [%d], datasize [%u], sendRecvTimes [%u].", myRank_, dataSize_, sendRecvTimes);

    for (u32 idx = 0; idx < sendRecvTimes; idx++) {
        u64 currDataSize = (idx == sendRecvTimes - 1) ? (dataSize_ - idx * scratchInputMemSize) : scratchInputMemSize;

        buffInfo.inBuffBaseOff = 0;
        buffInfo.outBuffBaseOff = currDataSize;
        buffInfo.scratchBuffBaseOff = currDataSize;

        RankSliceInfo sliceInfoVec;
        AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
        CHK_RET(tempRSAlg.CalcSliceInfo(allignInfo, true, currDataSize, sliceInfoVec));
        TempFuncs tempFuncs;
        tempFuncs.opMode = opMode_;
        tempFuncs.enableCounterNotify = IsEnableCounterNotify();
        tempFuncs.forAllReduce = true;
        tempFuncs.isForepart = true; // Usr Buff to CCL Buff required

        UsrData usrData;
        DataSlice usrInSlice = DataSlice(BufferType::INPUT, idx * scratchInputMemSize, currDataSize);
        DataSlice scratchInSlice = DataSlice(BufferType::SCRATCH, 0, currDataSize);
        usrData.usrInSlices.push_back(usrInSlice);
        usrData.scratchInSlices.push_back(scratchInSlice);

        tempFuncs.usrData = usrData;
        CHK_RET(tempRSAlg.GenPrimQue(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));

        buffInfo.outBuffBaseOff = 0;
        buffInfo.inBuffBaseOff = currDataSize; // will not be used in allgather
        buffInfo.scratchBuffBaseOff = currDataSize;
        tempFuncs.isForepart = false; // Usr Buff to CCL Buff required
        tempFuncs.isBottom = true;    // CCL Buff to Usr Buff required

        DataSlice scratchOutSlice = DataSlice(BufferType::SCRATCH, 0, currDataSize);
        DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, idx * scratchInputMemSize, currDataSize);
        tempFuncs.usrData.scratchOutSlices.push_back(scratchOutSlice);
        tempFuncs.usrData.usrOutSlices.push_back(usrOutSlice);
        CHK_RET(tempAGAlg.GenPrimQue(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    }

    return HcclResult::HCCL_SUCCESS;
}

REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLREDUCE, AllReduceConcurrMesh, AllReduceCombExecutor, TopoMatchConcurrMesh, TempReduceScatterConcurrMesh,
    TempAllGatherConcurrMesh);
REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLREDUCE, AllReduceMesh, AllReduceCombExecutor, TopoMatchMesh, TempReduceScatterMesh, TempAllGatherMesh);
} // namespace Hccl
