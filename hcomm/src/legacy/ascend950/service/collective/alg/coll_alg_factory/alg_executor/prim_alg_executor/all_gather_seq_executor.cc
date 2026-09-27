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
#include "all_gather_seq_executor.h"

namespace Hccl {
template <typename AlgTopoMatch, typename AlgTemp0, typename AlgTemp1>
AllGatherSeqExecutor<AlgTopoMatch, AlgTemp0, AlgTemp1>::AllGatherSeqExecutor() : CollAlgBase()
{}

template <typename AlgTopoMatch, typename AlgTemp0, typename AlgTemp1>
AllGatherSeqExecutor<AlgTopoMatch, AlgTemp0, AlgTemp1>::~AllGatherSeqExecutor()
{}

// dataSize_ as input
template <typename AlgTopoMatch, typename AlgTemp0, typename AlgTemp1>
HcclResult AllGatherSeqExecutor<AlgTopoMatch, AlgTemp0, AlgTemp1>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    resReq.requiredScratchMemSize = 0;

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    HCCL_INFO("[CollAlgFactory] Rank[%d], [%s].", myRank_, topoMatch.Describe().c_str());

    // instantiate templates
    auto virtRankMapIter = virtRankMap_.begin();
    auto vTopoIter = vTopo_.begin();
    tempRankSizes_.push_back((*virtRankMapIter).size());

    AlgTemp0 tempAlg0(myRank_, tempRankSizes_[0], (*vTopoIter), (*virtRankMapIter));

    virtRankMapIter++;
    vTopoIter++;
    tempRankSizes_.push_back((*virtRankMapIter).size());

    AlgTemp1 tempAlg1(myRank_, tempRankSizes_[1], (*vTopoIter), (*virtRankMapIter));

    // calculate required primQues
    AlgTempResReq tempResReq0;
    CHK_RET(tempAlg0.CalcRes(tempResReq0));
    AlgTempResReq tempResReq1;
    CHK_RET(tempAlg1.CalcRes(tempResReq1));

    resReq.requiredSubQueNum = std::max(tempResReq0.queNum, tempResReq1.queNum) - 1;
    return HcclResult::HCCL_SUCCESS;
}

// dataSize_ as input
template <typename AlgTopoMatch, typename AlgTemp0, typename AlgTemp1>
HcclResult AllGatherSeqExecutor<AlgTopoMatch, AlgTemp0, AlgTemp1>::GenPrimQues(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, PrimQuePtr primQue)
{
    // init and check params
    CHK_RET(Init(op, params, primQue));

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    auto virtRankMapIter = virtRankMap_.begin();
    tempRankSizes_.push_back((*virtRankMapIter).size());
    virtRankMapIter++;
    tempRankSizes_.push_back((*virtRankMapIter).size());

    // instantiate templates
    AlgTemp0 tempAlg0(myRank_, tempRankSizes_[0], vTopo_[0], virtRankMap_[0]);
    tempAlg0.SetDmaMode(dmaMode_);
    AlgTemp1 tempAlg1(myRank_, tempRankSizes_[1], vTopo_[1], virtRankMap_[1]);
    tempAlg1.SetDmaMode(dmaMode_);

    // calculate required primQues and prepare queue
    AlgTempResReq tempResReq0;
    CHK_RET(tempAlg0.CalcRes(tempResReq0));

    std::vector<PrimQuePtr> requiredQue0;
    CHK_RET(InitQueue(tempResReq0.queNum, requiredQue0));
    HCCL_INFO(
        "[CollAlgFactory] Rank[%d], allGather template 0 [%s]: requiredQue Num [%u].", myRank_,
        tempAlg0.Describe().c_str(), tempResReq0.queNum);
    tempRequiredQues_.push_back(requiredQue0);

    ResLinks tempLinks0;
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq0.links, tempLinks0));
    tempResLinks_.push_back(tempLinks0);

    AlgTempResReq tempResReq1;
    CHK_RET(tempAlg1.CalcRes(tempResReq1));

    std::vector<PrimQuePtr> requiredQue1;
    CHK_RET(InitQueue(tempResReq1.queNum, requiredQue1));
    HCCL_INFO(
        "[CollAlgFactory] Rank[%d], allGather template 1 [%s]: requiredQue Num [%u].", myRank_,
        tempAlg1.Describe().c_str(), tempResReq1.queNum);
    tempRequiredQues_.push_back(requiredQue1);

    ResLinks tempLinks1;
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq1.links, tempLinks1));
    tempResLinks_.push_back(tempLinks1);

    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;

    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OFFLOAD Mode for Host.", myRank_);
        CHK_RET(GenPrimQues4Offload(tempAlg0, tempAlg1));
    } else { // OPBASE
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OPBASE Mode for Host.", myRank_);
        CHK_RET(GenPrimQues4Opbase(dataSizePerVolume, tempAlg0, tempAlg1));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemp0, typename AlgTemp1>
HcclResult
AllGatherSeqExecutor<AlgTopoMatch, AlgTemp0, AlgTemp1>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateMultiLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    // instantiate templates
    AlgTemp0 tempAlg0(myRank_, virtRankMap_[0].size(), vTopo_[0], virtRankMap_[0]);
    tempAlg0.SetDmaMode(dmaMode_);
    AlgTemp1 tempAlg1(myRank_, virtRankMap_[1].size(), vTopo_[1], virtRankMap_[1]);
    tempAlg1.SetDmaMode(dmaMode_);

    // calculate required resources
    AlgTempResReq tempResReq0;
    CHK_RET(tempAlg0.CalcRes(tempResReq0));
    AlgTempResReq tempResReq1;
    CHK_RET(tempAlg1.CalcRes(tempResReq1));

    algResReq.primQueueNum = std::max(tempResReq0.queNum, tempResReq1.queNum);

    LinkReq linkReqSeq = GetSeqLinksUnion(tempResReq0.links, tempResReq1.links);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, linkReqSeq, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemp0, typename AlgTemp1>
HcclResult AllGatherSeqExecutor<AlgTopoMatch, AlgTemp0, AlgTemp1>::GenPrimQuesAIC(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    PrimQuePtr primQue)
{
    // init and check params
    CHK_RET(Init(op, params, primQue));

    // Topo Match
    vTopo_ = topoInfo.vTopo;
    virtRanks_ = topoInfo.virtRanks;
    virtRankMap_ = topoInfo.virtRankMap;

    tempRankSizes_.push_back((virtRankMap_[0]).size());
    tempRankSizes_.push_back((virtRankMap_[1]).size());

    // instantiate templates
    AlgTemp0 tempAlg0(myRank_, tempRankSizes_[0], vTopo_[0], virtRankMap_[0]);
    tempAlg0.SetDmaMode(dmaMode_);
    AlgTemp1 tempAlg1(myRank_, tempRankSizes_[1], vTopo_[1], virtRankMap_[1]);
    tempAlg1.SetDmaMode(dmaMode_);

    // calculate required primQues and prepare queue
    AlgTempResReq tempResReq0;
    CHK_RET(tempAlg0.CalcRes(tempResReq0));

    AlgTempResReq tempResReq1;
    CHK_RET(tempAlg1.CalcRes(tempResReq1));

    std::vector<PrimQuePtr> requiredQue0;
    CHK_RET(InitQueue(tempResReq0.queNum, requiredQue0));
    HCCL_INFO(
        "[CollAlgFactory] Rank[%d], allGather template 0 [%s]: requiredQue Num [%u].", myRank_,
        tempAlg0.Describe().c_str(), tempResReq0.queNum);
    tempRequiredQues_.push_back(requiredQue0);

    std::vector<PrimQuePtr> requiredQue1;
    CHK_RET(InitQueue(tempResReq1.queNum, requiredQue1));
    HCCL_INFO(
        "[CollAlgFactory] Rank[%d], allGather template 1 [%s]: requiredQue Num [%u].", myRank_,
        tempAlg1.Describe().c_str(), tempResReq1.queNum);
    tempRequiredQues_.push_back(requiredQue1);

    ResLinks tempLinks0;
    CHK_RET(PrepResLinks(myRank_, tempResReq0.links, linkMgr, tempLinks0));
    tempResLinks_.push_back(tempLinks0);

    ResLinks tempLinks1;
    CHK_RET(PrepResLinks(myRank_, tempResReq1.links, linkMgr, tempLinks1));
    tempResLinks_.push_back(tempLinks1);

    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;

    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OFFLOAD Mode for AICPU.", myRank_);
        CHK_RET(GenPrimQues4Offload(tempAlg0, tempAlg1));
    } else { // OPBASE
        HCCL_INFO("[CollAlgFactory] Rank[%d], Generating Primitive Queues in OPBASE Mode for AICPU.", myRank_);
        CHK_RET(GenPrimQues4Opbase(dataSizePerVolume, tempAlg0, tempAlg1));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemp0, typename AlgTemp1>
HcclResult AllGatherSeqExecutor<AlgTopoMatch, AlgTemp0, AlgTemp1>::GenPrimQues4Offload(
    AlgTemplateBase& tempAlg0, AlgTemplateBase& tempAlg1)
{
    RankSliceInfo sliceInfoVec0;
    AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
    CHK_RET(tempAlg0.CalcSliceInfo(allignInfo, dataSize_, sliceInfoVec0));

    u64 outcomeSize = dataSize_ * tempRankSizes_[0];
    u32 outDataIdx = virtRankMap_[1][myRank_];

    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::INPUT;
    buffInfo.outBuffType = BufferType::OUTPUT;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = outDataIdx * outcomeSize;

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.forAlgSeqComb = false;

    CHK_RET(tempAlg0.GenPrimQue(tempFuncs, sliceInfoVec0, buffInfo, tempResLinks_[0], tempRequiredQues_[0]));

    // level 1
    RankSliceInfo sliceInfoVec1;
    CHK_RET(tempAlg1.CalcSliceInfo(allignInfo, outcomeSize, sliceInfoVec1));

    buffInfo.outBuffBaseOff = 0;

    tempFuncs.forAlgSeqComb = true;

    CHK_RET(tempAlg1.GenPrimQue(tempFuncs, sliceInfoVec1, buffInfo, tempResLinks_[1], tempRequiredQues_[1]));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemp0, typename AlgTemp1>
HcclResult AllGatherSeqExecutor<AlgTopoMatch, AlgTemp0, AlgTemp1>::GenPrimQues4Opbase(
    const u32 dataSizePerVolume, AlgTemplateBase& tempAlg0, AlgTemplateBase& tempAlg1)
{
    CHK_PRT_RET(
        dataSizePerVolume == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid dataSizePerVolume [%u].", myRank_, dataSizePerVolume),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(rankSize_ == 0, HCCL_ERROR("[CollAlgFactory] RankSize is zero!"), HcclResult::HCCL_E_INTERNAL);
    u32 scratchInputSize = static_cast<int>(
        (rankSize_ % dataSizePerVolume == 0) ?
            floor(maxTmpMemSize_ / rankSize_) :
            floor(maxTmpMemSize_ / (rankSize_ * dataSizePerVolume)) * dataSizePerVolume);

    CHK_PRT_RET(
        scratchInputSize == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid input maxTmpMemSize [%u].", myRank_, maxTmpMemSize_),
        HcclResult::HCCL_E_PARA);

    BuffInfo buffInfo;
    buffInfo.outBuffType = BufferType::SCRATCH;

    u32 sendRecvTimes = (dataSize_ / scratchInputSize) + ((dataSize_ % scratchInputSize) == 0 ? 0 : 1);
    HCCL_INFO("[CollAlgFactory] Rank [%d], sendRecvTimes [%u].", myRank_, sendRecvTimes);

    for (u32 idx = 0; idx < sendRecvTimes; idx++) {
        // datasize of level 0
        u64 currDataSize = (idx == (sendRecvTimes - 1)) ? (dataSize_ - idx * scratchInputSize) : scratchInputSize;

        // expected outcome of level 0 allgather
        u64 outcomeSize = currDataSize * tempRankSizes_[0];
        u32 outDataIdx = virtRankMap_[1][myRank_];

        // level 0
        RankSliceInfo sliceInfoVec0;
        AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
        CHK_RET(tempAlg0.CalcSliceInfo(allignInfo, currDataSize, sliceInfoVec0));

        buffInfo.outBuffBaseOff = outDataIdx * outcomeSize;

        TempFuncs tempFuncs;
        tempFuncs.opMode = opMode_;
        tempFuncs.enableCounterNotify = IsEnableCounterNotify();
        tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
        tempFuncs.isBottom = false;  // CCL Buff to Usr Buff required

        UsrData usrData;
        DataSlice usrInSlice = DataSlice(BufferType::INPUT, idx * scratchInputSize, currDataSize);
        DataSlice scratchInSlice = DataSlice(
            BufferType::SCRATCH, virtRankMap_[0][myRank_] * currDataSize + outDataIdx * outcomeSize, currDataSize);
        usrData.usrInSlices.push_back(usrInSlice);
        usrData.scratchInSlices.push_back(scratchInSlice);

        tempFuncs.usrData = usrData;

        CHK_RET(tempAlg0.GenPrimQue(tempFuncs, sliceInfoVec0, buffInfo, tempResLinks_[0], tempRequiredQues_[0]));

        // level 1
        RankSliceInfo sliceInfoVec1;
        CHK_RET(tempAlg1.CalcSliceInfo(allignInfo, outcomeSize, sliceInfoVec1));

        buffInfo.outBuffBaseOff = 0;

        tempFuncs.isForepart = false; // Usr Buff to CCL Buff required
        tempFuncs.isBottom = true;    // CCL Buff to Usr Buff required

        for (u32 rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            DataSlice scratchOutSlice = DataSlice(BufferType::SCRATCH, rankIdx * currDataSize, currDataSize);
            DataSlice usrOutSlice
                = DataSlice(BufferType::OUTPUT, rankIdx * dataSize_ + idx * scratchInputSize, currDataSize);
            tempFuncs.usrData.scratchOutSlices.push_back(scratchOutSlice);
            tempFuncs.usrData.usrOutSlices.push_back(usrOutSlice);
        }

        CHK_RET(tempAlg1.GenPrimQue(tempFuncs, sliceInfoVec1, buffInfo, tempResLinks_[1], tempRequiredQues_[1]));
    }

    return HcclResult::HCCL_SUCCESS;
}

REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLGATHER, AllGatherSeqMeshRing, AllGatherSeqExecutor, TopoMatchMeshRing, TempAllGatherMesh,
    TempAllGatherRing);
} // namespace Hccl
