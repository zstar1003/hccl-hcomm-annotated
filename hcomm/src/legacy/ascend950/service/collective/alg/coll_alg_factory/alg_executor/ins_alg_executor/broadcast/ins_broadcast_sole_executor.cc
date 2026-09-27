/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_broadcast_sole_executor.h"
#include "log.h"
#include "ins_coll_alg_registry.h"
#ifndef CCL_KERNEL_AICPU
#include "ccu_temp_broadcast_mesh_1D.h"
#include "ccu_temp_broadcast_mesh_2D.h"
#endif
#include "topo_match_mesh.h"
#include "topo_match_concurr_mesh.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsBroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsBroadcastSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsBroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsBroadcastSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsBroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    resReq.requiredScratchMemSize = 0;

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    // calculate required insQueues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsBroadcastSoleExecutor], CalcRes with detouring enabled.");
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] [InsBroadcastSoleExecutor], CalcRes with detouring disabled.");
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsBroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetRoot(op_.root);
    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsBroadcastSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsBroadcastSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    HCCL_DEBUG(
        "[InsCollAlgFactory] [InsBroadcastSoleExecutor] Rank[%d], requiredQueNum [%u].", myRank_,
        algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

// dataSize_ as input
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsBroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsCollAlgFactory] [InsBroadcastSoleExecutor] Host Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], [%s].", myRank_, topoMatch.Describe().c_str());
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataTypeSize_;
    CHK_PRT_RET(
        dataTypeSize_ == 0,
        HCCL_ERROR("Broadcast_[CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u].", myRank_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "[InsBroadcastSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.SetCollOp(op); // CCU template需要传递op信息
    tempAlg.SetRoot(root_);

    // 计算算法模板所需资源
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsBroadcastSoleExecutor] Rank[%d], CalcRes with detouring enabled for Orchestrate.",
            myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsBroadcastSoleExecutor] Rank[%d], CalcRes with detouring disabled for Orchestrate.",
            myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));

    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OFFLOAD Mode for HOST.", myRank_);
        CHK_RET(OrchestrateOffload(tempAlg));
    } else { // OPBASE
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OPBASE Mode for HOST.", myRank_);
        CHK_RET(OrchestrateOpbase(tempAlg));
    }

    return HcclResult::HCCL_SUCCESS;
}

// 算子执行aicpu接口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsBroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsCollAlgFactory] [InsBroadcastSoleExecutor] [InsReduceSoleExecutor] AiCpu Orchestrate begins.");
    // 参数校验和初始化
    CHK_RET(Init(op, params, insQue));

    // soleEsecutor 只支持单层拓扑, 所以只取第 0 级通信域的信息
    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataTypeSize_;
    CHK_PRT_RET(
        dataTypeSize_ == 0,
        HCCL_ERROR("Broadcast_[CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u].", myRank_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "Broadcast_[InsReduceSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.SetCollOp(op); // CCU template需要传递op信息
    tempAlg.SetRoot(root_);

    // 计算算法模板所需资源
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsBroadcastSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsBroadcastSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));

    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_DEBUG(
            "[InsReduceSoleExecutor] Rank[%d], Generating Instruction Queues in OFFLOAD Mode for AICPU.", myRank_);
        CHK_RET(OrchestrateOffload(tempAlg));
    } else { // OPBASE
        HCCL_DEBUG(
            "[InsReduceSoleExecutor] Rank[%d], Generating Instruction Queues in OPBASE Mode for AICPU.", myRank_);
        CHK_RET(OrchestrateOpbase(tempAlg));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsBroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateOpbase(InsAlgTemplate& tempAlg)
{
    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::SCRATCH;
    buffInfo.outBuffType = BufferType::SCRATCH;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;

    // 基本参数配置
    AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
    tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required

    // 根据CCL Buffer 大小，计算出一轮中最多能输出多少数据
    u64 maxLoopOutputCount
        = std::min(static_cast<u64>(maxTmpMemSize_), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_;
    CHK_PRT_RET(
        maxLoopOutputCount == 0, HCCL_ERROR("[InsReduceSoleExecutor] maxLoopOutputCount is zero."),
        HcclResult::HCCL_E_PARA);
    u64 dataSize = dataCount_ * dataTypeSize_;

    // offload模式根据UB_MAX_DATA_SIZE，计算出一轮中最多能输出多少数据
    u64 maxLoopOutputSize = maxLoopOutputCount * dataTypeSize_;

    u64 loopTimes = dataSize / maxLoopOutputSize + static_cast<u64>(dataSize % maxLoopOutputSize != 0);

    for (u32 loop = 0; loop < loopTimes; loop++) {
        u64 currloopOffset = loop * maxLoopOutputSize;
        u64 currSize = (loop == (loopTimes - 1)) ? dataSize - currloopOffset : maxLoopOutputSize;
        UsrData usrData;
        usrData.usrInSlices.emplace_back(BufferType::INPUT, currloopOffset, currSize);
        usrData.scratchInSlices.emplace_back(BufferType::SCRATCH, 0, currSize);
        usrData.scratchOutSlices.emplace_back(BufferType::SCRATCH, 0, currSize);
        usrData.usrOutSlices.emplace_back(BufferType::INPUT, currloopOffset, currSize);
        tempFuncs.usrData = usrData;

        RankSliceInfo sliceInfoVec;
        CHK_RET(tempAlg.CalcSliceInfo(allignInfo, currSize, sliceInfoVec));
        CHK_RET(tempAlg.Run(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsBroadcastSoleExecutor] Rank[%d], done generating instruction queues, "
            "currSize[%llu], currOffset[%llu].",
            myRank_, currSize, currloopOffset);
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsBroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateOffload(InsAlgTemplate& tempAlg)
{
    HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], done calculating slice information.", myRank_);

    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::INPUT;
    buffInfo.outBuffType = BufferType::INPUT;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;
    HCCL_DEBUG(
        "[InsCollAlgFactory] Rank[%d], input buffer type [%s], output buffer type [%s], input buffer base "
        "offset [%u], output buffer base offset [%u].",
        myRank_, buffInfo.inBuffType.Describe().c_str(), buffInfo.outBuffType.Describe().c_str(),
        buffInfo.inBuffBaseOff, buffInfo.outBuffBaseOff);

    AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
    tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required

    u64 dataSize = dataCount_ * dataTypeSize_;

    // offload模式根据UB_MAX_DATA_SIZE，计算出一轮中最多能输出多少数据
    u64 maxLoopOutputSize = (UB_MAX_DATA_SIZE / dataTypeSize_) * dataTypeSize_;

    u64 loopTimes = dataSize / maxLoopOutputSize + static_cast<u64>(dataSize % maxLoopOutputSize != 0);

    for (u32 loop = 0; loop < loopTimes; loop++) {
        u64 currloopOffset = loop * maxLoopOutputSize;
        u64 currSize = (loop == (loopTimes - 1)) ? dataSize - currloopOffset : maxLoopOutputSize;
        UsrData usrData;
        usrData.usrInSlices.emplace_back(BufferType::INPUT, currloopOffset, currSize);
        usrData.scratchInSlices.emplace_back(BufferType::SCRATCH, 0, currSize);
        usrData.scratchOutSlices.emplace_back(BufferType::SCRATCH, 0, currSize);
        usrData.usrOutSlices.emplace_back(BufferType::INPUT, currloopOffset, currSize);
        tempFuncs.usrData = usrData;
        RankSliceInfo sliceInfoVec;
        CHK_RET(tempAlg.CalcSliceInfo(allignInfo, currSize, sliceInfoVec));
        CHK_RET(tempAlg.Run(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsBroadcastSoleExecutor] Rank[%d], done generating instruction queues, "
            "currSize[%llu], currOffset[%llu].",
            myRank_, currSize, currloopOffset);
    }

    return HcclResult::HCCL_SUCCESS;
}

#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::BROADCAST, CcuBroadcastMesh1D, InsBroadcastSoleExecutor, TopoMatchMesh, CcuTempBroadcastMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::BROADCAST, CcuBroadcastMesh2D, InsBroadcastSoleExecutor, TopoMatchConcurrMesh, CcuTempBroadcastMesh2D);
#endif
} // namespace Hccl
