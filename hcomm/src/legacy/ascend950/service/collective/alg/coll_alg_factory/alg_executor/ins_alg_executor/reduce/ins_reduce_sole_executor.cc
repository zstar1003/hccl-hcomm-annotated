/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_reduce_sole_executor.h"
#include "log.h"
#include "ins_coll_alg_registry.h"
#ifndef CCL_KERNEL_AICPU
#include "ccu_temp_reduce_mesh_2D.h"
#endif
#include "topo_match_concurr_mesh.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsReduceSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsReduceSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    constexpr u64 needScratchSize = 200 * 1024 * 1024; // 需要申请200MB临时内存
    resReq.requiredScratchMemSize = needScratchSize;

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.InitReduceInfo(redOp_, dataType_);

    // calculate required insQueues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsReduceSoleExecutor], CalcRes with detouring enabled.");
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] [InsReduceSoleExecutor], CalcRes with detouring disabled.");
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.InitReduceInfo(redOp_, dataType_);

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsReduceSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] [InsReduceSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    HCCL_DEBUG(
        "[InsCollAlgFactory] [InsReduceSoleExecutor] Rank[%d], requiredQueNum [%u].", myRank_, algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

// dataSize_ as input
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsCollAlgFactory] [InsReduceSoleExecutor] Host Orchestrate begins.");
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
        HCCL_ERROR("Reduce_[CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u].", myRank_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "Reduce_[InsReduceSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.InitReduceInfo(redOp_, dataType_);
    tempAlg.SetCollOp(op); // CCU template需要传递op信息
    tempAlg.SetRoot(root_);

    // 计算算法模板所需资源
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsReduceSoleExecutor] Rank[%d], CalcRes with detouring enabled for Orchestrate.",
            myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsReduceSoleExecutor] Rank[%d], CalcRes with detouring disabled for Orchestrate.",
            myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));

    ParamPool paramPool = {op_, params};
    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OFFLOAD Mode for HOST.", myRank_);
        CHK_RET(OrchestrateOffload(tempAlg, paramPool));
    } else { // OPBASE
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OPBASE Mode for HOST.", myRank_);
        CHK_RET(OrchestrateOpbase(tempAlg, paramPool));
    }

    return HcclResult::HCCL_SUCCESS;
}

// 算子执行aicpu接口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsCollAlgFactory] [InsReduceSoleExecutor] [InsReduceSoleExecutor] AiCpu Orchestrate begins.");
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
        HCCL_ERROR("Reduce_[CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u].", myRank_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "Reduce_[InsReduceSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.InitReduceInfo(redOp_, dataType_);
    tempAlg.SetCollOp(op); // CCU template需要传递op信息
    tempAlg.SetRoot(root_);

    // 计算算法模板所需资源
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsReduceSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] [InsReduceSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));

    ParamPool paramPool = {op_, params};
    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_DEBUG(
            "[InsReduceSoleExecutor] Rank[%d], Generating Instruction Queues in OFFLOAD Mode for AICPU.", myRank_);
        CHK_RET(OrchestrateOffload(tempAlg, paramPool));
    } else { // OPBASE
        HCCL_DEBUG(
            "[InsReduceSoleExecutor] Rank[%d], Generating Instruction Queues in OPBASE Mode for AICPU.", myRank_);
        CHK_RET(OrchestrateOpbase(tempAlg, paramPool));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateOffload(
    InsAlgTemplate& tempAlg, const ParamPool& paramPool)
{
    (void)paramPool;
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    CHK_PRT_RET(
        dataSizePerVolume == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid dataSizePerVolume [%u].", myRank_, dataSizePerVolume),
        HcclResult::HCCL_E_INTERNAL);
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::INPUT;
    buffInfo.outBuffType = BufferType::OUTPUT;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;
    HCCL_INFO(
        "[InsCollAlgFactory] Rank[%d], input buffer type [%s], output buffer type [%s], input buffer base "
        "offset [%u], output buffer base offset [%u].",
        myRank_, buffInfo.inBuffType.Describe().c_str(), buffInfo.outBuffType.Describe().c_str(),
        buffInfo.inBuffBaseOff, buffInfo.outBuffBaseOff);
    u64 sendRecvTimes = (dataSize_ / transportBoundDataSize) + ((dataSize_ % transportBoundDataSize) == 0 ? 0 : 1);
    HCCL_INFO("[CollAlgFactory] Rank [%d], sendRecvTimes [%u].", myRank_, sendRecvTimes);
    for (u64 idx = 0; idx < sendRecvTimes; idx++) {
        u64 currDataSize = (idx == (sendRecvTimes - 1)) ? (dataSize_ - idx * transportBoundDataSize) :
                                                          transportBoundDataSize; // 判断是否为最后一轮
        RankSliceInfo sliceInfoVec;
        AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
        TempFuncs tempFuncs;
        tempFuncs.opMode = opMode_;
        tempFuncs.enableCounterNotify = IsEnableCounterNotify();
        tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
        tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required
        UsrData usrData;
        // 将整块数据一次性从 UserIn 搬运到 CclIn 上
        usrData.usrInSlices.emplace_back(BufferType::INPUT, idx * transportBoundDataSize, currDataSize);
        usrData.scratchInSlices.emplace_back(BufferType::SCRATCH, 0, currDataSize);
        // 将整块数据一次性从 CclOut 搬运到 UserOut 上
        usrData.scratchOutSlices.emplace_back(BufferType::SCRATCH, 0, currDataSize);
        usrData.usrOutSlices.emplace_back(BufferType::OUTPUT, idx * transportBoundDataSize, currDataSize);
        tempFuncs.usrData = usrData;
        CHK_RET(tempAlg.CalcSliceInfo(allignInfo, transportBoundDataSize, sliceInfoVec));
        CHK_RET(tempAlg.Run(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    }
    HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], done generating instruction queues.", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateOpbase(InsAlgTemplate& tempAlg, ParamPool& paramPool)
{
    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::SCRATCH;
    buffInfo.outBuffType = BufferType::SCRATCH;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;

    // 基本参数配置
    AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
    TempFuncs tempFuncs;
    tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
    tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();

    u64 outputCount = dataCount_;
    // 根据CCL Buffer 大小，计算出一轮中最多能输出多少数据
    u64 maxLoopOutputCount = tempAlg.CalcLoopMaxCount(paramPool);
    if (maxLoopOutputCount == 0) {
        HCCL_ERROR("[InsReduceSoleExecutor][OrchestrateOpbase] maxLoopOutputCount is zero!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsReduceSoleExecutor][OrchestrateOpbase] Actual maxLoopOutputCount: [%llu].", maxLoopOutputCount);

    u64 loopTimes = outputCount / maxLoopOutputCount + static_cast<u64>(outputCount % maxLoopOutputCount != 0);
    for (u32 loop = 0; loop < loopTimes; loop++) {
        u64 loopOffsetCount = loop * maxLoopOutputCount;
        u64 loopOffsetSize = loopOffsetCount * dataTypeSize_;
        // 本轮需要处理的数据量
        u64 currOutputCount = (loop == (loopTimes - 1)) ? outputCount - loopOffsetCount : maxLoopOutputCount;
        u64 currOutputSize = currOutputCount * dataTypeSize_;
        UsrData usrData;
        // 将整块数据一次性从 UserIn 搬运到 CclIn 上
        usrData.usrInSlices.emplace_back(BufferType::INPUT, loopOffsetSize, currOutputSize);
        usrData.scratchInSlices.emplace_back(BufferType::SCRATCH, 0, currOutputSize);
        // 将整块数据一次性从 CclOut 搬运到 UserOut 上
        usrData.scratchOutSlices.emplace_back(BufferType::SCRATCH, 0, currOutputSize);
        usrData.usrOutSlices.emplace_back(BufferType::OUTPUT, loopOffsetSize, currOutputSize);
        tempFuncs.usrData = usrData;

        RankSliceInfo sliceInfoVec;
        CHK_RET(tempAlg.CalcSliceInfo(allignInfo, currOutputSize, sliceInfoVec));
        CHK_RET(tempAlg.Run(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    }
    return HcclResult::HCCL_SUCCESS;
}

#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCE, CcuReduceMesh2D, InsReduceSoleExecutor, TopoMatchConcurrMesh, CcuTempReduceMesh2D);
#endif
} // namespace Hccl
