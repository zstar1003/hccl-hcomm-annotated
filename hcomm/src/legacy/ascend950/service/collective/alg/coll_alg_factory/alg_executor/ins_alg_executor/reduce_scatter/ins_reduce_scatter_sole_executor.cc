/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "ins_coll_alg_registry.h"
#include "ins_reduce_scatter_sole_executor.h"
#ifndef CCL_KERNEL_AICPU
#include "ccu_temp_reduce_scatter_mesh_1D.h"
#include "ccu_temp_reduce_scatter_mesh_2D.h"
#include "ccu_temp_reduce_scatter_mesh_2D_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_detour_1D.h"
#endif

#include "topo_match_mesh.h"
#include "topo_match_concurr_mesh.h"

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsReduceScatterSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsReduceScatterSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    resReq.requiredScratchMemSize = 200 * 1024 * 1024; // scratch memory size 200 * 1024K

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.InitReduceInfo(redOp_, dataType_);

    // calculate required insQueues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsReduceScatterSoleExecutor], CalcRes with detouring enabled.");
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] [InsReduceScatterSoleExecutor], CalcRes with detouring disabled.");
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    const RankGraph* rankGraph, CollAlgResReq& algResReq)
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
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsReduceScatterSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsReduceScatterSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    HCCL_DEBUG(
        "[InsCollAlgFactory] [InsReduceScatterSoleExecutor] Rank[%d], requiredQueNum [%u].", myRank_,
        algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

// dataSize_ as input
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsCollAlgFactory] [InsReduceScatterSoleExecutor] Host Orchestrate begins.");
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
        HCCL_ERROR("Reducescatter_[CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u].", myRank_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "[InsReduceScatterSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.InitReduceInfo(redOp_, dataType_);
    tempAlg.SetCollOp(op); // CCU template需要传递op信息

    // 计算算法模板所需资源
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsReduceScatterSoleExecutor] Rank[%d], CalcRes with detouring enabled for "
            "Orchestrate.",
            myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsReduceScatterSoleExecutor] Rank[%d], CalcRes with detouring disabled for "
            "Orchestrate.",
            myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));
    ParamPool paramPool = {op, params};
    HCCL_DEBUG(
        "[InsCollAlgFactory] [InsReduceScatterSoleExecutor] Rank[%d], Generating Instruction Queues in OFFLOAD Mode "
        "for HOST.",
        myRank_);
    CHK_RET(OrchestrateLoop(tempAlg, paramPool));

    return HcclResult::HCCL_SUCCESS;
}

// 算子执行aicpu接口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    CHK_PTR_NULL(linkMgr);
    HCCL_INFO("[InsCollAlgFactory] [InsReduceScatterSoleExecutor] AiCpu Orchestrate begins.");
    // 参数校验和初始化
    CHK_RET(Init(op, params, insQue));

    // soleEsecutor 只支持单层拓扑, 所以只取第0级通信域的信息
    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataTypeSize_;
    CHK_PRT_RET(
        dataTypeSize_ == 0,
        HCCL_ERROR("Reducescatter_[CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u].", myRank_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "[InsReduceScatterSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.InitReduceInfo(redOp_, dataType_);
    tempAlg.SetCollOp(op); // CCU template需要传递op信息

    // 计算算法模板所需资源
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsReduceScatterSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsReduceScatterSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));
    ParamPool paramPool = {op, params};
    HCCL_DEBUG(
        "[InsReduceScatterSoleExecutor] Rank[%d], Generating Instruction Queues in OFFLOAD Mode for AICPU.", myRank_);
    CHK_RET(OrchestrateLoop(tempAlg, paramPool));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    InsAlgTemplate& tempAlg, const ParamPool& paramPool)
{
    (void)paramPool;
    BuffInfo buffInfo;
    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    if (opMode_ == OpMode::OFFLOAD) {
        buffInfo.inBuffType = BufferType::INPUT;
        buffInfo.outBuffType = BufferType::OUTPUT;
    } else {
        buffInfo.inBuffType = BufferType::SCRATCH;
        buffInfo.outBuffType = BufferType::SCRATCH;
        tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
        tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required
    }
    buffInfo.scratBuffType = BufferType::SCRATCH;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;
    HCCL_DEBUG(
        "[InsCollAlgFactory] Rank[%d], input buffer type [%s], output buffer type [%s], input buffer base "
        "offset [%u], output buffer base offset [%u].",
        myRank_, buffInfo.inBuffType.Describe().c_str(), buffInfo.outBuffType.Describe().c_str(),
        buffInfo.inBuffBaseOff, buffInfo.outBuffBaseOff);
    u64 outputCount = dataCount_;
    u64 outputSize = outputCount * dataTypeSize_;
    CHK_PRT_RET(rankSize_ == 0, HCCL_ERROR("[CollAlgFactory] RankSize is zero!"), HcclResult::HCCL_E_PARA);
    AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
    HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], done calculating slice information.", myRank_);

    u64 maxLoopOutputCount = 0;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_INFO("[InsV2ReduceScatterSoleExecutor]transportBoundDataSize [%u]", transportBoundDataSize);
        maxLoopOutputCount = transportBoundDataSize / dataTypeSize_;
    } else {
        HCCL_INFO("[InsV2ReduceScatterSoleExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
        maxLoopOutputCount = maxTmpMemSize_ / (rankSize_ * dataTypeSize_);
    }
    CHK_PRT_RET(
        maxLoopOutputCount == 0, HCCL_ERROR("[InsReduceScatterSoleExecutor] maxLoopOutputCount is zero!"),
        HcclResult::HCCL_E_PARA);
    u64 loopTimes = outputCount / maxLoopOutputCount + static_cast<u64>(outputCount % maxLoopOutputCount != 0);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 loopOffsetCount = loop * maxLoopOutputCount;
        u64 loopOffsetSize = loopOffsetCount * dataTypeSize_;
        // 本轮需要处理的output数据总量
        u64 currOutputCount = (loop == (loopTimes - 1)) ? outputCount - loopOffsetCount : maxLoopOutputCount;
        u64 currSliceSize = currOutputCount * dataTypeSize_;
        UsrData usrData;
        for (RankId rankId : virtRanks_) {
            u32 rankIdx = virtRankMap_[rankId];
            u64 rankOffsetSize = rankIdx * outputSize;
            // 需要处理每一个rank的userIn数据
            usrData.usrInSlices.emplace_back(BufferType::INPUT, rankOffsetSize + loopOffsetSize, currSliceSize);
            // userIn数据搬到CCLIn上时的对应位置
            usrData.scratchInSlices.emplace_back(BufferType::SCRATCH, rankIdx * currSliceSize, currSliceSize);
        }
        // 直接将对应rank的整块数据搬到userOut上
        usrData.scratchOutSlices.emplace_back(
            BufferType::SCRATCH, virtRankMap_[myRank_] * currSliceSize, currSliceSize);
        usrData.usrOutSlices.emplace_back(BufferType::OUTPUT, loopOffsetSize, currSliceSize);
        tempFuncs.usrData = usrData;
        RankSliceInfo sliceInfoVec;
        CHK_RET(tempAlg.CalcSliceInfo(allignInfo, currSliceSize, sliceInfoVec));
        CHK_RET(tempAlg.Run(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    }
    return HcclResult::HCCL_SUCCESS;
}

#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, CcuReduceScatterMesh1D, InsReduceScatterSoleExecutor, TopoMatchMesh,
    CcuTempReduceScatterMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, CcuReduceScatterMesh2D, InsReduceScatterSoleExecutor, TopoMatchConcurrMesh,
    CcuTempReduceScatterMesh2D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, CcuReduceScatterMeshMem2Mem2D, InsReduceScatterSoleExecutor, TopoMatchConcurrMesh,
    CcuTempReduceScatterMeshMem2Mem2D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, CcuReduceScatterMeshDetour1D, InsReduceScatterSoleExecutor, TopoMatchMesh,
    CcuTempReduceScatterMeshDetour1D);
#endif
} // namespace Hccl
