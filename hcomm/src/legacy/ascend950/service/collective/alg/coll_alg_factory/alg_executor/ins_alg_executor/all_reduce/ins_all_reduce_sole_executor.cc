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
#include "ins_all_reduce_sole_executor.h"
#ifndef CCL_KERNEL_AICPU
#include "ccu_temp_all_reduce_mesh_1D.h"
#include "ccu_temp_all_reduce_mesh_1D_one_shot.h"
#include "ccu_temp_all_reduce_mesh_2D_one_shot.h"
#include "ccu_temp_all_reduce_mesh_2D_two_shot.h"
#include "ccu_temp_all_reduce_mesh_detour_1D.h"
#include "ccu_temp_all_reduce_mesh_2D_two_shot_mem2mem.h"
#endif
#include "topo_match_mesh.h"
#include "topo_match_concurr_mesh.h"

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsAllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsAllReduceSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsAllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsAllReduceSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
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

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[CalcResOffload] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[CalcResOffload] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

// 算子执行aicpu接口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsCollAlgFactory] [InsAllReduceSoleExecutor] Host Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));

    // soleEsecutor 只支持单层拓扑, 所以只取第 0 级通信域的信息
    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataTypeSize_;
    CHK_PRT_RET(
        dataTypeSize_ == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u] for AICPU.", myRank_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "[InsAllReduceSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s] for AICPU.",
        myRank_, rankSize_, dmaMode_.Describe().c_str());
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.InitReduceInfo(redOp_, dataType_);
    tempAlg.SetCollOp(op); // CCU template需要传递op信息

    // 计算算法模板所需资源
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsAllReduceSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsAllReduceSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));

    ParamPool paramPool = {op, params};
    HCCL_DEBUG("[InsAllReduceSoleExecutor] Rank[%d], Generating Instruction Queues for AICPU.", myRank_);
    CHK_RET(OrchestrateCommon(tempAlg, paramPool));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsAllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
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
        HCCL_DEBUG("[CalcRes] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[CalcRes] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    HCCL_DEBUG("[CalcRes] Rank[%d], requiredQueNum [%u].", myRank_, algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

// dataSize_ as input
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsCollAlgFactory] [InsAllReduceSoleExecutor] Host Orchestrate begins.");
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
        HCCL_ERROR("Allreduce_[CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u].", myRank_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "[InsAllReduceSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.InitReduceInfo(redOp_, dataType_);
    tempAlg.SetCollOp(op); // CCU template需要传递op信息

    // 计算算法模板所需资源
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsAllReduceSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsAllReduceSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));

    ParamPool paramPool = {op, params};
    HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues for HOST.", myRank_);
    CHK_RET(tempAlg.GetScratchBufferInfo(maxTmpMemSize_, dataType_));
    CHK_RET(OrchestrateCommon(tempAlg, paramPool));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateCommon(
    InsAlgTemplate& tempAlg, const ParamPool& paramPool)
{
    (void)paramPool;
    BuffInfo buffInfo;
    buffInfo.scratBuffType = BufferType::SCRATCH;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;
    buffInfo.scratchBuffBaseOff = 0;
    // 基本参数配置
    AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    if (opMode_ == OpMode::OFFLOAD) {
        buffInfo.inBuffType = BufferType::INPUT;
        buffInfo.outBuffType = BufferType::OUTPUT;
    } else {
        tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
        tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required
        buffInfo.inBuffType = BufferType::SCRATCH;
        buffInfo.outBuffType = BufferType::SCRATCH;
    }

    u64 outputCount = dataCount_;
    // 计算出一轮中最多能输出多少数据
    u64 maxLoopOutputCount = 0;
    if (opMode_ == OpMode::OFFLOAD) {
        maxLoopOutputCount = static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_;
    } else {
        maxLoopOutputCount = maxTmpMemSize_ / (rankSize_ * dataTypeSize_);
    }
    CHK_PRT_RET(
        maxLoopOutputCount == 0,
        HCCL_ERROR("[OrchestrateCommon] maxLoopOutputCount is zero, dataType[%s]", dataType_.Describe().c_str()),
        HcclResult::HCCL_E_PARA);
    u64 loopTimes = outputCount / maxLoopOutputCount + static_cast<u64>(outputCount % maxLoopOutputCount != 0);

    for (u32 loop = 0; loop < loopTimes; loop++) {
        // loopOffsetCount 是已经处理过的数据量
        u64 loopOffsetCount = loop * maxLoopOutputCount;
        u64 loopOffsetSize = loopOffsetCount * dataTypeSize_;

        // 本轮需要处理的数据量
        u64 currOutputCount = (loop == (loopTimes - 1)) ? outputCount - loopOffsetCount : maxLoopOutputCount;
        u64 currDataSize = currOutputCount * dataTypeSize_;

        UsrData usrData;
        // 将整块数据一次性从 UserIn 搬运到 CclIn 上
        usrData.usrInSlices.emplace_back(BufferType::INPUT, loopOffsetSize, currDataSize);
        usrData.scratchInSlices.emplace_back(BufferType::SCRATCH, 0, currDataSize);
        if (opMode_ == OpMode::OPBASE) {
            // 将整块数据一次性从 CclOut 搬运到 UserOut 上
            usrData.scratchOutSlices.emplace_back(BufferType::SCRATCH, 0, currDataSize);
        }
        usrData.usrOutSlices.emplace_back(BufferType::OUTPUT, loopOffsetSize, currDataSize);
        tempFuncs.usrData = usrData;

        RankSliceInfo sliceInfoVec;
        CHK_RET(tempAlg.CalcSliceInfo(allignInfo, currDataSize, sliceInfoVec));
        CHK_RET(tempAlg.Run(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    }
    return HcclResult::HCCL_SUCCESS;
}

// === CCU 算法注册 ===
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, CcuAllReduceMesh1D, InsAllReduceSoleExecutor, TopoMatchMesh, CcuTempAllReduceMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, CcuAllReduceMesh1DOneShot, InsAllReduceSoleExecutor, TopoMatchMesh,
    CcuTempAllReduceMesh1DOneShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, CcuAllReduceMesh2DOneShot, InsAllReduceSoleExecutor, TopoMatchConcurrMesh,
    CcuTempAllReduceMesh2DOneShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, CcuAllReduceMesh2DTwoShot, InsAllReduceSoleExecutor, TopoMatchConcurrMesh,
    CcuTempAllReduceMesh2DTwoShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, CcuAllReduceMeshDetour1D, InsAllReduceSoleExecutor, TopoMatchMesh, CcuTempAllReduceMeshDetour1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, CcuAllReduceMeshTwoShotMem2Mem2D, InsAllReduceSoleExecutor, TopoMatchConcurrMesh,
    CcuTempAllReduceMeshTwoShotMem2Mem2D);
#endif

} // namespace Hccl
