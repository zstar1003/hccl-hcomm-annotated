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
#include "topo_match_concurr_mesh.h"
#include "ins_temp_reduce_mesh_1D.h"
#include "ins_temp_reduce_mesh_1D_two_shot.h"
#include "ins_temp_reduce_mesh_2D.h"
#include "ins_temp_reduce_aicpu_reduce.h"
#include "ins_temp_reduce_aicpu_reduce_mesh_2D.h"
#include "topo_match_mesh.h"
#include "topo_match_nhr.h"
#include "ins_temp_reduce_nhr.h"
#include "ins_v2_reduce_sole_executor.h"
#ifndef CCL_KERNEL_AICPU
#include "aiv_temp_reduce_mesh_1D.h"
#include "ccu_temp_reduce_mesh_1D.h"
#include "ccu_temp_reduce_nhr_1D_mem2mem.h"
#include "ccu_temp_reduce_mesh_1D_mem2mem.h"
#include "ccu_temp_reduce_mesh_2D_mem2mem.h"
#include "ccu_temp_reduce_mesh_1D_two_shot_mem2mem.h"
#endif

namespace Hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2ReduceSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsV2ReduceSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const RankGraph* rankGraph)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const AlgTopoInfo& topoInfo)
{
    CHK_PRT_RET(
        topoInfo.vTopo.empty(), HCCL_ERROR("[InsV2ReduceSoleExecutor][InitCommInfo] vTopo size is invalid"),
        HCCL_E_PARA);
    CHK_PRT_RET(
        topoInfo.virtRankMap.empty(), HCCL_ERROR("[InsV2ReduceSoleExecutor][InitCommInfo] virtRankMap size is invalid"),
        HCCL_E_PARA);
    CHK_PRT_RET(
        topoInfo.virtRanks.empty(), HCCL_ERROR("[InsV2ReduceSoleExecutor][InitCommInfo] virtRanks size is invalid"),
        HCCL_E_PARA);
    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CreateTemplates(std::shared_ptr<InsAlgTemplate>& algTemplatePtr)
{
    algTemplatePtr = std::make_shared<InsAlgTemplate>(myRank_, rankSize_, vTopo_, virtRankMap_);
    CHK_PTR_NULL(algTemplatePtr); // 检查是否成功分配内存
    algTemplatePtr->SetDmaMode(dmaMode_);
    algTemplatePtr->InitReduceInfo(redOp_, dataType_);
    algTemplatePtr->SetCollOp(op_);
    algTemplatePtr->SetRoot(op_.root);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    const RankGraph* rankGraph, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        HCCL_DEBUG("[InsV2ReduceSoleExecutor] [%s] Rank[%d], CalcRes with detouring enabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsV2ReduceSoleExecutor] [%s] Rank[%d], CalcRes with detouring disabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    ConnectedLinkMgr* linkMgr, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detouring enabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detouring disabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

// AICPU 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(topoInfo));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(linkMgr, algTemplate, tempResReq));
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

// 切分数据并调用 template
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(std::shared_ptr<InsAlgTemplate> algTemplate)
{
    HCCL_INFO("[InsReduceSoleExecutor][OrchestrateLoop] Start, template[%s]", algTemplate->Describe().c_str());
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;

    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParams.repeatNum = 1; // 不需要重复
    tempAlgParams.outputRepeatStride = 0;
    tempAlgParams.inputRepeatStride = 0;

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.isForepart = true;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.isBottom = true;

    u64 maxDataSizePerLoop = 0;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE; // algTemplate->CalcLoopMaxCount();
    u32 templateScratchMultiplier = algTemplate->CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    HCCL_INFO(
        "[InsReduceSoleExecutor][OrchestrateLoop] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(
        maxDataCountPerLoop == 0, HCCL_ERROR("[InsReduceSoleExecutor][OrchestrateLoop] maxDataCountPerLoop is 0"),
        HCCL_E_INTERNAL);

    tempAlgParams.buffInfo.scratchBuffBaseOff = 0;
    tempAlgParams.inputSliceStride = 0;          // 输入数据仅有 1 个 slice, 不需要 stride
    tempAlgParams.outputSliceStride = dataSize_; // 每张卡的数据间隔为算子输入大小

    u64 processedDataCount = 0;
    u64 loopTimes = dataCount_ / maxDataCountPerLoop + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;

        CHK_RET(algTemplate->GenExtIns(tempFuncs, tempAlgParams, tempResLinks_, tempInsQue_));
        processedDataCount += currDataCount;
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));
    u32 templateScratchMultiplier = algTemplate->CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    resReq.requiredScratchMemSize = std::min(dataSize * templateScratchMultiplier, transportBoundDataSize);
    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

INS_REGISTER_IMPL_BY_TEMP(OpType::REDUCE, InsReduceMesh1D, InsV2ReduceSoleExecutor, TopoMatchMesh, InsTempReduceMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCE, InsReduceMesh1DTwoShot, InsV2ReduceSoleExecutor, TopoMatchMesh, InsTempReduceMesh1DTwoShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCE, InsReduceMesh2D, InsV2ReduceSoleExecutor, TopoMatchConcurrMesh, InsTempReduceMesh2D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCE, InsReduceAicpuReduce, InsV2ReduceSoleExecutor, TopoMatchMesh, InsTempReduceAicpuReduce);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCE, InsReduceAicpuReduceMesh2D, InsV2ReduceSoleExecutor, TopoMatchConcurrMesh,
    InsTempReduceAicpuReduceMesh2D);
INS_REGISTER_IMPL_BY_TEMP(OpType::REDUCE, InsReduceNHR, InsV2ReduceSoleExecutor, TopoMatchNHR, InsTempReduceNHR);
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(OpType::REDUCE, CcuReduceMesh1D, InsV2ReduceSoleExecutor, TopoMatchMesh, CcuTempReduceMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCE, CcuReduceNHR1D, InsV2ReduceSoleExecutor, TopoMatchMesh, CcuTempReduceNHRMem2Mem1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCE, CcuReduceMeshMem2Mem1D, InsV2ReduceSoleExecutor, TopoMatchMesh, CcuTempReduceMeshMem2Mem1D);
INS_REGISTER_IMPL_BY_TEMP(OpType::REDUCE, AivReduceMesh1D, InsV2ReduceSoleExecutor, TopoMatchMesh, AivTempReduceMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCE, CcuReduceMeshMem2Mem2D, InsV2ReduceSoleExecutor, TopoMatchConcurrMesh, CcuTempReduceMeshMem2Mem2D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCE, CcuReduceMeshTwoShotMem2Mem1D, InsV2ReduceSoleExecutor, TopoMatchMesh,
    CcuTempReduceMeshTwoShotMem2Mem1D);
#endif
} // namespace Hccl
