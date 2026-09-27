/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_sole_executor.h"
#include "log.h"
#include "ins_coll_alg_registry.h"

#include "topo_match_mesh.h"
#include "topo_match_nhr.h"
#include "topo_match_concurr_mesh.h"
#include "ccu_temp_broadcast_mesh_1D_mem2mem.h"
#include "ins_temp_broadcast_mesh1D_oneshot.h"
#include "ins_temp_broadcast_mesh_2D_two_shot.h"
#include "ins_temp_broadcast_mesh_1D_two_shot.h"
#include "ins_temp_broadcast_nhr.h"

#ifndef CCL_KERNEL_AICPU
#include "ccu_temp_broadcast_mesh_2D_mem2mem.h"
#include "aiv_temp_broadcast_mesh_1D.h"
#include "ccu_temp_broadcast_nhr_1D_mem2mem.h"
#endif

namespace Hccl {
constexpr u64 MAX_OFFLOAD_SCRATCH_SIZE = 200 * 1024 * 1024; // 200M
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2BroadcastSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsV2BroadcastSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const RankGraph* rankGraph)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const AlgTopoInfo& topoInfo)
{
    CHK_PRT_RET(
        topoInfo.vTopo.size() == 0, HCCL_ERROR("[InsV2BroadcastSoleExecutor] [Orchestrate] vTopo size is 0."),
        HcclResult::HCCL_E_INTERNAL);
    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    const RankGraph* rankGraph, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        HCCL_DEBUG("[InsV2BroadcastSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsV2BroadcastSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    ConnectedLinkMgr* linkMgr, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d]. CalcRes with detouring enabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detouring disabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    resReq.requiredScratchMemSize = 0;

    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));
    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));
    resReq.requiredScratchMemSize = MAX_OFFLOAD_SCRATCH_SIZE;
    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CreateTemplates(
    std::shared_ptr<InsAlgTemplate>& algTemplatePtr)
{
    algTemplatePtr = std::make_shared<InsAlgTemplate>(myRank_, rankSize_, vTopo_, virtRankMap_);
    CHK_PTR_NULL(algTemplatePtr); // 检查是否成功分配内存
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    algTemplatePtr->SetDmaMode(dmaMode_);
    algTemplatePtr->SetCollOp(op_); // CCU template需要传递op信息
    algTemplatePtr->SetRoot(root_);
    algTemplatePtr->SetDataType(dataType_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    HCCL_DEBUG(
        "[InsV2BroadcastSoleExecutor][InsCollAlgFactory] Rank[%d], requiredQueNum [%u].", myRank_,
        algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

// host
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsCollAlgFactory] [InsV2BroadcastSoleExecutor] Host Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));

    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataTypeSize_;
    CHK_PRT_RET(
        dataTypeSize_ == 0,
        HCCL_ERROR(
            "[InsV2BroadcastSoleExecutor] [CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u].", myRank_,
            dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "[InsV2BroadcastSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    std::shared_ptr<InsAlgTemplate> tempAlg = nullptr;
    CHK_RET(CreateTemplates(tempAlg));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, tempAlg, tempResReq));
    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));

    CHK_RET(OrchestrateLoop(tempAlg));

    return HcclResult::HCCL_SUCCESS;
}

// 算子执行aicpu接口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsCollAlgFactory] [InsV2BroadcastSoleExecutor] AiCpu Orchestrate begins.");
    // 参数校验和初始化
    CHK_RET(Init(op, params, insQue));
    // soleEsecutor 只支持单层拓扑, 所以只取第 0 级通信域的信息
    CHK_RET(InitCommInfo(topoInfo));
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataTypeSize_;
    CHK_PRT_RET(
        dataTypeSize_ == 0,
        HCCL_ERROR(
            "[InsV2BroadcastSoleExecutor] [CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u].", myRank_,
            dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "[InsV2BroadcastSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    std::shared_ptr<InsAlgTemplate> tempAlg = nullptr;
    CHK_RET(CreateTemplates(tempAlg));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(linkMgr, tempAlg, tempResReq));

    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));

    CHK_RET(OrchestrateLoop(tempAlg));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2BroadcastSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(std::shared_ptr<InsAlgTemplate>& tempAlg)
{
    // 基本参数配置
    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
    tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required

    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParams.buffInfo.scratchBuffBaseOff = 0;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    // 不需要重复
    tempAlgParams.repeatNum = 1;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;

    // 根据CCL Buffer大小和UB_MAX_DATA_SIZE，计算出一轮中最多能输出多少数据
    u64 maxDataSizePerLoop = 0;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE; // algTemplate->CalcLoopMaxCount();
    u32 templateScratchMultiplier = tempAlg->CalcScratchMultiple(BufferType::INPUT, BufferType::INPUT);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier;
        maxDataSizePerLoop = min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;

    u64 dataSize = dataCount_ * dataTypeSize_;

    u64 maxLoopOutputSize = maxDataCountPerLoop * dataTypeSize_;

    u64 loopTimes = dataSize / maxLoopOutputSize + static_cast<u64>(dataSize % maxLoopOutputSize != 0);

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currloopOffset = loop * maxLoopOutputSize;
        u64 currSize = (loop == (loopTimes - 1)) ? dataSize - currloopOffset : maxLoopOutputSize;
        // 当前搬运的数据片
        tempAlgParams.buffInfo.inBuffBaseOff = currloopOffset;
        tempAlgParams.buffInfo.outBuffBaseOff = currloopOffset;

        tempAlgParams.sliceSize = currSize;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;

        CHK_RET(tempAlg->GenExtIns(tempFuncs, tempAlgParams, tempResLinks_, requiredQue_));
        HCCL_DEBUG(
            "[InsCollAlgFactory] Rank[%d], done generating instruction queues, currSize[%llu], currOffset[%llu].",
            myRank_, currSize, currloopOffset);
    }

    return HcclResult::HCCL_SUCCESS;
}

INS_REGISTER_IMPL_BY_TEMP(
    OpType::BROADCAST, InsBroadcastMesh1DOneShot, InsV2BroadcastSoleExecutor, TopoMatchMesh,
    InsTempBroadcastMesh1DOneShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::BROADCAST, InsBroadcastMesh1DTwoShot, InsV2BroadcastSoleExecutor, TopoMatchMesh,
    InsTempBroadcastMesh1DTwoShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::BROADCAST, InsBroadcastMesh2DTwoShot, InsV2BroadcastSoleExecutor, TopoMatchConcurrMesh,
    InsTempBroadcastMesh2DTwoShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::BROADCAST, InsBroadcastNHR, InsV2BroadcastSoleExecutor, TopoMatchNHR, InsTempBroadcastNHR);
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::BROADCAST, CcuBroadcastMeshMem2Mem1D, InsV2BroadcastSoleExecutor, TopoMatchMesh,
    CcuTempBroadcastMesh1DMem2Mem);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::BROADCAST, CcuBroadcastMeshMem2Mem2D, InsV2BroadcastSoleExecutor, TopoMatchConcurrMesh,
    CcuTempBroadcastMeshMem2Mem2D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::BROADCAST, AivBroadcastMesh1D, InsV2BroadcastSoleExecutor, TopoMatchMesh, AivTempBroadcastMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::BROADCAST, CcuBroadcastNHRMem2Mem1D, InsV2BroadcastSoleExecutor, TopoMatchMesh,
    CcuTempBroadcastNHRMem2Mem1D);
#endif
} // namespace Hccl
