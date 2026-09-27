/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_sole_executor.h"

#include "log.h"

#include "ins_coll_alg_registry.h"

#include "topo_match_mesh.h"
#include "topo_match_nhr.h"
#include "topo_match_concurr_mesh.h"

#include "ins_temp_all_gather_mesh.h"
#include "ins_temp_all_gather_mesh_2D.h"
#include "ins_temp_all_gather_nhr.h"
#ifndef CCL_KERNEL_AICPU
#include "ccu_temp_all_gather_mesh_1D_mem2mem_with_stride.h"
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"
#include "ccu_temp_all_gather_mesh_2D_mem2mem.h"
#include "aiv_temp_all_gather_mesh_1D.h"
#include "ccu_temp_all_gather_mesh_1D_2die.h"
#endif

namespace Hccl {
constexpr u64 MAX_OFFLOAD_SCRATCH_SIZE = 200 * 1024 * 1024; // 200M

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AllGatherSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsV2AllGatherSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const RankGraph* rankGraph)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const AlgTopoInfo& topoInfo)
{
    if (topoInfo.vTopo.size() < 1) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CreateTemplates(
    std::shared_ptr<InsAlgTemplate>& algTemplatePtr)
{
    HCCL_DEBUG("[InsV2AllGatherSoleExecutor][CreateTemplates]");
    algTemplatePtr = std::make_shared<InsAlgTemplate>(myRank_, rankSize_, vTopo_, virtRankMap_);
    CHK_PTR_NULL(algTemplatePtr); // 检查是否成功分配内存
    algTemplatePtr->SetDmaMode(dmaMode_);
    algTemplatePtr->SetDataType(dataType_);
    algTemplatePtr->SetCollOp(op_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    const RankGraph* rankGraph, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        HCCL_DEBUG("[InsV2AllGatherSoleExecutor] [%s] Rank[%d], CalcRes with detouring enabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsV2AllGatherSoleExecutor] [%s] Rank[%d], CalcRes with detouring disabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    ConnectedLinkMgr* linkMgr, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        HCCL_DEBUG("[InsV2AllGatherSoleExecutor] [%s] Rank[%d], CalcRes with detouring enabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_DEBUG("[InsV2AllGatherSoleExecutor] [%s] Rank[%d], CalcRes with detouring disabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));
    CHK_RET(algTemplate->CalNumBlocks(numBlocks, dataSize, numBlocksLimit));
    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_DEBUG("[InsV2AllGatherSoleExecutor][Orchestrate] Orchestrate HOST Start");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));

    HCCL_DEBUG(
        "[InsV2AllGatherSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

// AICPU 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_DEBUG("[InsV2AllGatherSoleExecutor][Orchestrate] Orchestrate AICPU Start");
    CHK_RET(Init(op, params, insQue));
    dataType_ = op.dataType;

    CHK_RET(InitCommInfo(topoInfo));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));
    algTemplate->SetDataType(dataType_);
    std::map<u32, u32> rank2PathNumMap;
    CHK_RET(SetPathNumMapByLinkMgrMultiLevel(linkMgr, virtRanks_, myRank_, rank2PathNumMap));
    algTemplate->setPathNumMap(rank2PathNumMap);

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(linkMgr, algTemplate, tempResReq));

    HCCL_DEBUG(
        "[InsV2AllGatherSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

// 切分数据并调用 template
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(std::shared_ptr<InsAlgTemplate> algTemplate)
{
    HCCL_INFO("[InsV2AllGatherSoleExecutor][OrchestrateOpbase] Start, template[%s]", algTemplate->Describe().c_str());
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;

    TemplateDataParams tempAlgParams;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParams.repeatNum = 1; // 不需要重复

    TempFuncs tempFuncs;
    tempFuncs.isForepart = true;
    tempFuncs.isBottom = true;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();

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
        "[InsV2AllGatherSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(
        maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2AllGatherSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);

    u64 processedDataCount = 0; // 已经处理的数据count
    u64 loopTimes = dataCount_ / maxDataCountPerLoop + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;

        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.scratchBuffBaseOff = 0;
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        tempAlgParams.inputSliceStride = 0;          // 输入数据仅有 1 个 slice, 不需要 stride
        tempAlgParams.outputSliceStride = dataSize_; // 每张卡的数据间隔为算子输入大小

        CHK_RET(algTemplate->GenExtIns(tempFuncs, tempAlgParams, tempResLinks_, tempInsQue_));
        processedDataCount += currDataCount;
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    // 通过判断哪层通信域能有到所有remoteRank的path，判断当前算法跑在哪一层
    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsV2AllGatherSoleExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    algTemplate->setPathNumMap(rank2PathNumMap);
    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    HCCL_DEBUG(
        "[InsV2AllGatherSoleExecutor] [%s] Rank[%d], requiredQueNum [%u].", __func__, myRank_, algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    // Topo Match
    (void)dataSize;
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    // 通过判断哪层通信域能有到所有remoteRank的path，判断当前算法跑在哪一层
    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsV2AllGatherSoleExecutor] CalcResOffload SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    algTemplate->setPathNumMap(rank2PathNumMap);
    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));
    resReq.requiredScratchMemSize = UB_MAX_DATA_SIZE;
    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLGATHER, InsAllGatherMesh, InsV2AllGatherSoleExecutor, TopoMatchMesh, InsTempAllGatherMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLGATHER, InsAllGatherMesh2D, InsV2AllGatherSoleExecutor, TopoMatchConcurrMesh, InsTempAllGatherMesh2D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLGATHER, InsAllGatherNHR, InsV2AllGatherSoleExecutor, TopoMatchNHR, InsTempAllGatherNHR);
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLGATHER, AivAllGatherMesh1D, InsV2AllGatherSoleExecutor, TopoMatchMesh, AivTempAllGatherMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLGATHER, CcuAllGatherMesh1DMem2MemWithStride, InsV2AllGatherSoleExecutor, TopoMatchMesh,
    CcuTempAllGatherMesh1DMem2MemWithStride);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLGATHER, CcuAllGatherNHR1D, InsV2AllGatherSoleExecutor, TopoMatchMesh, CcuTempAllGatherNHRMem2Mem1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLGATHER, CcuAllGatherMeshMem2Mem2D, InsV2AllGatherSoleExecutor, TopoMatchConcurrMesh,
    CcuTempAllGatherMeshMem2Mem2D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLGATHER, CcuAllGatherMesh1D2Die, InsV2AllGatherSoleExecutor, TopoMatchMesh, CcuTempAllGatherMesh1D2Die);
#endif
} // namespace Hccl
