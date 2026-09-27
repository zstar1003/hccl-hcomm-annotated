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
#include "topo_match_nhr.h"
#include "topo_match_mesh.h"
#include "topo_match_concurr_mesh.h"
#include "ins_temp_reduce_scatter_mesh_1D.h"
#include "ins_temp_reduce_scatter_mesh_1D_meshchunk.h"
#include "ins_temp_reduce_scatter_mesh_2D.h"
#include "ins_temp_reduce_scatter_nhr.h"
#ifndef CCL_KERNEL_AICPU
#include "aiv_temp_reduce_scatter_mesh_1D.h"
#include "ccu_temp_reduce_scatter_nhr_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_2die.h"
#endif
#include "ins_v2_reduce_scatter_sole_executor.h"
#include "ins_temp_reduce_scatter_aicpu_reduce.h"
#include "ins_temp_reduce_scatter_aicpu_reduce_mesh_2D.h"
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"

namespace Hccl {
constexpr u64 MAX_OFFLOAD_SCRATCH_SIZE = 200 * 1024 * 1024; // 200M
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2ReduceScatterSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsV2ReduceScatterSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const RankGraph* rankGraph)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const AlgTopoInfo& topoInfo)
{
    CHK_PRT_RET(
        topoInfo.vTopo.empty(), HCCL_ERROR("[InsV2ReduceScatterSoleExecutor][InitCommInfo] vTopo size is invalid"),
        HCCL_E_PARA);
    CHK_PRT_RET(
        topoInfo.virtRankMap.empty(),
        HCCL_ERROR("[InsV2ReduceScatterSoleExecutor][InitCommInfo] virtRankMap size is invalid"), HCCL_E_PARA);
    CHK_PRT_RET(
        topoInfo.virtRanks.empty(),
        HCCL_ERROR("[InsV2ReduceScatterSoleExecutor][InitCommInfo] virtRanks size is invalid"), HCCL_E_PARA);
    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CreateTemplates(
    std::shared_ptr<InsAlgTemplate>& algTemplatePtr)
{
    HCCL_DEBUG("[InsV2ReduceScatterSoleExecutor][CreateTemplates]");
    algTemplatePtr = std::make_shared<InsAlgTemplate>(myRank_, rankSize_, vTopo_, virtRankMap_);
    CHK_PTR_NULL(algTemplatePtr);
    algTemplatePtr->SetDmaMode(dmaMode_);
    algTemplatePtr->SetDataType(dataType_);
    algTemplatePtr->SetCollOp(op_);
    algTemplatePtr->InitReduceInfo(redOp_, dataType_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalNumBlocks(
    u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));
    CHK_RET(algTemplate->CalNumBlocks(numBlocks, dataSize, numBlocksLimit));
    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口，将对应的instruction添加到指令队列中
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][Orchestrate] Orchestrate host Start");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(rankGraph));
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;
    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsV2ReduceScatterSoleExecutor][Orchestrate] [%s] Rank[%d], CalcRes with detouring enabled.", __func__,
            myRank_);
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsV2ReduceScatterSoleExecutor][Orchestrate] [%s] Rank[%d], CalcRes with detouring disabled.", __func__,
            myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }

    HCCL_DEBUG(
        "[InsV2ReduceScatterSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor][Orchestrate] Orchestrate AICPU Start");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(topoInfo));
    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;
    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    std::map<u32, u32> rank2PathNumMap;
    CHK_RET(SetPathNumMapByLinkMgrMultiLevel(linkMgr, virtRanks_, myRank_, rank2PathNumMap));
    algTemplate->setPathNumMap(rank2PathNumMap);

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        CHK_RET(algTemplate->CalcResDetour(linkMgr, tempResReq));
    } else {
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

// 单算子模式资源计算接口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    std::shared_ptr<InsAlgTemplate> algTemplate)
{
    HCCL_INFO(
        "[InsV2ReduceScatterSoleExecutor][OrchestrateOpbase] Start, template[%s]", algTemplate->Describe().c_str());

    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;

    u64 maxDataSizePerLoop = 0;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
    u32 templateScratchMultiplier
        = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);
    if (templateScratchMultiplier != 0) {
        // reduce 时 保障scratch上偏移按照dataTypeSize_对齐
        u64 scratchBoundDataSize = (maxTmpMemSize_ / templateScratchMultiplier) / dataTypeSize_ * dataTypeSize_;
        maxDataSizePerLoop = min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_; // 单次循环处理的数据量大小，同时会处理两片数据
    HCCL_INFO(
        "[InsV2ReduceScatterSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(
        maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2ReduceScatterSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);
    // 这里处理的数据量，是单次循环所处理的总数据量，包括两个数据片，每一半stream处理一个数据片
    TempFuncs tempFuncs;
    tempFuncs.isForepart = true;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.isBottom = true;
    // maxDataCountPerLoop是一次循环所处理的一片数据量大小
    u64 processedDataCount = 0;
    u64 allDataCountPerLoop = maxDataCountPerLoop;
#ifdef CCL_KERNEL_AICPU
    if (vTopo_.size() > 1) { // aicpu mesh 2d
        allDataCountPerLoop = maxDataCountPerLoop * 2;
    }
#endif
    u64 loopTimes = dataCount_ / allDataCountPerLoop + static_cast<u64>(dataCount_ % allDataCountPerLoop != 0);
    HCCL_INFO(
        "[InsV2ReduceScatterSoleExecutor]allDataCountPerLoop [%u],dataCount_ [%u],loopTimes [%u]", allDataCountPerLoop,
        dataCount_, loopTimes);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : allDataCountPerLoop;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.scratchBuffBaseOff = 0;

        tempAlgParams.sliceSize = currDataCount * dataTypeSize_; // 这里是单次循环处理的两片数据的大小
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        // 这里的stride当成传统意义上的sreide间隔
        tempAlgParams.inputSliceStride = dataSize_; // 如果是输入，偏移是算子的output datasize
        tempAlgParams.outputSliceStride = maxDataSizePerLoop; // 如果是scratchbuffer，偏移是单次循环所处理的最大数据量
        HCCL_INFO(
            "[InsV2ReduceScatterSoleExecutor] loop [%u] tempAlgParams.inputSliceStride "
            "[%u],tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
            loop, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride, tempAlgParams.sliceSize);
        HCCL_INFO(
            "[InsV2ReduceScatterSoleExecutor] loop [%u] tempAlgParams.buffInfo.inBuffBaseOff "
            "[%u],tempAlgParams.buffInfo.outBuffBaseOff [%u]",
            loop, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParams.repeatNum = 1;
        tempAlgParams.inputRepeatStride = 0;
        tempAlgParams.outputRepeatStride = 0;

        CHK_RET(algTemplate->GenExtIns(tempFuncs, tempAlgParams, tempResLinks_, tempInsQue_));
        processedDataCount += currDataCount;
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    // 通过判断哪层通信域能有到所有remoteRank的path，判断当前算法跑在哪一层
    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    tempAlg.setPathNumMap(rank2PathNumMap);

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsV2ReduceScatterSoleExecutor][CalcRes] [%s] Rank[%d], CalcRes with detouring enabled.", __func__,
            myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsV2ReduceScatterSoleExecutor][CalcRes] [%s] Rank[%d], CalcRes with detouring disabled.", __func__,
            myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    HCCL_DEBUG("[%s] Rank[%d], requiredQueNum [%u].", __func__, myRank_, algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;

    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    // 通过判断哪层通信域能有到所有remoteRank的path，判断当前算法跑在哪一层
    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsV2ReduceScatterSoleExecutor] CalcResOffload SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    algTemplate->setPathNumMap(rank2PathNumMap);

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsV2ReduceScatterSoleExecutor][CalcResOffload] [%s] Rank[%d], CalcRes with detouring enabled.", __func__,
            myRank_);
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsV2ReduceScatterSoleExecutor][CalcResOffload] [%s] Rank[%d], CalcRes with detouring disabled.",
            __func__, myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    resReq.requiredScratchMemSize = UB_MAX_DATA_SIZE;
    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, InsReduceScatterMesh1D, InsV2ReduceScatterSoleExecutor, TopoMatchMesh,
    InsTempReduceScatterMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, InsReduceScatterMesh1DMeshChunk, InsV2ReduceScatterSoleExecutor, TopoMatchMesh,
    InsTempReduceScatterMesh1DMeshChunk);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, InsReduceScatterNHR, InsV2ReduceScatterSoleExecutor, TopoMatchNHR, InsTempReduceScatterNHR);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, InsReduceScatterMesh2D, InsV2ReduceScatterSoleExecutor, TopoMatchConcurrMesh,
    InsTempReduceScatterMesh2D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, InsReduceScatterAicpuReduce, InsV2ReduceScatterSoleExecutor, TopoMatchMesh,
    InsTempReduceScatterAicpuReduce);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, InsReduceScatterAicpuReduceMesh2D, InsV2ReduceScatterSoleExecutor, TopoMatchConcurrMesh,
    InsTempReduceScatterAicpuReduceMesh2D);
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, AivReduceScatterMesh1D, InsV2ReduceScatterSoleExecutor, TopoMatchMesh,
    AivTempReduceScatterMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, CcuReduceScatterMeshMem2Mem1D, InsV2ReduceScatterSoleExecutor, TopoMatchMesh,
    CcuTempReduceScatterMeshMem2Mem1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, CcuReduceScatterNHR1DMem2Mem, InsV2ReduceScatterSoleExecutor, TopoMatchMesh,
    CcuTempReduceScatterNHR1DMem2Mem);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTER, CcuReduceScatterMesh1D2Die, InsV2ReduceScatterSoleExecutor, TopoMatchMesh,
    CcuTempReduceScatterMesh1D2Die);
#endif
} // namespace Hccl
