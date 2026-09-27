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
#include "topo_match_mesh.h"
#include "ins_v2_reduce_scatter_v_sole_executor.h"

#ifndef CCL_KERNEL_AICPU
#include "ccu_temp_reduce_scatter_v_mesh_1D.h"
#include "ccu_temp_reduce_scatter_v_mesh_1D_mem2mem.h"
#endif

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ReduceScatterVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2ReduceScatterVSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ReduceScatterVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsV2ReduceScatterVSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const RankGraph* rankGraph)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const AlgTopoInfo& topoInfo)
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
HcclResult InsV2ReduceScatterVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CreateTemplates(
    std::shared_ptr<InsAlgTemplate>& algTemplatePtr)
{
    HCCL_DEBUG("[InsV2ReduceScatterVSoleExecutor][CreateTemplates]");
    algTemplatePtr = std::make_shared<InsAlgTemplate>(myRank_, rankSize_, vTopo_, virtRankMap_);
    CHK_PTR_NULL(algTemplatePtr);
    algTemplatePtr->SetDmaMode(dmaMode_);
    algTemplatePtr->SetDataType(dataType_);
    algTemplatePtr->SetCollOp(op_);
    algTemplatePtr->InitReduceInfo(redOp_, dataType_);
    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口，将对应的 instruction 添加到指令队列中
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsV2ReduceScatterVSoleExecutor][Orchestrate] Orchestrate host Start");

    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(rankGraph));
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;
    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsV2ReduceScatterVSoleExecutor][Orchestrate] [%s] Rank[%d], CalcRes with detouring enabled.", __func__,
            myRank_);
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsV2ReduceScatterVSoleExecutor][Orchestrate] [%s] Rank[%d], CalcRes with detouring disabled.", __func__,
            myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }

    HCCL_DEBUG(
        "[InsV2ReduceScatterVSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsV2ReduceScatterVSoleExecutor][Orchestrate] Orchestrate AICPU Start");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(topoInfo));
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;
    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        CHK_RET(algTemplate->CalcResDetour(linkMgr, tempResReq));
    } else {
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    HCCL_DEBUG(
        "[InsV2ReduceScatterVSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

// 单算子模式资源计算接口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    std::shared_ptr<InsAlgTemplate> algTemplate)
{
    HCCL_INFO(
        "[InsV2ReduceScatterVSoleExecutor][OrchestrateOpbase] Start, template[%s]", algTemplate->Describe().c_str());

    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParams.repeatNum = 1; // 不需要重复
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.isForepart = true;
    tempFuncs.isBottom = true;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();

    u64 maxDataSizePerLoop = 0;
    u64 transportBoundDataSize;
    CHK_RET(algTemplate->GetMaxTransPortDataSize(transportBoundDataSize));
    u64 templateScratchMultiplier
        = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);
    if (templateScratchMultiplier != 0) {
        Buffer* scratchBuffer = op_.GetBuffer(BufferType::SCRATCH);
        CHK_PRT_RET(
            scratchBuffer == nullptr || scratchBuffer->GetAddr() == 0 || scratchBuffer->GetSize() == 0,
            HCCL_ERROR("Scratch buffer is null while ScratchMultiplier is not 0"), HCCL_E_MEMORY);
        u64 scratchBoundDataSize = scratchBuffer->GetSize() / templateScratchMultiplier;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    HCCL_INFO(
        "[InsAllReduceCombExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(
        maxDataCountPerLoop == 0, HCCL_ERROR("[InsAllReduceCombExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(
        op_.vDataDes.counts == nullptr || op_.vDataDes.displs == nullptr,
        HCCL_ERROR("[InsAllReduceCombExecutor][OrchestrateOpbase] counts or displs is nullptr"), HCCL_E_PTR);
    u64 myRankRecvCount = static_cast<u64*>(op_.vDataDes.counts)[myRank_];
    u64 myDisplacementCount = static_cast<u64*>(op_.vDataDes.displs)[myRank_];
    u64 maxRecvDataCount = 0;
    for (u64 i = 0; i < rankSize_; i++) {
        maxRecvDataCount = max(maxRecvDataCount, static_cast<u64*>(op_.vDataDes.counts)[i]);
    }

    u64 loopTimes = 1 + ((maxRecvDataCount - 1) / maxDataCountPerLoop); // 向上取整
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount
            = processedDataCount < myRankRecvCount ? min(maxDataCountPerLoop, myRankRecvCount - processedDataCount) : 0;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.scratchBuffBaseOff = 0;
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParams.tailSize = 0; // 变长算子不涉及
        tempAlgParams.inputSliceStride
            = myDisplacementCount * dataTypeSize_; // 变长算子, 表示自己的这片输入数据的起始位置
        tempAlgParams.outputSliceStride = 0;       // 变长算子， 表示自己的这片输出数据的起始位置

        CHK_RET(algTemplate->GenExtIns(tempFuncs, tempAlgParams, tempResLinks_, tempInsQue_));
        processedDataCount += maxDataCountPerLoop;
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsV2ReduceScatterVSoleExecutor][CalcRes] [%s] Rank[%d], CalcRes with detouring enabled.", __func__,
            myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsV2ReduceScatterVSoleExecutor][CalcRes] [%s] Rank[%d], CalcRes with detouring disabled.", __func__,
            myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    HCCL_INFO("[%s] Rank[%d], requiredQueNum [%u].", __func__, myRank_, algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsV2ReduceScatterVSoleExecutor][CalcResOffload] [%s] Rank[%d], CalcRes with detouring enabled.",
            __func__, myRank_);
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsV2ReduceScatterVSoleExecutor][CalcResOffload] [%s] Rank[%d], CalcRes with detouring disabled.",
            __func__, myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    resReq.requiredScratchMemSize = UB_MAX_DATA_SIZE;
    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTERV, CcuReduceScatterVMeshMem2Mem1D, InsV2ReduceScatterVSoleExecutor, TopoMatchMesh,
    CcuTempReduceScatterVMeshMem2Mem1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::REDUCESCATTERV, CcuReduceScatterVMesh1D, InsV2ReduceScatterVSoleExecutor, TopoMatchMesh,
    CcuTempReduceScatterVMesh1D);
#endif

} // namespace Hccl
