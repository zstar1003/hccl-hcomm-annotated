/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_v_sole_executor.h"

#include "log.h"

#include "ins_coll_alg_registry.h"

#include "topo_match_mesh.h"

#ifndef CCL_KERNEL_AICPU
#include "ccu_temp_all_gather_v_mesh_1D.h"
#endif

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AllGatherVSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsV2AllGatherVSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const RankGraph* rankGraph)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const AlgTopoInfo& topoInfo)
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
HcclResult InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CreateTemplates(
    std::shared_ptr<InsAlgTemplate>& algTemplatePtr)
{
    algTemplatePtr = std::make_shared<InsAlgTemplate>(myRank_, rankSize_, vTopo_, virtRankMap_);
    CHK_PTR_NULL(algTemplatePtr); // 检查是否成功分配内存
    algTemplatePtr->SetCollOp(op_);
    algTemplatePtr->SetDmaMode(dmaMode_);
    algTemplatePtr->SetDataType(dataType_);
    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_DEBUG("[InsV2AllGatherVSoleExecutor][Orchestrate] Orchestrate host Start");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsV2AllGatherVSoleExecutor] [%s] Rank[%d], CalcRes with detouring enabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsV2AllGatherVSoleExecutor] [%s] Rank[%d], CalcRes with detouring disabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }

    HCCL_DEBUG(
        "[InsV2AllGatherVSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

// AICPU 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_DEBUG("[InsV2AllGatherVSoleExecutor][Orchestrate] Orchestrate AICPU Start");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(topoInfo));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        CHK_RET(algTemplate->CalcResDetour(linkMgr, tempResReq));
    } else {
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }

    HCCL_DEBUG(
        "[InsV2AllGatherVSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

// 切分数据并调用 template
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(std::shared_ptr<InsAlgTemplate> algTemplate)
{
    HCCL_INFO("[InsV2AllGatherVSoleExecutor][OrchestrateOpbase] Start, template[%s]", algTemplate->Describe().c_str());
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;

    TemplateDataParams tempAlgParams;
    tempAlgParams.repeatNum = 1; // 不需要重复
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;

    TempFuncs tempFuncs;
    tempFuncs.isForepart = true;
    tempFuncs.isBottom = true;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();

    u64 maxDataSizePerLoop = 0;
    u64 transportBoundDataSize;
    CHK_RET(algTemplate->GetMaxTransPortDataSize(transportBoundDataSize));
    u64 templateScratchMultiplier
        = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, tempAlgParams.buffInfo.outBuffType);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    HCCL_INFO(
        "[InsV2AllGatherVSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(
        maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2AllGatherVSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);

    CHK_PRT_RET(
        op_.vDataDes.counts == nullptr || op_.vDataDes.displs == nullptr,
        HCCL_ERROR("[InsAllReduceCombExecutor][OrchestrateOpbase] counts or displs is nullptr"), HCCL_E_PTR);
    u64 myRankSendCount = static_cast<u64*>(op_.vDataDes.counts)[myRank_];
    u64 myDisplacement = static_cast<u64*>(op_.vDataDes.displs)[myRank_] * dataTypeSize_;
    u64 maxSendDataCount = 0;
    for (u64 i = 0; i < rankSize_; i++) {
        maxSendDataCount = max(maxSendDataCount, static_cast<u64*>(op_.vDataDes.counts)[i]);
    }

    u64 processedDataCount = 0;
    u64 loopTimes = 1 + ((maxSendDataCount - 1) / maxDataCountPerLoop); // 向上取整

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount
            = processedDataCount < myRankSendCount ? min(maxDataCountPerLoop, myRankSendCount - processedDataCount) : 0;

        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.scratchBuffBaseOff = 0;
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParams.tailSize = 0;         // 变长算子不涉及
        tempAlgParams.inputSliceStride = 0; // 变长算子, 表示自己的这片输入数据的起始位置
        tempAlgParams.outputSliceStride = myDisplacement; // 变长算子， 表示自己的这片输出数据的起始位置

        CHK_RET(algTemplate->GenExtIns(tempFuncs, tempAlgParams, tempResLinks_, tempInsQue_));
        processedDataCount += maxDataCountPerLoop;
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detouring enabled.", __func__, myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detouring disabled.", __func__, myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    algResReq.primQueueNum = tempResReq.streamNum;
    HCCL_DEBUG("[%s] Rank[%d], requiredQueNum [%u].", __func__, myRank_, algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    resReq.requiredScratchMemSize = 0;
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsV2AllGatherVSoleExecutor] CalcRes with detouring enabled.");
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsV2AllGatherVSoleExecutor] CalcRes with detouring disabled.");
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLGATHERV, CcuAllGatherVMesh1D, InsV2AllGatherVSoleExecutor, TopoMatchMesh, CcuTempAllGatherVMesh1D);
#endif
} // namespace Hccl
