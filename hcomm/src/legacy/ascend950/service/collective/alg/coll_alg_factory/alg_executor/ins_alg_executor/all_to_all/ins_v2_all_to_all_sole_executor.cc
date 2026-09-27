/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_to_all_sole_executor.h"
#include "log.h"
#include "ins_coll_alg_registry.h"
#include "topo_match_mesh.h"
#include "topo_match_concurr_mesh.h"
#include "ins_temp_all_to_all_mesh_2D.h"
#ifndef CCL_KERNEL_AICPU
#include "aiv_temp_all_to_all_mesh_1D.h"
#include "ccu_temp_all_to_all_mesh_1D_2Die.h"
#endif

namespace Hccl {
constexpr u64 MAX_OFFLOAD_SCRATCH_SIZE = 200 * 1024 * 1024; // 200M

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AlltoAllSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsV2AlltoAllSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitParams(
    const CollAlgOperator& op, const CollAlgParams& params)
{
    op_ = op;
    opMode_ = params.opMode;
    maxTmpMemSize_ = params.maxTmpMemSize;
    CHK_PRT_RET(
        (maxTmpMemSize_ == 0), HCCL_ERROR("[InitParams] maxTmpMemSize equals to zero."), HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        (op.opType != OpType::ALLTOALL), HCCL_ERROR("[InitParams] opType is invalid."), HcclResult::HCCL_E_PARA);

    dataType_ = op.all2AllDataDes.sendType;
    dataCount_ = op.all2AllDataDes.sendCount; // 本卡数据量/rankSize
    outputDataType_ = op.all2AllDataDes.sendType;
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_DEBUG("dataType_ is [%u], dataCount_ is [%u]", dataType_, dataCount_);

    CHK_PRT_RET(
        InitOpInfo(op, opType_, redOp_, root_), HCCL_ERROR("[InitParams] unable to init OpInfo."),
        HcclResult::HCCL_E_PARA);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const RankGraph* rankGraph)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const AlgTopoInfo& topoInfo)
{
    CHK_PRT_RET(
        (topoInfo.vTopo.empty()), HCCL_ERROR("[InsV2AlltoAllSoleExecutor][InitCommInfo], topoInfo.vTopo is empty"),
        HcclResult::HCCL_E_PARA);
    CHK_PRT_RET(
        (topoInfo.virtRankMap.empty()),
        HCCL_ERROR("[InsV2AlltoAllSoleExecutor][InitCommInfo], topoInfo.virtRankMap is empty"),
        HcclResult::HCCL_E_PARA);
    CHK_PRT_RET(
        (topoInfo.virtRanks.empty()),
        HCCL_ERROR("[InsV2AlltoAllSoleExecutor][InitCommInfo], topoInfo.virtRanks is empty"), HcclResult::HCCL_E_PARA);

    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CreateTemplates(
    std::shared_ptr<InsAlgTemplate>& algTemplatePtr)
{
    HCCL_DEBUG("[InsV2AlltoAllSoleExecutor][CreateTemplates]");
    algTemplatePtr = std::make_shared<InsAlgTemplate>(myRank_, rankSize_, vTopo_, virtRankMap_);
    CHK_PTR_NULL(algTemplatePtr); // 检查是否成功分配内存
    algTemplatePtr->SetDmaMode(dmaMode_);
    algTemplatePtr->SetDataType(dataType_);
    algTemplatePtr->SetCollOp(op_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    const RankGraph* rankGraph, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    ConnectedLinkMgr* linkMgr, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        CHK_RET(algTemplate->CalcResDetour(linkMgr, tempResReq));
    } else {
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));
    CHK_RET(algTemplate->CalNumBlocks(numBlocks, dataSize, numBlocksLimit));
    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_DEBUG("[InsV2AlltoAllSoleExecutor][Orchestrate] Orchestrate HOST Start");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));

    HCCL_DEBUG(
        "[InsV2AlltoAllSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));

    CHK_RET(OrchestrateLoop(algTemplate));
    HCCL_DEBUG("[InsV2AlltoAllSoleExecutor][Orchestrate] Orchestrate HOST End");
    return HcclResult::HCCL_SUCCESS;
}

// AICPU 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
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
InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(std::shared_ptr<InsAlgTemplate> algTemplate)
{
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    HCCL_INFO("[InsV2AlltoAllSoleExecutor][Orchestrate] Start, template[%s]", algTemplate->Describe().c_str());
    dataSize_ = dataCount_ * dataSizePerVolume;

    TemplateDataParams tempAlgParams;
    tempAlgParams.repeatNum = 1; // 不需要重复
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.isBottom = true;
    tempFuncs.isForepart = true;

    u64 maxDataSizePerLoop = 0;
    u64 transportBoundDataSize_ = UB_MAX_DATA_SIZE;
    u32 templateScratchMultiplier = algTemplate->CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier;
        maxDataSizePerLoop = std::min(transportBoundDataSize_, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize_;
    }

    // 先将cclBuffer切块，看每一块大小
    u64 scratchDataCountPerLoopPerRank = maxDataSizePerLoop / dataTypeSize_ / rankSize_;
    HCCL_DEBUG(
        "[InsV2AlltoAllSoleExecutor][OrchestrateLoop] maxTmpMemSize_[%llu], templateScratchMultiplier[%llu], "
        "maxDataSizePerLoop[%llu], "
        "transportBoundDataSize_[%llu], scratchDataCountPerLoopPerRank[%llu]",
        maxTmpMemSize_, templateScratchMultiplier, maxDataSizePerLoop, transportBoundDataSize_,
        scratchDataCountPerLoopPerRank);
    CHK_PRT_RET(
        scratchDataCountPerLoopPerRank == 0,
        HCCL_ERROR("[InsV2AlltoAllSoleExecutor][OrchestrateLoop] scratchDataCountPerLoopPerRank is 0"),
        HCCL_E_INTERNAL);

    // 将usrIn数据切块，看每一块大小，这里要保障是能整除的
    u64 dataCountPerRank = dataCount_;
    u64 allToAllProcessedDataCount = 0;
    u64 loopTimes = dataCountPerRank / scratchDataCountPerLoopPerRank
                    + static_cast<u64>(dataCountPerRank % scratchDataCountPerLoopPerRank != 0);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount
            = (loop == loopTimes - 1) ? dataCountPerRank - allToAllProcessedDataCount : scratchDataCountPerLoopPerRank;

        tempAlgParams.buffInfo.inBuffBaseOff = allToAllProcessedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = allToAllProcessedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.scratchBuffBaseOff = 0;
        tempAlgParams.sliceSize = currDataCount * dataTypeSize_; // 这里就是 cclbuf 一块数据的大小
        tempAlgParams.inputSliceStride = dataCountPerRank * dataTypeSize_;
        tempAlgParams.outputSliceStride = dataCountPerRank * dataTypeSize_;

        CHK_RET(algTemplate->GenExtIns(tempFuncs, tempAlgParams, tempResLinks_, tempInsQue_));
        allToAllProcessedDataCount += currDataCount;
    }

    HCCL_INFO("[InsV2AlltoAllSoleExecutor][Orchestrate] End, template[%s]", algTemplate->Describe().c_str());

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
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
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));
    resReq.requiredScratchMemSize = MAX_OFFLOAD_SCRATCH_SIZE; // 最大 200M
    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALL, InsAlltoAllMesh2D, InsV2AlltoAllSoleExecutor, TopoMatchConcurrMesh, InsTempAlltoAllMesh2D);
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALL, AivAlltoAllMesh1D, InsV2AlltoAllSoleExecutor, TopoMatchMesh, AivTempAlltoAllMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALL, CcuAlltoAllMesh1D2Die, InsV2AlltoAllSoleExecutor, TopoMatchMesh, CcuTempAllToAllMesh1D2Die);
#endif
} // namespace Hccl
