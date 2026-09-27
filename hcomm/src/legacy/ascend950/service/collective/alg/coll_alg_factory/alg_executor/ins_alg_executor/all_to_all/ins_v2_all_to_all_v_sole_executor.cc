/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_to_all_v_sole_executor.h"
#include "log.h"
#include "ins_coll_alg_registry.h"
#include "topo_match_mesh.h"
#ifndef CCL_KERNEL_AICPU
#include "aiv_temp_all_to_all_v_mesh_1D.h"
#endif

namespace Hccl {
constexpr u64 MAX_OFFLOAD_SCRATCH_SIZE = 200 * 1024 * 1024; // 200M

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AlltoAllVSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsV2AlltoAllVSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitParams(
    const CollAlgOperator& op, const CollAlgParams& params)
{
    op_ = op;
    opMode_ = params.opMode;
    maxTmpMemSize_ = params.maxTmpMemSize;
    CHK_PRT_RET(
        (maxTmpMemSize_ == 0), HCCL_ERROR("[InitParams] maxTmpMemSize equals to zero."), HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        (op.opType != OpType::ALLTOALLV), HCCL_ERROR("[InitParams] opType is invalid."), HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        op.all2AllVDataDes.sendCounts == nullptr || op.all2AllVDataDes.sdispls == nullptr,
        HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][InitParams] sendCounts or sdispls is nullptr"), HCCL_E_PTR);

    dataType_ = op.all2AllVDataDes.sendType;
    dataCount_ = static_cast<u64*>(op.all2AllVDataDes.sendCounts)[myRank_]; // send留给本卡的那一片数据
    outputDataType_ = op.all2AllVDataDes.recvType;
    CHK_PRT_RET(
        dataType_ != outputDataType_,
        HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][InitParams] dataType_ != outputDataType_"), HCCL_E_PTR);

    dataTypeSize_ = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_DEBUG("dataType_ is [%u], dataCount_ is [%u]", dataType_, dataCount_);

    CHK_PRT_RET(
        InitOpInfo(op, opType_, redOp_, root_), HCCL_ERROR("[InitParams] unable to init OpInfo."),
        HcclResult::HCCL_E_PARA);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const RankGraph* rankGraph)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const AlgTopoInfo& topoInfo)
{
    CHK_PRT_RET(
        (topoInfo.vTopo.empty()), HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][InitCommInfo], topoInfo.vTopo is empty"),
        HcclResult::HCCL_E_PARA);
    CHK_PRT_RET(
        (topoInfo.virtRankMap.empty()),
        HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][InitCommInfo], topoInfo.virtRankMap is empty"),
        HcclResult::HCCL_E_PARA);
    CHK_PRT_RET(
        (topoInfo.virtRanks.empty()),
        HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][InitCommInfo], topoInfo.virtRanks is empty"), HcclResult::HCCL_E_PARA);

    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CreateTemplates(
    std::shared_ptr<InsAlgTemplate>& algTemplatePtr)
{
    HCCL_DEBUG("[InsV2AlltoAllVSoleExecutor][CreateTemplates]");
    algTemplatePtr = std::make_shared<InsAlgTemplate>(myRank_, rankSize_, vTopo_, virtRankMap_);
    CHK_PTR_NULL(algTemplatePtr); // 检查是否成功分配内存
    algTemplatePtr->SetDmaMode(dmaMode_);
    algTemplatePtr->SetDataType(dataType_);
    algTemplatePtr->SetCollOp(op_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    const RankGraph* rankGraph, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        HCCL_DEBUG("[InsV2AlltoAllVSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsV2AlltoAllVSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    ConnectedLinkMgr* linkMgr, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        HCCL_INFO("[InsV2AlltoAllVSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(algTemplate->CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_INFO("[InsV2AlltoAllVSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][Orchestrate] Orchestrate HOST Start");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));

    HCCL_DEBUG(
        "[InsV2AlltoAllVSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));

    CHK_RET(OrchestrateLoop(algTemplate));
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][Orchestrate] Orchestrate HOST End");
    return HcclResult::HCCL_SUCCESS;
}

// AICPU 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][Orchestrate] Orchestrate AICPU Start");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(topoInfo));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(linkMgr, algTemplate, tempResReq));

    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));

    CHK_RET(OrchestrateLoop(algTemplate));
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][Orchestrate] Orchestrate AICPU End");
    return HcclResult::HCCL_SUCCESS;
}

// 切分数据并调用 template
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(std::shared_ptr<InsAlgTemplate> algTemplate)
{
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][Orchestrate] Start, template[%s]", algTemplate->Describe().c_str());
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;

    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.repeatNum = 1; // 不需要重复
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;

    TempFuncs tempFuncs;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.isForepart = true;
    tempFuncs.isBottom = true;
    tempFuncs.opMode = opMode_;

    u64 maxDataSizePerLoop = 0;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u32 templateScratchMultiplier = algTemplate->CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }

    // 先将cclBuffer切块，看每一块大小
    u64 scratchDataCountPerLoopPerRank = maxDataSizePerLoop / dataTypeSize_ / rankSize_;
    HCCL_INFO(
        "[InsV2AlltoAllVSoleExecutor][OrchestrateLoop] scratchDataCountPerLoopPerRank[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        scratchDataCountPerLoopPerRank, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(
        scratchDataCountPerLoopPerRank == 0,
        HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][OrchestrateLoop] scratchDataCountPerLoopPerRank is 0"),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(
        op_.all2AllVDataDes.sendCounts == nullptr || op_.all2AllVDataDes.sdispls == nullptr
            || op_.all2AllVDataDes.recvCounts == nullptr || op_.all2AllVDataDes.rdispls == nullptr,
        HCCL_ERROR(
            "[InsV2AlltoAllVSoleExecutor][OrchestrateLoop] sendCounts or sdispls or recvCounts or rdispls is nullptr"),
        HCCL_E_PTR);

#ifndef CCL_KERNEL_AICPU
    // aiv不在这里切分数据，先全量下kernel，到kernel去做循环
    tempAlgParams.buffInfo.inBuffBaseOff = 0;
    tempAlgParams.buffInfo.outBuffBaseOff = 0;
    tempAlgParams.buffInfo.scratchBuffBaseOff = 0;
    tempAlgParams.sliceSize
        = scratchDataCountPerLoopPerRank * dataTypeSize_; // 这里就是每个rank可以用的ccl buffer的大小;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;

    CHK_RET(algTemplate->GenExtIns(tempFuncs, tempAlgParams, tempResLinks_, tempInsQue_));
#endif

    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][Orchestrate] End, template[%s]", algTemplate->Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
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
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][CalcResOffload] dataSize is [%u]", dataSize);
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

#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALLV, AivAlltoAllVMesh1D, InsV2AlltoAllVSoleExecutor, TopoMatchMesh, AivTempAlltoAllVMesh1D);
#endif
} // namespace Hccl
