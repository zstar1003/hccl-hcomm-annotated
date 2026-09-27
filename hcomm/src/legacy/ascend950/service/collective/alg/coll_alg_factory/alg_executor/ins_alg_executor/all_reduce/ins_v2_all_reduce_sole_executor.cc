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
#include "topo_match_mesh.h"
#include "topo_match_nhr.h"

#include "ins_temp_all_reduce_mesh_1D_two_shot.h"
#include "ins_temp_all_reduce_mesh_2D_two_shot.h"
#include "ins_temp_all_reduce_mesh_1D_one_shot.h"
#include "ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.h"
#include "ins_temp_all_reduce_nhr.h"
#include "ins_v2_all_reduce_sole_executor.h"
#ifndef CCL_KERNEL_AICPU
#include "aiv_temp_all_reduce_mesh_1D_oneshot.h"
#include "aiv_temp_all_reduce_mesh_1D_twoshot.h"
#include "ccu_temp_all_reduce_nhr_1D_mem2mem.h"
#include "ccu_temp_all_reduce_mesh_1D_mem2mem.h"
#endif
#include "ins_temp_all_reduce_aicpu_reduce.h"
#include "ins_temp_all_reduce_aicpu_reduce_mesh_2D.h"

namespace Hccl {
constexpr u64 MAX_OFFLOAD_SCRATCH_SIZE = 200 * 1024 * 1024; // 200M

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AllReduceSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsV2AllReduceSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const RankGraph* rankGraph)
{
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(const AlgTopoInfo& topoInfo)
{
    CHK_PRT_RET(
        topoInfo.vTopo.size() == 0 || topoInfo.virtRankMap.size() == 0 || topoInfo.virtRanks.size() == 0,
        HCCL_ERROR("[InsV2AllReduceSoleExecutor][InitCommInfo] topoInfo vector member size is 0 !"), HCCL_E_INTERNAL);
    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CreateTemplates(
    std::shared_ptr<InsAlgTemplate>& algTemplatePtr)
{
    algTemplatePtr = std::make_shared<InsAlgTemplate>(myRank_, rankSize_, vTopo_, virtRankMap_);
    CHK_PTR_NULL(algTemplatePtr); // 检查是否成功分配内存
    algTemplatePtr->SetDmaMode(dmaMode_);
    algTemplatePtr->InitReduceInfo(redOp_, dataType_);
    algTemplatePtr->SetCollOp(op_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    const RankGraph* rankGraph, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d]. CalcRes with detouring enabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[%s] Rank[%d]. CalcRes with detouring disabled.", __func__, myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetTemplateResRequest(
    ConnectedLinkMgr* linkMgr, std::shared_ptr<InsAlgTemplate>& algTemplate, AlgTempResReq& tempResReq) const
{
    if (enableDetour_) {
        HCCL_DEBUG("[InsV2AllReduceSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(algTemplate->CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_DEBUG("[InsV2AllReduceSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(algTemplate->CalcRes(tempResReq));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    // 通过判断哪层通信域能有到所有remoteRank的path，判断当前算法跑在哪一层
    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsV2AllReduceSoleExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    algTemplate->setPathNumMap(rank2PathNumMap);
    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    algResReq.queueNotifys = tempResReq.queNotifys;
    HCCL_DEBUG("[%s] Rank[%d], requiredQueNum [%u].", __func__, myRank_, algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;

    // Topo Match
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    // 通过判断哪层通信域能有到所有remoteRank的path，判断当前算法跑在哪一层
    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsV2AllReduceSoleExecutor] CalcResOffload SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    algTemplate->setPathNumMap(rank2PathNumMap);
    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    resReq.requiredScratchMemSize = transportBoundDataSize;
    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));
    algTemplate->CalNumBlocks(numBlocks, dataSize, numBlocksLimit);
    return HcclResult::HCCL_SUCCESS;
}
// HOST 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsV2AllReduceSoleExecutor][Orchestrate] Orchestrate HOST Start.");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(rankGraph));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(rankGraph, algTemplate, tempResReq));

    HCCL_DEBUG(
        "[InsV2AllReduceSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

// AICPU 侧算法入口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsV2AllReduceSoleExecutor][Orchestrate] Orchestrate AICPU Start.");
    CHK_RET(Init(op, params, insQue));
    CHK_RET(InitCommInfo(topoInfo));

    std::shared_ptr<InsAlgTemplate> algTemplate = nullptr;
    CHK_RET(CreateTemplates(algTemplate));

    std::map<u32, u32> rank2PathNumMap;
    CHK_RET(SetPathNumMapByLinkMgrMultiLevel(linkMgr, virtRanks_, myRank_, rank2PathNumMap));
    algTemplate->setPathNumMap(rank2PathNumMap);

    AlgTempResReq tempResReq;
    CHK_RET(GetTemplateResRequest(linkMgr, algTemplate, tempResReq));

    HCCL_DEBUG(
        "[InsV2AllReduceSoleExecutor][Orchestrate] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        algTemplate->Describe().c_str(), tempResReq.queNum);
    CHK_RET(InitQueue(tempResReq.queNum, tempInsQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));
    CHK_RET(OrchestrateLoop(algTemplate));
    return HcclResult::HCCL_SUCCESS;
}

// 切分数据并调用 template
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(std::shared_ptr<InsAlgTemplate> algTemplate)
{
    HCCL_INFO("[InsV2AllReduceSoleExecutor][OrchestrateOpbase] Start, template[%s]", algTemplate->Describe().c_str());
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;

    TemplateDataParams tempAlgParams;
    tempAlgParams.repeatNum = 1; // 不需要重复
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    // template 中2D未适配
    tempFuncs.enableCounterNotify = false;
    tempFuncs.isForepart = true;
    tempFuncs.isBottom = true;

    u64 maxDataSizePerLoop = 0;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE; // algTemplate->CalcLoopMaxCount();
    u32 templateScratchMultiplier = algTemplate->CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    if (templateScratchMultiplier != 0) {
        // maxTmpMemSize_大小的buffer当前template仍然可以用，只是传入的待处理数据只有scratchBoundDataSize大小
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }

    // 如果有template融合可以多种取整策略融合: maxDataSizePerLoop / (MN2*dataSizePerVolume)*MN2;
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataSizePerVolume;
    HCCL_INFO(
        "[InsV2AllReduceSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(
        maxDataCountPerLoop == 0,
        HCCL_ERROR(
            "[InsV2AllReduceSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0, scratch buffer "
            "size:%u,maxDataSizePerLoop:%llu, dataSize:%llu",
            maxTmpMemSize_, maxDataSizePerLoop, dataSize_),
        HCCL_E_INTERNAL);

    u64 processedDataCount = 0;
    u64 loopTimes = dataCount_ / maxDataCountPerLoop + static_cast<u32>(dataCount_ % maxDataCountPerLoop != 0);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;

        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataSizePerVolume;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataSizePerVolume;
        tempAlgParams.buffInfo.scratchBuffBaseOff = 0;
        tempAlgParams.buffInfo.scratchBuffSize = maxTmpMemSize_;
        tempAlgParams.sliceSize = currDataCount * dataSizePerVolume;
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        tempAlgParams.inputSliceStride = 0;  // 输入数据仅有 1 个 slice, 不需要 stride
        tempAlgParams.outputSliceStride = 0; // 每张卡的数据间隔为算子输入大小

        CHK_RET(algTemplate->GenExtIns(tempFuncs, tempAlgParams, tempResLinks_, tempInsQue_));
        processedDataCount += currDataCount;
    }

    return HcclResult::HCCL_SUCCESS;
}

INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, InsAllReduceMesh1DOneShot, InsV2AllReduceSoleExecutor, TopoMatchMesh,
    InsTempAllReduceMesh1DOneShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, InsAllReduceMesh2DTwoShot, InsV2AllReduceSoleExecutor, TopoMatchConcurrMesh,
    InsTempAllReduceMesh2DTwoShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, InsAllReduceMesh1DTwoShot, InsV2AllReduceSoleExecutor, TopoMatchMesh,
    InsTempAllReduceMesh1DTwoShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, InsAllReduceMesh1DTwoShotMeshChunk, InsV2AllReduceSoleExecutor, TopoMatchMesh,
    InsTempAllReduceMesh1DTwoShotMeshChunk);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, InsAllReduceNHR, InsV2AllReduceSoleExecutor, TopoMatchNHR, InsTempAllReduceNHR);
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, AivAllReduceMesh1DOneShot, InsV2AllReduceSoleExecutor, TopoMatchMesh,
    AivTempAllReduceMesh1DOneShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, AivAllReduceMesh1DTwoShot, InsV2AllReduceSoleExecutor, TopoMatchMesh,
    AivTempAllReduceMesh1DTwoShot);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, CcuAllReduceNHR1D, InsV2AllReduceSoleExecutor, TopoMatchMesh, CcuTempAllReduceNHRMem2Mem1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, CcuAllReduceMeshMem2Mem1D, InsV2AllReduceSoleExecutor, TopoMatchMesh,
    CcuTempAllReduceMeshMem2Mem1D);
#endif
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, InsAllReduceAicpuReduce, InsV2AllReduceSoleExecutor, TopoMatchMesh, InsTempAllReduceAicpuReduce);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLREDUCE, InsAllReduceAicpuReduceMesh2D, InsV2AllReduceSoleExecutor, TopoMatchConcurrMesh,
    InsTempAllReduceAicpuReduceMesh2D);

} // namespace Hccl
