/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_reduce_parallel_executor.h"
#include <cmath>
#include "log.h"
#include "ins_coll_alg_registry.h"
#include "topo_match_mesh_nhr.h"
#include "topo_match_mesh_nhr_pcie.h"
#include "alg_data_trans_wrapper.h"
#include "ins_temp_reduce_nhr.h"
#include "ins_temp_reduce_mesh_1D.h"
#include "ccu_temp_reduce_nhr_1D_mem2mem.h"
#include "ccu_temp_reduce_mesh_1D_mem2mem.h"

namespace Hccl {
constexpr u64 MAX_OFFLOAD_SCRATCH_SIZE = 200 * 1024 * 1024; // 200M

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsReduceParallelExecutor()
    : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsReduceParallelExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    HCCL_INFO("[InsReduceParallelExecutor] CalcResOffload begins.");
    (void)dataSize;
    u64 scratchMemSize = MAX_OFFLOAD_SCRATCH_SIZE;
    resReq.requiredScratchMemSize = scratchMemSize; // 200MB
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, intraLocalRankSize_, interLocalRankSize_));
    InsAlgTemplate0 intraTempAlg(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlg(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]);

    // calculate required insQues and prepare queue
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsReduceParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[InsReduceParallelExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }

    CHK_RET(interTempAlg.CalcRes(resReqInter));
    resReq.requiredSubQueNum = resReqIntra.streamNum + resReqInter.streamNum - 1;
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    HCCL_INFO("[InsReduceParallelExecutor] CalcRes begins.");
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateMultiLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, intraLocalRankSize_, interLocalRankSize_));

    // instantiate a template
    InsAlgTemplate0 intraTempAlg(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlg(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]);

    // calculate required insQues and prepare queue
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsReduceParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[InsReduceParallelExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }
    CHK_RET(interTempAlg.CalcRes(resReqInter));

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqIntra.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqInter.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = resReqIntra.streamNum + resReqInter.streamNum;
    std::vector<std::tuple<QId, QId, u32>> notifyRequests;

    u32 slaveNum = algResReq.primQueueNum - 1;
    notifyRequests.reserve(slaveNum); // 每个从流需要1个
    for (QId q = 1; q < algResReq.primQueueNum; q++) {
        notifyRequests.emplace_back(std::make_tuple(0, q, 0));
        notifyRequests.emplace_back(std::make_tuple(q, 0, 0));
    }

    // nhr算法只有一个stream
    for (QId q = resReqIntra.streamNum; q < algResReq.primQueueNum; q++) {
        if (resReqIntra.streamNum == q) {
            continue;
        }
        notifyRequests.emplace_back(std::make_tuple(resReqIntra.streamNum, q, 0));
        notifyRequests.emplace_back(std::make_tuple(q, resReqIntra.streamNum, 0));
        HCCL_DEBUG(
            "[InsReduceParallelExecutor] CalcRes notifyRequests:%u->%u. %u->%u", resReqIntra.streamNum, q, q,
            resReqIntra.streamNum);
    }

    algResReq.queueNotifys = notifyRequests;
    HCCL_DEBUG("[InsReduceParallelExecutor] algResReq.primQueueNum %u", algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口，将对应的 instruction 添加到指令队列中
// 传入的insQue为一条主流
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParams0(
    const u64 dataOffset, const u64 dataCount, const u64 scratchOffset, TemplateDataParams& tempAlgParams) const
{
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParams.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParams.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParams.buffInfo.scratchBuffBaseOff = scratchOffset;
    tempAlgParams.sliceSize = dataCount * dataTypeSize_;
    tempAlgParams.tailSize = tempAlgParams.sliceSize;
    tempAlgParams.inputSliceStride = 0; // 输入数据仅有 1 个 slice, 不需要 stride
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.repeatNum = 1;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParams1(
    const u64 dataOffset, const u64 dataCount, const u64 scratchOffset, TemplateDataParams& tempAlgParams) const
{
    tempAlgParams.buffInfo.inBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParams.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParams.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParams.buffInfo.scratchBuffBaseOff = scratchOffset;
    tempAlgParams.sliceSize = dataCount * dataTypeSize_;
    tempAlgParams.tailSize = tempAlgParams.sliceSize;
    tempAlgParams.inputSliceStride = 0; // 输入数据仅有 1 个 slice, 不需要 stride
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.repeatNum = 1;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplitRate(
    std::vector<float>& splitDataSize) const
{
    // 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(static_cast<float>(splitData));
    splitDataSize.push_back(static_cast<float>(splitData));
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcLocalRoot()
{
    CHK_PRT_RET(
        root_ >= rankSize_, HCCL_ERROR("[CalcLocalRoot] root[%u] is out of rankSize[%u]", root_, rankSize_),
        HcclResult::HCCL_E_INTERNAL);

    u32 intraLocalRootIdx = root_ % intraLocalRankSize_;
    intraLocalRoot_ = static_cast<u32>(vTopo_.at(0).at(0).at(intraLocalRootIdx));
    u32 interLocalRootIdx = root_ / intraLocalRankSize_;
    interLocalRoot_ = static_cast<u32>(vTopo_.at(1).at(0).at(interLocalRootIdx));

    HCCL_INFO(
        "[CalcLocalRoot] localRoot: myRank[%d] intraLocalRoot[%u] interLocalRoot[%u]", myRank_, intraLocalRoot_,
        interLocalRoot_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const RankGraph* rankGraph, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], detouring enabled.", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[%s] Rank[%d], detouring disabled.", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));

    // 申请算法模板所需资源
    if (!(resReqIntra.queNum > 0 && resReqInter.queNum > 0)) {
        HCCL_ERROR("[InsReduceParallelExecutor]resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    u32 totalQueueNum = resReqIntra.queNum + resReqInter.queNum;
    CHK_RET(InitQueue(totalQueueNum, reqQue_));
    for (u32 i = 0; i < reqQue_.size(); i++) {
        if (i < resReqIntra.queNum) {
            intraQue_.push_back(reqQue_[i]);
        } else {
            interQue_.push_back(reqQue_[i]);
        }
    }
    syncQueues_.emplace_back(intraQue_[0]);
    syncQueues_.emplace_back(interQue_[0]);

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, interLinks_));
    HCCL_INFO(
        "[InsReduceParallelExecutor] intraLinks_ size[%zu], interLinks_ size[%zu]", intraLinks_.size(),
        interLinks_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    ConnectedLinkMgr* linkMgr, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detour enabled", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(linkMgr, resReqIntra));
    } else {
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));

    // 申请算法模板所需资源
    if (!(resReqIntra.queNum > 0 && resReqInter.queNum > 0)) {
        HCCL_ERROR("[InsReduceParallelExecutor]resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    u32 totalQueueNum = resReqIntra.queNum + resReqInter.queNum;
    CHK_RET(InitQueue(totalQueueNum, reqQue_));
    for (u32 i = 0; i < reqQue_.size(); i++) {
        if (i < resReqIntra.queNum) {
            intraQue_.push_back(reqQue_[i]);
        } else {
            interQue_.push_back(reqQue_[i]);
        }
    }
    syncQueues_.emplace_back(intraQue_[0]);
    syncQueues_.emplace_back(interQue_[0]);

    CHK_RET(PrepResLinks(myRank_, resReqIntra.links, linkMgr, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, resReqInter.links, linkMgr, interLinks_));
    HCCL_INFO(
        "[InsReduceParallelExecutor] intraLinks_ size[%zu], interLinks_ size[%zu]", intraLinks_.size(),
        interLinks_.size());
    return HCCL_SUCCESS;
}

// Aicpu展开
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsReduceParallelExecutor] AICPU Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));
    // 所以获取取级通信域的信息
    vTopo_ = topoInfo.vTopo;             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks;     // 本通信域内的 rank 集合

    // 计算localRankSize和localRoot
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, intraLocalRankSize_, interLocalRankSize_));
    CHK_RET(CalcLocalRoot());

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]); // server内算法，比如mesh
    InsAlgTemplate1 tempAlgInter(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]); // server间算法，比如nhr

    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.InitReduceInfo(redOp_, dataType_);
    tempAlgIntra.SetRoot(intraLocalRoot_);
    tempAlgIntra.SetCollOp(op);

    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.InitReduceInfo(redOp_, dataType_);
    tempAlgInter.SetRoot(interLocalRoot_);
    tempAlgInter.SetCollOp(op);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(linkMgr, tempAlgIntra, tempAlgInter));
    CHK_RET(GenInsQues(tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsReduceParallelExecutor] AICPU Orchestrate success.");
    return HcclResult::HCCL_SUCCESS;
}

// Host展开
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsReduceParallelExecutor] Host Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // 计算localRankSize和localRoot
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, intraLocalRankSize_, interLocalRankSize_));
    CHK_RET(CalcLocalRoot());

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]); // server内算法，比如mesh
    InsAlgTemplate1 tempAlgInter(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]); // server间算法，比如nhr

    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.SetCollOp(op);
    tempAlgIntra.InitReduceInfo(redOp_, dataType_);
    tempAlgIntra.SetRoot(intraLocalRoot_);

    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.InitReduceInfo(redOp_, dataType_);
    tempAlgInter.SetCollOp(op);
    tempAlgInter.SetRoot(interLocalRoot_);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(rankGraph, tempAlgIntra, tempAlgInter));
    CHK_RET(GenInsQues(tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsReduceParallelExecutor] Host Orchestrate success.");
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenInsQues(
    InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    std::vector<float> dataSplitSize;
    GetParallelDataSplitRate(dataSplitSize);
    u64 alignedSize = 16 * 1024; // 假设需要16K对齐
    BufferType inBuffType = BufferType::INPUT;
    BufferType outBuffType = BufferType::OUTPUT;
    u32 intraScatchteMultipleStage0 = tempAlgIntra.CalcScratchMultiple(inBuffType, outBuffType);
    u32 interScatchteMultipleStage0 = tempAlgInter.CalcScratchMultiple(inBuffType, outBuffType);
    u32 intraScatchteMultipleStage1 = tempAlgIntra.CalcScratchMultiple(outBuffType, outBuffType);
    u32 interScatchteMultipleStage1 = tempAlgInter.CalcScratchMultiple(outBuffType, outBuffType);
    u32 scratchMultipleIntra = static_cast<u32>(std::max(
        std::ceil(dataSplitSize[0] * intraScatchteMultipleStage0),
        std::ceil(dataSplitSize[1] * intraScatchteMultipleStage1)));
    u32 scratchMultipleInter = static_cast<u32>(std::max(
        std::ceil(dataSplitSize[1] * interScatchteMultipleStage0),
        std::ceil(dataSplitSize[0] * interScatchteMultipleStage1)));
    u32 totalScratchMultiple = scratchMultipleIntra + scratchMultipleInter;
    u64 scratchMemBlockSize = maxTmpMemSize_;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / alignedSize / totalScratchMultiple) * alignedSize;
    }
    u64 intraScratchOffset = 0;
    u64 interScratchOffset = scratchMultipleIntra * scratchMemBlockSize;

    // dataSplitSize为分数，这里maxCountPerLoop对10取整
    u64 maxCountPerLoop
        = (std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_ / 10)
          * 10;

    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TemplateDataParams tempAlgParamsIntra0, tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1, tempAlgParamsIntra1;
    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = false;
    tempFuncs.isBottom = true;
    tempFuncs.isForepart = true;
    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex == loopTimes - 1) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataCountPerLoopAixs0 = static_cast<u64>(dataSplitSize[0] * currCount);
        u64 dataCountPerLoopAixs1 = currCount - dataCountPerLoopAixs0;
        // 第一步开始前同步

        CHK_RET(PreSyncQues(syncQueues_, 0));
        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + dataCountPerLoopAixs0 * dataTypeSize_;
        // 数据0的server内的mesh算法
        GenTemplateAlgParams0(dataOffset0, dataCountPerLoopAixs0, intraScratchOffset, tempAlgParamsIntra0);
        // 把每个template需要的queue传进去，比如stars的mesh要传多条queue
        CHK_RET(tempAlgIntra.GenExtIns(tempFuncs, tempAlgParamsIntra0, intraLinks_, intraQue_));
        // 数据1的server间的nhr算法
        GenTemplateAlgParams0(dataOffset1, dataCountPerLoopAixs1, interScratchOffset, tempAlgParamsInter1);
        CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter1, interLinks_, interQue_));
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncQues(syncQueues_, 0));
        // 只有真正root节点的横纵坐标所在的卡，需要做第二步骤，担任过其中一个root节点的，只需要负责发就行了
        if ((static_cast<u32>(myRank_) != intraLocalRoot_) && (static_cast<u32>(myRank_) != interLocalRoot_)) {
            continue;
        }

        // 第二步开始前同步
        CHK_RET(PreSyncQues(syncQueues_, 0));
        if (static_cast<u32>(myRank_) == intraLocalRoot_) {
            // 数据0的server间的nhr算法
            GenTemplateAlgParams1(dataOffset0, dataCountPerLoopAixs0, interScratchOffset, tempAlgParamsInter0);
            CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter0, interLinks_, interQue_));
        }
        if (static_cast<u32>(myRank_) == interLocalRoot_) {
            // 数据1的server内的mesh算法
            GenTemplateAlgParams1(dataOffset1, dataCountPerLoopAixs1, intraScratchOffset, tempAlgParamsIntra1);
            CHK_RET(tempAlgIntra.GenExtIns(tempFuncs, tempAlgParamsIntra1, intraLinks_, intraQue_));
        }
        // 尾同步
        CHK_RET(PostSyncQues(syncQueues_, 0));
    }
    return HcclResult::HCCL_SUCCESS;
}

// 算法注册
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::REDUCE, InsReduceParallelMesh1DNHR, InsReduceParallelExecutor, TopoMatchMeshNHR, InsTempReduceMesh1D,
    InsTempReduceNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::REDUCE, InsReduceParallelMesh1DNHRPcie, InsReduceParallelExecutor, TopoMatchMeshNHRPcie,
    InsTempReduceMesh1D, InsTempReduceNHR);
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::REDUCE, CcuReduceParallelMesh1DNHR, InsReduceParallelExecutor, TopoMatchMeshNHR, CcuTempReduceMeshMem2Mem1D,
    CcuTempReduceNHRMem2Mem1D);
#endif
} // namespace Hccl
