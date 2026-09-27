/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_all_reduce_parallel_executor.h"
#include <cmath>
#include "log.h"
#include "ins_coll_alg_registry.h"
#include "topo_match_mesh_nhr.h"
#include "topo_match_concurr_mesh_nhr.h"
#include "topo_match_mesh_nhr_pcie.h"
#include "alg_data_trans_wrapper.h"
#include "ins_temp_all_reduce_nhr.h"
#include "ins_temp_all_reduce_mesh_1D_two_shot.h"
#include "ins_temp_all_reduce_mesh_2D_two_shot.h"
#include "ccu_temp_all_reduce_nhr_1D_mem2mem.h"
#include "ccu_temp_all_reduce_mesh_1D_mem2mem.h"

namespace Hccl {
constexpr u64 MAX_OFFLOAD_SCRATCH_SIZE = 200 * 1024 * 1024; // 200M

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsAllReduceParallelExecutor()
    : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsAllReduceParallelExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    HCCL_INFO("[InsAllReduceParallelExecutor] CalcResOffload begins.");
    (void)dataSize;
    uint64_t tempSize = 2;
    u64 scratchMemSize = MAX_OFFLOAD_SCRATCH_SIZE;
    resReq.requiredScratchMemSize = scratchMemSize; // 200MB
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));
    InsAlgTemplate0 intraTempAlg(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlg(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);

    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsV2AllGatherSoleExecutor] CalcResOffload SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    intraTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interTempAlg.setPathNumMap(rank2PathNumMap[1]);

    // calculate required insQues and prepare queue
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsAllReduceParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[InsAllReduceParallelExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }

    CHK_RET(interTempAlg.CalcRes(resReqInter));

    // 算法从流数量 = Σ(temp的que数量 + temp的从流数量 * temp调用次数) - 算法主流数量
    resReq.requiredSubQueNum = resReqIntra.queNum + (resReqIntra.streamNum - resReqIntra.queNum) * tempSize
                               + resReqInter.queNum + (resReqInter.streamNum - resReqInter.queNum) * tempSize - 1;
    HCCL_INFO("[InsAllReduceParallelExecutor::CalcResOffload]requiredSubQueNum = %llu", resReq.requiredSubQueNum);

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    HCCL_INFO("[InsAllReduceParallelExecutor] CalcRes begins.");
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateMultiLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));

    // instantiate a template
    InsAlgTemplate0 intraTempAlg(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlg(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);

    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsAllReduceParallelExecutor] CalcResOffload SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    intraTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interTempAlg.setPathNumMap(rank2PathNumMap[1]);

    // calculate required insQues and prepare queue
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsAllReduceParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[InsAllReduceParallelExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }
    CHK_RET(interTempAlg.CalcRes(resReqInter));

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqIntra.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqInter.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = resReqIntra.queNum + resReqInter.queNum;
    CHK_RET(CalcParallelNotifyReq(algResReq.primQueueNum, resReqIntra.queNum, algResReq.queueNotifys));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, algResReq.links));
    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口，将对应的 instruction 添加到指令队列中
// 传入的insQue为一条主流
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParams0(
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

    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParams1(
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
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplitRate(
    std::vector<float>& splitDataSize) const
{
    // to do 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const RankGraph* rankGraph, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqInter;
    AlgTempResReq resReqIntra;
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
        HCCL_ERROR("[InsAllReduceParallelExecutor]resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    u32 totalQueueNum = resReqIntra.queNum + resReqInter.queNum;
    CHK_RET(InitQueue(totalQueueNum, requiredQue_));
    for (u32 qIdx = 0; qIdx < requiredQue_.size(); qIdx++) {
        if (qIdx < resReqIntra.queNum) {
            intraQue_.push_back(requiredQue_[qIdx]);
        } else {
            interQue_.push_back(requiredQue_[qIdx]);
        }
    }
    syncQueues_.emplace_back(intraQue_[0]);
    syncQueues_.emplace_back(interQue_[0]);

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, interLinks_));
    HCCL_INFO(
        "[InsAllReduceParallelExecutor] intraLinks_ size[%zu], interLinks_ size[%zu]", intraLinks_.size(),
        interLinks_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    ConnectedLinkMgr* linkMgr, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsAllReduceParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(linkMgr, resReqIntra));
    } else {
        HCCL_DEBUG("[InsAllReduceParallelExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));

    // 申请算法模板所需资源
    if (!(resReqIntra.queNum > 0 && resReqInter.queNum > 0)) {
        HCCL_ERROR("[InsAllReduceParallelExecutor] Intra queNum and Inter queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    u32 totalQueueNum = resReqIntra.queNum + resReqInter.queNum;
    CHK_RET(InitQueue(totalQueueNum, requiredQue_));
    for (u32 i = 0; i < requiredQue_.size(); i++) {
        if (i < resReqIntra.queNum) {
            intraQue_.push_back(requiredQue_[i]);
        } else {
            interQue_.push_back(requiredQue_[i]);
        }
    }
    syncQueues_.emplace_back(intraQue_[0]);
    syncQueues_.emplace_back(interQue_[0]);

    CHK_RET(PrepResLinks(myRank_, resReqIntra.links, linkMgr, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, resReqInter.links, linkMgr, interLinks_));
    HCCL_INFO(
        "[InsAllReduceParallelExecutor] intraLinks_ size[%zu], interLinks_ size[%zu]", intraLinks_.size(),
        interLinks_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcSendDataSize(
    u64& memBlockSize, float& SplitRate, u32& multipleIntra, u32& multipleInter)
{
    std::vector<float> dataSplitSize;
    GetParallelDataSplitRate(dataSplitSize);
    uint64_t templateNum = 2;
    if (multipleIntra == 0 && multipleInter == 0) {
        memBlockSize = UB_MAX_DATA_SIZE + UB_MAX_DATA_SIZE;
    } else if ((multipleIntra == 0 && multipleInter > 0) || (multipleInter == 0 && multipleIntra > 0)) {
        // 因为数据要交替在两个template中执行，因此最终要以数据处理量小的template为准
        if (multipleIntra > 0) {
            memBlockSize = std::min(static_cast<u64>(UB_MAX_DATA_SIZE), maxTmpMemSize_ / multipleIntra) * templateNum;
            Intra0ScratchSize = maxTmpMemSize_;
            Intra1ScratchSize = maxTmpMemSize_;
        } else {
            memBlockSize = std::min(static_cast<u64>(UB_MAX_DATA_SIZE), maxTmpMemSize_ / multipleInter) * templateNum;
            Inter0ScratchSize = maxTmpMemSize_;
            Inter1ScratchSize = maxTmpMemSize_;
        }
    } else { // multipleIntra >0 && multipleInter >0, 理论上dataSplitSize[0]=0.5时，scratch buffer利用率最大
        SplitRate = dataSplitSize[0];
        u32 subMultiple0 = static_cast<u32>(std::ceil(SplitRate * multipleIntra + (1 - SplitRate) * multipleInter));
        u32 subMultiple1 = static_cast<u32>(std::ceil((1 - SplitRate) * multipleIntra + SplitRate * multipleInter));
        u64 totalScratchMultiple = std::max(subMultiple0, subMultiple1);
        memBlockSize = std::min(static_cast<u64>(UB_MAX_DATA_SIZE), maxTmpMemSize_ / totalScratchMultiple);

        interScratchOffset0 = static_cast<u64>(memBlockSize * SplitRate * multipleIntra);
        interScratchOffset1 = static_cast<u64>(memBlockSize * (1 - SplitRate) * multipleIntra);
        Intra0ScratchSize = interScratchOffset0;
        Inter0ScratchSize = interScratchOffset1;
        Intra1ScratchSize = interScratchOffset1;
        Inter1ScratchSize = interScratchOffset0;
    }
    return HCCL_SUCCESS;
}

/*
 *@Desc: AICPU算法编排
 */
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsAllReduceParallelExecutor] AICPU Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));
    // 所以获取取级通信域的信息
    vTopo_ = topoInfo.vTopo;             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks;     // 本通信域内的 rank 集合
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]); // server内算法，比如mesh
    InsAlgTemplate1 tempAlgInter(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]); // server间算法，比如nhr

    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.InitReduceInfo(redOp_, dataType_);
    tempAlgInter.SetCollOp(op);

    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.InitReduceInfo(redOp_, dataType_);
    tempAlgIntra.SetCollOp(op);

    std::vector<std::map<u32, u32>> rank2PathNumMap;
    SetPathNumMapByLinkMgrMultiLevel(linkMgr, virtRanks_, myRank_, rank2PathNumMap);
    tempAlgIntra.setPathNumMap(rank2PathNumMap[0]);
    tempAlgInter.setPathNumMap(rank2PathNumMap[1]);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(linkMgr, tempAlgIntra, tempAlgInter));
    CHK_RET(GenInsQues(tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsAllReduceParallelExecutor] Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

/*
 *@Desc: Host算法编排
 */
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsAllReduceParallelExecutor] Host Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]); // server内算法，比如mesh
    InsAlgTemplate1 tempAlgInter(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]); // server间算法，比如nhr

    tempAlgIntra.InitReduceInfo(redOp_, dataType_);
    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.SetCollOp(op);

    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.SetCollOp(op); // CCU template需要传递op信息
    tempAlgInter.InitReduceInfo(redOp_, dataType_);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(rankGraph, tempAlgIntra, tempAlgInter));

    CHK_RET(GenInsQues(tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsAllReduceParallelExecutor] Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

/*
@Desc: 本方法主要实现的是跨框算法实现，如下图，框内和框间分别用不同的算法实现
/-------------------\    /-------------------\
|   /----\ /----\   |    |   /----\ /----\   |
|   |card| |card|   |    |   |card| |card|   |
|   \----/ \----/   |    |   \----/ \----/   |
|                   |    |                   |
|      Machine 1    |    |      Machine 2    |
\-------------------/    \-------------------/
*/
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllReduceParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenInsQues(
    InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    u64 alignedSize = 128; // 假设需要128字节对齐，太大会导致后续maxCountPerLoop计算有问题
    u32 multipleIntra = tempAlgIntra.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 multipleInter = tempAlgInter.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u64 memBlockSize = UB_MAX_DATA_SIZE;
    CalcSendDataSize(memBlockSize, dataSplitRate, multipleIntra, multipleInter);
    // dataSplitSize为分数，这里maxCountPerLoop对10取整，ScratchBufferSize为1M时可能会导致maxCountPerLoop为0；
    u64 maxCountPerLoop = (memBlockSize / dataTypeSize_ / 10 / alignedSize) * 10 * alignedSize;
    CHK_PRT_RET(
        maxCountPerLoop == 0,
        HCCL_ERROR("[InsAllReduceParallelExecutor] memBlockSize:%llu,maxCountPerLoop==0!.", memBlockSize),
        HcclResult::HCCL_E_INTERNAL);
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TemplateDataParams tempAlgParamsIntra0, tempAlgParamsInter0, tempAlgParamsInter1, tempAlgParamsIntra1;
    TempFuncs tempFuncs;
    tempFuncs.enableCounterNotify = false;
    tempFuncs.opMode = opMode_;
    tempFuncs.isBottom = true;
    tempFuncs.isForepart = true;
    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex == loopTimes - 1) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataCountPerLoopAixs0 = static_cast<u64>(dataSplitRate * currCount);
        u64 dataCountPerLoopAixs1 = currCount - dataCountPerLoopAixs0;
        // 第一步开始前同步
        CHK_RET(PreSyncQues(syncQueues_, 0));
        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + dataCountPerLoopAixs0 * dataTypeSize_;

        tempAlgParamsIntra0.buffInfo.scratchBuffSize = Intra0ScratchSize;
        GenTemplateAlgParams0(dataOffset0, dataCountPerLoopAixs0, 0, tempAlgParamsIntra0);
        // 把每个template需要的queue传进去，比如stars的mesh要传多条queue
        CHK_RET(tempAlgIntra.GenExtIns(tempFuncs, tempAlgParamsIntra0, intraLinks_, intraQue_));
        tempAlgParamsInter0.buffInfo.scratchBuffSize = Inter0ScratchSize;
        GenTemplateAlgParams0(dataOffset1, dataCountPerLoopAixs1, interScratchOffset0, tempAlgParamsInter0);
        CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter0, interLinks_, interQue_));
        CHK_RET(PostSyncQues(syncQueues_, 0));

        // 第二步开始前同步
        CHK_RET(PreSyncQues(syncQueues_, 0));
        tempAlgParamsInter1.buffInfo.scratchBuffSize = Inter1ScratchSize;
        GenTemplateAlgParams1(dataOffset0, dataCountPerLoopAixs0, interScratchOffset1, tempAlgParamsInter1);
        CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter1, interLinks_, interQue_));
        tempAlgParamsIntra1.buffInfo.scratchBuffSize = Intra1ScratchSize;
        GenTemplateAlgParams1(dataOffset1, dataCountPerLoopAixs1, 0, tempAlgParamsIntra1);
        CHK_RET(tempAlgIntra.GenExtIns(tempFuncs, tempAlgParamsIntra1, intraLinks_, intraQue_));
        CHK_RET(PostSyncQues(syncQueues_, 0));
    }
    return HcclResult::HCCL_SUCCESS;
}

// 算法注册
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLREDUCE, InsAllReduceParallelMesh1DNHR, InsAllReduceParallelExecutor, TopoMatchMeshNHR,
    InsTempAllReduceMesh1DTwoShot, InsTempAllReduceNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLREDUCE, InsAllReduceParallelMesh2DNHR, InsAllReduceParallelExecutor, TopoMatchConcurrMeshNHR,
    InsTempAllReduceMesh2DTwoShot, InsTempAllReduceNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLREDUCE, InsAllReduceParallelNHRNHR, InsAllReduceParallelExecutor, TopoMatchMeshNHR, InsTempAllReduceNHR,
    InsTempAllReduceNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLREDUCE, InsAllReduceParallelMesh1DNHRPcie, InsAllReduceParallelExecutor, TopoMatchMeshNHRPcie,
    InsTempAllReduceMesh1DTwoShot, InsTempAllReduceNHR);

#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLREDUCE, CcuAllReduceParallelMesh1DNHR, InsAllReduceParallelExecutor, TopoMatchMeshNHR,
    CcuTempAllReduceMeshMem2Mem1D, CcuTempAllReduceNHRMem2Mem1D);
#endif
} // namespace Hccl
