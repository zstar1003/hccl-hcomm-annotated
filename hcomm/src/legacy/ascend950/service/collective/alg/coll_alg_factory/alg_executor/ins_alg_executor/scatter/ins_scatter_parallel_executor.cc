/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_scatter_parallel_executor.h"

#include <cmath>

#include "log.h"

#include "ins_coll_alg_registry.h"

#include "topo_match_mesh_nhr.h"
#include "topo_match_mesh_nhr_pcie.h"
#include "alg_data_trans_wrapper.h"

#include "ins_temp_scatter_mesh_1d.h"
#include "ins_temp_scatter_nhr.h"

#include "ccu_temp_scatter_mesh_1D.h"
#include "ccu_temp_scatter_nhr_1D_mem2mem.h"

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsScatterParallelExecutor()
    : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsScatterParallelExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    u64 scratchMemSize = 200 * 1024 * 1024;
    resReq.requiredScratchMemSize = scratchMemSize; // 200MB
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));
    InsAlgTemplate0 intraTempAlg(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlg(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);

    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsScatterParallelExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    intraTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interTempAlg.setPathNumMap(rank2PathNumMap[1]);
    // calculate required insQues and prepare queue
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsReduceScatterParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[InsReduceScatterParallelExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }

    CHK_RET(interTempAlg.CalcRes(resReqInter));

    resReq.requiredSubQueNum = resReqIntra.streamNum + resReqInter.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateMultiLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));
    // instantiate a template
    InsAlgTemplate0 intraTempAlg(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlg(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);
    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsScatterParallelExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    intraTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interTempAlg.setPathNumMap(rank2PathNumMap[1]);
    // calculate required insQues and prepare queue
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsScatterParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[InsScatterParallelExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }
    CHK_RET(interTempAlg.CalcRes(resReqInter));

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqIntra.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqInter.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = resReqIntra.streamNum + resReqInter.streamNum;
    CHK_RET(CalcParallelNotifyReq(algResReq.primQueueNum, resReqIntra.queNum, algResReq.queueNotifys));

    HCCL_DEBUG("[InsScatterParallelExecutor] algResReq.primQueueNum %u", algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口，将对应的 instruction 添加到指令队列中
// 传入的insQue为一条主流
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra0(
    const u64 dataOffset, const u64 dataCountPerLoopAixs0, const u64 scratchOffset,
    TemplateDataParams& tempAlgParamsIntra0) const
{
    tempAlgParamsIntra0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsIntra0.buffInfo.outBuffType = BufferType::SCRATCH;
    tempAlgParamsIntra0.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsIntra0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra0.buffInfo.outBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.buffInfo.scratchBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.tailSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.inputSliceStride = dataSize_;
    tempAlgParamsIntra0.outputSliceStride = 0;
    tempAlgParamsIntra0.repeatNum = rankSizeLevel1_;
    tempAlgParamsIntra0.inputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsIntra0.outputRepeatStride = tempAlgParamsIntra0.sliceSize;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter0(
    const u64 dataOffset, const u64 dataCountPerLoopAixs0, const u64 scratchOffset,
    TemplateDataParams& tempAlgParamsInter0) const
{
    tempAlgParamsInter0.buffInfo.inBuffType = BufferType::SCRATCH;
    tempAlgParamsInter0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsInter0.buffInfo.inBuffBaseOff = scratchOffset;
    tempAlgParamsInter0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.scratchBuffBaseOff = scratchOffset;
    tempAlgParamsInter0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.tailSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.inputSliceStride = tempAlgParamsInter0.sliceSize;
    tempAlgParamsInter0.outputSliceStride = 0;
    tempAlgParamsInter0.repeatNum = 1;
    tempAlgParamsInter0.inputRepeatStride = 0;
    tempAlgParamsInter0.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(
    const u64 dataOffset, const u64 dataCountPerLoopAixs1, const u64 scratchOffset,
    TemplateDataParams& tempAlgParamsInter1) const
{
    tempAlgParamsInter1.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsInter1.buffInfo.outBuffType = BufferType::SCRATCH;
    tempAlgParamsInter1.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsInter1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter1.buffInfo.outBuffBaseOff = scratchOffset;
    tempAlgParamsInter1.buffInfo.scratchBuffBaseOff = scratchOffset;
    tempAlgParamsInter1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsInter1.tailSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsInter1.inputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter1.outputSliceStride = 0;
    tempAlgParamsInter1.repeatNum = rankSizeLevel0_;
    tempAlgParamsInter1.inputRepeatStride = dataSize_;
    tempAlgParamsInter1.outputRepeatStride = tempAlgParamsInter1.sliceSize;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra1(
    const u64 dataOffset, const u64 dataCountPerLoopAixs1, const u64 scratchOffset,
    TemplateDataParams& tempAlgParamsIntra1) const
{
    tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::SCRATCH;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsIntra1.buffInfo.inBuffBaseOff = scratchOffset;
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.scratchBuffBaseOff = scratchOffset;
    tempAlgParamsIntra1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsIntra1.tailSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsIntra1.inputSliceStride = tempAlgParamsIntra1.sliceSize;
    tempAlgParamsIntra1.outputSliceStride = 0;
    tempAlgParamsIntra1.repeatNum = 1;
    tempAlgParamsIntra1.inputRepeatStride = 0;
    tempAlgParamsIntra1.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<double>& splitDataSize) const
{
    // to do 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const RankGraph* rankGraph, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqInter;
    AlgTempResReq resReqIntra;
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], detour enabled.", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[%s] Rank[%d], detour disabled.", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));

    // 申请算法模板所需资源
    if (!(resReqIntra.queNum > 0 && resReqInter.queNum > 0)) {
        HCCL_ERROR("[InsScatterParallelExecutor]resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    u32 totalQueueNum = resReqIntra.queNum + resReqInter.queNum;
    CHK_RET(InitQueue(totalQueueNum, requiredQue_));
    for (u32 idx = 0; idx < requiredQue_.size(); idx++) {
        if (idx < resReqIntra.queNum) {
            intraQue_.push_back(requiredQue_[idx]);
        } else {
            interQue_.push_back(requiredQue_[idx]);
        }
    }
    syncQueues_.emplace_back(intraQue_[0]);
    syncQueues_.emplace_back(interQue_[0]);

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, interLinks_));
    HCCL_INFO(
        "[InsScatterParallelExecutor] intraLinks_ size[%zu], interLinks_ size[%zu]", intraLinks_.size(),
        interLinks_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    ConnectedLinkMgr* linkMgr, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detour enabled.", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(linkMgr, resReqIntra));
    } else {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detouring disabled.", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));

    // 申请算法模板所需资源
    if (!(resReqIntra.queNum > 0 && resReqInter.queNum > 0)) {
        HCCL_ERROR("[InsScatterParallelExecutor]resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
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
        "[InsScatterParallelExecutor] intraLinks_ size[%zu], interLinks_ size[%zu]", intraLinks_.size(),
        interLinks_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    // init and check params
    CHK_RET(Init(op, params, insQue));
    dataType_ = op.dataType;
    virtRanks_ = topoInfo.virtRanks;
    vTopo_ = topoInfo.vTopo;
    virtRankMap_ = topoInfo.virtRankMap;
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));
    rankIdxLevel0_ = myRank_ % vTopo_[0][0].size();
    rankIdxLevel1_ = myRank_ / vTopo_[0][0].size();

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]); // server内算法，比如mesh
    InsAlgTemplate1 tempAlgInter(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]); // server间算法，比如nhr
    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsScatterParallelExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByLinkMgrMultiLevel(linkMgr, virtRanks_, myRank_, rank2PathNumMap));
    tempAlgIntra.setPathNumMap(rank2PathNumMap[0]);
    tempAlgInter.setPathNumMap(rank2PathNumMap[1]);

    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.SetCollOp(op); // CCU template需要传递op信息
    tempAlgIntra.SetDataType(dataType_);
    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.SetCollOp(op); // CCU template需要传递op信息
    tempAlgInter.SetDataType(dataType_);
    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(linkMgr, tempAlgIntra, tempAlgInter));

    CHK_RET(GenInsQuesHost(tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsScatterParallelExecutor] Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsScatterParallelExecutor] Host Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));
    dataType_ = op.dataType;
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));
    rankIdxLevel0_ = myRank_ % vTopo_[0][0].size();
    rankIdxLevel1_ = myRank_ / vTopo_[0][0].size();

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]); // server内算法，比如mesh
    InsAlgTemplate1 tempAlgInter(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]); // server间算法，比如nhr

    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsScatterParallelExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    tempAlgIntra.setPathNumMap(rank2PathNumMap[0]);
    tempAlgInter.setPathNumMap(rank2PathNumMap[1]);

    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.SetCollOp(op); // CCU template需要传递op信息
    tempAlgIntra.SetDataType(dataType_);
    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.SetCollOp(op); // CCU template需要传递op信息
    tempAlgInter.SetDataType(dataType_);
    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(rankGraph, tempAlgIntra, tempAlgInter));

    CHK_RET(GenInsQuesHost(tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsScatterParallelExecutor] Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenInsQuesHost(
    InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    HCCL_INFO("[InsScatterParallelExecutor] AlgTemplate intra server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsScatterParallelExecutor] AlgTemplate inter server is [%s]", tempAlgInter.Describe().c_str());
    std::vector<double> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);
    double scratchMultipleIntra = std::max(dataSplitSize[0] * rankSizeLevel1_, dataSplitSize[1] * rankSizeLevel0_);
    double scratchMultipleInter = std::max(dataSplitSize[1] * rankSize_, dataSplitSize[0] * rankSize_);
    double totalScratchMultiple = scratchMultipleIntra + scratchMultipleInter;
    u64 scratchMemBlockSize = maxTmpMemSize_;
    if (totalScratchMultiple > 0) {
        // data0和data1的count需要和申请的scratch mem大小对应
        u64 tmpMemBlockCount = u64(maxTmpMemSize_ / totalScratchMultiple) / dataTypeSize_;
        scratchMemBlockSize
            = (u64(dataSplitSize[0] * tmpMemBlockCount) + u64(dataSplitSize[1] * tmpMemBlockCount)) * dataTypeSize_;
    }
    u64 intraScratchOffset = 0;
    u64 interScratchOffset = static_cast<u64>(scratchMultipleIntra * scratchMemBlockSize);
    u64 maxCountPerLoop
        = std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_;

    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TemplateDataParams tempAlgParamsInter1, tempAlgParamsIntra1, tempAlgParamsIntra0, tempAlgParamsInter0;
    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.isForepart = true;
    tempFuncs.isBottom = true;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex == loopTimes - 1) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataCountPerLoopAixs0 = static_cast<u64>(dataSplitSize[0] * currCount);
        u64 dataCountPerLoopAixs1 = currCount - dataCountPerLoopAixs0;

        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + dataCountPerLoopAixs0 * dataTypeSize_;

        // 第一步开始前同步
        CHK_RET(PreSyncQues(syncQueues_, 0));
        // 数据0的server内的mesh算法
        if (rankIdxLevel1_ == root_ / rankSizeLevel0_) {
            GenTemplateAlgParamsIntra0(dataOffset0, dataCountPerLoopAixs0, intraScratchOffset, tempAlgParamsIntra0);
            tempAlgIntra.SetRoot(root_);
            // 把每个template需要的queue传进去，比如stars的mesh要传多条queue
            CHK_RET(tempAlgIntra.GenExtIns(
                tempFuncs, tempAlgParamsIntra0, intraLinks_, intraQue_)); // Todo: 这里要把tempFuncs去掉
        }
        if (rankIdxLevel0_ == root_ % rankSizeLevel0_) {
            // 数据1的server间的nhr算法
            GenTemplateAlgParamsInter1(dataOffset1, dataCountPerLoopAixs1, interScratchOffset, tempAlgParamsInter1);
            tempAlgInter.SetRoot(root_);
            CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter1, interLinks_, interQue_));
        }
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncQues(syncQueues_, 0));
        // 第二步开始前同步
        CHK_RET(PreSyncQues(syncQueues_, 0));
        // 数据0的server间的nhr算法
        GenTemplateAlgParamsInter0(dataOffset0, dataCountPerLoopAixs0, intraScratchOffset, tempAlgParamsInter0);
        tempAlgInter.SetRoot(root_ / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
        CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter0, interLinks_, interQue_));
        // 数据1的server内的mesh算法
        GenTemplateAlgParamsIntra1(dataOffset1, dataCountPerLoopAixs1, interScratchOffset, tempAlgParamsIntra1);
        tempAlgIntra.SetRoot(root_ % rankSizeLevel0_ + rankIdxLevel1_ * rankSizeLevel0_);
        CHK_RET(tempAlgIntra.GenExtIns(tempFuncs, tempAlgParamsIntra1, intraLinks_, intraQue_));
        // 尾同步
        CHK_RET(PostSyncQues(syncQueues_, 0));
    }
    return HcclResult::HCCL_SUCCESS;
}

INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::SCATTER, InsScatterParallelMesh1DNHR, InsScatterParallelExecutor, TopoMatchMeshNHR, InsTempScatterMesh1D,
    InsTempScatterNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::SCATTER, InsScatterParallelMesh1DNHRPcie, InsScatterParallelExecutor, TopoMatchMeshNHRPcie,
    InsTempScatterMesh1D, InsTempScatterNHR);
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::SCATTER, CcuScatterParallelMesh1DNHR, InsScatterParallelExecutor, TopoMatchMeshNHR, CcuTempScatterMesh1D,
    CcuTempScatterNHRMem2Mem1D);
#endif
} // namespace Hccl
