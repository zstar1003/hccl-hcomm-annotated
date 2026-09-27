/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "ins_coll_alg_registry.h"
#include "topo_match_mesh_nhr.h"
#include "alg_data_trans_wrapper.h"

#include "ins_temp_scatter_mesh_1d.h"
#include "ins_temp_scatter_nhr.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh.h"
#include "ins_broadcast_parallel_aicpu_executor.h"

namespace Hccl {

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    PreCalcRes(
        const RankGraph* rankGraph, AlgTempResReq& resReqIntraScatter, AlgTempResReq& resReqInterScatter,
        AlgTempResReq& resReqIntraAllGather, AlgTempResReq& resReqInterAllGather)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // 计算localRankSize
    CHK_RET(CalcLocalRankSize());

    // 实例化算法模板类
    InsAlgTemplate0 intraScatterTempAlg(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interScatterTempAlg(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]);
    InsAlgTemplate2 intraAllGatherTempAlg(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate3 interAllGatherTempAlg(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]);
    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    intraAllGatherTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interAllGatherTempAlg.setPathNumMap(rank2PathNumMap[1]);
    intraScatterTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interScatterTempAlg.setPathNumMap(rank2PathNumMap[1]);
    CHK_RET(CalcSingleAlgRes(
        intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg, rankGraph,
        resReqIntraScatter, resReqInterScatter, resReqIntraAllGather, resReqInterAllGather));
    return HcclResult::HCCL_SUCCESS;
}
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
template <typename T>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcSingleAlgRes(
        InsAlgTemplate0& intraScatter, InsAlgTemplate1& interScatter, InsAlgTemplate2& intraAllGather,
        InsAlgTemplate3& interAllGather, T* type, AlgTempResReq& resReqIntraScatter, AlgTempResReq& resReqInterScatter,
        AlgTempResReq& resReqIntraAllGather, AlgTempResReq& resReqInterAllGather) const
{
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detouring enabled.", __func__, myRank_);
        CHK_RET(intraScatter.CalcResDetour(type, resReqIntraScatter));
        CHK_RET(intraAllGather.CalcResDetour(type, resReqIntraAllGather));
    } else {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detouring disabled.", __func__, myRank_);
        CHK_RET(intraScatter.CalcRes(resReqIntraScatter));
        CHK_RET(intraAllGather.CalcRes(resReqIntraAllGather));
    }
    CHK_RET(interScatter.CalcRes(resReqInterScatter));
    CHK_RET(interAllGather.CalcRes(resReqInterAllGather));
    return HcclResult::HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] CalcRes start, rank[%d]", myRank_);

    // 计算和准备Queue资源
    AlgTempResReq resReqIntraScatter;
    AlgTempResReq resReqInterScatter;
    AlgTempResReq resReqIntraAllGather;
    AlgTempResReq resReqInterAllGather;

    CHK_RET(PreCalcRes(rankGraph, resReqIntraScatter, resReqInterScatter, resReqIntraAllGather, resReqInterAllGather));

    algResReq.topoInfo.UpdateMultiLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqIntraScatter.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqInterScatter.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqIntraAllGather.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqInterAllGather.links, algResReq.levelRankPairs));
    u32 intraQueNum = max(resReqIntraScatter.queNum, resReqIntraAllGather.queNum);
    u32 interQueNum = max(resReqInterScatter.queNum, resReqInterAllGather.queNum);

    algResReq.primQueueNum = intraQueNum + interQueNum;

    std::vector<std::tuple<QId, QId, u32>> notifyRequests;

    CHK_RET(CalcParallelNotifyReq(algResReq.primQueueNum, resReqIntraAllGather.queNum, algResReq.queueNotifys));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqIntraScatter.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqInterScatter.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqIntraAllGather.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqInterAllGather.links, algResReq.links));

    HCCL_INFO(
        "[InsBroadcastParallelAiCpuExecutor] CalcRes end, rank[%d], required total que num [%u], que notify num [%u]",
        myRank_, algResReq.primQueueNum, algResReq.queueNotifys.size());

    return HcclResult::HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] CalcResOffload start, rank[%d]", myRank_);

    (void)dataSize;
    u64 scratchMemSize = 200 * 1024 * 1024;
    resReq.requiredScratchMemSize = scratchMemSize; // 200MB

    // 计算和准备Queue资源
    AlgTempResReq resReqIntraScatter;
    AlgTempResReq resReqInterScatter;
    AlgTempResReq resReqIntraAllGather;
    AlgTempResReq resReqInterAllGather;

    CHK_RET(PreCalcRes(rankGraph, resReqIntraScatter, resReqInterScatter, resReqIntraAllGather, resReqInterAllGather));
    resReq.requiredSubQueNum = resReqIntraScatter.streamNum + resReqInterScatter.streamNum
                               + resReqIntraAllGather.streamNum + resReqInterAllGather.streamNum - 1;

    HCCL_INFO(
        "[InsBroadcastParallelAiCpuExecutor] CalcResOffload end, rank[%d], required sub que num is [%u]", myRank_,
        resReq.requiredSubQueNum);

    return HcclResult::HCCL_SUCCESS;
}

// Host展开
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    Orchestrate(const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] Host orchestrate begins.");

    // 初始化参数
    CHK_RET(Init(op, params, insQue));

    // 获取算法Topo信息
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // 计算localRankSize和localRoot
    CHK_RET(CalcLocalRankSize());
    CHK_RET(CalcLocalRoot());

    // 实例化算法模板类
    InsAlgTemplate0 intraScatterTempAlg(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interScatterTempAlg(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]);
    InsAlgTemplate2 intraAllGatherTempAlg(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate3 interAllGatherTempAlg(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]);
    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] Orchestrate SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    intraAllGatherTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interAllGatherTempAlg.setPathNumMap(rank2PathNumMap[1]);
    intraScatterTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interScatterTempAlg.setPathNumMap(rank2PathNumMap[1]);
    // 传入Template参数
    AlgTemplateInitPara(op, intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg);
    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(
        rankGraph, intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg));

    // 算法展开
    CHK_RET(GenInsQues(intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg));

    HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] Host orchestrate success.");
    return HcclResult::HCCL_SUCCESS;
}

// Aicpu展开
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue)
{
    HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] Aicpu orchestrate begins.");

    // 初始化参数
    CHK_RET(Init(op, params, insQue));

    // 获取算法Topo信息
    vTopo_ = topoInfo.vTopo;             // 本通信域内的通信平面
    virtRanks_ = topoInfo.virtRanks;     // 本通信域内的 rank 集合
    virtRankMap_ = topoInfo.virtRankMap; // 本通信域内的 rank 映射表

    // 计算localRankSize和localRoot
    CHK_RET(CalcLocalRankSize());
    CHK_RET(CalcLocalRoot());

    // 实例化算法模板类
    InsAlgTemplate0 intraScatterTempAlg(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interScatterTempAlg(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]);
    InsAlgTemplate2 intraAllGatherTempAlg(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate3 interAllGatherTempAlg(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]);

    // 传入Template参数
    AlgTemplateInitPara(op, intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg);
    std::vector<std::map<u32, u32>> rank2PathNumMap;
    CHK_RET(SetPathNumMapByLinkMgrMultiLevel(linkMgr, virtRanks_, myRank_, rank2PathNumMap));
    intraAllGatherTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interAllGatherTempAlg.setPathNumMap(rank2PathNumMap[1]);
    intraScatterTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interScatterTempAlg.setPathNumMap(rank2PathNumMap[1]);
    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(
        linkMgr, intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg));

    // 算法展开
    CHK_RET(GenInsQues(intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg));

    HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] Aicpu orchestrate success.");
    return HcclResult::HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
template <typename T>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    PrepareRes(
        T* type, AlgTempResReq& resReqIntraScatter, AlgTempResReq& resReqInterScatter,
        AlgTempResReq& resReqIntraAllGather, AlgTempResReq& resReqInterAllGather)
{
    // 申请算法模板所需资源
    if (resReqIntraScatter.queNum == 0 || resReqInterScatter.queNum == 0 || resReqIntraAllGather.queNum == 0
        || resReqInterAllGather.queNum == 0) {
        HCCL_ERROR("queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    u32 intraQueNum = max(resReqIntraScatter.queNum, resReqIntraAllGather.queNum);
    u32 interQueNum = max(resReqInterScatter.queNum, resReqInterAllGather.queNum);

    u32 totalQueueNum = intraQueNum + interQueNum;
    CHK_RET(InitQueue(totalQueueNum, requiredQue_));
    for (u32 i = 0; i < requiredQue_.size(); i++) {
        if (i < intraQueNum) {
            intraQue_.push_back(requiredQue_.at(i));
        } else {
            interQue_.push_back(requiredQue_.at(i));
        }
    }

    // 每个算法的第0条流用于同步
    syncQueues_.emplace_back(intraQue_.at(0));
    syncQueues_.emplace_back(interQue_.at(0));

    CHK_RET(WrapPrepResLinks(type, resReqIntraScatter.links, scatterIntraLinks_));
    CHK_RET(WrapPrepResLinks(type, resReqInterScatter.links, scatterInterLinks_));
    CHK_RET(WrapPrepResLinks(type, resReqIntraAllGather.links, allGatherIntraLinks_));
    CHK_RET(WrapPrepResLinks(type, resReqInterAllGather.links, allGatherInterLinks_));
    HCCL_INFO(
        "[InsBroadcastParallelAiCpuExecutor] scatterIntraLinks size[%zu], scatterInterLinks size[%zu], "
        "allGatherIntraLinks size[%zu], allGatherInterLinks size[%zu]",
        scatterIntraLinks_.size(), scatterInterLinks_.size(), allGatherIntraLinks_.size(), allGatherInterLinks_.size());
    return HcclResult::HCCL_SUCCESS;
}

// Host
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    PrepareResForTemplate(
        const RankGraph* rankGraph, InsAlgTemplate0& intraScatterTempAlg, InsAlgTemplate1& interScatterTempAlg,
        InsAlgTemplate2& intraAllGatherTempAlg, InsAlgTemplate3& interAllGatherTempAlg)
{
    AlgTempResReq resReqIntraScatter;
    AlgTempResReq resReqInterScatter;
    AlgTempResReq resReqIntraAllGather;
    AlgTempResReq resReqInterAllGather;
    CHK_RET(CalcSingleAlgRes(
        intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg, rankGraph,
        resReqIntraScatter, resReqInterScatter, resReqIntraAllGather, resReqInterAllGather));
    CHK_RET(PrepareRes(rankGraph, resReqIntraScatter, resReqInterScatter, resReqIntraAllGather, resReqInterAllGather));

    return HcclResult::HCCL_SUCCESS;
}

// Aicpu
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    PrepareResForTemplate(
        ConnectedLinkMgr* linkMgr, InsAlgTemplate0& intraScatterTempAlg, InsAlgTemplate1& interScatterTempAlg,
        InsAlgTemplate2& intraAllGatherTempAlg, InsAlgTemplate3& interAllGatherTempAlg)
{
    AlgTempResReq resReqIntraScatter;
    AlgTempResReq resReqInterScatter;
    AlgTempResReq resReqIntraAllGather;
    AlgTempResReq resReqInterAllGather;
    CHK_RET(CalcSingleAlgRes(
        intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg, linkMgr,
        resReqIntraScatter, resReqInterScatter, resReqIntraAllGather, resReqInterAllGather));
    CHK_RET(PrepareRes(linkMgr, resReqIntraScatter, resReqInterScatter, resReqIntraAllGather, resReqInterAllGather));

    return HcclResult::HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsBroadcastParallelAiCpuExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,
    InsAlgTemplate3>::CalcSlice(std::vector<double>& splitDataSize, float scratchMaxMultiple, SliceConfig& slice)
{
    // 数据切分
    u64 sliceCount = std::min(static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_, dataCount_);
    if (scratchMaxMultiple > 0 && maxTmpMemSize_ > 0) {
        u64 scratchCount = maxTmpMemSize_ / dataTypeSize_; // 按照count来切分
        sliceCount = min(
            sliceCount,
            static_cast<u64>(static_cast<double>(scratchCount) / scratchMaxMultiple)); // 向下取整，防止Scratch溢出
    }
    /* 刷新sliceCountPart0 和sliceCountPart1确保是interLocalRankSize_ * intraLocalRankSize_整倍数 */
    u64 sliceCountPart0 = static_cast<u64>(sliceCount * splitDataSize.at(0));
    sliceCountPart0
        = (sliceCountPart0 / interLocalRankSize_ / intraLocalRankSize_) * interLocalRankSize_ * intraLocalRankSize_;
    u64 sliceCountPart1 = static_cast<u64>(sliceCount * splitDataSize.at(1));
    sliceCountPart1
        = (sliceCountPart1 / interLocalRankSize_ / intraLocalRankSize_) * interLocalRankSize_ * intraLocalRankSize_;
    sliceCount = sliceCountPart0 + sliceCountPart1;
    // 计算循环次数, 如果sliceCountPart0和liceCountPart1为0说明只有一块数据都是尾块
    u32 loopTimes = sliceCount == 0 ? 1 : (dataCount_ + sliceCount - 1) / sliceCount;
    // 计算尾块
    u64 finalSliceCount = dataCount_ - (loopTimes - 1) * sliceCount;
    u64 finalTailCount = finalSliceCount % (interLocalRankSize_ * intraLocalRankSize_);
    u64 finalSliceCountPart1 = static_cast<u64>(finalSliceCount * splitDataSize.at(1));
    //  刷新sliceCountPart0 和sliceCountPart1确保是interLocalRankSize_ * intraLocalRankSize_整倍数
    finalSliceCountPart1 = (finalSliceCountPart1 / interLocalRankSize_ / intraLocalRankSize_) * interLocalRankSize_
                           * intraLocalRankSize_;
    u64 finalSliceCountPart0 = finalSliceCount - finalSliceCountPart1 - finalTailCount;
    slice.loopTimes = loopTimes;
    slice.sliceCount = sliceCount;
    slice.sliceCountPart0 = sliceCountPart0;
    slice.sliceCountPart1 = sliceCountPart1;
    slice.finalSliceCount = finalSliceCount;
    slice.finalSliceCountPart0 = finalSliceCountPart0;
    slice.finalSliceCountPart1 = finalSliceCountPart1;
    // 结构体定义中必须确保finalTailCountPart0和finalTailCountPart1初始化为0
    (finalSliceCountPart0 < finalSliceCountPart1 ? slice.finalTailCountPart0 : slice.finalTailCountPart1)
        = finalTailCount;
    return;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    StageProcess(DataParameters& dataParameters, std::vector<StageProcAlgPara>& algParaVec)
{
    TemplateDataParams tempAlgParams;
    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = false;
    for (u32 step = 0; step < algParaVec.size(); step++) {
        bool isFirst = (step == 0);
        // 先处理part0数据
        CHK_RET(PreSyncQues(syncQueues_, 0));
        // 第一步的时候server间topo包含root_的rank进行展开，其它rank不展开
        u64 sliceSizePart0 = max(dataParameters.sliceSize.at(0).at(step), dataParameters.tailSize.at(0).at(step));
        if ((!isFirst || interLocalRoot_ == root_) && (sliceSizePart0 > 0)) {
            GenDataParamsStage(0, step, dataParameters, tempAlgParams);
            CHK_RET(algParaVec.at(step).part0FuncPtr(
                tempFuncs, tempAlgParams, algParaVec.at(step).part0links, algParaVec.at(step).part0Que));
        }
        // 再处理part1数据, 第一步的时候server内topo包含root_的rank进行展开，其它rank不展开
        u64 sliceSizePart1 = max(dataParameters.sliceSize.at(1).at(step), dataParameters.tailSize.at(1).at(step));
        if ((!isFirst || intraLocalRoot_ == root_) && sliceSizePart1 > 0) {
            // 数据1的server内的scatter算法
            GenDataParamsStage(1, step, dataParameters, tempAlgParams);
            CHK_RET(algParaVec.at(step).part1FuncPtr(
                tempFuncs, tempAlgParams, algParaVec.at(step).part1links, algParaVec.at(step).part1Que));
        }
        CHK_RET(PostSyncQues(syncQueues_, 0));
    }
    return HcclResult::HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsBroadcastParallelAiCpuExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GenInsQues(
        InsAlgTemplate0& intraScatterTempAlg, InsAlgTemplate1& interScatterTempAlg,
        InsAlgTemplate2& intraAllGatherTempAlg, InsAlgTemplate3& interAllGatherTempAlg)
{
    LogAlgInfo(intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg);

    std::vector<double> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);
    ScratchMultiple scratchMultiple;
    CalcScratchMultiple(
        dataSplitSize, scratchMultiple, intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg,
        interAllGatherTempAlg);
    SliceConfig slice;
    CalcSlice(dataSplitSize, scratchMultiple.maxMultiple, slice);

    std::vector<StageProcAlgPara> stageProcAlgParaVec;
    InitStageProcAlgParaVec(
        stageProcAlgParaVec, intraScatterTempAlg, interScatterTempAlg, intraAllGatherTempAlg, interAllGatherTempAlg);
    DataParameters dataParameters;
    InitDataParameters(slice, scratchMultiple, dataParameters);
    for (u32 loopIndex = 0; loopIndex < slice.loopTimes - 1; loopIndex++) {
        dataParameters.dataOffset[0] = loopIndex * slice.sliceCount * dataTypeSize_;
        dataParameters.dataOffset[1] = dataParameters.dataOffset[0] + slice.sliceCountPart0 * dataTypeSize_;
        CHK_RET(StageProcess(dataParameters, stageProcAlgParaVec));
    }
    InitFinalSliceDataParameters(slice, scratchMultiple, dataParameters);
    dataParameters.dataOffset[0] = (slice.loopTimes - 1) * slice.sliceCount * dataTypeSize_;
    dataParameters.dataOffset[1]
        = dataParameters.dataOffset[0] + (slice.finalSliceCountPart0 + slice.finalTailCountPart0) * dataTypeSize_;
    CHK_RET(StageProcess(dataParameters, stageProcAlgParaVec));
    return HcclResult::HCCL_SUCCESS;
}

// 算法注册
INS_REGISTER_IMPL_BY_FOUR_TEMPS(
    OpType::BROADCAST, AiCpuInsBroadcastParallelMesh1DNHR, InsBroadcastParallelAiCpuExecutor, TopoMatchMeshNHR,
    InsTempScatterMesh1D, InsTempScatterNHR, InsTempAllGatherMesh1D, InsTempAllGatherNHR);
} // namespace Hccl
