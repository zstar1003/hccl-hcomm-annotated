/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_all_reduce_parallel_executor_opt.h"

#include "log.h"

#include "ins_coll_alg_registry.h"

#include "topo_match_mesh_nhr.h"
#include "alg_data_trans_wrapper.h"

#include "ins_temp_reduce_scatter_mesh_1D.h"

#include "ins_temp_all_gather_mesh.h"

namespace Hccl {
constexpr u64 MAX_OFFLOAD_SCRATCH_SIZE = 200 * 1024 * 1024; // 200M

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
InsAllReduceParallelExecutorV2<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::InsAllReduceParallelExecutorV2()
    : InsCollAlgBase()
{}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
InsAllReduceParallelExecutorV2<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::~InsAllReduceParallelExecutorV2()
{}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcResOffload(const RankGraph* rankGraph, [[maybe_unused]] const u64& dataSize, CollOffloadOpResReq& resReq)
{
    HCCL_INFO("[InsAllReduceParallelExecutorV2] CalcResOffload begins.");
    resReq.requiredScratchMemSize = MAX_OFFLOAD_SCRATCH_SIZE; // 200MB
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    CHK_RET(CalcLocalRankSize());

    InsAlgTemplate0 intraTempAlgRS(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlgRS(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);
    InsAlgTemplate2 intraTempAlgAG(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate3 interTempAlgAG(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);

    // 设置链路信息
    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsAllReduceParallelExecutorV2] CalcResOffload SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    intraTempAlgRS.setPathNumMap(rank2PathNumMap[0]);
    interTempAlgRS.setPathNumMap(rank2PathNumMap[1]);
    intraTempAlgAG.setPathNumMap(rank2PathNumMap[0]);
    interTempAlgAG.setPathNumMap(rank2PathNumMap[1]);

    AlgTempResReq resReqIntraRS;
    AlgTempResReq resReqInterRS;

    CHK_RET(intraTempAlgRS.CalcRes(resReqIntraRS));
    CHK_RET(interTempAlgRS.CalcRes(resReqInterRS));

    AlgTempResReq resReqIntraAG;
    AlgTempResReq resReqInterAG;

    CHK_RET(intraTempAlgAG.CalcRes(resReqIntraAG));
    CHK_RET(interTempAlgAG.CalcRes(resReqInterAG));

    // 算法从流数量 = Σ(temp的que数量 + temp的从流数量 * temp调用次数) - 算法主流数量
    resReq.requiredSubQueNum
        = std::max((resReqIntraAG.queNum + resReqInterAG.queNum), (resReqIntraRS.queNum + resReqInterRS.queNum)) - 1;
    HCCL_INFO("[InsAllReduceParallelExecutorV2::CalcResOffload]requiredSubQueNum = %llu", resReq.requiredSubQueNum);

    return HcclResult::HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    HCCL_INFO("[InsFourTemplateAllReduceExecutor] CalcRes begins.");

    // 拓扑匹配
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateMultiLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    CHK_RET(CalcLocalRankSize());

    // 创建四个模板实例
    InsAlgTemplate0 tempAlgRSIntra(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 tempAlgRSInter(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);
    InsAlgTemplate2 tempAlgAGIntra(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate3 tempAlgAGInter(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);

    // 设置链路信息
    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsAllReduceParallelExecutorV2] CalcResOffload SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    tempAlgRSIntra.setPathNumMap(rank2PathNumMap[0]);
    tempAlgRSInter.setPathNumMap(rank2PathNumMap[1]);
    tempAlgAGIntra.setPathNumMap(rank2PathNumMap[0]);
    tempAlgAGInter.setPathNumMap(rank2PathNumMap[1]);

    // 计算各模板资源需求
    AlgTempResReq resReqRSIntra, resReqRSInter, resReqAGIntra, resReqAGInter;

    CHK_RET(tempAlgRSIntra.CalcRes(resReqRSIntra));
    CHK_RET(tempAlgRSInter.CalcRes(resReqRSInter));
    CHK_RET(tempAlgAGIntra.CalcRes(resReqAGIntra));
    CHK_RET(tempAlgAGInter.CalcRes(resReqAGInter));

    // 计算链接信息
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqRSIntra.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqRSInter.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqAGIntra.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqAGInter.links, algResReq.levelRankPairs));

    algResReq.primQueueNum
        = std::max(resReqRSIntra.queNum, resReqAGIntra.queNum) + std::max(resReqRSInter.queNum, resReqAGInter.queNum);
    HCCL_INFO("[InsFourTemplateAllReduceExecutor::CalcRes] primQueueNum = %u", algResReq.primQueueNum);

    std::vector<std::tuple<QId, QId, u32>> notifyRequests;
    for (QId q = 1; q < algResReq.primQueueNum; q++) {
        notifyRequests.emplace_back(std::make_tuple(0, q, 0));
        notifyRequests.emplace_back(std::make_tuple(q, 0, 0));
    }

    u32 tempMasterQId = std::max(resReqRSIntra.queNum, resReqAGIntra.queNum);
    for (QId q = tempMasterQId + 1; q < algResReq.primQueueNum; q++) {
        notifyRequests.emplace_back(std::make_tuple(tempMasterQId, q, 0));
        notifyRequests.emplace_back(std::make_tuple(q, tempMasterQId, 0));
    }

    algResReq.queueNotifys = notifyRequests;

    // 计算链接资源
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqRSIntra.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqRSInter.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqAGIntra.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqAGInter.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GetParallelDataSplitRate(std::vector<float>& splitDataSize) const
{
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult InsAllReduceParallelExecutorV2<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::CalcLocalRankSize()
{
    uint64_t virtRanks_2 = 2;
    CHK_PRT_RET(
        virtRanks_.size() < virtRanks_2, HCCL_ERROR("[CalcLocalRankSize] virtRanks level num is smaller than 2."),
        HcclResult::HCCL_E_INTERNAL);

    rankSizeLevel0_ = virtRanks_.at(0).size();
    rankSizeLevel1_ = virtRanks_.at(1).size();
    CHK_PRT_RET(
        rankSizeLevel0_ == 0 || rankSizeLevel1_ == 0,
        HCCL_ERROR(
            "[CalcLocalRankSize] rankSizeLevel0_[%llu] or rankSizeLevel1_[%llu] is 0, rankSize will be 0.",
            rankSizeLevel0_, rankSizeLevel1_),
        HcclResult::HCCL_E_INTERNAL);
    rankSize_ = rankSizeLevel0_ * rankSizeLevel1_;

    // 计算当前 rank 在各层级中的索引
    if (virtRankMap_[0].find(myRank_) != virtRankMap_[0].end()) {
        rankIdxLevel0_ = virtRankMap_[0][myRank_];
    } else {
        HCCL_ERROR("[CalcLocalRankSize] rank [%d] is not in level 0 topo", myRank_);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (virtRankMap_[1].find(myRank_) != virtRankMap_[1].end()) {
        rankIdxLevel1_ = virtRankMap_[1][myRank_];
    } else {
        HCCL_ERROR("[CalcLocalRankSize] rank [%d] is not in level 1 topo", myRank_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_INFO(
        "[CalcLocalRankSize] localRankSize: myRank[%d] rankSizeLevel0_[%u] rankSizeLevel1_[%u] "
        "rankIdxLevel0_[%llu] rankIdxLevel1_[%llu]",
        myRank_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    return HcclResult::HCCL_SUCCESS;
};

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    PrepareResForTemplate(
        const RankGraph* rankGraph, InsAlgTemplate0& tempAlgIntraRS, InsAlgTemplate1& tempAlgInterRS,
        InsAlgTemplate2& tempAlgIntraAG, InsAlgTemplate3& tempAlgInterAG)
{
    AlgTempResReq resReqIntraRS, resReqInterRS, resReqIntraAG, resReqInterAG;

    CHK_RET(tempAlgIntraRS.CalcRes(resReqIntraRS));
    CHK_RET(tempAlgInterRS.CalcRes(resReqInterRS));
    CHK_RET(tempAlgIntraAG.CalcRes(resReqIntraAG));
    CHK_RET(tempAlgInterAG.CalcRes(resReqInterAG));

    CHK_RET(CalcQue(resReqIntraRS, resReqInterRS, resReqIntraAG, resReqInterAG));

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqIntraRS.links, intraRSLinks_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqInterRS.links, interRSLinks_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqIntraAG.links, intraAGLinks_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqInterAG.links, interAGLinks_));
    HCCL_INFO(
        "[InsAllReduceParallelExecutorV2] intraRSLinks_ size[%zu], interRSLinks_ size[%zu], intraAGLinks_ size[%zu], "
        "interAGLinks_ size[%zu]",
        intraRSLinks_.size(), interRSLinks_.size(), intraAGLinks_.size(), interAGLinks_.size());
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    PrepareResForTemplate(
        ConnectedLinkMgr* linkMgr, InsAlgTemplate0& tempAlgIntraRS, InsAlgTemplate1& tempAlgInterRS,
        InsAlgTemplate2& tempAlgIntraAG, InsAlgTemplate3& tempAlgInterAG)
{
    AlgTempResReq resReqIntraRS, resReqInterRS, resReqIntraAG, resReqInterAG;

    CHK_RET(tempAlgIntraRS.CalcRes(resReqIntraRS));
    CHK_RET(tempAlgInterRS.CalcRes(resReqInterRS));
    CHK_RET(tempAlgIntraAG.CalcRes(resReqIntraAG));
    CHK_RET(tempAlgInterAG.CalcRes(resReqInterAG));

    CHK_RET(CalcQue(resReqIntraRS, resReqInterRS, resReqIntraAG, resReqInterAG));

    CHK_RET(PrepResLinks(myRank_, resReqIntraRS.links, linkMgr, intraRSLinks_));
    CHK_RET(PrepResLinks(myRank_, resReqInterRS.links, linkMgr, interRSLinks_));
    CHK_RET(PrepResLinks(myRank_, resReqIntraAG.links, linkMgr, intraAGLinks_));
    CHK_RET(PrepResLinks(myRank_, resReqInterAG.links, linkMgr, interAGLinks_));
    HCCL_INFO(
        "[InsAllReduceParallelExecutorV2] intraRSLinks_ size[%zu], interRSLinks_ size[%zu], intraAGLinks_ size[%zu], "
        "interAGLinks_ size[%zu]",
        intraRSLinks_.size(), interRSLinks_.size(), intraAGLinks_.size(), interAGLinks_.size());
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GenRSIntraParams0(const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const
{
    u64 sliceCount = dataCount / rankSize_;
    u64 sliceBytes = sliceCount * dataTypeSize_;
    u64 tailSize = dataCount * dataTypeSize_ - sliceBytes * (rankSize_ - 1);
    SetTemplateDataParams(
        params, BufferType::INPUT, BufferType::OUTPUT, dataOffset, dataOffset + rankIdxLevel0_ * sliceBytes, scratchOff,
        sliceBytes, sliceBytes, sliceBytes, rankSizeLevel1_, sliceBytes * rankSizeLevel0_, sliceBytes * rankSizeLevel0_,
        tailSize);
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GenRSInterParams0(const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const
{
    u64 sliceCount = dataCount / rankSize_;
    u64 sliceBytes = sliceCount * dataTypeSize_;
    u64 tailSize = dataCount * dataTypeSize_ - sliceBytes * (rankSize_ - 1);
    SetTemplateDataParams(
        params, BufferType::OUTPUT, BufferType::OUTPUT, dataOffset + rankIdxLevel0_ * sliceBytes,
        dataOffset + myRank_ * sliceBytes, scratchOff, sliceBytes, sliceBytes * rankSizeLevel0_,
        sliceBytes * rankSizeLevel0_, 1, 0, 0, tailSize);
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GenAGInterParams0(const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const
{
    u64 sliceCount = dataCount / rankSize_;
    u64 sliceBytes = sliceCount * dataTypeSize_;
    u64 tailSize = dataCount * dataTypeSize_ - sliceBytes * (rankSize_ - 1);
    SetTemplateDataParams(
        params, BufferType::OUTPUT, BufferType::OUTPUT, dataOffset + rankIdxLevel0_ * sliceBytes,
        dataOffset + rankIdxLevel0_ * sliceBytes, scratchOff, sliceBytes, sliceBytes * rankSizeLevel0_,
        sliceBytes * rankSizeLevel0_, 1, 0, 0, tailSize);
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GenAGIntraParams0(const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const
{
    u64 sliceCount = dataCount / rankSize_;
    u64 sliceBytes = sliceCount * dataTypeSize_;
    u64 tailSize = dataCount * dataTypeSize_ - sliceBytes * (rankSize_ - 1);
    SetTemplateDataParams(
        params, BufferType::OUTPUT, BufferType::OUTPUT, dataOffset, dataOffset, scratchOff, sliceBytes, sliceBytes,
        sliceBytes, rankSizeLevel1_, rankSizeLevel0_ * sliceBytes, rankSizeLevel0_ * sliceBytes, tailSize);
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GenRSInterParams1(const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const
{
    u64 sliceCount = dataCount / rankSize_ * rankSizeLevel0_;
    u64 sliceBytes = sliceCount * dataTypeSize_;
    u64 tailSize = dataCount * dataTypeSize_ - sliceBytes * (rankSizeLevel1_ - 1);
    SetTemplateDataParams(
        params, BufferType::INPUT, BufferType::OUTPUT, dataOffset, dataOffset + rankIdxLevel1_ * sliceBytes, scratchOff,
        sliceBytes, sliceBytes, sliceBytes, 1, 0, 0, tailSize);
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GenRSIntraParams1(const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const
{
    HCCL_INFO("InsAllReduceParallelExecutorV2 GenRSIntraParams1 start");
    u64 dataCountTmp = CalcDataCountTmp1(dataCount);
    u64 sliceBytes = sliceCount_ * dataTypeSize_;
    SetTemplateDataParams(
        params, BufferType::OUTPUT, BufferType::OUTPUT, dataOffset + rankIdxLevel1_ * sliceBytes * rankSizeLevel0_,
        dataOffset + sliceBytes * myRank_, scratchOff, sliceBytes, sliceBytes, sliceBytes, 1, 0, 0,
        (dataCountTmp - sliceCount_ * (rankSizeLevel0_ - 1)) * dataTypeSize_);
    HCCL_INFO("InsAllReduceParallelExecutorV2 GenRSIntraParams1 end");
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GenAGIntraParams1(const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const
{
    HCCL_INFO("InsAllReduceParallelExecutorV2 GenAGIntraParams1 start");
    u64 dataCountTmp = CalcDataCountTmp1(dataCount);
    u64 sliceBytes = sliceCount_ * dataTypeSize_;
    SetTemplateDataParams(
        params, BufferType::OUTPUT, BufferType::OUTPUT, dataOffset + rankIdxLevel1_ * sliceBytes * rankSizeLevel0_,
        dataOffset + rankIdxLevel1_ * sliceBytes * rankSizeLevel0_,
        scratchOff + rankIdxLevel1_ * rankSizeLevel0_ * sliceBytes, sliceBytes, sliceBytes, sliceBytes, 1, 0, 0,
        (dataCountTmp - sliceCount_ * (rankSizeLevel0_ - 1)) * dataTypeSize_);
    HCCL_INFO("InsAllReduceParallelExecutorV2 GenAGIntraParams1 end");
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
void InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GenAGInterParams1(const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const
{
    u64 sliceCount = dataCount / rankSize_ * rankSizeLevel0_;
    u64 sliceBytes = sliceCount * dataTypeSize_;
    u64 tailSize = dataCount * dataTypeSize_ - sliceBytes * (rankSizeLevel1_ - 1);
    SetTemplateDataParams(
        params, BufferType::OUTPUT, BufferType::OUTPUT, dataOffset, dataOffset, scratchOff, sliceBytes, sliceBytes,
        sliceBytes, 1, 0, 0, tailSize);
    HCCL_INFO("InsAllReduceParallelExecutorV2 GenAGInterParams1 end");
}

/*
 *@Desc: HOST算法编排
 */
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    Orchestrate(const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsAllReduceParallelExecutorV2] Host Orchestrate begins.");

    // init and check params
    CHK_RET(Init(op, params, insQue));

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    CHK_RET(CalcLocalRankSize());

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntraRS(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 tempAlgInterRS(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);
    InsAlgTemplate2 tempAlgIntraAG(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate3 tempAlgInterAG(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);

    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsAllReduceParallelExecutorV2] Orchestrate SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    tempAlgIntraRS.setPathNumMap(rank2PathNumMap[0]);
    tempAlgInterRS.setPathNumMap(rank2PathNumMap[1]);
    tempAlgIntraAG.setPathNumMap(rank2PathNumMap[0]);
    tempAlgInterAG.setPathNumMap(rank2PathNumMap[1]);

    InitAlgCommonParams(tempAlgIntraRS, tempAlgInterRS, tempAlgIntraAG, tempAlgInterAG, op);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(rankGraph, tempAlgIntraRS, tempAlgInterRS, tempAlgIntraAG, tempAlgInterAG));
    CHK_RET(GenInsQues(tempAlgIntraRS, tempAlgInterRS, tempAlgIntraAG, tempAlgInterAG));
    HCCL_INFO("[InsAllReduceParallelExecutorV2] Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

/*
 *@Desc: AICPU算法编排
 */
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue)
{
    HCCL_INFO("[InsAllReduceParallelExecutorV2] AICPU Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));
    // 获取当前通信域的信息
    vTopo_ = topoInfo.vTopo;
    virtRankMap_ = topoInfo.virtRankMap;
    virtRanks_ = topoInfo.virtRanks;
    CHK_RET(CalcLocalRankSize());

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntraRS(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 tempAlgInterRS(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);
    InsAlgTemplate2 tempAlgIntraAG(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate3 tempAlgInterAG(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);

    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsAllReduceParallelExecutorV2] Orchestrate SetPathNumMap");
    CHK_RET(SetPathNumMapByLinkMgrMultiLevel(linkMgr, virtRanks_, myRank_, rank2PathNumMap));
    tempAlgIntraRS.setPathNumMap(rank2PathNumMap[0]);
    tempAlgInterRS.setPathNumMap(rank2PathNumMap[1]);
    tempAlgIntraAG.setPathNumMap(rank2PathNumMap[0]);
    tempAlgInterAG.setPathNumMap(rank2PathNumMap[1]);

    InitAlgCommonParams(tempAlgIntraRS, tempAlgInterRS, tempAlgIntraAG, tempAlgInterAG, op);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(linkMgr, tempAlgIntraRS, tempAlgInterRS, tempAlgIntraAG, tempAlgInterAG));
    CHK_RET(GenInsQues(tempAlgIntraRS, tempAlgInterRS, tempAlgIntraAG, tempAlgInterAG));
    HCCL_INFO("[InsAllReduceParallelExecutorV2] Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsAllReduceParallelExecutorV2<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GenInsQues(
        InsAlgTemplate0& tempAlgIntraRS, InsAlgTemplate1& tempAlgInterRS, InsAlgTemplate2& tempAlgIntraAG,
        InsAlgTemplate3& tempAlgInterAG)
{
    std::vector<float> dataSplitSize;
    GetParallelDataSplitRate(dataSplitSize);
    u64 alignedSize = 16 * 1024; // 16K 对齐
    u64 UB_DATA_SIZE_LIMIT = static_cast<u64>(UB_MAX_DATA_SIZE) * rankSize_ / rankSizeLevel0_;

    u64 dataCount0 = (static_cast<u64>((dataCount_ * dataSplitSize[0])) / rankSize_) * rankSize_;
    u64 dataCount1 = dataCount_ - dataCount0;

    u64 scratchSize0 = (static_cast<u64>(maxTmpMemSize_ * dataSplitSize[0]) / alignedSize / dataTypeSize_) * alignedSize
                       * dataTypeSize_;
    scratchSize0 = std::min(scratchSize0, UB_DATA_SIZE_LIMIT);
    u64 maxCountPerLoop0 = scratchSize0 / dataTypeSize_ / rankSize_ * rankSize_;
    maxCountPerLoop0 = std::min(dataCount0, maxCountPerLoop0);

    u64 scratchSize1 = ((maxTmpMemSize_ - scratchSize0) / alignedSize / dataTypeSize_) * alignedSize * dataTypeSize_;
    scratchSize1 = std::min(scratchSize1, UB_DATA_SIZE_LIMIT);
    u64 maxCountPerLoop1 = scratchSize1 / dataTypeSize_ / rankSize_ * rankSize_;
    maxCountPerLoop1 = std::min(dataCount1, maxCountPerLoop1);

    u32 loopTimes0 = 0;
    if (maxCountPerLoop0 != 0) {
        loopTimes0 = dataCount0 / maxCountPerLoop0 + ((dataCount0 % maxCountPerLoop0 == 0) ? 0 : 1);
    }

    u32 loopTimes1 = 0;
    if (maxCountPerLoop1 != 0) {
        loopTimes1 = dataCount1 / maxCountPerLoop1 + ((dataCount1 % maxCountPerLoop1 == 0) ? 0 : 1);
    }
    u32 loopTimes = std::max(loopTimes0, loopTimes1);

    u64 scratchOffset0 = 0;
    u64 scratchOffset1 = scratchSize0;

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = false;

    TemplateDataParams rsIntra0Params, rsInter1Params;
    TemplateDataParams rsInter0Params, rsIntra1Params;
    TemplateDataParams agIntra0Params, agInter1Params;
    TemplateDataParams agInter0Params, agIntra1Params;

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount0 = 0;
        u64 dataOffset0 = 0;
        if (loopIndex < loopTimes) {
            currCount0 = (loopIndex == loopTimes - 1) ? (dataCount0 - loopIndex * maxCountPerLoop0) : maxCountPerLoop0;
            dataOffset0 = loopIndex * maxCountPerLoop0 * dataTypeSize_;
        }

        u64 currCount1 = 0;
        u64 dataOffset1 = 0;
        if (loopIndex < loopTimes) {
            currCount1 = (loopIndex == loopTimes - 1) ? (dataCount1 - loopIndex * maxCountPerLoop1) : maxCountPerLoop1;
            dataOffset1 = loopIndex * maxCountPerLoop1 * dataTypeSize_ + dataCount0 * dataTypeSize_;
        }

        // 计算统一sliceCount数量
        u64 totalSliceCount = currCount1 / rankSize_ * rankSizeLevel0_;
        sliceCount_ = (currCount1 >= rankSize_) ? totalSliceCount / rankSizeLevel0_ :
                                                  totalSliceCount / rankSize_ * rankSizeLevel1_;

        // ────────────── Phase 1: RS-1 ──────────────
        // 前半: 框内 Mesh RS,  后半: 框间 NHR RS
        CHK_RET(PreSyncQues(syncQueues_, 0));

        GenRSIntraParams0(dataOffset0, currCount0, scratchOffset0, rsIntra0Params);
        CHK_RET(tempAlgIntraRS.GenExtIns(tempFuncs, rsIntra0Params, intraRSLinks_, intraQue_));

        GenRSInterParams1(dataOffset1, currCount1, scratchOffset1, rsInter1Params);
        CHK_RET(tempAlgInterRS.GenExtIns(tempFuncs, rsInter1Params, interRSLinks_, interQue_));

        CHK_RET(PostSyncQues(syncQueues_, 0));

        // ────────────── Phase 2: RS-2 ──────────────
        // 前半: 框间 NHR RS,  后半: 框内 Mesh RS
        CHK_RET(PreSyncQues(syncQueues_, 0));

        GenRSInterParams0(dataOffset0, currCount0, scratchOffset0, rsInter0Params);
        CHK_RET(tempAlgInterRS.GenExtIns(tempFuncs, rsInter0Params, interRSLinks_, interQue_));

        GenRSIntraParams1(dataOffset1, currCount1, scratchOffset1, rsIntra1Params);
        CHK_RET(tempAlgIntraRS.GenExtIns(tempFuncs, rsIntra1Params, intraRSLinks_, intraQue_));

        CHK_RET(PostSyncQues(syncQueues_, 0));

        // ────────────── Phase 3: AG-1 ──────────────
        // 前半: 框间 NHR AG,  后半: 框内 Mesh AG
        CHK_RET(PreSyncQues(syncQueues_, 0));

        GenAGInterParams0(dataOffset0, currCount0, scratchOffset0, agInter0Params);
        CHK_RET(tempAlgInterAG.GenExtIns(tempFuncs, agInter0Params, interAGLinks_, interQue_));

        GenAGIntraParams1(dataOffset1, currCount1, scratchOffset1, agIntra1Params);
        CHK_RET(tempAlgIntraAG.GenExtIns(tempFuncs, agIntra1Params, intraAGLinks_, intraQue_));

        CHK_RET(PostSyncQues(syncQueues_, 0));

        // ────────────── Phase 4: AG-2 ──────────────
        // 前半: 框内 Mesh AG,  后半: 框间 NHR AG
        CHK_RET(PreSyncQues(syncQueues_, 0));

        GenAGIntraParams0(dataOffset0, currCount0, scratchOffset0, agIntra0Params);
        CHK_RET(tempAlgIntraAG.GenExtIns(tempFuncs, agIntra0Params, intraAGLinks_, intraQue_));

        GenAGInterParams1(dataOffset1, currCount1, scratchOffset1, agInter1Params);
        CHK_RET(tempAlgInterAG.GenExtIns(tempFuncs, agInter1Params, interAGLinks_, interQue_));

        CHK_RET(PostSyncQues(syncQueues_, 0));
    }

    return HcclResult::HCCL_SUCCESS;
}

// 算法注册
INS_REGISTER_IMPL_BY_FOUR_TEMPS(
    OpType::ALLREDUCE, InsAllReduceFourTemplateMesh1DNHR, InsAllReduceParallelExecutorV2, TopoMatchMeshNHR,
    InsTempReduceScatterMesh1D, InsTempReduceScatterMesh1D, InsTempAllGatherMesh1D, InsTempAllGatherMesh1D);
} // namespace Hccl
