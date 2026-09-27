/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_all_gather_parallel_executor.h"

#include <cmath>

#include "log.h"

#include "ins_coll_alg_registry.h"

#include "topo_match_mesh_nhr.h"
#include "topo_match_concurr_mesh_nhr.h"
#include "topo_match_mesh_nhr_pcie.h"

#include "alg_data_trans_wrapper.h"

#include "ins_temp_all_gather_mesh.h"
#include "ins_temp_all_gather_mesh_2D.h"
#include "ins_temp_all_gather_nhr.h"

#ifndef CCL_KERNEL_AICPU
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem_with_stride.h"
#endif

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsAllGatherParallelExecutor()
    : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsAllGatherParallelExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;

    u64 scratchMemSize = 200 * 1024 * 1024;
    resReq.requiredScratchMemSize = scratchMemSize; // 200MB

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    rankSizeLevel0_ = virtRanks_[0].size();
    rankSizeLevel1_ = virtRanks_[1].size();
    InsAlgTemplate0 intraTempAlg(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlg(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);
    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsAllGatherParallelExecutor] CalcResOffload SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    intraTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interTempAlg.setPathNumMap(rank2PathNumMap[1]);

    // calculate required insQues and prepare queue
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], CalcResOffload with detouring enabled.", __func__, myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG(
            "[%s] Rank[%d], CalcResOffload with detouring disabled. rankSizeLevel0[%u] rankSizeLevel1[%u]", __func__,
            myRank_, rankSizeLevel0_, rankSizeLevel1_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }

    CHK_RET(interTempAlg.CalcRes(resReqInter));

    resReq.requiredSubQueNum = resReqIntra.streamNum + resReqInter.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    algResReq.topoInfo.UpdateMultiLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    rankSizeLevel0_ = virtRanks_[0].size();
    rankSizeLevel1_ = virtRanks_[1].size();

    // instantiate a template
    InsAlgTemplate0 intraTempAlg(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlg(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]);
    std::vector<map<u32, u32>> rank2PathNumMap;
    HCCL_INFO("[InsAllGatherParallelExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    intraTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interTempAlg.setPathNumMap(rank2PathNumMap[1]);

    // calculate required insQues and prepare queue
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsAllGatherParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG(
            "[InsAllGatherParallelExecutor] Rank[%d], CalcRes with detouring disabled."
            "rankSizeLevel0[%u] rankSizeLevel1[%u]",
            myRank_, rankSizeLevel0_, rankSizeLevel1_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }
    CHK_RET(interTempAlg.CalcRes(resReqInter));

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqIntra.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqInter.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = resReqIntra.streamNum + resReqInter.streamNum;
    CHK_RET(CalcParallelNotifyReq(algResReq.primQueueNum, resReqIntra.queNum, algResReq.queueNotifys));
    HCCL_DEBUG("[InsAllGatherParallelExecutor] algResReq.primQueueNum %u", algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口，将对应的 instruction 添加到指令队列中
// 传入的insQue为一条主流
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra0(
    const u64 dataOffset, const u64 dataCountPerLoopAixs0, const u64 scratchOffset,
    TemplateDataParams& tempAlgParamsIntra0) const
{
    tempAlgParamsIntra0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsIntra0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra0.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsIntra0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra0.buffInfo.outBuffBaseOff = rankIdxLevel1_ * rankSizeLevel0_ * dataSize_ + dataOffset;
    tempAlgParamsIntra0.buffInfo.scratchBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.tailSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.inputSliceStride = 0;
    tempAlgParamsIntra0.outputSliceStride = dataSize_;
    tempAlgParamsIntra0.repeatNum = 1;
    tempAlgParamsIntra0.inputRepeatStride = 0;
    tempAlgParamsIntra0.outputRepeatStride = 0;

    HCCL_DEBUG(
        "[InsAllGatherParallelExecutor][GenTemplateAlgParamsIntra0] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] rankSizeLevel0[%u] "
        "rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        myRank_, tempAlgParamsIntra0.buffInfo.inBuffBaseOff, tempAlgParamsIntra0.buffInfo.outBuffBaseOff,
        tempAlgParamsIntra0.buffInfo.scratchBuffBaseOff, tempAlgParamsIntra0.sliceSize,
        tempAlgParamsIntra0.outputSliceStride, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter0(
    const u64 dataOffset, const u64 dataCountPerLoopAixs0, const u64 scratchOffset,
    TemplateDataParams& tempAlgParamsInter0) const
{
    tempAlgParamsInter0.buffInfo.inBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsInter0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.scratchBuffBaseOff = scratchOffset;
    tempAlgParamsInter0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.tailSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.inputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter0.outputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter0.repeatNum = rankSizeLevel0_;
    tempAlgParamsInter0.inputRepeatStride = dataSize_;
    tempAlgParamsInter0.outputRepeatStride = dataSize_;
    HCCL_DEBUG(
        "[InsAllGatherParallelExecutor][GenTemplateAlgParamsInter0] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] "
        "outputRepeatStride[%llu]",
        myRank_, tempAlgParamsInter0.buffInfo.inBuffBaseOff, tempAlgParamsInter0.buffInfo.outBuffBaseOff,
        tempAlgParamsInter0.buffInfo.scratchBuffBaseOff, tempAlgParamsInter0.sliceSize,
        tempAlgParamsInter0.outputSliceStride, tempAlgParamsInter0.outputRepeatStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(
    const u64 dataOffset, const u64 dataCountPerLoopAixs1, const u64 scratchOffset,
    TemplateDataParams& tempAlgParamsInter1) const
{
    tempAlgParamsInter1.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsInter1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter1.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsInter1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter1.buffInfo.outBuffBaseOff = rankIdxLevel0_ * dataSize_ + dataOffset; // for example 0 2 4 | 1 3 5
    tempAlgParamsInter1.buffInfo.scratchBuffBaseOff = scratchOffset;
    tempAlgParamsInter1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsInter1.tailSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsInter1.inputSliceStride = 0;
    tempAlgParamsInter1.outputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter1.repeatNum = 1;
    tempAlgParamsInter1.inputRepeatStride = 0;
    tempAlgParamsInter1.outputRepeatStride = 0;
    HCCL_DEBUG(
        "[InsAllGatherParallelExecutor][GenTemplateAlgParamsInter1] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu]",
        myRank_, tempAlgParamsInter1.buffInfo.inBuffBaseOff, tempAlgParamsInter1.buffInfo.outBuffBaseOff,
        tempAlgParamsInter1.buffInfo.scratchBuffBaseOff, tempAlgParamsInter1.sliceSize,
        tempAlgParamsInter1.outputSliceStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra1(
    const u64 dataOffset, const u64 dataCountPerLoopAixs1, const u64 scratchOffset,
    TemplateDataParams& tempAlgParamsIntra1) const
{
    tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsIntra1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.scratchBuffBaseOff = scratchOffset;
    tempAlgParamsIntra1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsIntra1.tailSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsIntra1.inputSliceStride = dataSize_;
    tempAlgParamsIntra1.outputSliceStride = dataSize_;
    tempAlgParamsIntra1.repeatNum = rankSizeLevel1_;
    tempAlgParamsIntra1.inputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsIntra1.outputRepeatStride = dataSize_ * rankSizeLevel0_;
    HCCL_DEBUG(
        "[InsAllGatherParallelExecutor][GenTemplateAlgParamsIntra1] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] outputSliceStride[%llu] "
        "outputRepeatStride[%llu]",
        myRank_, tempAlgParamsIntra1.buffInfo.inBuffBaseOff, tempAlgParamsIntra1.buffInfo.outBuffBaseOff,
        tempAlgParamsIntra1.buffInfo.scratchBuffBaseOff, tempAlgParamsIntra1.sliceSize,
        tempAlgParamsIntra1.outputSliceStride, tempAlgParamsIntra1.outputRepeatStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float>& splitDataSize) const
{
    // to do 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const RankGraph* rankGraph, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqInter;
    AlgTempResReq resReqIntra;
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detour enabled.", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(rankGraph, resReqIntra));
    } else {
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));

    // 申请算法模板所需资源
    if (!(resReqIntra.queNum > 0 && resReqInter.queNum > 0)) {
        HCCL_ERROR("Intra queNum and Inter queNum must be larger than 0.");
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

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, interLinks_));
    HCCL_INFO(
        "[InsAllGatherParallelExecutor] intraLinks_ size[%zu], interLinks_ size[%zu]", intraLinks_.size(),
        interLinks_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    ConnectedLinkMgr* linkMgr, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], detouring enabled.", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(linkMgr, resReqIntra));
    } else {
        HCCL_DEBUG("[%s] Rank[%d], detouring disabled.", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));

    // 申请算法模板所需资源
    if (!(resReqIntra.queNum > 0 && resReqInter.queNum > 0)) {
        HCCL_ERROR("[InsAllGatherParallelExecutor]resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    u32 totalQueueNum = resReqIntra.queNum + resReqInter.queNum;
    CHK_RET(InitQueue(totalQueueNum, requiredQue_));
    for (u32 q = 0; q < requiredQue_.size(); q++) {
        if (q < resReqIntra.queNum) {
            intraQue_.push_back(requiredQue_[q]);
        } else {
            interQue_.push_back(requiredQue_[q]);
        }
    }
    syncQueues_.emplace_back(intraQue_[0]);
    syncQueues_.emplace_back(interQue_[0]);

    CHK_RET(PrepResLinks(myRank_, resReqIntra.links, linkMgr, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, resReqInter.links, linkMgr, interLinks_));
    HCCL_INFO(
        "[InsAllGatherParallelExecutor] intraLinks_ size[%zu], interLinks_ size[%zu]", intraLinks_.size(),
        interLinks_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsAllGatherParallelExecutor] Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));
    dataType_ = op.dataType;
    virtRanks_ = topoInfo.virtRanks;
    vTopo_ = topoInfo.vTopo;
    virtRankMap_ = topoInfo.virtRankMap;

    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));
    rankIdxLevel0_ = myRank_ % virtRanks_[0].size();
    rankIdxLevel1_ = myRank_ / virtRanks_[0].size();

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]); // server内算法，比如mesh
    InsAlgTemplate1 tempAlgInter(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]); // server间算法，比如nhr

    // 实例化算法模板类

    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.SetCollOp(op); // CCU template需要传递op信息
    tempAlgIntra.SetDataType(dataType_);

    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.SetCollOp(op); // CCU template需要传递op信息
    tempAlgInter.SetDataType(dataType_);

    std::vector<std::map<u32, u32>> rank2PathNumMap;
    CHK_RET(SetPathNumMapByLinkMgrMultiLevel(linkMgr, virtRanks_, myRank_, rank2PathNumMap));
    tempAlgIntra.setPathNumMap(rank2PathNumMap[0]);
    tempAlgInter.setPathNumMap(rank2PathNumMap[1]);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(linkMgr, tempAlgIntra, tempAlgInter));

    CHK_RET(GenInsQuesHost(tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsAllGatherParallelExecutor] Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsAllGatherParallelExecutor] Host Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));
    rankIdxLevel0_ = myRank_ % virtRanks_[0].size();
    rankIdxLevel1_ = myRank_ / virtRanks_[0].size();
    HCCL_DEBUG(
        "[InsAllGatherParallelExecutor] my rank is [%d] ranksize is [%u], rankIdxLevel0_ = [%u], rankIdxLevel1_ "
        "= [%u] .",
        myRank_, rankSize_, rankIdxLevel0_, rankIdxLevel1_);
    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]); // server内算法，比如mesh
    InsAlgTemplate1 tempAlgInter(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]); // server间算法，比如nhr

    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.SetDataType(dataType_);
    tempAlgIntra.SetCollOp(op); // CCU template需要传递op信息

    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.SetDataType(dataType_);
    tempAlgInter.SetCollOp(op); // CCU template需要传递op信息

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(rankGraph, tempAlgIntra, tempAlgInter));

    CHK_RET(GenInsQuesHost(tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsAllGatherParallelExecutor]Host Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenInsQuesHost(
    InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    HCCL_INFO("[InsAllGatherParallelExecutor] AlgTemplate inter server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsAllGatherParallelExecutor] AlgTemplate intra server is [%s]", tempAlgInter.Describe().c_str());
    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);
    u64 alignedSize = 16 * 1024; // 假设需要16K对齐
    BufferType inBuffType = BufferType::INPUT;
    BufferType outBuffType = BufferType::OUTPUT;
    u32 intraScatchteMultipleStage0 = tempAlgIntra.CalcScratchMultiple(inBuffType, outBuffType);
    u32 interScatchteMultipleStage0 = tempAlgInter.CalcScratchMultiple(inBuffType, outBuffType);
    u32 intraScatchteMultipleStage1 = tempAlgIntra.CalcScratchMultiple(outBuffType, outBuffType);
    u32 interScatchteMultipleStage1 = tempAlgInter.CalcScratchMultiple(outBuffType, outBuffType);
    u32 scratchMultipleIntra = static_cast<u32>(std::max(
        std::ceil(dataSplitSize[0] * intraScatchteMultipleStage0),
        std::ceil(dataSplitSize[1] * intraScatchteMultipleStage1 * rankSizeLevel1_)));
    u32 scratchMultipleInter = static_cast<u32>(std::max(
        std::ceil(dataSplitSize[1] * interScatchteMultipleStage0),
        std::ceil(dataSplitSize[0] * interScatchteMultipleStage1 * rankSizeLevel0_)));
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
        GenTemplateAlgParamsIntra0(dataOffset0, dataCountPerLoopAixs0, intraScratchOffset, tempAlgParamsIntra0);
        // 把每个template需要的queue传进去，比如stars的mesh要传多条queue
        CHK_RET(tempAlgIntra.GenExtIns(
            tempFuncs, tempAlgParamsIntra0, intraLinks_, intraQue_)); // Todo: 这里要把tempFuncs去掉
        // 数据1的server间的nhr算法
        GenTemplateAlgParamsInter1(dataOffset1, dataCountPerLoopAixs1, interScratchOffset, tempAlgParamsInter1);
        CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter1, interLinks_, interQue_));
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncQues(syncQueues_, 0));

        // 第二步开始前同步
        CHK_RET(PreSyncQues(syncQueues_, 0));
        // 数据0的server间的nhr算法
        GenTemplateAlgParamsInter0(dataOffset0, dataCountPerLoopAixs0, interScratchOffset, tempAlgParamsInter0);
        CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter0, interLinks_, interQue_));
        // 数据1的server内的mesh算法
        GenTemplateAlgParamsIntra1(dataOffset1, dataCountPerLoopAixs1, intraScratchOffset, tempAlgParamsIntra1);
        CHK_RET(tempAlgIntra.GenExtIns(tempFuncs, tempAlgParamsIntra1, intraLinks_, intraQue_));
        // 尾同步
        CHK_RET(PostSyncQues(syncQueues_, 0));
    }
    return HcclResult::HCCL_SUCCESS;
}

INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLGATHER, InsAllGatherParallelMesh1DNHR, InsAllGatherParallelExecutor, TopoMatchMeshNHR,
    InsTempAllGatherMesh1D, InsTempAllGatherNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLGATHER, InsAllGatherParallelMesh2DNHR, InsAllGatherParallelExecutor, TopoMatchConcurrMeshNHR,
    InsTempAllGatherMesh2D, InsTempAllGatherNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLGATHER, InsAllGatherParallelNHRNHR, InsAllGatherParallelExecutor, TopoMatchMeshNHR,
    InsTempAllGatherMesh2D, InsTempAllGatherNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLGATHER, InsAllGatherParallelMesh1DNHRPcie, InsAllGatherParallelExecutor, TopoMatchMeshNHRPcie,
    InsTempAllGatherMesh1D, InsTempAllGatherNHR);

// 算法注册
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::ALLGATHER, CcuAllGatherParallelMesh1DNHR, InsAllGatherParallelExecutor, TopoMatchMeshNHR,
    CcuTempAllGatherMesh1DMem2MemWithStride, CcuTempAllGatherNHRMem2Mem1D);
#endif

} // namespace Hccl
