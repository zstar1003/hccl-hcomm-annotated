/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_reduce_scatter_parallel_executor.h"

#include <cmath>

#include "log.h"

#include "ins_coll_alg_registry.h"

#include "topo_match_mesh_nhr.h"
#include "topo_match_concurr_mesh_nhr.h"
#include "topo_match_mesh_nhr_pcie.h"
#include "alg_data_trans_wrapper.h"
#include "ins_temp_reduce_scatter_mesh_1D.h"
#include "ins_temp_reduce_scatter_mesh_2D.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_nhr_1D_mem2mem.h"

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsReduceScatterParallelExecutor()
    : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsReduceScatterParallelExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcResOffload(
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
    HCCL_INFO("[InsReduceScatterParallelExecutor] CalcResOffload SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    intraTempAlg.setPathNumMap(rank2PathNumMap[0]);
    interTempAlg.setPathNumMap(rank2PathNumMap[1]);

    // calculate required insQues and prepare queue
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("InsReduceScatterParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[InsReduceScatterParallelExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }

    CHK_RET(interTempAlg.CalcRes(resReqInter));
    resReq.requiredSubQueNum = resReqIntra.streamNum + resReqInter.streamNum - 1;
    HCCL_DEBUG(
        "CalResOffload resReqIntra.streamNum [%u], resReqInter.streamNum [%u]", resReqIntra.streamNum,
        resReqInter.streamNum);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
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
    HCCL_INFO("[InsReduceScatterParallelExecutor] CalcRes SetPathNumMap");
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

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqIntra.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqInter.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = resReqIntra.streamNum + resReqInter.streamNum;
    CHK_RET(CalcParallelNotifyReq(algResReq.primQueueNum, resReqIntra.queNum, algResReq.queueNotifys));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, algResReq.links));
    HCCL_DEBUG(
        "CalRes resReqIntra.streamNum [%u], resReqInter.streamNum [%u]", resReqIntra.streamNum, resReqInter.streamNum);
    return HcclResult::HCCL_SUCCESS;
}

// HOST 侧算法入口，将对应的 instruction 添加到指令队列中
// 传入的insQue为一条主流
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra0(
    const u64 dataOffset, const u64 dataCountPerLoopAixs0, std::vector<u64>& scratchOffVec,
    TemplateDataParams& tempAlgParamsIntra0) const
{
    tempAlgParamsIntra0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsIntra0.buffInfo.outBuffType = BufferType::SCRATCH; // 第一步最后的数据存储在scratch buffer上
    tempAlgParamsIntra0.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsIntra0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra0.buffInfo.outBuffBaseOff
        = scratchOffVec[0] + rankIdxLevel0_ * dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.buffInfo.scratchBuffBaseOff = scratchOffVec[0];
    tempAlgParamsIntra0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;

    tempAlgParamsIntra0.inputSliceStride = dataSize_;
    tempAlgParamsIntra0.outputSliceStride = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.repeatNum = rankSizeLevel1_;
    tempAlgParamsIntra0.inputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsIntra0.outputRepeatStride = dataCountPerLoopAixs0 * dataTypeSize_ * rankSizeLevel0_;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter0(
    const u64 dataOffset, const u64 dataCountPerLoopAixs0, std::vector<u64>& scratchOffVec,
    TemplateDataParams& tempAlgParamsInter0) const
{
    tempAlgParamsInter0.buffInfo.inBuffType = BufferType::SCRATCH;
    tempAlgParamsInter0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsInter0.buffInfo.inBuffBaseOff
        = scratchOffVec[0] + rankIdxLevel0_ * dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.scratchBuffBaseOff = scratchOffVec[2];
    tempAlgParamsInter0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;

    tempAlgParamsInter0.inputSliceStride = dataCountPerLoopAixs0 * dataTypeSize_ * rankSizeLevel0_;
    tempAlgParamsInter0.outputSliceStride = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.repeatNum = 1;
    tempAlgParamsInter0.inputRepeatStride = 0;
    tempAlgParamsInter0.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(
    const u64 dataOffset, const u64 dataCountPerLoopAixs1, std::vector<u64>& scratchOffVec,
    TemplateDataParams& tempAlgParamsInter1) const
{
    tempAlgParamsInter1.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsInter1.buffInfo.outBuffType = BufferType::SCRATCH;
    tempAlgParamsInter1.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsInter1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter1.buffInfo.outBuffBaseOff = scratchOffVec[3];
    tempAlgParamsInter1.buffInfo.scratchBuffBaseOff = scratchOffVec[3];
    tempAlgParamsInter1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;

    tempAlgParamsInter1.inputSliceStride = dataSize_;
    tempAlgParamsInter1.outputSliceStride = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsInter1.repeatNum = rankSizeLevel0_;
    tempAlgParamsInter1.inputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter1.outputRepeatStride = dataCountPerLoopAixs1 * dataTypeSize_ * rankSizeLevel1_;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra1(
    const u64 dataOffset, const u64 dataCountPerLoopAixs1, std::vector<u64>& scratchOffVec,
    TemplateDataParams& tempAlgParamsIntra1) const
{
    tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::SCRATCH;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.scratBuffType = BufferType::SCRATCH;
    tempAlgParamsIntra1.buffInfo.inBuffBaseOff = scratchOffVec[3];
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.scratchBuffBaseOff = scratchOffVec[1];
    tempAlgParamsIntra1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;

    tempAlgParamsIntra1.inputSliceStride = dataCountPerLoopAixs1 * dataTypeSize_ * rankSizeLevel1_;
    tempAlgParamsIntra1.outputSliceStride = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsIntra1.repeatNum = 1;
    tempAlgParamsIntra1.inputRepeatStride = 0;
    tempAlgParamsIntra1.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float>& splitDataSize) const
{
    // to do 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const RankGraph* rankGraph, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsReduceScatterParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[InsReduceScatterParallelExecutor] Rank[%d], CalcRes with detour disabled.", myRank_);
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));

    // 申请算法模板所需资源
    if (!(resReqIntra.queNum > 0 && resReqInter.queNum > 0)) {
        HCCL_ERROR(
            "[InsReduceScatterParallelExecutor]resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    u32 totalQueueNum = resReqIntra.queNum + resReqInter.queNum;
    CHK_RET(InitQueue(totalQueueNum, requireQue_));
    for (u32 i = 0; i < requireQue_.size(); i++) {
        if (i < resReqIntra.queNum) {
            intraQue_.push_back(requireQue_[i]);
        } else {
            interQue_.push_back(requireQue_[i]);
        }
    }
    syncQueues_.emplace_back(intraQue_[0]);
    syncQueues_.emplace_back(interQue_[0]);

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, interLinks_));
    HCCL_INFO(
        "[InsReduceScatterParallelExecutor] intraLinks_ size[%zu], interLinks_ size[%zu]", intraLinks_.size(),
        interLinks_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    ConnectedLinkMgr* linkMgr, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqInter;
    AlgTempResReq resReqIntra;
    if (enableDetour_) {
        HCCL_DEBUG("[InsReduceScatterParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(linkMgr, resReqIntra));
    } else {
        HCCL_DEBUG("[InsReduceScatterParallelExecutor] Rank[%d], CalcRes with detour disabled.", myRank_);
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));

    // 申请算法模板所需资源
    if (!(resReqIntra.queNum > 0 && resReqInter.queNum > 0)) {
        HCCL_ERROR(
            "[InsReduceScatterParallelExecutor]resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    u32 totalQueueNum = resReqIntra.queNum + resReqInter.queNum;
    CHK_RET(InitQueue(totalQueueNum, requireQue_));
    u32 intraQueNum = resReqIntra.queNum;
    for (u32 i = 0; i < requireQue_.size(); i++) {
        if (i < intraQueNum) {
            intraQue_.push_back(requireQue_[i]);
        } else {
            interQue_.push_back(requireQue_[i]);
        }
    }
    syncQueues_.emplace_back(intraQue_[0]);
    syncQueues_.emplace_back(interQue_[0]);

    CHK_RET(PrepResLinks(myRank_, resReqIntra.links, linkMgr, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, resReqInter.links, linkMgr, interLinks_));
    HCCL_INFO(
        "[InsReduceScatterParallelExecutor] intraLinks_ size[%zu], interLinks_ size[%zu]", intraLinks_.size(),
        interLinks_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    // init and check params
    CHK_RET(Init(op, params, insQue));

    virtRanks_ = topoInfo.virtRanks;
    vTopo_ = topoInfo.vTopo;
    virtRankMap_ = topoInfo.virtRankMap;
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));
    if (virtRankMap_[0].find(myRank_) != virtRankMap_[0].end()) {
        rankIdxLevel0_ = virtRankMap_[0][myRank_];
    } else {
        HCCL_ERROR("rank [%d] is not in level 0 topo", myRank_);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (virtRankMap_[1].find(myRank_) != virtRankMap_[1].end()) {
        rankIdxLevel1_ = virtRankMap_[1][myRank_];
    } else {
        HCCL_ERROR("rank [%d] is not in level 1 topo", myRank_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]); // server内算法，比如mesh
    InsAlgTemplate1 tempAlgInter(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]); // server间算法，比如nhr

    // 实例化算法模板类

    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.SetCollOp(op); // CCU template需要传递op信息
    tempAlgIntra.InitReduceInfo(redOp_, dataType_);

    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.SetCollOp(op); // CCU template需要传递op信息
    tempAlgInter.InitReduceInfo(redOp_, dataType_);

    std::vector<std::map<u32, u32>> rank2PathNumMap;
    SetPathNumMapByLinkMgrMultiLevel(linkMgr, virtRanks_, myRank_, rank2PathNumMap);
    tempAlgIntra.setPathNumMap(rank2PathNumMap[0]);
    tempAlgInter.setPathNumMap(rank2PathNumMap[1]);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(linkMgr, tempAlgIntra, tempAlgInter));

    CHK_RET(GenInsQuesHost(tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsReduceScatterParallelExecutor] Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsReduceScatterParallelExecutor] Host Orchestrate begins.");
    // init and check params
    CHK_RET(Init(op, params, insQue));

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, rankSizeLevel0_, rankSizeLevel1_));

    if (virtRankMap_[0].find(myRank_) != virtRankMap_[0].end()) {
        rankIdxLevel0_ = virtRankMap_[0][myRank_];
    } else {
        HCCL_ERROR("rank [%d] is not in level 0 topo", myRank_);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (virtRankMap_[1].find(myRank_) != virtRankMap_[1].end()) {
        rankIdxLevel1_ = virtRankMap_[1][myRank_];
    } else {
        HCCL_ERROR("rank [%d] is not in level 1 topo", myRank_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, rankSizeLevel0_, vTopo_[0], virtRankMap_[0]); // server内算法，比如mesh
    InsAlgTemplate1 tempAlgInter(myRank_, rankSizeLevel1_, vTopo_[1], virtRankMap_[1]); // server间算法，比如nhr

    // 实例化算法模板类

    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.SetCollOp(op); // CCU template需要传递op信息
    tempAlgIntra.InitReduceInfo(redOp_, dataType_);

    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.SetCollOp(op); // CCU template需要传递op信息
    tempAlgInter.InitReduceInfo(redOp_, dataType_);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(rankGraph, tempAlgIntra, tempAlgInter));

    CHK_RET(GenInsQuesHost(tempAlgIntra, tempAlgInter));
    HCCL_INFO("[InsReduceScatterParallelExecutor] Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsReduceScatterParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenInsQuesHost(
    InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    HCCL_INFO("[InsReduceScatterParallelExecutor] AlgTemplate inter server is [%s]", tempAlgInter.Describe().c_str());
    HCCL_INFO("[InsReduceScatterParallelExecutor] AlgTemplate intra server is [%s]", tempAlgIntra.Describe().c_str());
    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);
    u64 alignedSize = 16 * 1024; // 假设需要16K对齐
    BufferType inBuffType = BufferType::INPUT;
    BufferType outBuffType = BufferType::OUTPUT;
    u32 intraScatchteMultipleStage0 = tempAlgIntra.CalcScratchMultiple(inBuffType, outBuffType);
    u32 interScatchteMultipleStage0 = tempAlgInter.CalcScratchMultiple(inBuffType, outBuffType);
    u32 intraScatchteMultipleStage1 = tempAlgIntra.CalcScratchMultiple(outBuffType, outBuffType);
    u32 interScatchteMultipleStage1 = tempAlgInter.CalcScratchMultiple(outBuffType, outBuffType);
    if (interScatchteMultipleStage0 == 0 || interScatchteMultipleStage1 == 0) {
        interScatchteMultipleStage0 = rankSizeLevel1_;
        interScatchteMultipleStage1 = rankSizeLevel1_;
    }
    u32 scratchMultipleIntra0
        = static_cast<u32>(std::ceil(dataSplitSize[0] * intraScatchteMultipleStage0 * rankSizeLevel1_));
    u32 scratchMultipleIntra1 = static_cast<u32>(std::ceil(dataSplitSize[1] * intraScatchteMultipleStage1));
    u32 scratchMultipleInter1
        = static_cast<u32>(std::ceil(dataSplitSize[1] * interScatchteMultipleStage0 * rankSizeLevel0_));
    u32 scratchMultipleInter0 = static_cast<u32>(std::ceil(dataSplitSize[0] * interScatchteMultipleStage1));
    u32 totalScratchMultiple
        = scratchMultipleIntra0 + scratchMultipleIntra1 + scratchMultipleInter0 + scratchMultipleInter1;
    u64 scratchMemBlockSize = maxTmpMemSize_;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / alignedSize / totalScratchMultiple) * alignedSize;
    }
    u64 intra0ScratchOffset = 0;
    u64 intra1ScratchOffset = intra0ScratchOffset + scratchMultipleIntra0 * scratchMemBlockSize;
    u64 inter0ScratchOffset = intra1ScratchOffset + scratchMultipleIntra1 * scratchMemBlockSize;
    u64 inter1ScratchOffset = inter0ScratchOffset + scratchMultipleInter0 * scratchMemBlockSize;
    std::vector<u64> scratchOffVec
        = {intra0ScratchOffset, intra1ScratchOffset, inter0ScratchOffset, inter1ScratchOffset};

    // dataSplitSize为分数，这里maxCountPerLoop对10取整
    u64 maxCountPerLoop
        = (std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_ / 10)
          * 10;
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = false;
    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsIntra1;
    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex == loopTimes - 1) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        u64 dataCountPerLoopAixs0 = static_cast<u64>(dataSplitSize[0] * currCount);
        u64 dataCountPerLoopAixs1 = currCount - dataCountPerLoopAixs0;
        // 第一步开始前同步
        CHK_RET(PreSyncQues(syncQueues_, 0));
        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + dataCountPerLoopAixs0 * dataTypeSize_;
        // 数据0的server内的mesh算法
        GenTemplateAlgParamsIntra0(dataOffset0, dataCountPerLoopAixs0, scratchOffVec, tempAlgParamsIntra0);
        // 把每个template需要的queue传进去，比如stars的mesh要传多条queue
        CHK_RET(tempAlgIntra.GenExtIns(tempFuncs, tempAlgParamsIntra0, intraLinks_, intraQue_));
        // 数据1的server间的nhr算法
        GenTemplateAlgParamsInter1(dataOffset1, dataCountPerLoopAixs1, scratchOffVec, tempAlgParamsInter1);
        CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter1, interLinks_, interQue_));
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncQues(syncQueues_, 0));

        // 第二步开始前同步
        CHK_RET(PreSyncQues(syncQueues_, 0));
        // 数据0的server间的nhr算法
        tempFuncs.isBottom = true;
        GenTemplateAlgParamsInter0(dataOffset0, dataCountPerLoopAixs0, scratchOffVec, tempAlgParamsInter0);
        CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter0, interLinks_, interQue_));
        tempFuncs.isBottom = false;
        // 数据1的server内的mesh算法
        GenTemplateAlgParamsIntra1(dataOffset1, dataCountPerLoopAixs1, scratchOffVec, tempAlgParamsIntra1);
        CHK_RET(tempAlgIntra.GenExtIns(tempFuncs, tempAlgParamsIntra1, intraLinks_, intraQue_));
        // 尾同步
        CHK_RET(PostSyncQues(syncQueues_, 0));
    }
    return HcclResult::HCCL_SUCCESS;
}

// 算法注册
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::REDUCESCATTER, InsReduceScatterParallelMesh1DNHR, InsReduceScatterParallelExecutor, TopoMatchMeshNHR,
    InsTempReduceScatterMesh1D, InsTempReduceScatterNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::REDUCESCATTER, InsReduceScatterParallelMesh2DNHR, InsReduceScatterParallelExecutor, TopoMatchConcurrMeshNHR,
    InsTempReduceScatterMesh2D, InsTempReduceScatterNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::REDUCESCATTER, InsReduceScatterParallelNHRNHR, InsReduceScatterParallelExecutor, TopoMatchConcurrMeshNHR,
    InsTempReduceScatterNHR, InsTempReduceScatterNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::REDUCESCATTER, InsReduceScatterParallelMesh1DNHRPcie, InsReduceScatterParallelExecutor,
    TopoMatchMeshNHRPcie, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR);
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::REDUCESCATTER, CcuReduceScatterParallelMesh1DNHR, InsReduceScatterParallelExecutor, TopoMatchMeshNHR,
    CcuTempReduceScatterMeshMem2Mem1D, CcuTempReduceScatterNHR1DMem2Mem);
#endif
} // namespace Hccl
