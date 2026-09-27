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
#include "topo_match_mesh_nhr.h"
#include "topo_match_mesh_nhr_pcie.h"
#include "alg_data_trans_wrapper.h"

#include "ins_temp_broadcast_mesh_1D_two_shot.h"
#include "ins_temp_broadcast_nhr.h"
#include "ccu_temp_broadcast_mesh_1D_mem2mem.h"
#include "ccu_temp_broadcast_nhr_1D_mem2mem.h"
#include "ins_broadcast_parallel_executor.h"

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsBroadcastParallelExecutor()
    : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::~InsBroadcastParallelExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    HCCL_INFO("[InsBroadcastParallelExecutor] CalcRes start, rank[%d]", myRank_);

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateMultiLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    // 计算localRankSize
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, intraLocalRankSize_, interLocalRankSize_));

    // 实例化算法模板类
    InsAlgTemplate0 intraTempAlg(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlg(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]);

    // 计算和准备Queue资源
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsBroadcastParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[InsBroadcastParallelExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }
    CHK_RET(interTempAlg.CalcRes(resReqInter));

    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqIntra.links, algResReq.levelRankPairs));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, resReqInter.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = resReqIntra.streamNum + resReqInter.streamNum;

    CHK_RET(CalcParallelNotifyReq(algResReq.primQueueNum, resReqIntra.queNum, algResReq.queueNotifys));

    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, algResReq.links));
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, algResReq.links));

    HCCL_INFO(
        "[InsBroadcastParallelExecutor] CalcRes end, rank[%d], required total que num [%u], que notify num [%u]",
        myRank_, algResReq.primQueueNum, algResReq.queueNotifys.size());

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    HCCL_INFO("[InsBroadcastParallelExecutor] CalcResOffload start, rank[%d]", myRank_);

    (void)dataSize;
    u64 scratchMemSize = 200 * 1024 * 1024;
    resReq.requiredScratchMemSize = scratchMemSize; // 200MB
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // 计算localRankSize
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, intraLocalRankSize_, interLocalRankSize_));

    // 实例化算法模板类
    InsAlgTemplate0 intraTempAlg(myRank_, intraLocalRankSize_, vTopo_[0], virtRankMap_[0]);
    InsAlgTemplate1 interTempAlg(myRank_, interLocalRankSize_, vTopo_[1], virtRankMap_[1]);

    // 计算和准备Queue资源
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[InsBroadcastParallelExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(intraTempAlg.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[InsBroadcastParallelExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(intraTempAlg.CalcRes(resReqIntra));
    }
    CHK_RET(interTempAlg.CalcRes(resReqInter));

    resReq.requiredSubQueNum = resReqIntra.streamNum + resReqInter.streamNum - 1;

    HCCL_INFO(
        "[InsBroadcastParallelExecutor] CalcResOffload end, rank[%d], required sub que num is [%u]", myRank_,
        resReq.requiredSubQueNum);

    return HcclResult::HCCL_SUCCESS;
}

// Host展开
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsBroadcastParallelExecutor] Host orchestrate begins.");

    // 初始化参数
    CHK_RET(Init(op, params, insQue));

    // 获取算法Topo信息
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // 计算localRankSize和localRoot
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, intraLocalRankSize_, interLocalRankSize_));
    CHK_RET(CalcLocalRoot());

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, intraLocalRankSize_, vTopo_.at(0), virtRankMap_.at(0)); // server内算法
    InsAlgTemplate1 tempAlgInter(myRank_, interLocalRankSize_, vTopo_.at(1), virtRankMap_.at(1)); // server间算法

    // 传入Template参数
    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.SetCollOp(op);
    tempAlgIntra.SetDataType(dataType_);
    tempAlgIntra.SetRoot(intraLocalRoot_);

    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.SetCollOp(op);
    tempAlgInter.SetDataType(dataType_);
    tempAlgInter.SetRoot(interLocalRoot_);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(rankGraph, tempAlgIntra, tempAlgInter));

    // 算法展开
    CHK_RET(GenInsQues(tempAlgIntra, tempAlgInter));

    HCCL_INFO("[InsBroadcastParallelExecutor] Host orchestrate success.");
    return HcclResult::HCCL_SUCCESS;
}

// Aicpu展开
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsBroadcastParallelExecutor] Aicpu orchestrate begins.");

    // 初始化参数
    CHK_RET(Init(op, params, insQue));

    // 获取算法Topo信息
    vTopo_ = topoInfo.vTopo;             // 本通信域内的通信平面
    virtRanks_ = topoInfo.virtRanks;     // 本通信域内的 rank 集合
    virtRankMap_ = topoInfo.virtRankMap; // 本通信域内的 rank 映射表

    // 计算localRankSize和localRoot
    CHK_RET(CalcLocalRankSize(myRank_, virtRanks_, intraLocalRankSize_, interLocalRankSize_));
    CHK_RET(CalcLocalRoot());

    // 实例化算法模板类
    InsAlgTemplate0 tempAlgIntra(myRank_, intraLocalRankSize_, vTopo_.at(0), virtRankMap_.at(0)); // server内算法
    InsAlgTemplate1 tempAlgInter(myRank_, interLocalRankSize_, vTopo_.at(1), virtRankMap_.at(1)); // server间算法

    // 传入Template参数
    tempAlgIntra.SetDmaMode(dmaMode_);
    tempAlgIntra.SetCollOp(op);
    tempAlgIntra.SetDataType(dataType_);
    tempAlgIntra.SetRoot(intraLocalRoot_);

    tempAlgInter.SetDmaMode(dmaMode_);
    tempAlgInter.SetCollOp(op);
    tempAlgInter.SetDataType(dataType_);
    tempAlgInter.SetRoot(interLocalRoot_);

    // 计算算法模板所需资源
    CHK_RET(PrepareResForTemplate(linkMgr, tempAlgIntra, tempAlgInter));

    // 算法展开
    CHK_RET(GenInsQues(tempAlgIntra, tempAlgInter));

    HCCL_INFO("[InsBroadcastParallelExecutor] Aicpu orchestrate success.");
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float>& splitDataSize) const
{
    // to do 先做等分，后续根据性能做调整
    double splitData = 0.5;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcLocalRoot()
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

// Host
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const RankGraph* rankGraph, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detouring enabled", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(rankGraph, resReqIntra));
    } else {
        HCCL_DEBUG("[%s] Rank[%d], CalcRes with detouring disabled", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));

    // 申请算法模板所需资源
    if (!(resReqIntra.queNum > 0 && resReqInter.queNum > 0)) {
        HCCL_ERROR("[InsBroadcastParallelExecutor]resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    u32 totalQueueNum = resReqIntra.queNum + resReqInter.queNum;
    CHK_RET(InitQueue(totalQueueNum, requiredQue_));
    for (u32 i = 0; i < requiredQue_.size(); i++) {
        if (i < resReqIntra.queNum) {
            intraQue_.push_back(requiredQue_.at(i));
        } else {
            interQue_.push_back(requiredQue_.at(i));
        }
    }
    // 每个算法的第0条流用于同步
    syncQueues_.emplace_back(intraQue_.at(0));
    syncQueues_.emplace_back(interQue_.at(0));

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqIntra.links, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, resReqInter.links, interLinks_));
    HCCL_INFO(
        "[InsBroadcastParallelExecutor] intraLinks size[%zu], interLinks size[%zu]", intraLinks_.size(),
        interLinks_.size());

    return HcclResult::HCCL_SUCCESS;
}

// Aicpu
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    ConnectedLinkMgr* linkMgr, InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    AlgTempResReq resReqIntra;
    AlgTempResReq resReqInter;
    if (enableDetour_) {
        HCCL_DEBUG("[%s] Rank[%d], with detouring enabled", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcResDetour(linkMgr, resReqIntra));
    } else {
        HCCL_DEBUG("[%s] Rank[%d], with detouring disabled", __func__, myRank_);
        CHK_RET(tempAlgIntra.CalcRes(resReqIntra));
    }
    CHK_RET(tempAlgInter.CalcRes(resReqInter));
    u32 intraQueNum = resReqIntra.queNum;
    u32 interQueNum = resReqInter.queNum;
    // 申请算法模板所需资源
    if (!(intraQueNum > 0 && interQueNum > 0)) {
        HCCL_ERROR("[InsBroadcastParallelExecutor]resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }
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

    CHK_RET(PrepResLinks(myRank_, resReqIntra.links, linkMgr, intraLinks_));
    CHK_RET(PrepResLinks(myRank_, resReqInter.links, linkMgr, interLinks_));
    HCCL_INFO(
        "[InsBroadcastParallelExecutor] intraLinks size[%zu], interLinks size[%zu]", intraLinks_.size(),
        interLinks_.size());

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenDataParams(
    const u64 dataOffset, const u64 sliceCount, const u64 scratchOffsetCount, TemplateDataParams& dataParams) const
{
    dataParams.buffInfo.inBuffType = BufferType::INPUT;
    dataParams.buffInfo.outBuffType = BufferType::INPUT;
    dataParams.buffInfo.scratBuffType = BufferType::SCRATCH;
    dataParams.buffInfo.inBuffBaseOff = dataOffset;
    dataParams.buffInfo.outBuffBaseOff = dataOffset;
    dataParams.buffInfo.scratchBuffBaseOff = scratchOffsetCount * dataTypeSize_;
    dataParams.sliceSize = sliceCount * dataTypeSize_;

    dataParams.inputSliceStride = 0;
    dataParams.outputSliceStride = 0;
    dataParams.repeatNum = 1;
    dataParams.inputRepeatStride = 0;
    dataParams.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsBroadcastParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenInsQues(
    InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter)
{
    HCCL_INFO("[InsBroadcastParallelExecutor] AlgTemplate intra server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsBroadcastParallelExecutor] AlgTemplate inter server is [%s]", tempAlgInter.Describe().c_str());

    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);

    u32 multipleIntra = tempAlgIntra.CalcScratchMultiple(BufferType::INPUT, BufferType::INPUT);
    u32 multipleInter = tempAlgInter.CalcScratchMultiple(BufferType::INPUT, BufferType::INPUT);

    // 按照intraData0+interData1，以及intraData1+interData0两种方式分别计算，取multiple最大需求
    float multiple0 = dataSplitSize.at(0) * float(multipleIntra) + dataSplitSize.at(1) * float(multipleInter);
    float multiple1 = dataSplitSize.at(1) * float(multipleIntra) + dataSplitSize.at(0) * float(multipleInter);
    float multiple = std::max(multiple0, multiple1);

    // 数据切分
    u64 sliceCount = std::min(static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_, dataCount_);
    if (multiple > 0 && maxTmpMemSize_ > 0) {
        u64 scratchCount = maxTmpMemSize_ / dataTypeSize_;             // 按照count来切分
        sliceCount = static_cast<u64>(float(scratchCount) / multiple); // 向下取整，防止Scratch溢出
    }
    u64 sliceCountPart0 = static_cast<u64>(float(sliceCount) * dataSplitSize.at(0));
    u64 sliceCountPart1 = sliceCount - sliceCountPart0;

    if (sliceCount == 0) {
        HCCL_WARNING("The divisor cannot be zero.");
        return HcclResult::HCCL_SUCCESS;
    }
    // 计算循环次数
    u32 loopTimes = dataCount_ / sliceCount + ((dataCount_ % sliceCount == 0) ? 0 : 1);
    // 计算尾块
    u64 finalSliceCount = dataCount_ - (loopTimes - 1) * sliceCount;
    u64 finalSliceCountPart0 = static_cast<u64>(float(finalSliceCount) * dataSplitSize.at(0));
    u64 finalSliceCountPart1 = finalSliceCount - finalSliceCountPart0;
    // 计算Scratch偏移，数据尾块必然小于常规块，不用额外计算尾块时的Scratch偏移
    u64 scratchOffsetCountIntraStage0 = 0;
    u64 scratchOffsetCountInterStage0 = sliceCountPart0 * multipleIntra;
    u64 scratchOffsetCountInterStage1 = 0;
    u64 scratchOffsetCountIntraStage1 = sliceCountPart0 * multipleInter;

    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsIntra1;
    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = false;

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCountPart0 = (loopIndex == loopTimes - 1) ? finalSliceCountPart0 : sliceCountPart0;
        u64 currCountPart1 = (loopIndex == loopTimes - 1) ? finalSliceCountPart1 : sliceCountPart1;
        u64 dataOffset0 = loopIndex * sliceCount * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + currCountPart0 * dataTypeSize_;

        // 第一步开始前同步
        CHK_RET(PreSyncQues(syncQueues_, 0));
        // server内topo包含root_的rank进行展开，其它rank不展开
        if (intraLocalRoot_ == root_ && currCountPart0 > 0) {
            // 数据0的server内的mesh算法
            GenDataParams(dataOffset0, currCountPart0, scratchOffsetCountIntraStage0, tempAlgParamsIntra0);
            CHK_RET(tempAlgIntra.GenExtIns(tempFuncs, tempAlgParamsIntra0, intraLinks_, intraQue_));
        }
        // server间topo包含root_的rank进行展开，其它rank不展开
        if (interLocalRoot_ == root_ && currCountPart1 > 0) {
            // 数据1的server间的nhr算法
            GenDataParams(dataOffset1, currCountPart1, scratchOffsetCountInterStage0, tempAlgParamsInter1);
            CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter1, interLinks_, interQue_));
        }
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncQues(syncQueues_, 0));

        // 第二步开始前同步
        CHK_RET(PreSyncQues(syncQueues_, 0));
        if (currCountPart0 > 0) {
            // 数据0的server间的nhr算法
            GenDataParams(dataOffset0, currCountPart0, scratchOffsetCountInterStage1, tempAlgParamsInter0);
            CHK_RET(tempAlgInter.GenExtIns(tempFuncs, tempAlgParamsInter0, interLinks_, interQue_));
        }
        if (currCountPart1 > 0) {
            // 数据1的server内的mesh算法
            GenDataParams(dataOffset1, currCountPart1, scratchOffsetCountIntraStage1, tempAlgParamsIntra1);
            CHK_RET(tempAlgIntra.GenExtIns(tempFuncs, tempAlgParamsIntra1, intraLinks_, intraQue_));
        }
        // 尾同步
        CHK_RET(PostSyncQues(syncQueues_, 0));
    }

    return HcclResult::HCCL_SUCCESS;
}

// 算法注册
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::BROADCAST, InsBroadcastParallelMesh1DNHR, InsBroadcastParallelExecutor, TopoMatchMeshNHR,
    InsTempBroadcastMesh1DTwoShot, InsTempBroadcastNHR);
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::BROADCAST, InsBroadcastParallelMesh1DNHRPcie, InsBroadcastParallelExecutor, TopoMatchMeshNHRPcie,
    InsTempBroadcastMesh1DTwoShot, InsTempBroadcastNHR);

#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TWO_TEMPS(
    OpType::BROADCAST, CcuBroadcastParallelMesh1DNHR, InsBroadcastParallelExecutor, TopoMatchMeshNHR,
    CcuTempBroadcastMesh1DMem2Mem, CcuTempBroadcastNHRMem2Mem1D);
#endif
} // namespace Hccl
