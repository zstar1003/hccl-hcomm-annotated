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
#ifndef CCL_KERNEL_AICPU
#include "ccu_temp_scatter_mesh_2D.h"
#endif
#include "topo_match_concurr_mesh.h"
#include "ins_scatter_sole_executor.h"

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsScatterSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsScatterSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    // 通过判断哪层通信域能有到所有remoteRank的path，判断当前算法跑在哪一层
    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsV2ScatterSoleExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    tempAlg.setPathNumMap(rank2PathNumMap);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.SetRoot(root_);

    if (tempAlg.GetExpandedMode() == DeviceMode::CCU) {
        resReq.requiredScratchMemSize = dataSize * rankSize_;
        HCCL_DEBUG(
            "[InsScatterSoleExecutor][CalcResOffload][CCU] requiredScratchSize:[%llu], dataSize:[%llu], "
            "rankSize:[%llu]",
            resReq.requiredScratchMemSize, dataSize, rankSize_);
    } else {
        (void)dataSize;
        resReq.requiredScratchMemSize = 0;
    }

    // calculate required insQueues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsScatterSoleExecutor], CalcRes with detouring enabled.");
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] [InsScatterSoleExecutor], CalcRes with detouring disabled.");
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    // 通过判断哪层通信域能有到所有remoteRank的path，判断当前算法跑在哪一层
    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsV2ScatterSoleExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    tempAlg.setPathNumMap(rank2PathNumMap);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.SetRoot(root_);

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsScatterSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] [InsScatterSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    algResReq.localWaitGroupCntNotify = tempResReq.localWaitGroupCntNotify;
    algResReq.localBcastPostCntNotify = tempResReq.localBcastPostCntNotify;
    HCCL_DEBUG(
        "[InsCollAlgFactory] [InsScatterSoleExecutor] Rank[%d], requiredQueNum [%u].", myRank_, algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));

    return HcclResult::HCCL_SUCCESS;
}

// dataSize_ as input
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_INFO("[InsScatterSoleExecutor]ScatterSoleExecutor Orchestrate begin");
    // init and check params
    CHK_RET(Init(op, params, insQue));

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    HCCL_INFO("[InsScatterSoleExecutor] Rank[%d], [%s].", myRank_, topoMatch.Describe().c_str());
    dataType_ = op.dataType;
    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.SetCollOp(op); // ccu需要传递op信息
    tempAlg.SetRoot(root_);
    tempAlg.SetDataType(dataType_);
    // 通过判断哪层通信域能有到所有remoteRank的path，判断当前算法跑在哪一层
    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsV2ScatterSoleExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByRankGraphMultiLevel(rankGraph, virtRanks_, myRank_, rank2PathNumMap));
    tempAlg.setPathNumMap(rank2PathNumMap);
    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        tempAlg.SetDataType(dataType_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    HCCL_DEBUG(
        "[InsCollAlgFactory] [InsScatterSoleExecutor] Rank[%d], template [%s], requiredQue Num [%u].", myRank_,
        tempAlg.Describe().c_str(), tempResReq.queNum);

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));

    // 令Scatter算子的dataSize_为outputSize
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;
    HCCL_DEBUG("[InsScatterSoleExecutor][Orchestrate] dataSize[%llu]", dataSize_);

    if (tempAlg.GetExpandedMode() == DeviceMode::CCU) {
        HCCL_DEBUG("[InsScatterSoleExecutor] Rank[%d], Generating Instruction Queues for CCU.", myRank_);
        CHK_RET(GenInsQues4Ccu(tempAlg));
        return HcclResult::HCCL_SUCCESS;
    }
    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsScatterSoleExecutor] Rank[%d], Generating Instruction Queues in OFFLOAD Mode for "
            "HOST.",
            myRank_);
        CHK_RET(GenInsQues4Offload(tempAlg));
    } else { // OPBASE
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsScatterSoleExecutor] Rank[%d], Generating Instruction Queues in OPBASE Mode for "
            "HOST.",
            myRank_);
        CHK_RET(GenInsQues4Opbase(tempAlg));
    }
    return HcclResult::HCCL_SUCCESS;
}

// 算子执行aicpu接口
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsCollAlgFactory] [InsScatterSoleExecutor] [InsScatterSoleExecutor] AiCpu Orchestrate begins.");
    // 参数校验和初始化
    CHK_RET(Init(op, params, insQue));

    // soleEsecutor 只支持单层拓扑, 所以只取第 0 级通信域的信息
    vTopo_ = topoInfo.vTopo[0];             // 本通信域内的通信平面
    virtRankMap_ = topoInfo.virtRankMap[0]; // 本通信域内的 rank 映射表
    virtRanks_ = topoInfo.virtRanks[0];     // 本通信域内的 rank 集合
    dataTypeSize_ = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = op.dataType;
    CHK_PRT_RET(
        dataTypeSize_ == 0,
        HCCL_ERROR("Scatter_[CollAlgFactory] Rank [%d], Invalid dataTypeSize_ [%u].", myRank_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    // 实例化算法模板类
    HCCL_DEBUG(
        "Scatter_[InsScatterSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDataType(dataType_);
    // 通过判断哪层通信域能有到所有remoteRank的path，判断当前算法跑在哪一层
    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsV2ScatterSoleExecutor] CalcRes SetPathNumMap");
    CHK_RET(SetPathNumMapByLinkMgrMultiLevel(linkMgr, virtRanks_, myRank_, rank2PathNumMap));
    tempAlg.setPathNumMap(rank2PathNumMap);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.SetCollOp(op); // ccu需要传递op信息
    tempAlg.SetRoot(root_);

    // 计算算法模板所需资源
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsScatterSoleExecutor] Rank[%d], CalcRes with detouring enabled for Orchestrate.",
            myRank_);
        CHK_RET(tempAlg.CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_DEBUG(
            "[InsCollAlgFactory] [InsScatterSoleExecutor] Rank[%d], CalcRes with detouring disabled for Orchestrate.",
            myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    // 申请算法模板所需资源
    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));

    // 令Scatter算子的dataSize_为outputSize
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    dataSize_ = dataCount_ * dataSizePerVolume;

    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_DEBUG(
            "[InsScatterSoleExecutor] Rank[%d], Generating Instruction Queues in OFFLOAD Mode for AICPU.", myRank_);
        CHK_RET(GenInsQues4Offload(tempAlg));
    } else { // OPBASE
        HCCL_DEBUG(
            "[InsScatterSoleExecutor] Rank[%d], Generating Instruction Queues in OPBASE Mode for AICPU.", myRank_);
        CHK_RET(GenInsQues4Opbase(tempAlg));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GenInsQues4Offload(InsAlgTemplate& tempAlg)
{
    RankSliceInfo sliceInfoVec;
    AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
    CHK_RET(tempAlg.CalcSliceInfo(allignInfo, dataSize_, sliceInfoVec));
    HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], done calculating slice information.", myRank_);

    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::INPUT;
    buffInfo.outBuffType = BufferType::OUTPUT;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;
    HCCL_DEBUG("[CollAlgFactory] AlgTemplate is [%s]", tempAlg.Describe().c_str());
    HCCL_DEBUG(
        "[InsCollAlgFactory] Rank[%d], input buffer type [%s], output buffer type [%s], input buffer base "
        "offset [%u], output buffer base offset [%u].",
        myRank_, buffInfo.inBuffType.Describe().c_str(), buffInfo.outBuffType.Describe().c_str(),
        buffInfo.inBuffBaseOff, buffInfo.outBuffBaseOff);

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.isForepart = true; // only have one Temp, soleExecutor is always true
    tempFuncs.isBottom = true;   // only have one Temp, soleExecutor is always true
    HCCL_DEBUG("[CollAlgFactory] AlgTemplate is [%s]", tempAlg.Describe().c_str());

    CHK_RET(tempAlg.Run(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], done generating instruction queues.", myRank_);

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GenInsQues4Opbase(InsAlgTemplate& tempAlg)
{
    HCCL_DEBUG("[CollAlgFactory] AlgTemplate is [%s]", tempAlg.Describe().c_str());
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    CHK_PRT_RET(
        dataSizePerVolume == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid dataSizePerVolume [%u].", myRank_, dataSizePerVolume),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(rankSize_ == 0, HCCL_ERROR("[CollAlgFactory] RankSize is zero!"), HcclResult::HCCL_E_PARA);
    // maxTmpMemSize_为整个Scratch的大小
    u64 scratchOutputMemSize
        = static_cast<u64>(floor(maxTmpMemSize_ / (rankSize_ * dataSizePerVolume)) * dataSizePerVolume);

    CHK_PRT_RET(
        scratchOutputMemSize == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid input maxTmpMemSize [%u].", myRank_, maxTmpMemSize_),
        HcclResult::HCCL_E_PARA);

    // 统一管理基地址偏移
    BuffInfo buffInfo;
    buffInfo.outBuffType = BufferType::SCRATCH;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;
    buffInfo.scratchBuffBaseOff = 0;

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotify();
    tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
    tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required

    // 计算CCL的循环次数，dataSize_为Scatter的outputSize（小的）,看CCLout与ScatterOut的倍数关系
    u64 sendRecvTimes = (dataSize_ / scratchOutputMemSize) + ((dataSize_ % scratchOutputMemSize) == 0 ? 0 : 1);
    HCCL_INFO("[insScatterSoleExecutor] Rank [%d], sendRecvTimes [%u].", myRank_, sendRecvTimes);

    for (u32 idx = 0; idx < sendRecvTimes; idx++) {
        // 本轮的ScratchOut的大小（小的）
        u64 currDataSize
            = (idx == (sendRecvTimes - 1)) ? (dataSize_ - idx * scratchOutputMemSize) : scratchOutputMemSize;

        RankSliceInfo sliceInfoVec;
        AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};

        // 每轮cclLoop，准备好usrData的input本地拷贝到scratch的dataSlices，和scratch本地拷贝到output的dataSlices；存放在tempFunc.usrData中
        UsrData usrData;
        u64 usrInOffset = idx * scratchOutputMemSize;
        u64 usrInRankStride = dataSize_;
        for (RankId r : virtRanks_) {
            u32 rankId = virtRankMap_[r];
            usrData.usrInSlices.emplace_back(
                DataSlice(BufferType::INPUT, usrInOffset + rankId * usrInRankStride, currDataSize));
            usrData.scratchInSlices.emplace_back(DataSlice(BufferType::SCRATCH, rankId * currDataSize, currDataSize));
        }

        usrData.scratchOutSlices.emplace_back(DataSlice(BufferType::SCRATCH, myRank_ * currDataSize, currDataSize));
        usrData.usrOutSlices.emplace_back(DataSlice(BufferType::OUTPUT, usrInOffset, currDataSize));
        tempFuncs.usrData = usrData;

        // 计算SliceInfo,nhr也按照mesh的方式，分rankSize片，每片的大小为curDataSize(按照output计算)
        CHK_RET(tempAlg.CalcSliceInfo(allignInfo, currDataSize, sliceInfoVec));
        CHK_RET(tempAlg.Run(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsScatterSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GenInsQues4Ccu(InsAlgTemplate& tempAlg)
{
    HCCL_DEBUG("[ScatterSoleExecutor][GenInsQues4Ccu] Gen InsQue start");
    u32 dataSizePerVolume = DataTypeSizeGet(dataType_);
    CHK_PRT_RET(
        dataSizePerVolume == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid dataSizePerVolume [%u].", myRank_, dataSizePerVolume),
        HcclResult::HCCL_E_INTERNAL);

    // maxTmpMemSize_为整个Scratch的大小,按scatter的output的计算
    u64 scratchOutputMemSize
        = static_cast<u64>(floor(maxTmpMemSize_ / (rankSize_ * dataSizePerVolume)) * dataSizePerVolume);

    CHK_PRT_RET(
        scratchOutputMemSize == 0,
        HCCL_ERROR("[CollAlgFactory] Rank [%d], Invalid input maxTmpMemSize [%u].", myRank_, maxTmpMemSize_),
        HcclResult::HCCL_E_PARA);

    // 统一管理基地址偏移
    BuffInfo buffInfo;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = 0;
    buffInfo.scratchBuffBaseOff = 0;

    // CCLBuf的切分，UB的最大传输值的切分，统一在executor中处理
    // dataSize_为Scatter的outputSize（小的，已含DataType）
    uint64_t tempMaxSliceSize = tempAlg.GetMaxSliceSize();
    uint64_t blockSize = dataSize_ < tempMaxSliceSize ? dataSize_ : tempMaxSliceSize;
    blockSize = blockSize < scratchOutputMemSize ?
                    blockSize :
                    scratchOutputMemSize; // 按blockSize切分，则可以同时满足UB传输上限、CCLbuff上限
    // 将dataSize_按照blockSize切分
    u32 loopTimes = (dataSize_ / blockSize) + ((dataSize_ % blockSize) == 0 ? 0 : 1);
    HCCL_DEBUG(
        "[ins_scatter_sole_executor][GenInsQues4Ccu] dataSize_[%llu], blockSize[%llu], loopTimes[%u], "
        "scratchOutputMemSize[%u], maxTmpMemSize[%u] ",
        dataSize_, blockSize, loopTimes, scratchOutputMemSize, maxTmpMemSize_);
    TempFuncs tempFuncs;
    for (uint64_t idx = 0; idx < loopTimes; idx++) {
        uint64_t sliceSize = ((idx == loopTimes - 1) ? (dataSize_ - idx * blockSize) : blockSize);
        uint64_t offset = idx * blockSize;
        // tempAlg从op_中可以获取input，output，scratch的基地址， 从dataSize_获取Stride
        // 从buffInfo中可以获取每次的偏移
        buffInfo.inBuffBaseOff = offset;
        buffInfo.outBuffBaseOff = offset;
        RankSliceInfo sliceInfoVec;
        AllignInfo allignInfo = {enableAllign_, allignSize_, dataType_};
        // 从sliceInfoVec中获取sliceSize
        CHK_RET(tempAlg.CalcSliceInfo(allignInfo, sliceSize, sliceInfoVec));
        CHK_RET(tempAlg.Run(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    }
    return HcclResult::HCCL_SUCCESS;
}

#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::SCATTER, CcuScatterMesh2D, InsScatterSoleExecutor, TopoMatchConcurrMesh, CcuTempScatterMesh2D);
#endif
} // namespace Hccl
