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
#include "ins_all_to_all_sole_executor.h"

#include "ccu_temp_all_to_all_mesh_1D.h"
#include "ccu_temp_all_to_all_v_mesh_1D.h"
#include "ccu_temp_all_to_all_mesh2d.h"
#include "ccu_temp_all_to_all_v_mesh_2D.h"
#include "ccu_temp_all_to_all_v_mesh_2Die.h"

namespace Hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsAlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsAlltoAllSoleExecutor() : InsCollAlgBase()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsAlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::~InsAlltoAllSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InitParams(
    const CollAlgOperator& op, const CollAlgParams& params)
{
    opMode_ = params.opMode;
    maxTmpMemSize_ = params.maxTmpMemSize;
    CHK_PRT_RET(
        (maxTmpMemSize_ == 0), HCCL_ERROR("[InitParams] maxTmpMemSize equals to zero for OPBASE."),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        GetAlltoAllLocalSendRecvInfo(op, myRank_, rankSize_, localSendRecvInfo_),
        HCCL_ERROR("[InitParams] unable to init DataInfo."), HcclResult::HCCL_E_PARA);
    if (op.opType == OpType::ALLTOALL) {
        sendType_ = op.all2AllDataDes.sendType;
        recvType_ = op.all2AllDataDes.recvType;
    } else if (op.opType == OpType::ALLTOALLV) {
        sendType_ = op.all2AllVDataDes.sendType;
        recvType_ = op.all2AllVDataDes.recvType;
    } else if (op.opType == OpType::ALLTOALLVC) {
        sendType_ = op.all2AllVCDataDes.sendType;
        recvType_ = op.all2AllVCDataDes.recvType;
    } else if (op.opType != OpType::HALFALLTOALLV) {
        HCCL_ERROR("[InsAlltoAllSoleExecutor] opType [%s] is invalid.", op.opType.Describe().c_str());
        return HcclResult::HCCL_E_PARA;
    }
    CHK_PRT_RET(
        InitOpInfo(op, opType_, redOp_, root_), HCCL_ERROR("[InitParams] unable to init OpInfo."),
        HcclResult::HCCL_E_PARA);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcResOffload(
    const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)dataSize;
    resReq.requiredScratchMemSize = 200 * 1024 * 1024; //  200 * 1024*1024 = 200M

    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsAlltoAllSoleExecutor] CalcResOffload SetPathNumMap");
    for (auto rankIdx : virtRanks_) {
        if (rankIdx == myRank_) {
            continue;
        }
        std::vector<NetInstance::Path> tmpPaths0 = rankGraph->GetPaths(0, myRank_, rankIdx);
        std::vector<NetInstance::Path> tmpPaths1 = rankGraph->GetPaths(1, myRank_, rankIdx);
        HCCL_INFO(
            "[InsAlltoAllSoleExecutor]tmpPaths0.size() = %zu,tmpPaths1.size() = %zu", tmpPaths0.size(),
            tmpPaths1.size());
        rank2PathNumMap[rankIdx] = tmpPaths0.size() + tmpPaths1.size();
    }

    tempAlg.setPathNumMap(rank2PathNumMap);

    // calculate required insQueues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsAlltoAllSoleExecutor], CalcResOffload with detouring enabled.");
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] [InsAlltoAllSoleExecutor], CalcResOffload with detouring disabled.");
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    resReq.requiredSubQueNum = tempResReq.streamNum - 1;

    HCCL_INFO(
        "[InsAlltoAllSoleExecutor][CalcResOffload] requiredSubQueNum[%llu], requiredScratchMemSize[%llu].",
        resReq.requiredSubQueNum, resReq.requiredScratchMemSize);

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult
InsAlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    algResReq.topoInfo.UpdateSingleLevelTopo(virtRanks_, virtRankMap_, vTopo_);
    HCCL_DEBUG(
        "[InsAlltoAllSoleExecutor][CalcRes]topoInfo.virtRanks[%u], topoInfo.virtRankMap[%u], topoInfo.vTopo[%u].",
        algResReq.topoInfo.virtRanks.size(), algResReq.topoInfo.virtRankMap.size(), algResReq.topoInfo.vTopo.size());
    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);

    std::map<u32, u32> rank2PathNumMap;
    HCCL_INFO("[InsAlltoAllSoleExecutor] CalcRes SetPathNumMap");
    for (auto rankIdx : virtRanks_) {
        if (rankIdx == myRank_) {
            continue;
        }
        std::vector<NetInstance::Path> tmpPaths0 = rankGraph->GetPaths(0, myRank_, rankIdx);
        std::vector<NetInstance::Path> tmpPaths1 = rankGraph->GetPaths(1, myRank_, rankIdx);
        rank2PathNumMap[rankIdx] = tmpPaths0.size() + tmpPaths1.size();
    }
    tempAlg.setPathNumMap(rank2PathNumMap);

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsAlltoAllSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsAlltoAllSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    algResReq.primQueueNum = tempResReq.streamNum;
    algResReq.queueNotifys = tempResReq.queNotifys;
    HCCL_DEBUG("[InsAlltoAllSoleExecutor] Rank[%d], requiredQueNum [%u].", myRank_, algResReq.primQueueNum);
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));
    return HcclResult::HCCL_SUCCESS;
}

// dataSize_ as input
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    // init and check params
    CHK_RET(Init(op, params, insQue));
    // Topo Match
    AlgTopoMatch topoMatch(myRank_, rankSize_, rankGraph, devType_);
    CHK_RET(topoMatch.MatchTopo(vTopo_, virtRanks_, virtRankMap_));
    HCCL_DEBUG("[InsAlltoAllSoleExecutor] Rank[%d], [%s].", myRank_, topoMatch.Describe().c_str());

    // instantiate a template
    InsAlgTemplate tempAlg(myRank_, rankSize_, vTopo_, virtRankMap_);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.SetCollOp(op); // CCU template需要传递op信息
    tempAlg.SetA2ASendRecvInfo(localSendRecvInfo_);
    tempAlg.SetLoadInfo(params);

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsAlltoAllSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(rankGraph, tempResReq));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] [InsAlltoAllSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    HCCL_DEBUG(
        "[InsAlltoAllSoleExecutor] Rank[%d], template [%s], requiredQue Num [%u].", myRank_, tempAlg.Describe().c_str(),
        tempResReq.queNum);

    CHK_RET(PrepResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, tempResLinks_));

    CHK_RET(OrchestrateOpbase(tempAlg));

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    HCCL_INFO("[InsAlltoAllSoleExecutor] Begin to orchestrate.");
    // init and check params
    CHK_RET(Init(op, params, insQue));
    dataType_ = op.dataType;
    // instantiate a template
    if (topoInfo.vTopo.size() == 0) {
        HCCL_ERROR("[InsAlltoAllSoleExecutor] Rank[%d], vTopo size is zero.", myRank_);
        return HcclResult::HCCL_E_PARA;
    }
    if (topoInfo.virtRankMap.size() == 0) {
        HCCL_ERROR("[InsAlltoAllSoleExecutor] Rank[%d], virtRankMap size is zero.", myRank_);
        return HcclResult::HCCL_E_PARA;
    }
    InsAlgTemplate tempAlg(myRank_, rankSize_, topoInfo.vTopo[0], topoInfo.virtRankMap[0]);
    tempAlg.SetDmaMode(dmaMode_);
    tempAlg.SetCollOp(op);
    tempAlg.SetA2ASendRecvInfo(localSendRecvInfo_);
    tempAlg.SetLoadInfo(params);
    tempAlg.SetDataType(dataType_);
    HCCL_DEBUG(
        "[InsAlltoAllSoleExecutor] Rank[%d], Init insAlgTemplate with rankSize [%u] and dmaMode [%s].", myRank_,
        rankSize_, dmaMode_.Describe().c_str());
    virtRankMap_ = topoInfo.virtRankMap[0];
    virtRanks_ = topoInfo.virtRanks[0];
    std::map<u32, u32> rank2PathNumMap;
    for (u32 rankIdx : virtRanks_) {
        auto links0 = linkMgr->GetLinks(0, rankIdx);
        auto links1 = linkMgr->GetLinks(1, rankIdx);
        if (links0.size() + links1.size() != 0) {
            rank2PathNumMap[rankIdx] = links0.size() + links1.size();
        }
    }
    tempAlg.setPathNumMap(rank2PathNumMap);

    // calculate required insQues and prepare queue
    AlgTempResReq tempResReq;
    if (enableDetour_) {
        HCCL_DEBUG("[InsCollAlgFactory] [InsAlltoAllSoleExecutor] Rank[%d], CalcRes with detouring enabled.", myRank_);
        CHK_RET(tempAlg.CalcResDetour(linkMgr, tempResReq));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] [InsAlltoAllSoleExecutor] Rank[%d], CalcRes with detouring disabled.", myRank_);
        CHK_RET(tempAlg.CalcRes(tempResReq));
    }

    CHK_RET(InitQueue(tempResReq.queNum, requiredQue_));
    HCCL_DEBUG(
        "[InsAlltoAllSoleExecutor] Rank[%d], template [%s], requiredQue Num [%u].", myRank_, tempAlg.Describe().c_str(),
        tempResReq.queNum);

    CHK_RET(PrepResLinks(myRank_, tempResReq.links, linkMgr, tempResLinks_));

    CHK_RET(OrchestrateOpbase(tempAlg));
    HCCL_INFO("[InsAlltoAllSoleExecutor] Orchestrate success.");

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsAlltoAllSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateOpbase(InsAlgTemplate& tempAlg)
{
    HCCL_DEBUG("[CollAlgFactory][InsAlltoAllSoleExecutor] AlgTemplate is [%s]", tempAlg.Describe().c_str());
    CHK_PRT_RET(
        maxTmpMemSize_ == 0,
        HCCL_ERROR("[InsAlltoAllSoleExecutor] Rank [%d], Invalid input maxTmpMemSize [%u].", myRank_, maxTmpMemSize_),
        HcclResult::HCCL_E_PARA);

    CHK_RET(tempAlg.GetScratchBufferInfo(maxTmpMemSize_, sendType_));

    BuffInfo buffInfo;
    buffInfo.inBuffType = BufferType::SCRATCH;
    buffInfo.outBuffType = BufferType::SCRATCH;
    buffInfo.inBuffBaseOff = 0;
    buffInfo.outBuffBaseOff = maxTmpMemSize_ / 2; // 占据scratch memory的后半部分，除以2
    RankSliceInfo sliceInfoVec;

    TempFuncs tempFuncs;
    tempFuncs.opMode = opMode_;
    tempFuncs.enableCounterNotify = IsEnableCounterNotifyByDevType(myRank_, devType_);
    tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
    tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required
    CHK_RET(tempAlg.Run(tempFuncs, sliceInfoVec, buffInfo, tempResLinks_, requiredQue_));
    HCCL_INFO("[InsAlltoAllSoleExecutor][OrchestrateOpbase] Run template success.");

    return HcclResult::HCCL_SUCCESS;
}

INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALL, InsAlltoAllMesh, InsAlltoAllSoleExecutor, TopoMatchMesh, InsTempAlltoAllMesh);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALLV, InsAlltoAllvMesh, InsAlltoAllSoleExecutor, TopoMatchMesh, InsTempAlltoAllMesh);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALLVC, InsAlltoAllvcMesh, InsAlltoAllSoleExecutor, TopoMatchMesh, InsTempAlltoAllMesh);
#ifndef CCL_KERNEL_AICPU
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALL, CcuAlltoAllMesh1D, InsAlltoAllSoleExecutor, TopoMatchMesh, CcuTempAllToAllMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALLV, CcuAlltoAllVMesh1D, InsAlltoAllSoleExecutor, TopoMatchMesh, CcuTempAlltoAllVMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALL, CcuAlltoAllMesh2D, InsAlltoAllSoleExecutor, TopoMatchConcurrMesh, CcuTempAlltoAllMesh2D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALLV, CcuAlltoAllVMesh2D, InsAlltoAllSoleExecutor, TopoMatchConcurrMesh, CcuTempAlltoAllVMesh2D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::HALFALLTOALLV, CcuHalfAll2AllVMesh1D, InsAlltoAllSoleExecutor, TopoMatchMesh, CcuTempHalfAllToAllVMesh1D);
INS_REGISTER_IMPL_BY_TEMP(
    OpType::ALLTOALLV, CcuAlltoAllVMesh2Die, InsAlltoAllSoleExecutor, TopoMatchMesh, CcuTempAlltoAllVMesh2Die);
#endif

} // namespace Hccl
