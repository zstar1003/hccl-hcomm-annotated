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
#include "ins_recv_executor.h"
#include "alg_data_trans_wrapper.h"

using namespace std;

namespace Hccl {
InsRecvExecutor::InsRecvExecutor() : InsCollAlgBase() {}

InsRecvExecutor::~InsRecvExecutor() {}

HcclResult InsRecvExecutor::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    dmaMode_ = DmaMode::DEFAULT;
    HCCL_DEBUG("[InsCollAlgFactory][InsRecvExecutor][Orchestrate] Begin to Generate Instruction Queue for RECV.");
    CHK_RET(Init(op, params, insQue));
    // 从集合通信算子op中获得remote rank和data type/count相关信息
    RankId remoteRank = op.sendRecvRemoteRank;
    u32 dataElemSize = DATA_TYPE_SIZE_MAP.at(op.dataType);
    u64 totalDataSize = static_cast<u64>(dataElemSize) * op.dataCount;
    if (totalDataSize == 0) {
        HCCL_WARNING("[InsCollAlgFactory][InsRecvExecutor][Orchestrate] totalDataSize is 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }
    // 判断是不是自发自收这种情况，若是，则什么都不做，直接返回
    if (myRank_ == remoteRank) {
        HCCL_WARNING("[InsCollAlgFactory][InsRecvExecutor][Orchestrate] Self send, Self recv, Do nothing");
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_DEBUG("[InsCollAlgFactory][InsRecvExecutor][Orchestrate] Other send, Self recv");

    // 从virtualTopo里面拿到 link，转成LinkData格式
    const std::vector<NetInstance::Path> recvPath = GetPathsFromRankGraph(rankGraph, myRank_, remoteRank);
    CHK_PRT_RET(
        recvPath.size() == 0,
        HCCL_ERROR("[InsCollAlgFactory] Unable to obtain valid link, srcRank [%d], dstRank [%d].", myRank_, remoteRank),
        HcclResult::HCCL_E_INTERNAL);
    LinkData recvLinkData(recvPath[0]);
    if (recvLinkData.GetType() == PortDeploymentType::P2P && recvLinkData.GetLinkProtocol() == LinkProtocol::PCIE) {
        dmaMode_ = DmaMode::GET;
    }
    HCCL_DEBUG(
        "[InsCollAlgFactory][InsRecvExecutor][Orchestrate] Total transfer data size [%llu], Max scratch buffer size "
        "[%u].",
        totalDataSize, params.maxTmpMemSize);

    // 初始化循环参数
    u64 resDataSize = totalDataSize;
    u64 currentOffset = 0;
    u32 roundIdx = 0;
    // 模式判断
    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OFFLOAD Mode for HOST.", myRank_);
        u64 transferSize = resDataSize;
        // 根据本轮数据搬运量声明相关DataSlice--图模式一次搬运全部数据 从inputbuffer直接send到对面outputbuffer
        DataSlice outputBuffer(BufferType::OUTPUT, currentOffset, transferSize);
        DataSlice remoteInputBuffer(BufferType::INPUT, currentOffset, transferSize);
        SlicesList recvSlicesList({remoteInputBuffer}, {outputBuffer});
        DataInfo recvInfo(recvLinkData, recvSlicesList);
        CHK_RET(Recv(recvInfo, insQue, 0, true, dmaMode_));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OPBASE Mode for HOST.", myRank_);
        // 当需要多轮搬运时，需保证一次数据的搬运量需为单个数据size的整数倍
        u64 maxRoundTransferSize = params.maxTmpMemSize - params.maxTmpMemSize % dataElemSize;
        while (resDataSize > 0) {
            // 判断本轮需搬运的数据量
            u64 transferSize = resDataSize > params.maxTmpMemSize ? maxRoundTransferSize : resDataSize;
            HCCL_DEBUG(
                "[InsCollAlgFactory][InsRecvExecutor][Orchestrate] Recv round [%u], transfer data size [%llu]",
                roundIdx, transferSize);
            // 根据本轮数据搬运量创建相关DataSlice
            DataSlice outputBuffer(BufferType::OUTPUT, currentOffset, transferSize);
            DataSlice scratchBuffer(BufferType::SCRATCH, 0, transferSize);
            DataSlice remoteScratchBuffer(BufferType::SCRATCH, 0, transferSize);
            SlicesList recvSlicesList({remoteScratchBuffer}, {scratchBuffer});
            DataInfo recvInfo(recvLinkData, recvSlicesList);
            CHK_RET(Recv(recvInfo, insQue, 0, true, dmaMode_));
            // local copy
            CHK_RET(LocalCopy(insQue, scratchBuffer, outputBuffer));
            // 更新循环参数
            currentOffset = currentOffset + transferSize;
            resDataSize = resDataSize - transferSize;
            roundIdx = roundIdx + 1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsRecvExecutor::CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)rankGraph;
    (void)dataSize;
    resReq.requiredScratchMemSize = 0; // 图模式不用scratch memory
    resReq.requiredSubQueNum = 0;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsRecvExecutor::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    u32 linkNumBtwPeers = 1;
    algResReq.primQueueNum = 1;
    AlgTempResReq tempResReq;
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.streamNum;
    if (static_cast<u32>(sendRecvRemoteRank_) > rankSize_ - 1) {
        HCCL_ERROR(
            "[InsCollAlgFactory][InsRecvExecutor][CalcRes] Rank[%d] get dest[%d] is invalid", myRank_,
            sendRecvRemoteRank_);
        return HcclResult::HCCL_E_PARA;
    }
    tempResReq.links[sendRecvRemoteRank_] = linkNumBtwPeers;
    uint32_t linkNum = GetPathsFromRankGraph(rankGraph, myRank_, sendRecvRemoteRank_).size();
    if (linkNum == 0) {
        HCCL_ERROR(
            "[InsCollAlgFactory][InsRecvExecutor][CalcRes] Rank[%d] get path num to dest[%d] is zero", myRank_,
            sendRecvRemoteRank_);
    }
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsRecvExecutor::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    dmaMode_ = DmaMode::DEFAULT;
    (void)topoInfo;
    HCCL_DEBUG(
        "[InsCollAlgFactory][InsRecvExecutor][Orchestrate] Begin to Generate Instruction Queue for RECV AICPU mode.");
    CHK_RET(Init(op, params, insQue));
    // 从集合通信算子op中获得remote rank和data type/count相关信息
    RankId remoteRank = op.sendRecvRemoteRank;
    u32 dataElemSize = DATA_TYPE_SIZE_MAP.at(op.dataType);
    u64 totalDataSize = static_cast<u64>(dataElemSize) * op.dataCount;
    if (totalDataSize == 0) {
        HCCL_WARNING("[InsCollAlgFactory][InsRecvExecutor][Orchestrate] totalDataSize is 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }
    // 判断是不是自发自收这种情况，若是，则什么都不做，直接返回
    if (myRank_ == remoteRank) {
        HCCL_WARNING("[InsCollAlgFactory][InsRecvExecutor][Orchestrate] Self send, Self recv, Do nothing");
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_DEBUG("[InsCollAlgFactory][InsRecvExecutor][Orchestrate] Other send, Self recv.");

    // 从linkMgr里面拿到 linkData
    const vector<LinkData> recvPath = linkMgr->GetLinks(remoteRank);
    CHK_PRT_RET(
        recvPath.size() == 0,
        HCCL_ERROR("[InsCollAlgFactory] Unable to obtain valid link, srcRank [%d], dstRank [%d].", myRank_, remoteRank),
        HcclResult::HCCL_E_INTERNAL);
    LinkData recvLinkData(recvPath[0]);
    if (recvLinkData.GetType() == PortDeploymentType::P2P && recvLinkData.GetLinkProtocol() == LinkProtocol::PCIE) {
        dmaMode_ = DmaMode::GET;
    }
    HCCL_DEBUG(
        "[InsCollAlgFactory][InsRecvExecutor][Orchestrate] Total transfer data size [%llu], Max scratch buffer size "
        "[%u].",
        totalDataSize, params.maxTmpMemSize);

    // 初始化循环参数
    u64 resDataSize = totalDataSize;
    u64 currentOffset = 0;
    u32 roundIdx = 0;
    // 模式判断
    if (opMode_ == OpMode::OFFLOAD) {
        u64 maxLoopOutputSize = 256 * 1024 * 1024; // 256m为一轮
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OFFLOAD Mode for HOST.", myRank_);
        while (resDataSize > 0) {
            u64 transferSize = resDataSize > maxLoopOutputSize ? maxLoopOutputSize : resDataSize;
            DataSlice outputBuffer(BufferType::OUTPUT, currentOffset, transferSize);
            DataSlice remoteInputBuffer(BufferType::INPUT, currentOffset, transferSize);
            SlicesList recvSlicesList({remoteInputBuffer}, {outputBuffer});
            DataInfo recvInfo(recvLinkData, recvSlicesList);
            CHK_RET(Recv(recvInfo, insQue, 0, true, dmaMode_));
            currentOffset = currentOffset + transferSize;
            resDataSize = resDataSize - transferSize;
            roundIdx = roundIdx + 1;
        }
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OPBASE Mode for HOST.", myRank_);
        // 当需要多轮搬运时，需保证一次数据的搬运量需为单个数据size的整数倍
        u64 maxRoundTransferSize = params.maxTmpMemSize - params.maxTmpMemSize % dataElemSize;
        while (resDataSize > 0) {
            // 判断本轮需搬运的数据量
            u64 transferSize = resDataSize > params.maxTmpMemSize ? maxRoundTransferSize : resDataSize;
            HCCL_DEBUG(
                "[InsCollAlgFactory][InsRecvExecutor][Orchestrate] Recv round [%u], transfer data size [%llu]",
                roundIdx, transferSize);
            // 根据本轮数据搬运量创建相关DataSlice
            DataSlice outputBuffer(BufferType::OUTPUT, currentOffset, transferSize);
            DataSlice scratchBuffer(BufferType::SCRATCH, 0, transferSize);
            DataSlice remoteScratchBuffer(BufferType::SCRATCH, 0, transferSize);

            SlicesList recvSlicesList({remoteScratchBuffer}, {scratchBuffer});
            DataInfo recvInfo(recvLinkData, recvSlicesList);
            CHK_RET(Recv(recvInfo, insQue, 0, true, dmaMode_));
            // local copy
            CHK_RET(LocalCopy(insQue, scratchBuffer, outputBuffer));
            // 更新循环参数
            currentOffset = currentOffset + transferSize;
            resDataSize = resDataSize - transferSize;
            roundIdx = roundIdx + 1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// 注册
INS_REGISTER_IMPL(OpType::RECV, InsRecv, InsRecvExecutor);

} // namespace Hccl
