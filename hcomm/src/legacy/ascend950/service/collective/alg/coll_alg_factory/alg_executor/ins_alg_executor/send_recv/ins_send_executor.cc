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
#include "ins_send_executor.h"
#include "alg_data_trans_wrapper.h"

using namespace std;

namespace Hccl {
InsSendExecutor::InsSendExecutor() : InsCollAlgBase() {}

InsSendExecutor::~InsSendExecutor() {}

HcclResult InsSendExecutor::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    dmaMode_ = DmaMode::DEFAULT;
    HCCL_DEBUG("[InsCollAlgFactory][InsSendExecutor][Orchestrate] Begin to Generate Instruction Queue for SEND.");
    CHK_RET(Init(op, params, insQue));
    // 从集合通信算子op中获得remote rank和data type/count相关信息
    RankId remoteRank = op.sendRecvRemoteRank;
    u32 dataElemSize = DATA_TYPE_SIZE_MAP.at(op.dataType);
    u64 totalDataSize = static_cast<u64>(dataElemSize) * op.dataCount;
    if (totalDataSize == 0) {
        HCCL_WARNING("[InsCollAlgFactory][InsSendExecutor][Orchestrate] totalDataSize is 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }
    // 判断是不是自发自收这种情况，若是，则什么都不做，直接返回
    if (myRank_ == remoteRank) {
        HCCL_WARNING("[InsCollAlgFactory][InsSendExecutor][Orchestrate] Self send, Self recv, Do nothing");
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_DEBUG("[InsCollAlgFactory][InsSendExecutor][Orchestrate] Self send, Other recv");

    // 从rankGraph里面拿到 link，转成LinkData格式
    const std::vector<NetInstance::Path> sendPath = GetPathsFromRankGraph(rankGraph, myRank_, remoteRank);
    CHK_PRT_RET(
        sendPath.size() == 0,
        HCCL_ERROR("[InsCollAlgFactory] Unable to obtain valid link, srcRank [%d], dstRank [%d].", myRank_, remoteRank),
        HcclResult::HCCL_E_INTERNAL);
    LinkData sendLinkData(sendPath[0]);
    if (sendLinkData.GetType() == PortDeploymentType::P2P && sendLinkData.GetLinkProtocol() == LinkProtocol::PCIE) {
        dmaMode_ = DmaMode::GET;
    }
    HCCL_DEBUG(
        "[InsCollAlgFactory][InsSendExecutor][Orchestrate] Total transfer data size [%llu], Max scratch buffer size "
        "[%u].",
        totalDataSize, params.maxTmpMemSize);

    // 初始化循环参数
    u64 resDataSize = totalDataSize;
    u64 currentOffset = 0;
    u32 roundIdx = 0;
    // 模式判断
    HCCL_DEBUG("[InsCollAlgFactory] opmode_ is [%d], ", opMode_);
    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OFFLOAD Mode for HOST.", myRank_);
        u64 transferSize = resDataSize;
        // 根据本轮数据搬运量声明相关DataSlice--图模式一次搬运全部数据 从inputbuffer直接send到对面outputbuffer
        DataSlice inputBuffer(BufferType::INPUT, currentOffset, transferSize);
        DataSlice remoteOutputBuffer(BufferType::OUTPUT, currentOffset, transferSize);
        SlicesList sendSlicesList({inputBuffer}, {remoteOutputBuffer});
        DataInfo sendInfo(sendLinkData, sendSlicesList);
        CHK_RET(Send(sendInfo, insQue, 0, true, dmaMode_));
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OPBASE Mode for HOST.", myRank_);
        // 当需要多轮搬运时，需保证一次数据的搬运量需为单个数据size的整数倍
        u64 maxRoundTransferSize = params.maxTmpMemSize - params.maxTmpMemSize % dataElemSize;
        while (resDataSize > 0) {
            // 判断本轮需搬运的数据量
            u64 transferSize = resDataSize > params.maxTmpMemSize ? maxRoundTransferSize : resDataSize;
            HCCL_DEBUG(
                "[InsCollAlgFactory][InsSendExecutor][Orchestrate] Send round [%u], transfer data size [%llu].",
                roundIdx, transferSize);
            // 根据本轮数据搬运量声明相关DataSlice
            DataSlice inputBuffer(BufferType::INPUT, currentOffset, transferSize);
            DataSlice scratchBuffer(BufferType::SCRATCH, 0, transferSize);
            DataSlice remoteScratchBuffer(BufferType::SCRATCH, 0, transferSize);
            // local copy
            CHK_RET(LocalCopy(insQue, inputBuffer, scratchBuffer));
            SlicesList sendSlicesList({scratchBuffer}, {remoteScratchBuffer});
            DataInfo sendInfo(sendLinkData, sendSlicesList);
            CHK_RET(Send(sendInfo, insQue, 0, true, dmaMode_));

            currentOffset = currentOffset + transferSize;
            resDataSize = resDataSize - transferSize;
            roundIdx = roundIdx + 1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsSendExecutor::CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)rankGraph;
    (void)dataSize;
    resReq.requiredScratchMemSize = 0; // 图模式不用scratch memory
    resReq.requiredSubQueNum = 0;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsSendExecutor::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    u32 linkNumBtwPeers = 1;
    algResReq.primQueueNum = 1;
    AlgTempResReq tempResReq;
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.streamNum;
    if (static_cast<u32>(sendRecvRemoteRank_) > rankSize_ - 1) {
        HCCL_ERROR(
            "[InsCollAlgFactory][InsSendExecutor][CalcRes] Rank[%d] get dest[%d] is invalid", myRank_,
            sendRecvRemoteRank_);
        return HcclResult::HCCL_E_PARA;
    }
    tempResReq.links[sendRecvRemoteRank_] = linkNumBtwPeers;
    uint32_t linkNum = GetPathsFromRankGraph(rankGraph, myRank_, sendRecvRemoteRank_).size();
    if (linkNum == 0) {
        HCCL_ERROR(
            "[InsCollAlgFactory][InsSendExecutor][CalcRes] Rank[%d] get path num to dest[%d] is zero", myRank_,
            sendRecvRemoteRank_);
    }
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsSendExecutor::Orchestrate(
    const AlgTopoInfo& topoInfo, // aicpu
    const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr, InsQuePtr insQue)
{
    dmaMode_ = DmaMode::DEFAULT;
    (void)topoInfo;
    HCCL_DEBUG(
        "[InsCollAlgFactory][InsSendExecutor][Orchestrate] Begin to Generate Instruction Queue for SEND AICPU mode.");
    CHK_RET(Init(op, params, insQue));
    // 从集合通信算子op中获得remote rank和data type/count相关信息
    RankId remoteRank = op.sendRecvRemoteRank;
    u32 dataElemSize = DATA_TYPE_SIZE_MAP.at(op.dataType);
    u64 totalDataSize = static_cast<u64>(dataElemSize) * op.dataCount;
    if (totalDataSize == 0) {
        HCCL_WARNING("[InsCollAlgFactory][InsSendExecutor][Orchestrate] totalDataSize is 0, do nothing.");
        return HcclResult::HCCL_SUCCESS;
    }
    // 判断是不是自发自收这种情况，若是，则什么都不做，直接返回
    if (myRank_ == remoteRank) {
        HCCL_WARNING("[InsCollAlgFactory][InsSendExecutor][Orchestrate] Self send, Self recv, Do nothing");
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_DEBUG("[InsCollAlgFactory][InsSendExecutor][Orchestrate] Self send, Other recv.");

    // 从linkMgr里面拿到 linkData
    const vector<LinkData> sendPath = linkMgr->GetLinks(remoteRank);
    CHK_PRT_RET(
        sendPath.size() == 0,
        HCCL_ERROR("[InsCollAlgFactory] Unable to obtain valid link, srcRank [%d], dstRank [%d].", myRank_, remoteRank),
        HcclResult::HCCL_E_INTERNAL);
    LinkData sendLinkData(sendPath[0]);
    if (sendLinkData.GetType() == PortDeploymentType::P2P && sendLinkData.GetLinkProtocol() == LinkProtocol::PCIE) {
        dmaMode_ = DmaMode::GET;
    }
    HCCL_DEBUG(
        "[InsCollAlgFactory][InsSendExecutor][Orchestrate] Total transfer data size [%llu], Max scratch buffer size "
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
            DataSlice inputBuffer(BufferType::INPUT, currentOffset, transferSize);
            DataSlice remoteOutputBuffer(BufferType::OUTPUT, currentOffset, transferSize);
            SlicesList sendSlicesList({inputBuffer}, {remoteOutputBuffer});
            DataInfo sendInfo(sendLinkData, sendSlicesList);
            CHK_RET(Send(sendInfo, insQue, 0, true, dmaMode_));
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
                "[InsCollAlgFactory][InsSendExecutor][Orchestrate] Send round [%u], transfer data size [%llu]",
                roundIdx, transferSize);
            // 根据本轮数据搬运量声明相关DataSlice
            DataSlice inputBuffer(BufferType::INPUT, currentOffset, transferSize);
            DataSlice scratchBuffer(BufferType::SCRATCH, 0, transferSize);
            DataSlice remoteScratchBuffer(BufferType::SCRATCH, 0, transferSize);
            // local copy
            CHK_RET(LocalCopy(insQue, inputBuffer, scratchBuffer));
            SlicesList sendSlicesList({scratchBuffer}, {remoteScratchBuffer});
            DataInfo sendInfo(sendLinkData, sendSlicesList);
            CHK_RET(Send(sendInfo, insQue, 0, true, dmaMode_));

            currentOffset = currentOffset + transferSize;
            resDataSize = resDataSize - transferSize;
            roundIdx = roundIdx + 1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// 注册
INS_REGISTER_IMPL(OpType::SEND, InsSend, InsSendExecutor);

} // namespace Hccl
