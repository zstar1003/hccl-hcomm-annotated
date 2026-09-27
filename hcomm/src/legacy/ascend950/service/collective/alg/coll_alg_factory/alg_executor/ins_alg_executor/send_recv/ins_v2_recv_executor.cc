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
#include "ins_v2_recv_executor.h"
#include "alg_data_trans_wrapper.h"

#include "hccl_aiv_utils.h"
#include "aiv_ins.h"
#include "executor_utils.h"

using namespace std;

namespace Hccl {
InsV2RecvExecutor::InsV2RecvExecutor() : InsCollAlgBase() {}

InsV2RecvExecutor::~InsV2RecvExecutor() {}

HcclResult InsV2RecvExecutor::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;

    if (numBlocksLimit < 1) {
        HCCL_ERROR("[InsV2RecvExecutor] core num[%u] is less than 1", numBlocksLimit);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    numBlocks = numBlocksLimit;
    HCCL_INFO("[InsV2RecvExecutor] Actually use core num[%u]", numBlocks);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsV2RecvExecutor::CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq)
{
    (void)rankGraph;
    (void)dataSize;
    (void)resReq;
    HCCL_ERROR("[InsCollAlgFactory][InsV2RecvExecutor][CalcResOffload] offload is not supported");
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult InsV2RecvExecutor::CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq)
{
    u32 linkNumBtwPeers = 1;
    algResReq.primQueueNum = 1;
    AlgTempResReq tempResReq;
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    if (static_cast<u32>(sendRecvRemoteRank_) > rankSize_ - 1) {
        HCCL_ERROR(
            "[InsCollAlgFactory][InsV2RecvExecutor][CalcRes] Rank[%d] get dest[%d] is invalid", myRank_,
            sendRecvRemoteRank_);
        return HcclResult::HCCL_E_PARA;
    }
    tempResReq.links[sendRecvRemoteRank_] = linkNumBtwPeers;
    uint32_t linkNum = GetPathsFromRankGraph(rankGraph, myRank_, sendRecvRemoteRank_).size();
    if (linkNum == 0) {
        HCCL_ERROR(
            "[InsCollAlgFactory][InsV2RecvExecutor][CalcRes] Rank[%d] get path num to dest[%d] is zero", myRank_,
            sendRecvRemoteRank_);
    }
    CHK_RET(CalcResLinks(myRank_, rankGraph, linkPriority_, tempResReq.links, algResReq.links));
    CHK_RET(CalcLinkInfo(myRank_, rankGraph, tempResReq.links, algResReq.levelRankPairs));
    return HcclResult::HCCL_SUCCESS;
}

// host
HcclResult InsV2RecvExecutor::Orchestrate(
    const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue)
{
    HCCL_DEBUG("[InsCollAlgFactory][InsV2RecvExecutor][Orchestrate] Begin to Generate Instruction Queue for RECV.");
    CHK_RET(Init(op, params, insQue));
    // 从集合通信算子op中获得remote rank和data type/count相关信息
    RankId remoteRank = op.sendRecvRemoteRank;
    u32 dataElemSize = DATA_TYPE_SIZE_MAP.at(op.dataType);
    u64 totalDataSize = static_cast<u64>(dataElemSize) * op.dataCount;

    // 判断是不是自发自收这种情况，若是，则什么都不做，直接返回
    if (myRank_ == remoteRank) {
        HCCL_WARNING("[InsCollAlgFactory][InsV2RecvExecutor][Orchestrate] Self send, Self recv, Do nothing");
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_DEBUG("[InsCollAlgFactory][InsV2RecvExecutor][Orchestrate] Other send, Self recv");

    // 从virtualTopo里面拿到 link，转成LinkData格式
    const std::vector<NetInstance::Path> recvPath = GetPathsFromRankGraph(rankGraph, myRank_, remoteRank);
    CHK_PRT_RET(
        recvPath.size() == 0,
        HCCL_ERROR("[InsCollAlgFactory] Unable to obtain valid link, srcRank [%d], dstRank [%d].", myRank_, remoteRank),
        HcclResult::HCCL_E_INTERNAL);
    LinkData recvLinkData(recvPath[0]);
    HCCL_DEBUG(
        "[InsCollAlgFactory][InsV2RecvExecutor][Orchestrate] Total transfer data size [%llu], Max scratch buffer size "
        "[%u].",
        totalDataSize, params.maxTmpMemSize);

    // 模式判断
    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_ERROR("[InsCollAlgFactory][InsV2RecvExecutor][Orchestrate] offload is not supported");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OPBASE Mode for HOST.", myRank_);

        CHK_RET(ExecAiv(op, params, recvLinkData, insQue));
        return HcclResult::HCCL_SUCCESS;
    }
    return HcclResult::HCCL_SUCCESS;
}

// aicpu
HcclResult InsV2RecvExecutor::Orchestrate(
    const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
    InsQuePtr insQue)
{
    (void)topoInfo;
    HCCL_DEBUG(
        "[InsCollAlgFactory][InsV2RecvExecutor][Orchestrate] Begin to Generate Instruction Queue for RECV AICPU mode.");
    CHK_RET(Init(op, params, insQue));
    // 从集合通信算子op中获得remote rank和data type/count相关信息
    RankId remoteRank = op.sendRecvRemoteRank;
    u32 dataElemSize = DATA_TYPE_SIZE_MAP.at(op.dataType);
    u64 totalDataSize = static_cast<u64>(dataElemSize) * op.dataCount;

    // 判断是不是自发自收这种情况，若是，则什么都不做，直接返回
    if (myRank_ == remoteRank) {
        HCCL_WARNING("[InsCollAlgFactory][InsV2RecvExecutor][Orchestrate] Self send, Self recv, Do nothing");
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_DEBUG("[InsCollAlgFactory][InsV2RecvExecutor][Orchestrate] Other send, Self recv.");

    // 从linkMgr里面拿到 linkData
    const vector<LinkData> recvPath = linkMgr->GetLinks(remoteRank);
    CHK_PRT_RET(
        recvPath.size() == 0,
        HCCL_ERROR("[InsCollAlgFactory] Unable to obtain valid link, srcRank [%d], dstRank [%d].", myRank_, remoteRank),
        HcclResult::HCCL_E_INTERNAL);
    LinkData recvLinkData(recvPath[0]);
    HCCL_DEBUG(
        "[InsCollAlgFactory][InsV2RecvExecutor][Orchestrate] Total transfer data size [%llu], Max scratch buffer size "
        "[%u].",
        totalDataSize, params.maxTmpMemSize);

    // 模式判断
    if (opMode_ == OpMode::OFFLOAD) {
        HCCL_ERROR("[InsCollAlgFactory][InsV2RecvExecutor][Orchestrate] offload is not supported");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    } else {
        HCCL_DEBUG("[InsCollAlgFactory] Rank[%d], Generating Instruction Queues in OPBASE Mode for HOST.", myRank_);

        CHK_RET(ExecAiv(op, params, recvLinkData, insQue));
        return HcclResult::HCCL_SUCCESS;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsV2RecvExecutor::ExecAiv(
    const CollAlgOperator& op, const CollAlgParams& params, LinkData& recvLinkData, InsQuePtr insQue)
{
    HCCL_INFO(
        "[InsV2RecvExecutor][ExecAiv] start: rank is %d, count is %u, dataType is %u, srcRank is %d", myRank_,
        op.dataCount, static_cast<u32>(op.dataType), op.sendRecvRemoteRank);

    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxScratchDataSize = std::min(transportBoundDataSize, params.maxTmpMemSize);
    u32 dataElemSize = DATA_TYPE_SIZE_MAP.at(op.dataType);
    u64 maxScratchDataCount = maxScratchDataSize / dataElemSize;
    CHK_PRT_RET(
        maxScratchDataCount == 0, HCCL_ERROR("[InsV2RecvExecutor][Orchestrate] maxScratchDataCount is 0"),
        HCCL_E_INTERNAL);

    std::vector<LinkData> allLinks;
    allLinks.emplace_back(recvLinkData);

    u64 loopTimes = op.dataCount / maxScratchDataCount + static_cast<u64>(op.dataCount % maxScratchDataCount != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        sliceId_++; // 自动增长sliceId，传入aivTag
        u64 currDataCount = (loop == loopTimes - 1) ? op.dataCount - processedDataCount : maxScratchDataCount;
        HCCL_INFO(
            "[InsV2RecvExecutor][ExecAiv] myRank[%d], loop[%llu] sliceId_[%llu] currDataCount[%llu], "
            "processedDataCount[%llu]",
            myRank_, loop, sliceId_, currDataCount, processedDataCount);

        AivOpArgs aivRecvArgs;
        aivRecvArgs.cmdType = HcclCMDType::HCCL_CMD_RECEIVE;
        aivRecvArgs.input = 0;
        aivRecvArgs.output = processedDataCount * dataElemSize;
        aivRecvArgs.rank = u32(myRank_);
        aivRecvArgs.sendRecvRemoteRank = op.sendRecvRemoteRank;
        aivRecvArgs.rankSize = rankSize_;
        aivRecvArgs.count = currDataCount; // 需要传输的数据量
        aivRecvArgs.dataType = op.dataType;
        aivRecvArgs.aivTag = sliceId_; // 传入aivTag，Launch时重新组装为aivTag
        aivRecvArgs.isOpBase = (opMode_ == OpMode::OPBASE);
        aivRecvArgs.xRankSize = rankSize_;
        aivRecvArgs.yRankSize = 0;
        aivRecvArgs.zRankSize = 0;
        CHK_RET(CalNumBlocks(aivRecvArgs.numBlocks, 0, op.numBlocksLimit));

        aivRecvArgs.inputSliceStride = 0;
        aivRecvArgs.outputSliceStride = 0;
        aivRecvArgs.repeatNum = 1; // 不重复
        aivRecvArgs.inputRepeatStride = 0;
        aivRecvArgs.outputRepeatStride = 0;

        std::unique_ptr<Instruction> aivInsRecvMesh1D = std::make_unique<AivInstruction>(allLinks, aivRecvArgs);

        insQue->Append(std::move(aivInsRecvMesh1D));
        processedDataCount += currDataCount;
    }

    HCCL_INFO("[InsV2RecvExecutor][ExecAiv] end: rank[%d]", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

// 注册
INS_REGISTER_IMPL(OpType::RECV, AivRecv, InsV2RecvExecutor);

} // namespace Hccl
