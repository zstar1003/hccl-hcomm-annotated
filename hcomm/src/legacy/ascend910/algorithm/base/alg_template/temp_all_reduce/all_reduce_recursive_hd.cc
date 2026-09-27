/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "all_reduce_recursive_hd.h"

namespace hccl {
AllReduceRecursiveHalvingDoubling::AllReduceRecursiveHalvingDoubling(const HcclDispatcher dispatcher)
    : RecursiveHalvingDoublingBase(dispatcher)
{}

AllReduceRecursiveHalvingDoubling::~AllReduceRecursiveHalvingDoubling() {}

HcclResult AllReduceRecursiveHalvingDoubling::Prepare(u64 reduceAttrBitMap, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    reduceAttr = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

// 算法的主入口
HcclResult
AllReduceRecursiveHalvingDoubling::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_RET(PrepareRunAsync(rank, rankSize, links));
    CHK_PRT_RET(
        rankSize == 1,
        HCCL_INFO(
            "[AllReduceRecursiveHalvingDoubling][RunAsync]"
            "rankSize[%u], do nothing.",
            rankSize),
        HCCL_SUCCESS);

    CHK_RET(ReduceInPartOne(rank, links));

    CHK_RET(ReduceScatterInBlock(rank, rankSize, links));

    CHK_RET(AllGatherInBlock(rank, rankSize, links));

    CHK_RET(GatherInPartOne(rank, links));

    HCCL_INFO("AllReduceRecursiveHalvingDoubling finished: rank[%u] finished", rank);
    return HCCL_SUCCESS;
}

HcclResult AllReduceRecursiveHalvingDoubling::RunAsyncStaged(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, RunStage stage)
{
    CHK_PRT_RET(
        rankSize == 1 && stage != RunStage::RUN_PREPARE,
        HCCL_INFO(
            "[AllReduceRecursiveHalvingDoubling][RunAsyncStaged] rankSize[%u], stage[%d], do nothing.", rankSize,
            stage),
        HCCL_SUCCESS);
    switch (stage) {
        case RunStage::RUN_PREPARE:
            CHK_RET(PrepareRunAsync(rank, rankSize, links));
            break;
        case RunStage::RUN_REDUCE_SCATTER:
            // 先执行reducescater
            CHK_RET(ReduceInPartOne(rank, links));
            CHK_RET(ReduceScatterInBlock(rank, rankSize, links));
            break;
        case RunStage::RUN_ALLGATHER:
            // 再执行allgather
            CHK_RET(AllGatherInBlock(rank, rankSize, links));
            CHK_RET(GatherInPartOne(rank, links));
            break;
        default:
            HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][RunAsyncStaged]stage[%d] is not supported", stage);
            return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO(
        "AllReduceRecursiveHalvingDoubling RunAsyncStaged stage[%d] finished: rank[%u] ranksize[%u]", stage, rank,
        rankSize);
    return HCCL_SUCCESS;
}

HcclResult
AllReduceRecursiveHalvingDoubling::PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR(
            "[AllReduceRecursiveHalvingDoubling][RunAsync]rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO(
        "AllReduceRecursiveHalvingDoubling run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank,
        rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    HcclResult ret = HCCL_SUCCESS;

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return ret;
    }

    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr));
    CHK_SMART_PTR_NULL(reducerInfo_);

    bool bRetSize = (links.size() < rankSize);
    CHK_PRT_RET(
        bRetSize,
        HCCL_ERROR(
            "[AllReduceRecursiveHalvingDoubling][RunAsync]rank[%u] linksize[%llu] is "
            "error",
            rank, links.size()),
        HCCL_E_INTERNAL);

    CHK_RET(CalcPartOneSizeAndBlockSize(rankSize));

    u32 bytesPerData = SIZE_TABLE[dataType_];
    u64 dataBytes = count_ * bytesPerData;
    CHK_RET(CalculateSlices(dataBytes));
    HCCL_INFO("AllReduceRecursiveHalvingDoubling PrepareRunAsync finished: rank[%u] finished", rank);
    return HCCL_SUCCESS;
}

HcclResult AllReduceRecursiveHalvingDoubling::ReduceInPartOne(u32 rank, const std::vector<LINK>& links)
{
    // 本rank属于第一部分，并且是2的整数倍
    if (rank < part1Size_ && rank % 2 == 0) { // 1.从下一个rank接收数据到output，2. reduce到本rank的input
        u32 peerRank = rank + 1;
        HCCL_DEBUG(
            "rank[%u] outputMem receives from PeerRank[%u] inputMem, Offset[%llu], Size[%llu]", rank, peerRank,
            baseOffset_, outputMem_.size());

        if (peerRank < links.size()) {
            const LINK& link = links[peerRank];
            CHK_SMART_PTR_NULL(link);

            HcclResult ret = link->TxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] tx ack from peerRank[%u] failed", rank, peerRank), ret);
            ret = link->RxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] rx ack from peerRank[%u] failed", rank, peerRank), ret);
            //  接收数据到本端的 output
            HCCL_DEBUG("send mem[%p] size[%llu] to peerRank[%u]", outputMem_.ptr(), outputMem_.size(), peerRank);
            ret = link->TxAsync(UserMemType::INPUT_MEM, baseOffset_, outputMem_.ptr(), 0, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Reduce][InPartOneToEven]TxAsync: tx async size[%llu] "
                    "failed",
                    0),
                ret);
            CHK_RET(reducerInfo_->run(dispatcher_, link, baseOffset_, inputMem_, inputMem_, outputMem_, stream_));
            ret = link->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]RxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) { //  向上一个rank的output发数据 2
        u32 peerRank = rank - 1;

        if (peerRank < links.size()) {
            const LINK& link = links[peerRank];
            CHK_SMART_PTR_NULL(link);
            HcclResult ret = link->TxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] tx ack from peerRank[%u] failed", rank, peerRank), ret);
            ret = link->RxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] rx ack from peerRank[%u] failed", rank, peerRank), ret);
            //  发送到对端的output
            HCCL_DEBUG(
                "rank[%u] sends inputMem[%p] to PeerRank[%u] Offset[%llu], Size[%llu]", rank, inputMem_.ptr(), peerRank,
                baseOffset_, inputMem_.size());
            ret = senderInfo_->run(link, baseOffset_, inputMem_, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]tx sync to peerRank[%u] failed", peerRank), ret);
            ret = link->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_, inputMem_.ptr(), 0, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[AlgTemplateBase][ExecuteTxSync]ExecuteTxSync: rx async size[%llu] failed", 0), ret);
            ret = link->DataReceivedAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[AlgTemplateBase][ExecuteTxSync]ExecuteTxSync: data received ack failed"), ret);
            ret = link->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]TxWaitDone failed"), ret);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult
AllReduceRecursiveHalvingDoubling::ReduceScatterInBlock(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 rankInBlock = 0;
    if (rank < part1Size_ && (rank % 2) == 1) { // 模2判断奇偶性，本rank处于第一部分，并且为奇数rank
        return HCCL_SUCCESS;
    } else if (rank < part1Size_ && (rank % 2) == 0) { // 模2判断奇偶性，本rank 处于第一部分，并且为偶数rank
        rankInBlock = rank / 2;                        // 除2计算block内的rank值
    } else {                                           // 本rank不属于第一部分
        rankInBlock = rank - part1Size_ / 2;           // 除2计算block内的part1的范围
    }
    // 直接调用block的reducscatterhd算法
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_HD, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(
        inputMem_, outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_,
        blockSize_, reduceAttr, UserMemType::INPUT_MEM, UserMemType::OUTPUT_MEM));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    // 重新建立reducscatterscatter需要的链接
    std::vector<LINK> subLinks;
    CHK_RET(BuildSubLinks(links, subLinks, rankSize));

    CHK_PRT_RET(
        subLinks.size() == 0,
        HCCL_ERROR(
            "[AllReduceRecursiveHalvingDoubling][ReduceScatterInBlock]rank[%u] BuildSubLinks "
            "failed",
            rank),
        HCCL_E_PARA);
    CHK_RET(tempAlg->RunAsync(rankInBlock, blockSize_, subLinks));
    return HCCL_SUCCESS;
}

HcclResult AllReduceRecursiveHalvingDoubling::AllGatherInBlock(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 rankInBlock = 0;
    if (rank < part1Size_ && (rank % 2) == 1) { // 模2判断奇偶性，本rank 处于第一部分，并且为奇数rank
        return HCCL_SUCCESS;
    } else if (rank < part1Size_ && (rank % 2) == 0) { // 模2判断奇偶性，本rank 处于第一部分，并且为偶数rank
        rankInBlock = rank / 2;                        // 在block内的rank为实际rank除以2
    } else {
        rankInBlock = rank - part1Size_ / 2; // 除2计算block内的part1的范围
    }
    // 直接调用block的allgatherhd算法
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_GATHER_HALVING_DOUBLING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(blockSize_, UserMemType::OUTPUT_MEM, UserMemType::OUTPUT_MEM));
    CHK_RET(tempAlg->Prepare(
        outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    // 重新建立allgather需要的链接
    std::vector<LINK> subLinks;
    CHK_RET(BuildSubLinks(links, subLinks, rankSize));

    CHK_PRT_RET(
        subLinks.size() == 0,
        HCCL_ERROR(
            "[AllReduceRecursiveHalvingDoubling][AllGatherInBlock]rank[%u] build sub "
            "links failed",
            rank),
        HCCL_E_PARA);

    CHK_RET(tempAlg->RunAsync(rankInBlock, blockSize_, subLinks));
    return HCCL_SUCCESS;
}

HcclResult AllReduceRecursiveHalvingDoubling::GatherInPartOne(u32 rank, const std::vector<LINK>& links)
{
    if (rank < part1Size_ && rank % 2 == 0) { // 模2判断奇偶性，本rank 处于第一部分，并且为偶数rank
        u32 peerRank = rank + 1;
        //  发送到对端的output
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] tx ack from peerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] rx ack from peerRank[%u] failed", rank, peerRank), ret);
            HCCL_DEBUG(
                "rank[%u] outputMem[%p] sends to peerRank[%u] outputMem, offset[%llu], size[%llu]", rank,
                outputMem_.ptr(), peerRank, baseOffset_, outputMem_.size());
            ret = ExecuteTxSync(
                links[peerRank], UserMemType::OUTPUT_MEM, baseOffset_, outputMem_.ptr(), outputMem_.size(), stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceRecursiveHalvingDoubling][GatherInPartOne]rank[%u] tx "
                    "sync to PeerRank[%u] failed",
                    rank, peerRank),
                ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][GatherInPartOne]TxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) { // 模2判断奇偶性，本rank 处于第一部分，并且为奇数rank
        u32 peerRank = rank - 1;
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] tx ack from peerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] rx ack from peerRank[%u] failed", rank, peerRank), ret);
            // 等待对端可以接收数据
            HCCL_DEBUG(
                "rank[%u] outputMem[%p] receive from PeerRank[%u] outputMem, Offset[%llu], "
                "Size[%llu]",
                rank, outputMem_.ptr(), peerRank, baseOffset_, outputMem_.size());
            ret = ExecuteRxSync(
                links[peerRank], UserMemType::OUTPUT_MEM, baseOffset_, outputMem_.ptr(), outputMem_.size(), stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceRecursiveHalvingDoubling][GatherInPartOne]rank[%u] rx "
                    "sync from PeerRank[%u] failed",
                    rank, peerRank),
                ret);
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][GatherInPartOne]RxWaitDone failed"), ret);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AllReduceRecursiveHalvingDoubling::GetCommonNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) const
{
    u32 stepNum = 0;
    while ((rankSize >> (stepNum + 1)) != 0) {
        stepNum++;
    }
    // 执行reducescatter流程
    for (u32 step = 0; step < stepNum; step++) {
        u32 peerRankBitmask = 1 << (stepNum - step - 1);
        u32 peerRank = rank ^ peerRankBitmask;
        NslbDpAdjInfo adjInfoStep = {};
        u32 remoteuserRank = links[peerRank]->GetRemoteRank();
        adjInfoStep.dstLocalRankId = remoteuserRank;
        adjInfoStep.phaseId = step + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    u32 begin = stepNum;
    // 后续执行allgather流程
    for (u32 step = 0; step < stepNum; step++) {
        u32 peerRankBitmask = (1 << step);
        u32 peerRank = rank ^ peerRankBitmask;
        NslbDpAdjInfo adjInfoStep = {};
        u32 remoteuserRank = links[peerRank]->GetRemoteRank();
        adjInfoStep.dstLocalRankId = remoteuserRank;
        adjInfoStep.phaseId = step + begin + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();
    return HCCL_SUCCESS;
}
HcclResult AllReduceRecursiveHalvingDoubling::GetOddNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) const
{
    (void)rankSize;
    u32 peerRank = rank - 1;
    if (peerRank < links.size()) {
        NslbDpAdjInfo adjInfoStep = {};
        adjInfoStep.dstLocalRankId = links[peerRank]->GetRemoteRank();
        adjInfoStep.phaseId = 1;
        adjInfoStep.rev = 0;
        HCCL_INFO("AllGatherHDR-nslb: peerRank[%u]", peerRank);
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
        nslbAdjInfo.dstRankNum = 1;
    }
    return HCCL_SUCCESS;
}
HcclResult AllReduceRecursiveHalvingDoubling::GetNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    u32 nslbRound = 0;
    u32 base = 1;
    const u32 minExponent = 1;
    while ((base << nslbRound) <= rankSize) {
        nslbRound++;
    }
    if (nslbRound >= minExponent) {
        nslbRound = nslbRound - minExponent;
    }
    u32 nslbBlockSize = base << nslbRound;
    // 获取第一部分：rank数减block数乘2
    u32 nslbPart1Size = (rankSize - nslbBlockSize) * NSLBDP_ALL_REDUCE_MOLD2;
    // 2的次幂场景下处理流程
    if (nslbPart1Size == 0) {
        GetCommonNslbAdjInfo(rank, rankSize, links, nslbAdjInfo);
        return HCCL_SUCCESS;
    }
    // 非2的次幂场景下，被合并部分的奇数rank处理流程
    if (rank < nslbPart1Size && rank % NSLBDP_ALL_REDUCE_MOLD2 == 1) {
        GetOddNslbAdjInfo(rank, rankSize, links, nslbAdjInfo);
        return HCCL_SUCCESS;
    }
    // 针对合并后映射成2的次幂场景处理
    u32 rankInBlock = 0;
    if (rank < nslbPart1Size && (rank % NSLBDP_ALL_REDUCE_MOLD2) == 0) {
        rankInBlock = rank / NSLBDP_ALL_REDUCE_MOLD2; // 直接除以2即为本rank的在block内的排序
    } else {
        rankInBlock
            = rank
              - nslbPart1Size / NSLBDP_ALL_REDUCE_MOLD2; // 通过rank减去part1除2的大小即不处于第一部分的block内rank号
    }
    std::vector<LINK> subLinks;
    std::vector<LINK>::const_iterator iter = links.begin();
    subLinks.resize(nslbBlockSize);
    for (u32 i = 0; i < rankSize; i++) {
        if (i < nslbPart1Size
            && (i % NSLBDP_ALL_REDUCE_MOLD2) == 1) { // 模2余1代表当前rank在part1的奇数位置上，不参与block内的建链
            continue;
        } else if (i < nslbPart1Size && (i % NSLBDP_ALL_REDUCE_MOLD2) == 0) { // 模2余0代表当前rank在part1的偶数位置上
            std::vector<LINK>::const_iterator niter = std::next(iter, i);
            if (niter != links.end()) {
                subLinks[i / NSLBDP_ALL_REDUCE_MOLD2] = *niter;
            }
        } else {
            std::vector<LINK>::const_iterator niter = std::next(iter, i);
            if (niter != links.end()) {
                subLinks[i - nslbPart1Size / NSLBDP_ALL_REDUCE_MOLD2] = *niter;
            }
        }
    }
    u32 stepNum = 0;
    while ((rankSize >> (stepNum + 1)) != 0) {
        stepNum++;
    }
    // 映射完成后针对以新的通信域进行邻接表获取
    u32 begin = 1;
    for (u32 step = 0; step < stepNum; step++) {
        u32 peerRankBitmask = 1 << (stepNum - step - 1);
        u32 peerRank = rankInBlock ^ peerRankBitmask;
        if (subLinks[peerRank] == nullptr) {
            continue;
        }
        NslbDpAdjInfo adjInfoStep = {};
        u32 remoteuserRank = subLinks[peerRank]->GetRemoteRank();
        adjInfoStep.dstLocalRankId = remoteuserRank;
        adjInfoStep.phaseId = step + 1 + begin;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    nslbAdjInfo.dstRankNum = stepNum;

    if (nslbAdjInfo.nsAdjInfo.size() == 0) {
        return HCCL_SUCCESS;
    }
    // 上面处理完成后，紧接着处理合并部分的偶数rank同步到奇数rank增加phaseId
    if (rank < nslbPart1Size && rank % NSLBDP_ALL_REDUCE_MOLD2 == 0) {
        u32 peerRank = rank + 1;
        uint16_t phaseSize = nslbAdjInfo.nsAdjInfo.size();
        if (peerRank < links.size()) {
            NslbDpAdjInfo adjInfoStep = {};
            adjInfoStep.dstLocalRankId = links[peerRank]->GetRemoteRank();
            adjInfoStep.phaseId = nslbAdjInfo.nsAdjInfo[phaseSize - 1].phaseId + 1;
            adjInfoStep.rev = 0;
            HCCL_INFO("Scatter-nslb: peerRank[%u]", peerRank);
            nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
            nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();
        }
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, AllReduceRecursiveHalvingDoubling);
} // namespace hccl
