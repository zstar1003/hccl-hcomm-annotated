/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_recursive_hd.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherRecursiveHalvingDoubling::AllGatherRecursiveHalvingDoubling(const HcclDispatcher dispatcher)
    : RecursiveHalvingDoublingBase(dispatcher)
{}

AllGatherRecursiveHalvingDoubling::~AllGatherRecursiveHalvingDoubling() {}

// 服务器间allreduce的入口函数
HcclResult AllGatherRecursiveHalvingDoubling::RunAsync(
    const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "AllGatherRecursiveHalvingDoubling run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank,
        rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    HcclResult ret = HCCL_SUCCESS;

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return ret;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[AllGatherRecursiveHalvingDoubling][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank,
            links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    ret = CalcPartOneSizeAndBlockSize(rankSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllGatherRecursiveHalvingDoubling][RunAsync]Calculate part1Size[%u] "
            "And BlockSize[%u] Failed! rankSize[%u]",
            part1Size_, blockSize_, rankSize),
        ret);

    ret = CalculateSlices(dataBytes_, rankSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllGatherRecursiveHalvingDoubling][RunAsync]Calculate slices failed, "
            "dataBytes[%llu], rankSize[%u]",
            dataBytes_, rankSize),
        ret);

    CHK_RET(GatherInPartOneToEven(rank, links));

    CHK_RET(AllGatherInBlock(rank, rankSize, links));

    CHK_RET(GatherInPartOneToOdd(rank, links));

    HCCL_INFO("AllGatherRecursiveHalvingDoubling finished: rank[%u] finished", rank);
    return HCCL_SUCCESS;
}

HcclResult AllGatherRecursiveHalvingDoubling::CalculateSlices(u64 dataBytes, const u32 rankSize) const
{
    slices_.resize(blockSize_);
    u64 bytesPerSlice = dataBytes;
    u64 totalBytes = dataBytes * rankSize;
    u64 bytesLeft = totalBytes;
    u32 i = 0;
    while (bytesLeft > 0 && i < part1Size_ / 2) { // 除2计算part1在做完操作后block内slice数
        slices_[i].size = 2 * bytesPerSlice < bytesLeft ? 2 * bytesPerSlice : bytesLeft; // 乘2表示slice为part2两倍
        slices_[i].offset = totalBytes - bytesLeft;
        bytesLeft -= slices_[i].size;
        i++;
    }

    while (bytesLeft > 0) {
        slices_[i].size = bytesPerSlice < bytesLeft ? bytesPerSlice : bytesLeft;
        slices_[i].offset = totalBytes - bytesLeft;
        bytesLeft -= slices_[i].size;
        i++;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherRecursiveHalvingDoubling::GatherInPartOneToEven(u32 rank, const std::vector<LINK>& links)
{
    if (rank < part1Size_ && rank % 2 == 0) { // 模2判断奇偶性，从下一个rank的output收数据到output
        u32 peerRank = rank + 1;              // 加1计算下一个rank号
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
            DeviceMem gatherOutputMem = outputMem_.range(dataBytes_ * rank, dataBytes_);
            //  接收数据到本端的 output
            HCCL_DEBUG(
                "rank[%u] outputMem[%p] receive from PeerRank[%u] outputMem, Offset[%llu], Size[%llu]", rank,
                gatherOutputMem.ptr(), peerRank, baseOffset_ + dataBytes_ * rank, gatherOutputMem.size());

            ret = ExecuteRxSync(
                links[peerRank], UserMemType::OUTPUT_MEM, dataBytes_ * rank, gatherOutputMem.ptr(), dataBytes_,
                stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Gather][InPartOneToEven]rank[%u] rx sync from PeerRank[%u] "
                    "failed",
                    rank, peerRank),
                ret);
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InPartOneToEven]RxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) { // 模2判断奇偶性，向上一个rank发送数据
        u32 peerRank = rank - 1;                     // 减1计算上一个rank号
        //  发送到对端的output
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] tx ack from peerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            HCCL_DEBUG("[AllGatherRecursiveHalvingDoubling][GatherInPartOneToEven]peerRank is %u", peerRank);
            //  等待对端可以接收数据
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] rx ack from peerRank[%u] failed", rank, peerRank), ret);
            //  设置gather的发送内存范围
            DeviceMem gatherOutputMem = outputMem_.range(dataBytes_ * rank, dataBytes_);
            //  发送数据到对端的 output
            HCCL_DEBUG(
                "rank[%u] outputMem[%p] sends to PeerRank[%u] outputMem, Offset[%llu], Size[%llu]", rank,
                gatherOutputMem.ptr(), peerRank, baseOffset_ + dataBytes_ * rank, gatherOutputMem.size());

            ret = ExecuteTxSync(
                links[peerRank], UserMemType::OUTPUT_MEM, dataBytes_ * rank, gatherOutputMem.ptr(), dataBytes_,
                stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] tx sync to PeerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InPartOneToEven]TxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherRecursiveHalvingDoubling::GatherInPartOneToOdd(u32 rank, const std::vector<LINK>& links)
{
    if (rank < part1Size_ && rank % 2 == 0) { // 模2判断奇偶性，向下一个rank发送数据
        u32 peerRank = rank + 1;              // 加1计算下一个rank号
        //  发送到对端的output
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] tx ack from peerRank[%u] failed.", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            //  等待对端可以接收数据
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] rx ack from peerRank[%u] failed", rank, peerRank), ret);

            HCCL_DEBUG(
                "rank[%u] outputMem[%p] sends to PeerRank[%u] outputMem, Offset[%llu], Size[%llu]", rank,
                outputMem_.ptr(), peerRank, baseOffset_, outputMem_.size());
            ret = ExecuteTxSync(
                links[peerRank], UserMemType::OUTPUT_MEM, baseOffset_, outputMem_.ptr(), outputMem_.size(), stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] tx sync to PeerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InPartOneToOdd]TxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) { // 模2判断奇偶性，从上一个rank的output收数据到output
        u32 peerRank = rank - 1;                     // 减1计算上一个rank号
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            //  知会对端本人可以接收数据
            HcclResult ret = links[peerRank]->TxAck(stream_);
            //  等待对端可以接收数据
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] tx ack from peerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] rx ack from peerRank[%u] failed", rank, peerRank), ret);
            //  接收数据到本端的 output
            HCCL_DEBUG(
                "rank[%u] outputMem[%p] receive from PeerRank[%u] outputMem, Offset[%llu], "
                "Size[%llu]",
                rank, outputMem_.ptr(), peerRank, baseOffset_, outputMem_.size());
            ret = ExecuteRxSync(
                links[peerRank], UserMemType::OUTPUT_MEM, baseOffset_, outputMem_.ptr(), outputMem_.size(), stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] rx sync from PeerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InPartOneToOdd]RxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherRecursiveHalvingDoubling::AllGatherInBlock(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 rankInBlock = 0;
    if (rank < part1Size_ && (rank % 2) == 1) { // 模2余1代表当前rank在part1的奇数位置上，不参与block内的计算
        return HCCL_SUCCESS;
    } else if (rank < part1Size_ && (rank % 2) == 0) { // 模2余0代表当前rank在part1的偶数位置上，参与block内的计算
        rankInBlock = rank / 2;                        // 除2计算出在block内的rank号
    } else {
        rankInBlock = rank - part1Size_ / 2; // rank在part2中，用原始rank减part1除2，计算出在block内的rank号
    }

    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_GATHER_HALVING_DOUBLING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(blockSize_, UserMemType::OUTPUT_MEM, UserMemType::OUTPUT_MEM));
    CHK_RET(tempAlg->Prepare(
        outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    std::vector<LINK> subLinks;
    CHK_RET(BuildSubLinks(links, subLinks, rankSize));

    CHK_PRT_RET(
        subLinks.size() == 0,
        HCCL_ERROR("[AllGatherRecursiveHalvingDoubling][AllGatherInBlock]rank[%u] BuildSubLinks failed", rank),
        HCCL_E_PARA);

    CHK_RET(tempAlg->RunAsync(rankInBlock, blockSize_, subLinks));

    return HCCL_SUCCESS;
}

HcclResult AllGatherRecursiveHalvingDoubling::GetNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    u32 nslbRound = 0;
    u32 base = 1;
    const u32 minExponent = 1;
    HCCL_DEBUG("[AllGatherRecursiveHalvingDoubling]GetNslbAdjInfo begins");
    while ((base << nslbRound) <= rankSize) {
        nslbRound++;
    }
    if (nslbRound >= minExponent) {
        nslbRound = nslbRound - minExponent;
    }
    u32 nslbBlockSize = base << nslbRound;
    // 获取第一部分：rank数减block数乘2
    u32 nslbPart1Size = (rankSize - nslbBlockSize) * NSLBDP_ALL_GATHER_MOLD2;
    // 2的次幂场景下处理流程
    if (nslbPart1Size == 0) {
        u32 stepNum = 0;
        while ((rankSize >> (stepNum + 1)) != 0) {
            stepNum++;
        }
        for (u32 step = 0; step < stepNum; step++) {
            HCCL_DEBUG("[AllGatherRecursiveHalvingDoubling]current step is %u", step);
            u32 peerRankBitmask = 1 << (stepNum - step - 1);
            u32 peerRank = rank ^ peerRankBitmask;
            NslbDpAdjInfo adjInfoStep = {};
            u32 remoteuserRank = links[peerRank]->GetRemoteRank();
            adjInfoStep.dstLocalRankId = remoteuserRank;
            adjInfoStep.phaseId = step + 1;
            adjInfoStep.rev = 0;
            nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
            HCCL_DEBUG("[AllGatherRecursiveHalvingDoubling]current step %u success", step);
        }
        nslbAdjInfo.dstRankNum = stepNum;
        return HCCL_SUCCESS;
    }
    // 非2的次幂场景下，被合并部分的奇数rank处理流程
    if (rank < nslbPart1Size && rank % NSLBDP_ALL_GATHER_MOLD2 == 1) {
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
    // 针对合并后映射成2的次幂场景处理
    u32 rankInBlock = 0;
    if (rank < nslbPart1Size && (rank % NSLBDP_ALL_GATHER_MOLD2) == 0) {
        rankInBlock = rank / NSLBDP_ALL_GATHER_MOLD2; // 直接除以2即为本rank的在block内的排序
    } else {
        rankInBlock
            = rank
              - nslbPart1Size / NSLBDP_ALL_GATHER_MOLD2; // 通过rank减去part1除2的大小即不处于第一部分的block内rank号
    }
    std::vector<LINK> subLinks;
    std::vector<LINK>::const_iterator iter = links.begin();
    subLinks.resize(nslbBlockSize);
    for (u32 i = 0; i < rankSize; i++) {
        if (i < nslbPart1Size
            && (i % NSLBDP_ALL_GATHER_MOLD2) == 1) { // 模2余1代表当前rank在part1的奇数位置上，不参与block内的建链
            continue;
        } else if (i < nslbPart1Size && (i % NSLBDP_ALL_GATHER_MOLD2) == 0) { // 模2余0代表当前rank在part1的偶数位置上
            std::vector<LINK>::const_iterator niter = std::next(iter, i);
            if (niter != links.end()) {
                subLinks[i / NSLBDP_ALL_GATHER_MOLD2] = *niter;
            }
        } else {
            std::vector<LINK>::const_iterator niter = std::next(iter, i);
            if (niter != links.end()) {
                subLinks[i - nslbPart1Size / NSLBDP_ALL_GATHER_MOLD2] = *niter;
            }
        }
    }
    u32 stepNum = 0;
    while ((rankSize >> (stepNum + 1)) != 0) {
        stepNum++;
    }
    // 映射完成后针对以新的通信域进行邻接表获取
    for (u32 step = 0; step < stepNum; step++) {
        u32 peerRankBitmask = (1 << step);
        u32 peerRank = rankInBlock ^ peerRankBitmask;
        if (subLinks[peerRank] == nullptr) {
            continue;
        }
        NslbDpAdjInfo adjInfoStep = {};
        u32 remoteuserRank = subLinks[peerRank]->GetRemoteRank();
        HCCL_DEBUG("[AllGatherRecursiveHalvingDoubling][GetNslbAdjInfo]remoteuserRank is %u", remoteuserRank);
        adjInfoStep.dstLocalRankId = remoteuserRank;
        adjInfoStep.phaseId = step + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();

    if (nslbAdjInfo.nsAdjInfo.size() == 0) {
        return HCCL_SUCCESS;
    }
    // 上面处理完成后，紧接着处理合并部分的偶数rank同步到奇数rank增加phaseId
    if (rank < nslbPart1Size && rank % NSLBDP_ALL_GATHER_MOLD2 == 0) {
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
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, AllGatherRecursiveHalvingDoubling);
} // namespace hccl
