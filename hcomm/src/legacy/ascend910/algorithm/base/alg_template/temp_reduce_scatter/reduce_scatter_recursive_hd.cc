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
#include "reduce_scatter_recursive_hd.h"

namespace hccl {
ReduceScatterRecursiveHalvingDoubling::ReduceScatterRecursiveHalvingDoubling(const HcclDispatcher dispatcher)
    : RecursiveHalvingDoublingBase(dispatcher)
{}

ReduceScatterRecursiveHalvingDoubling::~ReduceScatterRecursiveHalvingDoubling() {}

HcclResult ReduceScatterRecursiveHalvingDoubling::Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo)
{
    (void)opInfo;
    reduceAttr = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

// reducescatter recursiveHD 入口函数
HcclResult ReduceScatterRecursiveHalvingDoubling::RunAsync(
    const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize, inputMem_.ptr(),
        outputMem_.ptr(), count_);
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR(
            "[ReduceScatterRecursiveHalvingDoubling][RunAsync]rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }

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
            "[ReduceScatterRecursiveHalvingDoubling][RunAsync]rank[%u] linksize[%llu] is "
            "error",
            rank, links.size()),
        HCCL_E_INTERNAL);

    ret = CalcPartOneSizeAndBlockSize(rankSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ReduceScatterRecursiveHalvingDoubling][RunAsync]calculate part1size[%u] "
            "and blocksize[%u] Failed! rankSize[%u]",
            part1Size_, blockSize_, rankSize),
        ret);

    HCCL_DEBUG("rank[%u] calculate part1size[%u] blocksize[%u] ranksize[%u]", rank, part1Size_, blockSize_, rankSize);

    CHK_RET(ReduceInPartOne(rank, links));

    CHK_RET(CalculateSlices(dataBytes_, rankSize));

    CHK_RET(ReduceScatterInBlock(rank, rankSize, links));

    CHK_RET(ScatterInPartOne(rank, rankSize, links));

    HCCL_INFO("ReduceScatterRecursiveHalvingDoubling finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRecursiveHalvingDoubling::CalculateSlices(u64 dataBytes, const u32 rankSize) const
{
    CHK_PRT_RET((blockSize_ == 0), HCCL_ERROR("[Calculate][Slices]calculate_slices para error"), HCCL_E_INTERNAL);

    slices_.resize(blockSize_);
    u64 bytesPerSlice = dataBytes / rankSize; // input大小 / server数 = 服务器内rank数*count (4p mesh以4*count为粒度)
    u32 i = 0;
    u32 halfPart1Size = (part1Size_ / 2); // 除2计算一半part1的大小

    /* 先给属于part1的block rank分配slice。每个rank有两份数据 */
    while (i < halfPart1Size) {
        slices_[i].size = 2 * bytesPerSlice;       // 乘2计算2倍数据大小
        slices_[i].offset = i * 2 * bytesPerSlice; // 乘2计算2倍数据大小
        i++;
    }

    /* 再给剩余的block rank分配slice。每个rank有一份数据 */
    while (i < blockSize_) {
        slices_[i].size = bytesPerSlice;
        slices_[i].offset = (i * bytesPerSlice) + (halfPart1Size * bytesPerSlice);
        i++;
    }

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRecursiveHalvingDoubling::ReduceInPartOne(u32 rank, const std::vector<LINK>& links)
{
    if (rank < part1Size_ && rank % 2 == 0) { // 模2判断奇偶性，rank属于第一部分，并且为偶数rank
        u32 peerRank = rank + 1;
        HCCL_DEBUG(
            "rank[%u] outputmem receives from peerrank[%u] inputmem, offset[%llu], size[%llu]", rank, peerRank,
            baseOffset_, scratchMem_.size());
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            // 接收数据到本端的input
            HCCL_DEBUG("send mem[%p] size[%llu] to peerank[%u]", scratchMem_.ptr(), scratchMem_.size(), peerRank);
            ret = links[peerRank]->TxAsync(UserMemType::INPUT_MEM, baseOffset_, scratchMem_.ptr(), 0, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Reduce][InPartOneToEven]TxAsync: tx async size[%llu] "
                    "failed",
                    0),
                ret);
            CHK_RET(reducerInfo_->run(
                dispatcher_, links[peerRank], baseOffset_, inputMem_, inputMem_, scratchMem_, stream_));
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]RxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) { //  向上一个rank的input发数据 2
        u32 peerRank = rank - 1;
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            //  发送到对端的input
            HCCL_DEBUG(
                "rank[%u] sends inputMem[%p] to peerrank[%u] offset[%llu], size[%llu]", rank, inputMem_.ptr(), peerRank,
                baseOffset_, inputMem_.size());
            ret = senderInfo_->run(links[peerRank], baseOffset_, inputMem_, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]tx sync to peerank[%u] failed", peerRank), ret);
            ret = links[peerRank]->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_, inputMem_.ptr(), 0, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[AlgTemplateBase][ExecuteTxSync]ExecuteTxSync: rx async size[%llu] failed", 0), ret);
            ret = links[peerRank]->DataReceivedAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[AlgTemplateBase][ExecuteTxSync]ExecuteTxSync: data received ack failed"), ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]TxWaitDone failed"), ret);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterRecursiveHalvingDoubling::ReduceScatterInBlock(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 rankInBlock = 0;
    if (rank < part1Size_ && (rank % 2) == 1) { // rank号对2求余，rank为奇数
        return HCCL_SUCCESS;
    } else if (rank < part1Size_ && (rank % 2) == 0) { // rank对2求余，rank为偶数
        rankInBlock = rank / 2;                        // 直接除以2即为本rank的在block内的排序
    } else {
        rankInBlock = rank - part1Size_ / 2; // 通过rank减去part1除2的大小即不处于第一部分的block内rank号
    }

    std::unique_ptr<AlgTemplateBase> executor
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_HD, dispatcher_);
    CHK_SMART_PTR_NULL(executor);
    CHK_RET(executor->Prepare(
        inputMem_, inputMem_, scratchMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_,
        blockSize_, reduceAttr, UserMemType::INPUT_MEM, UserMemType::OUTPUT_MEM));

    CHK_RET(executor->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    std::vector<LINK> subLinks;
    CHK_RET(BuildSubLinks(links, subLinks, rankSize));

    CHK_PRT_RET(
        subLinks.size() == 0,
        HCCL_ERROR(
            "[ReduceScatterRecursiveHalvingDoubling][ReduceScatterInBlock]rank[%u] "
            "build sub links failed",
            rank),
        HCCL_E_PARA);
    CHK_RET(executor->RunAsync(rankInBlock, blockSize_, subLinks));
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterRecursiveHalvingDoubling::ScatterInPartOne(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 bytesPerData = DataUnitSize(dataType_);
    u64 dataBytes = count_ * bytesPerData;
    u64 bytesPerSlice = dataBytes / rankSize;

    if (rank < part1Size_ && rank % 2 == 0) { // 模2计算奇偶性，偶数rank把自己第二份数据给下一个奇数rank
        u32 peerRank = rank + 1;
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            //  发送到对端的input
            HCCL_DEBUG(
                "rank[%u] sends inputmem[%p] to peerrank[%u] Offset[%llu], Size[%llu]", rank, inputMem_.ptr(), peerRank,
                baseOffset_, inputMem_.size());

            u64 offset = peerRank * bytesPerSlice; // 计算对端rank的slice偏移
            void* srcAddr = reinterpret_cast<s8*>(inputMem_.ptr()) + offset;
            ret = ExecuteTxSync(
                links[peerRank], UserMemType::INPUT_MEM, offset + baseOffset_, srcAddr, bytesPerSlice, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[Scatter][InPartOne]tx sync to peerRank[%u] failed", peerRank), ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Scatter][InPartOne]TxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) { // 模2计算奇偶性，奇数rank接收偶数rank发过来下半份的数据
        u32 peerRank = rank - 1;
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            //  接收数据到本端的 inputMem_
            HCCL_DEBUG("rx mem[%p] size[%llu] from peerank[%u]", inputMem_.ptr(), inputMem_.size(), peerRank);

            u64 offset = rank * bytesPerSlice; // 本rank slice偏移
            void* dstAddr = reinterpret_cast<s8*>(inputMem_.ptr()) + offset;
            ret = ExecuteRxSync(
                links[peerRank], UserMemType::INPUT_MEM, offset + baseOffset_, dstAddr, bytesPerSlice, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[Scatter][InPartOne]rx sync from peerRank[%u] failed", peerRank), ret);
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Scatter][InPartOne]RxWaitDone failed"), ret);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRecursiveHalvingDoubling::GetNslbAdjInfo(
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
    u32 nslbPart1Size = (rankSize - nslbBlockSize) * 2;
    // 2的次幂场景下处理流程
    if (nslbPart1Size == 0) {
        u32 stepNum = 0;
        while ((rankSize >> (stepNum + 1)) != 0) {
            stepNum++;
        }
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
        nslbAdjInfo.dstRankNum = stepNum;
        return HCCL_SUCCESS;
    }
    // 非2的次幂场景下，被合并部分的奇数rank处理流程
    if (rank < nslbPart1Size && rank % NSLBDP_REDUCE_SCATTER_MOLD2 == 1) {
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
    if (rank < nslbPart1Size && (rank % NSLBDP_REDUCE_SCATTER_MOLD2) == 0) {
        rankInBlock = rank / NSLBDP_REDUCE_SCATTER_MOLD2; // 直接除以2即为本rank的在block内的排序
    } else {
        rankInBlock = rank
                      - nslbPart1Size
                            / NSLBDP_REDUCE_SCATTER_MOLD2; // 通过rank减去part1除2的大小即不处于第一部分的block内rank号
    }
    std::vector<LINK> subLinks;
    std::vector<LINK>::const_iterator iter = links.begin();
    subLinks.resize(nslbBlockSize);
    for (u32 i = 0; i < rankSize; i++) {
        if (i < nslbPart1Size
            && (i % NSLBDP_REDUCE_SCATTER_MOLD2) == 1) { // 模2余1代表当前rank在part1的奇数位置上，不参与block内的建链
            continue;
        } else if (
            i < nslbPart1Size && (i % NSLBDP_REDUCE_SCATTER_MOLD2) == 0) { // 模2余0代表当前rank在part1的偶数位置上
            std::vector<LINK>::const_iterator niter = std::next(iter, i);
            if (niter != links.end()) {
                subLinks[i / NSLBDP_REDUCE_SCATTER_MOLD2] = *niter;
            }
        } else {
            std::vector<LINK>::const_iterator niter = std::next(iter, i);
            if (niter != links.end()) {
                subLinks[i - nslbPart1Size / NSLBDP_REDUCE_SCATTER_MOLD2] = *niter;
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
        adjInfoStep.phaseId = step + begin + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();

    if (nslbAdjInfo.nsAdjInfo.size() == 0) {
        return HCCL_SUCCESS;
    }
    // 上面处理完成后，紧接着处理合并部分的偶数rank同步到奇数rank增加phaseId
    if (rank < nslbPart1Size && rank % NSLBDP_REDUCE_SCATTER_MOLD2 == 0) {
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
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_RECURSIVE_HD, ReduceScatterRecursiveHalvingDoubling);
} // namespace hccl
