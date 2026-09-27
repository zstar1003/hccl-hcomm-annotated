/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_nb.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherNB::AllGatherNB(const HcclDispatcher dispatcher) : NBBase(dispatcher) {}

AllGatherNB::~AllGatherNB() {}

HcclResult AllGatherNB::Tx(const LINK& link, const std::vector<Slice>& txSlices)
{
    std::vector<TxMemoryInfo> txMems;
    for (const Slice& txSlice : txSlices) {
        DeviceMem srcMem = outputMem_.range(txSlice.offset, txSlice.size);
        HCCL_DEBUG("tx srcMem[%p] range[%llu] size[%llu] ", srcMem.ptr(), txSlice.offset, txSlice.size);
        txMems.emplace_back(
            TxMemoryInfo{UserMemType::OUTPUT_MEM, txSlice.offset + baseOffset_, srcMem.ptr(), txSlice.size});
    }

    CHK_RET(link->TxAsync(txMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherNB::Rx(const LINK& link, const std::vector<Slice>& rxSlices)
{
    std::vector<RxMemoryInfo> rxMems;
    for (const Slice& rxSlice : rxSlices) {
        DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
        HCCL_DEBUG("rx dstMem[%p] range[%llu], size[%llu] ", dstMem.ptr(), rxSlice.offset, rxSlice.size);
        rxMems.emplace_back(
            RxMemoryInfo{UserMemType::OUTPUT_MEM, rxSlice.offset + baseOffset_, dstMem.ptr(), rxSlice.size});
    }

    CHK_RET(link->RxAsync(rxMems, stream_));
    return HCCL_SUCCESS;
}

// 服务器间allgather的入口函数
HcclResult AllGatherNB::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "[AllGatherNB] run_async rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        links.size() < rankSize, HCCL_ERROR("[AllGatherNB][RunAsync]rank[%u] linkSize is less than rankSize", rank),
        HCCL_E_INTERNAL);

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(unitSize == 0, HCCL_ERROR("[AllGatherNB][RunAsync]unitSize is zero"), HCCL_E_INTERNAL);

    std::vector<Slice> inputSlices(slices_);
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        inputSlices.resize(rankSize);

        u64 sliceSize = count_ * unitSize;
        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = sliceSize * i;
            inputSlices[i].size = sliceSize;
            inputSlices[i].offset = (inputMem_.size() < outputMem_.size()) ? 0 : (sliceSize * i);
            HCCL_DEBUG(
                "rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", rank, i, slices_[i].offset, i,
                slices_[i].size);
        }
    }

    // 双buffer下, 先将input拷贝到output的合适位置
    if (inputMem_ != outputMem_) {
        DeviceMem dst = outputMem_.range(slices_[rank].offset, slices_[rank].size);
        DeviceMem src = inputMem_.range(inputSlices[rank].offset, inputSlices[rank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    // 运行all-gather, ring算法
    CHK_RET(RunAllGather(rank, rankSize, slices_, links));

    HCCL_INFO("[AllGatherNB] finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult AllGatherNB::RunAllGather(
    u32 rank, u32 rankSize, const std::vector<Slice>& outputSlices, const std::vector<LINK>& links)
{
    CHK_PRT_RET(
        outputSlices.size() < rankSize,
        HCCL_ERROR("[Run][AllGather]rank[%u] OutputSlice Size is less than rank size", rank), HCCL_E_INTERNAL);
    HcclResult ret = HCCL_SUCCESS;

    u32 nSteps = CalcCeilLog2(rankSize);
    u32 nRealSliceSize = outputSlices.size() / rankSize;
    HCCL_DEBUG(
        "[AllGatherNB][RunAllGather]Starts, outputSlices.size[%u], rankSize[%u], nRealSliceSize[%u]",
        outputSlices.size(), rankSize, nRealSliceSize);

    // 逐步编排任务
    for (u32 step = 0; step < nSteps; step++) {
        // 计算通信对象
        u32 deltaRank = 1 << step;
        u32 recvFrom = (rankSize + rank - deltaRank) % rankSize;
        u32 sendTo = (rank + deltaRank) % rankSize;

        // 数据份数和数据编号增量
        // 节点6个，nSteps = 3，step = 2
        u32 nSlices = 1 << step;
        if (step == (nSteps - 1) && rankSize != u32(1 << nSteps)) {
            nSlices = rankSize - (1 << step);
        }

        LINK linkLeft = links[recvFrom];
        CHK_SMART_PTR_NULL(linkLeft);

        LINK linkRight = links[sendTo];
        CHK_SMART_PTR_NULL(linkRight);

        std::vector<Slice> txSlices;
        std::vector<Slice> rxSlices;
        for (u32 i = 0; i < nSlices; i++) {
            u32 rxSliceIndex = (rank - (1 << step) + rankSize - i) % rankSize;
            u32 txSliceIndex = (rank + rankSize - i) % rankSize;
            for (u32 j = 0; j < nRealSliceSize; j++) {
                u32 rxIndex = rxSliceIndex * nRealSliceSize + j;
                u32 txIndex = txSliceIndex * nRealSliceSize + j;
                if (outputSlices[txIndex].size > 0) {
                    txSlices.push_back(outputSlices[txIndex]);
                }
                if (outputSlices[rxIndex].size > 0) {
                    rxSlices.push_back(outputSlices[rxIndex]);
                }
                HCCL_DEBUG(
                    "rank[%u] round[%u] slice[%u] realSlice[%u] rx data outputSlice[%u] offset[%llu] size[%llu]", rank,
                    step, i, j, rxIndex, outputSlices[rxIndex].offset, outputSlices[rxIndex].size);
            }
        }
        if (rxSlices.size() > 0) {
            CHK_RET(linkLeft->TxAck(stream_));
        }
        if (txSlices.size() > 0) {
            CHK_RET(linkRight->RxAck(stream_));
            ret = Tx(linkRight, txSlices);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][AllGather]rank[%u] round[%u] tx %u slices failed", rank, step, nSlices), ret);
        }
        if (rxSlices.size() > 0) {
            ret = Rx(linkLeft, rxSlices);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][AllGather]rank[%u] round[%u] rx %u slices failed", rank, step, nSlices), ret);

            ret = linkLeft->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]RxWaitDone failed"), ret);
            ret = linkLeft->PostFinAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]PostFinAck failed"), ret);
        }
        if (txSlices.size() > 0) {
            ret = linkRight->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]TxWaitDone failed"), ret);
            ret = linkRight->WaitFinAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]WaitFinAck failed"), ret);
        }

        if (linkRight->IsSpInlineReduce() || linkLeft->IsSpInlineReduce()) {
            // SDMA场景同步
            CHK_RET(ExecuteBarrier(linkLeft, linkRight));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
AllGatherNB::GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }
    if (links.size() < rankSize) {
        return HCCL_SUCCESS;
    }
    u32 nSteps = 0;
    for (u32 temp = rankSize - 1; temp != 0; temp >>= 1, ++nSteps) {
    }

    for (u32 step = 0; step < nSteps; step++) {
        u32 deltaRank = 1 << step;
        u32 sendTo = (rank + deltaRank) % rankSize;
        LINK linkRight = links[sendTo];
        CHK_SMART_PTR_NULL(linkRight);

        NslbDpAdjInfo adjInfoStep = {};
        adjInfoStep.dstLocalRankId = linkRight->GetRemoteRank();
        adjInfoStep.phaseId = step + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    nslbAdjInfo.dstRankNum = nSteps;
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_NB, AllGatherNB);
} // namespace hccl
