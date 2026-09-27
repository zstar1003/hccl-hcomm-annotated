/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_nb.h"
#include "alg_template_register.h"

namespace hccl {
ScatterNB::ScatterNB(const HcclDispatcher dispatcher) : NBBase(dispatcher), interRank_(0), interRankSize_(0) {}

ScatterNB::~ScatterNB() {}

HcclResult ScatterNB::RunScatterNB(const std::vector<std::shared_ptr<Transport>>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    // 需要判断input不等于outputmem，scatter 输入只有一个input时不用拷贝
    if (inputMem_ != outputMem_) {
        ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Run][ScatterOnRootRank]root rank[%u] memcpy async from input[%p] "
                "failed to output[%p]",
                interRank_, inputMem_.ptr(), outputMem_.ptr()),
            ret);
    }

    // 计算通信步数：ceiling(log2(rankSize))
    u32 nSteps = CalcCeilLog2(interRankSize_);

    // 逐步编排任务
    u32 deltaRoot = (interRank_ + interRankSize_ - root_) % interRankSize_;
    for (u32 step = 0; step < nSteps; step++) {
        if (deltaRoot < u32(1 << step)) {
            if (step != nSteps - 1 || deltaRoot < (interRankSize_ - (1 << step))) {
                RunScatterTx(step, links);
            }
        } else if (deltaRoot < u32(1 << (step + 1)) && interRank_ != root_) {
            RunScatterRx(step, links);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterNB::RunScatterTx(const u32 step, const std::vector<std::shared_ptr<Transport>>& links)
{
    HcclResult ret = HCCL_SUCCESS;

    // 计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (interRank_ + deltaRank) % interRankSize_;
    // 数据份数和数据编号增量
    u32 nSlices = (interRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 sliceIdx = (interRank_ + (1 << step)) % interRankSize_;
    LINK linkRight = links[sendTo];
    CHK_SMART_PTR_NULL(linkRight);

    std::vector<Slice> txSlices;
    for (u32 i = 0; i < nSlices; i++) {
        if (slicesFlag_[sliceIdx] == false) {
            continue;
        }
        txSlices.push_back(slices_[sliceIdx]);
        sliceIdx = (sliceIdx + deltaSliceIndex) % interRankSize_;
    }

    CHK_RET(linkRight->RxAck(stream_));
    ret = Tx(linkRight, txSlices);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Run][Scatter]rank[%u] step[%u] RightLink tx slices count [%u] Failed", interRank_, step, nSlices),
        ret);
    ret = linkRight->TxWaitDone(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][Scatter]TxWaitDone failed"), ret);

    // 为了避免在大数据量场景下触发网卡轮询机制，这里添加一组Data Notify，确保对端数据接收完成才进行下一次通信任务
    CHK_RET(linkRight->RxDataSignal(stream_));
    return HCCL_SUCCESS;
}

HcclResult ScatterNB::RunScatterRx(const u32 step, const std::vector<std::shared_ptr<Transport>>& links)
{
    HcclResult ret = HCCL_SUCCESS;

    // 计算通信对象
    u32 deltaRank = 1 << step;
    u32 recvFrom = (interRank_ + interRankSize_ - deltaRank) % interRankSize_;
    // 数据份数和数据编号增量
    u32 nSlices = (interRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 sliceIdx = interRank_;
    LINK linkLeft = links[recvFrom];
    CHK_SMART_PTR_NULL(linkLeft);

    std::vector<Slice> rxSlices;
    for (u32 i = 0; i < nSlices; i++) {
        rxSlices.push_back(slices_[sliceIdx]);
        sliceIdx = (sliceIdx + deltaSliceIndex) % interRankSize_;
        slicesFlag_[sliceIdx] = true;
    }

    CHK_RET(linkLeft->TxAck(stream_));
    ret = Rx(linkLeft, rxSlices);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Run][Scatter]rank[%u] step[%u] Right Link rx slices count [%u] "
            "Failed",
            interRank_, step, nSlices),
        ret);

    ret = linkLeft->RxWaitDone(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][Scatter]RxWaitDone failed"), ret);

    // 为了避免在大数据量场景下触发网卡轮询机制，这里添加一组Data Notify，确保对端数据接收完成才进行下一次通信任务
    CHK_RET(linkLeft->TxDataSignal(stream_));
    return HCCL_SUCCESS;
}

void ScatterNB::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = (totalCount / rankSize) * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
    }
}

// scatter的入口函数
HcclResult ScatterNB::RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[ScatterNB][RunAsync]run_async inputmem or outputmem is null");
        return HCCL_E_PTR;
    }

    interRank_ = rank;
    interRankSize_ = rankSize;
    if (interRank_ == root_) {
        slicesFlag_.resize(interRankSize_, true);
    } else {
        slicesFlag_.resize(interRankSize_, false);
    }

    // ranksize为1时，只有当input!=output 时候进行拷贝
    if (interRankSize_ == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(
        unitSize == 0, HCCL_ERROR("[ScatterNB][RunAsync]rank[%u] unit data size is zero", rank), HCCL_E_INTERNAL);

    // 带入vector为空，计算每个rank的结果偏移和大小
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, interRankSize_);
    }

    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR("[ScatterNB][RunAsync]rank[%u] link size[%llu] is less than rank size", rank, links.size()),
        HCCL_E_INTERNAL);

    CHK_RET(RunScatterNB(links));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(
            links[(interRank_ + interRankSize_ - 1) % interRankSize_], links[(interRank_ + 1) % interRankSize_]));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterNB::Tx(const LINK& link, const std::vector<Slice>& txSlices)
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

HcclResult ScatterNB::Rx(const LINK& link, const std::vector<Slice>& rxSlices)
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

HcclResult
ScatterNB::GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }
    if (links.size() < rankSize) {
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[ScatterNB]GetNslbAdjInfo start");
    u32 nSteps = 0;
    for (u32 temp = rankSize - 1; temp != 0; temp >>= 1, ++nSteps) {
    }

    u32 deltaRoot = (rank + rankSize - root_) % rankSize;
    for (u32 step = 0; step < nSteps; step++) {
        if (deltaRoot >= u32(1 << step)) {
            continue;
        }
        HCCL_DEBUG("[ScatterNB]now step is %u start", step);
        if (step != nSteps - 1 || deltaRoot < (rankSize - (1 << step))) {
            u32 deltaRank = 1 << step;
            u32 sendTo = (rank + deltaRank) % rankSize;
            HCCL_DEBUG("[ScatterNB]sendTo is %u", sendTo);
            LINK linkRight = links[sendTo];
            CHK_SMART_PTR_NULL(linkRight);

            NslbDpAdjInfo adjInfoStep = {};
            adjInfoStep.dstLocalRankId = linkRight->GetRemoteRank();
            adjInfoStep.phaseId = step + 1;
            adjInfoStep.rev = 0;
            nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
        }
    }
    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_SCATTER_NB, ScatterNB);
} // namespace hccl
