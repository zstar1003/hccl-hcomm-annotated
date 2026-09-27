/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_slim_ring.h"
#include "alg_template_register.h"

namespace hccl {
ReduceScatterSlimRing::ReduceScatterSlimRing(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ReduceScatterSlimRing::~ReduceScatterSlimRing() {}

HcclResult ReduceScatterSlimRing::Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo)
{
    (void)opInfo;
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterSlimRing::RunVectorSourceReducer(
    const LINK& link, const std::vector<Slice>& txSlices, const std::vector<Slice>& txSlicetemp)
{
    /* 1、对外reduce_scatter，output的大小为每块数据*rank_size。只能发送到对端地址偏移为0开始。
      2、allreduce中使用reduce_scatter，output与Input大小相等，接收和发送偏移相等都为slice.offset */
    std::vector<SenderMemoryInfo> txMems;
    for (u32 i = 0; i < txSlices.size(); i++) {
        DeviceMem srcMem = inputMem_.range(txSlices[i].offset, txSlices[i].size);
        HCCL_DEBUG(
            "send inputmem range[%llu], size[%llu] tx dstmem offset[%llu]", txSlices[i].offset, txSlices[i].size,
            txSlicetemp[i].offset);
        txMems.emplace_back(SenderMemoryInfo{baseOffset_ + txSlicetemp[i].offset, srcMem});
    }
    CHK_RET(senderInfo_->run(link, txMems, notifyIdx_, stream_));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterSlimRing::RunVectorDestRducer(
    const LINK& link, const std::vector<Slice>& rxSlices, const std::vector<Slice>& rxSlicetemp)
{
    std::vector<ReducerMemoryInfo> rxReduceMems;
    for (u32 i = 0; i < rxSlices.size(); i++) {
        DeviceMem dstMem = inputMem_.range(rxSlices[i].offset, rxSlices[i].size);
        DeviceMem srcMemTemp = scratchMem_.range(rxSlicetemp[i].offset, rxSlicetemp[i].size);
        HCCL_DEBUG(
            "rcv offset[%llu], size[%llu] ,then reduce with "
            "offset[%llu] size[%llu] ",
            rxSlicetemp[i].offset, rxSlicetemp[i].size, rxSlices[i].offset, rxSlices[i].size);
        rxReduceMems.emplace_back(ReducerMemoryInfo{baseOffset_ + rxSlices[i].offset, dstMem, dstMem, srcMemTemp});
    }
    CHK_RET(reducerInfo_->run(dispatcher_, link, rxReduceMems, notifyIdx_, stream_));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterSlimRing::RunVectorFinRducer(
    const u32 rank, const LINK& link, const u32 sliceSize, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& outputSlices)
{
    std::vector<ReducerMemoryInfo> rxReduceMems;
    for (u32 i = 0; i < sliceSize; i++) {
        DeviceMem dstMem
            = outputMem_.range(outputSlices[rank * sliceSize + i].offset, outputSlices[rank * sliceSize + i].size);
        // reduce目的操作
        DeviceMem srcMem
            = inputMem_.range(inputSlices[rank * sliceSize + i].offset, inputSlices[rank * sliceSize + i].size);
        DeviceMem scratchMem
            = scratchMem_.range(outputSlices[rank * sliceSize + i].offset, outputSlices[rank * sliceSize + i].size);
        rxReduceMems.emplace_back(
            ReducerMemoryInfo{baseOffset_ + inputSlices[rank * sliceSize + i].offset, srcMem, dstMem, scratchMem});
    }
    CHK_RET(reducerInfo_->run(dispatcher_, link, rxReduceMems, notifyIdx_, stream_));

    notifyIdx_++;

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterSlimRing::RunSourceReducer(const LINK& link, const Slice& txSlice, const Slice& txSlicetemp)
{
    /* 1、对外reduce_scatter，output的大小为每块数据*rank_size。只能发送到对端地址偏移为0开始。
      2、allreduce中使用reduce_scatter，output与Input大小相等，接收和发送偏移相等都为slice.offset */
    DeviceMem srcMem = inputMem_.range(txSlice.offset, txSlice.size);
    HCCL_DEBUG(
        " send inputmem range[%llu], size[%llu] tx dstmem offset[%llu]", txSlice.offset, txSlice.size,
        txSlicetemp.offset);
    CHK_RET(senderInfo_->run(link, baseOffset_ + txSlicetemp.offset, srcMem, stream_));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterSlimRing::RunDestRducer(const LINK& link, const Slice& rxSlice, const Slice& rxSlicetemp)
{
    DeviceMem dstMem = inputMem_.range(rxSlice.offset, rxSlice.size);
    DeviceMem srcMemTemp = scratchMem_.range(rxSlicetemp.offset, rxSlicetemp.size);
    HCCL_DEBUG(
        "rcv offset[%llu], size[%llu] ,then reduce with "
        "offset[%llu] size[%llu] ",
        rxSlicetemp.offset, rxSlicetemp.size, rxSlice.offset, rxSlice.size);
    CHK_RET(reducerInfo_->run(dispatcher_, link, baseOffset_ + rxSlice.offset, dstMem, dstMem, srcMemTemp, stream_));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterSlimRing::InitSlice(std::vector<Slice>& outputSlices, u32 rank, u32 rankSize, u32 unitSize)
{
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        outputSlices.resize(rankSize);
        u64 sliceSize = count_ * unitSize;
        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = (i * sliceSize);
            outputSlices[i].size = sliceSize;
            outputSlices[i].offset = (inputMem_.size() > outputMem_.size()) ? 0 : (i * sliceSize);
            HCCL_DEBUG(
                "rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu] outputSlices[%u].offset=[%llu], "
                "outputSlices[%u].size=[%llu] ",
                rank, i, slices_[i].offset, i, slices_[i].size, i, outputSlices[i].offset, i, outputSlices[i].size);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterSlimRing::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[ReduceScatterSlimRing][RunAsync]rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO(
        "ReduceScatterRing run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    // 判断rank_size == 1
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            return HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return HCCL_SUCCESS;
    }

    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[ReduceScatterSlimRing][RunAsync]rank[%u] link size[%llu] is less than rank size[%u]", rank, links.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    u32 ringPrevRank = (rank + rankSize - 1) % rankSize;
    linkLeft_ = links[ringPrevRank];
    CHK_SMART_PTR_NULL(linkLeft_);

    u32 ringNextRank = (rank + 1) % rankSize;
    linkRight_ = links[ringNextRank];
    CHK_SMART_PTR_NULL(linkRight_);

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[ReduceScatterSlimRing][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    std::vector<Slice> outputSlices(slices_);
    InitSlice(outputSlices, rank, rankSize, unitSize);
    // 运行reduce-scatter, ring算法
    // 单环场景下 nicRankList_ 长度默认为 8。
    // 多环场景下 nicRankList_ 长度为网口数量。此时若 rankSize != nicRankList_ 则为网口裁剪场景
    if (rankSize != HCCL_NIC_MAX_NUM || nicRankList_.size() == HCCL_NIC_MAX_NUM) {
        // 非网口裁剪场景:
        CHK_RET(RunReduceScatter(rank, rankSize, slices_, outputSlices));
    }

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(linkRight_, linkLeft_, notifyIdx_));
        notifyIdx_++;
    }

    HCCL_INFO("ReduceScatterRing finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterSlimRing::RunReduceScatter(
    const u32 rank, const u32 rankSize, const std::vector<Slice>& inputSlices, const std::vector<Slice>& outputSlices)
{
    bool bRetSize = (inputSlices.size() < rankSize);
    CHK_PRT_RET(
        bRetSize,
        HCCL_ERROR(
            "[Run][ReduceScatter]rank[%u] inputslice size[%llu] is less than rank size[%u]", rank, inputSlices.size(),
            rankSize),
        HCCL_E_INTERNAL);

    bRetSize = (outputSlices.size() < rankSize);
    CHK_PRT_RET(
        bRetSize,
        HCCL_ERROR(
            "[Run][ReduceScatter]rank[%u] outputslice size[%llu] is less than rank size[%u]", rank, outputSlices.size(),
            rankSize),
        HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;

    u32 sliceSize = inputSlices.size() / rankSize;

    // 获取rx_slice, 首先向本rank前2个rank处发ack消息
    u32 rxSliceIndex = (rank + rankSize - 2) % rankSize;

    // reduce源操作, 获取tx_slice，从本rank前一rank开始接收ack
    u32 txSliceIndex = (rank + rankSize - 1) % rankSize;

    std::vector<Slice> txInputSegsSlice;
    std::vector<Slice> txOutputSegsSlice;
    for (u32 j = 0; j < sliceSize; j++) {
        txInputSegsSlice.push_back(inputSlices[txSliceIndex * sliceSize + j]);
        txOutputSegsSlice.push_back(outputSlices[txSliceIndex * sliceSize + j]);
    }
    ret = RunVectorSourceReducer(linkRight_, txInputSegsSlice, txOutputSegsSlice); // NotifyRecord
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Run][ReduceScatter]rank[%u] txSliceIndex[%u] Reducer src run failed", rank, txSliceIndex), ret);

    // 本rank既当reduce源, 也当reduce操作的目的
    for (u32 i = 0; i < (rankSize - 2); i++) { // 中间rank_size - 2次传输
        // reduce目的操作
        std::vector<Slice> rxInputSegsSlice;
        std::vector<Slice> rxOutputSegsSlice;
        for (u32 j = 0; j < sliceSize; j++) {
            rxInputSegsSlice.push_back(inputSlices[rxSliceIndex * sliceSize + j]);
            rxOutputSegsSlice.push_back(outputSlices[rxSliceIndex * sliceSize + j]);
        }
        ret = RunVectorDestRducer(linkLeft_, rxInputSegsSlice, rxOutputSegsSlice);

        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Run][ReduceScatter]rank[%u] round[%u] rxSlice[%u] Reducer dst run failed", rank, i, rxSliceIndex),
            ret);

        notifyIdx_++;

        // 获取rx_slice
        rxSliceIndex = (rxSliceIndex + rankSize - 1) % rankSize;

        // reduce源操作, 获取tx_slice
        txSliceIndex = (txSliceIndex + rankSize - 1) % rankSize;

        std::vector<Slice> txInputSlice;
        std::vector<Slice> txOutputSlice;
        for (u32 j = 0; j < sliceSize; j++) {
            txInputSlice.push_back(inputSlices[txSliceIndex * sliceSize + j]);
            txOutputSlice.push_back(outputSlices[txSliceIndex * sliceSize + j]);
        }
        ret = RunVectorSourceReducer(linkRight_, txInputSlice, txOutputSlice);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]rank[%u] round[%u] Reducer src run failed", rank, i),
            ret);
    }

    /* * 末尾传输, 本rank只当reduce目的, 根据单buffer还是双buffer来决定如何搬移
        当前简化处理, 只考虑单buffer的场景, 双buffer则在run_async中多拷贝一次 */
    RunVectorFinRducer(rank, linkLeft_, sliceSize, inputSlices, outputSlices);

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterSlimRing::SetNotifyIdx(u32 notifyIdx)
{
    notifyIdx_ = notifyIdx;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterSlimRing::GetNotifyIdx(u32& notifyIdx)
{
    notifyIdx = notifyIdx_;
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_SLIM_RING, ReduceScatterSlimRing);
} // namespace hccl
