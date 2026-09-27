/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_mesh_mix_single_stream.h"
#include "alg_template_register.h"

namespace hccl {
ReduceScatterMeshMixSingleStream::ReduceScatterMeshMixSingleStream(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher),
      reduceAttr_(0),
      streamIndex_(0)
{}

ReduceScatterMeshMixSingleStream::~ReduceScatterMeshMixSingleStream() {}

HcclResult ReduceScatterMeshMixSingleStream::Prepare(u64 reduceAttrBitMap, u32 streamIndex)
{
    reduceAttr_ = reduceAttrBitMap;
    streamIndex_ = streamIndex;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshMixSingleStream::RunSourceReducer(
    const LINK& link, const std::vector<Slice>& txSlices, const std::vector<Slice>& dstSlices)
{
    // 发送inputmem
    std::vector<SenderMemoryInfo> txMems;
    for (u64 i = 0; i < txSlices.size(); i++) {
        DeviceMem srcMem = inputMem_.range(txSlices[i].offset, txSlices[i].size);
        HCCL_DEBUG(
            "[ReduceScatterMeshMixSingleStream][RunSourceReducer] send inputmem range[%llu], size[%llu] "
            "tx dstmem offset[%llu]",
            txSlices[i].offset, txSlices[i].size, dstSlices[i].offset);
        txMems.emplace_back(SenderMemoryInfo{baseOffset_ + dstSlices[i].offset, srcMem});
    }

    CHK_RET(senderInfo_->run(link, txMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshMixSingleStream::RunDestReducer(
    const LINK& link, const std::vector<Slice>& rxSlices, const std::vector<Slice>& dstSlices)
{
    // 使用scratchmem接收数据，并同inputmem数据做reduce
    std::vector<ReducerMemoryInfo> rxReduceMems;
    for (u64 i = 0; i < rxSlices.size(); i++) {
        DeviceMem dstMem = inputMem_.range(rxSlices[i].offset, rxSlices[i].size);
        DeviceMem srcMem = scratchMem_.range(dstSlices[i].offset, dstSlices[i].size);
        HCCL_DEBUG(
            "[ReduceScatterMeshMixSingleStream][RunDestReducer] rcv offset[%llu], size[%llu] ,then reduce with "
            "offset[%llu] size[%llu] ",
            dstSlices[i].offset, dstSlices[i].size, rxSlices[i].offset, rxSlices[i].size);
        rxReduceMems.emplace_back(ReducerMemoryInfo{baseOffset_ + rxSlices[i].offset, dstMem, dstMem, srcMem});
    }

    CHK_RET(reducerInfo_->run(dispatcher_, link, rxReduceMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshMixSingleStream::RunReduceScatter(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, const std::vector<Slice>& inputSlices,
    const std::vector<Slice>& scratchSlices)
{
    u32 interRankSize = inputSlices.size() / rankSize;
    std::vector<u32> txRankOpOrder;
    std::vector<u32> rxRankOpOrder;
    // 计算默认的每轮接收的源rank和发送的目的rank
    for (u32 round = 1; round < rankSize; round++) {
        u32 srcRank = ForwardRank(rank, rankSize, round);
        u32 dstRank = BackwardRank(rank, rankSize, round);
        HCCL_INFO("<multiDie>RunReduceScatter:srcRank[%u] dstRank[%u]", srcRank, dstRank);
        rxRankOpOrder.push_back(srcRank);
        txRankOpOrder.push_back(dstRank);
    }

    HcclResult ret = HCCL_SUCCESS;
    for (u32 round = 1; round < rankSize; round++) {
        // 不同的stream依次轮训默认的顺序数组
        u32 orderIndex = (round + streamIndex_ - 1) % (rankSize - 1);
        u32 srcRank = rxRankOpOrder[orderIndex];
        s32 dstRank = txRankOpOrder[orderIndex];
        CHK_SMART_PTR_NULL(links[srcRank]);
        HCCL_INFO("rank[%u] will tx_ack to rank[%u]", rank, srcRank);
        ret = links[srcRank]->TxAck(stream_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]rank[%u] tx ack to rank[%u] failed", rank, srcRank),
            ret);
        CHK_SMART_PTR_NULL(links[dstRank]);
        HCCL_INFO("rank[%u] will rx_ack from rank[%d]", rank, dstRank);

        ret = links[dstRank]->RxAck(stream_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]rank[%u] rx ack from rank[%d] failed", rank, dstRank),
            ret);
        HCCL_INFO(
            "rank:%u round[%u] send to rank:[%d], inputSlices offset[%llu]"
            "size[%llu] scratchSlice offset[%llu] size[%llu] ",
            rank, round, dstRank, inputSlices[dstRank].offset, inputSlices[dstRank].size, scratchSlices[dstRank].offset,
            scratchSlices[dstRank].size);
        // 发送数据
        std::vector<Slice> txSlices;
        std::vector<Slice> txDstSlices;
        for (u32 i = dstRank * interRankSize; i < (dstRank + 1) * interRankSize; i++) {
            txSlices.push_back(inputSlices[i]);
            txDstSlices.push_back(scratchSlices[i]);
        }
        ret = RunSourceReducer(links[dstRank], txSlices, txDstSlices);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][ReduceScatter]rank:%u round[%u] reducer src run failed", rank, round), ret);
        HCCL_INFO(
            "rank[%u] round[%u] rx from rank[%u], inSlicesoffset[%llu] size[%llu] "
            "scratchSlices offset[%llu] size[%llu]",
            rank, round, srcRank, inputSlices[rank].offset, inputSlices[rank].size, scratchSlices[rank].offset,
            scratchSlices[rank].size);

        std::vector<Slice> rxSlices;
        std::vector<Slice> rxDstSlices;
        for (u32 i = rank * interRankSize; i < (rank + 1) * interRankSize; i++) {
            rxSlices.push_back(inputSlices[i]);
            rxDstSlices.push_back(scratchSlices[i]);
        }
        ret = RunDestReducer(links[srcRank], rxSlices, rxDstSlices);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][ReduceScatter]rank[%u] round[%u] reducer dst run failed", rank, round), ret);

        ret = links[srcRank]->RxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]RxWaitDone failed"), ret);
        ret = links[dstRank]->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]TxWaitDone failed"), ret);
    }
    if (barrierSwitchOn_) {
        for (u32 round = 1; round < rankSize; round++) {
            u32 orderIndex = (round + streamIndex_ - 1) % (rankSize - 1);
            u32 srcRank = rxRankOpOrder[orderIndex];
            s32 dstRank = txRankOpOrder[orderIndex];

            ret = ExecuteBarrier(links[srcRank], links[dstRank]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Run][ReduceScatter]rank[%u] run ReduceScatter executor barrier "
                    "failed. srcRank:%u dstRank:%d",
                    rank, srcRank, dstRank),
                ret);
        }
    }

    return HCCL_SUCCESS;
}

// reducescatter的入口函数
HcclResult
ReduceScatterMeshMixSingleStream::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR(
            "[ReduceScatterMeshMixSingleStream][RunAsync]rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO(
        "[ReduceScatterMeshMixSingleStream][RunAsync]rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] "
        "count[%llu]",
        rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            return HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return HCCL_SUCCESS;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[ReduceScatterMeshMixSingleStream][RunAsync]rank[%u] linksize error", rank);
        return HCCL_E_INTERNAL;
    }

    if (streamIndex_ >= rankSize - 1) {
        HCCL_ERROR(
            "[ReduceScatterMeshMixSingleStream][RunAsync]rank[%u] stream index[%u] is out of range when ranksize[%u]",
            rank, streamIndex_, rankSize);
        return HCCL_E_INTERNAL;
    }

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[ReduceScatterMeshMixSingleStream][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    std::vector<Slice> scratchSlices(slices_);

    // 运行reduce-scatter, mesh算法
    CHK_RET(RunReduceScatter(rank, rankSize, links, slices_, scratchSlices));

    HCCL_INFO("ReduceScatterMeshMixSingleStream finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_MESH_MIX_SS, ReduceScatterMeshMixSingleStream);
} // namespace hccl
