/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_nb_binary.h"
#include "device_capacity.h"
#include "alg_template_register.h"

namespace hccl {
constexpr float LATENCY = 60; // 静态时延 60 us;

BroadcastNBBinary::BroadcastNBBinary(const HcclDispatcher dispatcher) : NBBase(dispatcher) {}

BroadcastNBBinary::~BroadcastNBBinary() {}

HcclResult
BroadcastNBBinary::RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("BroadcastNBBinary run: rank[%u] totalrank[%u] count[%llu]", rank, rankSize, count_);

    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR("[BroadcastNBBinary][RunAsync]rank[%u] linksize[%llu] is less than rank size", rank, links.size()),
        HCCL_E_INTERNAL);

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(
        unitSize == 0, HCCL_ERROR("[BroadcastNBBinary][RunAsync]rank[%u] unit data size is zero", rank),
        HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;
    if (inputMem_ != outputMem_) {
        ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Run][BroadcastOnRootRank]root rank[%u] memcpy async from input[%p] "
                "failed to output[%p]",
                rank, inputMem_.ptr(), outputMem_.ptr()),
            ret);
    }

    CHK_RET(RunBroadcastNBBinary(rank, rankSize, links));
    HCCL_INFO("BroadcastNBBinary finished: rank[%u] end count[%llu]", rank, count_);
    return HCCL_SUCCESS;
}

HcclResult BroadcastNBBinary::RunBroadcastNBBinary(
    const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    if (rank == root_) {
        hasData_ = true;
        CHK_PRT_RET(!inputMem_, HCCL_ERROR("[BroadcastNBBinary][RunAsync]rank[%u] inputmem is null", rank), HCCL_E_PTR);
    } else {
        CHK_PRT_RET(
            !outputMem_, HCCL_ERROR("[BroadcastNBBinary][RunAsync]rank[%u] outputmem is null", rank), HCCL_E_PTR);
    }

    HCCL_DEBUG("root[%u], hasData[%u]", root_, hasData_);

    u64 dataBytes = count_ * DataUnitSize(dataType_);     // 总数据量
    u32 nSteps = CalcCeilLog2(rankSize);                  // 通信步数
    u32 deltaRoot = (rank + rankSize - root_) % rankSize; // 与Root节点的距离

    for (u32 step = 0; step < nSteps; step++) {
        if (deltaRoot < u32(1 << step)) { // 该节点需要发送数据
            if ((step != nSteps - 1 || deltaRoot < (rankSize - (1 << step))) && hasData_) {
                u32 deltaRank = 1 << step;
                u32 sendTo = (rank + deltaRank) % rankSize;
                LINK linkRight = links[sendTo];
                CHK_SMART_PTR_NULL(linkRight);

                std::vector<Slice> txSlices;
                txSlices.resize(1);
                txSlices[0].offset = baseOffset_;
                txSlices[0].size = dataBytes;

                CHK_RET(linkRight->RxAck(stream_));
                ret = Tx(linkRight, txSlices);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Run][Broadcast]rank[%u] step[%u] Right Link tx slices Failed", rank, step), ret);

                ret = linkRight->TxWaitDone(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][Broadcast]TxWaitDone failed"), ret);
                ret = linkRight->GetLinkType() == LinkType::LINK_HCCS ?
                          linkRight->WaitFin(stream_) :
                          linkRight->WaitFinAck(stream_); // P2P和Roce场景都需要同步
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][Broadcast]WaitFinAck failed"), ret);
            }
        } else if (deltaRoot < u32(1 << (step + 1)) && rank != root_) { // 该节点需要接收数据
            u32 deltaRank = 1 << step;
            u32 recvFrom = (rank + rankSize - deltaRank) % rankSize;
            LINK linkLeft = links[recvFrom];
            CHK_SMART_PTR_NULL(linkLeft);

            std::vector<Slice> rxSlices;
            rxSlices.resize(1);
            rxSlices[0].offset = baseOffset_;
            rxSlices[0].size = dataBytes;

            CHK_RET(linkLeft->TxAck(stream_));
            ret = Rx(linkLeft, rxSlices);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][Broadcast]rank[%u] step[%u] Right Link rx slices Failed", rank, step), ret);

            ret = linkLeft->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][Broadcast]RxWaitDone failed"), ret);
            ret = linkLeft->GetLinkType() == LinkType::LINK_HCCS ?
                      linkLeft->PostFin(stream_) :
                      linkLeft->PostFinAck(stream_); // P2P和Roce场景都需要同步
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][Broadcast]PostFinAck failed"), ret);
            hasData_ = true;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastNBBinary::Tx(const LINK& link, const std::vector<Slice>& txSlices)
{
    std::vector<TxMemoryInfo> txMems;
    for (const Slice& txSlice : txSlices) {
        DeviceMem srcMem = outputMem_.range(txSlice.offset, txSlice.size);
        HCCL_DEBUG("tx srcMem[%p] range[%llu] size[%llu] ", srcMem.ptr(), txSlice.offset, txSlice.size);
        txMems.emplace_back(TxMemoryInfo{UserMemType::OUTPUT_MEM, txSlice.offset, srcMem.ptr(), txSlice.size});
    }

    CHK_RET(link->TxAsync(txMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult BroadcastNBBinary::Rx(const LINK& link, const std::vector<Slice>& rxSlices)
{
    std::vector<RxMemoryInfo> rxMems;
    for (const Slice& rxSlice : rxSlices) {
        DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
        HCCL_DEBUG("rx dstMem[%p] range[%llu], size[%llu] ", dstMem.ptr(), rxSlice.offset, rxSlice.size);
        rxMems.emplace_back(RxMemoryInfo{UserMemType::OUTPUT_MEM, rxSlice.offset, dstMem.ptr(), rxSlice.size});
    }

    CHK_RET(link->RxAsync(rxMems, stream_));
    return HCCL_SUCCESS;
}

bool ShouldUseBinaryBroadcastOfNB(
    const u64 dataSize, const u32 rankSize, const u32 userRankSize, const float deviceNumPerAggregation)
{
    // 小数据量和rank size为2时使用Binary broadcast
    HCCL_INFO(
        "datasize[%llu], ranksize[%u], userRankSize[%u], deviceNumPerAggregation[%f]", dataSize, rankSize, userRankSize,
        deviceNumPerAggregation);

    constexpr u32 TWO_RANK_SIZE = 2;
    if (rankSize == TWO_RANK_SIZE) {
        return true;
    }

    // 通信步数为log_2(rankSize)向上取整
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, ++nSteps) {
    }

    float bandWidth; // 网卡出口带宽，用于计算大小包切片策略的阈值
    CHK_RET(GetBandWidthPerNPU(1, userRankSize, static_cast<u32>(deviceNumPerAggregation), bandWidth)); // 单位：GB/s

    // (公式解释)：
    // bandwidth_ - 网卡出口带宽，单位GB/s
    // LATENCY - 链路端到端静态时延
    // * 1000 - 转换为ns
    const float dataSizeBaseNum = bandWidth * LATENCY * 1000;
    constexpr u32 rankSizeOfSmallScale = 4; // 4以下节点数为小规模
    constexpr u32 dataSizeMultiple = 2;     // 通信数据量倍数为2

    // 计算阈值，用于判断大小包
    float thresholdDataSize;
    if (rankSize <= rankSizeOfSmallScale) {
        thresholdDataSize = rankSize * dataSizeBaseNum;
    } else {
        // 公式：用于计算大包处理的阈值
        thresholdDataSize = (nSteps + nSteps - dataSizeMultiple) / (nSteps - dataSizeMultiple) * dataSizeBaseNum;
    }

    return dataSize < thresholdDataSize; // 小数据量使用Binary broadcast
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_BROADCAST_NB_BINARY, BroadcastNBBinary);
} // namespace hccl
