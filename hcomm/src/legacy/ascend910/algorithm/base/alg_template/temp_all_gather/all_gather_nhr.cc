/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_nhr.h"
#include <cmath>
#include "alg_template_register.h"

namespace hccl {
AllGatherNHR::AllGatherNHR(const HcclDispatcher dispatcher) : NHRBase(dispatcher) {}

AllGatherNHR::~AllGatherNHR() {}

HcclResult AllGatherNHR::Prepare(bool needMerge)
{
    isNeedMerge = needMerge;
    return HCCL_SUCCESS;
}

// 服务器间allgather的入口函数
HcclResult AllGatherNHR::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 从AllReduce或者Broadcast调用AllGather需要merge
    if (isNeedMerge) {
        // 获取tree映射，存储到类对象的成员变量中
        GetRankMapping(rankSize);
    }
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "[AllGatherNHR][RunAsync] rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        links.size() < rankSize, HCCL_ERROR("[AllGatherNHR][RunAsync] rank[%u] linkSize is less than rankSize", rank),
        HCCL_E_INTERNAL);

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(unitSize == 0, HCCL_ERROR("[AllGatherNHR][RunAsync]rank[%u] unitSize is zero", rank), HCCL_E_INTERNAL);

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
                "[AllGatherNHR][RunAsync] rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", rank, i,
                slices_[i].offset, i, slices_[i].size);
        }
    }

    if (sliceMap_.size() != rankSize) {
        GetRankMapping(rankSize, true); // 没有初始化过，说明不是由allreduce或者bcast调入，需要保序
    }

    // 双buffer下, 先将input拷贝到output的合适位置
    if (inputMem_ != outputMem_) {
        u32 targetIdx = sliceMap_[rank];
        DeviceMem dst = outputMem_.range(slices_[targetIdx].offset, slices_[targetIdx].size);
        DeviceMem src = inputMem_.range(inputSlices[targetIdx].offset, inputSlices[targetIdx].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    // 运行all-gather, ring算法
    CHK_RET(RunAllGather(rank, rankSize, slices_, links));

    HCCL_INFO("[AllGatherNHR][RunAsync] AllGatherNHR finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHR::SdmaRx(const LINK& linkLeft, const LINK& linkRight, std::vector<Slice>& rxSlices)
{
    CHK_RET(linkRight->TxAck(stream_)); // 告知right可以从我这读了
    CHK_RET(linkLeft->RxAck(stream_));  // 等left告知可以从他那读了
    void* srcMemPtr = nullptr;
    CHK_RET(linkLeft->GetRemoteMem(UserMemType::OUTPUT_MEM, &srcMemPtr));
    for (const Slice& rxSlice : rxSlices) {
        DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
        DeviceMem srcMem(static_cast<s8*>(srcMemPtr) + baseOffset_ + rxSlice.offset, rxSlice.size);
        HCCL_DEBUG("[AllGatherNHR] rx dstMem[%p] range[%llu], size[%llu] ", dstMem.ptr(), rxSlice.offset, rxSlice.size);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dstMem, srcMem, stream_, linkLeft->GetRemoteRank(), // Memecpy
            linkLeft->GetLinkType()));
    }
    CHK_RET(linkLeft->TxDataSignal(stream_));  // 告知left我读完了
    CHK_RET(linkRight->RxDataSignal(stream_)); // 等right读完

    return HCCL_SUCCESS;
}

HcclResult AllGatherNHR::RdmaTxRx(
    const LINK& linkLeft, const LINK& linkRight, InterServerAlgoStep& stepInfo, std::vector<Slice>& txSlices,
    std::vector<Slice>& rxSlices)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_RET(linkLeft->TxAck(stream_));
    CHK_RET(linkRight->RxAck(stream_));
    ret = Tx(linkRight, txSlices);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllGatherNHR][RunAllGather] rank[%u] round[%u] "
            "tx %u slices failed",
            stepInfo.myRank, stepInfo.step, stepInfo.nSlices),
        ret);
    ret = Rx(linkLeft, rxSlices);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllGatherNHR][RunAllGather] rank[%u] round[%u] "
            "rx %u slices failed",
            stepInfo.myRank, stepInfo.step, stepInfo.nSlices),
        ret);

    ret = linkLeft->PostFinAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherNHR][RunAllGather] PostFinAck failed"), ret);
    ret = linkRight->WaitFinAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherNHR][RunAllGather] WaitFinAck failed"), ret);
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHR::RunAllGather(
    u32 rank, u32 rankSize, const std::vector<Slice>& outputSlices, const std::vector<LINK>& links)
{
    CHK_PRT_RET(
        outputSlices.size() < rankSize,
        HCCL_ERROR("[AllGatherNHR][RunAllGather] rank[%u] OutputSlice Size is less than rank size", rank),
        HCCL_E_INTERNAL);
    HcclResult ret = HCCL_SUCCESS;

    // 计算通信步数
    u32 nSteps = GetStepNumInterServer(rankSize);

    // 逐步编排任务
    for (u32 step = 0; step < nSteps; step++) {
        InterServerAlgoStep stepInfo;
        GetStepInfo(step, nSteps, rank, rankSize, stepInfo);

        LINK linkLeft = links[stepInfo.fromRank];
        CHK_SMART_PTR_NULL(linkLeft);

        LINK linkRight = links[stepInfo.toRank];
        CHK_SMART_PTR_NULL(linkRight);

        std::vector<Slice> txSlices;
        std::vector<Slice> rxSlices;

        HCCL_DEBUG(
            "[AllGatherNHR][RunAllGather] rank[%u] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] "
            "nSlices[%u]",
            rank, rankSize, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            txSlices.push_back(outputSlices[stepInfo.txSliceIdxs[i]]);
            rxSlices.push_back(outputSlices[stepInfo.rxSliceIdxs[i]]);

            HCCL_DEBUG(
                "[AllGatherNHR][RunAllGather] i[%u] rxSliceIndex[%u] txSliceIndex[%u] rx data offset[%llu] "
                "size[%llu]",
                i, stepInfo.rxSliceIdxs[i], stepInfo.txSliceIdxs[i], outputSlices[stepInfo.rxSliceIdxs[i]].offset,
                outputSlices[stepInfo.rxSliceIdxs[i]].size);
        }

        // 合并连续slices
        MergeSlices(rxSlices);
        MergeSlices(txSlices);

        if (linkLeft->IsSpInlineReduce() && linkRight->IsSpInlineReduce()) {
            ret = SdmaRx(linkLeft, linkRight, rxSlices);
        } else {
            ret = RdmaTxRx(linkLeft, linkRight, stepInfo, txSlices, rxSlices);
        }
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherNHR][RunAllGather] RunAllGather failed"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHR::Tx(const LINK& link, std::vector<Slice>& txSlices)
{
    std::vector<TxMemoryInfo> txMems;
    for (const Slice& txSlice : txSlices) {
        DeviceMem srcMem = outputMem_.range(txSlice.offset, txSlice.size);
        HCCL_DEBUG(
            "[AllGatherNHR][Tx] tx srcMem[%p] offset[%llu] size[%llu]", srcMem.ptr(), txSlice.offset, txSlice.size);
        txMems.emplace_back(
            TxMemoryInfo{UserMemType::OUTPUT_MEM, txSlice.offset + baseOffset_, srcMem.ptr(), txSlice.size});
    }

    CHK_RET(link->TxAsync(txMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHR::Rx(const LINK& link, std::vector<Slice>& rxSlices)
{
    std::vector<RxMemoryInfo> rxMems;
    for (const Slice& rxSlice : rxSlices) {
        DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
        HCCL_DEBUG(
            "[AllGatherNHR][Rx] rx dstMem[%p] range[%llu], size[%llu] ", dstMem.ptr(), rxSlice.offset, rxSlice.size);
        rxMems.emplace_back(
            RxMemoryInfo{UserMemType::OUTPUT_MEM, rxSlice.offset + baseOffset_, dstMem.ptr(), rxSlice.size});
    }

    CHK_RET(link->RxAsync(rxMems, stream_));
    return HCCL_SUCCESS;
}

// NHR每步的算法描述原理函数
HcclResult AllGatherNHR::GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep& stepInfo)
{
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    u32 sliceSize = slices_.size() / rankSize;
    stepInfo.step = step;
    stepInfo.myRank = rank;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (rank + rankSize - deltaRank) % rankSize;
    u32 sendTo = (rank + deltaRank) % rankSize;

    // 数据份数和数据编号增量
    u32 nSlices = (rankSize - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = rank;
    u32 rxSliceIdx = (rank - (1 << (nSteps - 1 - step)) + rankSize) % rankSize;

    stepInfo.nSlices = nSlices * sliceSize;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (u32 i = 0; i < nSlices; i++) {
        for (u32 j = 0; j < sliceSize; j++) {
            u32 targetTxSliceIdx = sliceMap_[txSliceIdx];
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx * sliceSize + j);

            u32 targetRxSliceIdx = sliceMap_[rxSliceIdx];
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx * sliceSize + j);

            HCCL_DEBUG(
                "[AllGatherNHR][GetStepInfo] i[%u] txSliceIdx[%u]->targetTxSliceIdx[%u] rxSliceIdx[%u]->"
                "targetRxSliceIdx[%u]",
                i, txSliceIdx, targetTxSliceIdx, rxSliceIdx, targetRxSliceIdx);
        }
        txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
    }
    return HCCL_SUCCESS;
}

HcclResult
AllGatherNHR::GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
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
        u32 deltaRank = 1 << (nSteps - 1 - step);
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
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_NHR, AllGatherNHR);
} // namespace hccl
