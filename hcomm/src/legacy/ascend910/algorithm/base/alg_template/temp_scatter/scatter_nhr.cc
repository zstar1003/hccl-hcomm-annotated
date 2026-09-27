/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_nhr.h"
#include "alg_template_register.h"

namespace hccl {
ScatterNHR::ScatterNHR(const HcclDispatcher dispatcher) : NHRBase(dispatcher), interRank_(0), interRankSize_(0) {}

ScatterNHR::~ScatterNHR() {}

HcclResult ScatterNHR::Prepare(bool needMerge)
{
    isNeedMerge = needMerge;
    return HCCL_SUCCESS;
}

// scatter的入口函数
HcclResult
ScatterNHR::RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links)
{
    // 从Broadcast调用Scatter需要merge
    if (isNeedMerge) {
        // 获取tree映射，存储到类对象的成员变量中
        GetRankMapping(rankSize);
    }
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[ScatterNHR][RunAsync] run_async inputmem or outputmem is null");
        return HCCL_E_PTR;
    }

    interRank_ = rank;
    interRankSize_ = rankSize;

    // ranksize为1时，只有当input!=output 时候进行拷贝
    if (interRankSize_ == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(
        unitSize == 0, HCCL_ERROR("[ScatterNHR][RunAsync] rank[%u] unit data size is zero", rank), HCCL_E_INTERNAL);

    // 带入vector为空，计算每个rank的结果偏移和大小
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, interRankSize_);
    }

    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR("[ScatterNHR][RunAsync] rank[%u] link size[%llu] is less than rank size", rank, links.size()),
        HCCL_E_INTERNAL);

    if (sliceMap_.size() != rankSize) {
        GetRankMapping(rankSize, true); // 没有初始化过，说明不是由allreduce或者bcast调入，需要保序
    }

    DeviceMem src;

    HcclResult ret = HCCL_SUCCESS;
    // 需要判断input不等于outputmem，scatter 输入只有一个input时不用拷贝
    if (inputMem_ != outputMem_) {
        u32 targetIdx = sliceMap_[interRank_];

        src = inputMem_.range(slices_[targetIdx].offset, slices_[targetIdx].size);
        ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, src, stream_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ScatterNHR][RunAsync] root rank[%u] memcpy async from input[%p] "
                "failed to output[%p]",
                interRank_, inputMem_.ptr(), outputMem_.ptr()),
            ret);
    }

    // 运行scatter, NHR 算法
    CHK_RET(RunScatterNHR(links));
    return HCCL_SUCCESS;
}

HcclResult ScatterNHR::SdmaRx(
    LINK& linkLeft, LINK& linkRight, InterServerAlgoStep& stepInfo, [[maybe_unused]] const std::vector<LINK>& links)
{
    if (linkRight != nullptr) {
        CHK_RET(linkRight->TxAck(stream_));
    }
    if (linkLeft != nullptr) {
        CHK_RET(linkLeft->RxAck(stream_));
        std::vector<Slice> rxSlices;
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            rxSlices.push_back(slices_[stepInfo.rxSliceIdxs[i]]);
        }
        MergeSlices(rxSlices);
        void* srcMemPtr = nullptr;
        CHK_RET(linkLeft->GetRemoteMem(UserMemType::OUTPUT_MEM, &srcMemPtr));
        for (const Slice& rxSlice : rxSlices) {
            DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
            DeviceMem srcMem(static_cast<s8*>(srcMemPtr) + baseOffset_ + rxSlice.offset, rxSlice.size);
            HCCL_DEBUG(
                "[ScatterNHR] rx dstMem[%p] range[%llu], size[%llu] ", dstMem.ptr(), rxSlice.offset, rxSlice.size);
            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, dstMem, srcMem, stream_, linkLeft->GetRemoteRank(), // Memecpy
                linkLeft->GetLinkType()));
        }
        CHK_RET(linkLeft->TxDataSignal(stream_)); // 告知left读完了
    }
    if (linkRight != nullptr) {
        CHK_RET(linkRight->RxDataSignal(stream_)); // 等right读完
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterNHR::RdmaTxRx(
    LINK& linkLeft, LINK& linkRight, InterServerAlgoStep& stepInfo, [[maybe_unused]] const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;

    if (linkLeft != nullptr) {
        CHK_RET(linkLeft->TxAck(stream_));
    }

    if (linkRight != nullptr) {
        CHK_RET(linkRight->RxAck(stream_));

        std::vector<Slice> txSlices;
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            txSlices.push_back(slices_[stepInfo.txSliceIdxs[i]]);
        }
        ret = Tx(linkRight, txSlices);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterNHR][RunScatterNHR] Tx failed"), ret);

        CHK_RET(linkRight->TxWaitDone(stream_));

        ret = linkRight->WaitFinAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterNHR][RunScatterNHR] WaitFinAck failed"), ret);
    }

    if (linkLeft != nullptr) {
        std::vector<Slice> rxSlices;
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            rxSlices.push_back(slices_[stepInfo.rxSliceIdxs[i]]);
        }
        ret = Rx(linkLeft, rxSlices);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterNHR][RunScatterNHR] Rx failed"), ret);

        CHK_RET(linkLeft->RxWaitDone(stream_));

        ret = linkLeft->PostFinAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterNHR][RunScatterNHR] PostFinAck failed"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterNHR::RunScatterNHR(const std::vector<std::shared_ptr<Transport>>& links)
{
    // 计算通信步数
    u32 nSteps = GetStepNumInterServer(interRankSize_);

    // 逐步编排任务
    for (u32 step = 0; step < nSteps; step++) {
        InterServerAlgoStep stepInfo;
        GetStepInfo(step, nSteps, interRank_, interRankSize_, stepInfo);

        HCCL_DEBUG(
            "[ScatterNHR][RunScatterNHR] rank[%u] recvFrom[%u] sendTo[%u] step[%u]", interRank_, stepInfo.fromRank,
            stepInfo.toRank, step);

        LINK linkLeft;
        LINK linkRight;
        if (stepInfo.txSliceIdxs.size() > 0) {
            linkRight = links[stepInfo.toRank];
            CHK_SMART_PTR_NULL(linkRight);
        }
        if (stepInfo.rxSliceIdxs.size() > 0) {
            linkLeft = links[stepInfo.fromRank];
            CHK_SMART_PTR_NULL(linkLeft);
        }

        if ((linkRight != nullptr && linkRight->IsSpInlineReduce())
            || (linkLeft != nullptr && linkLeft->IsSpInlineReduce())) {
            CHK_RET(SdmaRx(linkLeft, linkRight, stepInfo, links));
        } else {
            CHK_RET(RdmaTxRx(linkLeft, linkRight, stepInfo, links));
        }
    }
    return HCCL_SUCCESS;
}

void ScatterNHR::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = (totalCount / rankSize) * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
    }
    return;
}

HcclResult ScatterNHR::Tx(const LINK& link, std::vector<Slice>& txSlices)
{
    std::vector<TxMemoryInfo> txMems;

    HCCL_DEBUG("[ScatterNHR][Tx] txSlices size [%u]", txSlices.size());
    // 合并连续slices
    MergeSlices(txSlices);
    HCCL_DEBUG("[ScatterNHR][Tx] merged txSlices size [%u]", txSlices.size());

    for (const Slice& txSlice : txSlices) {
        DeviceMem srcMem = outputMem_.range(txSlice.offset, txSlice.size);
        HCCL_DEBUG("[ScatterNHR][Tx] tx srcMem[%p] range[%llu] size[%llu]", srcMem.ptr(), txSlice.offset, txSlice.size);
        txMems.emplace_back(
            TxMemoryInfo{UserMemType::OUTPUT_MEM, txSlice.offset + baseOffset_, srcMem.ptr(), txSlice.size});
    }

    CHK_RET(link->TxAsync(txMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult ScatterNHR::Rx(const LINK& link, std::vector<Slice>& rxSlices)
{
    std::vector<RxMemoryInfo> rxMems;

    HCCL_DEBUG("[ScatterNHR][Rx] rxslices size [%u]", rxSlices.size());
    // 合并连续slices
    MergeSlices(rxSlices);
    HCCL_DEBUG("[ScatterNHR][Rx] merged rxslices size [%u]", rxSlices.size());

    for (const Slice& rxSlice : rxSlices) {
        DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
        HCCL_DEBUG("[ScatterNHR][Rx] rx dstMem[%p] range[%llu] size[%llu]", dstMem.ptr(), rxSlice.offset, rxSlice.size);
        rxMems.emplace_back(
            RxMemoryInfo{UserMemType::OUTPUT_MEM, rxSlice.offset + baseOffset_, dstMem.ptr(), rxSlice.size});
    }

    CHK_RET(link->RxAsync(rxMems, stream_));
    return HCCL_SUCCESS;
}

// NHR每步的算法描述原理函数
HcclResult ScatterNHR::GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep& stepInfo)
{
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.step = step;
    stepInfo.myRank = rank;

    u32 deltaRoot = (root_ + rankSize - rank) % rankSize;
    u32 deltaRankPair = 1 << step;

    // 数据份数和数据编号增量
    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    // 判断是否是2的幂
    u32 nRanks = 0; // 本步需要进行收/发的rank数
    bool isPerfect = (rankSize & (rankSize - 1)) == 0;
    if (!isPerfect && step == nSteps - 1) {
        nRanks = rankSize - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) { // 需要发
        u32 sendTo = (rank + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = sliceMap_[txSliceIdx];
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.toRank = sendTo;
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom = (rank + deltaRankPair) % rankSize;
        u32 rxSliceIdx = rank;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = sliceMap_[rxSliceIdx];
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.fromRank = recvFrom;
        stepInfo.nSlices = nSlices;
    }
    return HCCL_SUCCESS;
}

HcclResult
ScatterNHR::GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
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

    u32 deltaRoot = (rankSize - rank) % rankSize;
    for (u32 step = 0; step < nSteps; step++) {
        u32 deltaRankPair = 1 << step;
        u32 nRanks = 0;
        bool isPerfect = (rankSize & (rankSize - 1)) == 0;
        if (!isPerfect && step == nSteps - 1) {
            nRanks = rankSize - deltaRankPair;
        } else {
            nRanks = deltaRankPair;
        }
        if (deltaRoot >= nRanks) {
            continue;
        }

        u32 sendTo = (rank + rankSize - deltaRankPair) % rankSize;
        LINK linkRight = links[sendTo];
        CHK_SMART_PTR_NULL(linkRight);

        NslbDpAdjInfo adjInfoStep = {};
        adjInfoStep.dstLocalRankId = linkRight->GetRemoteRank();
        adjInfoStep.phaseId = step + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_SCATTER_NHR, ScatterNHR);
} // namespace hccl
