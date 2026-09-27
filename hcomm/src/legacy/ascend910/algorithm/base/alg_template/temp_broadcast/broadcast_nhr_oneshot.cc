/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_nhr_oneshot.h"
#include <cmath>
#include "alg_template_register.h"

namespace hccl {
BroadcastNHROneshot::BroadcastNHROneshot(const HcclDispatcher dispatcher)
    : NHRBase(dispatcher),
      localBaseOffset_(0),
      isForAllReduce_(false)
{}

BroadcastNHROneshot::~BroadcastNHROneshot() {}

HcclResult BroadcastNHROneshot::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(SimpleCheck(rank, rankSize, links));
    HCCL_INFO(
        "[BroadcastNHROneshot][RunAsync] rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(unitSize == 0, HCCL_ERROR("[BroadcastNHROneshot][RunAsync] unitSize is zero"), HCCL_E_INTERNAL);

    if (!isForAllReduce_) {
        localBaseOffset_ = baseOffset_; // broadcast的本地偏移量和baseOffset_一致
    }

    // 双buffer下, 先将input拷贝到output的合适位置
    if (inputMem_ != outputMem_ && rank == root_) {
        u64 totalSize = count_ * SIZE_TABLE[dataType_];
        DeviceMem src = inputMem_.range(localBaseOffset_, totalSize);
        DeviceMem dst = outputMem_.range(localBaseOffset_, totalSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    // 如果ranksize为1, 从input->output就结束
    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }

    // 运行bcast， rst算法
    CHK_RET(RunBroadcastNHROneshot(rank, rankSize, links));

    HCCL_INFO("[BroadcastNHROneshot][RunAsync] finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult BroadcastNHROneshot::RunAsyncForAllReduce(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    isForAllReduce_ = true;
    return RunAsync(rank, rankSize, links);
}

HcclResult BroadcastNHROneshot::SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // 检查memory
    CHK_PRT_RET(
        !outputMem_ || !inputMem_,
        HCCL_ERROR("[BroadcastNHROneshot][SimpleCheck] rank[%u] inputmem or outputmem is null", rank), HCCL_E_PTR);

    // 判断links数量是否正确
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[BroadcastNHROneshot][SimpleCheck] rank[%u] link size[%llu] is "
            "less than rank size[%u]",
            rank, links.size(), rankSize),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult BroadcastNHROneshot::SdmaRx(
    LINK& linkLeft, LINK& linkRight, [[maybe_unused]] InterServerAlgoStep& stepInfo,
    [[maybe_unused]] const std::vector<LINK>& links)
{
    u64 totalSize = count_ * SIZE_TABLE[dataType_];
    DeviceMem srcMem = outputMem_.range(localBaseOffset_, totalSize);

    if (linkRight != nullptr) {
        CHK_RET(linkRight->TxAck(stream_));
    }
    if (linkLeft != nullptr) {
        CHK_RET(linkLeft->RxAck(stream_));
        void* srcMemPtr = nullptr;
        CHK_RET(linkLeft->GetRemoteMem(UserMemType::OUTPUT_MEM, &srcMemPtr));
        DeviceMem srcMemLeft(static_cast<s8*>(srcMemPtr) + baseOffset_, totalSize);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, srcMem, srcMemLeft, stream_, linkLeft->GetRemoteRank(), // Memecpy
            linkLeft->GetLinkType()));
        CHK_RET(linkLeft->TxDataSignal(stream_));
    }
    if (linkRight != nullptr) {
        CHK_RET(linkRight->RxDataSignal(stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastNHROneshot::RdmaTxRx(
    LINK& linkLeft, LINK& linkRight, [[maybe_unused]] InterServerAlgoStep& stepInfo,
    [[maybe_unused]] const std::vector<LINK>& links)
{
    u64 totalSize = count_ * SIZE_TABLE[dataType_];
    DeviceMem srcMem = outputMem_.range(localBaseOffset_, totalSize);

    if (linkLeft != nullptr) {
        CHK_RET(linkLeft->TxAck(stream_));
    }

    if (linkRight != nullptr) {
        CHK_RET(linkRight->RxAck(stream_));
        CHK_RET(linkRight->TxAsync(UserMemType::OUTPUT_MEM, baseOffset_, srcMem.ptr(), srcMem.size(), stream_));
        CHK_RET(linkRight->WaitFinAck(stream_));
    }

    if (linkLeft != nullptr) {
        CHK_RET(linkLeft->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_, srcMem.ptr(), srcMem.size(), stream_));
        CHK_RET(linkLeft->PostFinAck(stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastNHROneshot::RunBroadcastNHROneshot(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    // 计算通信步数
    u32 nSteps = GetStepNumInterServer(rankSize);
    HCCL_DEBUG(
        "[BroadcastNHROneshot][RunBroadcastNHROneshot] rank[%u] rankSize[%u] nSteps[%u]", rank, rankSize, nSteps);

    // 逐步编排任务
    for (u32 step = 0; step < nSteps; step++) {
        InterServerAlgoStep stepInfo;
        GetStepInfo(step, nSteps, rank, rankSize, stepInfo);

        HCCL_DEBUG(
            "[BroadcastNHROneshot][RunBroadcastNHROneshot] recvFrom[%u] sendTo[%u] step[%u]", stepInfo.fromRank,
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

// NHR每步的算法描述原理函数
HcclResult BroadcastNHROneshot::GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep& stepInfo)
{
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 1;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.step = step;
    stepInfo.myRank = rank;

    u32 nRanks = (rankSize - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step)); // 本步需要进行收/发的rank数

    u32 deltaRoot = (rank + rankSize - root_) % rankSize;

    u32 deltaRankPair = 1 << (nSteps - 1 - step);
    u32 deltaRankGroup = 1 << (nSteps - step);

    if (deltaRoot / deltaRankGroup < nRanks) {
        if (deltaRoot % deltaRankGroup == 0) {
            stepInfo.toRank = (rank + deltaRankPair) % rankSize;
            stepInfo.txSliceIdxs.push_back(0);
        }

        if ((deltaRoot + deltaRankPair) % deltaRankGroup == 0) {
            stepInfo.fromRank = (rank + rankSize - deltaRankPair) % rankSize;
            stepInfo.rxSliceIdxs.push_back(0);
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_BROADCAST_NHR_ONESHOT, BroadcastNHROneshot);
} // namespace hccl
