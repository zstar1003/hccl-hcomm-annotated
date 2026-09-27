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
#include "reduce_nhr_oneshot.h"

namespace hccl {

ReduceNHROneshot::ReduceNHROneshot(const HcclDispatcher dispatcher) : NHRBase(dispatcher) {}

ReduceNHROneshot::~ReduceNHROneshot() {}

HcclResult ReduceNHROneshot::Prepare(u64 reduceAttrBitMap, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

HcclResult ReduceNHROneshot::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(SimpleCheck(rank, rankSize, links));
    HCCL_INFO(
        "[ReduceNHROneshot][RunAsync] run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank,
        rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(
        unitSize == 0, HCCL_ERROR("[ReduceNHROneshot][RunAsync] rank[%u] unit data size is zero", rank),
        HCCL_E_INTERNAL);

    // 如果ranksize为1, 从input->output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            return HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        HCCL_DEBUG("[ReduceNHROneshot]RunAsync for rankSize is 1 success");
        return HCCL_SUCCESS;
    }

    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);

    // 运行reduce, NHR 算法
    CHK_RET(RunReduceNHROneshot(rank, rankSize, links));

    HCCL_INFO("[ReduceNHROneshot][RunAsync] finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult ReduceNHROneshot::SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // 检查memory
    CHK_PRT_RET(
        !outputMem_ || !inputMem_,
        HCCL_ERROR("[ReduceNHROneshot][SimpleCheck] rank[%u] inputmem or outputmem is null", rank), HCCL_E_PTR);

    // 判断links数量是否正确
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[ReduceNHROneshot][SimpleCheck] rank[%u] link size[%llu] is "
            "less than rank size[%u]",
            rank, links.size(), rankSize),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceNHROneshot::SdmaRx(
    LINK& linkLeft, LINK& linkRight, [[maybe_unused]] InterServerAlgoStep& stepInfo,
    [[maybe_unused]] const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    u64 totalSize = count_ * SIZE_TABLE[dataType_];
    DeviceMem srcMem = inputMem_.range(0, totalSize);
    DeviceMem tempMem = scratchMem_.range(0, totalSize);

    if (linkRight != nullptr) {
        CHK_RET(linkRight->TxAck(stream_));
    }

    if (linkLeft != nullptr) {
        CHK_RET(linkLeft->RxAck(stream_));
        void* remoteMem = nullptr;
        CHK_RET(linkLeft->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMem));
        if ((INLINE_REDUCE_BITMASK & reduceAttr_) == 1) { // inlineReduce
            CHK_RET(HcclReduceAsync(
                dispatcher_, static_cast<s8*>(remoteMem) + baseOffset_, tempMem.size() / SIZE_TABLE[dataType_],
                dataType_, reductionOp_, stream_, srcMem.ptr(), linkLeft->GetRemoteRank(), linkLeft->GetLinkType(),
                INLINE_REDUCE_BIT));
        } else { // tbeReduce
            DeviceMem srcMemLeft(static_cast<s8*>(remoteMem) + baseOffset_, totalSize);
            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, tempMem, srcMemLeft, stream_,
                linkLeft->GetRemoteRank(), // left的inputMem拷到本端的scratchMem
                linkLeft->GetLinkType()));
            u64 dataCount = srcMem.size() / SIZE_TABLE[dataType_];
            ret = HcclReduceAsync(
                dispatcher_, tempMem.ptr(), dataCount, dataType_, reductionOp_, stream_, srcMem.ptr(),
                INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, reduceAttr_);
        }
        CHK_RET(linkLeft->TxDataSignal(stream_));
    }
    if (linkRight != nullptr) {
        CHK_RET(linkRight->RxDataSignal(stream_));
    }
    return ret;
}

HcclResult ReduceNHROneshot::RdmaTxRx(
    LINK& linkLeft, LINK& linkRight, [[maybe_unused]] InterServerAlgoStep& stepInfo,
    [[maybe_unused]] const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    u64 totalSize = count_ * SIZE_TABLE[dataType_];
    DeviceMem srcMem = inputMem_.range(0, totalSize);
    DeviceMem tempMem = scratchMem_.range(0, totalSize);

    if (linkLeft != nullptr) {
        ret = linkLeft->TxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceNHROneshot][RunReduceNHROneshot] TxAck failed"), ret);
    }

    if (linkRight != nullptr) {
        ret = linkRight->RxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceNHROneshot][RunReduceNHROneshot] RxAck failed"), ret);
        ret = senderInfo_->run(linkRight, baseOffset_, srcMem, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceNHROneshot][RunReduceNHROneshot] Tx failed"), ret);
    }

    if (linkLeft != nullptr) {
        ret = reducerInfo_->run(dispatcher_, linkLeft, baseOffset_, srcMem, srcMem, tempMem, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceNHROneshot][RunReduceNHROneshot] Rx failed"), ret);
    }

    if (barrierSwitchOn_) {
        CHK_RET(ExecuteBarrier(linkLeft, linkRight));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceNHROneshot::RunReduceNHROneshot(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 计算通信步数
    u32 nSteps = GetStepNumInterServer(rankSize);
    HCCL_DEBUG("[ReduceNHROneshot][RunReduceNHROneshot] rank[%u] rankSize[%u] nSteps[%u]", rank, rankSize, nSteps);

    // 逐步编排任务
    for (u32 step = 0; step < nSteps; step++) {
        InterServerAlgoStep stepInfo;
        GetStepInfo(step, nSteps, rank, rankSize, stepInfo);

        u32 sendTo = stepInfo.toRank;
        u32 recvFrom = stepInfo.fromRank;

        // 当前每个数据块发送一次ACK、reduce一次、同步一次
        HCCL_DEBUG("[ReduceNHROneshot][RunReduceNHROneshot] recvFrom[%u] sendTo[%u] step[%u]", recvFrom, sendTo, step);

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
HcclResult ReduceNHROneshot::GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep& stepInfo)
{
    (void)nSteps;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 1;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.step = step;
    stepInfo.myRank = rank;

    u32 nRanks = (rankSize - 1 + (1 << step)) / (1 << (step + 1)); // 本步需要进行收/发的rank数

    // 以0为root，第i步，0+deltaRankPair开始，每隔deltaRankGroup的rank需要发给rank-deltaRankPair
    u32 deltaRoot = (rank + rankSize - root_) % rankSize;

    u32 deltaRankPair = 1 << step;
    u32 deltaRankGroup = 1 << (step + 1);

    if (deltaRoot / deltaRankGroup < nRanks) {
        if ((deltaRoot + deltaRankPair) % deltaRankGroup == 0) {
            stepInfo.toRank = (rank + rankSize - deltaRankPair) % rankSize;
            stepInfo.txSliceIdxs.push_back(0);
        }

        if (deltaRoot % deltaRankGroup == 0) {
            stepInfo.fromRank = (rank + deltaRankPair) % rankSize;
            stepInfo.rxSliceIdxs.push_back(0);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
ReduceNHROneshot::ExecuteBarrier(const std::shared_ptr<Transport>& preLink, const std::shared_ptr<Transport>& aftLink)
{
    if (preLink != nullptr) {
        CHK_RET(preLink->TxAck(stream_));
    }
    if (aftLink != nullptr) {
        CHK_RET(aftLink->RxAck(stream_));
        CHK_RET(aftLink->TxDataSignal(stream_));
    }
    if (preLink != nullptr) {
        CHK_RET(preLink->RxDataSignal(stream_));
        CHK_RET(preLink->PostFinAck(stream_));
    }
    if (aftLink != nullptr) {
        CHK_RET(aftLink->WaitFinAck(stream_));
    }

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCE_NHR_ONE_SHOT, ReduceNHROneshot);
} // namespace hccl
