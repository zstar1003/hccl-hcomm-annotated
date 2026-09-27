/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "alg_template_register.h"
#include "all_reduce_hd_optim_pub.h"
namespace hccl {
AllReduceHDOptim::AllReduceHDOptim(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllReduceHDOptim::~AllReduceHDOptim() {}

HcclResult AllReduceHDOptim::Prepare(
    u64 reduceAttrBitMap, std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
    std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 userRank, HcomCollOpInfo* opInfo, bool aicpu)
{
    reduceAttr_ = reduceAttrBitMap;
    userRank_ = userRank;
    meshStreams_ = meshStreams;
    meshSignal_ = &meshSignal;
    meshSignalAux_ = &meshSignalAux;
    opInfo_ = opInfo;
    aicpu_ = aicpu;
    return HCCL_SUCCESS;
}

HcclResult AllReduceHDOptim::MainRecordSub(u32 streamNum)
{
    if (aicpu_) {
        return HCCL_SUCCESS;
    }
    for (u32 signalIndex = 0; signalIndex < streamNum; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAux_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceHDOptim::SubWaitMain(u32 streamNum)
{
    if (aicpu_) {
        return HCCL_SUCCESS;
    }
    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAux_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceHDOptim::MainWaitSub(u32 streamNum)
{
    if (aicpu_) {
        return HCCL_SUCCESS;
    }
    for (u32 signalIndex = 0; signalIndex < streamNum; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignal_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceHDOptim::SubRecordMain(u32 streamNum)
{
    if (aicpu_) {
        return HCCL_SUCCESS;
    }
    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignal_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// allreduce算法的函数入口
HcclResult AllReduceHDOptim::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "AllReduceHDOptim run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        outputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[AllReduceHDOptim][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    if (meshStreams_.size() < base) {
        HCCL_ERROR(
            "[AllReduceHDOptim][RunAsync]rank[%u] meshStreams_[%llu] is less than need[%u]", rank, meshStreams_.size(),
            base);
        return HCCL_E_INTERNAL;
    }
    u32 totalSize = SIZE_TABLE[dataType_] * count_;
    userMemIn = DeviceMem::create(opInfo_->inputAddr, totalSize);
    userMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize);
    emptyMem_ = outputMem_.range(0, 0);
    nSteps = static_cast<u32>(log2(rankSize));
    stepPow = static_cast<u32>(pow(base, nSteps));

    ret = RunPreCopy(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceHDOptim][RunAsync]rank[%u] count[%llu] failed RunPreCopy step", rank, count_), ret);

    if (rank < stepPow) {
        ret = RunAllReduceHDOptim(rank, rankSize, links);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[AllReduceHDOptim][RunAsync]rank[%u] count[%llu] failed RunAllReduceHDOptim step", rank, count_),
            ret);
    }

    if (stepPow != rankSize) {
        ret = RunFinalStep(rank, rankSize, links);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[AllReduceHDOptim][RunAsync]rank[%u] count[%llu] failed RunFinalStep step", rank, count_), ret);
    }

    HCCL_INFO("AllReduceHDOptim finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceHDOptim::RunPreCopy(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 totalSize = SIZE_TABLE[dataType_] * count_;

    DeviceMem src = userMemIn.range(0, totalSize);
    DeviceMem dst = outputMem_.range(0, totalSize);
    DeviceMem nextDst = outputMem_.range(totalSize, totalSize);

    if (stepPow == rankSize) {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, nextDst, src, stream_));
        return HCCL_SUCCESS;
    }

    u32 neighCur = rank ^ (1 << nSteps);
    if (neighCur < rankSize) {
        // reduce写
        if (rank >= pow(base, nSteps)) {
            CHK_PTR_NULL(links[neighCur]);
            CHK_RET(links[neighCur]->RxAck(stream_));
            void* remMemPtr = nullptr;
            CHK_RET(links[neighCur]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
            dst = DeviceMem::create(static_cast<u8*>(remMemPtr), totalSize);
            CHK_RET(HcclReduceAsync(
                dispatcher_, static_cast<void*>(src.ptr()), count_, dataType_, reductionOp_, stream_,
                static_cast<void*>(dst.ptr()), links[neighCur]->GetRemoteRank(), links[neighCur]->GetLinkType(),
                INLINE_REDUCE_BIT));
            CHK_RET(links[neighCur]->TxDataSignal(stream_));
        } else {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
            CHK_RET(links[neighCur]->TxAck(stream_));
            CHK_RET(links[neighCur]->RxDataSignal(stream_));
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, nextDst, dst, stream_));
        }
    } else {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, nextDst, dst, stream_));
    }

    return HCCL_SUCCESS;
}

HcclResult AllReduceHDOptim::RunBetweenStep(
    u32 rank, u32 step, u32 neighBefore, u32 neighNext, u32 rankSize, const std::vector<LINK>& links)
{
    (void)rank;
    u32 totalSize = SIZE_TABLE[dataType_] * count_;

    DeviceMem src;
    DeviceMem dst;

    if ((step == 1) && (stepPow == rankSize)) {
        // 二次幂整第一步写 串行同步
        CHK_RET(links[neighBefore]->TxDataSignal(stream_));
        CHK_RET(links[neighBefore]->RxDataSignal(stream_));
        CHK_RET(MainRecordSub(base));
        CHK_RET(SubWaitMain(base));
    } else {
        CHK_RET(MainRecordSub(base));
        CHK_RET(SubWaitMain(base));
        CHK_RET(links[neighBefore]->TxDataSignal(stream_));
        CHK_RET(links[neighBefore]->RxDataSignal(stream_));
    }

    src = outputMem_.range(step * totalSize, totalSize);
    if ((step == (nSteps - 1)) && (static_cast<u32>(pow(base, nSteps)) == rankSize)) {
        dst = userMemOut.range(0, totalSize);
    } else {
        dst = outputMem_.range((step + 1) * totalSize, totalSize);
    }
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, aicpu_ ? stream_ : meshStreams_[0]));

    CHK_RET(links[neighNext]->TxAck(aicpu_ ? stream_ : meshStreams_[1]));
    CHK_RET(links[neighNext]->RxAck(aicpu_ ? stream_ : meshStreams_[1]));

    CHK_RET(SubRecordMain(base));
    CHK_RET(MainWaitSub(base));
    return HCCL_SUCCESS;
}

HcclResult AllReduceHDOptim::RunAllReduceHDOptim(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 unitSize = SIZE_TABLE[dataType_];
    u32 totalSize = unitSize * count_;

    DeviceMem src;
    DeviceMem dst;

    u32 neighCur = rank ^ (1 << 0);
    CHK_RET(links[neighCur]->TxAck(stream_));
    CHK_RET(links[neighCur]->RxAck(stream_));

    u32 neighNext = 0;
    void* remMemPtr = nullptr;
    for (u32 step = 1; step <= nSteps; step++) {
        if ((step != nSteps) || (stepPow != rankSize)) {
            dst = outputMem_.range(step * totalSize, totalSize);
        } else {
            dst = userMemOut.range(0, totalSize);
        }
        CHK_RET(links[neighCur]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        src = DeviceMem::create(static_cast<u8*>(remMemPtr) + (step - 1) * totalSize, totalSize);
        if ((step == 1) && (stepPow == rankSize)) {
            // 二次幂整第一步写
            src = userMemIn.range(0, totalSize);
            dst = DeviceMem::create(static_cast<u8*>(remMemPtr) + totalSize, totalSize);
        }

        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<void*>(src.ptr()), count_, dataType_, reductionOp_, stream_,
            static_cast<void*>(dst.ptr()), links[neighCur]->GetRemoteRank(), links[neighCur]->GetLinkType(),
            INLINE_REDUCE_BIT));

        if (step != nSteps) {
            neighNext = rank ^ (1 << (step));
            CHK_RET(RunBetweenStep(rank, step, neighCur, neighNext, rankSize, links));
            neighCur = neighNext;
        }
    }

    CHK_RET(links[neighCur]->TxDataSignal(stream_));
    CHK_RET(links[neighCur]->RxDataSignal(stream_));

    return HCCL_SUCCESS;
}

HcclResult AllReduceHDOptim::RunFinalStep(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 unitSize = SIZE_TABLE[dataType_];
    u32 totalSize = unitSize * count_;

    DeviceMem src = outputMem_.range(nSteps * totalSize, totalSize);
    DeviceMem dst = userMemOut.range(0, totalSize);

    u32 neighCur = rank ^ (1 << nSteps);
    if (neighCur >= rankSize) {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    } else if (rank < pow(base, nSteps)) {
        CHK_RET(links[neighCur]->TxAck(stream_));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
        CHK_RET(links[neighCur]->RxDataSignal(stream_));
    } else {
        CHK_RET(links[neighCur]->RxAck(stream_));
        void* remMemPtr = nullptr;
        CHK_RET(links[neighCur]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        src = DeviceMem::create(static_cast<u8*>(remMemPtr) + nSteps * totalSize, totalSize);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, stream_, links[neighCur]->GetRemoteRank(), links[neighCur]->GetLinkType()));
        CHK_RET(links[neighCur]->TxDataSignal(stream_));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_HD_OPTIM, AllReduceHDOptim);
} // namespace hccl
