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
#include "broadcast_oneshot_pub.h"
#include "alg_template_register.h"

namespace hccl {
BroadcastHD::BroadcastHD(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

BroadcastHD::~BroadcastHD() {}

HcclResult BroadcastHD::Prepare(
    DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
    const Stream& stream, const HcclReduceOp reductionOp, const u32 root, std::vector<Stream>& meshStreams,
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 interRank, const HcomCollOpInfo* opInfo)
{
    localRank_ = interRank;
    meshStreams_ = meshStreams;
    meshSignalPtr_ = &meshSignal;
    meshSignalAuxPtr_ = &meshSignalAux;
    opInfo_ = opInfo;
    return AlgTemplateBase::Prepare(inputMem, outputMem, scratchMem, count, dataType, stream, reductionOp, root);
}

HcclResult BroadcastHD::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalAuxPtr_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAuxPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastHD::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalAuxPtr_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastHD::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalPtr_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignalPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastHD::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalPtr_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastHD::PrepareStep(u32 rankSize)
{
    u32 step;
    for (u32 rank = 0; rank < rankSize; rank++) {
        step = (rank == root_) ? 0 : static_cast<u32>(log2((rank - root_ + rankSize) % rankSize));
        stepMap_[rank] = step;
    }

    return HCCL_SUCCESS;
}

// 算法的函数入口
HcclResult BroadcastHD::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "BroadcastHD run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[BroadcastHD][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    if (meshStreams_.size() < 1) {
        HCCL_ERROR(
            "[BroadcastHD][RunAsync]rank[%u] meshStreams_[%llu] is less than need[1]", rank, meshStreams_.size());
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PrepareStep(rankSize));

    emptyMem_ = outputMem_.range(0, 0);
    nSteps_ = static_cast<u32>(log2(rankSize * base - 1));

    for (u32 step = stepMap_[rank]; step < nSteps_ - 1; step++) {
        if (step == stepMap_[rank]) {
            if (step != 0) {
                ret = RunReceive(rank, step, rankSize, links);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[BroadcastHD][RunAsync]rank[%u] count[%llu] step [%llu] failed in RunReceive step", rank,
                        count_, step),
                    ret);
            } else if (rank != root_) {
                ret = RunReceiveFirst(rank, rankSize, links);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[BroadcastHD][RunAsync]rank[%u] count[%llu] step [%llu] failed in RunReceiveFirst step", rank,
                        count_, step),
                    ret);
            } else {
                ret = RunSendFirst(rank, rankSize, links);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[BroadcastHD][RunAsync]rank[%u] count[%llu] step [%llu] failed in RunSendFirst step", rank,
                        count_, step),
                    ret);
            }
        } else {
            ret = RunSend(rank, step, rankSize, links);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[BroadcastHD][RunAsync]rank[%u] count[%llu] step [%llu] failed in RunSend step", rank, count_,
                    step),
                ret);
        }
    }
    ret = RunFinalStep(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadcastHD][RunAsync]rank[%u] count[%llu] failed in RunFinalStep", rank, count_), ret);
    HCCL_INFO("BroadcastHD finished: rank[%u] ranksize[%u].", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult BroadcastHD::RunFinalStep(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 half = static_cast<u32>(pow(2, nSteps_ - 1));
    u32 logicRank = (rank - root_ + rankSize) % rankSize;
    if ((logicRank % half) < (rankSize - half)) {
        if (stepMap_[rank] == (nSteps_ - 1)) {
            ret = RunReceive(rank, nSteps_ - 1, rankSize, links);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[BroadcastHD][RunAsync]rank[%u] count[%llu] step [%llu] failed in RunReceive step", rank, count_,
                    nSteps_ - 1),
                ret);
        } else {
            ret = RunSend(rank, nSteps_ - 1, rankSize, links);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[BroadcastHD][RunAsync]rank[%u] count[%llu] step [%llu] failed in RunSend step", rank, count_,
                    nSteps_ - 1),
                ret);
        }
    } else {
        u32 unitSize = SIZE_TABLE[dataType_];
        DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);
        DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), count_ * unitSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemIn, commMemOut, stream_));
        HCCL_INFO("final local cpy step %llu, rank %llu", nSteps_ - 1, rank);
    }
    return HCCL_SUCCESS;
}

u32 BroadcastHD::GetDstRank(u32 rank, u32 step, u32 rankSize)
{
    u32 logicRank = (rank - root_ + rankSize) % rankSize;
    u32 logicDstRank = logicRank ^ (1 << step);
    return (logicDstRank + root_) % rankSize;
}

HcclResult BroadcastHD::RunSend(u32 rank, u32 step, u32 rankSize, const std::vector<LINK>& links)
{
    u32 dstRank = GetDstRank(rank, step, rankSize);
    HCCL_INFO("RunSend: rank[%u] dstRank[%u] step [%u] count[%llu].", rank, dstRank, step, count_);
    // 数据准备
    u32 unitSize = SIZE_TABLE[dataType_];

    if (step == (nSteps_ - 1)) {
        CHK_RET(MainRecordSub());
        CHK_RET(SubWaitMain());
        if (rank != root_) {
            DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);
            DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), count_ * unitSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemIn, commMemOut, meshStreams_[0]));
        }
    }

    CHK_RET(links[dstRank]->TxAck(stream_));
    CHK_RET(links[dstRank]->RxDataSignal(stream_));

    if (step == (nSteps_ - 1)) {
        CHK_RET(SubRecordMain());
        CHK_RET(MainWaitSub());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyMem_, emptyMem_, stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastHD::RunReceive(u32 rank, u32 step, u32 rankSize, const std::vector<LINK>& links)
{
    u32 dstRank = GetDstRank(rank, step, rankSize);
    HCCL_INFO("RunReceive: rank[%u] step[%u] outputMem[%p] count[%llu].", rank, step, outputMem_.ptr(), count_);

    // 数据准备
    u32 unitSize = SIZE_TABLE[dataType_];
    DeviceMem dst;
    if (step == nSteps_ - 1) {
        dst = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);
    } else {
        dst = outputMem_.range(0, count_ * unitSize);
    }

    CHK_RET(links[dstRank]->RxAck(stream_));
    void* remMemPtr = nullptr;
    CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
    DeviceMem src = DeviceMem::create(static_cast<u8*>(remMemPtr), count_ * unitSize);
    CHK_RET(HcclD2DMemcpyAsync(
        dispatcher_, dst, src, stream_, links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
    CHK_RET(links[dstRank]->TxDataSignal(stream_));
    return HCCL_SUCCESS;
}

HcclResult BroadcastHD::RunSendFirst(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 dstRank = GetDstRank(rank, 0, rankSize);
    HCCL_INFO("RunSendFirst: rank[%u] dstRank[%u] count[%llu].", rank, dstRank, count_);
    // 数据准备
    u32 unitSize = SIZE_TABLE[dataType_];
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyMem_, emptyMem_, stream_));
    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), count_ * unitSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, commMemOut, userMemIn, meshStreams_[0]));

    CHK_RET(links[dstRank]->RxAck(stream_));
    void* remMemPtr = nullptr;
    CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
    DeviceMem dst = DeviceMem::create(static_cast<u8*>(remMemPtr), count_ * unitSize);
    CHK_RET(HcclD2DMemcpyAsync(
        dispatcher_, dst, userMemIn, stream_, links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
    CHK_RET(links[dstRank]->TxDataSignal(stream_));

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    return HCCL_SUCCESS;
}

HcclResult BroadcastHD::RunReceiveFirst(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 dstRank = GetDstRank(rank, 0, rankSize);
    HCCL_INFO("RunReceiveFirst: rank[%u] dstRank[%u] count[%llu].", rank, dstRank, count_);
    // 数据准备
    u32 unitSize = SIZE_TABLE[dataType_];

    CHK_RET(links[dstRank]->TxAck(stream_));
    CHK_RET(links[dstRank]->RxDataSignal(stream_));

    if (nSteps_ == 1) {
        DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);
        DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), count_ * unitSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemIn, commMemOut, stream_));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_BROADCAST_HD, BroadcastHD);
} // namespace hccl
