/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_reduce_broadcast.h"

namespace hccl {
AllReduceReduceBcast::AllReduceReduceBcast(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllReduceReduceBcast::~AllReduceReduceBcast() {}

HcclResult AllReduceReduceBcast::Prepare(PrepareData& param)
{
    reduceAttr_ = param.reduceAttr;
    localRank_ = param.interRank;
    localRankSize_ = param.interRankSize;
    userRank_ = param.userRank;
    meshStreams_ = *param.subStreamsPtr;
    meshSignalPtr_ = param.signalPtr;
    meshSignalAuxPtr_ = param.signalAuxPtr;
    opInfo_ = param.opInfo;

    return AlgTemplateBase::Prepare(
        param.inputMem, param.outputMem, param.scratchMem, param.count, param.dataType, param.stream, param.reductionOp,
        LEVEL0_BRIDGE_RANK_ID, *param.slicesPtr, 0);
}

HcclResult AllReduceReduceBcast::MainRecordSub()
{
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux = *meshSignalAuxPtr_;
    for (u32 signalIndex = 0; signalIndex < meshSignalAux.size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, meshSignalAux[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceReduceBcast::SubWaitMain()
{
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux = *meshSignalAuxPtr_;
    for (u32 streamIndex = 0; streamIndex < meshSignalAux.size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, meshSignalAux[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceReduceBcast::MainWaitSub()
{
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignal = *meshSignalPtr_;
    for (u32 signalIndex = 0; signalIndex < meshSignal.size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, meshSignal[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceReduceBcast::SubRecordMain()
{
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignal = *meshSignalPtr_;
    for (u32 streamIndex = 0; streamIndex < meshSignal.size(); streamIndex++) {
        CHK_RET(
            LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, meshSignal[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// 将数据均分，最小单位是128

// ring allreduce算法的函数入口
HcclResult AllReduceReduceBcast::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "AllReduceReduceBcast run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[AllReduceReduceBcast][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    // 如果ranksize为1, 从input->output
    if (rankSize == 1) {
        HCCL_DEBUG("[AllReduceReduceBcast][RunAsync]rankSize is %u", rankSize);
        if (opInfo_->inputAddr != opInfo_->outputAddr) {
            DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * DataUnitSize(dataType_));
            DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, count_ * DataUnitSize(dataType_));
            ret = HcclD2DMemcpyAsync(dispatcher_, userMemOut, userMemIn, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceReduceBcast][RunAsync]rank[%u] memcpy async failed", rank),
                ret);
        }
        return ret;
    }

    ret = RunReduce(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceReduceBcast][RunAsync]rank[%u] count[%llu] failed in Reduce "
            "step",
            rank, count_),
        ret);

    ret = RunBroadcast(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceReduceBcast][RunAsync]rank[%u] count[%llu] failed in Broadcast "
            "step",
            rank, count_),
        ret);

    HCCL_INFO("AllReduceReduceBcast finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceReduceBcast::RunReduce(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO("AllReduceReduceBcast RunReduce: rank[%u] totalrank[%u] count[%llu]", rank, rankSize, count_);

    u32 unitSize = DataUnitSize(dataType_);

    DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem src;
    DeviceMem dst;

    if (rank == 0) {
        src = DeviceMem::create(static_cast<char*>(opInfo_->inputAddr), count_ * unitSize);
        dst = commMemOut.range(0, count_ * unitSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    // 数据准备
    HcclResult ret;
    if (rank == 0) {
        ret = RunAllReduceBDReduceReceive(rank, 0, links);
    } else {
        ret = RunAllReduceBDReduceSend(rank, 0, links);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceReduceBcastReduce]rank[%u] failed", rank), ret);
    return HCCL_SUCCESS;
}

HcclResult AllReduceReduceBcast::RunBroadcast(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO("AllReduceReduceBcast RunBroadcast: rank[%u] totalrank[%u] count[%llu]", rank, rankSize, count_);
    u32 unitSize = DataUnitSize(dataType_);

    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, count_ * unitSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem src;
    DeviceMem dst;

    if (userMemOut.ptr() != commMemOut.ptr()) {
        if (rank == 0) {
            src = commMemOut.range(0, count_ * unitSize);
            dst = userMemOut.range(0, count_ * unitSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
        }
    }
    HcclResult ret;
    if (rank == 0) {
        ret = RunAllReduceBDMemcpySend(rank, 0, links);
    } else {
        ret = RunAllReduceBDMemcpyReceive(rank, 0, links);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceReduceBcast]rank[%u] failed", rank), ret);

    HCCL_INFO("AllReduceReduceBcast RunBroadcast: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult AllReduceReduceBcast::RunAllReduceBDReduceSend(u32 rank, u32 peer, const std::vector<LINK>& links)
{
    HCCL_INFO("AllReduceReduceBcast RunAllReduceBDReduceSend: rank[%u] peer[%u] count[%llu]", rank, peer, count_);

    // 数据准备
    u32 unitSize = DataUnitSize(dataType_);
    u32 totalSize = count_ * unitSize;

    CHK_RET(links[peer]->RxAck(stream_));

    void* remMemPtr = nullptr;
    CHK_RET(links[peer]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));

    DeviceMem src = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);
    DeviceMem dst = DeviceMem::create(static_cast<char*>(remMemPtr), totalSize);

    CHK_RET(HcclReduceAsync(
        dispatcher_, static_cast<void*>(src.ptr()), count_, dataType_, reductionOp_, stream_,
        static_cast<void*>(dst.ptr()), links[peer]->GetRemoteRank(), links[peer]->GetLinkType(), INLINE_REDUCE_BIT));

    CHK_RET(links[peer]->TxDataSignal(stream_));
    return HCCL_SUCCESS;
}

HcclResult AllReduceReduceBcast::RunAllReduceBDReduceReceive(u32 rank, u32 peer, const std::vector<LINK>& links)
{
    HCCL_INFO("AllReduceReduceBcast RunAllReduceBDReduceReceive: rank[%u] peer[%u] count[%llu]", rank, peer, count_);

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < localRankSize_; round++) {
        Stream& subStream = (round == localRankSize_ - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[round]->TxAck(subStream));
        CHK_RET(links[round]->RxDataSignal(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());
    return HCCL_SUCCESS;
}

HcclResult AllReduceReduceBcast::RunAllReduceBDMemcpyReceive(u32 rank, u32 peer, const std::vector<LINK>& links)
{
    HCCL_INFO("AllReduceReduceBcast RunAllReduceBDMemcpyReceive: rank[%u] peer[%u] count[%llu]", rank, peer, count_);
    u32 unitSize = DataUnitSize(dataType_);

    CHK_RET(links[peer]->RxAck(stream_));

    void* remMemPtr = nullptr;
    CHK_RET(links[peer]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
    DeviceMem src = DeviceMem::create(static_cast<char*>(remMemPtr), count_ * unitSize);
    DeviceMem dst = DeviceMem::create(opInfo_->outputAddr, count_ * unitSize);
    CHK_RET(
        HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_, links[peer]->GetRemoteRank(), links[peer]->GetLinkType()));
    CHK_RET(links[peer]->TxDataSignal(stream_));

    HCCL_INFO("AllReduceReduceBcast RunAllReduceBDMemcpyReceive finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult AllReduceReduceBcast::RunAllReduceBDMemcpySend(u32 rank, u32 peer, const std::vector<LINK>& links)
{
    HCCL_INFO("AllReduceReduceBcast RunAllReduceBDMemcpySend: rank[%u] peer[%u] count[%llu]", rank, peer, count_);

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < localRankSize_; round++) {
        Stream& subStream = (round == localRankSize_ - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[round]->TxAck(subStream));
        CHK_RET(links[round]->RxDataSignal(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    HCCL_INFO("AllReduceReduceBcast RunAllReduceBDMemcpySend finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_REDUCE_BCAST, AllReduceReduceBcast);
} // namespace hccl
