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
#include "all_reduce_local_reduce_bcast.h"

namespace hccl {
AllReduceLocalReduceBcast::AllReduceLocalReduceBcast(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllReduceLocalReduceBcast::~AllReduceLocalReduceBcast() {}

HcclResult AllReduceLocalReduceBcast::Prepare(
    u64 reduceAttrBitMap, std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
    std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 interRank, u32 interRankSize, u32 userRank,
    HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    localRank_ = interRank;
    localRankSize_ = interRankSize;
    userRank_ = userRank;
    meshStreams_ = meshStreams;
    meshSignal_ = &meshSignal;
    meshSignalAux_ = &meshSignalAux;
    opInfo_ = opInfo;
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::MainRecordSub(u32 streamNum)
{
    if (streamNum == 0) {
        for (u32 signalIndex = 0; signalIndex < meshSignalAux_->size(); signalIndex++) {
            CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAux_)[signalIndex], profilerInput_.stage));
        }
    } else {
        for (u32 signalIndex = 0; signalIndex < streamNum; signalIndex++) {
            CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAux_)[signalIndex], profilerInput_.stage));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::SubWaitMain(u32 streamNum)
{
    if (streamNum == 0) {
        for (u32 streamIndex = 0; streamIndex < meshSignalAux_->size(); streamIndex++) {
            CHK_RET(LocalNotify::Wait(
                meshStreams_[streamIndex], dispatcher_, (*meshSignalAux_)[streamIndex], profilerInput_.stage));
        }
    } else {
        for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
            CHK_RET(LocalNotify::Wait(
                meshStreams_[streamIndex], dispatcher_, (*meshSignalAux_)[streamIndex], profilerInput_.stage));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::MainWaitSub(u32 streamNum)
{
    if (streamNum == 0) {
        for (u32 signalIndex = 0; signalIndex < meshSignal_->size(); signalIndex++) {
            CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignal_)[signalIndex], profilerInput_.stage));
        }
    } else {
        for (u32 signalIndex = 0; signalIndex < streamNum; signalIndex++) {
            CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignal_)[signalIndex], profilerInput_.stage));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::SubRecordMain(u32 streamNum)
{
    if (streamNum == 0) {
        for (u32 streamIndex = 0; streamIndex < meshSignal_->size(); streamIndex++) {
            CHK_RET(LocalNotify::Post(
                meshStreams_[streamIndex], dispatcher_, (*meshSignal_)[streamIndex], profilerInput_.stage));
        }
    } else {
        for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
            CHK_RET(LocalNotify::Post(
                meshStreams_[streamIndex], dispatcher_, (*meshSignal_)[streamIndex], profilerInput_.stage));
        }
    }
    return HCCL_SUCCESS;
}

// 将数据均分，最小单位是128

// ring allreduce算法的函数入口
HcclResult AllReduceLocalReduceBcast::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "AllReduceLocalReduceBcast run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[AllReduceLocalReduceBcast][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank,
            links.size(), rankSize),
        HCCL_E_INTERNAL);

    // 如果ranksize为1, 从input->output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            DeviceMem userMemIn = DeviceMem::create(inputMem_.ptr(), count_ * DataUnitSize(dataType_));
            DeviceMem userMemOut = DeviceMem::create(outputMem_.ptr(), count_ * DataUnitSize(dataType_));
            ret = HcclD2DMemcpyAsync(dispatcher_, userMemOut, userMemIn, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceLocalReduceBcast][RunAsync]rank[%u] memcpy async failed", rank), ret);
        }
        return ret;
    }

    ret = RunReduce(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceLocalReduceBcast][RunAsync]rank[%u] count[%llu] failed in Reduce step", rank, count_),
        ret);

    ret = RunBroadcast(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceLocalReduceBcast][RunAsync]rank[%u] count[%llu] failed in Broadcast "
            "step",
            rank, count_),
        ret);

    HCCL_INFO("AllReduceLocalReduceBcast finished: rank[%u] ranksize[%u].", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::RunReduce(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO("AllReduceLocalReduceBcast RunReduce: rank[%u] totalrank[%u] count[%llu].", rank, rankSize, count_);

    u32 unitSize = SIZE_TABLE[dataType_];

    DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    if (rank == 0) {
        DeviceMem src = userMemIn.range(0, count_ * unitSize);
        DeviceMem dst = commMemOut.range(0, count_ * unitSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    // 数据准备
    HcclResult ret;
    if (rank == 0) {
        CHK_RET(RunAllReduceBDReduceReceive(rank, 0, links));
        ret = RunLocalReduce(rank, rankSize);
    } else {
        ret = RunAllReduceBDReduceSend(rank, 0, links);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceLocalReduceBcastReduce]rank[%u] RunAllReduceBDReduce failed", rank),
        ret);
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::RunBroadcast(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO("AllReduceLocalReduceBcast RunBroadcast: rank[%u] totalrank[%u] count[%llu].", rank, rankSize, count_);

    HcclResult ret;
    if (rank == 0) {
        ret = RunAllReduceBDMemcpySend(rank, 0, links);
    } else {
        ret = RunAllReduceBDMemcpyReceive(rank, 0, links);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceLocalReduceBcast]rank[%u] RunBroadcast failed", rank), ret);

    HCCL_INFO("AllReduceLocalReduceBcast RunBroadcast: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::RunAllReduceBDReduceSend(u32 rank, u32 peer, const std::vector<LINK>& links)
{
    HCCL_INFO("AllReduceLocalReduceBcast RunAllReduceBDReduceSend: rank[%u] peer[%u] count[%llu].", rank, peer, count_);

    // 数据准备
    u32 unitSize = SIZE_TABLE[dataType_];
    u32 totalSize = count_ * unitSize;

    DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);

    CHK_RET(links[peer]->RxAck(stream_));

    void* remMemPtr = nullptr;
    CHK_RET(links[peer]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));

    DeviceMem src = userMemIn;
    DeviceMem dst = DeviceMem::create(static_cast<char*>(remMemPtr) + (rank - 1) * totalSize, totalSize);

    if (rank != 1) {
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, stream_, links[peer]->GetRemoteRank(), links[peer]->GetLinkType()));
    } else {
        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<void*>(src.ptr()), count_, dataType_, reductionOp_, stream_,
            static_cast<void*>(dst.ptr()), links[peer]->GetRemoteRank(), links[peer]->GetLinkType(),
            INLINE_REDUCE_BIT));
    }

    CHK_RET(links[peer]->TxDataSignal(stream_));
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::RunAllReduceBDReduceReceive(u32 rank, u32 peer, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "AllReduceLocalReduceBcast RunAllReduceBDReduceReceive: rank[%u] peer[%u] count[%llu].", rank, peer, count_);

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < localRankSize_; round++) {
        Stream& subStream = (round == localRankSize_ - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[round]->TxAck(subStream));
        CHK_RET(links[round]->RxDataSignal(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());
    HCCL_DEBUG("[AllReduceLocalReduceBcast]RunAllReduceBDReduceReceive success");
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::RunLocalReduce(u32 rank, u32 rankSize)
{
    (void)rank;
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());
    u32 power = static_cast<u32>(log2(rankSize - 1));
    u32 rankPower = static_cast<u32>(pow(2, power));
    u32 unitSize = SIZE_TABLE[dataType_];
    u32 totalSize = count_ * unitSize;
    DeviceMem src;
    DeviceMem dst;
    if (rankPower < rankSize - 1) {
        src = commMemOut.range(rankPower * totalSize, (rankSize - rankPower - 1) * totalSize);
        dst = commMemOut.range(0, (rankSize - rankPower - 1) * totalSize);
        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<void*>(src.ptr()), count_ * (rankSize - rankPower - 1), dataType_, reductionOp_,
            stream_, static_cast<void*>(dst.ptr()), INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));
    }
    for (u32 round = 0; round < power; round++) {
        u32 sliceNum = rankPower / static_cast<u32>(pow(2, round + 1));
        src = commMemOut.range(sliceNum * totalSize, sliceNum * totalSize);
        dst = commMemOut.range(0, sliceNum * totalSize);
        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<void*>(src.ptr()), count_ * sliceNum, dataType_, reductionOp_, stream_,
            static_cast<void*>(dst.ptr()), INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::RunAllReduceBDMemcpyReceive(u32 rank, u32 peer, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "AllReduceLocalReduceBcast RunAllReduceBDMemcpyReceive: rank[%u] peer[%u] count[%llu]", rank, peer, count_);
    u32 unitSize = SIZE_TABLE[dataType_];

    CHK_RET(links[peer]->RxAck(stream_));
    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, count_ * unitSize);

    u32 totalSize = count_ * unitSize;

    void* remMemPtr = nullptr;
    CHK_RET(links[peer]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
    DeviceMem src;

    src = DeviceMem::create(static_cast<char*>(remMemPtr), totalSize);

    DeviceMem dst = userMemOut;
    CHK_RET(
        HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_, links[peer]->GetRemoteRank(), links[peer]->GetLinkType()));
    CHK_RET(links[peer]->TxDataSignal(stream_));

    HCCL_INFO("AllReduceLocalReduceBcast RunAllReduceBDMemcpyReceive finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduceBcast::RunAllReduceBDMemcpySend(u32 rank, u32 peer, const std::vector<LINK>& links)
{
    HCCL_INFO("AllReduceLocalReduceBcast RunAllReduceBDMemcpySend: rank[%u] peer[%u] count[%llu]", rank, peer, count_);

    u32 unitSize = SIZE_TABLE[dataType_];
    u32 totalSize = count_ * unitSize;

    if (opInfo_->outputAddr != outputMem_.ptr()) {
        DeviceMem dst = DeviceMem::create(opInfo_->outputAddr, totalSize);
        DeviceMem src = DeviceMem::create(outputMem_.ptr(), totalSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }
    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < localRankSize_; round++) {
        Stream& subStream = (round == localRankSize_ - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[round]->TxAck(subStream));
        CHK_RET(links[round]->RxDataSignal(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    HCCL_INFO("AllReduceLocalReduceBcast RunAllReduceBDMemcpySend finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_LOCAL_REDUCE_BCAST, AllReduceLocalReduceBcast);
} // namespace hccl
