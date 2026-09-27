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
#include "reduce_scatter_local_reduce_pub.h"
#include "alg_template_register.h"

namespace hccl {
ReduceScatterLocalReduce::ReduceScatterLocalReduce(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ReduceScatterLocalReduce::~ReduceScatterLocalReduce() {}

HcclResult ReduceScatterLocalReduce::Prepare(
    DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
    const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
    const u64 baseOffset, const u64 reduceAttrBitMap, std::vector<Stream>& meshStreams,
    std::vector<std::shared_ptr<LocalNotify>>& meshSignal, std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux,
    u32 userRank, const HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    userRank_ = userRank;
    meshStreams_ = meshStreams;
    meshSignalPtr_ = &meshSignal;
    meshSignalAuxPtr_ = &meshSignalAux;
    opInfo_ = opInfo;
    return AlgTemplateBase::Prepare(
        inputMem, outputMem, scratchMem, count, dataType, stream, reductionOp, root, slices, baseOffset);
}

HcclResult ReduceScatterLocalReduce::MainRecordSub(u32 streamNum)
{
    u32 totalTask = streamNum;
    CHK_PRT_RET(
        (totalTask > meshSignalAuxPtr_->size()),
        HCCL_ERROR(
            "[ReduceScatterLocalReduce][MainRecordSub]totalTask[%u] is over range of meshSignalAux[%zu]", totalTask,
            meshSignalAuxPtr_->size()),
        HCCL_E_PARA);
    for (u32 signalIndex = 0; signalIndex < totalTask; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAuxPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterLocalReduce::SubWaitMain(u32 streamNum)
{
    u32 totalTask = streamNum;
    CHK_PRT_RET(
        (totalTask > meshSignalAuxPtr_->size() || totalTask > meshStreams_.size()),
        HCCL_ERROR(
            "[ReduceScatterLocalReduce][SubWaitMain]totalTask[%u] is over range of meshSignalAux[%zu]"
            "or meshStreams_[%zu]",
            totalTask, meshSignalAuxPtr_->size(), meshStreams_.size()),
        HCCL_E_PARA);
    for (u32 streamIndex = 0; streamIndex < totalTask; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterLocalReduce::MainWaitSub(u32 streamNum)
{
    u32 totalTask = streamNum;
    CHK_PRT_RET(
        (totalTask > meshSignalPtr_->size()),
        HCCL_ERROR(
            "[ReduceScatterLocalReduce][MainWaitSub]totalTask[%u] is over range of meshSignal[%zu]", totalTask,
            meshSignalPtr_->size()),
        HCCL_E_PARA);
    for (u32 signalIndex = 0; signalIndex < totalTask; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignalPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterLocalReduce::SubRecordMain(u32 streamNum)
{
    u32 totalTask = streamNum;
    CHK_PRT_RET(
        (totalTask > meshSignalPtr_->size() || totalTask > meshStreams_.size()),
        HCCL_ERROR(
            "[ReduceScatterLocalReduce][SubWaitMain]totalTask[%u] is over range of meshSignal[%zu]"
            "or meshStreams_[%zu]",
            totalTask, meshSignalPtr_->size(), meshStreams_.size()),
        HCCL_E_PARA);
    for (u32 streamIndex = 0; streamIndex < totalTask; streamIndex++) {
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// 计算每片数据的offset
HcclResult ReduceScatterLocalReduce::PrepareOffset(u32 rankSize)
{
    Slice temp;
    u32 unitSize = SIZE_TABLE[dataType_];
    u64 totalSize = (opInfo_->count) * unitSize;
    slices_.clear();
    slices_.reserve(rankSize);
    if (rankSize == 0) {
        HCCL_ERROR("[Prepare][Offset]data slice prepare, sliceNum is 0");
        return HCCL_E_PARA;
    }
    for (u32 i = 0; i < rankSize; i++) {
        if (count_ * SIZE_TABLE[dataType_] > HCCL_SMALL_COUNT_32_KB) {
            temp.offset = (i * totalSize) % HCCL_MIN_SLICE_ALIGN_910B;
        } else {
            temp.offset = 0;
        }
        temp.size = 0;
        slices_.push_back(temp);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterLocalReduce::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "ReduceScatterLocalReduce run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu].", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[ReduceScatterLocalReduce][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    ret = PrepareOffset(rankSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ReduceScatterLocalReduce][RunAsync]rank[%u] count[%llu] failed in PrepareOffset step", rank, count_),
        ret);

    ret = RunReduceScatter(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ReduceScatterLocalReduce][RunAsync]rank[%u] count[%llu] failed in ReduceScatter step", rank, count_),
        ret);

    ret = RunLocalReduce(rank, rankSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterlocalReduce]rank[%u] LocalReduce failed", rank), ret);

    HCCL_INFO("ReduceScatterLocalReduce finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterLocalReduce::CalAlign(u64 totalSize, u32 rankSize, u64& alignSize)
{
    auto maxIt = std::max_element(slices_.begin(), slices_.end(), [](const Slice& slice1, const Slice& slice2) {
        return slice1.offset < slice2.offset;
    });

    u64 maxOffset = maxIt->offset;

    alignSize = RoundUpWithDivisor(totalSize, HCCL_MIN_SLICE_ALIGN_910B);
    if (alignSize * (rankSize - 1) > (outputMem_.size() - maxOffset)) {
        alignSize = RoundUpWithDivisor(totalSize, HCCL_MIN_SLICE_ALIGN_ONCHIP);
    }
    if (alignSize * (rankSize - 1) > (outputMem_.size() - maxOffset)) {
        alignSize = totalSize;
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterLocalReduce::RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "ReduceScatterLocalReduce run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu].", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    // 数据准备
    u64 unitSize = SIZE_TABLE[dataType_];
    u64 totalSize = count_ * unitSize;
    u64 alignSize = totalSize;
    CHK_RET(CalAlign(totalSize, rankSize, alignSize));
    u64 offset = (opInfo_->count) * unitSize;
    DeviceMem UserMemIn = DeviceMem::create(opInfo_->inputAddr, offset * rankSize);
    DeviceMem CommMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());
    DeviceMem UserMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize);

    DeviceMem src;
    DeviceMem dst;

    DeviceMem emptySrc = UserMemIn.range(0, 0);
    DeviceMem emptyDst = CommMemOut.range(0, 0);

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(MainRecordSub(rankSize - base));
    CHK_RET(SubWaitMain(rankSize - base));

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = (round + rank) % rankSize;
        Stream& subStream = (round == rankSize - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    CHK_RET(SubRecordMain(rankSize - base));
    CHK_RET(MainWaitSub(rankSize - base));

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(MainRecordSub(meshStreams_.size()));
    CHK_RET(SubWaitMain(meshStreams_.size()));

    for (u32 round = 1; round < rankSize; round++) {
        Stream& subStream = (round == rankSize - 1) ? stream_ : meshStreams_[round - 1];

        u32 dstRank = (round + rank) % rankSize;

        void* remMemPtr = nullptr;
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));

        dst = DeviceMem::create(
            static_cast<u8*>(remMemPtr) + alignSize * (round - 1) + slices_[dstRank].offset, totalSize);
        src = UserMemIn.range(offset * dstRank, totalSize);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, subStream, links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));

        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }

    src = UserMemIn.range(offset * rank, totalSize);
    dst = UserMemOut.range(0, totalSize);
    Stream& subStream = (meshStreams_.size() > 0) ? meshStreams_[meshStreams_.size() - 1] : stream_;
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStream));

    CHK_RET(SubRecordMain(meshStreams_.size()));
    CHK_RET(MainWaitSub(meshStreams_.size()));

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterLocalReduce::RunLocalReduce(u32 rank, u32 rankSize)
{
    u32 power = static_cast<u32>(log2(rankSize));
    u32 rankPower = static_cast<u32>(pow(base, power));
    u64 totalSize = count_ * SIZE_TABLE[dataType_];
    u64 alignSize = totalSize;
    CHK_RET(CalAlign(totalSize, rankSize, alignSize));
    DeviceMem CommMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());
    DeviceMem UserMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize);
    CommMemOut = CommMemOut.range(slices_[rank].offset, outputMem_.size() - slices_[rank].offset);
    DeviceMem src;
    DeviceMem dst;
    DeviceMem emptySrc = CommMemOut.range(0, 0);
    DeviceMem emptyDst = CommMemOut.range(0, 0);
    if (rankPower < rankSize) {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));
        CHK_RET(MainRecordSub(rankSize - rankPower - 1));
        CHK_RET(SubWaitMain(rankSize - rankPower - 1));
        for (u32 add = 0; add < (rankSize - rankPower); add++) {
            Stream& subStream = (add == 0) ? stream_ : meshStreams_[add - 1];
            src = CommMemOut.range(alignSize * (add + rankPower - 1), totalSize);
            if (add == 0) {
                dst = UserMemOut.range(0, totalSize);
            } else {
                dst = CommMemOut.range(alignSize * (add - 1), totalSize);
            }
            CHK_RET(HcclReduceAsync(
                dispatcher_, static_cast<void*>(src.ptr()), count_, dataType_, reductionOp_, subStream,
                static_cast<void*>(dst.ptr()), INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));
        }
        CHK_RET(SubRecordMain(rankSize - rankPower - 1));
        CHK_RET(MainWaitSub(rankSize - rankPower - 1));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));
    }
    for (u32 round = 0; round < power; round++) {
        rankPower = static_cast<u32>(pow(base, power - round - 1));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));
        CHK_RET(MainRecordSub(rankPower - 1));
        CHK_RET(SubWaitMain(rankPower - 1));
        for (u32 add = 0; add < rankPower; add++) {
            Stream& subStream = (add == 0) ? stream_ : meshStreams_[add - 1];
            src = CommMemOut.range(alignSize * (add + rankPower - 1), totalSize);
            dst = (add == 0) ? UserMemOut.range(0, totalSize) : CommMemOut.range(alignSize * (add - 1), totalSize);
            CHK_RET(HcclReduceAsync(
                dispatcher_, static_cast<void*>(src.ptr()), count_, dataType_, reductionOp_, subStream,
                static_cast<void*>(dst.ptr()), INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));
        }
        CHK_RET(SubRecordMain(rankPower - 1));
        CHK_RET(MainWaitSub(rankPower - 1));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_LOCAL_REDUCE, ReduceScatterLocalReduce);
} // namespace hccl
