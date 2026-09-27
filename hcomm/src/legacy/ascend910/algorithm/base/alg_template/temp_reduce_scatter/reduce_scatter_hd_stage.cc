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
#include "reduce_scatter_hd_stage_pub.h"
#include "alg_template_register.h"

namespace hccl {
ReduceScatterHDStage::ReduceScatterHDStage(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ReduceScatterHDStage::~ReduceScatterHDStage() {}

HcclResult ReduceScatterHDStage::Prepare(
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

HcclResult ReduceScatterHDStage::MainRecordSub(u32 streamNum)
{
    for (u32 signalIndex = 0; signalIndex < streamNum; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAuxPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHDStage::SubWaitMain(u32 streamNum)
{
    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHDStage::MainWaitSub(u32 streamNum)
{
    for (u32 signalIndex = 0; signalIndex < streamNum; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignalPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHDStage::SubRecordMain(u32 streamNum)
{
    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// ring allreduce算法的函数入口
HcclResult ReduceScatterHDStage::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "ReduceScatterHDStage run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[ReduceScatterHDStage][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    ret = PrepareSliceData(rankSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ReduceScatterHDStage][RunAsync]rank[%u] count[%llu] failed in PrepareSliceData "
            "step",
            rank, count_),
        ret);

    ret = RunReduceScatterStage(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ReduceScatterHDStage][RunAsync]rank[%u] count[%llu] failed"
            "step",
            rank, count_),
        ret);

    HCCL_INFO("ReduceScatterHDStage finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHDStage::PrepareSliceData(u32 rankSize)
{
    Slice temp;
    u32 unitSize = SIZE_TABLE[dataType_];
    u64 totalSize = count_ * unitSize;
    u32 power = static_cast<u32>(log2(rankSize));
    u32 half = static_cast<u32>(pow(base, power - 1));
    u64 offset;
    for (u32 round = 1; round <= power; round++) {
        u32 sliceNum = rankSize / static_cast<u32>(pow(base, round));
        sliceMap_[round - 1].clear();
        sliceMap_[power - round].reserve(rankSize);
        for (u32 sliceGroup = 0; sliceGroup < pow(base, round); sliceGroup++) {
            for (u32 sliceCount = 0; sliceCount < sliceNum; sliceCount++) {
                temp.size = totalSize * sliceNum;
                offset = totalSize * sliceNum * sliceGroup;
                if (sliceGroup == 0) {
                    temp.offset = offset;
                } else {
                    if (round != 1) {
                        temp.offset = (offset >= half * totalSize) ? (offset - half * totalSize) : offset;
                    } else {
                        temp.offset = offset;
                    }
                }
                sliceMap_[round - 1].push_back(temp);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHDStage::RunReduceScatterStage(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "RunReduceScatterStage run: rank[%u] totalrank[%u] outputMem[%p] count[%llu]", rank, rankSize, outputMem_.ptr(),
        count_);
    nSteps_ = static_cast<u32>(log2(rankSize));
    CHK_RET(RunReduceScatterStage1st(rank, rankSize, links));
    CHK_RET(RunReduceScatterRead(rank, rankSize, links));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHDStage::RunReduceScatterStage1st(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    u32 unitSize = SIZE_TABLE[dataType_];
    u32 totalSize = unitSize * count_;

    DeviceMem UserMemIn = DeviceMem::create(opInfo_->inputAddr, rankSize * totalSize);
    DeviceMem CommMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem src;
    DeviceMem dst;

    src = UserMemIn.range(sliceMap_[0][rank].offset, sliceMap_[0][rank].size);
    dst = CommMemOut.range(0, sliceMap_[0][rank].size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));

    u32 dstRank = rank ^ (1 << (nSteps_ - 1));
    CHK_RET(links[dstRank]->TxAck(stream_));
    CHK_RET(links[dstRank]->RxAck(stream_));

    void* remMemPtr = nullptr;
    CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
    src = UserMemIn.range(sliceMap_[0][dstRank].offset, sliceMap_[0][dstRank].size);
    dst = DeviceMem::create(static_cast<u8*>(remMemPtr), sliceMap_[0][dstRank].size);
    CHK_RET(HcclReduceAsync(
        dispatcher_, static_cast<void*>(src.ptr()), sliceMap_[0][dstRank].size / unitSize, dataType_, reductionOp_,
        stream_, static_cast<void*>(dst.ptr()), links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType(),
        INLINE_REDUCE_BIT));

    CHK_RET(links[dstRank]->TxDataSignal(stream_));
    CHK_RET(links[dstRank]->RxDataSignal(stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHDStage::RunBetweenStep(u32 rank, u32 neighCur, u32 neighNext, const std::vector<LINK>& links)
{
    (void)rank;
    CHK_RET(MainRecordSub(1));
    CHK_RET(SubWaitMain(1));

    CHK_RET(links[neighCur]->TxDataSignal(meshStreams_[0]));
    CHK_RET(links[neighCur]->RxDataSignal(meshStreams_[0]));

    CHK_RET(links[neighNext]->TxAck(stream_));
    CHK_RET(links[neighNext]->RxAck(stream_));

    CHK_RET(SubRecordMain(1));
    CHK_RET(MainWaitSub(1));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHDStage::RunReduceScatterRead(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    (void)rankSize;
    u32 unitSize = SIZE_TABLE[dataType_];
    u32 totalSize = unitSize * count_;

    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, totalSize);

    void* remMemPtr = nullptr;
    DeviceMem dst;
    DeviceMem src;
    u32 dstRank;
    CHK_RET(links[rank ^ (1 << (nSteps_ - 1 - 1))]->TxAck(stream_));
    CHK_RET(links[rank ^ (1 << (nSteps_ - 1 - 1))]->RxAck(stream_));
    for (u32 step = 1; step < nSteps_; step++) {
        dstRank = rank ^ (1 << (nSteps_ - 1 - step));
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        dst = outputMem_.range(sliceMap_[step][rank].offset, sliceMap_[step][rank].size);
        src = DeviceMem::create(static_cast<u8*>(remMemPtr) + sliceMap_[step][rank].offset, sliceMap_[step][rank].size);
        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<void*>(src.ptr()), sliceMap_[step][rank].size / unitSize, dataType_, reductionOp_,
            stream_, static_cast<void*>(dst.ptr()), links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType(),
            INLINE_REDUCE_BIT));
        if (step != (nSteps_ - 1)) {
            CHK_RET(RunBetweenStep(rank, dstRank, rank ^ (1 << (nSteps_ - 1 - step - 1)), links));
        }
    }

    CHK_RET(MainRecordSub(1));
    CHK_RET(SubWaitMain(1));

    CHK_RET(links[rank ^ (1 << 0)]->TxDataSignal(stream_));
    CHK_RET(links[rank ^ (1 << 0)]->RxDataSignal(stream_));

    src = outputMem_.range(sliceMap_[nSteps_ - 1][rank].offset, totalSize);
    dst = userMemOut.range(0, totalSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, meshStreams_[0]));

    CHK_RET(SubRecordMain(1));
    CHK_RET(MainWaitSub(1));

    DeviceMem emptyMem = outputMem_.range(0, 0);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyMem, emptyMem, stream_));

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_HDSTAGE, ReduceScatterHDStage);
} // namespace hccl
