/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_pipeline.h"
#include "alg_template_register.h"

constexpr u32 STEP_OFFSET_TWO = 2;

namespace hccl {
ReduceScatterPipeline::ReduceScatterPipeline(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ReduceScatterPipeline::~ReduceScatterPipeline() {}

HcclResult ReduceScatterPipeline::MainWaitSub(u32 begin)
{
    u32 subStreamNum = intraRankSize_;
    for (u32 signalIndex = begin; signalIndex < subStreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, streamNotifyMain_[signalIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPipeline::SubRecordMain(u32 begin)
{
    u32 subStreamNum = intraRankSize_;
    for (u32 streamIndex = begin; streamIndex < subStreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Post(subStream_[streamIndex], dispatcher_, streamNotifyMain_[streamIndex], -1));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPipeline::MainRecordSub(u32 begin)
{
    u32 subStreamNum = intraRankSize_;
    for (u32 signalIndex = begin; signalIndex < subStreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, streamNotifySub_[signalIndex], -1));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPipeline::SubWaitMain(u32 begin)
{
    u32 subStreamNum = intraRankSize_;
    for (u32 streamIndex = begin; streamIndex < subStreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            subStream_[streamIndex], dispatcher_, streamNotifySub_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPipeline::RunIntraServer(u32 step, u64 remoteOffset)
{
    for (u32 i = 1; i < intraRankSize_; i++) {
        u32 remIntraRankId = (intraRankId_ + i) % intraRankSize_;
        CHK_RET(intraLinks_[remIntraRankId]->TxAck(subStream_[i]));
        CHK_RET(intraLinks_[remIntraRankId]->RxAck(subStream_[i]));
        void* remoteMemPtr = nullptr;
        CHK_RET(intraLinks_[remIntraRankId]->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMemPtr));
        u64 srcOffset
            = (((interRankId_ + step + 1) % interRankSize_) * intraRankSize_ + remIntraRankId) * memSliceSize_;
        u64 offset = (srcOffset + offset_) % HCCL_MIN_SLICE_ALIGN_910B;
        DeviceMem src = DeviceMem::create(static_cast<u8*>(usrInMem_) + srcOffset, curSize_);
        DeviceMem dst = DeviceMem::create(static_cast<u8*>(remoteMemPtr) + remoteOffset + offset, curSize_);

        CHK_RET(HcclReduceAsync(
            dispatcher_, src.ptr(), count_, dataType_, reductionOp_, subStream_[i], dst.ptr(),
            intraLinks_[remIntraRankId]->GetRemoteRank(), intraLinks_[remIntraRankId]->GetLinkType(),
            INLINE_REDUCE_BIT));

        CHK_RET(intraLinks_[remIntraRankId]->TxDataSignal(subStream_[i]));
        CHK_RET(intraLinks_[remIntraRankId]->RxDataSignal(subStream_[i]));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPipeline::RunInterServer(u32 step, const LINK& prevInterLink, const LINK& nextInterLink)
{
    u32 dmaMemSliceNum = dmaMem_.size();
    u32 rxDMAMemSliceId = (step + 1) % dmaMemSliceNum;
    u32 txDMAMemSliceId = step % dmaMemSliceNum;
    u64 sliceMemOffset = memSliceSize_ * (((interRankId_ + 1 + step) % interRankSize_) * intraRankSize_ + intraRankId_);
    u64 offset = (sliceMemOffset + offset_) % HCCL_MIN_SLICE_ALIGN_910B;
    u64 rxInterOffset = rxDMAMemSliceId * blockSize_ + offset;
    void* txLocalAddr = static_cast<u8*>(dmaMem_[txDMAMemSliceId].ptr()) + offset;
    DeviceMem srcMem = DeviceMem::create(txLocalAddr, curSize_);
    CHK_RET(senderInfo_->run(nextInterLink, rxInterOffset, srcMem, subStream_[0]));
    HCCL_DEBUG(
        "[ReduceScatterPipeline][RunInterServer] local rank[%u] localOffset[%llu]tx with slice[%llu]", rankId_,
        rxInterOffset, curSize_);

    u64 rxSliceOffset = memSliceSize_ * (((interRankId_ + 2 + step) % interRankSize_) * intraRankSize_ + intraRankId_);
    u64 rxOffset = (rxSliceOffset + offset_) % HCCL_MIN_SLICE_ALIGN_910B;
    u64 rxMemOffset = txDMAMemSliceId * blockSize_ + rxOffset;
    void* rxLocalAddr = static_cast<u8*>(dmaMem_[rxDMAMemSliceId].ptr()) + rxOffset;
    DeviceMem rxLocalMem = DeviceMem::create(rxLocalAddr, curSize_);
    CHK_RET(
        reducerInfo_->run(dispatcher_, prevInterLink, rxMemOffset, rxLocalMem, rxLocalMem, rxLocalMem, subStream_[0]));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPipeline::CopyToScratchBuffer(u32 step)
{
    u32 dmaMemSliceNum = dmaMem_.size();
    u32 dmaMemSliceId = step % dmaMemSliceNum;
    u64 sliceMemOffset = memSliceSize_ * (((interRankId_ + 1 + step) % interRankSize_) * intraRankSize_ + intraRankId_);
    u64 offset = (sliceMemOffset + offset_) % HCCL_MIN_SLICE_ALIGN_910B;
    // 把一块切片从userIn 做拷贝到CCLBuffer
    void* srcAddr = static_cast<u8*>(usrInMem_) + sliceMemOffset;
    DeviceMem locSrc = DeviceMem::create(srcAddr, curSize_);
    DeviceMem locDst = DeviceMem::create(static_cast<u8*>(dmaMem_[dmaMemSliceId].ptr()) + offset, curSize_);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDst, locSrc, stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPipeline::RunAsync()
{
    // inter ring algo
    u32 prevInterRankId = (interRankId_ + 1) % interRankSize_;
    u32 nextInterRankId = (interRankId_ - 1 + interRankSize_) % interRankSize_;
    LINK prevInterLink = interLinks_[prevInterRankId];
    LINK nextInterLink = interLinks_[nextInterRankId];
    // 当前使用3块DMAMem buffer
    u32 dmaMemSliceNum = dmaMem_.size();
    HCCL_DEBUG("RunAsync begin.");

    for (u32 step = 0; step < interRankSize_; step++) {
        u32 begin = 0;
        if (step == 0) {
            // 把第一块切片从userIn 做拷贝到CCLBuffer
            begin = 1;
            CHK_RET(CopyToScratchBuffer(step));
            CHK_RET(MainRecordSub(begin));
            CHK_RET(SubWaitMain(begin));
        }
        // server内做SDMA的reduce
        u64 remoteOffset = (step % dmaMemSliceNum) * blockSize_;
        HCCL_DEBUG("[RunAsync]remoteOffset is [%llu]", remoteOffset);
        CHK_RET(RunIntraServer(step, remoteOffset));
        CHK_RET(SubRecordMain(begin));
        CHK_RET(MainWaitSub(begin));
        if (step < interRankSize_ - 1) {
            // 把下一块切片从userIn 做拷贝到CCLBuffer
            CHK_RET(CopyToScratchBuffer(step + 1));
            // 全部流同步，确保SDMA执行完成
            CHK_RET(MainRecordSub(0));
            CHK_RET(SubWaitMain(0));
            CHK_RET(prevInterLink->TxAck(subStream_[0]));
            CHK_RET(nextInterLink->RxAck(subStream_[0]));
            // server间做RDMA的reduce，可与下一个step的SDMA并发执行
            CHK_RET(RunInterServer(step, prevInterLink, nextInterLink));
            CHK_RET(prevInterLink->PostFinAck(subStream_[0]));
            CHK_RET(nextInterLink->WaitFinAck(subStream_[0]));
            // inter的最后一步需要barrier确保数据发完
            if (step == interRankSize_ - STEP_OFFSET_TWO) {
                CHK_RET(ExecuteBarrier(prevInterLink, nextInterLink, subStream_[0]));
            }
        }
    }
    // 把对应的切片从CCLBuffer拷贝到userOut
    DeviceMem locDst = DeviceMem::create(static_cast<u8*>(usrOutMem_), curSize_);
    u64 srcOffset
        = (memSliceSize_ * (interRankId_ * intraRankSize_ + intraRankId_) + offset_) % HCCL_MIN_SLICE_ALIGN_910B;
    void* locSrcAddr = static_cast<u8*>(dmaMem_[(interRankSize_ - 1) % dmaMemSliceNum].ptr()) + srcOffset;
    DeviceMem locSrc = DeviceMem::create(locSrcAddr, curSize_);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDst, locSrc, stream_));
    HCCL_INFO("[ReduceScatterPipeline][RunAsync]ReduceScatterPipeline finished groupRankId[%u] ", rankId_);
    return HCCL_SUCCESS;
}

// 适配新CollExecutor接口
HcclResult ReduceScatterPipeline::Prepare(
    HcomCollOpInfo* opInfo, DeviceMem& cclBuffer, const u64 count, const u64 bufferSize, const u64 offset,
    const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo, Stream& mainStream,
    std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
    std::vector<std::shared_ptr<LocalNotify>>& notifySub, u64 reduceAttrBitMap)
{
    reduceAttr_ = reduceAttrBitMap;
    opInfo_ = opInfo;

    unitSize_ = SIZE_TABLE[opInfo_->dataType];
    memSliceSize_ = opInfo_->count * unitSize_;
    usrInMem_ = opInfo_->inputAddr;
    usrOutMem_ = opInfo_->outputAddr;
    reductionOp_ = opInfo_->reduceOp;
    dataType_ = opInfo_->dataType;
    offset_ = offset;

    // needed resource
    // stream: 1 * mainStream + n * subStream
    // mem: usrInMem_, usrOutMem, DMAMem
    // interNotify, streamNotify

    // stream
    // mainStream负责locMemCpy以及subStream同步控制
    stream_ = mainStream;
    // subStream负责:
    // streamId[0]: inter执行
    // streamId[1:intraRankSize]: intraRankSize-1个intra执行
    subStream_ = subStream;

    // DMAMem + interNotify from Link
    intraRankSize_ = level0CommInfo.localRankSize;
    interRankSize_ = level1CommInfo.localRankSize;
    intraRankId_ = level0CommInfo.localRank;
    interRankId_ = level1CommInfo.localRank;
    rankId_ = intraRankId_ + interRankId_ * intraRankSize_;

    // streamNotify, size: n
    streamNotifyMain_ = notifyMain;
    if (streamNotifyMain_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[ReduceScatterPipeline][Prepare]rank[%u] streamNotifyMain_ size [%u] is smaller than "
            "intraRankSize_[%u]",
            rankId_, streamNotifyMain_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }
    streamNotifySub_ = notifySub;
    if (streamNotifySub_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[ReduceScatterPipeline][Prepare]rank[%u] streamNotifySub_ size [%u] is smaller than "
            "intraRankSize_[%u]",
            rankId_, streamNotifySub_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }
    // usrMem

    intraLinks_ = level0CommInfo.links;
    interLinks_ = level1CommInfo.links;

    // 3级流水,使用3块DMAMem
    cclBuffer_ = cclBuffer;
    count_ = count;
    curSize_ = count_ * unitSize_;
    bufferSize_ = bufferSize;
    blockSize_ = (bufferSize_ / (HCCL_MIN_SLICE_ALIGN_910B * PIPELINE_DEPTH)) * HCCL_MIN_SLICE_ALIGN_910B;

    for (u32 i = 0; i < pipDepth_; i++) {
        DeviceMem mem = DeviceMem::create(static_cast<u8*>(cclBuffer_.ptr()) + blockSize_ * i, blockSize_);
        dmaMem_.push_back(mem);
    }

    HCCL_INFO(
        "[ReduceScatterPipeline][Prepare]streamNum[%u], streamNotifyMainNum[%u], streamNotifySubNum[%u]",
        subStream_.size(), streamNotifyMain_.size(), streamNotifySub_.size());
    HCCL_INFO(
        "[ReduceScatterPipeline][Prepare]interLinksNum[%u], intraLinksNum[%u]", interLinks_.size(), intraLinks_.size());
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);
    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPipeline::GetNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    u32 ringNextRank = (rank + 1) % rankSize;
    LINK nslbNext = links[ringNextRank];
    HCCL_DEBUG("[ReduceScatterPipeline]GetNslbAdjInfo starts");

    // Pipeline 步长合并 等同于 ring
    NslbDpAdjInfo adjInfoStep = {};
    nslbAdjInfo.dstRankNum = 1;
    adjInfoStep.dstLocalRankId = nslbNext->GetRemoteRank();
    adjInfoStep.phaseId = 1;
    adjInfoStep.rev = 0;
    nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_PIPELINE, ReduceScatterPipeline);
} // namespace hccl
