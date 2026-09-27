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
#include "all_reduce_opbase_pipeline.h"

constexpr u32 STEP_OFFSET_TWO = 2;

namespace hccl {
AllReduceOpbasePipeline::AllReduceOpbasePipeline(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllReduceOpbasePipeline::~AllReduceOpbasePipeline() {}

HcclResult AllReduceOpbasePipeline::Prepare(u64 reduceAttrBitMap, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

HcclResult AllReduceOpbasePipeline::MainWaitSub()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 signalIndex = 0; signalIndex < subStreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, streamNotifyMain_[signalIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOpbasePipeline::SubRecordMain()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 streamIndex = 0; streamIndex < subStreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Post(subStreams_[streamIndex], dispatcher_, streamNotifyMain_[streamIndex], -1));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOpbasePipeline::MainRecordSub()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 signalIndex = 0; signalIndex < subStreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, streamNotifySub_[signalIndex], -1));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOpbasePipeline::SubWaitMain()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 streamIndex = 0; streamIndex < subStreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            subStreams_[streamIndex], dispatcher_, streamNotifySub_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOpbasePipeline::RunReduceScatterIntraServer(u32 step)
{
    u32 dmaMemSliceNum = dmaMem_.size();
    u32 dmaMemSliceId = step % dmaMemSliceNum;
    for (u32 i = 1; i < intraRankSize_; i++) {
        u32 remIntraRankId = (intraRankId_ + i) % intraRankSize_;
        CHK_RET(intraLinks_[remIntraRankId]->TxAck(subStreams_[i - 1]));
        CHK_RET(intraLinks_[remIntraRankId]->RxAck(subStreams_[i - 1]));
        void* remoteMemPtr = nullptr;
        CHK_RET(intraLinks_[remIntraRankId]->GetRemoteMem(
            dmaMemSliceId == 1 ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM, &remoteMemPtr));
        u32 sliceId = ((interRankId_ + step + 1) % interRankSize_) * intraRankSize_ + remIntraRankId;
        u64 srcOffset = sliceId * memSliceSize_;
        u64 offset = srcOffset % HCCL_MIN_SLICE_ALIGN_910B;
        u64 dataSize = memSliceSize_;
        u64 dataCount = sliceCount_;
        if (sliceId == (interRankSize_ * intraRankSize_ - 1)) {
            dataSize = lastSliceSize_;
            dataCount = lastSliceCount_;
        }
        DeviceMem src = DeviceMem::create(static_cast<u8*>(usrInMem_) + srcOffset, dataSize);
        DeviceMem dst = DeviceMem::create(static_cast<u8*>(remoteMemPtr) + offset, dataSize);

        CHK_RET(HcclReduceAsync(
            dispatcher_, src.ptr(), dataCount, dataType_, reductionOp_, subStreams_[i - 1], dst.ptr(),
            intraLinks_[remIntraRankId]->GetRemoteRank(), intraLinks_[remIntraRankId]->GetLinkType(),
            INLINE_REDUCE_BIT));

        CHK_RET(intraLinks_[remIntraRankId]->TxDataSignal(subStreams_[i - 1]));
        CHK_RET(intraLinks_[remIntraRankId]->RxDataSignal(subStreams_[i - 1]));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOpbasePipeline::RunAllGatherIntraServer(u32 step)
{
    u32 dmaMemSliceNum = dmaMem_.size();
    u32 dmaMemSliceId = (interRankSize_ + step - 1) % dmaMemSliceNum;
    for (u32 i = 1; i < intraRankSize_; i++) {
        u32 remIntraRankId = (intraRankId_ + i) % intraRankSize_;
        CHK_RET(intraLinks_[remIntraRankId]->TxAck(subStreams_[i - 1]));
        CHK_RET(intraLinks_[remIntraRankId]->RxAck(subStreams_[i - 1]));
        void* remoteMemPtr = nullptr;
        CHK_RET(intraLinks_[remIntraRankId]->GetRemoteMem(
            dmaMemSliceId == 1 ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM, &remoteMemPtr));
        u32 sliceId = ((interRankId_ + step) % interRankSize_) * intraRankSize_ + remIntraRankId;
        u64 dstOffset = sliceId * memSliceSize_;
        u64 offset = dstOffset % HCCL_MIN_SLICE_ALIGN_910B;
        u64 dataSize = memSliceSize_;
        if (sliceId == (interRankSize_ * intraRankSize_ - 1)) {
            dataSize = lastSliceSize_;
        }
        DeviceMem src = DeviceMem::create(static_cast<u8*>(remoteMemPtr) + offset, dataSize);
        DeviceMem dst = DeviceMem::create(static_cast<u8*>(usrOutMem_) + dstOffset, dataSize);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, subStreams_[i - 1], intraLinks_[remIntraRankId]->GetRemoteRank(),
            intraLinks_[remIntraRankId]->GetLinkType()));
        CHK_RET(intraLinks_[remIntraRankId]->TxDataSignal(subStreams_[i - 1]));
        CHK_RET(intraLinks_[remIntraRankId]->RxDataSignal(subStreams_[i - 1]));
    }
    return HCCL_SUCCESS;
}

HcclResult
AllReduceOpbasePipeline::RunReduceScatterInterServer(u32 step, const LINK& prevInterLink, const LINK& nextInterLink)
{
    CHK_RET(prevInterLink->TxAck(stream_));
    CHK_RET(nextInterLink->RxAck(stream_));
    u32 dmaMemSliceNum = dmaMem_.size();
    u32 rxDMAMemSliceId = (step + 1) % dmaMemSliceNum;
    u32 txDMAMemSliceId = step % dmaMemSliceNum;
    UserMemType srcMemType = txDMAMemSliceId == 1 ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM;
    UserMemType dstMemType = rxDMAMemSliceId == 1 ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM;
    u32 txSliceId = ((interRankId_ + 1 + step) % interRankSize_) * intraRankSize_ + intraRankId_;
    u64 sliceMemOffset = memSliceSize_ * txSliceId;
    u64 offset = sliceMemOffset % HCCL_MIN_SLICE_ALIGN_910B;
    void* txLocalAddr = static_cast<u8*>(dmaMem_[txDMAMemSliceId].ptr()) + offset;
    u64 txDataSize = memSliceSize_;
    if (txSliceId == (interRankSize_ * intraRankSize_ - 1)) {
        txDataSize = lastSliceSize_;
    }
    DeviceMem srcMem = DeviceMem::create(txLocalAddr, txDataSize);
    CHK_RET(senderInfo_->run(nextInterLink, offset, srcMem, stream_, dstMemType));
    HCCL_DEBUG(
        "[AllReduceOpbasePipeline][RunReduceScatterInterServer] local rank[%u], localOffset[%llu],"
        "tx with slice[%llu]",
        rankId_, sliceMemOffset, curSize_);
    u32 rxSliceId = ((interRankId_ + 2 + step) % interRankSize_) * intraRankSize_ + intraRankId_;
    u64 rxSliceOffset = memSliceSize_ * rxSliceId;
    u64 rxOffset = (rxSliceOffset) % HCCL_MIN_SLICE_ALIGN_910B;
    void* rxLocalAddr = static_cast<u8*>(dmaMem_[rxDMAMemSliceId].ptr()) + rxOffset;
    u64 rxDataSize = memSliceSize_;
    if (rxSliceId == (interRankSize_ * intraRankSize_ - 1)) {
        rxDataSize = lastSliceSize_;
    }
    DeviceMem rxLocalMem = DeviceMem::create(rxLocalAddr, rxDataSize);
    DstMemType resultMem = DstMemType::RESULT_INPUT_MEM;
    CHK_RET(reducerInfo_->run(
        dispatcher_, prevInterLink, rxOffset, rxLocalMem, rxLocalMem, rxLocalMem, stream_, resultMem, srcMemType));
    return HCCL_SUCCESS;
}

HcclResult
AllReduceOpbasePipeline::RunAllGatherInterServer(u32 step, const LINK& prevInterLink, const LINK& nextInterLink)
{
    CHK_RET(prevInterLink->TxAck(stream_));
    CHK_RET(nextInterLink->RxAck(stream_));
    u32 dmaMemSliceNum = dmaMem_.size();
    u32 rxDMAMemSliceId = (interRankSize_ + step) % dmaMemSliceNum;
    u32 txDMAMemSliceId = (interRankSize_ + step - 1) % dmaMemSliceNum;
    UserMemType srcMemType = txDMAMemSliceId == 1 ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM;
    UserMemType dstMemType = rxDMAMemSliceId == 1 ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM;
    u32 txSliceId = ((interRankId_ + step) % interRankSize_) * intraRankSize_ + intraRankId_;
    u64 sliceMemOffset = memSliceSize_ * txSliceId;
    u64 offset = sliceMemOffset % HCCL_MIN_SLICE_ALIGN_910B;
    void* txLocalAddr = static_cast<u8*>(dmaMem_[txDMAMemSliceId].ptr()) + offset;
    u64 txDataSize = memSliceSize_;
    if (txSliceId == (interRankSize_ * intraRankSize_ - 1)) {
        txDataSize = lastSliceSize_;
    }
    DeviceMem srcMem = DeviceMem::create(txLocalAddr, txDataSize);
    CHK_RET(nextInterLink->TxAsync(
        dstMemType, offset, static_cast<u8*>(dmaMem_[txDMAMemSliceId].ptr()) + offset, txDataSize, stream_));

    u32 rxSliceId = ((interRankId_ + step + 1) % interRankSize_) * intraRankSize_ + intraRankId_;
    u64 rxSliceOffset = memSliceSize_ * rxSliceId;
    u64 rxOffset = rxSliceOffset % HCCL_MIN_SLICE_ALIGN_910B;
    void* rxLocalAddr = static_cast<u8*>(dmaMem_[rxDMAMemSliceId].ptr()) + rxOffset;
    u64 rxDataSize = memSliceSize_;
    if (rxSliceId == (interRankSize_ * intraRankSize_ - 1)) {
        rxDataSize = lastSliceSize_;
    }
    DeviceMem rxLocalMem = DeviceMem::create(rxLocalAddr, rxDataSize);
    CHK_RET(prevInterLink->RxAsync(
        srcMemType, rxOffset, static_cast<u8*>(dmaMem_[rxDMAMemSliceId].ptr()) + rxOffset, rxDataSize,
        stream_)); // wait
    return HCCL_SUCCESS;
}

HcclResult AllReduceOpbasePipeline::CopyToScratchBuffer(u32 step)
{
    u32 dmaMemSliceNum = dmaMem_.size();
    u32 dmaMemSliceId = step % dmaMemSliceNum;
    u32 sliceId = ((interRankId_ + 1 + step) % interRankSize_) * intraRankSize_ + intraRankId_;
    u64 sliceMemOffset = memSliceSize_ * sliceId;
    u64 offset = sliceMemOffset % HCCL_MIN_SLICE_ALIGN_910B;
    // 把一块切片从userIn 做拷贝到CCLBuffer
    void* srcAddr = static_cast<u8*>(usrInMem_) + sliceMemOffset;
    u64 dataSize = memSliceSize_;
    if (sliceId == (interRankSize_ * intraRankSize_ - 1)) {
        dataSize = lastSliceSize_;
    }
    DeviceMem locSrc = DeviceMem::create(srcAddr, dataSize);
    DeviceMem locDst = DeviceMem::create(static_cast<u8*>(dmaMem_[dmaMemSliceId].ptr()) + offset, dataSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDst, locSrc, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllReduceOpbasePipeline::RunAsync()
{
    // inter ring algo
    u32 prevInterRankId = (interRankId_ + 1) % interRankSize_;
    u32 nextInterRankId = (interRankId_ - 1 + interRankSize_) % interRankSize_;
    LINK prevInterLink = interLinks_[prevInterRankId];
    LINK nextInterLink = interLinks_[nextInterRankId];
    // 当前使用2块DMAMem buffer
    u32 dmaMemSliceNum = dmaMem_.size();
    for (u32 step = 0; step < interRankSize_; step++) {
        if (step == 0) {
            // 把第一块切片从userIn 做拷贝到CCLBuffer
            CHK_RET(CopyToScratchBuffer(step));
            CHK_RET(MainRecordSub());
            CHK_RET(SubWaitMain());
        }
        // server内做SDMA的reduce
        CHK_RET(RunReduceScatterIntraServer(step));
        CHK_RET(SubRecordMain());
        CHK_RET(MainWaitSub());
        if (step < interRankSize_ - 1) {
            // 把下一块切片从userIn 做拷贝到CCLBuffer
            CHK_RET(CopyToScratchBuffer(step + 1));
            CHK_RET(MainRecordSub());
            CHK_RET(SubWaitMain());
            // server间做RDMA的reduce，可与下一个step的SDMA并发执行
            CHK_RET(RunReduceScatterInterServer(step, prevInterLink, nextInterLink));
            // 确保step[n+2]的SDMA之前step[n]的RDMA已经完成，防止内存踩踏
            CHK_RET(prevInterLink->PostFinAck(stream_));
            CHK_RET(nextInterLink->WaitFinAck(stream_));
        }
    }
    DeviceMem srcZero = DeviceMem::create(usrInMem_, 0);
    DeviceMem dstZero = DeviceMem::create(usrOutMem_, 0);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstZero, srcZero, stream_));

    for (u32 step = 0; step < interRankSize_; step++) {
        CHK_RET(MainRecordSub());
        CHK_RET(SubWaitMain());
        if (step < interRankSize_ - 1) {
            CHK_RET(RunAllGatherInterServer(step, prevInterLink, nextInterLink));
            CHK_RET(prevInterLink->PostFinAck(stream_));
            CHK_RET(nextInterLink->WaitFinAck(stream_));
            // inter的最后一步需要barrier确保数据发完
            if (step == interRankSize_ - STEP_OFFSET_TWO) {
                CHK_RET(ExecuteBarrier(prevInterLink, nextInterLink));
            }
        }
        CHK_RET(RunAllGatherIntraServer(step));
        CHK_RET(SubRecordMain());
        CHK_RET(MainWaitSub());
        u32 DMAMemSliceId = (interRankSize_ + step - 1) % dmaMemSliceNum;
        u32 sliceId = ((interRankId_ + step) % interRankSize_) * intraRankSize_ + intraRankId_;
        u64 sliceMemOffset = memSliceSize_ * sliceId;
        u64 offset = sliceMemOffset % HCCL_MIN_SLICE_ALIGN_910B;
        void* txLocalAddr = static_cast<u8*>(dmaMem_[DMAMemSliceId].ptr()) + offset;
        u64 dataSize = memSliceSize_;
        if (sliceId == (interRankSize_ * intraRankSize_ - 1)) {
            dataSize = lastSliceSize_;
        }
        DeviceMem srcMem = DeviceMem::create(txLocalAddr, dataSize);
        DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(usrOutMem_) + sliceMemOffset, dataSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream_));
        HCCL_INFO("[AllReducePipeline][RunAsync]AllReducePipeline finished groupRankId[%u] ", rankId_);
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOpbasePipeline::Prepare(
    const HcomCollOpInfo* opInfo, DeviceMem& cclBufferA, DeviceMem& cclBufferB, const u64 count,
    const SubCommInfo& level1CommInfo, const SubCommInfo& level0CommInfo, Stream& mainStream,
    std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
    std::vector<std::shared_ptr<LocalNotify>>& notifySub)
{
    unitSize_ = SIZE_TABLE[opInfo->dataType];
    sliceCount_ = count / (level0CommInfo.localRankSize * level1CommInfo.localRankSize);
    memSliceSize_ = sliceCount_ * unitSize_;
    lastSliceCount_ = count - sliceCount_ * (level0CommInfo.localRankSize * level1CommInfo.localRankSize - 1);
    lastSliceSize_ = lastSliceCount_ * unitSize_;

    usrInMem_ = opInfo->inputAddr;
    usrOutMem_ = opInfo->outputAddr;
    reductionOp_ = opInfo->reduceOp;
    dataType_ = opInfo->dataType;

    // needed resource
    // stream: 1 * mainStream + (n -1) * subStream
    // interNotify, streamNotify

    // stream
    // mainStream负责locMemCpy、inter执行以及subStream同步控制
    stream_ = mainStream;
    // subStream负责:
    // streamId[0:intraRankSize-1]: intraRankSize-1个intra执行
    subStreams_ = subStream;

    intraRankSize_ = level0CommInfo.localRankSize;
    interRankSize_ = level1CommInfo.localRankSize;
    intraRankId_ = level0CommInfo.localRank;
    interRankId_ = level1CommInfo.localRank;
    rankId_ = intraRankId_ + interRankId_ * intraRankSize_;

    // streamNotify, size: n
    streamNotifyMain_ = notifyMain;
    if (streamNotifyMain_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[AllReduceOpbasePipeline][Prepare]rank[%u] streamNotifyMain_ size [%u] is smaller than "
            "intraRankSize_[%u]",
            rankId_, streamNotifyMain_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }
    streamNotifySub_ = notifySub;
    if (streamNotifySub_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[AllReduceOpbasePipeline][Prepare]rank[%u] streamNotifySub_ size [%u] is smaller than "
            "intraRankSize_[%u]",
            rankId_, streamNotifySub_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }

    intraLinks_ = level0CommInfo.links;
    interLinks_ = level1CommInfo.links;

    count_ = count;
    // 2级流水,使用2块DMAMem
    dmaMem_.push_back(cclBufferA);
    dmaMem_.push_back(cclBufferB);

    HCCL_INFO(
        "[AllReduceOpbasePipeline][Prepare]streamNum[%u], streamNotifyMainNum[%u], streamNotifySubNum[%u]",
        subStreams_.size(), streamNotifyMain_.size(), streamNotifySub_.size());
    HCCL_INFO(
        "[AllReduceOpbasePipeline][Prepare]interLinksNum[%u], intraLinksNum[%u]", interLinks_.size(),
        intraLinks_.size());
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);
    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);
    return HCCL_SUCCESS;
}

HcclResult AllReduceOpbasePipeline::GetNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    u32 ringNextRank = (rank + 1) % rankSize;
    LINK nslbNext = links[ringNextRank];
    CHK_SMART_PTR_NULL(nslbNext);

    // Pipeline 步长合并 等同于 ring
    NslbDpAdjInfo adjInfoStep = {};
    adjInfoStep.dstLocalRankId = nslbNext->GetRemoteRank();
    adjInfoStep.phaseId = 1;
    adjInfoStep.rev = 0;
    nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);

    // Pipeline 步长合并 等同于 ring
    NslbDpAdjInfo nextInfoStep = {};
    nextInfoStep.dstLocalRankId = nslbNext->GetRemoteRank();
    nextInfoStep.phaseId = nslbAdjInfo.nsAdjInfo[0].phaseId + 1;
    nextInfoStep.rev = 0;
    nslbAdjInfo.nsAdjInfo.push_back(nextInfoStep);

    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_OPBASE_PIPELINE, AllReduceOpbasePipeline);
} // namespace hccl
