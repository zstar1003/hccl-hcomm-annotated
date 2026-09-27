/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_v_pipeline.h"
#include "alg_template_register.h"

constexpr u32 STEP_OFFSET_TWO = 2;

namespace hccl {
ReduceScatterVPipeline::ReduceScatterVPipeline(const HcclDispatcher dispatcher) : ReduceScatterPipeline(dispatcher) {}

ReduceScatterVPipeline::~ReduceScatterVPipeline() {}

HcclResult ReduceScatterVPipeline::RunIntraServer(u32 step, u64 remoteOffset)
{
    for (u32 i = 1; i < intraRankSize_; i++) {
        u32 remIntraRankId = (intraRankId_ + i) % intraRankSize_;
        CHK_RET(intraLinks_[remIntraRankId]->TxAck(subStream_[i]));
        CHK_RET(intraLinks_[remIntraRankId]->RxAck(subStream_[i]));
        void* remoteMemPtr = nullptr;
        CHK_RET(intraLinks_[remIntraRankId]->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMemPtr));

        // 本次机内待发送数据的index坐标
        u32 index = (((interRankId_ + 1 + step) % interRankSize_) * intraRankSize_ + remIntraRankId);
        Slice userSlice = slices_[index];
        u64 srcOffset = userSlice.offset;

        u64 offset = srcOffset % HCCL_MIN_SLICE_ALIGN_910B;
        DeviceMem src = DeviceMem::create(static_cast<u8*>(usrInMem_) + srcOffset, userSlice.size);
        DeviceMem dst = DeviceMem::create(static_cast<u8*>(remoteMemPtr) + remoteOffset + offset, userSlice.size);

        CHK_RET(HcclReduceAsync(
            dispatcher_, src.ptr(), userSlice.size / unitSize_, dataType_, reductionOp_, subStream_[i], dst.ptr(),
            intraLinks_[remIntraRankId]->GetRemoteRank(), intraLinks_[remIntraRankId]->GetLinkType(),
            INLINE_REDUCE_BIT));

        CHK_RET(intraLinks_[remIntraRankId]->TxDataSignal(subStream_[i]));
        CHK_RET(intraLinks_[remIntraRankId]->RxDataSignal(subStream_[i]));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVPipeline::RunInterServer(u32 step, const LINK& prevInterLink, const LINK& nextInterLink)
{
    u32 dmaMemSliceNum = dmaMem_.size();
    u32 rxDMAMemSliceId = (step + 1) % dmaMemSliceNum;
    u32 txDMAMemSliceId = step % dmaMemSliceNum;

    u32 txindex = (((interRankId_ + 1 + step) % interRankSize_) * intraRankSize_ + intraRankId_);
    Slice txUserSlice = slices_[txindex];
    u64 txSliceOffset = txUserSlice.offset;
    u64 offset = txSliceOffset % HCCL_MIN_SLICE_ALIGN_910B;
    u64 rxInterOffset = rxDMAMemSliceId * blockSize_ + offset;
    void* txLocalAddr = static_cast<u8*>(dmaMem_[txDMAMemSliceId].ptr()) + offset;

    DeviceMem srcMem = DeviceMem::create(txLocalAddr, txUserSlice.size);
    CHK_RET(senderInfo_->run(nextInterLink, rxInterOffset, srcMem, subStream_[0])); // 发 srcMem -> rxInterOffset(rem)
    HCCL_DEBUG(
        "[ReduceScatterVPipeline][RunInterServer] local rank[%u] localOffset[%llu]tx with slice[%llu]", rankId_,
        rxInterOffset, txUserSlice.size);

    u32 rxindex = (((interRankId_ + 2 + step) % interRankSize_) * intraRankSize_ + intraRankId_);
    Slice rxUserSlice = slices_[rxindex];
    u64 rxSliceOffset = rxUserSlice.offset;
    u64 rxOffset = rxSliceOffset % HCCL_MIN_SLICE_ALIGN_910B;
    u64 rxMemOffset = txDMAMemSliceId * blockSize_ + rxOffset;
    void* rxLocalAddr = static_cast<u8*>(dmaMem_[rxDMAMemSliceId].ptr()) + rxOffset;

    DeviceMem rxLocalMem = DeviceMem::create(rxLocalAddr, rxUserSlice.size);
    CHK_RET(reducerInfo_->run(
        dispatcher_, prevInterLink, rxMemOffset, rxLocalMem, rxLocalMem,
        rxLocalMem, // 收 rxMemOffset(rem) -> rxLocalMem
        subStream_[0]));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVPipeline::CopyToScratchBuffer(u32 step)
{
    u32 dmaMemSliceNum = dmaMem_.size();
    u32 dmaMemSliceId = step % dmaMemSliceNum;

    u32 index = (((interRankId_ + 1 + step) % interRankSize_) * intraRankSize_ + intraRankId_);
    Slice userslice = slices_[index];
    u64 srcOffset = userslice.offset;
    u64 offset = srcOffset % HCCL_MIN_SLICE_ALIGN_910B;

    void* srcAddr = static_cast<u8*>(usrInMem_) + srcOffset;
    DeviceMem locSrc = DeviceMem::create(srcAddr, userslice.size);
    DeviceMem locDst = DeviceMem::create(static_cast<u8*>(dmaMem_[dmaMemSliceId].ptr()) + offset, userslice.size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDst, locSrc, stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVPipeline::RunAsync()
{
    // inter ring algo
    u32 prevInterRankId = (interRankId_ + 1) % interRankSize_;
    u32 nextInterRankId = (interRankId_ - 1 + interRankSize_) % interRankSize_;
    LINK prevInterLink = interLinks_[prevInterRankId];
    LINK nextInterLink = interLinks_[nextInterRankId];
    // 当前使用3块DMAMem buffer
    u32 dmaMemSliceNum = dmaMem_.size();

    for (u32 step = 0; step < interRankSize_; step++) {
        u32 begin = 0;
        if (step == 0) {
            begin = 1;
            CHK_RET(CopyToScratchBuffer(step));
            CHK_RET(MainRecordSub(begin));
            CHK_RET(SubWaitMain(begin));
        }
        // // server内做SDMA的reduce
        u64 remoteOffset = (step % dmaMemSliceNum) * blockSize_;
        CHK_RET(RunIntraServer(step, remoteOffset));
        CHK_RET(SubRecordMain(begin));
        CHK_RET(MainWaitSub(begin));
        if (step < interRankSize_ - 1) {
            // 把下一块切片从userIn 做拷贝到CCLBuffer
            CHK_RET(CopyToScratchBuffer(step + 1));
            // // 全部流同步，确保SDMA执行完成
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
    Slice userSlice = slices_[rankId_];
    u64 srcOffset = userSlice.offset % HCCL_MIN_SLICE_ALIGN_910B;
    void* locSrcAddr = static_cast<u8*>(dmaMem_[(interRankSize_ - 1) % dmaMemSliceNum].ptr()) + srcOffset;

    DeviceMem locSrc = DeviceMem::create(locSrcAddr, userSlice.size);
    DeviceMem locDst = DeviceMem::create(static_cast<u8*>(usrOutMem_), userSlice.size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDst, locSrc, stream_));
    HCCL_INFO("[ReduceScatterVPipeline][RunAsync]ReduceScatterVPipeline finished rankId[%u] ", rankId_);
    return HCCL_SUCCESS;
}

// 适配新CollExecutor接口
HcclResult ReduceScatterVPipeline::Prepare(
    HcomCollOpInfo* opInfo, DeviceMem& cclBuffer, const u64 bufferSize, const std::vector<Slice>& slices,
    const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo, Stream& mainStream,
    std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
    std::vector<std::shared_ptr<LocalNotify>>& notifySub, u64 reduceAttrBitMap)
{
    reduceAttr_ = reduceAttrBitMap;
    opInfo_ = opInfo;

    unitSize_ = SIZE_TABLE[opInfo_->dataType];
    usrInMem_ = opInfo_->inputAddr;
    usrOutMem_ = opInfo_->outputAddr;
    reductionOp_ = opInfo_->reduceOp;
    dataType_ = opInfo_->dataType;
    slices_ = slices;

    stream_ = mainStream;
    subStream_ = subStream;

    intraRankSize_ = level0CommInfo.localRankSize;
    interRankSize_ = level1CommInfo.localRankSize;
    intraRankId_ = level0CommInfo.localRank;
    interRankId_ = level1CommInfo.localRank;
    rankId_ = intraRankId_ + interRankId_ * intraRankSize_;

    streamNotifyMain_ = notifyMain;
    if (streamNotifyMain_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[ReduceScatterVPipeline][Prepare]rank[%u] streamNotifyMain_ size [%u] error, is smaller than,"
            "intraRankSize_[%u]",
            rankId_, streamNotifyMain_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }
    streamNotifySub_ = notifySub;
    if (streamNotifySub_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[ReduceScatterVPipeline][Prepare]rank[%u] streamNotifySub_ size [%u] error, is smaller than,"
            "intraRankSize_[%u]",
            rankId_, streamNotifySub_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }

    intraLinks_ = level0CommInfo.links;
    interLinks_ = level1CommInfo.links;

    // 3级流水,使用3块DMAMem
    cclBuffer_ = cclBuffer;
    bufferSize_ = bufferSize;

    blockSize_ = (bufferSize_ / (HCCL_MIN_SLICE_ALIGN_910B * PIPELINE_DEPTH)) * HCCL_MIN_SLICE_ALIGN_910B;

    for (u32 i = 0; i < pipDepth_; i++) {
        DeviceMem mem = DeviceMem::create(static_cast<u8*>(cclBuffer_.ptr()) + blockSize_ * i, blockSize_);
        dmaMem_.push_back(mem);
    }

    HCCL_INFO(
        "[ReduceScatterVPipeline][Prepare]streamNum[%u], streamNotifyMainNum[%u], streamNotifySubNum[%u]",
        subStream_.size(), streamNotifyMain_.size(), streamNotifySub_.size());
    HCCL_INFO(
        "[ReduceScatterVPipeline][Prepare]interLinksNum[%u], intraLinksNum[%u]", interLinks_.size(),
        intraLinks_.size());
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);
    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_V_PIPELINE, ReduceScatterVPipeline);
} // namespace hccl
