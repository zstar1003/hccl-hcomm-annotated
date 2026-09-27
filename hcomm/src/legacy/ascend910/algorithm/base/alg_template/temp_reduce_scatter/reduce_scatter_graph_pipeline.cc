/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_graph_pipeline.h"
#include "alg_template_register.h"

constexpr u32 STEP_OFFSET_TWO = 2;

namespace hccl {
ReduceScatterGraphPipeline::ReduceScatterGraphPipeline(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ReduceScatterGraphPipeline::~ReduceScatterGraphPipeline() {}

HcclResult ReduceScatterGraphPipeline::MainWaitSub(u32 begin)
{
    u32 subStreamNum = intraRankSize_;
    for (u32 signalIndex = begin; signalIndex < subStreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, streamNotifyMain_[signalIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterGraphPipeline::SubRecordMain(u32 begin)
{
    u32 subStreamNum = intraRankSize_;
    for (u32 streamIndex = begin; streamIndex < subStreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Post(subStream_[streamIndex], dispatcher_, streamNotifyMain_[streamIndex], -1));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterGraphPipeline::MainRecordSub(u32 begin)
{
    u32 subStreamNum = intraRankSize_;
    for (u32 signalIndex = begin; signalIndex < subStreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, streamNotifySub_[signalIndex], -1));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterGraphPipeline::SubWaitMain(u32 begin)
{
    u32 subStreamNum = intraRankSize_;
    for (u32 streamIndex = begin; streamIndex < subStreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            subStream_[streamIndex], dispatcher_, streamNotifySub_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterGraphPipeline::RunIntraServer(u64 blockIdx)
{
    u64 blockOff = blockIdx * intraRankSize_;
    u64 memOffset = (blockOff + intraRankId_) * memSliceSize_;
    for (u32 i = 1; i < intraRankSize_; i++) {
        u32 remIntraRankId = (intraRankId_ + i) % intraRankSize_;
        CHK_RET(intraLinks_[remIntraRankId]->TxAck(subStream_[i]));
        CHK_RET(intraLinks_[remIntraRankId]->RxAck(subStream_[i]));
        void* remoteMemPtr = nullptr;
        CHK_RET(intraLinks_[remIntraRankId]->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMemPtr));
        DeviceMem dst = DeviceMem::create(static_cast<u8*>(usrInMem_) + memOffset, memSliceSize_);
        DeviceMem src = DeviceMem::create(static_cast<u8*>(remoteMemPtr) + memOffset, memSliceSize_);

        CHK_RET(HcclReduceAsync(
            dispatcher_, src.ptr(), count_, dataType_, reductionOp_, subStream_[i], dst.ptr(),
            intraLinks_[remIntraRankId]->GetRemoteRank(), intraLinks_[remIntraRankId]->GetLinkType(),
            INLINE_REDUCE_BIT));

        CHK_RET(intraLinks_[remIntraRankId]->TxDataSignal(subStream_[i]));
        CHK_RET(intraLinks_[remIntraRankId]->RxDataSignal(subStream_[i]));
    }
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterGraphPipeline::RunInterServer(u64 blockIdx, const LINK& prevInterLink, const LINK& nextInterLink)
{
    u64 blockOff = blockIdx * intraRankSize_;
    u64 memOffset = (blockOff + intraRankId_) * memSliceSize_;
    u64 preBlockOff = ((blockIdx + 1) % interRankSize_) * intraRankSize_;
    u64 preMemOffset = (preBlockOff + intraRankId_) * memSliceSize_;

    DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(usrInMem_) + memOffset, memSliceSize_);
    CHK_RET(senderInfo_->run(nextInterLink, memOffset, srcMem, subStream_[0], UserMemType::INPUT_MEM));
    HCCL_DEBUG(
        "[ReduceScatterGraphPipeline][RunInterServer] local rank[%u] localOffset[%llu]tx with slice[%llu]", rankId_,
        memOffset, memSliceSize_);

    DeviceMem rxLocalMem = DeviceMem::create(static_cast<u8*>(usrInMem_) + preMemOffset, memSliceSize_);
    CHK_RET(
        reducerInfo_->run(dispatcher_, prevInterLink, preMemOffset, rxLocalMem, rxLocalMem, rxLocalMem, subStream_[0]));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterGraphPipeline::RunAsync()
{
    // inter ring algo
    u32 prevInterRankId = (interRankId_ + 1) % interRankSize_;
    u32 nextInterRankId = (interRankId_ - 1 + interRankSize_) % interRankSize_;
    LINK prevInterLink = interLinks_[prevInterRankId];
    LINK nextInterLink = interLinks_[nextInterRankId];

    for (u32 step = 0; step < interRankSize_; step++) {
        u32 begin = 0;
        if (step == 0) {
            begin = 1;
            CHK_RET(MainRecordSub(begin));
            CHK_RET(SubWaitMain(begin));
        }
        // server内做SDMA的reduce
        u64 blockIdx = ((interRankId_ + step + 1) % interRankSize_);
        CHK_RET(RunIntraServer(blockIdx));
        CHK_RET(SubRecordMain(begin));
        CHK_RET(MainWaitSub(begin));
        if (step < interRankSize_ - 1) {
            // 全部流同步，确保SDMA执行完成
            CHK_RET(MainRecordSub(0));
            CHK_RET(SubWaitMain(0));
            CHK_RET(prevInterLink->TxAck(subStream_[0]));
            CHK_RET(nextInterLink->RxAck(subStream_[0]));
            // server间做RDMA的reduce，可与下一个step的SDMA并发执行
            CHK_RET(RunInterServer(blockIdx, prevInterLink, nextInterLink));
            CHK_RET(prevInterLink->PostFinAck(subStream_[0]));
            CHK_RET(nextInterLink->WaitFinAck(subStream_[0]));
            // inter的最后一步需要barrier确保数据发完
            if (step == interRankSize_ - STEP_OFFSET_TWO) {
                CHK_RET(ExecuteBarrier(prevInterLink, nextInterLink, subStream_[0]));
            }
        }
    }
    // 把对应的切片从usrIn拷贝到userOut
    DeviceMem locSrc = DeviceMem::create(static_cast<u8*>(usrInMem_) + rankId_ * memSliceSize_, memSliceSize_);
    DeviceMem locDst = DeviceMem::create(static_cast<u8*>(usrOutMem_), memSliceSize_);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDst, locSrc, stream_));
    HCCL_INFO("[ReduceScatterGraphPipeline][RunAsync]ReduceScatterGraphPipeline finished groupRankId[%u] ", rankId_);
    return HCCL_SUCCESS;
}

// 适配新CollExecutor接口
HcclResult ReduceScatterGraphPipeline::Prepare(
    HcomCollOpInfo* opInfo, [[maybe_unused]] DeviceMem& cclBuffer, [[maybe_unused]] const u64 count,
    [[maybe_unused]] const u64 bufferSize, const u64 offset, const SubCommInfo& level0CommInfo,
    const SubCommInfo& level1CommInfo, Stream& mainStream, std::vector<Stream>& subStream,
    std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub,
    u64 reduceAttrBitMap)
{
    reduceAttr_ = reduceAttrBitMap;
    opInfo_ = opInfo;

    unitSize_ = SIZE_TABLE[opInfo_->dataType];
    count_ = opInfo_->count;
    memSliceSize_ = opInfo_->count * unitSize_;
    usrInMem_ = opInfo_->inputAddr;
    usrOutMem_ = opInfo_->outputAddr;
    reductionOp_ = opInfo_->reduceOp;
    dataType_ = opInfo_->dataType;
    offset_ = offset;

    // needed resource
    // stream: 1 * mainStream + n * subStream
    // mem: usrInMem_, usrOutMem
    // interNotify, streamNotify

    // stream
    // mainStream负责locMemCPY以及subStream同步控制
    stream_ = mainStream;
    // subStream负责：
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
            "[ReduceScatterGraphPipeline][Prepare]rank[%u] streamNotifyMain_ size [%u] error, smaller than "
            "intraRankSize_[%u]",
            rankId_, streamNotifyMain_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }
    streamNotifySub_ = notifySub;
    if (streamNotifySub_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[ReduceScatterGraphPipeline][Prepare]rank[%u] streamNotifySub_ size [%u] error, smaller than "
            "intraRankSize_[%u]",
            rankId_, streamNotifySub_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }

    intraLinks_ = level0CommInfo.links;
    interLinks_ = level1CommInfo.links;

    HCCL_INFO(
        "[ReduceScatterGraphPipeline][Prepare]streamNum[%u], streamNotifyMainNum[%u], streamNotifySubNum[%u]",
        subStream_.size(), streamNotifyMain_.size(), streamNotifySub_.size());
    HCCL_INFO(
        "[ReduceScatterGraphPipeline][Prepare]interLinksNum[%u], intraLinksNum[%u]", interLinks_.size(),
        intraLinks_.size());
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);
    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_GRAPH_PIPELINE, ReduceScatterGraphPipeline);
} // namespace hccl
