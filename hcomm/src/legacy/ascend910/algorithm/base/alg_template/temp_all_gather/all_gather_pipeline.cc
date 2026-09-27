/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_pipeline.h"
#include "alg_template_register.h"

constexpr u32 STEP_OFFSET_TWO = 2;

namespace hccl {
AllGatherPipeline::AllGatherPipeline(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllGatherPipeline::~AllGatherPipeline() {}

HcclResult AllGatherPipeline::Prepare(
    HcomCollOpInfo* opInfo, u32 userRank, u64& count, DeviceMem& cclBufferPartOne, DeviceMem& cclBufferPartTwo,
    SubCommInfo& level0CommInfo, SubCommInfo& level1CommInfo, Stream& mainStream, std::vector<Stream>& subStream,
    std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub)
{
    opInfo_ = opInfo;
    memSliceCount_ = count;
    userRank_ = userRank;

    u32 unitSize = SIZE_TABLE[opInfo_->dataType];
    u64 memSliceSize = memSliceCount_ * unitSize;

    usrInMemAddr_ = opInfo_->inputAddr;
    usrOutMemAddr_ = opInfo_->outputAddr;

    // needed resource
    // stream: 1 * mainStream + n * subStream
    // mem: usrInMem, usrOutMem, DMAMem
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
    intraLinks_ = level0CommInfo.links;
    interLinks_ = level1CommInfo.links;

    // streamNotify, size: n
    streamNotifyMain_ = notifyMain;
    if (streamNotifyMain_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[AllGatherPipeline][Prepare]rank[%u] streamNotifyMain_ size[%u] is smaller than "
            "intraRankSize_[%u]",
            userRank_, streamNotifyMain_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }
    streamNotifySub_ = notifySub;
    if (streamNotifySub_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[AllGatherPipeline][Prepare]rank[%u] streamNotifySub_ size[%u] is smaller than "
            "intraRankSize_[%u]",
            userRank_, streamNotifySub_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }

    // 128byte align offset
    DeviceMem dmaMem0 = DeviceMem::create(cclBufferPartOne.ptr(), memSliceSize);
    DeviceMem dmaMem1 = DeviceMem::create(cclBufferPartTwo.ptr(), memSliceSize);

    dmaMem_.push_back(dmaMem0);
    dmaMem_.push_back(dmaMem1);

    HCCL_INFO(
        "[AllGatherPipeline][Prepare]streamNum[%zu], streamNotifyMainNum[%zu], streamNotifySubNum[%zu].",
        subStream_.size(), streamNotifyMain_.size(), streamNotifySub_.size());
    HCCL_INFO(
        "[AllGatherPipeline][Prepare]interLinksNum[%zu], intraLinksNum[%zu].", interLinks_.size(), intraLinks_.size());
    return HCCL_SUCCESS;
}

HcclResult AllGatherPipeline::MainWaitSub()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 signalIndex = 0; signalIndex < subStreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, streamNotifyMain_[signalIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherPipeline::SubRecordMain()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 streamIndex = 0; streamIndex < subStreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Post(
            subStream_[streamIndex], dispatcher_, streamNotifyMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherPipeline::MainRecordSub()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 signalIndex = 0; signalIndex < subStreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, streamNotifySub_[signalIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherPipeline::SubWaitMain()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 streamIndex = 0; streamIndex < subStreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            subStream_[streamIndex], dispatcher_, streamNotifySub_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherPipeline::RunAsync()
{
    HCCL_INFO("[AllGatherPipeline][RunAsync]AllGatherRingMesh starts groupRankId[%u]. ", userRank_);
    // inter ring algo
    u32 prevInterRankId = (interRankId_ - 1 + interRankSize_) % interRankSize_;
    u32 nextInterRankId = (interRankId_ + 1) % interRankSize_;
    LINK prevInterLink = interLinks_[prevInterRankId];
    LINK nextInterLink = interLinks_[nextInterRankId];

    // intra fullmesh algo
    // intra使用全部连接，不再映射

    u32 unitSize = SIZE_TABLE[opInfo_->dataType];
    u64 memSliceSize = memSliceCount_ * unitSize;
    u64 memSliceOffset = opInfo_->count * unitSize;

    // 仅使用两块DMAMem，为了方便切换使用
    u32 dmaMemSliceId = 0;
    u32 dmaMemSliceNum = dmaMem_.size();

    // step 0前置操作 : 所有卡本地数据从userIn-->DMAIn
    DeviceMem locSrc = DeviceMem::create(usrInMemAddr_, memSliceSize);
    u64 localOffset = (opInfo_->count * userRank_ * unitSize) % HCCL_MIN_SLICE_ALIGN_910B;
    DeviceMem locDMAInMem
        = DeviceMem::create(static_cast<u8*>(dmaMem_[dmaMemSliceId].ptr()) + localOffset, memSliceSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDMAInMem, locSrc, stream_));

    for (u32 step = 0; step < interRankSize_; step++) {
        // 主从流同步
        CHK_RET(MainRecordSub());
        CHK_RET(SubWaitMain());

        // 数据搬运及后同步
        u32 srcDMAMemSliceId = dmaMemSliceId;
        dmaMemSliceId = (dmaMemSliceId + 1) % dmaMemSliceNum;
        u32 dstDMAMemSliceId = dmaMemSliceId;

        u64 serverRankOffset = intraRankId_ + (interRankId_ + interRankSize_ - step) % interRankSize_ * intraRankSize_;
        u64 serverOffsetByte = (opInfo_->count * serverRankOffset * unitSize) % HCCL_MIN_SLICE_ALIGN_910B;
        u64 readRemoteOffset
            = intraRankId_
              + (prevInterRankId + interRankSize_ - step) % interRankSize_ * intraRankSize_; // sever间前通信rank偏移
        u64 readRemoteOffsetByte = (opInfo_->count * readRemoteOffset * unitSize) % HCCL_MIN_SLICE_ALIGN_910B;
        if (step < interRankSize_ - 1) {
            CHK_RET(prevInterLink->TxAck(subStream_[0])); // AckRecord
            CHK_RET(nextInterLink->RxAck(subStream_[0])); // AckWait
            // RdmaSend + Record 或 PCIE::Record
            CHK_RET(nextInterLink->TxAsync(
                (dstDMAMemSliceId == 1 ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM), serverOffsetByte,
                static_cast<u8*>(dmaMem_[srcDMAMemSliceId].ptr()) + serverOffsetByte, memSliceSize, subStream_[0]));
            HCCL_DEBUG(
                "[AllGatherPipeline][RunAsync] local rank[%u] localOffset[%llu] tx with remoteRank[%u],"
                "remoteOffset[%llu] with slice[%llu].",
                userRank_, serverOffsetByte, nextInterRankId, serverOffsetByte, memSliceSize);
            // 对于RDM RxAsync，内存属性入参无效 RDMA::Wait
            // 对于PCIE，需设置内存属性 PCIE::Read + Record
            CHK_RET(prevInterLink->RxAsync(
                (srcDMAMemSliceId == 0 ? UserMemType::INPUT_MEM : UserMemType::OUTPUT_MEM), readRemoteOffsetByte,
                static_cast<u8*>(dmaMem_[dstDMAMemSliceId].ptr()) + readRemoteOffsetByte, memSliceSize,
                subStream_[0])); // wait
            HCCL_DEBUG(
                "[AllGatherPipeline][RunAsync]read local rank[%u] localOffset[%llu] tx with remoteRank[%u],"
                "remoteOffset[%llu] with slice[%llu].",
                userRank_, readRemoteOffsetByte, readRemoteOffset, readRemoteOffsetByte, memSliceSize);
            CHK_RET(prevInterLink->PostFinAck(subStream_[0]));
            CHK_RET(nextInterLink->WaitFinAck(subStream_[0]));
            // inter的最后一步需要barrier确保数据发完
            if (step == interRankSize_ - STEP_OFFSET_TWO) {
                CHK_RET(ExecuteBarrier(prevInterLink, nextInterLink, subStream_[0]));
            }
        }

        for (u32 i = 1; i < intraRankSize_; i++) {
            u32 remIntraRankId = (intraRankId_ + i) % intraRankSize_;
            CHK_RET(intraLinks_[remIntraRankId]->TxAck(subStream_[i])); // ack record
            CHK_RET(intraLinks_[remIntraRankId]->RxAck(subStream_[i]));
            void* remDMAMemPtr = nullptr;
            CHK_RET(intraLinks_[remIntraRankId]->GetRemoteMem(
                srcDMAMemSliceId == 1 ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM, &remDMAMemPtr));
            void* dstAddr
                = static_cast<u8*>(usrOutMemAddr_)
                  + ((interRankId_ - step + interRankSize_) % interRankSize_ * intraRankSize_ + remIntraRankId)
                        * memSliceOffset;

            u64 remoteOffsetByte = (opInfo_->count * (remIntraRankId + serverRankOffset - intraRankId_) * unitSize)
                                   % HCCL_MIN_SLICE_ALIGN_910B;
            DeviceMem src = DeviceMem::create(static_cast<u8*>(remDMAMemPtr) + remoteOffsetByte, memSliceSize);
            DeviceMem dst = DeviceMem::create(dstAddr, memSliceSize);
            HCCL_DEBUG("[AllGatherPipeline][RunAsync]remoteOffsetByte is %llu", remoteOffsetByte);
            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, dst, src, subStream_[i], intraLinks_[remIntraRankId]->GetRemoteRank(),
                intraLinks_[remIntraRankId]->GetLinkType()));
            CHK_RET(intraLinks_[remIntraRankId]->TxDataSignal(subStream_[i])); // data record
            CHK_RET(intraLinks_[remIntraRankId]->RxDataSignal(subStream_[i])); // data wait
        }

        CHK_RET(SubRecordMain());
        CHK_RET(MainWaitSub());

        void* dstAddr = static_cast<u8*>(usrOutMemAddr_)
                        + ((interRankId_ - step + interRankSize_) % interRankSize_ * intraRankSize_ + intraRankId_)
                              * memSliceOffset;
        DeviceMem locDst = DeviceMem::create(dstAddr, memSliceSize);
        DeviceMem srcMem
            = DeviceMem::create(static_cast<u8*>(dmaMem_[srcDMAMemSliceId].ptr()) + serverOffsetByte, memSliceSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDst, srcMem, stream_));
    }

    HCCL_INFO("[AllGatherPipeline][RunAsync]AllGatherRingMesh finished groupRankId[%u] ", userRank_);
    return HCCL_SUCCESS;
}

HcclResult AllGatherPipeline::GetNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    HCCL_DEBUG("[AllGatherPipeline]GetNslbAdjInfo start");
    u32 ringNextRank = (rank + 1) % rankSize;
    LINK nslbNext = links[ringNextRank];
    CHK_SMART_PTR_NULL(nslbNext);

    // Pipeline 步长合并 等同于 ring
    NslbDpAdjInfo adjInfoStep = {};
    nslbAdjInfo.dstRankNum = 1;
    adjInfoStep.dstLocalRankId = nslbNext->GetRemoteRank();
    adjInfoStep.phaseId = 1;
    adjInfoStep.rev = 0;
    nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_PIPELINE, AllGatherPipeline);
} // namespace hccl
