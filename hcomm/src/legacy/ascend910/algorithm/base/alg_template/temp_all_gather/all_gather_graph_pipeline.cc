/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_graph_pipeline.h"
#include "alg_template_register.h"

constexpr u32 STEP_OFFSET_TWO = 2;

namespace hccl {
AllGatherGraphPipeline::AllGatherGraphPipeline(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllGatherGraphPipeline::~AllGatherGraphPipeline() {}

HcclResult AllGatherGraphPipeline::Prepare(
    HcomCollOpInfo* opInfo, u32 userRank, u64& count, DeviceMem& inputMem, DeviceMem& outputMem,
    SubCommInfo& level0CommInfo, SubCommInfo& level1CommInfo, Stream& mainStream, std::vector<Stream>& subStream,
    std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub)
{
    opInfo_ = opInfo;
    memSliceCount_ = count;
    userRank_ = userRank;

    u32 unitSize = SIZE_TABLE[opInfo->dataType];
    u64 memSliceSize = memSliceCount_ * unitSize;

    usrInMemAddr_ = opInfo_->inputAddr;
    usrOutMemAddr_ = opInfo_->outputAddr;

    // needed resource
    // stream: 1 * mainStream + n * subStream
    // mem: usrInMem, usrOutMem
    // intorNotify, streamNotify

    // stream
    // mainStream负责locMemCpy和subStream同步控制
    stream_ = mainStream;
    // subStream负责：
    // streamId[0]: inter执行
    // streamId[1:intraRankSize]: intraRankSize-1个intra执行
    subStream_ = subStream;

    HCCL_DEBUG("[AllGatherGraphPipeline]prepare for userRank is %u, memSliceCount is %llu", userRank_, memSliceCount_);
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
            "[AllGatherGraphPipeline][Prepare]rank[%u] streamNotifyMain_ size[%u] is smaller than "
            "intraRankSize_[%u]",
            userRank_, streamNotifyMain_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }
    streamNotifySub_ = notifySub;
    if (streamNotifySub_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[AllGatherGraphPipeline][Prepare]rank[%u] streamNotifySub_ size[%u] is smaller than "
            "intraRankSize_[%u]",
            userRank_, streamNotifySub_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }

    DeviceMem dmaMem0 = DeviceMem::create(inputMem.ptr(), memSliceSize);
    DeviceMem dmaMem1 = DeviceMem::create(outputMem.ptr(), memSliceSize);

    dmaMem_.push_back(dmaMem0);
    dmaMem_.push_back(dmaMem1);

    HCCL_INFO(
        "[AllGatherGraphPipeline][Prepare]streamNum[%zu], streamNotifyMainNum[%zu], streamNotifySubNum[%zu]",
        subStream_.size(), streamNotifyMain_.size(), streamNotifySub_.size());
    HCCL_INFO(
        "[AllGatherGraphPipeline][Prepare]interLinksNum[%zu], intraLinksNum[%zu]", interLinks_.size(),
        intraLinks_.size());
    return HCCL_SUCCESS;
}

HcclResult AllGatherGraphPipeline::MainWaitSub()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 signalIndex = 0; signalIndex < subStreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, streamNotifyMain_[signalIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherGraphPipeline::SubRecordMain()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 streamIndex = 0; streamIndex < subStreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Post(
            subStream_[streamIndex], dispatcher_, streamNotifyMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherGraphPipeline::MainRecordSub()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 signalIndex = 0; signalIndex < subStreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, streamNotifySub_[signalIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherGraphPipeline::SubWaitMain()
{
    u32 subStreamNum = intraRankSize_;
    for (u32 streamIndex = 0; streamIndex < subStreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            subStream_[streamIndex], dispatcher_, streamNotifySub_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherGraphPipeline::RunAsync()
{
    HCCL_INFO("[AllGatherGraphPipeline][RunAsync]AllGatherRingMesh starts groupRankId[%u]", userRank_);
    // inter ring algo
    u32 prevInterRankId = (interRankId_ + interRankSize_ - 1) % interRankSize_;
    u32 nextInterRankId = (interRankId_ + 1) % interRankSize_;
    LINK prevInterLink = interLinks_[prevInterRankId];
    CHK_SMART_PTR_NULL(prevInterLink);
    LINK nextInterLink = interLinks_[nextInterRankId];
    CHK_SMART_PTR_NULL(nextInterLink);

    // intra fullmesh algo
    // intra 使用全部连接，不再映射

    u32 unitSize = SIZE_TABLE[opInfo_->dataType];
    u64 memSliceSize = memSliceCount_ * unitSize;

    // step 0 前置操作：所有卡本地数据从userIn-->userOut
    DeviceMem locSrc = DeviceMem::create(usrInMemAddr_, memSliceSize);
    u64 localOffsetByte = memSliceCount_ * userRank_ * unitSize;
    DeviceMem locDst = DeviceMem::create(static_cast<u8*>(dmaMem_[1].ptr()) + localOffsetByte, memSliceSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDst, locSrc, stream_));

    for (u32 step = 0; step < interRankSize_; step++) {
        // 主从流同步
        CHK_RET(MainRecordSub());
        CHK_RET(SubWaitMain());

        u64 serverRankOffset = intraRankId_ + (interRankId_ + interRankSize_ - step) % interRankSize_ * intraRankSize_;
        u64 serverOffsetByte = memSliceCount_ * serverRankOffset * unitSize;
        u64 readRemoteOffset
            = intraRankId_
              + (prevInterRankId + interRankSize_ - step) % interRankSize_ * intraRankSize_; // server间前通信rank偏移
        u64 readRemoteOffsetByte = memSliceCount_ * readRemoteOffset * unitSize;

        if (step < interRankSize_ - 1) {
            CHK_RET(prevInterLink->TxAck(subStream_[0])); // AckRecord
            CHK_RET(nextInterLink->RxAck(subStream_[0])); // AckWait

            CHK_RET(nextInterLink->TxAsync(
                UserMemType::OUTPUT_MEM, serverOffsetByte, static_cast<u8*>(dmaMem_[1].ptr()) + serverOffsetByte,
                memSliceSize, subStream_[0]));
            HCCL_DEBUG(
                "[AllGatherGraphPipeline][RunAsync] local rank[%u] localOffset[%llu] tx with remoteRank[%u], "
                "remoteOffset[%llu] with slice[%llu]",
                userRank_, serverOffsetByte, nextInterRankId, serverOffsetByte, memSliceSize);

            CHK_RET(prevInterLink->RxAsync(
                UserMemType::OUTPUT_MEM, readRemoteOffsetByte,
                static_cast<u8*>(dmaMem_[1].ptr()) + readRemoteOffsetByte, memSliceSize,
                subStream_[0])); // wait
            HCCL_DEBUG(
                "[AllGatherGraphPipeline][RunAsync] read local rank[%u] localOffset[%llu] tx with remoteRank[%u], "
                "remoteOffset[%llu] with slice[%llu]",
                userRank_, readRemoteOffsetByte, prevInterRankId, readRemoteOffsetByte, memSliceSize);

            CHK_RET(prevInterLink->PostFinAck(subStream_[0]));
            CHK_RET(nextInterLink->WaitFinAck(subStream_[0]));

            // inter的最后一步需要barrier确保数据发完
            if (step == interRankSize_ - STEP_OFFSET_TWO) {
                CHK_RET(ExecuteBarrier(prevInterLink, nextInterLink, subStream_[0]));
            }
        }

        HCCL_DEBUG("[AllGatherGraphPipeline][RunAsync]now step is %u, intraRankSize is %u", step, intraRankSize_);
        for (u32 i = 1; i < intraRankSize_; i++) {
            u32 remIntraRankId = (intraRankId_ + i) % intraRankSize_;
            CHK_RET(intraLinks_[remIntraRankId]->TxAck(subStream_[i])); // ack record
            CHK_RET(intraLinks_[remIntraRankId]->RxAck(subStream_[i])); // ack wait

            void* remDMAMemPtr = nullptr;
            CHK_RET(intraLinks_[remIntraRankId]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remDMAMemPtr));
            u64 remoteOffset
                = (interRankId_ - step + interRankSize_) % interRankSize_ * intraRankSize_ + remIntraRankId;
            u64 remoteOffsetByte = memSliceCount_ * remoteOffset * unitSize;
            void* dstAddr = static_cast<u8*>(usrOutMemAddr_) + remoteOffsetByte;

            DeviceMem src = DeviceMem::create(static_cast<u8*>(remDMAMemPtr) + remoteOffsetByte, memSliceSize);
            DeviceMem dst = DeviceMem::create(dstAddr, memSliceSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStream_[i]));

            CHK_RET(intraLinks_[remIntraRankId]->TxDataSignal(subStream_[i])); // data record
            CHK_RET(intraLinks_[remIntraRankId]->RxDataSignal(subStream_[i])); // data wait
        }

        CHK_RET(SubRecordMain());
        CHK_RET(MainWaitSub());
    }

    HCCL_INFO("[AllGatherGraphPipeline][RunAsync]AllGatherRingMesh finished groupRankId[%u]", userRank_);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_GRAPH_PIPELINE, AllGatherGraphPipeline);
} // namespace hccl
