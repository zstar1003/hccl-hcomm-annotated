/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_v_graph_pipeline.h"
#include "alg_template_register.h"

constexpr u32 STEP_OFFSET_TWO = 2;

namespace hccl {
AllGatherVGraphPipeline::AllGatherVGraphPipeline(const HcclDispatcher dispatcher) : AllGatherVPipeline(dispatcher) {}

HcclResult AllGatherVGraphPipeline::RunAsync()
{
    HCCL_INFO("[AllGatherVGraphPipeline][RunAsync]AllGatherRingMesh starts groupRankId[%u] ", userRank_);

    u32 unitSize = SIZE_TABLE[opInfo_->dataType];

    // step 0前置操作 : 所有卡本地数据从userIn-->userout
    DeviceMem locSrc = DeviceMem::create(usrInMemAddr_, memSliceCount_ * unitSize);
    u64 localOffset = userMemSlice_[userRank_].offset;
    DeviceMem locDst = DeviceMem::create(
        static_cast<void*>(static_cast<u8*>(dmaMem_[1].ptr()) + localOffset), memSliceCount_ * unitSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDst, locSrc, stream_));

    for (u32 step = 0; step < interRankSize_; step++) {
        CHK_RET(MainRecordSub());
        CHK_RET(SubWaitMain());
        CHK_RET(ExecInterServer(step));
        CHK_RET(ExecIntraServer(step));
        CHK_RET(SubRecordMain());
        CHK_RET(MainWaitSub());
    }

    HCCL_INFO("[AllGatherVGraphPipeline][RunAsync]AllGatherVGraphPipeline finished groupRankId[%u] ", userRank_);
    return HCCL_SUCCESS;
}

HcclResult AllGatherVGraphPipeline::ExecInterServer(u32 step)
{
    // inter ring algo
    u32 prevInterRankId = (interRankId_ - 1 + interRankSize_) % interRankSize_;
    u32 nextInterRankId = (interRankId_ + 1) % interRankSize_;
    LINK prevInterLink = interLinks_[prevInterRankId];
    LINK nextInterLink = interLinks_[nextInterRankId];
    u64 serverRankOffset = intraRankId_ + (interRankId_ + interRankSize_ - step) % interRankSize_ * intraRankSize_;
    u64 serverOffsetByte = userMemSlice_[serverRankOffset].offset;
    u64 readRemoteOffset
        = intraRankId_
          + (prevInterRankId + interRankSize_ - step) % interRankSize_ * intraRankSize_; // sever间前通信rank偏移
    u64 readRemoteOffsetByte = userMemSlice_[readRemoteOffset].offset;
    if (step < interRankSize_ - 1) {
        CHK_RET(prevInterLink->TxAck(subStream_[0])); // AckRecord
        CHK_RET(nextInterLink->RxAck(subStream_[0])); // AckWait
        // RdmaSend + Record 或 PCIE::Record
        CHK_RET(nextInterLink->TxAsync(
            UserMemType::OUTPUT_MEM, serverOffsetByte, static_cast<u8*>(dmaMem_[1].ptr()) + serverOffsetByte,
            userMemSlice_[serverRankOffset].size, subStream_[0]));
        HCCL_DEBUG(
            "[AllGatherVGraphPipeline][RunAsync] local rank[%u] localOffset[%llu] tx with remoteRank[%u],"
            "remoteOffset[%llu] with slice[%llu]",
            userRank_, serverOffsetByte, nextInterRankId, serverOffsetByte, userMemSlice_[serverRankOffset].size);
        // 对于RDMA RxAsync，内存属性入参无效 RDMA::Wait
        // 对于PCIE，需设置内存属性 PCIE::Read + Record
        CHK_RET(prevInterLink->RxAsync(
            UserMemType::OUTPUT_MEM, readRemoteOffsetByte, static_cast<u8*>(dmaMem_[1].ptr()) + readRemoteOffsetByte,
            userMemSlice_[readRemoteOffset].size, subStream_[0])); // wait
        HCCL_DEBUG(
            "[AllGatherVGraphPipeline][RunAsync]read local rank[%u] localOffset[%llu] tx with remoteRank[%u],"
            "remoteOffset[%llu] with slice[%llu]",
            userRank_, readRemoteOffsetByte, readRemoteOffset, readRemoteOffsetByte,
            userMemSlice_[readRemoteOffset].size);
        CHK_RET(prevInterLink->PostFinAck(subStream_[0]));
        CHK_RET(nextInterLink->WaitFinAck(subStream_[0]));
        // inter的最后一步需要barrier确保数据发完
        if (step == interRankSize_ - STEP_OFFSET_TWO) {
            CHK_RET(ExecuteBarrier(prevInterLink, nextInterLink, subStream_[0]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherVGraphPipeline::ExecIntraServer(u32 step)
{
    for (u32 i = 1; i < intraRankSize_; i++) {
        u32 remIntraRankId = (intraRankId_ + i) % intraRankSize_;
        CHK_RET(intraLinks_[remIntraRankId]->TxAck(subStream_[i])); // ack record
        CHK_RET(intraLinks_[remIntraRankId]->RxAck(subStream_[i])); // ack wait

        void* remDMAMemPtr = nullptr;
        CHK_RET(intraLinks_[remIntraRankId]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remDMAMemPtr));
        u64 remoteRankOffset
            = (interRankId_ - step + interRankSize_) % interRankSize_ * intraRankSize_ + remIntraRankId;
        u64 remoteOffsetByte = userMemSlice_[remoteRankOffset].offset;
        void* dstAddr = static_cast<u8*>(usrOutMemAddr_) + remoteOffsetByte;

        DeviceMem src = DeviceMem::create(
            static_cast<u8*>(remDMAMemPtr) + remoteOffsetByte, userMemSlice_[remoteRankOffset].size);
        DeviceMem dst = DeviceMem::create(dstAddr, userMemSlice_[remoteRankOffset].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStream_[i]));

        CHK_RET(intraLinks_[remIntraRankId]->TxDataSignal(subStream_[i])); // data record
        CHK_RET(intraLinks_[remIntraRankId]->RxDataSignal(subStream_[i])); // data wait
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_V_GRAPH_PIPELINE, AllGatherVGraphPipeline);
} // namespace hccl
