/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base.h"

using namespace AscendC;

class AivAll2All910BDirectFullMesh : public AivCommBase {
public:
    __aicore__ inline AivAll2All910BDirectFullMesh() {}

    template <typename T>
    __aicore__ inline void
    Process(GM_ADDR input, GM_ADDR output, int32_t tag, int32_t serverNum, uint64_t rmaInfo, uint64_t len);
};

template <typename T>
__aicore__ inline void AivAll2All910BDirectFullMesh::Process(
    GM_ADDR input, GM_ADDR output, int32_t tag, int32_t serverNum, uint64_t rmaInfo, uint64_t len)
{
    // 内存准备
    if (blockIdx_ >= serverNum) {
        return;
    }
    uint32_t rankPerSever = rankSize_ / serverNum;
    uint64_t pingpongOffset = 0;
    int64_t flagOffsetBase = BASE_FLAG_OFFSET * AIV_ALL_TO_ALL_910B_DIRECT_FULLMESH;
    uint64_t ubDataBaseOffset = 2 * BUFFER_AREA; // 数据开始存储的偏移
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    uint64_t flagOffset = 2 * rank_ * FLAG_SIZE;
    uint64_t ranksizeOffset = 2 * rankSize_ * FLAG_SIZE;

    __gm__ T* cclGMInSelf = (__gm__ T*)(GM_IN_RDMA[rank_] + ubDataBaseOffset);

    for (uint32_t i = 0; i < rankPerSever; i++) {
        // 发数据和通知对端数据到达
        uint32_t targetRank = (rank_ + blockIdx_ + i * serverNum) % rankSize_;
        uint64_t srcOffset = targetRank * len; // 要发送的数据偏移是srcOffset;
        uint64_t dstOffset = rank_ * len;      // 第rank号卡会发数据到对应的CCL的dstOffset位置上;
        if (targetRank == rank_) {
            // 把数据从usrin搬运到对端的usrout
            CpGM2GM(outputGM + dstOffset, inputGM + srcOffset, len);
            PipeBarrier<PIPE_ALL>();
            continue;
        }
        __gm__ T* cclGMOutOther = (__gm__ T*)(GM_OUT_RDMA[targetRank] + ubDataBaseOffset);

        // 1. 先把数据搬到自己的cclin
        CpGM2GM(cclGMInSelf + srcOffset, inputGM + srcOffset, len);
        PipeBarrier<PIPE_ALL>();
        // 2. 再把数据从cclin搬到对端的cclOut
        uint64_t srcRdmaAddr = (uint64_t)(cclGMInSelf + srcOffset);
        uint64_t dstRdmaAddr = (uint64_t)(cclGMOutOther + dstOffset);
        AIVRDMAPostSend(
            (GM_ADDR)srcRdmaAddr, (GM_ADDR)dstRdmaAddr, targetRank, len * sizeof(T), (__gm__ HcclRMAInfo*)rmaInfo,
            false, true);
        PipeBarrier<PIPE_ALL>();
        // 3. 置flag，告知对端数据已发送
        uint64_t localFlagOffset = 2 * targetRank * FLAG_SIZE;
        __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT_RDMA[rank_] + pingpongOffset + flagOffsetBase
                                                       + localFlagOffset + FLAG_SIZE); // flag标志位
        SetSignalValue(ctrlFlagGM, localSetTensor, tag);
        PipeBarrier<PIPE_ALL>();
        AIVRDMAPostSend(
            (GM_ADDR)((uint64_t)ctrlFlagGM),
            (GM_ADDR)((uint64_t)(GM_OUT_RDMA[targetRank] + pingpongOffset + flagOffsetBase + flagOffset)), targetRank,
            UB_FLAG_PAD_COUNT, (__gm__ HcclRMAInfo*)rmaInfo, true, true);
    }

    __gm__ T* cclGMOutSelf = (__gm__ T*)(GM_OUT_RDMA[rank_] + ubDataBaseOffset);
    for (uint32_t i = 0; i < rankPerSever; i++) {
        // 等待数据到达
        uint32_t sourceRank
            = (rank_ + rankSize_ - (blockIdx_ % serverNum + i * serverNum)) % rankSize_; // 从哪个卡收数据
        if (sourceRank == rank_) {
            continue;
        }
        uint64_t srcFlagOffset = 2 * sourceRank * FLAG_SIZE;
        __gm__ int32_t* ctrlFlagGM
            = (__gm__ int32_t*)(GM_OUT_RDMA[rank_] + pingpongOffset + flagOffsetBase + srcFlagOffset);
        WaitSignalGEValue(ctrlFlagGM, localCheckTensor, tag);
        PipeBarrier<PIPE_ALL>();
        uint64_t dataOffset = sourceRank * len;
        CpGM2GM(outputGM + dataOffset, cclGMOutSelf + dataOffset, len); // 将数据从cclout搬到自己的output
        PipeBarrier<PIPE_ALL>();
    }

    for (uint32_t i = 0; i < rankPerSever; i++) {
        // 接收方通知发送方：已收到数据
        uint32_t sourceRank
            = (rank_ + rankSize_ - (blockIdx_ % serverNum + i * serverNum)) % rankSize_; // 从哪个卡收数据
        if (sourceRank == rank_) {
            continue;
        }
        // 置flag，告知对端数据已接收
        uint64_t localFlagOffset = 2 * sourceRank * FLAG_SIZE;
        __gm__ int32_t* ctrlPostFlagGM
            = (__gm__ int32_t*)(GM_OUT_RDMA[rank_] + pingpongOffset + flagOffsetBase + ranksizeOffset + localFlagOffset
                                + FLAG_SIZE); // flag标志位
        SetSignalValue(ctrlPostFlagGM, localSetTensor, tag);
        PipeBarrier<PIPE_ALL>();
        AIVRDMAPostSend(
            (GM_ADDR)((uint64_t)ctrlPostFlagGM),
            (GM_ADDR)((uint64_t)(GM_OUT_RDMA[sourceRank] + pingpongOffset + flagOffsetBase + ranksizeOffset
                                 + flagOffset)),
            sourceRank, UB_FLAG_PAD_COUNT, (__gm__ HcclRMAInfo*)rmaInfo, true, true);
    }

    for (uint32_t i = 0; i < rankPerSever; i++) {
        // 发送方等待接收方的通知，确保数据发完才结束
        uint32_t targetRank = (rank_ + blockIdx_ + i * serverNum) % rankSize_;
        if (targetRank == rank_) {
            continue;
        }
        uint64_t localFlagOffset = 2 * targetRank * FLAG_SIZE;
        __gm__ int32_t* ctrlPostFlagGM = (__gm__ int32_t*)(GM_OUT_RDMA[rank_] + pingpongOffset + flagOffsetBase
                                                           + ranksizeOffset + localFlagOffset);
        WaitSignalGEValue(ctrlPostFlagGM, localCheckTensor, tag);
    }
}

template <typename T>
__aicore__ inline void aiv_all2All_910b_direct_fullmesh(KERNEL_ARGS_DEF)
{
    AivAll2All910BDirectFullMesh op;
    op.InitForRDMA(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process<T>(input, output, tag, serverNum, rmaInfo, len);
    op.TailCounter();
}
