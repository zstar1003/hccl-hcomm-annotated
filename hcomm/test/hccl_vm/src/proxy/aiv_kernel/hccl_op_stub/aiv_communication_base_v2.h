/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_COMMUNICATION_BASE_V2_H
#define AIV_COMMUNICATION_BASE_V2_H

#include "sync_interface.h"
#include "sim_common_defs.h"
#include "aiv_defines.h"

class AivCommBase {
public:
    __aicore__ inline AivCommBase() {}

    __aicore__ inline void Init(
        GM_ADDR buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank, uint32_t rankSize,
        uint64_t xRankSize, uint64_t yRankSize, uint64_t zRankSize, uint64_t len, uint32_t dataType, uint32_t reduceOp,
        uint32_t root, uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum,
        uint64_t inputRepeatStride, uint64_t outputRepeatStride, GM_ADDR headCountMem, GM_ADDR tailCountMem,
        GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter, uint32_t numBlocks, bool useDoubleBuffer,
        bool pingpong = false)
    {
        rank_ = rank;
        sendRecvRemoteRank_ = sendRecvRemoteRank;
        root_ = root;
        rankSize_ = rankSize;
        xRankSize_ = xRankSize;
        yRankSize_ = yRankSize;
        zRankSize_ = zRankSize;
        reduceOp_ = reduceOp;
        len_ = len;
        input_ = input;
        output_ = output;
        dataType_ = dataType;
        useDoubleBuffer_ = useDoubleBuffer;
        numBlocks_ = numBlocks;

        inputSliceStride_ = inputSliceStride;
        outputSliceStride_ = outputSliceStride;
        repeatNum_ = repeatNum;
        inputRepeatStride_ = inputRepeatStride;
        outputRepeatStride_ = outputRepeatStride;

        localOffset = (rankSize_ * NUM_BLOCKS_FOUR_PER_RANK_A3 * FLAG_BUF_NUM) * FLAG_SIZE;
        multiOffset = MAX_NUM_BLOCKS * DOUBLE * FLAG_SIZE + localOffset;
        pingpongOffset = multiOffset + DOUBLE * DOUBLE * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * DOUBLE;
        countOffset = DOUBLE * pingpongOffset;
        seperateOffset = countOffset + NUM_BLOCKS_FOUR_PER_RANK_A3 * rankSize_ * FLAG_SIZE;

        pipe.InitBuffer(localFlagBuf, LOCAL_FLAG_BUF_LEN);
        localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_ONE_OFFSET);
        localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_TWO_OFFSET);
        localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_THREE_OFFSET);
        localGetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FOUR_OFFSET);
        localTagTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FIVE_OFFSET);
        pipe.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);

        uint64_t chunkSize = UB_MAX_DATA_SIZE / TILING_NUM / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
        pipe.InitBuffer(inQueueX, 1, chunkSize);
        pipe.InitBuffer(inQueueY, 1, chunkSize);
        pipe.InitBuffer(outQueueZ, 1, chunkSize);

        GetTag(buffIn);
        InitBuffArray(buffIn, pingpong);
    }

    __aicore__ inline void Init(GM_ADDR hiddenInput, GM_ADDR input, GM_ADDR output, bool pingpong = false)
    {
        // SuperKernel 当前不支持
        HCCL_VM_ERROR("Not support yet!");
    }

    __aicore__ inline void InitBuffArray(GM_ADDR buffIn, bool pingpong = false)
    {
        GlobalTensor<uint64_t> ipcBufferGlobal;
        ipcBufferGlobal.SetGlobalBuffer((__gm__ uint64_t*)(buffIn));
        if (!pingpong) {
            for (int i = 0; i < rankSize_; i++) {
                GM_IN[i] = (GM_ADDR)ipcBufferGlobal.GetValue(i);
                GM_OUT[i]
                    = (GM_ADDR)ipcBufferGlobal.GetValue(BUFFER_OUT_ADDR_OFFSET / sizeof(uint64_t) + i) + FLAG1_OFFSET;
                gmOutOffset = FLAG1_OFFSET;
            }
        } else {
            for (int i = 0; i < rankSize_; i++) {
                GM_IN[i] = tag_ % PING_PONG == 0 ?
                               (GM_ADDR)ipcBufferGlobal.GetValue(BUFFER_OUT_ADDR_OFFSET / sizeof(uint64_t) + i)
                                   + GM_OUT_PING_OFFSET :
                               (GM_ADDR)ipcBufferGlobal.GetValue(BUFFER_OUT_ADDR_OFFSET / sizeof(uint64_t) + i)
                                   + GM_OUT_PONG_OFFSET;
                GM_OUT[i] = tag_ % PING_PONG == 0 ?
                                (GM_ADDR)ipcBufferGlobal.GetValue(BUFFER_OUT_ADDR_OFFSET / sizeof(uint64_t) + i)
                                    + FLAG1_OFFSET :
                                (GM_ADDR)ipcBufferGlobal.GetValue(BUFFER_OUT_ADDR_OFFSET / sizeof(uint64_t) + i)
                                    + FLAG2_OFFSET;
                gmOutOffset = tag_ % PING_PONG == 0 ? FLAG1_OFFSET : FLAG2_OFFSET;
            }
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void GetTag(GM_ADDR buffIn)
    {
        (void)buffIn;
        tag_ = AivCommInfoLayout::FIXED_TAG;
    }

    __aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);

    template <typename T>
    __aicore__ inline void SetAtomicOp(uint32_t atomicOp);

    template <typename T>
    __aicore__ inline void
    DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const uint32_t calCount);

    template <typename T>
    __aicore__ inline void
    DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const uint32_t calCount);

    template <typename T>
    __aicore__ inline void CpGM2GM(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, uint32_t atomicOp);

    template <typename T>
    __aicore__ inline void CpGM2GM(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count);

    template <typename T>
    __aicore__ inline void Reduce64(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, uint32_t reduceOp);

    __aicore__ inline void BarrierForFirstOPInner(uint32_t barrierStage);

    __aicore__ inline void BarrierAll();

    __aicore__ inline void SendRecvBarrierAll(uint32_t myRank, uint32_t remoteRank);

    __aicore__ inline bool IsFirstOP(int32_t sliceId);

    __aicore__ inline void ClearGM();

    __aicore__ inline void BarrierForFirstOP();

    __aicore__ inline void SendRecvBarrierForFirstOP(uint32_t myRank, uint32_t remoteRank);

    __aicore__ inline void WaitFlag(uint32_t targetRank, uint64_t flag_offset, int32_t curTag);

    __aicore__ inline void Record(uint32_t targetRank, uint64_t flag_offset, int32_t curTag);

    __aicore__ inline void Barrier(uint32_t step);

    __aicore__ inline void ClearFlag();

    __aicore__ inline void ClearSyncBuf();

    GM_ADDR GM_IN[MAX_RANK_SIZE];
    GM_ADDR GM_OUT[MAX_RANK_SIZE];
    uint32_t rank_;
    uint32_t sendRecvRemoteRank_;
    uint32_t root_;
    uint32_t rankSize_;
    uint64_t xRankSize_;
    uint64_t yRankSize_;
    uint64_t zRankSize_;
    uint32_t reduceOp_;
    uint32_t dataType_;
    uint32_t unitSize_;

    uint64_t input_;
    uint64_t output_;
    uint64_t cclBufferSize_;
    uint64_t gmOutOffset;

    uint64_t len_;
    uint32_t tag_;
    uint32_t curTag_{0};
    int32_t numBlocks_;
    uint32_t blockIdx_ = GetBlockIdx(); // 在构造函数中初始化，以免漏初始化

    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t repeatNum_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;

    bool useDoubleBuffer_;

    TPipe pipe;
    TBuf<> localFlagBuf;
    LocalTensor<int32_t> localSetTensor;
    LocalTensor<int32_t> localCheckTensor;
    LocalTensor<int32_t> localCheckGETensor;
    LocalTensor<int32_t> localGetTensor;
    LocalTensor<int32_t> localTagTensor;
    GlobalTensor<int32_t> d2hGlobal;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inQueueX;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inQueueY;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> outQueueZ;

    uint32_t localOffset;
    uint32_t multiOffset;
    uint32_t pingpongOffset;
    uint32_t countOffset;
    uint32_t seperateOffset;
};

__aicore__ inline void AivCommBase::Record(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
{
    const uint64_t commInfoOffset = gmOutOffset + flag_offset * AivCommInfoLayout::SYNC_CELL_BYTES;
    send_flag(targetRank, commInfoOffset, curTag);
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void AivCommBase::ClearSyncBuf()
{
    // SuperKernel 当前不支持
    HCCL_VM_ERROR("Not support yet!");
}

__aicore__ inline void AivCommBase::Barrier(uint32_t step)
{
    (void)step;
    // SuperKernel 当前不支持
    HCCL_VM_ERROR("Not support yet!");
}

__aicore__ inline void AivCommBase::ClearFlag()
{
    // SuperKernel 当前不支持
    HCCL_VM_ERROR("Not support yet!");
}

__aicore__ inline void AivCommBase::WaitFlag(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
{
    const uint64_t commInfoOffset = gmOutOffset + flag_offset * AivCommInfoLayout::SYNC_CELL_BYTES;
    recv_flag(targetRank, commInfoOffset, curTag);
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline bool AivCommBase::IsFirstOP(int32_t sliceId) { return sliceId == 1 && tag_ == 1; }

__aicore__ inline void AivCommBase::ClearGM()
{
    // 无论pingpong，清零区域始终从FLAG1_OFFSET开始
    // GM_OUT[rank_] = base + gmOutOffset，需减去gmOutOffset再加FLAG1_OFFSET
    GM_ADDR flagBase = GM_OUT[rank_] - gmOutOffset + FLAG1_OFFSET;
    uint32_t emptyOffset = AIV_FLAG_EMPTY_OFFSET - FLAG1_OFFSET;
    uint32_t blockCount = (BASE_FLAG_OFFSET - FLAG1_OFFSET) / numBlocks_;
    uint32_t blockOffset = blockCount * blockIdx_;
    CpGM2GM(flagBase + blockOffset, flagBase + blockOffset + emptyOffset, blockCount);
}

__aicore__ inline void AivCommBase::BarrierForFirstOPInner(uint32_t barrierStage)
{
    uint32_t perCoreRankNum = rankSize_ / numBlocks_;
    uint32_t remainRankNum = rankSize_ % numBlocks_;
    uint32_t curCoreRankNum = blockIdx_ < remainRankNum ? perCoreRankNum + 1 : perCoreRankNum;
    uint32_t startRank
        = blockIdx_ < remainRankNum ? (perCoreRankNum + 1) * blockIdx_ : perCoreRankNum * blockIdx_ + remainRankNum;
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        uint64_t flagOffset = BASE_FLAG_OFFSET - gmOutOffset + rank * FLAG_SIZE + barrierStage * rankSize_ * FLAG_SIZE;
        Record(rank_, flagOffset / FLAG_SIZE, DOUBLE);
    }
    PipeBarrier<PIPE_ALL>();
    uint64_t flagOffset = BASE_FLAG_OFFSET - gmOutOffset + rank_ * FLAG_SIZE + barrierStage * rankSize_ * FLAG_SIZE;
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        WaitFlag(rank, flagOffset / FLAG_SIZE, DOUBLE);
        Record(rank, flagOffset / FLAG_SIZE, 0);
    }
}

__aicore__ inline void AivCommBase::BarrierForFirstOP()
{
    BarrierForFirstOPInner(0);
    SyncAll<true>();
    ClearGM();
    SyncAll<true>();
    BarrierForFirstOPInner(1);
    SyncAll<true>();
}

// 为sendRecv单独设计
__aicore__ inline void AivCommBase::SendRecvBarrierForFirstOP(uint32_t myRank, uint32_t remoteRank)
{
    // 清零标记区
    ClearGM();
    SyncAll<true>();

    if (blockIdx_ == 0) {
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            if (i == myRank || i == remoteRank) {
                uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset + i * FLAG_SIZE;
                Record(rank_, flag_offset / FLAG_SIZE, DOUBLE);
            }
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            if (i == myRank || i == remoteRank) {
                uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset + rank_ * FLAG_SIZE;
                WaitFlag(i, flag_offset / FLAG_SIZE, DOUBLE);
            }
        }
    }

    SyncAll<true>();
}

__aicore__ inline void AivCommBase::BarrierAll()
{
    SyncAll<true>();

    // 每个核分配多个rank
    uint32_t perCoreRankNum = rankSize_ / numBlocks_;
    uint32_t remainRankNum = rankSize_ % numBlocks_;
    uint32_t curCoreRankNum = blockIdx_ < remainRankNum ? perCoreRankNum + 1 : perCoreRankNum;
    uint32_t startRank
        = blockIdx_ < remainRankNum ? (perCoreRankNum + 1) * blockIdx_ : perCoreRankNum * blockIdx_ + remainRankNum;
    uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset + rank_ * FLAG_SIZE;
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        Record(rank, flag_offset / FLAG_SIZE, 1);
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset + rank * FLAG_SIZE;
        WaitFlag(rank_, flag_offset / FLAG_SIZE, 1);
        Record(rank_, flag_offset / FLAG_SIZE, 0);
    }
}

// 为sendRecv单独设计
__aicore__ inline void AivCommBase::SendRecvBarrierAll(uint32_t myRank, uint32_t remoteRank)
{
    SyncAll<true>();
    if (blockIdx_ == 0) {
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            if (i == myRank || i == remoteRank) {
                uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset + rank_ * FLAG_SIZE;
                Record(i, flag_offset / FLAG_SIZE, 1);
            }
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            if (i == myRank || i == remoteRank) {
                uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset + i * FLAG_SIZE;
                WaitFlag(rank_, flag_offset / FLAG_SIZE, 1);
                Record(rank_, flag_offset / FLAG_SIZE, 0);
            }
        }
    }
    SyncAll<true>(); // 确保非0核在同步完成前不会走进其他算子流程
}

__aicore__ inline uint64_t AivCommBase::CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline void AivCommBase::SetAtomicOp(uint32_t atomicOp)
{
    switch (atomicOp) {
        case HcclReduceOp::HCCL_REDUCE_SUM:
            SetAtomicAdd<T>();
            break;
        case HcclReduceOp::HCCL_REDUCE_MAX:
            SetAtomicMax<T>();
            break;
        case HcclReduceOp::HCCL_REDUCE_MIN:
            SetAtomicMin<T>();
            break;
        default:
            SetAtomicNone();
            break;
    }
}

template <typename T>
__aicore__ inline void
AivCommBase::DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const uint32_t calCount)
{
    DataCopy<T>(dstLocal, srcGlobal, calCount);
}

template <typename T>
__aicore__ inline void
AivCommBase::DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const uint32_t calCount)
{
    DataCopy<T>(dstGlobal, srcLocal, calCount);
}

template <typename T>
__aicore__ inline void AivCommBase::CpGM2GM(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count)
{
    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);
    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }
    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;
    }
    return;
}

template <typename T>
__aicore__ inline void AivCommBase::CpGM2GM(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, uint32_t atomicOp)
{
    if constexpr (Std::is_same<T, int64_t>::value) {
        Reduce64(outputGM, inputGM, count, atomicOp);
        return;
    } else {
        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, count);

        SetAtomicOp<T>(atomicOp);

        uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
        if (useDoubleBuffer_) {
            maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
        }
        uint64_t curOffset = 0;
        while (count > 0) {
            uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

            LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
            DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
            inOutQue.EnQue(localIn);
            LocalTensor<T> localOut = inOutQue.DeQue<T>();
            DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
            inOutQue.FreeTensor(localOut);

            count -= curCount;
            curOffset += curCount;
        }

        SetAtomicNone();
        return;
    }
}

template <typename T>
__aicore__ inline void AivCommBase::Reduce64(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count, uint32_t reduceOp)
{
    (void)outputGM;
    (void)inputGM;
    (void)count;
    (void)reduceOp;
    // Reduce64 当前不支持
    HCCL_VM_ERROR("Not support yet!");
}

// 910B支持的Atomic数据类型
#define AIV_ATOMIC_DATA_TYPE_DEF(func) \
    func(float);                       \
    func(half);                        \
    func(int16_t);                     \
    func(int32_t);                     \
    func(int8_t);                      \
    func(bfloat16_t);                  \
    func(int64_t)

// 910B支持的DataCopy数据类型
#define AIV_COPY_DATA_TYPE_DEF(func) \
    func(half);                      \
    func(int16_t);                   \
    func(uint16_t);                  \
    func(float);                     \
    func(int32_t);                   \
    func(uint32_t);                  \
    func(int8_t);                    \
    func(uint8_t);                   \
    func(bfloat16_t);                \
    func(uint64_t);                  \
    func(int64_t);                   \
    func(fp8_e4m3fn_t);              \
    func(fp8_e5m2_t);                \
    func(fp8_e8m0_t);                \
    func(hifloat8_t)

#endif /* AIV_COMMUNICATION_BASE_V2_H */
