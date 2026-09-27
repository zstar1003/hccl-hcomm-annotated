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

class AivAllReduceRdmaSmallGraph910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceRdmaSmallGraph910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, int32_t aivRdmaStep);

    template <typename T>
    __aicore__ inline void ReduceScatterForGraph(
        __gm__ T* inputGM, __gm__ T* cclGMSelf, __gm__ T* cclGMOther, __gm__ T* outputGM, uint64_t sliceCount,
        uint64_t avgLengthPerRank, uint64_t tailLength, int32_t tag);

    template <typename T>
    __aicore__ inline void
    AllReduceForGraph(__gm__ T* inputGM, __gm__ T* cclGMSelf, __gm__ T* outputGM, int64_t len, int32_t tag);

    template <typename T>
    __aicore__ inline void AllGatherForGraph(
        __gm__ T* inputGM, __gm__ T* cclGMSelf, __gm__ T* cclGMOther, __gm__ T* outputGM, uint64_t sliceCount,
        uint64_t avgLengthPerRank, uint64_t tailLength, int32_t tag);
};

template <typename T>
__aicore__ inline void AivAllReduceRdmaSmallGraph910B::ReduceScatterForGraph(
    __gm__ T* inputGM, __gm__ T* cclGMSelf, __gm__ T* cclGMOther, __gm__ T* outputGM, uint64_t sliceCount,
    uint64_t avgLengthPerRank, uint64_t tailLength, int32_t tag)
{
    // reduce scatter，数据从input输入，inputMem+0作为buffer，结果放在inputMem+2M处

    if (blockIdx_ == rank_) {
        int64_t curCount = CalActualCount(rank_, rankSize_, avgLengthPerRank, tailLength);

        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM, curCount);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, curCount);
        GlobalTensor<T> cclGTSelf;
        cclGTSelf.SetGlobalBuffer(cclGMSelf, curCount);

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[avgLengthPerRank * blockIdx_], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(cclGTSelf[avgLengthPerRank * blockIdx_], localOut, curCount);
        DataCopyUB2GM(outputGT, localOut, curCount);
        inOutQue.FreeTensor(localOut);

        // 卡内同步，本卡目的分片已经在output中
        pipe_barrier(PIPE_ALL);
        Record1vN(tag, CommPattern::intraRank);
    } else {
        int64_t curCount = CalActualCount(blockIdx_, rankSize_, avgLengthPerRank, tailLength);

        GlobalTensor<T> cclGTOther;
        cclGTOther.SetGlobalBuffer(cclGMOther, curCount);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, curCount);

        // 从input搬运到buffer
        CpGM2GM(cclGMSelf + avgLengthPerRank * blockIdx_, inputGM + avgLengthPerRank * blockIdx_, curCount);
        pipe_barrier(PIPE_ALL);
        Record(tag, blockIdx_, AivNotifyType::ACK); // 本卡该片数据已经可以被跨片读取

        // 对端数据就绪后先搬到自己的UB
        curCount = CalActualCount(rank_, rankSize_, avgLengthPerRank, tailLength);

        Wait(tag, blockIdx_, AivNotifyType::ACK);
        pipe_barrier(PIPE_ALL);
        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, cclGTOther[avgLengthPerRank * rank_], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();

        // 本端数据在output就绪后从UB中搬入
        WaitNv1(tag, rank_);
        pipe_barrier(PIPE_ALL);
        SetAtomicOp<T>(reduceOp_);
        DataCopyUB2GM(outputGT, localOut, curCount);
        SetAtomicNone();
        inOutQue.FreeTensor(localOut);
    }

    return;
}

template <typename T>
__aicore__ inline void AivAllReduceRdmaSmallGraph910B::AllReduceForGraph(
    __gm__ T* inputGM, __gm__ T* cclGMSelf, __gm__ T* outputGM, int64_t len, int32_t tag)
{
    // all reduce，仅适用于A+X单机跨aggregation场景，数据在buffer中，先本端数据拷贝到output中，再从对端拷贝到output中
    uint32_t peerRank = 1 - rank_;
    int64_t count = len;
    uint32_t flagBaseOffset = 0;
    uint32_t flagOffsetStart = flagBaseOffset;           //  起始同步
    uint32_t flagOffsetEnd = flagBaseOffset + FLAG_SIZE; // 末尾同步

    __gm__ T* cclGMPeer = (__gm__ T*)(GM_IN[peerRank]);

    if (blockIdx_ == 0) {
        // 本端数据已就绪
        SetSignalValue((__gm__ int32_t*)(GM_OUT[rank_] + flagOffsetStart), localSetTensor, tag);
        CpGM2GM(outputGM, cclGMSelf, count);

        // 起始同步，检查对端数据是否就绪
        WaitSignalValue((__gm__ int32_t*)(GM_OUT[peerRank] + flagOffsetStart), localCheckTensor, tag);
        pipe_barrier(PIPE_ALL);
        SetSignalValue((__gm__ int32_t*)(GM_OUT[peerRank] + flagOffsetStart), localSetTensor, 0);
        CpGM2GM(outputGM, cclGMPeer, count, true, 0);

        // 末尾同步
        pipe_barrier(PIPE_ALL);
        SetSignalValue((__gm__ int32_t*)(GM_OUT[rank_] + flagOffsetEnd), localSetTensor, tag);
        WaitSignalValue((__gm__ int32_t*)(GM_OUT[peerRank] + flagOffsetEnd), localCheckTensor, tag);
        pipe_barrier(PIPE_ALL);
        SetSignalValue((__gm__ int32_t*)(GM_OUT[peerRank] + flagOffsetEnd), localSetTensor, 0);
    }
    return;
}

template <typename T>
__aicore__ inline void AivAllReduceRdmaSmallGraph910B::AllGatherForGraph(
    __gm__ T* inputGM, __gm__ T* cclGMSelf, __gm__ T* cclGMOther, __gm__ T* outputGM, uint64_t sliceCount,
    uint64_t avgLengthPerRank, uint64_t tailLength, int32_t tag)
{
    if (blockIdx_ == rank_) {
        int64_t curCount = CalActualCount(rank_, rankSize_, avgLengthPerRank, tailLength);

        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM, curCount);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, curCount);
        GlobalTensor<T> cclGTSelf;
        cclGTSelf.SetGlobalBuffer(cclGMSelf, curCount);

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT, curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(cclGTSelf[avgLengthPerRank * blockIdx_], localOut, curCount);

        // 卡间同步
        pipe_barrier(PIPE_ALL);
        Record1vN(tag, CommPattern::interRank); // 本卡该片数据已经可以被跨片读取
        DataCopyUB2GM(outputGT[avgLengthPerRank * blockIdx_], localOut, curCount);
        inOutQue.FreeTensor(localOut);
    } else {
        int64_t curCount = CalActualCount(blockIdx_, rankSize_, avgLengthPerRank, tailLength);

        WaitNv1(tag, blockIdx_);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(outputGM + (blockIdx_ * avgLengthPerRank), cclGMOther + blockIdx_ * avgLengthPerRank, curCount);
        Record(tag, blockIdx_, AivNotifyType::DataSignal);
        // 检查本卡上是否有GetBlockIdx()号对端的读完标记
        pipe_barrier(PIPE_ALL);
        Wait(tag, blockIdx_, AivNotifyType::DataSignal);
        pipe_barrier(PIPE_ALL);
    }

    return;
}

template <typename T>
__aicore__ inline void
AivAllReduceRdmaSmallGraph910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, int32_t aivRdmaStep)
{
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_]);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[blockIdx_]);
    uint64_t avgLengthPerRank = CeilDiv(len, rankSize_);
    uint64_t sliceCount = CeilDiv(len, avgLengthPerRank);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerRank;

    switch (aivRdmaStep) {
        case 0:
            ReduceScatterForGraph(
                inputGM, cclGMSelf, cclGMOther, outputGM, sliceCount, avgLengthPerRank, tailLength, tag);
            break;
        case 1:
            AllReduceForGraph(inputGM, cclGMSelf, outputGM, len, tag);
            break;
        case 2:
            AllGatherForGraph(inputGM, cclGMSelf, cclGMOther, outputGM, sliceCount, avgLengthPerRank, tailLength, tag);
            break;
        default:
            break;
    }
    return;
}

template <typename T>
__aicore__ inline void aiv_all_reduce_910b_rdma_smalldata_graph(KERNEL_ARGS_DEF)
{
    AivAllReduceRdmaSmallGraph910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag, aivRdmaStep);
    op.TailCounter();
}
