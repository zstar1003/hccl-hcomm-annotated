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

class AivAllReduceDeterMid910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceDeterMid910B() {}

    template <typename T>
    __aicore__ inline void CPGM2GMAccordingFlag(
        __gm__ T* cclGMSelf, __gm__ T* cclGMOther, uint64_t count, __gm__ int32_t* ctrlFlagGMSelf,
        __gm__ int32_t* ctrlFlagGMOther, int32_t tag, uint64_t ff = 0);

    __aicore__ inline int64_t GetDeterministicRankOffset(int64_t x);

    __aicore__ inline void EndSync(int32_t tag);

    template <typename T>
    __aicore__ inline void
    SumByPairs(__gm__ T* cclGMSelf, int64_t x, int64_t count, int32_t tag, int64_t flagOffsetBase);

    template <typename T>
    __aicore__ inline void ReduceWithFlagWrap(
        __gm__ T* cclGMSelf, __gm__ T* cclGMOther, uint64_t count, int32_t tag, __gm__ int32_t* flagCntDoneBase,
        __gm__ int32_t* flagCntSelf, __gm__ int32_t* flagCntDoneSelf);

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize);

    template <typename T>
    __aicore__ inline void ProcessProxy(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize);

    template <typename T>
    __aicore__ inline void
    ProcessSingleRanksizeCore(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize);
};

__aicore__ inline void AivAllReduceDeterMid910B::EndSync(int32_t tag)
{
    uint32_t targetRank = blockIdx_ % rankSize_;

    int64_t flagOffsetBasic = separateOffset + BASE_FLAG_OFFSET * AIV_ALL_REDUCE_DETER_910B_MIDDATA;
    uint32_t flagOffset = (((tag % 2 == 0) ? 3 : 9) * rankSize_ * FLAG_SIZE) + flagOffsetBasic;

    if (blockIdx_ < rankSize_) {
        if (targetRank != rank_) {
            PipeBarrier<PIPE_ALL>();
            SetSignalValue((__gm__ int32_t*)(GM_OUT[targetRank] + flagOffset + rank_ * FLAG_SIZE), localSetTensor, tag);
            WaitSignalValue(
                (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset + targetRank * FLAG_SIZE), localCheckTensor, tag);
            PipeBarrier<PIPE_ALL>();
            SetSignalValue((__gm__ int32_t*)(GM_OUT[rank_] + flagOffset + targetRank * FLAG_SIZE), localSetTensor, 0);
        }
    }
}

// 根据ctrlFlagGMOther的值，从cclGMOther拷贝到cclGMSelf
// 如果ff大于0，则更新ctrlFlagGMSelf
template <typename T>
__aicore__ inline void AivAllReduceDeterMid910B::CPGM2GMAccordingFlag(
    __gm__ T* cclGMSelf, __gm__ T* cclGMOther, uint64_t count, __gm__ int32_t* ctrlFlagGMSelf,
    __gm__ int32_t* ctrlFlagGMOther, int32_t tag, uint64_t ff)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);
    uint64_t maxBatchCount = CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE);

    while (true) {
        if (processedBatchCount >= maxBatchCount) {
            break;
        }

        LocalTensor<int32_t> localFlagX = flagInQue.AllocTensor<int32_t>();

#ifndef OPEN_HCCL_TEST
        uint64_t localFlagValueX = GetSignalValue(ctrlFlagGMOther, localFlagX);
#else
        uint64_t localFlagValueX = GetSignalValueWithExpected(
            ctrlFlagGMOther, localFlagX, CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE) + tag);
#endif

        flagInQue.FreeTensor(localFlagX);

        if (localFlagValueX <= tag) {
            continue;
        }

        uint64_t preparedBatchCount = localFlagValueX - tag;
        if (processedBatchCount >= preparedBatchCount) {
            continue;
        }

        uint64_t curSize = (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;
        if (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > avgSizePerSlice) {
            curSize = avgSizePerSlice - processedBatchCount * UB_DB_DATA_BATCH_SIZE;
        }

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

        //  搬运数据
        uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
        CpGM2GM(cclGMSelf + curProcessedOffset, cclGMOther + curProcessedOffset, curSize / sizeof(T));

        // 设置已经搬运的数据量
        processedBatchCount = preparedBatchCount;
        if (ff > 0 && (processedBatchCount % ff == 0 || processedBatchCount >= maxBatchCount)) {
            SyncFunc<HardEvent::MTE3_S>();
            SetSignalValue(ctrlFlagGMSelf, localSetTensor, processedBatchCount + tag);
        }
    }
}

__aicore__ inline int64_t AivAllReduceDeterMid910B::GetDeterministicRankOffset(int64_t x)
{
    int64_t tmp = 1;
    while (!(x & 1)) {
        x >>= 1;
        tmp <<= 1;
    }
    return tmp;
}

template <typename T>
__aicore__ inline void
AivAllReduceDeterMid910B::SumByPairs(__gm__ T* cclGMSelf, int64_t x, int64_t count, int32_t tag, int64_t flagOffsetBase)
{
    int64_t multiple = GetDeterministicRankOffset(x);
    int64_t target = x - multiple;

    int64_t flagOffsetSelf = 0;
    int64_t flagOffsetTarget = 0;
    int64_t flagOffset2st = flagOffsetBase + (rankSize_ + x) * FLAG_SIZE;
    int64_t flagOffset3st = flagOffsetBase + (DOUBLE * rankSize_ + x) * FLAG_SIZE;
    if (x & 1) {
        if (target == 0) {
            flagOffsetTarget = flagOffsetBase + (DOUBLE * rankSize_ + target) * FLAG_SIZE;
        } else {
            flagOffsetTarget = flagOffsetBase + (rankSize_ + target) * FLAG_SIZE;
        }
        flagOffsetSelf = flagOffset2st;
    } else {
        flagOffsetTarget = flagOffsetBase + (DOUBLE * rankSize_ + target + multiple / DOUBLE) * FLAG_SIZE;
        int64_t multipleTemp = multiple / DOUBLE;
        while (x + multipleTemp >= rankSize_) {
            multipleTemp /= DOUBLE;
        }
        if (multipleTemp <= 0) {
            flagOffsetSelf = flagOffset2st;
        } else {
            flagOffsetSelf = flagOffset3st + multipleTemp * FLAG_SIZE;
        }
    }
    __gm__ int32_t* flagCntDonePre = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffsetTarget);
    __gm__ int32_t* flagCntSelf = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffsetSelf);
    __gm__ int32_t* flagCntDoneSelf = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset3st);
    ReduceWithFlagWrap(
        cclGMSelf + target * count, cclGMSelf + x * count, count, tag, flagCntDonePre, flagCntSelf, flagCntDoneSelf);
}

template <typename T>
__aicore__ inline void AivAllReduceDeterMid910B::ReduceWithFlagWrap(
    __gm__ T* cclGMSelf, __gm__ T* cclGMOther, uint64_t count, int32_t tag, __gm__ int32_t* flagCntDoneBase,
    __gm__ int32_t* flagCntSelf, __gm__ int32_t* flagCntDoneSelf)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);
    uint64_t maxBatchCount = CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE);

    while (true) {
        if (processedBatchCount >= maxBatchCount) {
            break;
        }

        LocalTensor<int32_t> localFlag = flagInQue.AllocTensor<int32_t>();
        LocalTensor<int32_t> localFlagY = flagInQue.AllocTensor<int32_t>();

#ifndef OPEN_HCCL_TEST
        uint64_t localFlagValue = GetSignalValue(flagCntDoneBase, localFlag);
        uint64_t localFlagYValue = GetSignalValue(flagCntSelf, localFlagY);
#else
        uint64_t localFlagValue = GetSignalValueWithExpected(
            flagCntDoneBase, localFlag, CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE) + tag);
        uint64_t localFlagYValue = GetSignalValueWithExpected(
            flagCntSelf, localFlagY, CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE) + tag);
#endif

        flagInQue.FreeTensor(localFlag);
        flagInQue.FreeTensor(localFlagY);

        if (localFlagValue <= tag || localFlagYValue <= tag) {
            continue;
        }

        uint64_t preparedBatchCount = (localFlagValue <= localFlagYValue) ? localFlagValue : localFlagYValue;
        preparedBatchCount -= tag;
        if (processedBatchCount >= preparedBatchCount) {
            continue;
        }

        uint64_t curSize = (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;
        if (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > avgSizePerSlice) {
            curSize = avgSizePerSlice - processedBatchCount * UB_DB_DATA_BATCH_SIZE;
        }

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

        uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
        CpGM2GM(cclGMSelf + curProcessedOffset, cclGMOther + curProcessedOffset, curSize / sizeof(T), true, reduceOp_);

        processedBatchCount = preparedBatchCount;
        if (processedBatchCount % 8 == 0 || processedBatchCount >= maxBatchCount) {
            SyncFunc<HardEvent::MTE3_S>();
            SetSignalValue(flagCntDoneSelf, localSetTensor, processedBatchCount + tag);
        }
    }
}

template <typename T>
__aicore__ inline void
AivAllReduceDeterMid910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize)
{
    int64_t curCount = len;
    int64_t blockNumPerGroup = rankSize_;
    int64_t x = blockIdx_ % blockNumPerGroup;
    int64_t avgDataNum = curCount / rankSize_;
    int64_t lastDataNum = curCount - (rankSize_ - 1) * avgDataNum;
    int64_t flagOffsetBasic = separateOffset + BASE_FLAG_OFFSET * AIV_ALL_REDUCE_DETER_910B_MIDDATA;

    uint32_t flagOffsetBase = ((tag % 2 == 0) ? 0 : 6 * rankSize_ * FLAG_SIZE) + flagOffsetBasic;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[x] + dataOffset);
    __gm__ T* outputGM = (__gm__ T*)output;

    int64_t flagOffset1stCount = flagOffsetBase + (x)*FLAG_SIZE;
    int64_t flagOffset2stCount = flagOffsetBase + (rankSize_ + x) * FLAG_SIZE;
    int64_t flagOffset3stCount = flagOffsetBase + (DOUBLE * rankSize_ + x) * FLAG_SIZE;

    // 第一组 先从input拷贝到cclbuffer
    if (blockIdx_ < blockNumPerGroup) {
        int64_t dataCntSize = (x == rankSize_ - 1) ? lastDataNum : avgDataNum;
        CpGM2GMWithFlagWrap(
            cclGMSelf + x * avgDataNum, inputGM + x * avgDataNum, dataCntSize,
            (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset1stCount), 8, tag);
    }
    // 第二组 拷贝cclbuffer前半部分到cllbuffer后半部分
    else if (blockNumPerGroup <= blockIdx_ && blockIdx_ < DOUBLE * blockNumPerGroup) {
        __gm__ int32_t* flagCntDoneOtner = (__gm__ int32_t*)(GM_OUT[x] + flagOffsetBase + (rank_)*FLAG_SIZE);
        __gm__ int32_t* flagCntSelf = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2stCount);
        if (x == 0) {
            // 更新到3st区域
            flagCntSelf = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset3stCount);
        }

        int64_t dataCntSize = (rank_ == rankSize_ - 1) ? lastDataNum : avgDataNum;
        CPGM2GMAccordingFlag(
            cclGMSelf + curCount + x * dataCntSize, cclGMOther + rank_ * avgDataNum, dataCntSize, flagCntSelf,
            flagCntDoneOtner, tag, 8);
    }
    // 第三组 进行reduce操作
    else {
        if (x == 0) {
            return;
        }

        int64_t dataCntSize = (rank_ == rankSize_ - 1) ? lastDataNum : avgDataNum;
        if (rankSize_ >= DETERMINISTIC_RANKSIZE) {
            SumByPairs(cclGMSelf + curCount, x, dataCntSize, tag, flagOffsetBase);
        } else {
            __gm__ int32_t* flagCntDonePre = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset3stCount - FLAG_SIZE);
            __gm__ int32_t* flagCntSelf = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2stCount);
            __gm__ int32_t* flagCntDoneSelf = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset3stCount);

            ReduceWithFlagWrap(
                cclGMSelf + curCount, cclGMSelf + curCount + x * dataCntSize, dataCntSize, tag, flagCntDonePre,
                flagCntSelf, flagCntDoneSelf);
        }
    }

    // 第2组搬运cclbuffer到output
    if (blockNumPerGroup <= blockIdx_ && blockIdx_ < DOUBLE * blockNumPerGroup) {
        int32_t lastOpCore = rankSize_ - 1;
        if (rankSize_ >= DETERMINISTIC_RANKSIZE) {
            lastOpCore = rankSize_ > DETERMINISTIC_RANKSIZE ? DETERMINISTIC_RANKSIZE : DOUBLE;
        }
        __gm__ int32_t* ctrlFlagGMDone
            = (__gm__ int32_t*)(GM_OUT[x] + flagOffsetBase + (DOUBLE * rankSize_ + lastOpCore) * FLAG_SIZE);
        int64_t dataCntSize = (x == rankSize_ - 1) ? lastDataNum : avgDataNum;

        CPGM2GMAccordingFlag(outputGM + x * avgDataNum, cclGMOther + curCount, dataCntSize, 0, ctrlFlagGMDone, tag, 0);
    }
}

template <typename T>
__aicore__ inline void AivAllReduceDeterMid910B::ProcessSingleRanksizeCore(
    GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize)
{
    int64_t curCount = len;
    int64_t blockNumPerGroup = rankSize_;
    int64_t x = blockIdx_ % blockNumPerGroup;
    int64_t avgDataNum = curCount / rankSize_;
    int64_t lastDataNum = curCount - (rankSize_ - 1) * avgDataNum;
    int64_t flagOffsetBasic = separateOffset + BASE_FLAG_OFFSET * AIV_ALL_REDUCE_DETER_910B_MIDDATA;

    uint32_t flagOffsetBase = ((tag % 2 == 0) ? 0 : 6 * rankSize_ * FLAG_SIZE) + flagOffsetBasic;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_] + dataOffset);
    __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[x] + dataOffset);
    __gm__ T* outputGM = (__gm__ T*)output;

    int64_t flagOffset1stCount = flagOffsetBase + (x)*FLAG_SIZE;
    int64_t flagOffset2stCount = flagOffsetBase + (rankSize_ + x) * FLAG_SIZE;
    int64_t flagOffset3stCount = flagOffsetBase + (DOUBLE * rankSize_ + x) * FLAG_SIZE;

    // 先从input拷贝到cclbuffer
    int64_t dataCntSize = (x == rankSize_ - 1) ? lastDataNum : avgDataNum;
    CpGM2GMWithFlagWrap(
        cclGMSelf + x * avgDataNum, inputGM + x * avgDataNum, dataCntSize,
        (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset1stCount), 8, tag);
    PipeBarrier<PIPE_ALL>();

    // 拷贝cclbuffer前半部分到cllbuffer后半部分
    __gm__ int32_t* flagCntDoneOtner = (__gm__ int32_t*)(GM_OUT[x] + flagOffsetBase + (rank_)*FLAG_SIZE);
    __gm__ int32_t* flagCntSelf = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2stCount);
    if (x == 0) {
        flagCntSelf = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset3stCount);
    }
    dataCntSize = (rank_ == rankSize_ - 1) ? lastDataNum : avgDataNum;

    CPGM2GMAccordingFlag(
        cclGMSelf + curCount + x * dataCntSize, cclGMOther + rank_ * avgDataNum, dataCntSize, flagCntSelf,
        flagCntDoneOtner, tag, 8);
    PipeBarrier<PIPE_ALL>();

    // 进行reduce操作
    if (x != 0) {
        dataCntSize = (rank_ == rankSize_ - 1) ? lastDataNum : avgDataNum;
        if (rankSize_ >= DETERMINISTIC_RANKSIZE) {
            SumByPairs(cclGMSelf + curCount, x, dataCntSize, tag, flagOffsetBase);
        } else {
            __gm__ int32_t* flagCntDonePre = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset3stCount - FLAG_SIZE);
            __gm__ int32_t* flagCntSelf = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset2stCount);
            __gm__ int32_t* flagCntDoneSelf = (__gm__ int32_t*)(GM_OUT[rank_] + flagOffset3stCount);

            ReduceWithFlagWrap(
                cclGMSelf + curCount, cclGMSelf + curCount + x * dataCntSize, dataCntSize, tag, flagCntDonePre,
                flagCntSelf, flagCntDoneSelf);
        }
    }
    PipeBarrier<PIPE_ALL>();

    // 搬运cclbuffer到output
    int32_t lastOpCore = rankSize_ - 1;
    if (rankSize_ >= DETERMINISTIC_RANKSIZE) {
        lastOpCore = rankSize_ > DETERMINISTIC_RANKSIZE ? DETERMINISTIC_RANKSIZE : DOUBLE;
    }
    __gm__ int32_t* ctrlFlagGMDone
        = (__gm__ int32_t*)(GM_OUT[x] + flagOffsetBase + (DOUBLE * rankSize_ + lastOpCore) * FLAG_SIZE);
    dataCntSize = (x == rankSize_ - 1) ? lastDataNum : avgDataNum;

    CPGM2GMAccordingFlag(outputGM + x * avgDataNum, cclGMOther + curCount, dataCntSize, 0, ctrlFlagGMDone, tag, 0);
}

template <typename T>
__aicore__ inline void
AivAllReduceDeterMid910B::ProcessProxy(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize)
{
    if (numBlocks_ == rankSize_) {
        ProcessSingleRanksizeCore<T>(input, output, len, tag, bufferSize);
    } else {
        Process<T>(input, output, len, tag, bufferSize);
    }
}

template <typename T>
__aicore__ inline void aiv_all_reduce_deter_910b_middata(KERNEL_ARGS_DEF)
{
    AivAllReduceDeterMid910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    int32_t curTag = (tag << 15);
    op.ProcessProxy<T>(input, output, len, curTag, bufferSize);
    if (tag == 1000) {
        op.EndSync(tag);
    }
    op.TailCounter();
}
