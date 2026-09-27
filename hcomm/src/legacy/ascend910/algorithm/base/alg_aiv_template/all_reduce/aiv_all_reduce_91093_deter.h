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
#include "aiv_crossnode_91093_base.h"

using namespace AscendC;

class AivAllReduce91093Deter : public AivCrossNode91093Base {
public:
    __aicore__ inline AivAllReduce91093Deter() {}

    template <typename T>
    __aicore__ inline void InitDataCopyOffset(uint64_t perRankBufferCount, uint64_t halfBufferCount, uint64_t len);

    __aicore__ inline uint32_t CeilLog2(uint32_t n);
    __aicore__ inline uint32_t GetLargestPowerOf2(uint32_t n);

    template <typename T>
    __aicore__ inline void Process(
        GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
        uint64_t halfBufferCount, uint64_t avgBuffercount, uint64_t len);
};

template <typename T>
__aicore__ inline void
AivAllReduce91093Deter::InitDataCopyOffset(uint64_t perRankBufferCount, uint64_t halfBufferCount, uint64_t len)
{
    // 以下根据不同情况，计算每个aiv核的数据搬运参数
    // 当rankSize大于总aiv核数，使用1个aiv服务一个对端，需要多次通信
    if (rankSize_ > usedBlockNum_) {
        CalcNumTargetsAndTargetRanksGroup();

        blockNumPerGroup = 1;
        blockIdxInGroup = 0;
        if (len <= halfBufferCount) { // ccl够用，只需要搬一轮的情况
            countMid = 0;

            countTail = len / rankSize_;
            countTailLast_ = len - (rankSize_ - 1) * countTail;
        } else if (len % halfBufferCount == 0) { // ccl不够用，要搬多轮的情况1: 能整除
            countMid = perRankBufferCount;

            countTail = perRankBufferCount;
            countTailLast_ = countTail;
        } else { // ccl不够用，要搬多轮的情况2: 不能整除
            countMid = perRankBufferCount;

            uint64_t remainLen = len % halfBufferCount;
            countTail = remainLen / rankSize_;
            countTailLast_ = remainLen - (rankSize_ - 1) * countTail;
        }
        blockOffsetMid = 0;
        blockOffsetTail = 0;
        blockOffsetTailLast_ = 0;
        flagOffsetInGroup = 0;
        countPerCore = len;
        blockOffset = 0;
        groupMid_ = countMid;
        groupTail_ = countTail;
        groupTailLast_ = countTailLast_;
        // 当rankSize小于等于总aiv核数时，根据ranksize和数据量大小选择使用多个aiv服务一个对端（多核并行），只需一次通信
    } else {
        numTargets = 1;
        blockNumPerGroup = numBlocks_ / rankSize_; // 多少个aiv服务一个rank
        targetRanks[0] = blockIdx_ / blockNumPerGroup;

        uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
        blockIdxInGroup = blockIdx_ % blockNumPerGroup;

        if (len <= halfBufferCount) { // ccl够用，只需要搬一轮的情况
            countMid = 0;
            blockOffsetMid = 0;
            groupMid_ = 0;

            groupTail_ = len / rankSize_;
            groupTailLast_ = len - (rankSize_ - 1) * groupTail_;
            CalCountAndBlockOffset(groupTail_, blockNumPerGroup, blockIdxInGroup, padCount, countTail, blockOffsetTail);
            CalCountAndBlockOffset(
                groupTailLast_, blockNumPerGroup, blockIdxInGroup, padCount, countTailLast_, blockOffsetTailLast_);
        } else if (len % halfBufferCount == 0) { // ccl不够用，要搬多轮的情况1: 能整除
            groupMid_ = perRankBufferCount;
            CalCountAndBlockOffset(groupMid_, blockNumPerGroup, blockIdxInGroup, padCount, countMid, blockOffsetMid);

            countTail = countMid;
            blockOffsetTail = blockOffsetMid;
            groupTail_ = groupMid_;
            groupTailLast_ = groupTail_;
            countTailLast_ = countTail;
            blockOffsetTailLast_ = blockOffsetTail;
        } else { // ccl不够用，要搬多轮的情况2: 不能整除
            groupMid_ = perRankBufferCount;
            CalCountAndBlockOffset(groupMid_, blockNumPerGroup, blockIdxInGroup, padCount, countMid, blockOffsetMid);

            uint64_t remainLen = len % halfBufferCount;
            groupTail_ = remainLen / rankSize_;
            groupTailLast_ = remainLen - (rankSize_ - 1) * groupTail_;

            CalCountAndBlockOffset(groupTail_, blockNumPerGroup, blockIdxInGroup, padCount, countTail, blockOffsetTail);
            CalCountAndBlockOffset(
                groupTailLast_, blockNumPerGroup, blockIdxInGroup, padCount, countTailLast_, blockOffsetTailLast_);
        }
        flagOffsetInGroup = blockIdxInGroup * FLAG_SIZE;
        CalCountAndBlockOffset(len, blockNumPerGroup, blockIdxInGroup, padCount, countPerCore, blockOffset);
    }
}

__aicore__ inline uint32_t AivAllReduce91093Deter::CeilLog2(uint32_t n)
{
    if (n <= 1)
        return 0;
    uint32_t result = 0;
    uint32_t value = 1;
    while (value < n) {
        value <<= 1;
        result++;
    }
    return result;
}

__aicore__ inline uint32_t AivAllReduce91093Deter::GetLargestPowerOf2(uint32_t n)
{
    if (n <= 1)
        return 0;
    uint32_t result = 1;
    while (result * 2 < n) { // 严格小于
        result <<= 1;
    }
    return result;
}

template <typename T>
__aicore__ inline void AivAllReduce91093Deter::Process(
    GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
    uint64_t halfBufferCount, uint64_t avgBufferCount, uint64_t len)
{
    TQue<AscendC::TPosition::VECIN, 1> syncQue;
    GlobalTensor<int32_t> syncGlobal;
    GlobalTensor<int32_t> syncGlobalSecond;
    uint32_t syncBufferSize = numBlocks_ * 32;
    LocalTensor<int32_t> workLocal;

    pipe.InitBuffer(syncQue, 1, syncBufferSize);
    syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(buffOut0 + syncAllOffset), syncBufferSize);
    syncGlobalSecond.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(buffOut0 + syncAllOffset + syncBufferSize), syncBufferSize);
    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)buffIn0;

    GetTargetBuffer();

    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    bool pingpong = false;
    uint64_t curOffset = 0;
    uint64_t curCount = 0;
    uint64_t curBlockOffset = 0;
    uint64_t curGroupCount = 0;
    uint64_t curCountLast = 0;
    uint64_t curBlockOffsetLast = 0;
    uint64_t curGroupCountLast = 0;

    uint64_t halfBufferCountCcl
        = halfBufferCount
          + rankSize_; // 某些场景，groupTailLast会比avgBufferCount大，前一半buffer的末尾和后一半buffer的开头可能踩踏
    uint32_t bufferLoopNum = (len + halfBufferCount - 1) / halfBufferCount;
    for (uint32_t loop = 0; loop < bufferLoopNum; loop++) {
        if (loop == bufferLoopNum - 1) { // 最后一轮ccl填充
            curCount = countTail;
            curBlockOffset = blockOffsetTail;
            curGroupCount = groupTail_; // Tail 最后一轮

            curCountLast = countTailLast_;
            curBlockOffsetLast = blockOffsetTailLast_;
            curGroupCountLast = groupTailLast_;
        } else {
            curCount = countMid;
            curBlockOffset = blockOffsetMid;
            curGroupCount = groupMid_;

            curCountLast = curCount;
            curBlockOffsetLast = curBlockOffset;
            curGroupCountLast = curGroupCount;
        }

        workLocal = syncQue.AllocTensor<int32_t>();
        // step1 本端 input -> 本端 ccl
        for (uint32_t i = 0; i < numTargets; i++) {
            uint64_t recvOffset = avgBufferCount * targetRanks[i];
            uint64_t sendOffset = curGroupCount * targetRanks[i];

            PipeBarrier<PIPE_ALL>();
            if (targetRanks[i] == rankSize_ - 1) {
                CpGM2GM(
                    cclGMSelf + recvOffset + curBlockOffsetLast, inputGM + curOffset + sendOffset + curBlockOffsetLast,
                    curCountLast);
            } else {
                CpGM2GM(
                    cclGMSelf + recvOffset + curBlockOffset, inputGM + curOffset + sendOffset + curBlockOffset,
                    curCount);
            }
        }

        PipeBarrier<PIPE_ALL>();
        BatchRecordWait(curTag, buffersOut, AivNotifyType::ACK);
        PipeBarrier<PIPE_ALL>();

        // step2 对端 ccl -> 本端 ccl
        for (uint32_t i = 0; i < numTargets; i++) {
            __gm__ T* cclGMOther = (__gm__ T*)(buffersIn[i]);
            uint64_t remoteSendOffset = avgBufferCount * rank_;
            uint64_t dataNum = curGroupCount;
            if (rank_ == rankSize_ - 1) {
                dataNum = curGroupCountLast;
            }
            uint64_t localRecvOffset = dataNum * targetRanks[i];

            PipeBarrier<PIPE_ALL>();
            if (rank_ == rankSize_ - 1) {
                CpGM2GM(
                    cclGMSelf + halfBufferCountCcl + localRecvOffset + curBlockOffsetLast,
                    cclGMOther + remoteSendOffset + curBlockOffsetLast, curCountLast);
            } else {
                CpGM2GM(
                    cclGMSelf + halfBufferCountCcl + localRecvOffset + curBlockOffset,
                    cclGMOther + remoteSendOffset + curBlockOffset, curCount);
            }
        }

        PipeBarrier<PIPE_ALL>();
        BatchRecordWait(curTag, buffersOut, AivNotifyType::DataSignal);

        // 卡内每个核同步一次
        PipeBarrier<PIPE_ALL>();
        SyncAll(syncGlobal, workLocal, numBlocks_);
        PipeBarrier<PIPE_ALL>();

        // step3 新的二分归约算法
        // 每轮: 后半部分(offset powerOf2 到 curBlocks-1) 加到 前半部分(offset 0 到 powerOf2-1)
        // 每个核负责连续的offset，每轮重新划分
        uint32_t numReduce = rankSize_ < usedBlockNum_ ? rankSize_ : usedBlockNum_;
        uint32_t totalRounds = CeilLog2(rankSize_);

        if (blockIdx_ < numReduce) {
            uint64_t dataNum = curGroupCount;
            if (rank_ == rankSize_ - 1) {
                dataNum = curGroupCountLast;
            }
            uint32_t curBlocks = rankSize_;
            for (uint32_t round = 0; round < totalRounds; round++) {
                uint32_t powerOf2 = GetLargestPowerOf2(curBlocks);
                if (powerOf2 == 0)
                    break;

                // 计算当前核负责的offset范围
                uint32_t offsetsPerCore = (curBlocks + numReduce - 1) / numReduce;
                uint32_t startOffset = blockIdx_ * offsetsPerCore;
                uint32_t endOffset = startOffset + offsetsPerCore;
                if (endOffset > curBlocks)
                    endOffset = curBlocks;

                // 再处理前半部分的offset: 等待并执行reduce
                for (uint32_t offset = startOffset; offset < endOffset; offset++) {
                    if (offset < powerOf2) {
                        // 处理所有对应的后半部分offset
                        uint32_t backIdx = powerOf2 + offset;
                        // 等待后半部分对应offset
                        if (round > 0) {
                            WaitSyncFlag(curTag + round, flagAddrSelf_, 1, backIdx, pingpong);
                            WaitSyncFlag(curTag + round, flagAddrSelf_, 1, offset, pingpong);
                        }

                        // 后半部分reduce到前半部分
                        uint64_t frontOffset = dataNum * offset;
                        uint64_t backOffset = dataNum * backIdx;

                        if (backIdx < curBlocks) {
                            PipeBarrier<PIPE_ALL>();
                            CpGM2GM<T>(
                                cclGMSelf + halfBufferCountCcl + frontOffset,
                                cclGMSelf + halfBufferCountCcl + backOffset, dataNum, true, reduceOp_);
                            PipeBarrier<PIPE_ALL>();
                        }
                        SetSyncRecord(curTag + round + 1, flagAddrSelf_, 1, offset, pingpong);
                    }
                }

                curBlocks = powerOf2;
                PipeBarrier<PIPE_ALL>();
            }
        }

        // step4 对端ccl -> 本端 output
        for (uint32_t i = 0; i < numTargets; i++) {
            uint32_t targetRank = targetRanks[i];
            __gm__ T* cclGMOther = (__gm__ T*)(buffersIn[i]);

            // 等待对端完成reduce (最后一轮核0的同步)
            WaitSyncFlag(curTag + totalRounds, buffersOut[i], 1, 0, pingpong);

            // 从对端的offset 0拷贝结果
            uint64_t srcOffset = 0;
            uint64_t dstOffset = curGroupCount * targetRank;

            PipeBarrier<PIPE_ALL>();
            if (targetRank == rankSize_ - 1) {
                CpGM2GM(
                    outputGM + curOffset + dstOffset + curBlockOffsetLast,
                    cclGMOther + halfBufferCountCcl + srcOffset + curBlockOffsetLast, curCountLast);
            } else {
                CpGM2GM(
                    outputGM + curOffset + dstOffset + curBlockOffset,
                    cclGMOther + halfBufferCountCcl + srcOffset + curBlockOffset, curCount);
            }
        }

        // 尾同步
        PipeBarrier<PIPE_ALL>();
        BatchRecordWait(curTag, buffersOut, AivNotifyType::Done);
        if (bufferLoopNum > 1) {
            PipeBarrier<PIPE_ALL>();
            SyncAll(syncGlobalSecond, workLocal, numBlocks_);
            PipeBarrier<PIPE_ALL>();
        }

        syncQue.FreeTensor(workLocal);
        curTag += totalRounds + 1;
        curOffset += halfBufferCount;
    }
}

template <typename T>
__aicore__ inline void aiv_all_reduce_91093_deter(KERNEL_ARGS_DEF_A3)
{
    AivAllReduce91093Deter op;
    uint64_t totalBufferCount = (uint64_t)bufferSize / sizeof(T);
    // 某些场景，groupTailLast 比 avgBufferCount 大，故预留空间
    uint64_t reservedCount = (rankSize + 1) * (rankSize + 1);

    uint64_t avgBufferCount = (totalBufferCount - reservedCount) / 2 / rankSize;
    uint64_t halfBufferCount = avgBufferCount * rankSize;

    op.InitDeter<T>(buffOut0, buffOut1, rank, rankSize, reduceOp, tag, numBlocks, false);
    op.InitDataCopyOffset<T>(avgBufferCount, halfBufferCount, len);
    op.InitOpCounter(headCountMem, tailCountMem, addOneMem, SIZE_OF_INT32, isEnableCounter);
    op.HeadCounter();
    op.Process<T>(buffIn0, buffOut0, buffOut1, input, output, tag, halfBufferCount, avgBufferCount, len);
    op.TailCounter();
}

__aicore__ inline void sk_all_reduce_deter(SUPERKERNEL_ARGS_DEF)
{
    AivAllReduce91093Deter op;

    op.InitSuperKernel(hiddenInput, false);
    // 每张卡的CCLBuffer大小为bufferSize, 平均分给ranksize*2块，每块的大小为avgBufferCount
    uint64_t avgBufferCount = op.len_ / op.rankSize_;
    uint64_t halfBufferCount = avgBufferCount * op.rankSize_;

    if (op.dataType_ == HcclDataType::HCCL_DATA_TYPE_INT8) {
        op.InitDataCopyOffset<int8_t>(avgBufferCount, halfBufferCount, op.len_);
        op.Process<int8_t>(
            op.dataAddrSelf_, op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, halfBufferCount, avgBufferCount,
            op.len_);
    } else if (op.dataType_ == HcclDataType::HCCL_DATA_TYPE_INT16) {
        op.InitDataCopyOffset<int16_t>(avgBufferCount, halfBufferCount, op.len_);
        op.Process<int16_t>(
            op.dataAddrSelf_, op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, halfBufferCount, avgBufferCount,
            op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_INT32) {
        op.InitDataCopyOffset<int32_t>(avgBufferCount, halfBufferCount, op.len_);
        op.Process<int32_t>(
            op.dataAddrSelf_, op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, halfBufferCount, avgBufferCount,
            op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_FP16) {
        op.InitDataCopyOffset<half>(avgBufferCount, halfBufferCount, op.len_);
        op.Process<half>(
            op.dataAddrSelf_, op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, halfBufferCount, avgBufferCount,
            op.len_);
    } else if (op.dataType_ == HCCL_DATA_TYPE_FP32) {
        op.InitDataCopyOffset<float>(avgBufferCount, halfBufferCount, op.len_);
        op.Process<float>(
            op.dataAddrSelf_, op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, halfBufferCount, avgBufferCount,
            op.len_);
    } else {
        op.InitDataCopyOffset<bfloat16_t>(avgBufferCount, halfBufferCount, op.len_);
        op.Process<bfloat16_t>(
            op.dataAddrSelf_, op.flagAddrSelf_, op.commAddr_, input, output, op.tag_, halfBufferCount, avgBufferCount,
            op.len_);
    }
}
