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

// 中间转发节点数: 总rank数减去首(root)尾两个数据端点
constexpr uint32_t BROADCAST_MID_NODE_COUNT = 2;

class AivBroadcastCrossNode91093 : public AivCrossNode91093Base {
public:
    __aicore__ inline AivBroadcastCrossNode91093() {}

    template <typename T>
    __aicore__ inline void ProcessSmallData(
        GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag,
        uint64_t len, uint64_t maxCountPerLoop, uint32_t root);

    __aicore__ inline void InitSelf(
        GM_ADDR buffOut0, GM_ADDR buffOut1, uint32_t rank, uint32_t rankSize, int32_t tag, uint32_t numBlocks,
        bool useDoubleBuffer, uint32_t root, bool isOpBase);

    __aicore__ inline void InitDataCopyOffset(uint64_t len, uint64_t maxCountPerLoop);

    __aicore__ inline void WaitRecordSync(int32_t tag, uint32_t root, GM_ADDR rootAddr, GM_ADDR* buffersOut);

    __aicore__ inline void CalcNumTargetsAndTargetRanksForBRC(uint32_t root);
};

__aicore__ inline void AivBroadcastCrossNode91093::InitSelf(
    GM_ADDR buffOut0, GM_ADDR buffOut1, uint32_t rank, uint32_t rankSize, int32_t tag, uint32_t numBlocks,
    bool useDoubleBuffer, uint32_t root, bool isOpBase)
{
    flagAddrSelf_ = buffOut0;
    rank_ = rank;
    tag_ = tag;
    rankSize_ = rankSize;
    useDoubleBuffer_ = useDoubleBuffer;
    usedBlockNum_ = numBlocks;
    numBlocks_ = numBlocks;
    pingpongOffset = 0;
    blockGroup_ = numBlocks;
    commAddr_ = buffOut1;

    InitSetCheckClearArgsTensor();
    if (rankSize_ - 1 >= numBlocks_) {
        blockNumPerGroup = 1;
    } else {
        blockNumPerGroup = numBlocks_ / (rankSize_ - 1);
    }
    InitOffset();
    // 每1000次的时候清零一次
    if (tag_ == 1) {
        CalcNumTargetsAndTargetRanks(); // 1000次清零操作与bc操作有些不一样，因此获取rank逻辑单独提取
        GetTargetBuffer(isOpBase);
        ClearCycle();
    }
    CalcNumTargetsAndTargetRanksForBRC(root);
    GetTargetBuffer(isOpBase);
}

__aicore__ inline void AivBroadcastCrossNode91093::CalcNumTargetsAndTargetRanksForBRC(uint32_t root)
{
    numTargets = (rankSize_ - 1) / usedBlockNum_;
    uint32_t tailRankSize = (rankSize_ - 1) % usedBlockNum_;
    if (tailRankSize > 0 && blockIdx_ < tailRankSize) {
        numTargets += 1;
    }

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = (blockIdx_ + i * usedBlockNum_) % rankSize_;
        if (targetRank >= root) {
            targetRank += 1;
        }
        targetRanks[i] = targetRank;
    }
}

__aicore__ inline void AivBroadcastCrossNode91093::InitDataCopyOffset(uint64_t len, uint64_t maxCountPerLoop)
{
    uint64_t countPerRank = maxCountPerLoop / (rankSize_ - 1);

    if (len < maxCountPerLoop) {
        countMid = 0;
        countTail = len / (rankSize_ - 1);
        countTailLast_ = len - (rankSize_ - BROADCAST_MID_NODE_COUNT) * countTail;
    } else if (len % maxCountPerLoop == 0) {
        countMid = countPerRank;
        countTail = countPerRank;
        countTailLast_ = countPerRank;
    } else {
        countMid = countPerRank;
        uint64_t remainLen = len % maxCountPerLoop;
        countTail = remainLen / (rankSize_ - 1);
        countTailLast_ = remainLen - (rankSize_ - BROADCAST_MID_NODE_COUNT) * countTail;
    }
}

__aicore__ inline void
AivBroadcastCrossNode91093::WaitRecordSync(int32_t tag, uint32_t root, GM_ADDR rootAddr, GM_ADDR* buffersOut)
{
    for (uint32_t i = 0; i < numTargets; ++i) {
        int32_t targetRank = targetRanks[i];
        if (rank_ != root && targetRank == rank_) {
            Wait1vN(
                tag * (rankSize_ - BROADCAST_MID_NODE_COUNT), CommPattern::interRank, true,
                AivNotifyType::DataSignal); // 等待n个对端拿走数据
            PipeBarrier<PIPE_ALL>();
            Record(tag, rootAddr, AivNotifyType::DataSignal);
        } else if (rank_ != root) {
            RecordNv1(tag, buffersOut[i], false, AivNotifyType::DataSignal); // 告诉对端已经拿来数据了
            PipeBarrier<PIPE_ALL>();
        } else {
            Wait(tag, targetRank, AivNotifyType::DataSignal);
        }
    }
}

template <typename T>
__aicore__ inline void AivBroadcastCrossNode91093::ProcessSmallData(
    GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR commInfoAddr, GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t len,
    uint64_t maxCountPerLoop, uint32_t root)
{
    // 内存准备
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* outputGM = (__gm__ T*)output;
    __gm__ T* cclGMSelf = (__gm__ T*)buffIn0;

    GlobalTensor<uint64_t> bufferArgsGT;
    __gm__ uint64_t* buffersGmAddr = (__gm__ uint64_t*)(commInfoAddr);
    bufferArgsGT.SetGlobalBuffer(buffersGmAddr, FLAG_SIZE * rankSize_ / sizeof(uint64_t));

    DataCopy(bufferArgsTensor[numTargets * IDX_4], bufferArgsGT[2 * root], IDX_4);
    SyncFunc<HardEvent::MTE2_S>();

    uint32_t rootIdx = numTargets * IDX_4;
    __gm__ T* cclGMRoot = (__gm__ T*)((GM_ADDR)(bufferArgsTensor.GetValue(rootIdx)));
    GM_ADDR ctrlFlagGMRoot = (GM_ADDR)(bufferArgsTensor.GetValue(rootIdx + 1));

    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    uint64_t curOffset = 0;
    uint64_t curCount = 0;

    uint64_t bufferLoopNum = CeilDiv(len, maxCountPerLoop);
    for (uint64_t loop = 0; loop < bufferLoopNum; loop++) {
        if (loop == bufferLoopNum - 1) {
            curCount = countTail;
        } else {
            curCount = countMid;
        }

        for (uint32_t i = 0; i < numTargets; ++i) {
            __gm__ T* cclGMOther = (__gm__ T*)(buffersIn[i]);
            GM_ADDR ctrlFlagGMOther = buffersOut[i];

            uint32_t targetRank = targetRanks[i];
            uint64_t dataCopyOffset = targetRank * curCount;
            if (targetRank > root) {
                dataCopyOffset = (targetRank - 1) * curCount;
            }

            if (loop == bufferLoopNum - 1) {
                if ((root == rankSize_ - 1 && targetRank == rankSize_ - 2)
                    || (root < rankSize_ - 1 && targetRank == rankSize_ - 1)) {
                    curCount = countTailLast_;
                }
            }

            // root卡cclBuffer -> 本端(cclbuffer && output)
            if ((rank_ < root && targetRank == rank_) || (rank_ > root && targetRank == rank_)) {
                // 等待root就绪
                CountWaitGE(ctrlFlagGMRoot, curTag, targetRank);
                CpGM2GM(cclGMSelf + dataCopyOffset, cclGMRoot + dataCopyOffset, curCount);
                PipeBarrier<PIPE_ALL>();
                CountRecord(curTag, targetRank);
                PipeBarrier<PIPE_ALL>();
                CpGM2GM(outputGM + curOffset + dataCopyOffset, cclGMSelf + dataCopyOffset, curCount);
                PipeBarrier<PIPE_ALL>();
            } else if (rank_ != root) { // 对端cclbuffer -> 本端output
                CountWaitGE(ctrlFlagGMOther, curTag, targetRank);
                CpGM2GM(outputGM + curOffset + dataCopyOffset, cclGMOther + dataCopyOffset, curCount);
                PipeBarrier<PIPE_ALL>();
            } else { // root卡userInput -> root卡cclBuffer
                CpGM2GM(cclGMRoot + dataCopyOffset, inputGM + curOffset + dataCopyOffset, curCount);
                PipeBarrier<PIPE_ALL>();
                // root已就绪
                CountRecord(curTag, targetRank);
                PipeBarrier<PIPE_ALL>();
            }
        }
        // 尾同步
        WaitRecordSync(curTag, root, ctrlFlagGMRoot, buffersOut);

        curOffset += maxCountPerLoop;
        curTag += 1;
    }
}

template <typename T>
__aicore__ inline void aiv_broadcast_crossnode_91093(KERNEL_ARGS_DEF_A3)
{
    AivBroadcastCrossNode91093 op;

    uint64_t basicLoopCount = (rankSize - 1) * UB_MAX_DATA_SIZE / sizeof(T);
    uint64_t bufferCount = (uint64_t)bufferSize / sizeof(T);
    uint64_t maxCountPerLoop = bufferCount / basicLoopCount * basicLoopCount;
    if (bufferCount < basicLoopCount) {
        maxCountPerLoop = basicLoopCount;
    }

    op.InitSelf(buffOut0, buffOut1, rank, rankSize, tag, numBlocks, false, root, isOpBase);
    op.InitDataCopyOffset(len, maxCountPerLoop);
    op.InitOpCounter(headCountMem, tailCountMem, addOneMem, SIZE_OF_INT32, isEnableCounter);
    op.HeadCounter();
    op.ProcessSmallData<T>(buffIn0, buffOut0, buffOut1, input, output, tag, len, maxCountPerLoop, root);
    op.TailCounter();
}
