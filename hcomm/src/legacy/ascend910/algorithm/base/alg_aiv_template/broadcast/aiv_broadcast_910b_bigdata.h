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
#include "sync_interface.h"

using namespace AscendC;

class AivBroadcastBig910B : public AivCommBase {
public:
    __aicore__ inline AivBroadcastBig910B() {}

    __aicore__ inline void WaitRecordSync(int32_t tag, uint32_t root);

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint32_t root);
    template <typename T>
    __aicore__ inline void Process2Rank(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint32_t root);
};

__aicore__ inline void AivBroadcastBig910B::WaitRecordSync(int32_t tag, uint32_t root)
{
    int32_t targetRank = (blockIdx_ < root) ? blockIdx_ : (blockIdx_ + 1);
    int32_t nowTag = tag >> TAG_MOVE_LEFT_BITS;
    bool ifPingpong = (nowTag % 2 == 0);
    if ((rank_ < root && blockIdx_ == rank_) || (rank_ > root && blockIdx_ == rank_ - 1)) {
        for (uint32_t remoteRank = 0; remoteRank < rankSize_; remoteRank += 1) {
            if (remoteRank == root || remoteRank == rank_) {
                continue;
            } else {
                Wait(tag, remoteRank, AivNotifyType::DataSignal, 0, ifPingpong);
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_ALL>();
        Record(tag, root, AivNotifyType::DataSignal, 0, ifPingpong); // 告诉root数据拿过来了，确保root卡最后推出
    } else if (rank_ != root) {
        Record(tag, targetRank, AivNotifyType::DataSignal, 0, ifPingpong);
        PipeBarrier<PIPE_ALL>();
    } else {
        Wait(tag, targetRank, AivNotifyType::DataSignal, 0, ifPingpong); // 等待对应卡的数据拿走
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void
AivBroadcastBig910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint32_t root)
{
    uint64_t blockNumPerGroup = rankSize_ - 1; // 每组使用ranksize-1个核
    uint64_t ubLength = UB_MAX_DATA_SIZE / sizeof(T);
    uint64_t blockTotal = CeilDiv(len, ubLength);                   // 总搬运次数(需要多少核)
    if (blockIdx_ >= blockNumPerGroup || blockIdx_ >= blockTotal) { // 无数据块的核直接返回，消除跨算子flag覆盖死锁
        return;
    }
    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMRoot = (__gm__ T*)(GM_IN[root]);  // root卡的cclbuffer
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_]); // 当前卡的cclbuffer
    __gm__ T* outputGM = (__gm__ T*)output;
    int32_t targetRank = (blockIdx_ < root) ? blockIdx_ : (blockIdx_ + 1);
    for (uint64_t curIndex = blockIdx_; curIndex < blockTotal; curIndex += blockNumPerGroup) {
        uint64_t dataOffset = curIndex * ubLength;
        uint64_t curCount = (curIndex == blockTotal - 1) ? (len - dataOffset) : ubLength;
        if ((rank_ < root && blockIdx_ == rank_) || (rank_ > root && blockIdx_ == rank_ - 1)) {
            // 从root的cclbuffer搬数据到自己的cclbuffer
            __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[root] + countOffset + blockIdx_ * FLAG_SIZE);
            WaitSignalGEValue(ctrlFlagGM, localCheckGETensor, tag + curIndex);
            PipeBarrier<PIPE_ALL>();
            GlobalTensor<T> inputGT;
            inputGT.SetGlobalBuffer(cclGMRoot + dataOffset, curCount);
            GlobalTensor<T> outputGT;
            outputGT.SetGlobalBuffer(outputGM + dataOffset, curCount);
            GlobalTensor<T> cclGT;
            cclGT.SetGlobalBuffer(cclGMSelf + dataOffset, curCount);
            LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
            DataCopyGM2UB(localIn, inputGT[0], curCount);
            inOutQue.EnQue(localIn);
            LocalTensor<T> localOut = inOutQue.DeQue<T>();
            DataCopyUB2GM(cclGT[0], localOut, curCount);
            PipeBarrier<PIPE_ALL>();
            CountRecord(tag, curIndex, blockIdx_);
            PipeBarrier<PIPE_ALL>();
            DataCopyUB2GM(outputGT[0], localOut, curCount);
            inOutQue.FreeTensor(localOut);
        } else if (rank_ != root) {
            // 从对应卡的cclbuffer拉数据到自己的output
            __gm__ T* cclGMOther = (__gm__ T*)(GM_IN[targetRank]); // targetRank号卡的cclbuffer
            __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[targetRank] + countOffset + blockIdx_ * FLAG_SIZE);
            WaitSignalGEValue(ctrlFlagGM, localCheckGETensor, tag + curIndex);
            PipeBarrier<PIPE_ALL>();
            CpGM2GM(outputGM + dataOffset, cclGMOther + dataOffset, curCount);
            PipeBarrier<PIPE_ALL>();
        } else {
            // 把root卡的数据搬到root的cclbuffer
            CpGM2GM(cclGMRoot + dataOffset, inputGM + dataOffset, curCount);
            PipeBarrier<PIPE_ALL>();
            CountRecord(tag, curIndex, blockIdx_);
            PipeBarrier<PIPE_ALL>();
        }
    }
    // 同步
    WaitRecordSync(tag, root);
}

template <typename T>
__aicore__ inline void
AivBroadcastBig910B::Process2Rank(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint32_t root)
{
    uint64_t ubLength = UB_MAX_DATA_SIZE / sizeof(T);
    uint64_t blockTotal = CeilDiv(len, ubLength); // 总搬运次数(需要多少核)

    __gm__ T* inputGM = (__gm__ T*)input;
    __gm__ T* cclGMRoot = (__gm__ T*)(GM_IN[root]);  // root卡的cclbuffer
    __gm__ T* cclGMSelf = (__gm__ T*)(GM_IN[rank_]); // 当前卡的cclbuffer
    __gm__ T* outputGM = (__gm__ T*)output;

    if (blockIdx_ >= rankSize_) {
        return;
    }
    for (uint64_t curIndex = blockIdx_; curIndex < blockTotal; curIndex += rankSize_) {
        uint64_t dataOffset = curIndex * ubLength;
        uint64_t curCount = (curIndex == blockTotal - 1) ? (len - dataOffset) : ubLength;
        if (rank_ == root) {
            // 把root数据拷贝到自己的cclbuffer
            CpGM2GM(cclGMRoot + dataOffset, inputGM + dataOffset, curCount);
            PipeBarrier<PIPE_ALL>();
            // 告诉另一张卡可以读取数据了
            CountRecord(tag, curIndex, blockIdx_);
            PipeBarrier<PIPE_ALL>();
        } else {
            __gm__ int32_t* ctrlFlagGM = (__gm__ int32_t*)(GM_OUT[root] + countOffset + blockIdx_ * FLAG_SIZE);
            WaitSignalGEValue(ctrlFlagGM, localCheckGETensor, tag + curIndex);
            PipeBarrier<PIPE_ALL>();
            CpGM2GM(outputGM + dataOffset, cclGMRoot + dataOffset, curCount);
            PipeBarrier<PIPE_ALL>();
        }
    }
    int32_t nowTag = tag >> TAG_MOVE_LEFT_BITS;
    bool ifPingpong = (nowTag % 2 == 0);
    if (rank_ == root) {
        int32_t anotherRank = (root == 0) ? 1 : 0;
        if (blockIdx_ == 0) {
            Wait(tag, anotherRank, AivNotifyType::Done, 0, ifPingpong);
        } else {
            Wait(tag, anotherRank, AivNotifyType::DataSignal, 0, ifPingpong);
        }
        PipeBarrier<PIPE_ALL>();
    } else {
        if (blockIdx_ == 0) {
            Record(tag, root, AivNotifyType::Done, 0, ifPingpong);
        } else {
            Record(tag, root, AivNotifyType::DataSignal, 0, ifPingpong);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void aiv_broadcast_910b_bigdata(KERNEL_ARGS_DEF)
{
    AivBroadcastBig910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.HeadCounter();
    uint64_t maxDataLength = bufferSize / UB_ALIGN_SIZE * UB_ALIGN_SIZE / sizeof(T); // 32位对齐
    uint64_t countLeft = len;
    GM_ADDR curInput = input;   // 当前输入地址
    GM_ADDR curOutput = output; // 当前输出地址
    int32_t curTag = tag << TAG_MOVE_LEFT_BITS;
    while (countLeft > 0) {
        uint64_t curCount = countLeft > maxDataLength ? maxDataLength : countLeft;
        if (rankSize == 2) {
            // 两张卡的时候，只要把root卡的数据复制到另一张卡去
            op.Process2Rank<T>(curInput, curOutput, curCount, curTag, root);
        } else {
            op.Process<T>(curInput, curOutput, curCount, curTag, root);
        }
        uint64_t curSize = curCount * sizeof(T);
        curInput += curSize;
        curOutput += curSize;
        curTag += (curSize + UB_MAX_DATA_SIZE - 1) / UB_MAX_DATA_SIZE + 1;
        countLeft -= curCount;
    }
    op.TailCounter();
}
