/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base_v2.h"

using namespace AscendC;

template <typename T>
class AivBatchSendRecvMesh1D : public AivCommBase {
public:
    __aicore__ inline AivBatchSendRecvMesh1D() {}

    __aicore__ inline void
    SendData(uint32_t dataSizePerVolume, uint64_t currDataCount, uint64_t offset, HcclSendRecvItemDevice& item)
    {
        coreIndex = block_idx;
        uint64_t actualDataRate = dataSizePerVolume / sizeof(T);
        uint64_t dataPerCore = currDataCount / coreNumPerRank;
        uint64_t remainder = currDataCount % coreNumPerRank;
        uint64_t innerDispls = 0;
        if (coreIndex < remainder) { // 这部分核需要多处理一个数据
            innerDispls = coreIndex * dataPerCore + coreIndex;
            sendCurCount = dataPerCore + 1;
        } else {
            innerDispls = coreIndex * dataPerCore + remainder;
            sendCurCount = dataPerCore;
        }

        // send 将数据放到自己的cclBuffer，等对端来读
        sendInputOffset = item.bufAddr + (offset + innerDispls) * dataSizePerVolume;
        sendOutputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (innerDispls)*dataSizePerVolume;

        if (sendCurCount > 0) {
            WaitFlag(rank_, item.remoteRank * coreNumPerRank + coreIndex, 0);
            CpGM2GM((__gm__ T*)sendOutputOffset, (__gm__ T*)sendInputOffset, sendCurCount * actualDataRate);
            PipeBarrier<PIPE_ALL>(); // 核内自己的同步
            Record(rank_, item.remoteRank * coreNumPerRank + coreIndex, curTag);
            // 写完之后要设置一个flag，等对应的卡来读，这个flag要能把对端卡区分出来
        }
    }
    __aicore__ inline void Producer(ExtraArgs& extraArgs)
    {
        for (uint64_t i = selfSendRecvCount; i < extraArgs.itemNum; i++) {
            auto& item = extraArgs.sendRecvInfo[i];
            if (item.sendRecvType != HcclSendRecvType::HCCL_SEND) {
                return;
            }
            // 单个任务先根据cclBuffer切片
            uint32_t dataSizePerVolume = item.dataTypeSize;
            uint64_t cclBufferCountPerRank = cclBufferSizePerRank / dataSizePerVolume;
            uint64_t processedDataCount = 0;
            uint64_t loopTimes
                = item.count / cclBufferCountPerRank + static_cast<uint64_t>(item.count % cclBufferCountPerRank != 0);
            for (uint64_t loop = 0; loop < loopTimes; loop++) {
                uint64_t currDataCount
                    = (loop == loopTimes - 1) ? item.count - processedDataCount : cclBufferCountPerRank;
                // 开始具体的发送任务
                SendData(dataSizePerVolume, currDataCount, processedDataCount, extraArgs.sendRecvInfo[i]);
                processedDataCount += currDataCount;
            }
        }
    }

    __aicore__ inline void
    RecvData(uint32_t dataSizePerVolume, uint64_t currDataCount, uint64_t offset, HcclSendRecvItemDevice& item)
    {
        coreIndex = block_idx - coreNumPerRank;
        uint64_t actualDataRate = dataSizePerVolume / sizeof(T);
        uint64_t dataPerCore = currDataCount / coreNumPerRank;
        uint64_t remainder = currDataCount % coreNumPerRank;
        uint64_t innerDispls = 0;
        if (coreIndex < remainder) { // 这部分核需要多处理一个数据
            innerDispls = coreIndex * dataPerCore + coreIndex;
            recvCurCount = dataPerCore + 1;
        } else {
            innerDispls = coreIndex * dataPerCore + remainder;
            recvCurCount = dataPerCore;
        }

        // recv去读对端cclBuffer的数据
        recvInputOffset = reinterpret_cast<uint64_t>(GM_IN[item.remoteRank]) + innerDispls * dataSizePerVolume;
        recvOutputOffset = item.bufAddr + (offset + innerDispls) * dataSizePerVolume;

        if (recvCurCount > 0) {
            WaitFlag(item.remoteRank, rank_ * coreNumPerRank + coreIndex, curTag);
            CpGM2GM((__gm__ T*)recvOutputOffset, (__gm__ T*)recvInputOffset, recvCurCount * actualDataRate);
            PipeBarrier<PIPE_ALL>(); // 核内自己的同步
            Record(item.remoteRank, rank_ * coreNumPerRank + coreIndex, 0);
        }
    }

    __aicore__ inline void Consumer(ExtraArgs& extraArgs)
    {
        for (uint64_t i = selfSendRecvCount + firstRecvIdx; i < extraArgs.itemNum; i++) {
            auto& item = extraArgs.sendRecvInfo[i];
            if (item.sendRecvType != HcclSendRecvType::HCCL_RECV) {
                return;
            }
            // 单个任务先根据cclBuffer切片
            uint32_t dataSizePerVolume = item.dataTypeSize;
            uint64_t cclBufferCountPerRank = cclBufferSizePerRank / dataSizePerVolume;
            uint64_t processedDataCount = 0;
            uint64_t loopTimes
                = item.count / cclBufferCountPerRank + static_cast<uint64_t>(item.count % cclBufferCountPerRank != 0);
            for (uint64_t loop = 0; loop < loopTimes; loop++) {
                uint64_t currDataCount
                    = (loop == loopTimes - 1) ? item.count - processedDataCount : cclBufferCountPerRank;
                // 开始具体的发送任务
                RecvData(dataSizePerVolume, currDataCount, processedDataCount, extraArgs.sendRecvInfo[i]);
                processedDataCount += currDataCount;
            }
        }
    }

    __aicore__ inline void SelfSendRecv(ExtraArgs& extraArgs)
    {
        // 自收发应该不用考虑cclBuffer的大小
        // 确定第几个任务开始收
        firstRecvIdx = 0;
        for (uint64_t i = 0; i < extraArgs.itemNum; i++) {
            auto& item = extraArgs.sendRecvInfo[i];
            if (item.sendRecvType == HcclSendRecvType::HCCL_RECV) {
                firstRecvIdx = i;
                break;
            }
        }

        if (firstRecvIdx == 0) {
            return; // 说明没有自收发任务
        }

        selfSendRecvCount = 0; // 当前已经处理了多少个自收发任务
        coreNumPerRank = numBlocks_;
        coreIndex = block_idx;

        while (extraArgs.sendRecvInfo[selfSendRecvCount].remoteRank == rank_
               && extraArgs.sendRecvInfo[selfSendRecvCount].sendRecvType == HcclSendRecvType::HCCL_SEND) {
            // 自收发可以用上自己所有的核

            uint64_t actualDataRate = extraArgs.sendRecvInfo[selfSendRecvCount].dataTypeSize / sizeof(T);
            uint64_t dataPerCore = extraArgs.sendRecvInfo[selfSendRecvCount].count / coreNumPerRank;
            uint64_t remainder = extraArgs.sendRecvInfo[selfSendRecvCount].count % coreNumPerRank;
            // 数据对不齐的情况
            uint64_t innerDispls = 0;
            if (coreIndex < remainder) { // 这部分核需要多处理一个数据
                innerDispls = coreIndex * dataPerCore + coreIndex;
                sendRecvCurCount = dataPerCore + 1;
            } else {
                innerDispls = coreIndex * dataPerCore + remainder;
                sendRecvCurCount = dataPerCore;
            }

            sendInput = extraArgs.sendRecvInfo[selfSendRecvCount].bufAddr
                        + innerDispls * extraArgs.sendRecvInfo[selfSendRecvCount].dataTypeSize;
            recvOutput = extraArgs.sendRecvInfo[selfSendRecvCount + firstRecvIdx].bufAddr
                         + innerDispls * extraArgs.sendRecvInfo[selfSendRecvCount + firstRecvIdx].dataTypeSize;
            if (sendRecvCurCount > 0) {
                CpGM2GM((__gm__ T*)recvOutput, (__gm__ T*)sendInput, sendRecvCurCount * actualDataRate);
                PipeBarrier<PIPE_ALL>(); // 核内自己的同步
            }

            selfSendRecvCount++;
        }
    }

    __aicore__ inline void Process(uint32_t tag, ExtraArgs& extraArgs)
    {
        uint32_t rankNum = 2;
        curTag = static_cast<int32_t>(tag);

        // 先去做自收发
        SelfSendRecv(extraArgs);

        // 接下来，一半的核send，一半的核recv
        coreNumPerRank = numBlocks_ / rankNum;
        if (coreNumPerRank < 1) { // 单核情况，暂不处理
            return;
        }

        if (block_idx >= coreNumPerRank * rankNum) {
            return;
        }

        cclBufferSizePerRank
            = len_; // 每一次send，recv的dataType都可能不同，所以这里应该把dataSize发过来，这里再自己去拼装
        if (block_idx < coreNumPerRank) {
            // 先把自己的flag清空
            // 每个核，都可能需要对其他rank发数据，所以要先清理处对应的位置,一共rankSize * coreNumPerRank 个位置
            for (uint64_t i = 0; i < rankSize_; i++) {
                Record(rank_, i * coreNumPerRank + block_idx, 0); // 位置会不会不够用
            }
            PipeBarrier<PIPE_ALL>();

            Producer(extraArgs);
        } else {
            Consumer(extraArgs);
        }
    }

    int32_t curTag;
    uint32_t coreCount;
    uint32_t coreNumPerRank;
    uint32_t targetRank;
    uint32_t coreIndex;
    uint64_t sendInputOffset;
    uint64_t sendOutputOffset;
    uint64_t sendCurCount;
    uint64_t recvInputOffset;
    uint64_t recvOutputOffset;
    uint64_t recvCurCount;
    uint64_t cclBufferCountPerRank;

    uint32_t firstRecvIdx;
    uint32_t selfSendRecvCount;
    uint64_t sendRecvCurCount;
    uint64_t sendInput;
    uint64_t recvOutput;
    uint64_t cclBufferSizePerRank;
};

// 现在executor做好排列：基于这个基础去做aiv的收发
// 先放send，再放recv,自收发的在executor做好校验，这里不再校验了（校验数据量和数据类型是否一一对应）
// 1.sendDeque元素顺序是:先放remoteRank号小于等于root rank的第一个任务，依次减小(循环索引)直至放完
// 2.recvDeque元素顺序是:先放remoteRank号大于等于root rank的第一个任务，依次增大(循环索引)直至放完
// 如果有rank间重复send/recv场景，按照收发数据从大到小排序
template <typename T>
__aicore__ inline void AivBatchSendRecvV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivBatchSendRecvMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    if (block_idx == 0 && tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    op.Process(tag, extraArgs);
    op.BarrierAll();
}

// 先整理出来队列
// 然后先做自收发任务
// 然后收发两个方向，按照各自的方向去做，每次都是占用全量CCL Buffer
// 在这个过程中，根据核的数量，分为几种情况
// 1、单核负责收发：单核场景先不管，当作暂不支持
// 2、单核或者多核负责一个卡的收/发（想象成两条流，先发一个，再发一下，所以去同时给多个核收/发，是没有意义的，要一个一个来）
