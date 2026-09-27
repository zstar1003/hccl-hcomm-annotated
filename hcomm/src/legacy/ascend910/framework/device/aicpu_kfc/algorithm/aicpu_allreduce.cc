/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_allreduce.h"
#include <cmath>
#include <algorithm>
#include "common/aicpu_hccl_common.h"

HcclResult AicpuAllreduce::RunAlgorithm(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType,
    [[maybe_unused]] u64 strideLen, AivAicpuOpParam* /* nextTask */)
{
    CHK_PTR_NULL(ctx_);
    if (CC_EXE_ONE_SHOT_8_STREAM == ctx_->commOpType) {
        return RunAllReduceReduceBcast(opType, sendBuffer, recvBuffer, dataCount * ctx_->unitSize, dataType);
    } else if (ctx_->commOpType == CC_EXE_ONE_SHOT_1_STREAM) {
        return RunAllReduceOneShot1Stream(opType, sendBuffer, recvBuffer, dataCount * ctx_->unitSize, dataType);
    } else if (ctx_->commOpType == CC_EXE_TWO_SHOT_1_STREAM) {
        return RunAllReduceTwoShot1Stream(opType, sendBuffer, recvBuffer, dataCount, dataType);
    } else if (ctx_->commOpType == CC_EXE_ONE_SHOT_HD) {
        return RunAllReduceOneshotHD(opType, sendBuffer, recvBuffer, dataCount * ctx_->unitSize, dataType);
    } else if (ctx_->commOpType == CC_EXE_ONE_SHOT_SINGLE_RING) {
        return RunAllReduceRing(opType, sendBuffer, recvBuffer, dataCount, dataType);
    }

    if (ctx_->useBufferType == MC2_BUFFER_TYPE_WINDOW_IN) {
        if (HCCL_SUCCESS == RunAllReduceAlignWin2Win(opType, recvBuffer, dataCount, dataType)) {
            return HCCL_SUCCESS;
        }
    } else {
        if (HCCL_SUCCESS == RunAllReduceAlign(opType, sendBuffer, recvBuffer, dataCount, dataType)) {
            return HCCL_SUCCESS;
        }
    }

    return RunAllReduce(opType, sendBuffer, recvBuffer, dataCount, dataType);
}

int64_t AicpuAllreduce::RoundUpWithDivisor(u64 value, u64 divisor) const
{
    if ((value == 0) || (divisor == 0)) {
        return divisor;
    }
    // divisor必须大于等于1, 返回value向上取divisor的整数倍的值
    return ((value + (divisor - 1)) / divisor) * divisor;
}

HcclResult
AicpuAllreduce::PrepareSlice(u64 dataCount, HcclDataType dataType, u32 sliceNum, std::vector<Slice>& dataSlice) const
{
    Slice temp;
    u32 unitSize = DataUnitSize(dataType);
    u64 totalSize = dataCount * unitSize;
    dataSlice.clear();
    dataSlice.reserve(sliceNum);
    if (sliceNum == 0) {
        HCCL_ERROR("[Prepare][SliceData]data slice prepare, sliceNum is 0.");
        return HCCL_E_PARA;
    }
    u64 sizePerSlice = (totalSize + sliceNum - 1) / sliceNum; /* 1是为了向上取整 */
    sizePerSlice = RoundUpWithDivisor(sizePerSlice, HCCL_MIN_SLICE_ALIGN);
    u64 residueSize = totalSize;
    u32 i = 0;
    while (residueSize > 0) {
        u64 sliceSize = sizePerSlice < residueSize ? sizePerSlice : residueSize;
        temp.size = sliceSize;
        temp.offset = totalSize - residueSize;
        i++;
        if (sliceSize <= 0) {
            HCCL_ERROR("[Prepare][SliceData]data_slice_prepare sliceSize[%llu]", sliceSize);
            return HCCL_E_PARA;
        }
        residueSize -= sliceSize;
        dataSlice.push_back(temp);
    }
    while (i < sliceNum) {
        temp.size = 0;
        temp.offset = totalSize;
        i++;
        dataSlice.push_back(temp);
    }
    return HCCL_SUCCESS;
}

void AicpuAllreduce::GetDataSizes16K(std::vector<u64>& dataSizes, u64 allDataSize) const
{
    u64 num16k = allDataSize / HCCL_COPY_ALIGN;
    u64 tailSize = allDataSize % HCCL_COPY_ALIGN;

    u64 baseNum = num16k / ctx_->rankNum;
    u64 tailNum = num16k % ctx_->rankNum;

    for (u32 i = 0; i < ctx_->rankNum; i++) {
        dataSizes[i] = baseNum * HCCL_COPY_ALIGN;
    }
    for (u32 i = 0; i < tailNum; i++) {
        dataSizes[i] += HCCL_COPY_ALIGN;
    }
    dataSizes[ctx_->rankNum - 1] += tailSize;
}

HcclResult AicpuAllreduce::RunAllReduceAlignWin2Win(
    HcclReduceOp opType, void* recvBuffer, u64 dataCount, HcclDataType dataType) const
{
    u32 unitSize = ctx_->unitSize;
    u64 allDataSize = dataCount * unitSize;
    u8* curOutputPtr = static_cast<u8*>(recvBuffer);

    if (ctx_->rankNum == 0) {
        return HCCL_E_PARA;
    }
    std::vector<u64> dataSizes(ctx_->rankNum, 0);
    GetDataSizes16K(dataSizes, allDataSize);

    if (dataSizes[0] > ctx_->windowSize || dataSizes[ctx_->rankNum - 1] > ctx_->windowSize
        || (ctx_->rankNum - 1) * HCCL_COPY_ALIGN >= allDataSize) {
        return HCCL_E_PARA;
    }

    std::vector<u64> dataOffsets(ctx_->rankNum, 0);
    for (u32 i = 1; i < ctx_->rankNum; i++) {
        dataOffsets[i] = dataOffsets[i - 1] + dataSizes[i - 1];
    }

    u64 winOffset = ctx_->winOffset;

    // 1. 前同步
    TaskOrchestrator::DoPreSync();

    // 2. 跨片SDMA，分批拷贝 + 分批结束同步
    TaskOrchestrator::IpcCpyWin2Win(dataSizes, winOffset, dataOffsets, opType, dataType);

    // 3. 后同步
    TaskOrchestrator::DoPostSync();

    // 4. 前同步
    TaskOrchestrator::DoPreSync();

    // 5. 片内数据 Win拷贝到Rcv
    TaskOrchestrator::SelfCpyWin2RcvEx1(
        curOutputPtr, dataSizes[ctx_->rankId], dataOffsets[ctx_->rankId], winOffset, HCCL_REDUCE_RESERVED, dataType);

    // 6. 跨片SDMA，分批拷贝 + 分批结束同步
    TaskOrchestrator::IpcCpyWin2RcvEx(curOutputPtr, dataSizes, dataOffsets, winOffset, HCCL_REDUCE_RESERVED, dataType);

    // 7. 后同步
    TaskOrchestrator::DoPostSync();

    TaskOrchestrator::LaunchTasks();

    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::RunAllReduceAlign(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType) const
{
    u32 unitSize = ctx_->unitSize;
    u64 allDataSize = dataCount * unitSize;

    u8* curInputPtr = static_cast<u8*>(sendBuffer);
    u8* curOutputPtr = static_cast<u8*>(recvBuffer);

    if (ctx_->rankNum == 0) {
        return HCCL_E_PARA;
    }
    std::vector<u64> dataSizes(ctx_->rankNum, 0);
    GetDataSizes16K(dataSizes, allDataSize);

    if (dataSizes[0] > ctx_->windowSize || dataSizes[ctx_->rankNum - 1] > ctx_->windowSize
        || (ctx_->rankNum - 1) * HCCL_COPY_ALIGN >= allDataSize) {
        return HCCL_E_PARA;
    }

    std::vector<u64> dataOffsets(ctx_->rankNum, 0);
    for (u32 i = 1; i < ctx_->rankNum; i++) {
        dataOffsets[i] = dataOffsets[i - 1] + dataSizes[i - 1];
    }

    // 1. 片内数据 Snd拷贝到Window
    TaskOrchestrator::SelfCpySnd2Win(
        curInputPtr, dataSizes[ctx_->rankId], dataOffsets[ctx_->rankId], 0, HCCL_REDUCE_RESERVED, dataType);

    // 2. 前同步
    TaskOrchestrator::DoPreSync();

    // 3. 跨片SDMA，分批拷贝 + 分批结束同步
    TaskOrchestrator::IpcCpySnd2Win(curInputPtr, dataSizes, dataOffsets, nullptr, opType, dataType);

    // 4. 后同步
    TaskOrchestrator::DoPostSync();

    // 5. 前同步
    TaskOrchestrator::DoPreSync();

    // 6. 片内数据 Win拷贝到Rcv
    TaskOrchestrator::SelfCpyWin2Rcv(
        curOutputPtr, dataSizes[ctx_->rankId], 0, dataOffsets[ctx_->rankId], HCCL_REDUCE_RESERVED, dataType);

    // 7. 跨片SDMA，分批拷贝 + 分批结束同步
    TaskOrchestrator::IpcCpyWin2Rcv(curOutputPtr, dataSizes, nullptr, dataOffsets, HCCL_REDUCE_RESERVED, dataType);

    // 8. 后同步
    TaskOrchestrator::DoPostSync();

    TaskOrchestrator::LaunchTasks();

    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::RunAllReduce(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType) const
{
    u64 windowSize = ctx_->windowSize; // window size default is 200M, maybe need read from cfg/env.
    u32 unitSize = ctx_->unitSize;
    u64 maxCountPerLoop = (windowSize / unitSize) * ctx_->rankNum; // 中转内存单次最多能够接受的output count

    u8* curInputPtr = static_cast<u8*>(sendBuffer);
    u8* curOutputPtr = static_cast<u8*>(recvBuffer);
    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft = dataCount;

    u64 dataSlice[AC_MAX_RANK_NUM] = {0};
    u64 sliceSize[AC_MAX_RANK_NUM] = {0};
    if (ctx_->rankNum <= 0) {
        return HCCL_E_UNAVAIL;
    }

    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte
        u64 curRankCnt = curCount / ctx_->rankNum;
        for (u32 i = 0; i < ctx_->rankNum; i++) {
            dataSlice[i] = i * curRankCnt * unitSize;
            sliceSize[i] = curRankCnt * unitSize;
        }
        sliceSize[ctx_->rankNum - 1] += (curCount - curRankCnt * ctx_->rankNum) * unitSize;

        HCCL_DEBUG(
            "RunAllReducev:curInputPtr[%p], curOutputPtr[%p], curCount[%llu], curSize[%llu]", curInputPtr, curOutputPtr,
            curCount, curSize);

        if (ctx_->useBufferType != MC2_BUFFER_TYPE_WINDOW_IN) {
            RunAllReduceSlice(curOutputPtr, curInputPtr, sliceSize, dataSlice, opType, dataType);
        } else {
            RunAllReduceSliceWin2Win(curOutputPtr, sliceSize, dataSlice, opType, dataType);
        }

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }

    return HCCL_SUCCESS;
}

void AicpuAllreduce::RunAllReduceSliceWin2Win(
    u8* curOutputPtr, u64* sliceSize, u64* dataSlice, HcclReduceOp opType, HcclDataType dataType) const
{
    u64 winOffset = ctx_->winOffset;
    // 1. 前同步
    TaskOrchestrator::DoPreSync();

    // 2. 跨片SDMA，分批拷贝 + 分批结束同步
    TaskOrchestrator::IpcCpyWin2Win(sliceSize, dataSlice, opType, winOffset, dataType);

    // 3. 后同步
    TaskOrchestrator::DoPostSync();

    // 4. 前同步
    TaskOrchestrator::DoPreSync();

    // 5. 片内数据 Win拷贝到Rcv
    TaskOrchestrator::SelfCpyWin2RcvEx1(
        curOutputPtr, sliceSize[ctx_->rankId], dataSlice[ctx_->rankId], winOffset, HCCL_REDUCE_RESERVED, dataType);

    // 6. 跨片SDMA，分批拷贝 + 分批结束同步
    TaskOrchestrator::IpcCpyWin2RcvEx(curOutputPtr, sliceSize, dataSlice, winOffset, HCCL_REDUCE_RESERVED, dataType);

    // 7. 后同步
    TaskOrchestrator::DoPostSync();

    TaskOrchestrator::LaunchTasks();
}

void AicpuAllreduce::RunAllReduceSlice(
    u8* curOutputPtr, u8* curInputPtr, u64* sliceSize, u64* dataSlice, HcclReduceOp opType, HcclDataType dataType) const
{
    // 1. 片内数据 Snd拷贝到Window
    TaskOrchestrator::SelfCpySnd2Win(
        curInputPtr, sliceSize[ctx_->rankId], dataSlice[ctx_->rankId], 0, HCCL_REDUCE_RESERVED, dataType);

    // 2. 前同步
    TaskOrchestrator::DoPreSync();

    // 3. 跨片SDMA，分批拷贝 + 分批结束同步
    TaskOrchestrator::IpcCpySnd2Win(curInputPtr, sliceSize, dataSlice, nullptr, opType, dataType);

    // 4. 后同步
    TaskOrchestrator::DoPostSync();

    // 5. 前同步
    TaskOrchestrator::DoPreSync();

    // 6. 片内数据 Win拷贝到Rcv
    TaskOrchestrator::SelfCpyWin2Rcv(
        curOutputPtr, sliceSize[ctx_->rankId], 0, dataSlice[ctx_->rankId], HCCL_REDUCE_RESERVED, dataType);

    // 7. 跨片SDMA，分批拷贝 + 分批结束同步
    TaskOrchestrator::IpcCpyWin2Rcv(curOutputPtr, sliceSize, nullptr, dataSlice, HCCL_REDUCE_RESERVED, dataType);

    // 8. 后同步
    TaskOrchestrator::DoPostSync();

    TaskOrchestrator::LaunchTasks();
}

HcclResult AicpuAllreduce::RunAllReduceOneShot4Stream(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataSize, HcclDataType dataType) const
{
    // 第一轮第一组
    u32 mainRankId = ctx_->rankId;
    u32 maxStreamNum = ctx_->rankNum / 2;
    u32 startRank = 0;
    u32 endRank = maxStreamNum - 1;

    if (mainRankId >= maxStreamNum) {
        startRank = maxStreamNum;
        endRank = ctx_->rankNum - 1;
    }

    // 第1轮
    // 1. 片内数据 拷贝到Window
    TaskOrchestrator::SelfCpySnd2WinEx(
        mainRankId, sendBuffer, dataSize, 0, 0, HCCL_REDUCE_RESERVED, dataType, maxStreamNum);
    TaskOrchestrator::MainSubPreSync(mainRankId, startRank, endRank, maxStreamNum);

    TaskOrchestrator::IpcPreSyncEx(startRank, endRank, maxStreamNum, false);
    // 2. 跨片SDMA 片内Send拷贝到对端Window
    TaskOrchestrator::IpcCpySnd2WinEx(
        sendBuffer, dataSize, nullptr, nullptr, opType, dataType, startRank, endRank, maxStreamNum, false);
    TaskOrchestrator::IpcPostSyncEx(startRank, endRank, maxStreamNum, false);

    TaskOrchestrator::MainSubPostSync(mainRankId, startRank, endRank, maxStreamNum);

    // 第2轮
    u32 remoteRank = (ctx_->rankNum - 1) - mainRankId; // 0-7; 1-6; 2-5; 3-4
    // 3. 片内数据 拷贝到recv
    TaskOrchestrator::SelfCpyWin2RcvEx(
        mainRankId, recvBuffer, dataSize, 0, 0, HCCL_REDUCE_RESERVED, dataType, maxStreamNum);
    TaskOrchestrator::IpcPreSyncEx(remoteRank, remoteRank, maxStreamNum, true);
    // 4. 跨片SDMA Window拷贝到对端Recv
    TaskOrchestrator::IpcCpyWin2RcvEx(
        recvBuffer, dataSize, nullptr, nullptr, opType, dataType, remoteRank, remoteRank, maxStreamNum, true);
    TaskOrchestrator::IpcPostSyncEx(remoteRank, remoteRank, maxStreamNum, true);

    // 5. 下发sqe
    TaskOrchestrator::LaunchTasksEx(0, maxStreamNum - 1, maxStreamNum);

    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::RunReduceBcastOnMainSq(
    u32 mainRankId, u32 maxStreamNum, u32 /* startRank */, u32 /* endRank */, void* sendBuffer, void* recvBuffer,
    u64 dataSize, HcclDataType dataType) const
{
    HCCL_DEBUG("run RunReduceBcastOnMainSq start");
    if (ctx_->useBufferType != MC2_BUFFER_TYPE_WINDOW_IN) {
        // 1. reduce
        TaskOrchestrator::SelfCpySnd2WinEx(
            mainRankId, sendBuffer, dataSize, 0, 0, HCCL_REDUCE_RESERVED, dataType, maxStreamNum);
    }

    // 2. 前同步
    TaskOrchestrator::MainSubPreSync();
    TaskOrchestrator::IpcPreRecordEx(0, maxStreamNum - 1, maxStreamNum, false);
    // 3. 后同步
    TaskOrchestrator::IpcPostWaitEx(0, maxStreamNum - 1, maxStreamNum, true);

    // 4. 前同步
    TaskOrchestrator::MainSubPreSync();
    TaskOrchestrator::IpcPreRecordEx(0, maxStreamNum - 1, maxStreamNum, false);
    // 5. bcast
    TaskOrchestrator::SelfCpyWin2RcvEx(
        mainRankId, recvBuffer, dataSize, ctx_->useBufferType != MC2_BUFFER_TYPE_WINDOW_IN ? 0 : ctx_->winOffset, 0,
        HCCL_REDUCE_RESERVED, dataType, maxStreamNum);
    // 6. 后同步
    TaskOrchestrator::IpcPostWaitEx(0, maxStreamNum - 1, maxStreamNum, true);

    HCCL_DEBUG("run RunReduceBcastOnMainSq end");
    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::RunReduceBcastOnOtherSq(
    HcclReduceOp opType, u32 mainRankId, u32 maxStreamNum, void* sendBuffer, void* recvBuffer, u64 dataSize,
    HcclDataType dataType) const
{
    HCCL_DEBUG("run RunReduceBcastOnOtherSq start");

    // reduce
    TaskOrchestrator::IpcPreWaitEx(mainRankId, mainRankId, maxStreamNum, true);
    if (ctx_->useBufferType != MC2_BUFFER_TYPE_WINDOW_IN) {
        TaskOrchestrator::SelfCpySnd2WinEx(mainRankId, sendBuffer, dataSize, 0, 0, opType, dataType, maxStreamNum);
    } else {
        TaskOrchestrator::IpcCpyWin2WinEx(mainRankId, dataSize, ctx_->winOffset, opType, dataType, maxStreamNum);
    }
    TaskOrchestrator::IpcPostRecordEx(mainRankId, mainRankId, maxStreamNum, true);

    // bcast
    TaskOrchestrator::IpcPreWaitEx(mainRankId, mainRankId, maxStreamNum, true);
    TaskOrchestrator::SelfCpyWin2RcvEx(
        mainRankId, recvBuffer, dataSize, ctx_->useBufferType != MC2_BUFFER_TYPE_WINDOW_IN ? 0 : ctx_->winOffset, 0,
        HCCL_REDUCE_RESERVED, dataType, maxStreamNum);
    TaskOrchestrator::IpcPostRecordEx(mainRankId, mainRankId, maxStreamNum, true);
    HCCL_DEBUG("run RunReduceBcastOnOtherSq end");
    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::RunAllReduceReduceBcast(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataSize, HcclDataType dataType) const
{
    HCCL_DEBUG("run RunAllReduceReduceBcast start");

    u32 mainRankId = 0;
    u32 maxStreamNum = ctx_->rankNum;
    u32 startRank = 0;
    u32 endRank = ctx_->rankNum - 1;

    if (ctx_->rankId == mainRankId) {
        RunReduceBcastOnMainSq(
            mainRankId, maxStreamNum, startRank, endRank, sendBuffer, recvBuffer, dataSize, dataType);
    } else {
        RunReduceBcastOnOtherSq(opType, mainRankId, maxStreamNum, sendBuffer, recvBuffer, dataSize, dataType);
    }

    // 下发sqe
    TaskOrchestrator::LaunchTasksEx(0, maxStreamNum - 1, maxStreamNum);

    HCCL_DEBUG("run RunAllReduceReduceBcast end");
    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::RunAllReduceOneShot1Stream(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataSize, HcclDataType dataType) const
{
    HCCL_INFO("run RunAllReduceOneShot1Stream start");
    u32 maxStreamNum = ctx_->rankNum;
    u8* curOutputPtr = static_cast<u8*>(recvBuffer);
    u32 startRank = 0;
    u32 endRank = maxStreamNum - 1;

    TaskOrchestrator::SelfCpySnd2WinEx1(sendBuffer, dataSize, 0, 0, HCCL_REDUCE_RESERVED, dataType, maxStreamNum);

    TaskOrchestrator::SelfCpySnd2RcvEx(sendBuffer, recvBuffer, 0, 0, dataSize, HCCL_REDUCE_RESERVED, dataType);

    TaskOrchestrator::IpcPreSyncEx(startRank, endRank, maxStreamNum, true);

    TaskOrchestrator::IpcCpyWin2RcvEx(
        curOutputPtr, dataSize, nullptr, nullptr, opType, dataType, startRank, endRank, maxStreamNum, true);

    TaskOrchestrator::IpcPostSyncEx(startRank, endRank, maxStreamNum, true);

    // 下发sqe
    TaskOrchestrator::LaunchTasksEx(0, maxStreamNum - 1, maxStreamNum);

    HCCL_INFO("run RunAllReduceOneShot1Stream end");
    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::RunAllReduceTwoShot1Stream(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType) const
{
    HCCL_INFO("run RunAllReduceTwoShot1Stream start");
    u32 maxStreamNum = ctx_->rankNum;
    u32 startRank = 0;
    u32 endRank = maxStreamNum - 1;
    u8* curInputPtr = static_cast<u8*>(sendBuffer);
    u8* curOutputPtr = static_cast<u8*>(recvBuffer);
    u32 unitSize = ctx_->unitSize;
    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 windowSize = ctx_->windowSize; // window size default is 200M, maybe need read from cfg/env.
    u64 maxCountPerLoop = windowSize / unitSize * ctx_->rankNum; // 中转内存单次最多能够接受的output count
    u64 countLeft = dataCount;

    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte
        std::vector<Slice> dataSlice;
        PrepareSlice(curCount, dataType, maxStreamNum, dataSlice);
        u64 sliceSize = dataSlice[ctx_->rankId].size;

        HCCL_INFO(
            "RunAllReducev:curInputPtr[%p], curOutputPtr[%p], curCount[%llu], curSize[%llu]", curInputPtr, curOutputPtr,
            curCount, curSize);

        TaskOrchestrator::SelfCpySnd2WinEx1(
            curInputPtr, sliceSize, dataSlice[ctx_->rankId].offset, 0, HCCL_REDUCE_RESERVED, dataType, maxStreamNum);

        TaskOrchestrator::IpcPreSyncEx(startRank, endRank, maxStreamNum, true);

        TaskOrchestrator::IpcCpySnd2WinSliceEx(
            curInputPtr, dataSlice, nullptr, opType, dataType, startRank, endRank, maxStreamNum, true);

        TaskOrchestrator::IpcCpyWin2RcvSliceEx(
            curOutputPtr, dataSlice, nullptr, HCCL_REDUCE_RESERVED, dataType, startRank, endRank, maxStreamNum, true);

        TaskOrchestrator::IpcPostSyncEx(startRank, endRank, maxStreamNum, true);

        TaskOrchestrator::SelfCpyWin2Rcv(
            curOutputPtr, sliceSize, 0, dataSlice[ctx_->rankId].offset, HCCL_REDUCE_RESERVED, dataType);

        // 下发sqe
        TaskOrchestrator::LaunchTasksEx(0, maxStreamNum - 1, maxStreamNum);

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }

    HCCL_INFO("run RunAllReduceTwoShot1Stream end");
    return HCCL_SUCCESS;
}

// 计算HD算法给定轮中给定rank的对端的rank号
u32 AicpuAllreduce::GetHdPeer(const u32 hdRound, const u32 curRank) const
{
    // 将所有的设备分成若干组，相邻的组之间的对位节点相互通信
    // 每个组中的设备数量为 2^当前轮次，即：1，2，4，8 ....
    u32 groupSize = std::pow(2, hdRound);
    // 获取当前节点在组内的位置 （同时也是对端节点在组内的位置）
    u32 rankOffset = curRank % groupSize;
    // 获取当前节点在第几组
    u32 curGroupIdx = curRank / groupSize;
    // 偶数号组内的节点与下一组的节点通信，对应的，奇数号组内的节点与上一组的节点通信
    u32 peerGroupIdx = (curGroupIdx % 2 == 0) ? curGroupIdx + 1 : curGroupIdx - 1;

    u32 peerRank = peerGroupIdx * groupSize + rankOffset;
    return peerRank;
}

// OneshotHD 算法, 使用了DMA消减
// 只支持卡数大于 2 且为 2 的幂数的场景
// datasize 需小于 ccl buffer 大小
HcclResult AicpuAllreduce::RunAllReduceOneshotHD(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataSize, HcclDataType dataType) const
{
    /* 分3个阶段：
     * 第一阶段：将 input 的数据拷贝到 window 上
     * 第二阶段：将 input 数据发送到对端 window 进行 reduce 并将 reduce 完的
     * window 拷贝到 output
     * 第三阶段：不断将对端 window 数据读取到本端 output，
     * 如果不是最后一轮，则将 reduce 完的 output 拷贝到 window
     */
    HCCL_INFO("run RunAllReduceOneshotHD start");

    u8* curOutputPtr = static_cast<u8*>(recvBuffer);
    const u32 curRank = ctx_->rankId;
    // 第一阶段：
    // 将输入数据拷贝到 Window
    CHK_RET(TaskOrchestrator::SelfCpySnd2Win(sendBuffer, dataSize, 0, 0, HCCL_REDUCE_RESERVED, dataType));
    // 第二阶段：
    // 片内拷贝 片内Send拷贝到对端Window - 卡内双 die 间 allreduce
    u32 hdRound = 0;
    u32 peerRank = GetHdPeer(hdRound, curRank);
    CHK_RET(TaskOrchestrator::IpcPreSyncEx(peerRank, peerRank, ctx_->rankNum, true));

    // 将输入数据写到对端
    CHK_RET(TaskOrchestrator::IpcCpySnd2WinP2P(sendBuffer, peerRank, dataSize, 0, 0, opType, dataType));
    TaskOrchestrator::IpcPostSyncEx(peerRank, peerRank, ctx_->rankNum, true);

    // 片内拷贝 Win拷贝到Rcv
    CHK_RET(TaskOrchestrator::SelfCpyWin2Rcv(curOutputPtr, dataSize, 0, 0, HCCL_REDUCE_RESERVED, dataType));
    // 第三阶段：
    // 循环 log2(rankNum) - 1 次
    u32 remainingHdRounds = ctx_->rankNum >> 1;
    while (remainingHdRounds >>= 1) { // 使用位移代替 log2
        hdRound++;
        peerRank = GetHdPeer(hdRound, curRank);
        // 跨片拷贝 对端 window 拷贝到 rcv
        CHK_RET(TaskOrchestrator::IpcPreSyncEx(peerRank, peerRank, ctx_->rankNum, true));
        CHK_RET(TaskOrchestrator::IpcCpyWin2RcvP2PMainStream(curOutputPtr, peerRank, dataSize, 0, 0, opType, dataType));
        CHK_RET(TaskOrchestrator::IpcPostSyncEx(peerRank, peerRank, ctx_->rankNum, true));
        // 如果这不是最后一轮，则需要将rcv里的数据同步到 win 里
        if (remainingHdRounds > 1) {
            CHK_RET(TaskOrchestrator::SelfCpyRcv2Win(curOutputPtr, dataSize, 0, 0, HCCL_REDUCE_RESERVED, dataType));
        }
    }
    // 下发sqe
    TaskOrchestrator::LaunchTasksEx(0, ctx_->rankNum - 1, ctx_->rankNum);
    HCCL_INFO("run RunAllReduceOneshotHD end");
    return HCCL_SUCCESS;
}

// 将 oriValue 对调整为 alignValue 的倍数
u64 AicpuAllreduce::AlignWith(u64 oriValue, u64 alignValue) const
{
    if (oriValue <= alignValue || alignValue == 0) {
        return oriValue;
    }
    u64 remain = oriValue % alignValue;
    return oriValue - remain;
}

// 按照 cclBuffer 大小将数据切分
HcclResult AicpuAllreduce::GetBurstDataCounts(u64 windowSize, u64 dataCount, std::vector<u64>& burstDataCounts) const
{
    u64 alignedWindowSize = AlignWith(windowSize, HCCL_COPY_ALIGN);
    u32 unitSize = ctx_->unitSize;
    CHK_PRT_RET(unitSize == 0, HCCL_ERROR("UnitSize is 0"), HCCL_E_UNAVAIL);
    u32 maxDataPerBurst = alignedWindowSize / unitSize;
    CHK_PRT_RET(maxDataPerBurst == 0, HCCL_ERROR("maxDataPerBurst is 0"), HCCL_E_UNAVAIL);
    burstDataCounts.insert(burstDataCounts.end(), dataCount / maxDataPerBurst, maxDataPerBurst);
    u64 tailSize = dataCount % maxDataPerBurst;
    if (tailSize != 0) {
        burstDataCounts.push_back(tailSize);
    }
    return HCCL_SUCCESS;
}

std::vector<std::vector<u32>> AicpuAllreduce::GetRingOrders() const
{
    // simple ring
    std::vector<std::vector<u32>> ringOrders;
    std::vector<u32> ringOrder(ctx_->rankNum);
    for (u32 i = 0; i < ctx_->rankNum; i++) {
        ringOrder[i] = i;
    }
    ringOrders.push_back(ringOrder);
    return ringOrders;
}

// 当前只支持从0开始的rankID
HcclResult AicpuAllreduce::reorderRingSlice(
    const std::vector<u32>& ringOrder, const std::vector<Slice>& ringSlices,
    std::vector<Slice>& orderedRingSlices) const
{
    size_t ringSize = ringOrder.size();
    CHK_PRT_RET(ringSize == 0, HCCL_ERROR("ringSize is 0"), HCCL_E_UNAVAIL);
    orderedRingSlices.resize(ringSize);
    for (size_t rankIdx = 0; rankIdx < ringSize; rankIdx++) {
        u32 currentRank = ringOrder[rankIdx];
        u32 previousRankIdx = (rankIdx + ringSize - 1) % ringSize;
        u32 previousRank = ringOrder[previousRankIdx];
        Slice currentSlice = ringSlices[previousRank];
        orderedRingSlices[currentRank] = currentSlice;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::PrepareRingSlice(
    const std::vector<std::vector<u32>>& ringOrders, u64 dataCount, HcclDataType dataType,
    std::vector<std::vector<Slice>>& orderedAllRingSlice) const
{
    u32 ringNum = ringOrders.size();
    u32 sliceNum = ctx_->rankNum * ringNum;
    std::vector<Slice> dataSlices;
    PrepareSlice(dataCount, dataType, sliceNum, dataSlices);
    std::vector<std::vector<Slice>> allRingSlices(ringNum);
    orderedAllRingSlice.resize(ringNum);

    for (size_t i = 0; i < dataSlices.size(); i++) {
        allRingSlices[i % ringNum].push_back(dataSlices[i]);
    }
    for (size_t i = 0; i < ringNum; i++) {
        reorderRingSlice(ringOrders[i], allRingSlices[i], orderedAllRingSlice[i]);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::RingIPCPreSync(const u32 stream, const u32 prevRank, const u32 nextRank) const
{
    // 通知下游
    CHK_RET(AicpuDispatcher::SignalRecord(stream, nextRank, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC));
    // 等待上游通知
    CHK_RET(AicpuDispatcher::SignalWait(stream, prevRank, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC));
    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::RingIPCPostSync(const u32 stream, const u32 prevRank, const u32 nextRank) const
{
    // 回复上游通知
    CHK_RET(AicpuDispatcher::SignalRecord(stream, prevRank, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC));
    // 等待下游回复，回收 notify
    CHK_RET(AicpuDispatcher::SignalWait(stream, nextRank, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC));
    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::GetPrevRankList(const std::vector<u32>& ringOrder, std::vector<u32>& previousRankList) const
{
    size_t ringSize = ringOrder.size();
    for (size_t rankIdx = 0; rankIdx < ringSize; rankIdx++) {
        u32 currRank = ringOrder[rankIdx];
        u32 previousRankIdx = (rankIdx + ringSize - 1) % ringSize;
        previousRankList[currRank] = ringOrder[previousRankIdx];
    }
    return HCCL_SUCCESS;
}

size_t AicpuAllreduce::FindNextRank(const std::vector<u32>& previousRankList, const u32 localRank) const
{
    return std::find(previousRankList.begin(), previousRankList.end(), localRank) - previousRankList.begin();
}

Slice* AicpuAllreduce::GetNextRingSlice(
    const std::vector<u32>& previousRankList, std::vector<Slice>& orderedRingSlices, u32& curSliceIdx) const
{
    curSliceIdx = previousRankList[curSliceIdx];
    Slice* nextSlice = &orderedRingSlices[curSliceIdx];
    return nextSlice;
}

HcclResult AicpuAllreduce::RunAllReduceRingAlg(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, std::vector<Slice>& orderedRingSlices,
    std::vector<u32>& ringOrder, HcclDataType dataType) const
{
    size_t ringSize = ringOrder.size();
    std::vector<u32> previousRankList(ringSize);
    // 计算每一个rank在ring环上的前一个rank
    CHK_RET(GetPrevRankList(ringOrder, previousRankList));
    const u32 prevRank = previousRankList[ctx_->rankId];
    const u32 nextRank = FindNextRank(previousRankList, ctx_->rankId);
    const u32 subStream = prevRank;
    u32 curSliceIdx = ctx_->rankId;
    Slice* localSlice = &orderedRingSlices[curSliceIdx];
    Slice* remoteSlice = nullptr;
    // 第一轮：主流准备前2片数据
    CHK_RET(TaskOrchestrator::SelfCpySnd2Win(
        sendBuffer, localSlice->size, localSlice->offset, localSlice->offset, HCCL_REDUCE_RESERVED, dataType));
    localSlice = GetNextRingSlice(previousRankList, orderedRingSlices, curSliceIdx);
    CHK_RET(TaskOrchestrator::SelfCpySnd2Win(
        sendBuffer, localSlice->size, localSlice->offset, localSlice->offset, HCCL_REDUCE_RESERVED, dataType));
    for (size_t rankOffset = 0; rankOffset < ringSize - 1; rankOffset++) {
        remoteSlice = localSlice;
        // 从流开始 reduce 操作
        CHK_RET(TaskOrchestrator::MainSubPreSync(subStream)); // 主流启动从流
        // 从流跨片 reduce
        CHK_RET(RingIPCPreSync(subStream, prevRank, nextRank));
        CHK_RET(TaskOrchestrator::IpcCpyWin2WinP2P(
            prevRank, remoteSlice->size, remoteSlice->offset, remoteSlice->offset, opType, dataType));
        CHK_RET(RingIPCPostSync(subStream, prevRank, nextRank));
        if (rankOffset < ringSize - 2) { // ringSize - 2： 最后一轮不进行本地搬运
            // 主流继续准备数据
            localSlice = GetNextRingSlice(previousRankList, orderedRingSlices, curSliceIdx);
            CHK_RET(TaskOrchestrator::SelfCpySnd2Win(
                sendBuffer, localSlice->size, localSlice->offset, localSlice->offset, HCCL_REDUCE_RESERVED, dataType));
        }
        CHK_RET(TaskOrchestrator::MainSubPostSync(subStream)); // 从流通知主流，回收 notify，主流继续执行
        TaskOrchestrator::LaunchTasksEx(0, ctx_->rankNum - 1, ctx_->rankNum); // 下发sqe
    }
    for (size_t rankOffset = 0; rankOffset < ringSize - 1; rankOffset++) {
        localSlice = remoteSlice; // 主流搬运上一轮从流准备的数据
        remoteSlice = GetNextRingSlice(
            previousRankList, orderedRingSlices, curSliceIdx); // 从流继续往下循环，搬运reduce好的数据
        CHK_RET(TaskOrchestrator::MainSubPreSync(subStream));  // 主流通知从流回收notify资源
        TaskOrchestrator::SelfCpyWin2Rcv(
            recvBuffer, localSlice->size, localSlice->offset, localSlice->offset, HCCL_REDUCE_RESERVED, dataType);
        CHK_RET(RingIPCPreSync(subStream, prevRank, nextRank));
        if (rankOffset < ringSize - 2) { // < RingSize - 2: 不是最后一轮，搬到 window 上，让下游读
            CHK_RET(TaskOrchestrator::IpcCpyWin2WinP2P(
                prevRank, remoteSlice->size, remoteSlice->offset, remoteSlice->offset, HCCL_REDUCE_RESERVED, dataType));
        } else { // 最后一轮，dma 消减，直接从对端读入 rcv
            CHK_RET(TaskOrchestrator::IpcCpyWin2RcvP2P(
                recvBuffer, prevRank, remoteSlice->size, remoteSlice->offset, remoteSlice->offset, HCCL_REDUCE_RESERVED,
                dataType));
        }
        CHK_RET(RingIPCPostSync(subStream, prevRank, nextRank));
        CHK_RET(TaskOrchestrator::MainSubPostSync(subStream));                // 通知主流继续
        TaskOrchestrator::LaunchTasksEx(0, ctx_->rankNum - 1, ctx_->rankNum); // 下发sqe
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuAllreduce::RunAllReduceRingSingleBurst(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType,
    std::vector<std::vector<u32>>& ringOrders) const
{
    std::vector<std::vector<Slice>> orderedRingSlices;
    PrepareRingSlice(ringOrders, dataCount, dataType, orderedRingSlices);
    for (size_t i = 0; i < ringOrders.size(); i++) {
        CHK_RET(RunAllReduceRingAlg(opType, sendBuffer, recvBuffer, orderedRingSlices[i], ringOrders[i], dataType));
    }
    return HCCL_SUCCESS;
}

// 单 ring 算法, 使用了DMA消减
HcclResult AicpuAllreduce::RunAllReduceRing(
    HcclReduceOp opType, void* sendBuffer, void* recvBuffer, u64 dataCount, HcclDataType dataType) const
{
    HCCL_INFO("run RunAllReduceRing start");
    // 数据准备阶段
    std::vector<u64> burstDataCounts;
    u64 windowSize = ctx_->windowSize;
    u32 unitSize = ctx_->unitSize;
    CHK_RET(GetBurstDataCounts(windowSize, dataCount, burstDataCounts));
    u8* currSendBuffer = static_cast<u8*>(sendBuffer);
    u8* currRecvBuffer = static_cast<u8*>(recvBuffer);
    u64 burstSize;
    std::vector<std::vector<u32>> ringOrders = GetRingOrders();
    for (u64 burstDataCount : burstDataCounts) {
        burstSize = burstDataCount * unitSize;
        CHK_RET(
            RunAllReduceRingSingleBurst(opType, currSendBuffer, currRecvBuffer, burstDataCount, dataType, ringOrders));
        currSendBuffer += burstSize;
        currRecvBuffer += burstSize;
    }
    HCCL_INFO("run RunAllReduceRing end");
    return HCCL_SUCCESS;
}
