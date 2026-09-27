/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TASK_ORCHESTRATE_H__
#define __TASK_ORCHESTRATE_H__

#include "common/aicpu_kfc_def.h"
#include "aicpu_dispatcher.h"

class TaskOrchestrator {
public:
    static HcclResult DoPreSync();
    static HcclResult DoPostSync();

    static HcclResult SelfCpySnd2Win(
        void* sndAddr, u64 dataSize, u64 sndOffset, u64 winOffset, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult SelfCpyRcv2Win(
        void* rcvAddr, u64 dataSize, u64 rcvOffset, u64 winOffset, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult SelfCpyWin2Rcv(
        void* rcvAddr, u64 dataSize, u64 winOffset, u64 rcvOffset, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpySnd2WinP2P(
        void* sndAddr, u32 dstRank, u64 dataSize, u64 sndOffsets, u64 winOffsets, HcclReduceOp opType,
        HcclDataType dataType);
    static HcclResult IpcCpySnd2Win(
        void* sndAddr, u64 dataSize, u64* sndOffsets, u64* winOffsets, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpySnd2Win(
        void* sndAddr, u64 dataSize, u64 srcOffset, u64 dstOffset, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpySnd2Win(
        void* sndAddr, const std::vector<u64>& dataSizes, const std::vector<u64>& sndOffsets, u64* winOffsets,
        HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpySnd2Win(
        void* sndAddr, u64 dataSize, u64* sndOffsets, u64 winOffsets, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpySnd2Win(
        void* sndAddr, u64* dataSize, u64* sndOffsets, u64* winOffsets, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult
    IpcCpyWin2Win(u64* dataSize, u64* winOffsets, HcclReduceOp opType, u64 sendOff, HcclDataType dataType);
    static HcclResult IpcCpyWin2Win(
        const std::vector<u64>& dataSizes, u64 sendOff, const std::vector<u64>& winOffsets, HcclReduceOp opType,
        HcclDataType dataType);
    static HcclResult IpcCpyWin2Rcv(
        void* rcvAddr, u64 dataSize, u64* winOffsets, u64* rcvOffsets, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpyWin2RcvEx(
        void* rcvAddr, u64 dataSize, u64* rcvOffsets, u64 winOffset, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpyWin2RcvEx(
        void* rcvAddr, u64* dataSize, u64* rcvOffsets, u64 winOffset, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpyWin2RcvEx(
        void* rcvAddr, const std::vector<u64>& dataSizes, const std::vector<u64>& rcvOffsets, u64 sendOff,
        HcclReduceOp opType, HcclDataType dataType);
    static HcclResult SelfCpyWin2RcvEx1(
        void* rcvAddr, u64 dataSize, u64 rcvOffset, u64 winOffset, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpyWin2Rcv(
        void* rcvAddr, const std::vector<u64>& dataSizes, u64* winOffsets, const std::vector<u64>& rcvOffsets,
        HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpyWin2Rcv(
        void* rcvAddr, u64 dataSize, u64 winOffsets, u64* rcvOffsets, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpyWin2Rcv(
        void* rcvAddr, u64* dataSize, u64* winOffsets, u64* rcvOffsets, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult IpcCpyWin2RcvP2P(
        void* rcvAddr, u32 srcRank, u64 dataSize, u64 srcOffset, u64 dstOffset, HcclReduceOp opType,
        HcclDataType dataType);
    static HcclResult IpcCpyWin2RcvP2PMainStream(
        void* rcvAddr, u32 srcRank, u64 dataSize, u64 srcOffset, u64 dstOffset, HcclReduceOp opType,
        HcclDataType dataType);
    static HcclResult SelfCpySnd2WinEx(
        u32 mainRankId, void* sndAddr, u64 dataSize, u64 sndOffset, u64 winOffset, HcclReduceOp opType,
        HcclDataType dataType, u32 maxStreamNum);
    static HcclResult SelfCpyWin2RcvEx(
        u32 mainRankId, void* rcvAddr, u64 dataSize, u64 winOffset, u64 rcvOffset, HcclReduceOp opType,
        HcclDataType dataType, u32 maxStreamNum);
    static HcclResult IpcCpySnd2WinEx(
        void* sndAddr, u64 dataSize, u64* sndOffsets, u64* winOffsets, HcclReduceOp opType, HcclDataType dataType,
        u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq);
    static HcclResult IpcCpyWin2RcvEx(
        void* rcvAddr, u64 dataSize, u64* winOffsets, u64* rcvOffsets, HcclReduceOp opType, HcclDataType dataType,
        u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq);
    static HcclResult IpcCpyWin2WinEx(
        u32 mainRankId, u64 dataSize, u64 winOffset, HcclReduceOp opType, HcclDataType dataType, u32 maxStreamNum);
    static HcclResult IpcCpySnd2WinSliceEx(
        void* sndAddr, std::vector<Slice>& dataSlice, u64* winOffsets, HcclReduceOp opType, HcclDataType dataType,
        u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq);
    static HcclResult IpcCpyWin2RcvSliceEx(
        void* rcvAddr, std::vector<Slice>& dataSlice, u64* winOffsets, HcclReduceOp opType, HcclDataType dataType,
        u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq);
    static HcclResult SelfLocalReduce(u64 dataSize, HcclReduceOp opType, HcclDataType dataType);

    static HcclResult LaunchTasks();
    static HcclResult LaunchTasksEx(u32 subStart, u32 subEnd, u32 maxStreamNum);

    static HcclResult ActiveRecordMain(u16 sqId);
    static HcclResult WaitMainStreamFinish(AicpuComContext* ctx);
    static HcclResult WaitFinishWhileLoop(AicpuComContext* ctx);
    static HcclResult DealKfcCommand(AicpuComContext* ctx);
    static HcclResult CheckTaskTimeout(AicpuComContext* ctx, uint64_t startUsec);
    static void PrintTimeOutSqInfo(AicpuComContext* ctx, u64 timeThreshold);
    static HcclResult WorkSpacePrint(AicpuComContext* ctx);
    static void OverflowAddrCheck(AicpuComContext* ctx, uint32_t& overflowFlag, uint32_t sqHead, uint32_t sqTail);

    static HcclResult MainSubPreSync();
    static HcclResult MainSubPreSync(const uint32_t subStream);
    static HcclResult MainSubPreSync(uint32_t mainStream, uint32_t subStart, uint32_t subEnd, uint32_t maxStream);
    static HcclResult MainSubPostSync();
    static HcclResult MainSubPostSync(const uint32_t subStream);
    static HcclResult MainSubPostSync(uint32_t mainStream, uint32_t subStart, uint32_t subEnd, uint32_t maxStream);

    static HcclResult IpcPreSyncEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq);
    static HcclResult IpcPostSyncEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq);
    static HcclResult IpcPreRecordEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq);
    static HcclResult IpcPostRecordEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq);
    static HcclResult IpcPreWaitEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq);
    static HcclResult IpcPostWaitEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq);

    static HcclResult IpcPreSync();
    static HcclResult IpcPostSync();
    static HcclResult IpcPreSyncOnMainStream();
    static HcclResult IpcPostSyncOnMainStream();
    static HcclResult SelfCpySnd2RcvEx(
        void* sndAddr, void* rcvAddr, u64 sndOffsets, u64 rcvOffsets, u64 dataSize, HcclReduceOp opType,
        HcclDataType dataType);
    static HcclResult SelfCpySnd2WinEx1(
        void* sndAddr, u64 dataSize, u64 sndOffset, u64 winOffset, HcclReduceOp opType, HcclDataType dataType,
        u32 maxStreamNum);
    static HcclResult IpcCpyWin2WinP2P(
        u32 srcRank, u64 dataSize, u64 srcOffsets, u64 dstOffsets, HcclReduceOp opType, HcclDataType dataType);
    static HcclResult SelfCpySnd2Rcv(
        void* sndAddr, void* rcvAddr, u64 sndOffsets, u64 rcvOffsets, u64 dataSize, HcclReduceOp opType,
        HcclDataType dataType);
    static bool IsTaskExceptionForHccs(AicpuComContext* ctx);

    static HcclResult AddBarrier(uint32_t mainStream, uint32_t rankId, uint32_t rankNum);
    static HcclResult IsSupportRDMAReduce(HcclCMDType commType, HcclDataType dataType, HcclReduceOp op);
    static HcclResult
    RunConcreteAlgorithm(AivAicpuOpParam* commParam, AivAicpuOpParam* commParamNext, AicpuComContext* ctx);

private:
    static HcclResult SubRecordMain();
    static HcclResult MainWaitSub();
    static void PrintWaitAddr(const AicpuComContext* ctx);
};
#endif
