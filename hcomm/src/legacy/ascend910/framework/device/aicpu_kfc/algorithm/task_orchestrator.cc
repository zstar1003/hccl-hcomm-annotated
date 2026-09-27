/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_orchestrator.h"
#include <cmath>
#include "common/aicpu_sqe_context.h"
#include "common/aicpu_hccl_common.h"
#include "dfx/mc2_trace_utils.h"
#include "utils/hccl_aicpu_utils.h"
#include "common/aicpu_kfc_utils.h"
#include "framework/aicpu_kfc_prof.h"
#include "log.h"
#include "utils/aicpu_hdc_utils.h"
#include "aicpu_operator_pub.h"
#include "hccl_types.h"
#include "aicpu_allgather.h"
#include "aicpu_reduce_scatter.h"
#include "aicpu_dmy_cal_allreduce.h"
#include "aicpu_allreduce.h"
#include "aicpu_alltoall.h"

using namespace hccl;
namespace {
#define KFC_GET_START_TIME() ((AicpuKfcUtils::NeedRecordTimeTaken(*AicpuGetComContext())) ? GetCurCpuTimestamp() : 0)

#define RECORD_FILL_SQE_TIME(START_TIME)                                                          \
    do {                                                                                          \
        AicpuComContext* commctx__ = AicpuGetComContext();                                        \
        if (!AicpuKfcUtils::NeedRecordTimeTaken(*commctx__)) {                                    \
            break;                                                                                \
        }                                                                                         \
        AicpuKfcProf::GetProInst(*commctx__).fillSqeTimes += GetCurCpuTimestamp() - (START_TIME); \
    } while (0)

#define RECORD_PROF_TIME(VAR)                                                                           \
    do {                                                                                                \
        AicpuComContext* commctx__ = AicpuGetComContext();                                              \
        if (!AicpuKfcUtils::NeedRecordTimeTaken(*commctx__)) {                                          \
            break;                                                                                      \
        }                                                                                               \
        uint32_t recordIndex = AicpuKfcProf::GetProInst(*commctx__).workCnt;                            \
        recordIndex = (recordIndex >= AC_MAX_PROF_COMM_CNT) ? (AC_MAX_PROF_COMM_CNT - 1) : recordIndex; \
        AicpuKfcProf::GetProInst(*commctx__).commLoop[recordIndex].VAR = GetCurCpuTimestamp(true);      \
    } while (0)
} // namespace

HcclResult TaskOrchestrator::DoPreSync()
{
    // 15 sqe on main, 35 sqe on sub
    CHK_RET(MainSubPreSync());

    CHK_RET(IpcPreSync());

    CHK_RET(MainSubPostSync());

    CHK_RET(MainSubPreSync());

    HCCL_INFO("[SQE]Do pre sync on main stream 21 tasks, sub stream 35 tasks");
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::DoPostSync()
{
    // 8 sqe on main, 21 sqe on sub
    CHK_RET(IpcPostSync());

    CHK_RET(MainSubPostSync());

    HCCL_INFO("[SQE]Do post sync on main stream 7 tasks, sub stream 21 tasks");
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpySnd2Win(
    void* sndAddr, u64 dataSize, u64 sndOffset, u64 winOffset, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;

    AicpuComRankInfo* rankInfo = &ctx->rankInfo[rankId];
    void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + sndOffset);
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + winOffset);

    CHK_RET(AicpuDispatcher::CopyData(rankId, src, dst, dataSize, dataType, opType, rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpyRcv2Win(
    void* rcvAddr, u64 dataSize, u64 rcvOffset, u64 winOffset, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;

    AicpuComRankInfo* rankInfo = &ctx->rankInfo[rankId];
    void* src = static_cast<void*>(static_cast<s8*>(rcvAddr) + rcvOffset);
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + winOffset);

    CHK_RET(AicpuDispatcher::CopyData(rankId, src, dst, dataSize, dataType, opType, rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult
TaskOrchestrator::IpcCpyWin2Win(u64* dataSize, u64* winOffsets, HcclReduceOp opType, u64 sendOff, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    AicpuComRankInfo* selfRankInfo = &ctx->rankInfo[ctx->rankId];
    u64 offset = (winOffsets == nullptr) ? 0 : winOffsets[ctx->rankId];
    void* selfWindow = reinterpret_cast<void*>(static_cast<uintptr_t>(selfRankInfo->window));
    void* dst = static_cast<void*>(static_cast<s8*>(selfWindow) + sendOff + offset);
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            void* otherRankWindow = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window));
            void* src = static_cast<void*>(static_cast<s8*>(otherRankWindow) + sendOff + offset);
            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize[ctx->rankId], dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Win(
    const std::vector<u64>& dataSizes, u64 sendOff, const std::vector<u64>& winOffsets, HcclReduceOp opType,
    HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    AicpuComRankInfo* selfRankInfo = &ctx->rankInfo[ctx->rankId];
    u64 offset = winOffsets.empty() ? 0 : winOffsets[ctx->rankId];
    void* selfWindow = reinterpret_cast<void*>(static_cast<uintptr_t>(selfRankInfo->window));
    void* dst = static_cast<void*>(static_cast<s8*>(selfWindow) + sendOff + offset);
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            void* otherRankWindow = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window));
            void* src = static_cast<void*>(static_cast<s8*>(otherRankWindow) + sendOff + offset);
            u64 dataSize = dataSizes.empty() ? 0 : dataSizes[ctx->rankId];
            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);

    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2WinEx(
    u32 mainRankId, u64 dataSize, u64 winOffset, HcclReduceOp opType, HcclDataType dataType, u32 maxStreamNum)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;

    AicpuComRankInfo* mainRankInfo = &ctx->rankInfo[mainRankId];
    AicpuComRankInfo* rankInfo = &ctx->rankInfo[rankId];
    void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + winOffset);
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(mainRankInfo->window) + winOffset);

    CHK_RET(AicpuDispatcher::CopyData(rankId % maxStreamNum, src, dst, dataSize, dataType, opType, mainRankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpySnd2WinEx(
    u32 mainRankId, void* sndAddr, u64 dataSize, u64 sndOffset, u64 winOffset, HcclReduceOp opType,
    HcclDataType dataType, u32 maxStreamNum)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;

    AicpuComRankInfo* rankInfo = &ctx->rankInfo[mainRankId];
    void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + sndOffset);
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + winOffset);

    CHK_RET(AicpuDispatcher::CopyData(rankId % maxStreamNum, src, dst, dataSize, dataType, opType, mainRankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpyWin2Rcv(
    void* rcvAddr, u64 dataSize, u64 winOffset, u64 rcvOffset, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;
    AicpuComRankInfo* rankInfo = &ctx->rankInfo[rankId];

    void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + winOffset);
    void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + rcvOffset);

    CHK_RET(AicpuDispatcher::CopyData(rankId, src, dst, dataSize, dataType, opType, rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpyWin2RcvEx1(
    void* rcvAddr, u64 dataSize, u64 rcvOffset, u64 winOffset, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;
    AicpuComRankInfo* rankInfo = &ctx->rankInfo[rankId];
    void* window = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window));
    void* src = static_cast<void*>(static_cast<s8*>(window) + winOffset + rcvOffset);
    void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + rcvOffset);

    CHK_RET(AicpuDispatcher::CopyData(rankId, src, dst, dataSize, dataType, opType, rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpySnd2WinEx1(
    void* sndAddr, u64 dataSize, u64 sndOffset, u64 winOffset, HcclReduceOp opType, HcclDataType dataType,
    u32 maxStreamNum)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();

    AicpuComRankInfo* rankInfo = &ctx->rankInfo[ctx->rankId];
    void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + sndOffset);
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + winOffset);

    CHK_RET(AicpuDispatcher::CopyData(ctx->rankId % maxStreamNum, src, dst, dataSize, dataType, opType, ctx->rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpySnd2RcvEx(
    void* sndAddr, void* rcvAddr, u64 sndOffsets, u64 rcvOffsets, u64 dataSize, HcclReduceOp opType,
    HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 maxStreamNum = ctx->rankNum;

    void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + sndOffsets);
    void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + rcvOffsets);
    CHK_RET(AicpuDispatcher::CopyData(ctx->rankId % maxStreamNum, src, dst, dataSize, dataType, opType, ctx->rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpyWin2RcvEx(
    u32 mainRankId, void* rcvAddr, u64 dataSize, u64 winOffset, u64 rcvOffset, HcclReduceOp opType,
    HcclDataType dataType, u32 maxStreamNum)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;
    AicpuComRankInfo* rankInfo = &ctx->rankInfo[mainRankId];

    void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + winOffset);
    void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + rcvOffset);

    CHK_RET(AicpuDispatcher::CopyData(rankId % maxStreamNum, src, dst, dataSize, dataType, opType, mainRankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfCpySnd2Rcv(
    void* sndAddr, void* rcvAddr, u64 sndOffsets, u64 rcvOffsets, u64 dataSize, HcclReduceOp opType,
    HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 maxStreamNum = ctx->rankNum;

    void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + sndOffsets);
    void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + rcvOffsets);
    CHK_RET(AicpuDispatcher::CopyData(ctx->rankId % maxStreamNum, src, dst, dataSize, dataType, opType, ctx->rankId));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2Win(
    void* sndAddr, u64 dataSize, u64* sndOffsets, u64* winOffsets, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (sndOffsets == nullptr) ? 0 : sndOffsets[index];
            void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + srcOffset);
            u64 dstOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + dstOffset);
            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2Win(
    void* sndAddr, u64 dataSize, u64 srcOffset, u64 dstOffset, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + srcOffset);
            void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + dstOffset);
            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2Win(
    void* sndAddr, const std::vector<u64>& dataSizes, const std::vector<u64>& sndOffsets, u64* winOffsets,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = sndOffsets.empty() ? 0 : sndOffsets[index];
            void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + srcOffset);
            u64 dstOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(
                index, src, dst, dataSizes.empty() ? 0 : dataSizes[index], dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2Win(
    void* sndAddr, u64 dataSize, u64* sndOffsets, u64 winOffsets, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (sndOffsets == nullptr) ? 0 : sndOffsets[index];
            void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + srcOffset);
            void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + winOffsets);
            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2Win(
    void* sndAddr, u64* dataSize, u64* sndOffsets, u64* winOffsets, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (sndOffsets == nullptr) ? 0 : sndOffsets[index];
            void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + srcOffset);
            u64 dstOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + dstOffset);
            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize[index], dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

// 将本端 snd 发送至对端 window
HcclResult TaskOrchestrator::IpcCpySnd2WinP2P(
    void* sndAddr, u32 dstRank, u64 dataSize, u64 sndOffsets, u64 winOffsets, HcclReduceOp opType,
    HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 selfRank = ctx->rankId;

    void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + sndOffsets);
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(ctx->rankInfo[dstRank].window) + winOffsets);
    // 下发到主流上
    CHK_RET(AicpuDispatcher::CopyData(selfRank, src, dst, dataSize, dataType, opType, dstRank));
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

// 从对端window拷贝到本端window
HcclResult TaskOrchestrator::IpcCpyWin2WinP2P(
    u32 srcRank, u64 dataSize, u64 srcOffsets, u64 dstOffsets, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 selfRank = ctx->rankId;

    void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(ctx->rankInfo[srcRank].window) + srcOffsets);
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(ctx->rankInfo[selfRank].window) + dstOffsets);
    CHK_RET(AicpuDispatcher::CopyData(srcRank, src, dst, dataSize, dataType, opType, srcRank));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Rcv(
    void* rcvAddr, u64 dataSize, u64* winOffsets, u64* rcvOffsets, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + srcOffset);
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvEx(
    void* rcvAddr, u64 dataSize, u64* rcvOffsets, u64 winOffset, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void* window = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window));
            void* src = static_cast<void*>(static_cast<s8*>(window) + winOffset + dstOffset);
            void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvEx(
    void* rcvAddr, u64* dataSize, u64* rcvOffsets, u64 winOffset, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void* window = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window));
            void* src = static_cast<void*>(static_cast<s8*>(window) + winOffset + dstOffset);
            void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize[index], dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvEx(
    void* rcvAddr, const std::vector<u64>& dataSizes, const std::vector<u64>& rcvOffsets, u64 recvOff,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 dstOffset = rcvOffsets.empty() ? 0 : rcvOffsets[index];
            void* window = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window));
            void* src = static_cast<void*>(static_cast<s8*>(window) + recvOff + dstOffset);
            void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);
            u64 dataSize = dataSizes.empty() ? 0 : dataSizes[index];
            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Rcv(
    void* rcvAddr, const std::vector<u64>& dataSizes, u64* winOffsets, const std::vector<u64>& rcvOffsets,
    HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + srcOffset);
            u64 dstOffset = rcvOffsets.empty() ? 0 : rcvOffsets[index];
            void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(
                index, src, dst, dataSizes.empty() ? 0 : dataSizes[index], dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Rcv(
    void* rcvAddr, u64 dataSize, u64 winOffsets, u64* rcvOffsets, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + winOffsets);
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2Rcv(
    void* rcvAddr, u64* dataSize, u64* winOffsets, u64* rcvOffsets, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + srcOffset);
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(index, src, dst, dataSize[index], dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvP2P(
    void* rcvAddr, u32 srcRank, u64 dataSize, u64 srcOffset, u64 dstOffset, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();

    void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(ctx->rankInfo[srcRank].window) + srcOffset);
    void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);
    CHK_RET(AicpuDispatcher::CopyData(srcRank, src, dst, dataSize, dataType, opType, srcRank));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvP2PMainStream(
    void* rcvAddr, u32 srcRank, u64 dataSize, u64 srcOffset, u64 dstOffset, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 selfRank = ctx->rankId;

    void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(ctx->rankInfo[srcRank].window) + srcOffset);
    void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);
    CHK_RET(AicpuDispatcher::CopyData(selfRank, src, dst, dataSize, dataType, opType, srcRank));

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2WinEx(
    void* sndAddr, u64 dataSize, u64* sndOffsets, u64* winOffsets, HcclReduceOp opType, HcclDataType dataType,
    u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 streamId = 0;
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            streamId = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (sndOffsets == nullptr) ? 0 : sndOffsets[index];
            void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + srcOffset);
            u64 dstOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(streamId, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvEx(
    void* rcvAddr, u64 dataSize, u64* winOffsets, u64* rcvOffsets, HcclReduceOp opType, HcclDataType dataType,
    u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 streamId = 0;
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            streamId = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + srcOffset);
            u64 dstOffset = (rcvOffsets == nullptr) ? 0 : rcvOffsets[index];
            void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(streamId, src, dst, dataSize, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpySnd2WinSliceEx(
    void* sndAddr, std::vector<Slice>& dataSlice, u64* winOffsets, HcclReduceOp opType, HcclDataType dataType,
    u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 streamId = 0;
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            streamId = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = dataSlice[index].offset;
            void* src = static_cast<void*>(static_cast<s8*>(sndAddr) + srcOffset);
            u64 dstOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(streamId, src, dst, dataSlice[index].size, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcCpyWin2RcvSliceEx(
    void* rcvAddr, std::vector<Slice>& dataSlice, u64* winOffsets, HcclReduceOp opType, HcclDataType dataType,
    u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 streamId = 0;
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            streamId = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            AicpuComRankInfo* rankInfo = &ctx->rankInfo[index];
            u64 srcOffset = (winOffsets == nullptr) ? 0 : winOffsets[index];
            void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + srcOffset);
            u64 dstOffset = dataSlice[index].offset;
            void* dst = static_cast<void*>(static_cast<s8*>(rcvAddr) + dstOffset);

            CHK_RET(AicpuDispatcher::CopyData(streamId, src, dst, dataSlice[index].size, dataType, opType, index));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::SelfLocalReduce(u64 dataSize, HcclReduceOp opType, HcclDataType dataType)
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    u32 rankId = ctx->rankId;
    AicpuComRankInfo* rankInfo = &ctx->rankInfo[rankId];
    void* src = nullptr;
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window));
    u64 srcOffset = 0LU;
    u64 cpySize = 0LU;

    u32 rankNum = ctx->rankNum;
    u32 power = static_cast<u32>(log2(rankNum));
    u32 rankPower = static_cast<u32>(pow(2, power));
    if (rankPower < rankNum) {
        srcOffset = rankPower * dataSize;
        cpySize = (rankNum - rankPower) * dataSize;
        HCCL_DEBUG(
            "SelfLocalReduce: rankNum %u, power %u, rankPower %u, srcOffset %lu, cpySize %lu", rankNum, power,
            rankPower, srcOffset, cpySize);
        src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + srcOffset);
        CHK_RET(AicpuDispatcher::CopyData(rankId, src, dst, cpySize, dataType, opType, rankId));
    }

    for (u32 round = 0u; round < power; round++) {
        u32 sliceNum = rankPower / static_cast<u32>(pow(2, round + 1));
        srcOffset = sliceNum * dataSize;
        cpySize = srcOffset;
        HCCL_DEBUG(
            "SelfLocalReduce: sliceNum %u, rankNum %u, power %u, rankPower %u, srcOffset %lu", sliceNum, rankNum, power,
            rankPower, srcOffset);
        src = reinterpret_cast<void*>(static_cast<uintptr_t>(rankInfo->window) + srcOffset);
        CHK_RET(AicpuDispatcher::CopyData(rankId, src, dst, cpySize, dataType, opType, rankId));
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::LaunchTasks()
{
    auto ctx = AicpuGetComContext();
    return LaunchTasksEx(0, ctx->rankNum - 1, ctx->rankNum);
}

HcclResult TaskOrchestrator::LaunchTasksEx(u32 subStart, u32 subEnd, u32 maxStreamNum)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = GetCurCpuTimestamp();
    auto ctx = AicpuGetComContext();
    if (AicpuKfcUtils::NeedRecordTimeTaken(*ctx)) {
        RECORD_PROF_TIME(sendTaskStartTime);
    }
    u32 activeRank = ctx->rankId % maxStreamNum;

    /* 两阶段模式，主流待正式执行时再下 */
    /* 一阶段第一次，可以先下主流 */
    if (ctx->directlySendMainSteramSqe) {
        CHK_PRT_RET(
            ActiveRecordMain(activeRank) != HCCL_SUCCESS, HCCL_ERROR("launch task failed, sqid:%u", activeRank),
            HCCL_E_INTERNAL);
    }

    auto profInst = AicpuKfcProf::GetProInst(*ctx);
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != activeRank) {
            if (AicpuKfcUtils::NeedRecordTimeTaken(*ctx)) {
                profInst.fillSqeCnt += GetSqeContext()->buffPtr[index].sqeCnt;
            }
            CHK_PRT_RET(
                AicpuDispatcher::LaunchTask(index) != HCCL_SUCCESS, HCCL_ERROR("launch task failed, sqid:%u", index),
                HCCL_E_INTERNAL);
        }
    }
    HCCL_INFO("LaunchTasksEx sqeBufferLocal, subStart=%u, subEnd=%u", subStart, subEnd);

    if (AicpuKfcUtils::NeedRecordTimeTaken(*ctx)) {
        const u64 endTime = GetCurCpuTimestamp();
        profInst.sendSqeTimes += endTime - startTime;
        profInst.sendSqeBatch += ctx->rankNum;
        RECORD_PROF_TIME(sendSqeFinishTime);
    }
    return HCCL_SUCCESS;
}

// 主流notify从流 从流wait主流
HcclResult TaskOrchestrator::MainSubPreSync()
{
    auto ctx = AicpuGetComContext();
    return MainSubPreSync(ctx->rankId, 0U, ctx->rankNum - 1U, ctx->rankNum);
}

HcclResult TaskOrchestrator::MainSubPreSync(const uint32_t subStream)
{
    auto ctx = AicpuGetComContext();
    return MainSubPreSync(ctx->rankId, subStream, subStream, ctx->rankNum);
}

HcclResult TaskOrchestrator::MainSubPreSync(uint32_t mainStream, uint32_t subStart, uint32_t subEnd, uint32_t maxStream)
{
    if (maxStream == 0U) {
        HCCL_ERROR("Max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != mainStream) {
            CHK_RET(AicpuDispatcher::SignalRecord(
                mainStream % maxStream, index, AicpuDispatcher::NO_IPC, AicpuDispatcher::PRE_SYNC));
            CHK_RET(AicpuDispatcher::SignalWait(
                index % maxStream, index, AicpuDispatcher::NO_IPC, AicpuDispatcher::PRE_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

// 从流notify主流 主流wait从流
HcclResult TaskOrchestrator::MainSubPostSync()
{
    auto ctx = AicpuGetComContext();
    return MainSubPostSync(ctx->rankId, 0U, ctx->rankNum - 1U, ctx->rankNum);
}

HcclResult TaskOrchestrator::MainSubPostSync(const uint32_t subStream)
{
    auto ctx = AicpuGetComContext();
    return MainSubPostSync(ctx->rankId, subStream, subStream, ctx->rankNum);
}

HcclResult
TaskOrchestrator::MainSubPostSync(uint32_t mainStream, uint32_t subStart, uint32_t subEnd, uint32_t maxStream)
{
    if (maxStream == 0U) {
        HCCL_ERROR("Max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    for (uint32_t index = subStart; index <= subEnd; index++) {
        if (index != mainStream) {
            CHK_RET(AicpuDispatcher::SignalRecord(
                index % maxStream, index, AicpuDispatcher::NO_IPC, AicpuDispatcher::POST_SYNC));
            CHK_RET(AicpuDispatcher::SignalWait(
                mainStream % maxStream, index, AicpuDispatcher::NO_IPC, AicpuDispatcher::POST_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPreSync()
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            CHK_RET(AicpuDispatcher::SignalRecord(index, index, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC));
            CHK_RET(AicpuDispatcher::SignalWait(index, index, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPreRecordEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(AicpuDispatcher::SignalRecord(stream_id, index, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPreWaitEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(AicpuDispatcher::SignalWait(stream_id, index, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPreSyncEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(AicpuDispatcher::SignalRecord(stream_id, index, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC));
            CHK_RET(AicpuDispatcher::SignalWait(stream_id, index, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPreSyncOnMainStream()
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            CHK_RET(AicpuDispatcher::SignalRecord(ctx->rankId, index, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC));
            CHK_RET(AicpuDispatcher::SignalWait(ctx->rankId, index, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPostSyncOnMainStream()
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            CHK_RET(
                AicpuDispatcher::SignalRecord(ctx->rankId, index, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC));
            CHK_RET(AicpuDispatcher::SignalWait(ctx->rankId, index, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPostSync()
{
    const u64 startTime = KFC_GET_START_TIME();
    auto ctx = AicpuGetComContext();
    for (u32 index = 0; index < ctx->rankNum; index++) {
        if (index != ctx->rankId) {
            CHK_RET(AicpuDispatcher::SignalRecord(index, index, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC));
            CHK_RET(AicpuDispatcher::SignalWait(index, index, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC));
        }
    }
    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPostRecordEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(AicpuDispatcher::SignalRecord(stream_id, index, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPostWaitEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(AicpuDispatcher::SignalWait(stream_id, index, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::IpcPostSyncEx(u32 subStart, u32 subEnd, u32 maxStreamNum, bool onMainSq)
{
    if (maxStreamNum == 0) {
        HCCL_ERROR("max stream num can not be zero");
        return HCCL_E_PARA;
    }
    const u64 startTime = KFC_GET_START_TIME();
    u32 stream_id = 0;
    auto ctx = AicpuGetComContext();
    for (u32 index = subStart; index <= subEnd; index++) {
        if (index != ctx->rankId) {
            stream_id = (onMainSq == true) ? (ctx->rankId % maxStreamNum) : (index % maxStreamNum);
            CHK_RET(AicpuDispatcher::SignalRecord(stream_id, index, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC));
            CHK_RET(AicpuDispatcher::SignalWait(stream_id, index, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC));
        }
    }

    RECORD_FILL_SQE_TIME(startTime);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::ActiveRecordMain(u16 sqId)
{
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo* streamInfo = &ctx->streamInfo[sqId];
    HCCL_DEBUG(
        "ActiveStream rankId:%d, devId:%d, sqId:%lu, sqeCnt:%d", sqId, ctx->devId, streamInfo->sqId,
        GetSqeContext()->buffPtr[sqId].sqeCnt);
    if (GetSqeContext()->buffPtr[sqId].sqeCnt == 0U) {
        return HCCL_SUCCESS;
    }
    if (AicpuKfcUtils::NeedRecordTimeTaken(*ctx)) {
        AicpuKfcProf::GetProInst(*ctx).fillSqeCnt += GetSqeContext()->buffPtr[sqId].sqeCnt;
    }
    CHK_PRT_RET(
        AicpuDispatcher::LaunchTask(sqId) != HCCL_SUCCESS, HCCL_ERROR("Launch task failed, sqid:%u", sqId),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::WaitMainStreamFinish(AicpuComContext* ctx)
{
    s32 sqId = ctx->streamInfo[ctx->rankId].sqId;
    HCCL_INFO("Start WaitMainStreamFinish..devId = %d rankId:%u, sqid:%d", ctx->devId, ctx->rankId, sqId);

    auto ret = WaitFinishWhileLoop(ctx);
    if (ret != HCCL_SUCCESS) {
        if (ret != HCCL_E_SUSPENDING) {
            HCCL_ERROR("WaitFinishWhileLoop failed, determinism %u, ret %u.", ctx->determinism, ret);
        }
        return ret;
    }
    HCCL_INFO("End WaitMainStreamFinish..devId = %d rankid:%u, sqid:%d", ctx->devId, ctx->rankId, sqId);

    return HCCL_SUCCESS;
}

bool TaskOrchestrator::IsTaskExceptionForHccs(AicpuComContext* ctx)
{
    if (ctx->dfxExtendInfo.cqeStatus != dfx::CqeStatus::kCqeException) {
        return false;
    }

    // NOTE: 需要task exception补全dfx能力，定位故障task的remote rank;
    // 目前暂不具备识别是否跨片的能力，默认失败的task均为跨片操作。
    if (ctx->dfxExtendInfo.cqeException.sqeType == RT_STARS_SQE_TYPE_WRITE_VALUE
        || ctx->dfxExtendInfo.cqeException.sqeType == RT_STARS_SQE_TYPE_SDMA) {
        return true;
    }
    return false;
}

HcclResult TaskOrchestrator::DealKfcCommand(AicpuComContext* ctx)
{
    KfcCommand cmd = KfcCommand::kNone;
    CHK_RET(AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd));
    if (cmd == KfcCommand::kStopLaunch) {
        HCCL_WARNING("hccl aicpu stop wait finish, for recv stop launch cmd");
        return HCCL_E_SUSPENDING;
    } else if ((cmd == KfcCommand::NsStopLaunch) && (ctx->commOpenStatus == true) && (ctx->endStopLaunch == false)) {
        HCCL_WARNING("Stop launch due to receiving stop launch cmd.");
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), true);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), true);
        return HCCL_E_SUSPENDING;
    } else if (cmd == KfcCommand::kDestroyComm) {
        HCCL_WARNING("hccl aicpu stop wait finish, for recv destroy comm cmd");
        return HCCL_E_SUSPENDING;
    } else if (cmd == KfcCommand::kExit) {
        HCCL_ERROR("hccl aicpu stop wait finish, for recv exit cmd.");
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::WaitFinishWhileLoop(AicpuComContext* ctx)
{
    static uint32_t logHead = UINT32_MAX;
    static uint32_t logTail = UINT32_MAX;
    const uint64_t startUsec = GetCurCpuTimestamp();

    int32_t sqId = ctx->streamInfo[ctx->rankId].sqId;
    uint32_t sqHead = 0;
    uint32_t sqTail = 0;
    CHK_RET(QuerySqStatusByType(ctx->devId, sqId, DRV_SQCQ_PROP_SQ_TAIL, sqTail));
    uint32_t loopCnt = 0;
    ctx->sendCntRecord[1] = AicpuKfcUtils::GetSendCnt(ctx); // 1 记录下发完任务后的sendCnt
    ctx->recvCntRecord[1] = AicpuKfcUtils::GetRecvCnt(ctx); // 1 记录下发完任务后的recvCnt
    do {
        if (ctx->dfxExtendInfo.pollStatus == PollStatus::kStopAsException) {
            if (IsTaskExceptionForHccs(ctx)) {
                HCCL_WARNING("hccl aicpu stop wait task exec finish, for task exception.");
                return HCCL_E_SUSPENDING;
            } else {
                HCCL_ERROR("hccl aicpu exec failed, for task exception.");
                return HCCL_E_INTERNAL;
            }
        }

        CHK_RET(DealKfcCommand(ctx));
        CHK_RET(QuerySqStatusByType(ctx->devId, sqId, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
        if (loopCnt > 10000) { // 10000 is max loop cnt
            uint32_t overflowFlag = 0;
            OverflowAddrCheck(ctx, overflowFlag, sqHead, sqTail);
            loopCnt = 0;
            if (logHead != sqHead || logTail != sqTail) {
                logHead = sqHead;
                logTail = sqTail;
                HCCL_INFO("Current state. devId:%u sqid:%d, head:%u, tail:%u", ctx->devId, sqId, sqHead, sqTail);
            }
            CHK_RET(WorkSpacePrint(ctx));
        }
        CHK_RET(CheckTaskTimeout(ctx, startUsec));
        HCCL_INFO(
            "Current state. loopCnt:%u, devId:%u sqid:%d, head:%u, tail:%u", loopCnt, ctx->devId, sqId, sqHead, sqTail);
        loopCnt++;
    } while (sqHead != sqTail);
    return HCCL_SUCCESS;
}

void TaskOrchestrator::PrintTimeOutSqInfo(AicpuComContext* ctx, u64 timeThreshold)
{
    uint32_t status = 0U;
    int32_t sqId = ctx->streamInfo[ctx->rankId].sqId;
    auto ret = QuerySqStatusByType(ctx->devId, sqId, DRV_SQCQ_PROP_SQ_CQE_STATUS, status);
    if (ret != 0) {
        HCCL_ERROR("QuerySqStatusByType status failed. ret = %u sqid:%d", ret, sqId);
    }
    for (uint32_t i = 0U; i < ctx->rankNum; i++) {
        uint32_t sqHead = 0U;
        uint32_t sqTail = 0U;
        (void)QuerySqStatus(ctx->devId, ctx->streamInfo[i].sqId, sqHead, sqTail);
        SqeInfo sqeInfo;
        auto headRet = AicpuSqeContext::QuerySqeInfoByHead(i, sqHead, &sqeInfo);
        if (headRet != HCCL_SUCCESS) {
            HCCL_ERROR("QuerySqeInfoByHead status failed. ret = %u sqHead:%d", headRet, sqHead);
            continue;
        }
        HCCL_ERROR(
            "KFC timeout..[%lu]s, commId %s, stream %u sqid %d head %u tail %u. SqeInfo:%s", timeThreshold, ctx->hcomId,
            i, ctx->streamInfo[i].sqId, sqHead, sqTail, AicpuSqeContext::GetString(sqeInfo).c_str());
    }
}

HcclResult TaskOrchestrator::CheckTaskTimeout(AicpuComContext* ctx, uint64_t startUsec)
{
    const uint64_t sqeTimeoutSec = ctx->dfxExtendInfo.dfxTimeOutConfig.sqeWaitTimeOut;
    if (GetCurCpuTimestamp() - startUsec > static_cast<uint64_t>(NSEC_PER_SEC) * sqeTimeoutSec) {
        PrintTimeOutSqInfo(ctx, sqeTimeoutSec);
        CHK_RET(MC2TraceUtils::Save());
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), DfxKfcStatus::kTimeOut);
        return HCCL_E_TIMEOUT;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskOrchestrator::WorkSpacePrint(AicpuComContext* ctx)
{
    static int staticSndCnt = -1;
    uint64_t waitAddr = ctx->workSpaceAddr + ctx->notifyOff;
    int sndCnt = static_cast<int>((reinterpret_cast<AivAicpuOpParam*>(waitAddr))->sendCnt);
    if (staticSndCnt != sndCnt) {
        staticSndCnt = sndCnt;
        std::stringstream recordLog;
        recordLog << "waitAddr:0x" << std::hex << waitAddr << ", sendCnt:" << std::dec << sndCnt;
        HCCL_INFO("%s", recordLog.str().c_str());
        CHK_RET(MC2TraceUtils::Submit(recordLog.str().c_str()));
    }

    static int staticRcvCnt = -1;
    uint64_t recordAddr = ctx->workSpaceAddr + ctx->notifyOff + ctx->notifyBeginCnt * sizeof(uint8_t) * AC_SQE_SIZE;
    int rcvCnt = static_cast<int>((reinterpret_cast<AivAicpuOpParam*>(recordAddr))->rcvCnt);
    if (staticRcvCnt != rcvCnt) {
        staticRcvCnt = rcvCnt;
        std::stringstream recordLog;
        recordLog << "recordAddr:0x" << std::hex << recordAddr << ", rcvCnt:" << std::dec << rcvCnt;
        HCCL_INFO("%s", recordLog.str().c_str());
        CHK_RET(MC2TraceUtils::Submit(recordLog.str().c_str()));
    }
    return HCCL_SUCCESS;
}

void TaskOrchestrator::OverflowAddrCheck(AicpuComContext* ctx, uint32_t& overflowFlag, uint32_t sqHead, uint32_t sqTail)
{
    if (ctx->devType != DevType::DEV_TYPE_310P1 && ctx->devType != DevType::DEV_TYPE_310P3) {
        return;
    }

    if (ctx->overflowAddr == 0) {
        return;
    }

    uint32_t overflowValTmp = *reinterpret_cast<uint32_t*>(ctx->overflowAddr);
    if ((overflowFlag == 0) && ((overflowValTmp & 0x11) == 0x11)) { // 与runtime对齐，溢出时会给该地址里填写0x11
        HCCL_WARNING(
            "data is overflow, sqHead cur head:%u tail:%u, overflowVal:%u overflowValTmp:%u", sqHead, sqTail,
            overflowFlag, overflowValTmp);
        overflowFlag = 1;
    }
}

HcclResult TaskOrchestrator::AddBarrier(uint32_t mainStream, uint32_t rankId, uint32_t rankNum)
{
    const uint32_t preRankId = (rankId + rankNum - 1U) % rankNum;
    const uint32_t postRankId = (rankId + 1U) % rankNum;
    // 片间同步 notify后卡 wait前卡
    auto ret = AicpuDispatcher::SignalRecord(mainStream, postRankId, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Add notify post rank failed"), ret);
    ret = AicpuDispatcher::SignalWait(mainStream, preRankId, AicpuDispatcher::IPC, AicpuDispatcher::PRE_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Add wait pre rank failed"), ret);
    // 片间同步 notify前卡 wait后卡
    ret = AicpuDispatcher::SignalRecord(mainStream, preRankId, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Add notify pre rank failed"), ret);
    ret = AicpuDispatcher::SignalWait(mainStream, postRankId, AicpuDispatcher::IPC, AicpuDispatcher::POST_SYNC);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Add wait post rank failed"), ret);
    return ret;
}

HcclResult TaskOrchestrator::IsSupportRDMAReduce(HcclCMDType commType, HcclDataType dataType, HcclReduceOp op)
{
    static const std::set<HcclCMDType> multiThreadComTypeWhiteList = {
        HcclCMDType::HCCL_CMD_BATCH_WRITE,
    };
    if (HcclAicpuUtils::GetBlockNum() > 1U
        && multiThreadComTypeWhiteList.find(commType) == multiThreadComTypeWhiteList.end()) {
        HCCL_ERROR("Unsupported comm type %u with multi threads.", commType);
        return HCCL_E_PARA;
    }

    if (commType != HcclCMDType::HCCL_CMD_ALLREDUCE && commType != HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        return HCCL_SUCCESS;
    }
    static const std::set<HcclDataType> dtypeWhiteList
        = {HCCL_DATA_TYPE_FP32,  HCCL_DATA_TYPE_FP16,  HCCL_DATA_TYPE_INT8,
           HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_BFP16};
    if (dtypeWhiteList.find(dataType) == dtypeWhiteList.end()) {
        HCCL_ERROR("Unsupported datatype %s for comm type %u.", GetDataTypeEnumStr(dataType).c_str(), commType);
        return HCCL_E_PARA;
    }

    static const std::set<HcclReduceOp> reduceTypeWhiteList = {HCCL_REDUCE_SUM, HCCL_REDUCE_MAX, HCCL_REDUCE_MIN};
    if (reduceTypeWhiteList.find(op) == reduceTypeWhiteList.end()) {
        HCCL_ERROR("Unsupported reduce op %s.", GetReduceOpEnumStr(op).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult
TaskOrchestrator::RunConcreteAlgorithm(AivAicpuOpParam* commParam, AivAicpuOpParam* commParamNext, AicpuComContext* ctx)
{
    void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(commParam->sendBuffer));
    void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(commParam->recvBuffer));
    RECORD_PROF_TIME(hccExecStartTime);

    const bool waitFlag
        = ((ctx->devType != DevType::DEV_TYPE_310P1 && ctx->devType != DevType::DEV_TYPE_310P3)
           && ctx->commAlg == COMM_ALG_FULL_MESH);
    HCCL_DEBUG(
        "startRunAlg src:%p, dst:%p, ctx commType:%d, commParam commType:%d, waitFlag:%u.", src, dst, ctx->commType,
        commParam->commType, static_cast<u32>(waitFlag));
    if (waitFlag) {
        CHK_RET(AicpuDispatcher::AddWaitStartTaskOnMainStream(ctx->rankId));
    }

    HcclResult result = HCCL_SUCCESS;
    switch (ctx->commType) {
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER: {
            CHK_RET(IsSupportRDMAReduce(commParam->commType, commParam->hcclDataType, commParam->opType));
            u64 strideLen = (commParam->strideLen != 0) ? commParam->strideLen : commParam->count / ctx->rankNum;
            AicpuReduceScatter reduceScatter(ctx);
            result = reduceScatter.RunAlgorithm(
                commParam->opType, src, dst, commParam->count, commParam->hcclDataType, strideLen);
            break;
        }
        case HcclCMDType::HCCL_CMD_ALLGATHER: {
            u64 strideLen = (commParam->strideLen != 0) ? commParam->strideLen : commParam->count;
            AicpuAllgather allgather(ctx);
            result = allgather.RunAlgorithm(
                commParam->opType, src, dst, commParam->count, commParam->hcclDataType, strideLen, commParamNext);
            break;
        }
        case HcclCMDType::HCCL_CMD_ALLREDUCE: {
            CHK_RET(IsSupportRDMAReduce(commParam->commType, commParam->hcclDataType, commParam->opType));
            if (ctx->determinism) {
                u64 strideLen = (commParam->strideLen != 0) ? commParam->strideLen : commParam->count;
                AicpuDmyCalAllreduce dmyCalAllreduce(ctx);
                result = dmyCalAllreduce.RunAlgorithm(
                    commParam->opType, src, dst, commParam->count, commParam->hcclDataType, strideLen, commParamNext);
            } else {
                AicpuAllreduce allreduce(ctx);
                result = allreduce.RunAlgorithm(commParam->opType, src, dst, commParam->count, commParam->hcclDataType);
            }
            break;
        }
        case HcclCMDType::HCCL_CMD_ALLTOALL: {
            u64 strideLen = (commParam->strideLen != 0) ? commParam->strideLen : commParam->count;
            AicpuAllToAll allToAll(ctx);
            result = allToAll.RunAlgorithm(
                commParam->opType, src, dst, commParam->count, commParam->hcclDataType, strideLen);
            break;
        }
        default: {
            HCCL_ERROR("commType [%d] is not supported.", commParam->commType);
            result = HCCL_E_PARA;
            break;
        }
    }

    ctx->curTurnCnt++;
    HCCL_DEBUG("addEndTask, curTurnCnt:%u, totalTurnCnt:%u", ctx->curTurnCnt, ctx->totalTurnCnt);
    if (waitFlag) {
        CHK_RET(AicpuDispatcher::AddExecEndTaskOnMainStream(ctx->rankId));
    }
    return result;
}
