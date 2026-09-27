/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_sqe_context.h"

#include <sstream>
#include <unordered_map>
#include "common/aicpu_hccl_common.h"
#include "utils/hccl_aicpu_utils.h"
#include "common/aicpu_kfc_utils.h"

struct SqeContextVariable {
    int32_t lastClusterId = -1;
    SqeLocalRingBuffer* variablePtr = nullptr;
};

static SqeLocalRingBuffer g_ringBuffer[CLUSTER_CNT][AC_MAX_RANK_NUM];
static SqeContext g_sqeContext[CLUSTER_CNT];
static SqeContextVariable g_sqeVariable;

SqeContext* GetSqeContext() { return &g_sqeContext[HcclAicpuUtils::GetCurClusterId()]; }

void AicpuSqeContext::InitSqeContext()
{
    for (uint32_t i = 0U; i < CLUSTER_CNT; i++) {
        SqeContext* context = &g_sqeContext[i];
        context->buffPtr = g_ringBuffer[i];
        (void)memset_s(
            context->buffPtr, sizeof(SqeLocalRingBuffer[AC_MAX_RANK_NUM]), 0,
            sizeof(SqeLocalRingBuffer[AC_MAX_RANK_NUM]));
        context->clusterId = i;
    }
}

void AicpuSqeContext::SyncVariable()
{
    SqeContext* context = GetSqeContext();
    HCCL_DEBUG(
        "SyncCtxVariable, cur clusterId %d, last ClusterId %d, buffPtr %p", context->clusterId,
        g_sqeVariable.lastClusterId, g_sqeVariable.variablePtr);
    if (context->clusterId == g_sqeVariable.lastClusterId) {
        return;
    }
    if (g_sqeVariable.lastClusterId < 0 || g_sqeVariable.lastClusterId >= CLUSTER_CNT) {
        HCCL_DEBUG("SyncCtxVariable, invalid lastClusterId = %d", g_sqeVariable.lastClusterId);
        return;
    }
    context->buffPtr = g_sqeVariable.variablePtr;
}

void AicpuSqeContext::SaveVariable()
{
    SqeContext* context = GetSqeContext();
    HCCL_DEBUG("Save sqe context variable, cur clusterId=%d, buffPtr=%p", context->clusterId, context->buffPtr);
    g_sqeVariable.lastClusterId = context->clusterId;
    g_sqeVariable.variablePtr = context->buffPtr;
}

HcclResult AicpuSqeContext::GetNextSqeBufferAddr(
    uint32_t streamId, uint8_t*& sqeBufferAddr, uint8_t*& sqeTypeAddr, uint16_t& taskId)
{
    CHK_PRT_RET(
        (streamId >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[AicpuSqeContext][GetNextSqeBufferAddr]Invalid streamId[%u] >= %u", streamId, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    SqeContext* context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    auto& buff = context->buffPtr[streamId];
    if (buff.tailSqeIdx >= AC_SQE_MAX_CNT) {
        HCCL_WARNING("Sqe cnt is overflow, need revise buff content, current streamid: %u", streamId);
        HCCL_INFO(
            "buffer modify before ==> sqTail: %u, sqHead: %u, sqeCnt: %u, tailSqeTaskId: %u, tailSqeIdx: %u",
            buff.sqTail, buff.sqHead, buff.sqeCnt, buff.tailSqeTaskId, buff.tailSqeIdx);
        CHK_RET(AicpuKfcUtils::TraceProfSubmit());
        CHK_RET(AicpuSqeContext::ModifyBuffer(streamId));
        HCCL_INFO(
            "buffer modify after ==> sqTail: %u, sqHead: %u, sqeCnt: %u, tailSqeTaskId: %u, tailSqeIdx: %u",
            buff.sqTail, buff.sqHead, buff.sqeCnt, buff.tailSqeTaskId, buff.tailSqeIdx);
    }
    // nextTaskId=0的时候下发PlaceHolder
    if (UNLIKELY(buff.tailSqeTaskId == 0 && buff.filpNum != 0)) {
        CHK_RET(AddFlipTask(streamId));
    }

    buff.profTimestap[buff.tailSqeIdx] = GetCurCpuTimestamp(true);
    sqeBufferAddr = buff.localBuff + buff.tailSqeIdx * AC_SQE_SIZE;
    sqeTypeAddr = &buff.sqeType[buff.tailSqeIdx];
    taskId = buff.tailSqeTaskId;
    HCCL_DEBUG(
        "Get stream:%u next idx:%u, taskId:%u, clusterId:%u", streamId, buff.tailSqeIdx, taskId, context->clusterId);
    if (buff.tailSqeTaskId == UINT16_MAX) {
        buff.filpNum++;
        HCCL_WARNING("Sqe context cur taskId is uint16_max");
    }
    buff.tailSqeTaskId++;
    buff.tailSqeIdx++;
    buff.sqeCnt++;
    return HCCL_SUCCESS;
}

HcclResult AicpuSqeContext::AddFlipTask(uint32_t streamId)
{
    if (!dfx::ProfilingManager::GetProfL0State()) {
        return HCCL_SUCCESS;
    }
    SqeContext* context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    auto& buff = context->buffPtr[streamId];
    uint16_t filpNum = buff.filpNum;
    uint16_t taskId = buff.tailSqeTaskId;
    auto ctx = AicpuGetComContext();
    HcclComStreamInfo* streamInfo = &ctx->streamInfo[streamId];
    CHK_RET(dfx::ProfilingManager::ReportFilpTask(streamInfo->actualStreamId, taskId, filpNum));

    buff.profTimestap[buff.tailSqeIdx] = GetCurCpuTimestamp(true);
    uint8_t* sqeBufferAddr = buff.localBuff + buff.tailSqeIdx * AC_SQE_SIZE;
    uint8_t* sqeTypeAddr = &buff.sqeType[buff.tailSqeIdx];
    AicpuAddOneFlipPlaceHolderSqe addOneFlipPlaceHolderSqe = AicpuGetAddOneFlipPlaceHolderSqe();
    if (addOneFlipPlaceHolderSqe == nullptr) {
        HCCL_WARNING("AicpuAddOneFlipPlaceHolderSqe is null");
        return HCCL_SUCCESS;
    }
    addOneFlipPlaceHolderSqe(streamInfo->actualStreamId, filpNum, taskId, sqeBufferAddr, sqeTypeAddr);
    buff.tailSqeTaskId++;
    buff.tailSqeIdx++;
    buff.sqeCnt++;

    HCCL_INFO(
        "[AicpuSqeContext][AddFlipTask] Call AddFlipTask. para: taskId[%u], streamId[%u], flipNum[%u]]", taskId,
        streamInfo->actualStreamId, filpNum);

    return HCCL_SUCCESS;
}

HcclResult AicpuSqeContext::RecordAddInfo(uint32_t streamId, uint32_t addInfo)
{
    CHK_PRT_RET(
        (streamId >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[AicpuSqeContext][RecordAddInfo]Invalid streamId[%u] >= %u", streamId, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    SqeContext* context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    auto& buff = context->buffPtr[streamId];
    CHK_PRT_RET(
        ((buff.tailSqeIdx == 0) || (buff.tailSqeIdx > AC_SQE_MAX_CNT)),
        HCCL_ERROR("[AicpuSqeContext][RecordAddInfo]Invalid tailSqeIdx[%u]", buff.tailSqeIdx), HCCL_E_PARA);
    buff.addInfo[buff.tailSqeIdx - 1] = addInfo;
    return HCCL_SUCCESS;
}

HcclResult AicpuSqeContext::QuerySqeInfoByHead(uint32_t streamId, uint32_t sqHead, SqeInfo* info)
{
    CHK_PRT_RET(
        (streamId >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[AicpuSqeContext][QuerySqeInfoByHead]Invalid streamId[%u] >= %u", streamId, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    CHK_PTR_NULL(info);
    SqeContext* context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    auto& buff = context->buffPtr[streamId];
    const uint32_t sqDepth = AicpuGetComContext()->streamInfo[streamId].sqDepth;
    uint32_t sqUnexecuted = (buff.sqTail + sqDepth - sqHead) % sqDepth;
    if (buff.tailSqeIdx < sqUnexecuted) {
        HCCL_WARNING("tail sqe idx %u is less than sq unexecuted num %u", buff.tailSqeIdx, sqUnexecuted);
        return HCCL_E_INTERNAL;
    }
    uint16_t idx = buff.tailSqeIdx - sqUnexecuted;
    HCCL_INFO("Query streamId:%u, sqeIdx:%u, actual idx:%u, type:%u", streamId, sqHead, idx, buff.sqeType[idx]);
    info->sqeHeadIdx = sqHead;
    return SqeContextUtils::QuerySqeInfo(
        buff.localBuff + idx * AC_SQE_SIZE, buff.sqeType[idx], buff.addInfo[idx], info);
}

HcclResult AicpuSqeContext::QuerySqeInfoByTaskId(uint32_t streamId, uint16_t taskId, SqeInfo* info)
{
    CHK_PRT_RET(
        (streamId >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[AicpuSqeContext][QuerySqeInfoByTaskId]Invalid streamId[%u] >= %u", streamId, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    CHK_PTR_NULL(info);
    SqeContext* context = GetSqeContext();
    CHK_PTR_NULL(context->buffPtr);
    auto& buff = context->buffPtr[streamId];
    uint16_t tailRemain = buff.tailSqeTaskId - taskId;
    const uint32_t sqDepth = AicpuGetComContext()->streamInfo[streamId].sqDepth;
    uint32_t sqHeadIdx = (buff.sqTail + sqDepth - tailRemain) % sqDepth;
    if (buff.tailSqeIdx < tailRemain) {
        HCCL_WARNING("tail sqe idx %u is less than tail remain num %u", buff.tailSqeIdx, tailRemain);
        return HCCL_E_INTERNAL;
    }
    uint16_t idx = buff.tailSqeIdx - tailRemain;
    HCCL_INFO("Query streamId:%u, sqeIdx:%u, actual idx:%u, type:%u", streamId, sqHeadIdx, idx, buff.sqeType[idx]);
    info->sqeHeadIdx = sqHeadIdx;
    return SqeContextUtils::QuerySqeInfo(
        buff.localBuff + idx * AC_SQE_SIZE, buff.sqeType[idx], buff.addInfo[idx], info);
}

HcclResult AicpuSqeContext::ClearCurBuff(uint32_t streamid, uint32_t leftBound)
{
    CHK_PRT_RET(
        (streamid >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[AicpuSqeContext][ClearCurBuff]Invalid streamId[%u] >= %u", streamid, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    SqeContext* context = GetSqeContext();
    auto& buff = context->buffPtr[streamid];
    HCCL_INFO(
        "leftBound:%u, buff.sqeCnt:%u, buff.sqHead:%u, buff.sqTail:%u, buff.tailSqeIdx:%u, buff.tailSqeTaskId:%u",
        leftBound, buff.sqeCnt, buff.sqHead, buff.sqTail, buff.tailSqeIdx, buff.tailSqeTaskId);
    if (memset_s(
            buff.localBuff + leftBound * AC_SQE_SIZE, sizeof(buff.localBuff) - leftBound * AC_SQE_SIZE, 0,
            (buff.tailSqeIdx - leftBound) * AC_SQE_SIZE)
        != EOK) {
        return HCCL_E_MEMORY;
    }
    if (memset_s(buff.sqeType + leftBound, sizeof(buff.sqeType) - leftBound, 0, buff.tailSqeIdx - leftBound) != EOK) {
        return HCCL_E_MEMORY;
    }
    if (memset_s(buff.addInfo + leftBound, sizeof(buff.addInfo) - leftBound, 0, buff.tailSqeIdx - leftBound) != EOK) {
        return HCCL_E_MEMORY;
    }
    buff.sqeCnt = 0;
    buff.tailSqeIdx = 0;
    AicpuGetComContext()->profilingExtendInfo.lastSqeIdxs[streamid] = 0;
    return HCCL_SUCCESS;
}

HcclResult AicpuSqeContext::ModifyBuffer(uint32_t streamid)
{
    CHK_PRT_RET(
        (streamid >= AC_MAX_RANK_NUM),
        HCCL_ERROR("[AicpuSqeContext][ModifyBuffer]Invalid streamId[%u] >= %u", streamid, AC_MAX_RANK_NUM),
        HCCL_E_PARA);
    // 未下发的sqe移到前面
    SqeContext* context = GetSqeContext();
    auto& buff = context->buffPtr[streamid];
    uint32_t cnt = buff.sqeCnt;
    uint32_t leftSrc = buff.tailSqeIdx - buff.sqeCnt;
    HCCL_DEBUG(
        "buff.sqeCnt:%d, buff.sqHead:%u, buff.sqTail:%u, buff.tailSqeIdx:%u, buff.tailSqeTaskId:%u", buff.sqeCnt,
        buff.sqHead, buff.sqTail, buff.tailSqeIdx, buff.tailSqeTaskId);
    if (memmove_s(buff.localBuff, sizeof(buff.localBuff), buff.localBuff + leftSrc * AC_SQE_SIZE, cnt * AC_SQE_SIZE)
        != EOK) {
        return HCCL_E_MEMORY;
    }
    if (memmove_s(buff.sqeType, sizeof(buff.sqeType), buff.sqeType + leftSrc, cnt) != EOK) {
        return HCCL_E_MEMORY;
    }
    if (memmove_s(buff.addInfo, sizeof(buff.addInfo), buff.addInfo + leftSrc, cnt) != EOK) {
        return HCCL_E_MEMORY;
    }
    // 队列后面已经拷贝到rtsq上的sqe清除掉
    CHK_RET(ClearCurBuff(streamid, cnt));
    // 更新index和sqeCnt
    buff.tailSqeIdx = cnt;
    buff.sqeCnt = cnt;
    AicpuGetComContext()->profilingExtendInfo.lastSqeIdxs[streamid] = cnt;
    return HCCL_SUCCESS;
}

HcclResult AicpuSqeContext::ClearLocalBuff()
{
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        CHK_RET(ClearCurBuff(i));
    }
    return HCCL_SUCCESS;
}

std::string AicpuSqeContext::GetString(const SqeInfo& sqeInfo)
{
    std::stringstream ss;
    ss << "SqeInfo ";
    ss << "sqeIdx:" << sqeInfo.sqeHeadIdx << ",";
    ss << "type:" << SqeContextUtils::RtsqTaskTypeToStr(sqeInfo.type) << ",";
    ss << "subType:" << static_cast<uint16_t>(sqeInfo.subType) << ",";
    ss << "streamId:" << sqeInfo.streamId << ",";
    ss << "taskId:" << sqeInfo.taskId << ",";
    ss << "notifyId:" << sqeInfo.notifyId << ",";
    ss << "eventId:" << sqeInfo.eventId << ",";
    ss << "partId:" << sqeInfo.partId << ",";
    ss << "length:" << sqeInfo.length << ",";
    ss << "condValue:" << sqeInfo.condValue << ",";
    ss << "isLast:" << static_cast<uint16_t>(sqeInfo.isLast) << ",";
    ss << "opCode:" << static_cast<uint16_t>(sqeInfo.opCode) << ",";
    ss << "sqeNum:" << static_cast<uint16_t>(sqeInfo.sqeNum) << ",";
    ss << "valid:" << static_cast<uint16_t>(sqeInfo.valid) << ",";
    ss << "addr1High:0x" << std::hex << sqeInfo.addr1High << ",";
    ss << "addr1Low:0x" << std::hex << sqeInfo.addr1Low << ",";
    ss << "addr2High:0x" << std::hex << sqeInfo.addr2High << ",";
    ss << "addr2Low:0x" << std::hex << sqeInfo.addr2Low << ".";
    return ss.str();
}
