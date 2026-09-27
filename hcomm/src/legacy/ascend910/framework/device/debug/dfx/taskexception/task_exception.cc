/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_exception.h"
#include "log.h"
#include "aicpu_hccl_sqcq.h"
#include "sqe_context_utils.h"

namespace hccl {
TaskException::TaskException() {}

TaskException::~TaskException() {}

HcclResult TaskException::Init(u32 devId, u32 localUserRank, const std::string& identifier)
{
    devId_ = devId;
    localUserRank_ = localUserRank;
    identifier_ = identifier;
    HCCL_INFO(
        "%s success, devId[%u], localUserRank[%u], identifier[%s]", __func__, devId_, localUserRank_,
        identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult TaskException::RegisterOpInfo(void* opInfo, u32 size)
{
    CHK_PTR_NULL(opInfo);
    CHK_PRT_RET(
        size == 0 || size > OP_INFO_MAX_SIZE,
        HCCL_ERROR("%s fail, size[%u], expect [1, %u]", __func__, size, OP_INFO_MAX_SIZE), HCCL_E_PARA);

    opRingBufferIdx_ = (opRingBufferIdx_ + 1) % OPINFO_RING_BUFFER_MAX;
    indOpInfos_[opRingBufferIdx_].opIndex = opRingBufferIdx_;
    CHK_SAFETY_FUNC_RET(memcpy_s(indOpInfos_[opRingBufferIdx_].opInfo, size, reinterpret_cast<uint8_t*>(opInfo), size));
    HCCL_DEBUG("%s success, opRingBufferIdx_[%u], opInfo[%p], size[%u]", __func__, opRingBufferIdx_, opInfo, size);
    return HCCL_SUCCESS;
}

HcclResult TaskException::RegisterOpInfoCallback(HcommGetOpInfoCallback callback)
{
    CHK_PTR_NULL(callback);
    indOpInfos_[opRingBufferIdx_].callback = callback;
    HCCL_DEBUG("%s success, opRingBufferIdx_[%u], callback[%p]", __func__, opRingBufferIdx_, callback);
    return HCCL_SUCCESS;
}

bool TaskException::IsRepeatPrint(u32 streamId, u32 opIndex, u32 sqHead)
{
    auto it = threadPrintState_.find(streamId);
    if (it != threadPrintState_.end()) {
        return it->second.first == opIndex && it->second.second == sqHead;
    }
    return false;
}

HcclResult TaskException::PrintTaskException(hccl::Stream& stream)
{
    u32 sqHead = 0U;
    u32 sqTail = 0U;
    CHK_RET(QuerySqStatus(devId_, stream.sqId(), sqHead, sqTail));
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    CHK_PTR_NULL(sqeContextBuffer);
    if (sqHead == sqTail) { // 流上task已经执行完，不打印
        HCCL_RUN_INFO(
            "%s skip, group:%s, streamId:%d, sqHead is equal to sqTail:%u", __func__, identifier_.c_str(), stream.id(),
            sqTail);
        // 打印最后执行的200个task
        PrintTaskExceptionTaskQue(sqTail, sqeContextBuffer);
        return HCCL_SUCCESS;
    }
    u32 opIndex = indOpInfos_[sqeContextBuffer->rtsDfxInfo[sqHead].opRingBufferIdx].opIndex;
    if (IsRepeatPrint(stream.id(), opIndex, sqHead)) { // 避免重复打印
        HCCL_RUN_INFO(
            "%s skip, group:%s, streamId:%d, opIndex:%u, sqHead:%u, has already been printed", __func__,
            identifier_.c_str(), stream.id(), opIndex, sqHead);
        return HCCL_SUCCESS;
    }

    threadPrintState_[stream.id()] = {opIndex, sqHead}; // 打印当前流的信息，记录流的位置
    HCCL_RUN_INFO(
        "%s start, group:%s, streamId:%d, opIndex:%u, sqHead:%u, sqTail:%u", __func__, identifier_.c_str(), stream.id(),
        opIndex, sqHead, sqTail);

    HCCL_ERROR(
        "%s base information is streamId:%d, sqid:%u, head:%u, tail:%u, %s", __func__, stream.id(), stream.sqId(),
        sqHead, sqTail, GetTaskExceptionTaskInfo(sqHead, sqeContextBuffer).c_str());
    PrintTaskExceptionTaskQue(sqHead, sqeContextBuffer);
    return HCCL_SUCCESS;
}

HcclResult TaskException::PrintTaskExceptionByTaskId(u8 sqeType, u16 taskId, hccl::Stream& stream, u32 tail)
{
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    CHK_PTR_NULL(sqeContext);
    HCCL_ERROR("%s streamId:%d tail:%u cqeType:%u", __func__, stream.id(), tail, sqeType);
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    CHK_PTR_NULL(sqeContextBuffer);
    uint8_t* sqeMirrorBufferAddr = sqeContextBuffer->rtsMirrorBuffer + (tail - 1) * HCCL_SQE_SIZE;
    rtStarsSqeHeader_t* const sqeHeader = reinterpret_cast<rtStarsSqeHeader_t*>(sqeMirrorBufferAddr);
    CHK_PTR_NULL(sqeHeader);

    s32 taskNum = sqeHeader->taskId - taskId;
    HCCL_DEBUG("%s tail sqe taskId[%u] cqe taskId[%u] cqe type[%u]", __func__, sqeHeader->taskId, taskId, sqeType);
    s32 sqeIdx = tail - taskNum - 1;
    u32 sqHead = (sqeIdx + HCCL_SQE_MAX_CNT) % HCCL_SQE_MAX_CNT;

    HCCL_ERROR(
        "[TaskException]base information is streamId:%d, sqid:%u, head:%u, tail:%u, %s", stream.id(), stream.sqId(),
        sqHead, tail, GetTaskExceptionTaskInfo(sqHead, sqeContextBuffer).c_str());
    PrintTaskExceptionTaskQue(sqHead, sqeContextBuffer);
    return HCCL_SUCCESS;
}

std::string TaskException::GetTaskExceptionTaskInfo(u32 sqHead, SqeRingBuffer* sqeContextBuffer)
{
    SqeInfo sqeInfo;
    SqeContextUtils::QuerySqeInfo(
        sqeContextBuffer->rtsMirrorBuffer + sqHead * HCCL_SQE_SIZE, sqeContextBuffer->rtsqSqeType[sqHead],
        sqeContextBuffer->addInfo[sqHead], &sqeInfo);

    std::stringstream ss;
    ss << "type:" << SqeContextUtils::RtsqTaskTypeToStr(sqeInfo.type) << ", ";
    ss << "localRank:" << localUserRank_ << ", ";
    ss << "remoteRank:" << sqeContextBuffer->rtsDfxInfo[sqHead].remoteRank << ", ";
    ss << "taskId:" << sqeInfo.taskId << ", ";
    ss << "notifyId:" << sqeInfo.notifyId << ", ";
    ss << "length:" << sqeInfo.length << ", ";
    ss << "addr1High:0x" << std::hex << sqeInfo.addr1High << ", ";
    ss << "addr1Low:0x" << std::hex << sqeInfo.addr1Low << ", ";
    ss << "addr2High:0x" << std::hex << sqeInfo.addr2High << ", ";
    ss << "addr2Low:0x" << std::hex << sqeInfo.addr2Low << ".";
    return ss.str();
}

void TaskException::PrintTaskExceptionTaskQue(u32 sqIdx, SqeRingBuffer* sqeContextBuffer)
{
    const u32 sqeNum = 200; // 打印当前位置的前200个task
    // 记录上一次打印的算子信息
    IndOpInfo& lastOpInfo = indOpInfos_[sqeContextBuffer->rtsDfxInfo[sqIdx].opRingBufferIdx];
    u32 opIndex = lastOpInfo.opIndex; // 算子序号
    std::stringstream ss;
    ss << "OP(" << opIndex << ")";
    u32 tasksInCurrentLine = 0;
    const u32 maxTasksPerLine = 50; // 每行最多50个task，避免日志过长
    for (u32 i = 0; i < sqeNum; i++) {
        u32 newSqIdx = (sqIdx - i + HCCL_SQE_MAX_CNT) % HCCL_SQE_MAX_CNT;
        IndOpInfo& newOpInfo = indOpInfos_[sqeContextBuffer->rtsDfxInfo[newSqIdx].opRingBufferIdx];
        u32 newOpIdx = newOpInfo.opIndex;
        if (newOpIdx != opIndex) { // 不同一个算子，或已经到打印的最后一个位置
            PrintTaskExceptionOpInfo(lastOpInfo);
            HCCL_ERROR("[TaskException]task sequence is %s", ss.str().c_str());
            opIndex = newOpIdx;
            lastOpInfo = newOpInfo;
            ss.str("");
            ss << "OP(" << opIndex << ")";
            tasksInCurrentLine = 0;
        }
        // 输入task缩写
        ss << "," << GetTaskBriefsInfo(newSqIdx, sqeContextBuffer);
        tasksInCurrentLine++;
        // 超过每行最大task数时切分打印
        if (tasksInCurrentLine >= maxTasksPerLine && i < sqeNum) {
            HCCL_ERROR("[TaskException]task sequence is %s", ss.str().c_str());
            ss.str("");
            ss << "OP(" << opIndex << ")";
            tasksInCurrentLine = 0;
        }
    }
    // 打印剩余的task
    if (tasksInCurrentLine > 0) {
        HCCL_ERROR("[TaskException]task sequence is %s", ss.str().c_str());
    }
    return;
}

void TaskException::PrintTaskExceptionOpInfo(IndOpInfo& indOp)
{
    if (indOp.callback == nullptr) {
        HCCL_ERROR(
            "[TaskException][%s] fail, indOp callback is nullptr, group:%s, opIndex:%u", __func__, identifier_.c_str(),
            indOp.opIndex);
        return;
    }
    char opInfoTmp[OPINFO_RING_BUFFER_MAX];
    indOp.callback(reinterpret_cast<void*>(indOp.opInfo), opInfoTmp, OPINFO_RING_BUFFER_MAX);
    HCCL_ERROR(
        "[TaskException]opData information is group:%s, opIndex:%u, %s", identifier_.c_str(), indOp.opIndex, opInfoTmp);
}

std::string TaskException::GetTaskBriefsInfo(u32 idx, SqeRingBuffer* sqeContextBuffer)
{
    uint8_t* sqeMirrorBufferAddr = sqeContextBuffer->rtsMirrorBuffer + idx * HCCL_SQE_SIZE;
    rtStarsSqeHeader_t* const sqeHeader = reinterpret_cast<rtStarsSqeHeader_t*>(sqeMirrorBufferAddr);
    uint8_t sqeType = sqeHeader->type;

    SqeInfo sqeInfo;
    SqeContextUtils::QuerySqeInfo(
        sqeContextBuffer->rtsMirrorBuffer + idx * HCCL_SQE_SIZE, sqeContextBuffer->rtsqSqeType[idx],
        sqeContextBuffer->addInfo[idx], &sqeInfo);
    uint8_t subType = sqeInfo.subType;

    std::stringstream ss;
    std::string taskName = "UN";
    switch (sqeType) {
        case RT_STARS_SQE_TYPE_NOTIFY_RECORD:
            taskName = "NR"; // Notify Record
            break;
        case RT_STARS_SQE_TYPE_WRITE_VALUE:
            if (subType == RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_NO_PCIE) {
                taskName = "NR";
            } else if (subType == RT_STARS_WRITE_VALUE_SUB_TYPE_EVENT_RESET) {
                taskName = "NW"; // Notify Wait
            } else if (subType == RT_STARS_WRITE_VALUE_SUB_TYPE_RDMA_DB_SEND) {
                taskName = "RS"; // Rdma Send
            }
            break;
        case RT_STARS_SQE_TYPE_NOTIFY_WAIT:
            taskName = "NW";
            break;
        case RT_STARS_SQE_TYPE_EVENT_WAIT:
            taskName = "NW";
            break;
        case RT_STARS_SQE_TYPE_SDMA:
            taskName = "SD"; // SDMA
            break;
        case RT_STARS_SQE_TYPE_PLACE_HOLDER:
            taskName = "PH";
            break;
        default:
            break;
    }

    ss << taskName << "(";
    if (sqeContextBuffer->rtsDfxInfo[idx].remoteRank != INVALID_VALUE_RANKID) {
        ss << sqeContextBuffer->rtsDfxInfo[idx].remoteRank;
    } else {
        ss << "/";
    }
    ss << ",";
    if (sqeContextBuffer->rtsDfxInfo[idx].notifyId != INVALID_VALUE_RANKID) {
        ss << sqeContextBuffer->rtsDfxInfo[idx].notifyId;
    } else {
        ss << "/";
    }
    ss << ")";
    return ss.str();
}
} // namespace hccl
