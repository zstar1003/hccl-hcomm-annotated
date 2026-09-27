/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_graph_executor.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "aiv_task_snapshot_loader.h"
#include "aiv_resource_manager.h"
#include "hccl_types.h"
#include "sim_log.h"
#include "sim_common_defs.h"

using HcclSim::HcclVmResult;

static AivSim::flag_t*
GetCommInfoCellPtr(const AivBufferResource& aivCommInfoBuffer, uint64_t commInfoOffset, uint32_t taskId)
{
    if (aivCommInfoBuffer.realAddr == nullptr || aivCommInfoBuffer.size < sizeof(AivSim::flag_t)) {
        HCCL_VM_ERROR("AIV commInfo buffer invalid, taskId={:d}, aivCommInfoSize={:d}", taskId, aivCommInfoBuffer.size);
        return nullptr;
    }
    if (commInfoOffset % AivCommInfoLayout::SYNC_CELL_BYTES != 0
        || commInfoOffset > aivCommInfoBuffer.size - sizeof(AivSim::flag_t)) {
        HCCL_VM_ERROR(
            "AIV commInfo cell offset invalid, taskId={:d}, commInfoOffset={:d}, aivCommInfoSize={:d}", taskId,
            commInfoOffset, aivCommInfoBuffer.size);
        return nullptr;
    }
    auto* commInfoBytes = static_cast<uint8_t*>(aivCommInfoBuffer.realAddr);
    return reinterpret_cast<AivSim::flag_t*>(commInfoBytes + commInfoOffset);
}

static void AppendPipeTasksToQueue(
    const std::vector<std::shared_ptr<AivSim::AivTask>>& pipeTasks,
    std::queue<std::shared_ptr<AivSim::AivTask>>& taskQueue, uint32_t& maxTaskId, uint32_t& maxEventId,
    uint32_t& maxSyncRound)
{
    for (const auto& task : pipeTasks) {
        maxTaskId = std::max(maxTaskId, task->GetTaskId());
        if (task->GetTaskType() == AivSim::AivTaskType::SET_FLAG) {
            auto setFlagTask = std::dynamic_pointer_cast<AivSim::AivTaskSetFlag>(task);
            maxEventId = std::max(maxEventId, static_cast<uint32_t>(setFlagTask->GetEventId()));
        }
        if (task->GetTaskType() == AivSim::AivTaskType::WAIT_FLAG) {
            auto waitFlagTask = std::dynamic_pointer_cast<AivSim::AivTaskWaitFlag>(task);
            maxEventId = std::max(maxEventId, static_cast<uint32_t>(waitFlagTask->GetEventId()));
        }
        if (task->GetTaskType() == AivSim::AivTaskType::SYNC_ALL) {
            auto syncAllTask = std::dynamic_pointer_cast<AivSim::AivTaskSyncAll>(task);
            maxSyncRound = std::max(maxSyncRound, syncAllTask->GetSyncRound());
        }
        taskQueue.push(task);
    }
}

AivBlock::AivBlock(uint32_t blockIdx, size_t maxEventId, size_t ubSize) : blockIdx_(blockIdx), ubSize_(ubSize)
{
    for (size_t i = 0; i <= maxEventId; ++i) {
        events_.push_back(false);
    }
    ub_ = malloc(ubSize_);
}

AivBlock::~AivBlock() { free(ub_); }

bool AivGraphExecutor::Init(uint32_t rankId, uint32_t launchIdx)
{
    HCCL_VM_DEBUG("begin, launchIdx={}, rankId={}", launchIdx_, rankId_);
    rankId_ = rankId;
    launchIdx_ = launchIdx;

    AivRuntimeTaskSnapshot taskSnapshot;
    std::string errorMessage;
    if (!AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(
            rankId_, static_cast<uint32_t>(launchIdx_), taskSnapshot, &errorMessage)) {
        HCCL_VM_ERROR(
            "failed to load runtime snapshot, launchIdx={}, rankId={}, reason={}", launchIdx_, rankId_, errorMessage);
        return false;
    }

    aivBlocks_.clear();
    pipeBarrierRegisters_.clear();
    syncAllRegisters_.clear();
    aivTaskQueues_.clear();
    isInitialized_ = false;

    rankId_ = taskSnapshot.rankId;
    rankSize_ = taskSnapshot.rankSize;
    HCCL_VM_DEBUG(
        "snapshot loaded, launchIdx={}, rankId={}, rankSize={}, file={}, aivBlockNum={}", launchIdx_, rankId_,
        rankSize_, taskSnapshot.filePath, taskSnapshot.blocks.size());

    uint32_t maxTaskId = 0;
    uint32_t maxSyncRound = 0;
    for (const auto& block : taskSnapshot.blocks) {
        uint32_t maxEventId = 0;

        aivTaskQueues_.emplace_back();
        AppendPipeTasksToQueue(block.scalarTasks, aivTaskQueues_.back(), maxTaskId, maxEventId, maxSyncRound);

        aivTaskQueues_.emplace_back();
        AppendPipeTasksToQueue(block.mte2Tasks, aivTaskQueues_.back(), maxTaskId, maxEventId, maxSyncRound);

        aivTaskQueues_.emplace_back();
        AppendPipeTasksToQueue(block.mte3Tasks, aivTaskQueues_.back(), maxTaskId, maxEventId, maxSyncRound);

        HCCL_VM_TRACE(
            "blockIdx={}, scalarTaskCount={}, mte2TaskCount={}, mte3TaskCount={}, maxEventId={}", block.blockIdx,
            block.scalarTasks.size(), block.mte2Tasks.size(), block.mte3Tasks.size(), maxEventId);
        aivBlocks_.emplace_back(std::make_unique<AivBlock>(block.blockIdx, maxEventId, AivSim::AIV_UB_SIZE));
    }

    for (uint32_t i = 0; i <= maxTaskId; ++i) {
        pipeBarrierRegisters_.push_back(false);
    }
    size_t pipeSize = aivBlocks_.size() * AivSim::AivCore::GetPipeNum();
    for (uint32_t i = 0; i <= maxSyncRound; ++i) {
        syncAllRegisters_.emplace_back(pipeSize, false);
    }

    HCCL_VM_DEBUG(
        "queueCount={}, maxTaskId={}, pipeBarrierRegisterCount={}, syncAllRegisterCount={}, aivBlockNum={}",
        aivTaskQueues_.size(), maxTaskId, pipeBarrierRegisters_.size(), syncAllRegisters_.size(), aivBlocks_.size());

    isInitialized_ = true;
    HCCL_VM_INFO("success, launchIdx={}, rankId={}", launchIdx_, rankId_);
    return true;
}

bool AivGraphExecutor::HasTask() const
{
    for (const auto& queue : aivTaskQueues_) {
        if (!queue.empty()) {
            return true;
        }
    }
    return false;
}

HcclVmResult AivGraphExecutor::Execute()
{
    HCCL_VM_DEBUG("[AivGraph Execute] rankId={} launchIdx={}", rankId_, launchIdx_);

    const size_t queueNum = aivTaskQueues_.size();
    while (HasTask()) {
        auto& queue = aivTaskQueues_[curQueueIdx_];
        if (!queue.empty()) {
            std::shared_ptr<AivSim::AivTask> task = queue.front();
            auto ret = ExecuteTask(task);
            if (ret == HcclVmResult::HCCL_SIM_SUCCESS) {
                // task done
                queue.pop();
            } else if (ret == HcclVmResult::HCCL_SIM_VRT_CONTINUE_CMD) {
                // task not finish
                HCCL_VM_TRACE("Task not finish, taskId={}", task->GetTaskId());
            } else if (ret == HcclVmResult::HCCL_SIM_VRT_HOLD_CMD) {
                // aiv graph hold
                HCCL_VM_DEBUG(
                    "AivGraph Hold, taskId={} rankId={} launchIdx={}", task->GetTaskId(), rankId_, launchIdx_);
                curQueueIdx_ = (curQueueIdx_ + 1) % queueNum;
                return ret;
            } else {
                // task failed
                HCCL_VM_ERROR("Task execute failed, taskId={} ret={}", task->GetTaskId(), static_cast<uint32_t>(ret));
                return ret;
            }
        }
        curQueueIdx_ = (curQueueIdx_ + 1) % queueNum;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTask> task)
{
    HCCL_VM_TRACE("Executing Task, taskId={}", task->GetTaskId());
    switch (task->GetTaskType()) {
        case AivSim::AivTaskType::MEM_COPY:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskMemCopy>(task));
        case AivSim::AivTaskType::REDUCE:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskReduce>(task));
        case AivSim::AivTaskType::WAIT_FLAG:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskWaitFlag>(task));
        case AivSim::AivTaskType::SET_FLAG:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskSetFlag>(task));
        case AivSim::AivTaskType::PIPE_BARRIER:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskPipeBarrier>(task));
        case AivSim::AivTaskType::SYNC_ALL:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskSyncAll>(task));
        case AivSim::AivTaskType::SEND_FLAG:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskSendFlag>(task));
        case AivSim::AivTaskType::RECV_FLAG:
            return ExecuteTask(std::dynamic_pointer_cast<AivSim::AivTaskRecvFlag>(task));
        default:
            HCCL_VM_ERROR("Execute failed, unsupported taskType={:d}", static_cast<uint16_t>(task->GetTaskType()));
            return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskMemCopy> task)
{
    HCCL_VM_DEBUG("{}", task->Describe());

    if (task->GetSrc().GetSize() != task->GetDst().GetSize()) {
        HCCL_VM_ERROR("task invalid, len not match, taskId={:d}", task->GetTaskId());
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }
    const size_t len = task->GetSrc().GetSize();

    void* src = GetMemPtr(task, true);
    if (src == nullptr) {
        HCCL_VM_ERROR("Get src memory failed, taskId={:d}", task->GetTaskId());
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }

    void* dst = GetMemPtr(task, false);
    if (dst == nullptr) {
        HCCL_VM_ERROR("Get dst memory failed, taskId={:d}", task->GetTaskId());
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }

    std::memcpy(dst, src, len);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

template <typename T, typename>
void* AivGraphExecutor::GetMemPtr(std::shared_ptr<T> task, bool isSrc)
{
    const auto& slice = isSrc ? task->GetSrc() : task->GetDst();
    const uint64_t accessLen = slice.GetOffset() + slice.GetSize();

    if (slice.GetType() == AivSim::AivBufferType::UB) {
        if (accessLen > aivBlocks_[task->GetBlockId()]->GetUBSize()) {
            HCCL_VM_ERROR("UB out-of-bounds, taskId={:d}", task->GetTaskId());
            return nullptr;
        }
        uint64_t addr = reinterpret_cast<uint64_t>(aivBlocks_[task->GetBlockId()]->GetUB()) + slice.GetOffset();
        return reinterpret_cast<void*>(addr);
    } else {
        const uint32_t rankId = isSrc ? task->GetSrcRank() : task->GetDstRank();
        auto* rankResource = AivResourceManager::GetInstance().GetRankResource(rankId);
        if (rankResource == nullptr) {
            HCCL_VM_ERROR("GetRankResource failed, taskId={:d} targetRank={:d}", task->GetTaskId(), rankId);
            return nullptr;
        }
        const AivBufferResource* rankMem = nullptr;
        if (slice.GetType() == AivSim::AivBufferType::INPUT) {
            rankMem = &rankResource->inputBuffer;
        } else if (slice.GetType() == AivSim::AivBufferType::OUTPUT) {
            rankMem = &rankResource->outputBuffer;
        } else if (slice.GetType() == AivSim::AivBufferType::CCL) {
            rankMem = &rankResource->cclBuffer;
        } else if (slice.GetType() == AivSim::AivBufferType::AIV_COMM) {
            rankMem = &rankResource->aivCommInfoBuffer;
        } else {
            HCCL_VM_ERROR(
                "Mem type invalid, taskId={:d} sliceType={:d}", task->GetTaskId(),
                static_cast<uint32_t>(slice.GetType()));
            return nullptr;
        }
        if (accessLen > rankMem->size) {
            HCCL_VM_ERROR(
                "Rank memory out-of-bounds, taskId={:d} sliceType={:d}", task->GetTaskId(),
                static_cast<uint32_t>(slice.GetType()));
            return nullptr;
        }
        uint64_t addr = reinterpret_cast<uint64_t>(rankMem->realAddr) + slice.GetOffset();
        return reinterpret_cast<void*>(addr);
    }
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskReduce> task)
{
    HCCL_VM_DEBUG("{}", task->Describe());

    if (task->GetSrc().GetSize() != task->GetDst().GetSize()) {
        HCCL_VM_ERROR("task invalid, len not match, taskId={:d}", task->GetTaskId());
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }
    const size_t len = task->GetSrc().GetSize();

    void* src = GetMemPtr(task, true);
    if (src == nullptr) {
        HCCL_VM_ERROR("Get src memory failed, taskId={:d}", task->GetTaskId());
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }

    void* dst = GetMemPtr(task, false);
    if (dst == nullptr) {
        HCCL_VM_ERROR("Get dst memory failed, taskId={:d}", task->GetTaskId());
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }

    switch (static_cast<HcclDataType>(task->GetDataType())) {
        case HcclDataType::HCCL_DATA_TYPE_HIF8:
            return Reduce<AscendC::half>(src, dst, len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return Reduce<int16_t>(src, dst, len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_UINT16:
            return Reduce<uint16_t>(src, dst, len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return Reduce<float>(src, dst, len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return Reduce<int32_t>(src, dst, len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_UINT32:
            return Reduce<uint32_t>(src, dst, len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return Reduce<int8_t>(src, dst, len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_UINT8:
            return Reduce<uint8_t>(src, dst, len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_BFP16:
            return Reduce<AscendC::bfloat16_t>(src, dst, len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return Reduce<int64_t>(src, dst, len, task->GetReduceOp());
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            return Reduce<uint64_t>(src, dst, len, task->GetReduceOp());
        default:
            HCCL_VM_ERROR(
                "Reduce DataType not supported, taskId={:d} dataType={:d}", task->GetTaskId(), task->GetDataType());
            return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }
}

template <typename T>
HcclVmResult AivGraphExecutor::Reduce(void* src, void* dst, size_t len, uint32_t reduceOp)
{
    if (len % sizeof(T) != 0) {
        HCCL_VM_ERROR("reduce length invalid, len={:d}", len);
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }
    const size_t count = len / sizeof(T);

    T* srcArr = static_cast<T*>(src);
    T* dstArr = static_cast<T*>(dst);

    if (reduceOp == static_cast<uint32_t>(AivSim::ReduceOp::REDUCE_SUM)) {
        for (size_t i = 0; i < count; ++i) {
            dstArr[i] += srcArr[i];
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    if (reduceOp == static_cast<uint32_t>(AivSim::ReduceOp::REDUCE_MAX)) {
        for (size_t i = 0; i < count; ++i) {
            dstArr[i] = std::max(srcArr[i], dstArr[i]);
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    if (reduceOp == static_cast<uint32_t>(AivSim::ReduceOp::REDUCE_MIN)) {
        for (size_t i = 0; i < count; ++i) {
            dstArr[i] = std::min(srcArr[i], dstArr[i]);
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    HCCL_VM_ERROR("ReduceOp not supported, reduceOp={:d}", reduceOp);
    return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskSetFlag> task)
{
    auto& events = aivBlocks_[task->GetBlockId()]->GetEvents();
    events[task->GetEventId()] = true;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskWaitFlag> task)
{
    auto& events = aivBlocks_[task->GetBlockId()]->GetEvents();
    if (events[task->GetEventId()]) {
        events[task->GetEventId()] = false;
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }
    return HcclVmResult::HCCL_SIM_VRT_CONTINUE_CMD;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskPipeBarrier> task)
{
    pipeBarrierRegisters_[task->GetTaskId()] = true;
    bool pass = true;
    for (const auto& groupTask : task->GetBarrierGroup()) {
        if (!pipeBarrierRegisters_[groupTask->GetTaskId()]) {
            pass = false;
            break;
        }
    }
    return pass ? HcclVmResult::HCCL_SIM_SUCCESS : HcclVmResult::HCCL_SIM_VRT_CONTINUE_CMD;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskSyncAll> task)
{
    size_t curRound = task->GetSyncRound();
    size_t registerIdx = task->GetBlockId() * AivSim::AivCore::GetPipeNum() + static_cast<size_t>(task->GetCurPipe());
    syncAllRegisters_[curRound][registerIdx] = true;
    HCCL_VM_TRACE(
        "Executing SyncAll, taskId={}, syncRound={}, registerIdx={}", task->GetTaskId(), curRound, registerIdx);

    bool pass = true;
    for (bool val : syncAllRegisters_[curRound]) {
        if (!val) {
            pass = false;
            break;
        }
    }
    return pass ? HcclVmResult::HCCL_SIM_SUCCESS : HcclVmResult::HCCL_SIM_VRT_CONTINUE_CMD;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskSendFlag> task)
{
    HCCL_VM_DEBUG("{}", task->Describe());

    auto* rankResource = AivResourceManager::GetInstance().GetRankResource(task->GetRank());
    if (rankResource == nullptr) {
        HCCL_VM_ERROR("GetRankResource failed, taskId={:d}, targetRank={:d}", task->GetTaskId(), task->GetRank());
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }

    auto* flagPtr = GetCommInfoCellPtr(rankResource->aivCommInfoBuffer, task->GetCommInfoOffset(), task->GetTaskId());
    if (flagPtr == nullptr) {
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }

    *flagPtr = task->GetFlagValue();
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AivGraphExecutor::ExecuteTask(std::shared_ptr<AivSim::AivTaskRecvFlag> task)
{
    HCCL_VM_DEBUG("{}", task->Describe());

    auto* rankResource = AivResourceManager::GetInstance().GetRankResource(task->GetRank());
    if (rankResource == nullptr) {
        HCCL_VM_ERROR("GetRankResource failed, taskId={:d}, targetRank={:d}", task->GetTaskId(), task->GetRank());
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }

    auto* flagPtr = GetCommInfoCellPtr(rankResource->aivCommInfoBuffer, task->GetCommInfoOffset(), task->GetTaskId());
    if (flagPtr == nullptr) {
        return HcclVmResult::HCCL_SIM_VRT_ERROR_CMD;
    }

    AivSim::flag_t curFlagValue = *flagPtr;

    if (curFlagValue == task->GetTargetValue()) {
        return HcclVmResult::HCCL_SIM_SUCCESS;
    } else {
        return HcclVmResult::HCCL_SIM_VRT_HOLD_CMD;
    }
}

void AivGraphExecutor::ShowCurrentTaskStatus()
{
    std::stringstream ss;
    ss << "curRank=" << rankId_;
    ss << ", TaskStatus(queueIdx, taskId)={";
    for (size_t i = 0; i < aivTaskQueues_.size(); ++i) {
        if (i != 0) {
            ss << ",";
        }
        ss << "[" << i << ",";
        if (aivTaskQueues_[i].empty()) {
            ss << "empty";
        } else {
            ss << aivTaskQueues_[i].front()->GetTaskId();
        }
        ss << "]";
    }
    ss << "}";
    HCCL_VM_DEBUG("{}", ss.str());
}
