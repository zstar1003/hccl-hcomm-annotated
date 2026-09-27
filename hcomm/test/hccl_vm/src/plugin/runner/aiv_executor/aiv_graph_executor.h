/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_AIVGRAPHEXECUTOR_H
#define AIV_AIVGRAPHEXECUTOR_H

#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

#include "ai_core_stub.h"
#include "aiv_task.h"
#include "sim_common_defs.h"

class AivBlock {
public:
    AivBlock(uint32_t blockIdx, size_t maxEventId, size_t ubSize);
    ~AivBlock();
    AivBlock(const AivBlock&) = delete;
    AivBlock& operator=(const AivBlock&) = delete;

    uint32_t GetBlockIdx() const { return blockIdx_; }
    std::vector<bool>& GetEvents() { return events_; }
    void* GetUB() { return ub_; }
    size_t GetUBSize() const { return ubSize_; }

private:
    uint32_t blockIdx_{UINT32_MAX};

    std::vector<bool> events_{};

    void* ub_{nullptr};
    size_t ubSize_{0};
};

class AivGraphExecutor {
public:
    explicit AivGraphExecutor(uint64_t launchIdx) : launchIdx_{launchIdx} {}
    ~AivGraphExecutor() = default;
    AivGraphExecutor(const AivGraphExecutor&) = delete;
    AivGraphExecutor& operator=(const AivGraphExecutor&) = delete;

    bool Init(uint32_t rankId, uint32_t launchIdx);
    bool IsInitialized() const { return isInitialized_; }

    HcclSim::HcclVmResult Execute();

private:
    bool HasTask() const;

    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTask> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskMemCopy> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskReduce> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskSetFlag> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskWaitFlag> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskPipeBarrier> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskSendFlag> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskRecvFlag> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskSyncAll> task);

    template <
        typename T,
        typename
        = std::enable_if_t<std::is_same_v<T, AivSim::AivTaskMemCopy> || std::is_same_v<T, AivSim::AivTaskReduce>>>
    void* GetMemPtr(std::shared_ptr<T> task, bool isSrc);

    template <typename T>
    HcclSim::HcclVmResult Reduce(void* src, void* dst, size_t len, uint32_t reduceOp);

    // For Debug
    void ShowCurrentTaskStatus();

private:
    bool isInitialized_{false};

    uint64_t launchIdx_{UINT64_MAX}; // launch index on stream
    uint32_t rankId_{UINT32_MAX};
    uint32_t rankSize_{0};

    std::vector<std::unique_ptr<AivBlock>> aivBlocks_{};
    std::vector<bool> pipeBarrierRegisters_{};
    std::vector<std::vector<bool>> syncAllRegisters_{};
    std::vector<std::queue<std::shared_ptr<AivSim::AivTask>>> aivTaskQueues_{};

    size_t curQueueIdx_{0}; // 当前执行的Task队列
};

#endif // AIV_AIVGRAPHEXECUTOR_H
