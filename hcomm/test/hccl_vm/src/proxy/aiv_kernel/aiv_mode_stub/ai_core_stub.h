/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_AI_CORE_STUB_H
#define AIV_AI_CORE_STUB_H

#include <atomic>
#include <map>
#include <memory>
#include <vector>

#include "aiv_mode_stub_base.h"
#include "aiv_task.h"

namespace AivSim {
constexpr uint64_t AIV_UB_SIZE = 190 * 1024;
constexpr uint32_t MAX_RANK_NUM = 128;

class AivCore {
public:
    explicit AivCore(int64_t blockIdx) : blockIdx_(blockIdx) {};
    ~AivCore() = default;
    AivCore(const AivCore&) = delete;
    AivCore& operator=(const AivCore&) = delete;

    static size_t GetPipeNum() { return 3; }
    int64_t GetBlockIdx() const { return blockIdx_; }
    ReduceOp GetAtomicOp() const { return atomicOp_; }
    void SetAtomicOp(const ReduceOp atomicOp = ReduceOp::REDUCE_RESERVED) { atomicOp_ = atomicOp; }
    uint32_t GetSyncRound() { return syncRound_++; }

    void AppendScalar(const std::shared_ptr<AivTask>& task);
    void AppendMTE2(const std::shared_ptr<AivTask>& task);
    void AppendMTE3(const std::shared_ptr<AivTask>& task);
    const std::vector<std::shared_ptr<AivTask>>& GetScalarPipe() const { return pipeScalar_; }
    const std::vector<std::shared_ptr<AivTask>>& GetMTE2Pipe() const { return pipeMTE2_; }
    const std::vector<std::shared_ptr<AivTask>>& GetMTE3Pipe() const { return pipeMTE3_; }

    void DumpAllTasks() const;

private:
    int64_t blockIdx_{0};
    uint32_t syncRound_{0};
    ReduceOp atomicOp_{ReduceOp::REDUCE_RESERVED};

    Mem unifiedBuffer_{0x0, AIV_UB_SIZE};

    std::vector<std::shared_ptr<AivTask>> pipeScalar_{};
    std::vector<std::shared_ptr<AivTask>> pipeMTE2_{};
    std::vector<std::shared_ptr<AivTask>> pipeMTE3_{};
};

class AivKernelExecutor {
public:
    ~AivKernelExecutor() = default;
    AivKernelExecutor(const AivKernelExecutor&) = delete;
    AivKernelExecutor& operator=(const AivKernelExecutor&) = delete;

    static AivKernelExecutor& GetInstance();

    void Init(RankId rankId, size_t blockNum, uint32_t rankSize = 0);
    void Reset();

    TaskId NextTaskId() { return taskIdGen_.fetch_add(1); }
    RankId GetRankId() const { return rankId_; }
    uint32_t GetRankSize() const { return rankSize_; }
    int64_t GetBlockNum() const { return aivCores_.size(); }
    std::shared_ptr<AivCore> GetAivCore(int64_t blockIdx);
    const std::vector<std::shared_ptr<AivCore>>& GetAivCores() const { return aivCores_; }
    std::vector<std::shared_ptr<AivCore>>& GetAivCores() { return aivCores_; }
    const AivOpParam& GetCurOp() const { return curOp_; }
    const Mem& GetInBuffer() const { return inBuffer_; }
    const Mem& GetOutBuffer() const { return outBuffer_; }
    uint64_t GetInputGlobalOffsetBase() const { return inputGlobalOffsetBase_; }
    uint64_t GetOutputGlobalOffsetBase() const { return outputGlobalOffsetBase_; }
    Mem GetCclBuffer(RankId rankId) const { return rankId < MAX_RANK_NUM ? cclBuffer_[rankId] : Mem{}; }
    Mem GetAivCommInfoBuffer(RankId rankId) const { return rankId < MAX_RANK_NUM ? aivCommInfoBuffer_[rankId] : Mem{}; }
    uint64_t GetUbBufferSize() const { return AIV_UB_SIZE; }
    void SetCurOp(const AivOpParam& opParam) { curOp_ = opParam; }
    void SetIoBuffer(
        uint64_t inBuffer, uint64_t inBufferSize, uint64_t outBuffer, uint64_t outBufferSize,
        uint64_t inputGlobalOffsetBase, uint64_t outputGlobalOffsetBase);
    void SetCommBuffer(
        RankId rankId, uint64_t cclBuffer, uint64_t cclBufferSize, uint64_t aivCommInfoBuffer,
        uint64_t aivCommInfoBufferSize);
    AivDataSlice ResolveGlobalDataSlice(uint64_t addr, uint64_t size, RankId* rankId = nullptr) const;

    void DumpAllTasks() const;

private:
    AivKernelExecutor() = default;

private:
    std::atomic<TaskId> taskIdGen_{0}; // TaskId Generator
    RankId rankId_{UINT32_MAX};
    uint32_t rankSize_{0};
    std::vector<std::shared_ptr<AivCore>> aivCores_{};
    AivOpParam curOp_{};

    Mem inBuffer_{};
    Mem outBuffer_{};
    uint64_t inputGlobalOffsetBase_{0};
    uint64_t outputGlobalOffsetBase_{0};
    Mem cclBuffer_[MAX_RANK_NUM]{};
    Mem aivCommInfoBuffer_[MAX_RANK_NUM]{};
};
} // namespace AivSim

#endif // AIV_AI_CORE_STUB_H
