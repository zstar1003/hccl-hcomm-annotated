/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ai_core_stub.h"

#include <algorithm>
#include <cstdint>

#include "sim_log.h"

namespace AivSim {
AivKernelExecutor& AivKernelExecutor::GetInstance()
{
    static AivKernelExecutor instance;
    return instance;
}

void AivKernelExecutor::Init(RankId rankId, size_t blockNum, uint32_t rankSize)
{
    Reset();
    rankId_ = rankId;
    rankSize_ = std::min(rankSize, MAX_RANK_NUM);
    if (rankSize > MAX_RANK_NUM) {
        HCCL_VM_WARN("Init rankSize={} out of range, clamp to MAX_RANK_NUM={}", rankSize, MAX_RANK_NUM);
    }
    for (uint32_t i = 0; i < blockNum; ++i) {
        aivCores_.emplace_back(std::make_shared<AivCore>(i));
    }
}

void AivKernelExecutor::Reset()
{
    aivCores_.clear();
    taskIdGen_.store(0);
    rankId_ = UINT32_MAX;
    rankSize_ = 0;
    curOp_ = {};
    inBuffer_ = {};
    outBuffer_ = {};
    inputGlobalOffsetBase_ = 0;
    outputGlobalOffsetBase_ = 0;
    std::fill(std::begin(cclBuffer_), std::end(cclBuffer_), Mem{});
    std::fill(std::begin(aivCommInfoBuffer_), std::end(aivCommInfoBuffer_), Mem{});
}

std::shared_ptr<AivCore> AivKernelExecutor::GetAivCore(int64_t blockIdx)
{
    return (blockIdx >= 0 && blockIdx < aivCores_.size()) ? aivCores_[blockIdx] : nullptr;
}

void AivKernelExecutor::SetIoBuffer(
    uint64_t inBuffer, uint64_t inBufferSize, uint64_t outBuffer, uint64_t outBufferSize,
    uint64_t inputGlobalOffsetBase, uint64_t outputGlobalOffsetBase)
{
    inBuffer_ = {inBuffer, inBufferSize};
    outBuffer_ = {outBuffer, outBufferSize};
    inputGlobalOffsetBase_ = inputGlobalOffsetBase;
    outputGlobalOffsetBase_ = outputGlobalOffsetBase;
}

void AivKernelExecutor::SetCommBuffer(
    RankId rankId, uint64_t cclBuffer, uint64_t cclBufferSize, uint64_t aivCommInfoBuffer,
    uint64_t aivCommInfoBufferSize)
{
    if (rankId >= MAX_RANK_NUM) {
        HCCL_VM_WARN("SetCommBuffer rankId={} out of range", rankId);
        return;
    }
    cclBuffer_[rankId] = {cclBuffer, cclBufferSize};
    aivCommInfoBuffer_[rankId] = {aivCommInfoBuffer, aivCommInfoBufferSize};
}

bool AivBufferContains(const Mem& buffer, uint64_t addr, uint64_t size)
{
    if (buffer.addr == 0 || buffer.size == 0 || addr < buffer.addr) {
        return false;
    }

    const uint64_t offset = addr - buffer.addr;
    if (offset >= buffer.size) {
        return false;
    }

    return size <= (buffer.size - offset);
}

bool AivBufferMatch(
    uint64_t addr, uint64_t size, const Mem& buffer, AivBufferType type, RankId rank, AivDataSlice& slice,
    RankId* matchedRank)
{
    if (!AivBufferContains(buffer, addr, size)) {
        return false;
    }

    slice = AivDataSlice(type, addr - buffer.addr, size);
    if (matchedRank != nullptr) {
        *matchedRank = rank;
    }
    return true;
}

AivDataSlice AivKernelExecutor::ResolveGlobalDataSlice(uint64_t addr, uint64_t size, RankId* rankId) const
{
    if (rankId != nullptr) {
        *rankId = UINT32_MAX;
    }

    AivDataSlice matchedSlice{};
    if (AivBufferMatch(addr, size, inBuffer_, AivBufferType::INPUT, rankId_, matchedSlice, rankId)) {
        return matchedSlice;
    }
    if (AivBufferMatch(addr, size, outBuffer_, AivBufferType::OUTPUT, rankId_, matchedSlice, rankId)) {
        return matchedSlice;
    }
    for (RankId i = 0; i < rankSize_; ++i) {
        if (AivBufferMatch(addr, size, cclBuffer_[i], AivBufferType::CCL, i, matchedSlice, rankId)) {
            return matchedSlice;
        }
        if (AivBufferMatch(addr, size, aivCommInfoBuffer_[i], AivBufferType::AIV_COMM, i, matchedSlice, rankId)) {
            return matchedSlice;
        }
    }

    return {};
}

void AivKernelExecutor::DumpAllTasks() const
{
    HCCL_VM_DEBUG("");
    HCCL_VM_DEBUG("[rankId={}]", GetRankId());

    for (const auto& core : aivCores_) {
        core->DumpAllTasks();
    }
}

void AivCore::AppendScalar(const std::shared_ptr<AivTask>& task)
{
    task->SetRankId(AivKernelExecutor::GetInstance().GetRankId());
    task->SetBlockId(blockIdx_);
    task->SetTaskId(AivKernelExecutor::GetInstance().NextTaskId());
    task->SetCurPipe(AscendC::pipe_t::PIPE_S);
    pipeScalar_.push_back(task);
}

void AivCore::AppendMTE2(const std::shared_ptr<AivTask>& task)
{
    task->SetRankId(AivKernelExecutor::GetInstance().GetRankId());
    task->SetBlockId(blockIdx_);
    task->SetTaskId(AivKernelExecutor::GetInstance().NextTaskId());
    task->SetCurPipe(AscendC::pipe_t::PIPE_MTE2);
    pipeMTE2_.push_back(task);
}

void AivCore::AppendMTE3(const std::shared_ptr<AivTask>& task)
{
    task->SetRankId(AivKernelExecutor::GetInstance().GetRankId());
    task->SetBlockId(blockIdx_);
    task->SetTaskId(AivKernelExecutor::GetInstance().NextTaskId());
    task->SetCurPipe(AscendC::pipe_t::PIPE_MTE3);
    pipeMTE3_.push_back(task);
}

void AivCore::DumpAllTasks() const
{
    HCCL_VM_DEBUG("[blockIdx={:d}]", blockIdx_);

    HCCL_VM_DEBUG("[SCALAR] ");
    for (const auto& task : pipeScalar_) {
        HCCL_VM_DEBUG("    {:s}", task->Describe());
    }

    HCCL_VM_DEBUG("[MTE2] ");
    for (const auto& task : pipeMTE2_) {
        HCCL_VM_DEBUG("    {:s}", task->Describe());
    }

    HCCL_VM_DEBUG("[MTE3] ");
    for (const auto& task : pipeMTE3_) {
        HCCL_VM_DEBUG("    {:s}", task->Describe());
    }
}
} // namespace AivSim
