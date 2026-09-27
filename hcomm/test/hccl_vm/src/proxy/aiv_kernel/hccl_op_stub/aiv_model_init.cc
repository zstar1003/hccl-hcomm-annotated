/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_model_init.h"

#include <cstdint>
#include <string>

#include "aiv_task_json.h"
#include "sim_log.h"
#include "sim_common_defs.h"

void aiv_env_init(
    uint32_t rankId, size_t blockNum, const void* buffIn, uint32_t rankSize, uint64_t input, uint64_t inputSize,
    uint64_t output, uint64_t outputSize, uint64_t inputGlobalOffsetBase, uint64_t outputGlobalOffsetBase,
    uint64_t cclBufferSize, uint64_t aivCommInfoSize, AivSim::AivOpParam opParam)
{
    AscendC::block_num = blockNum;
    auto& executor = AivSim::AivKernelExecutor::GetInstance();
    executor.Init(rankId, blockNum, rankSize);
    executor.SetCurOp(opParam);
    const std::string inputMemDesc = AivSim::Mem{input, inputSize}.Describe();
    const std::string outputMemDesc = AivSim::Mem{output, outputSize}.Describe();
    HCCL_VM_DEBUG("begin");
    HCCL_VM_DEBUG("rankId={}, blockNum={}, buffIn={:p}, rankSize={}", rankId, blockNum, buffIn, rankSize);
    HCCL_VM_DEBUG(
        "input={}, output={}, "
        "inputGlobalOffsetBase={}, outputGlobalOffsetBase={}, cclBufferSize={}, aivCommInfoSize={}",
        inputMemDesc, outputMemDesc, inputGlobalOffsetBase, outputGlobalOffsetBase, cclBufferSize, aivCommInfoSize);
    HCCL_VM_DEBUG(
        "curOp: dataType={}, len={}, reduceOp={}, root={}, sliceId={}, inputStride={}, "
        "outputStride={}, kernelName={}",
        opParam.dataType, opParam.len, opParam.reduceOp, opParam.root, opParam.sliceId, opParam.inputStride,
        opParam.outputStride, opParam.kernelName);

    HCCL_VM_DEBUG("SetIoBuffer");
    executor.SetIoBuffer(input, inputSize, output, outputSize, inputGlobalOffsetBase, outputGlobalOffsetBase);

    if (buffIn == nullptr) {
        HCCL_VM_DEBUG("buffIn is null, skip SetCommBuffer initialization");
        return;
    }

    const auto* cclBufferTable = static_cast<const uint64_t*>(buffIn);
    const auto* aivCommInfoTable = reinterpret_cast<const uint64_t*>(
        static_cast<const uint8_t*>(buffIn) + AivCommInfoLayout::GM_OUT_TABLE_OFFSET);
    for (uint32_t i = 0; i < rankSize; ++i) {
        const uint64_t cclBuffer = cclBufferTable[i];
        const uint64_t aivCommInfoBuffer = aivCommInfoTable[i];
        const std::string cclMemDesc = AivSim::Mem{cclBuffer, cclBufferSize}.Describe();
        const std::string aivCommInfoMemDesc = AivSim::Mem{aivCommInfoBuffer, aivCommInfoSize}.Describe();
        HCCL_VM_DEBUG("SetCommBuffer rank={}, cclBuffer={}, aivCommInfoBuffer={}", i, cclMemDesc, aivCommInfoMemDesc);
        executor.SetCommBuffer(i, cclBuffer, cclBufferSize, aivCommInfoBuffer, aivCommInfoSize);
    }
}

void aiv_set_block_idx(int64_t blockIdx) { AscendC::block_idx = blockIdx; }

void aiv_dump_tasks(uint32_t launchIndex)
{
    AivSim::AivKernelExecutor::GetInstance().DumpAllTasks();
    AivSim::DumpExecutorToJsonFile(AivSim::AivKernelExecutor::GetInstance(), launchIndex);
}
