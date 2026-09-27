/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_task_json.h"
#include "sim_common_api.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace AivSim {
namespace fs = std::filesystem;
using json = nlohmann::json;

uint64_t ResolveSerializedDataSliceOffset(const AivKernelExecutor& executor, const AivDataSlice& slice)
{
    if (slice.GetType() == AivBufferType::INPUT) {
        return slice.GetOffset() + executor.GetInputGlobalOffsetBase();
    }
    if (slice.GetType() == AivBufferType::OUTPUT) {
        return slice.GetOffset() + executor.GetOutputGlobalOffsetBase();
    }
    return slice.GetOffset();
}

json SerializeDataSlice(const AivKernelExecutor& executor, const AivDataSlice& slice)
{
    return json{
        {"bufferType", static_cast<uint32_t>(slice.GetType())},
        {"bufferTypeName", GetAivBufferTypeName(slice.GetType())},
        {"offset", ResolveSerializedDataSliceOffset(executor, slice)},
        {"size", slice.GetSize()},
    };
}

std::string SerializeKernelName(const AivOpParam& opParam)
{
    uint32_t kernelNameLen = 0;
    while (kernelNameLen < AIV_OP_KERNEL_NAME_MAX_LEN && opParam.kernelName[kernelNameLen] != '\0') {
        ++kernelNameLen;
    }
    return std::string(opParam.kernelName, kernelNameLen);
}

json SerializeOpParam(const AivOpParam& opParam)
{
    return json{
        {"dataType", opParam.dataType},         {"len", opParam.len},
        {"reduceOp", opParam.reduceOp},         {"root", opParam.root},
        {"sliceId", opParam.sliceId},           {"inputStride", opParam.inputStride},
        {"outputStride", opParam.outputStride}, {"kernelName", SerializeKernelName(opParam)},
    };
}

template <class TaskT>
std::vector<uint32_t> SerializeTaskIds(const std::vector<std::shared_ptr<TaskT>>& tasks)
{
    std::vector<uint32_t> taskIds;
    taskIds.reserve(tasks.size());
    for (const auto& task : tasks) {
        if (task != nullptr) {
            taskIds.push_back(task->GetTaskId());
        }
    }
    return taskIds;
}

json SerializeTaskBase(const AivTask& task)
{
    return json{
        {"taskType", static_cast<uint32_t>(task.GetTaskType())},
        {"taskTypeName", GetTypeName(task.GetTaskType())},
        {"taskId", task.GetTaskId()},
        {"rankId", task.GetRankId()},
        {"blockId", task.GetBlockId()},
        {"curPipe", static_cast<uint32_t>(task.GetCurPipe())},
        {"curPipeName", AscendC::GetPipeName(task.GetCurPipe())},
    };
}

json SerializeTaskByDynamicType(const AivKernelExecutor& executor, const std::shared_ptr<AivTask>& task)
{
    if (task == nullptr) {
        return json{};
    }

    json taskJson = SerializeTaskBase(*task);
    json payload = json::object();

    if (const auto* memCopyTask = dynamic_cast<const AivTaskMemCopy*>(task.get()); memCopyTask != nullptr) {
        payload["srcRank"] = memCopyTask->GetSrcRank();
        payload["src"] = SerializeDataSlice(executor, memCopyTask->GetSrc());
        payload["dstRank"] = memCopyTask->GetDstRank();
        payload["dst"] = SerializeDataSlice(executor, memCopyTask->GetDst());
    } else if (const auto* reduceTask = dynamic_cast<const AivTaskReduce*>(task.get()); reduceTask != nullptr) {
        payload["srcRank"] = reduceTask->GetSrcRank();
        payload["src"] = SerializeDataSlice(executor, reduceTask->GetSrc());
        payload["dstRank"] = reduceTask->GetDstRank();
        payload["dst"] = SerializeDataSlice(executor, reduceTask->GetDst());
        payload["dataType"] = reduceTask->GetDataType();
        payload["reduceOp"] = reduceTask->GetReduceOp();
        payload["reduceOpName"] = GetReduceOpName(static_cast<ReduceOp>(reduceTask->GetReduceOp()));
    } else if (const auto* setFlagTask = dynamic_cast<const AivTaskSetFlag*>(task.get()); setFlagTask != nullptr) {
        payload["srcPipe"] = static_cast<uint32_t>(setFlagTask->GetSrcPipe());
        payload["srcPipeName"] = AscendC::GetPipeName(setFlagTask->GetSrcPipe());
        payload["dstPipe"] = static_cast<uint32_t>(setFlagTask->GetDstPipe());
        payload["dstPipeName"] = AscendC::GetPipeName(setFlagTask->GetDstPipe());
        payload["eventId"] = setFlagTask->GetEventId();
    } else if (const auto* waitFlagTask = dynamic_cast<const AivTaskWaitFlag*>(task.get()); waitFlagTask != nullptr) {
        payload["srcPipe"] = static_cast<uint32_t>(waitFlagTask->GetSrcPipe());
        payload["srcPipeName"] = AscendC::GetPipeName(waitFlagTask->GetSrcPipe());
        payload["dstPipe"] = static_cast<uint32_t>(waitFlagTask->GetDstPipe());
        payload["dstPipeName"] = AscendC::GetPipeName(waitFlagTask->GetDstPipe());
        payload["eventId"] = waitFlagTask->GetEventId();
    } else if (const auto* pipeBarrierTask = dynamic_cast<const AivTaskPipeBarrier*>(task.get());
               pipeBarrierTask != nullptr) {
        payload["pipeType"] = static_cast<uint32_t>(pipeBarrierTask->GetPipeType());
        payload["pipeTypeName"] = AscendC::GetPipeName(pipeBarrierTask->GetPipeType());
        payload["barrierGroupTaskIds"] = SerializeTaskIds(pipeBarrierTask->GetBarrierGroup());
    } else if (const auto* syncAllTask = dynamic_cast<const AivTaskSyncAll*>(task.get()); syncAllTask != nullptr) {
        payload["syncRound"] = syncAllTask->GetSyncRound();
    } else if (const auto* sendFlagTask = dynamic_cast<const AivTaskSendFlag*>(task.get()); sendFlagTask != nullptr) {
        payload["rank"] = sendFlagTask->GetRank();
        payload["commInfoOffset"] = sendFlagTask->GetCommInfoOffset();
        payload["flagValue"] = sendFlagTask->GetFlagValue();
    } else if (const auto* recvFlagTask = dynamic_cast<const AivTaskRecvFlag*>(task.get()); recvFlagTask != nullptr) {
        payload["rank"] = recvFlagTask->GetRank();
        payload["commInfoOffset"] = recvFlagTask->GetCommInfoOffset();
        payload["targetValue"] = recvFlagTask->GetTargetValue();
    }

    taskJson["payload"] = std::move(payload);
    return taskJson;
}

json SerializeTaskArray(const AivKernelExecutor& executor, const std::vector<std::shared_ptr<AivTask>>& tasks)
{
    json taskArray = json::array();
    for (const auto& task : tasks) {
        taskArray.push_back(SerializeTaskByDynamicType(executor, task));
    }
    return taskArray;
}

json SerializeCore(const AivKernelExecutor& executor, const AivCore& core)
{
    return json{
        {"blockIdx", core.GetBlockIdx()},
        {"scalarTasks", SerializeTaskArray(executor, core.GetScalarPipe())},
        {"mte2Tasks", SerializeTaskArray(executor, core.GetMTE2Pipe())},
        {"mte3Tasks", SerializeTaskArray(executor, core.GetMTE3Pipe())},
    };
}

uint64_t ResolveBufferSize(const AivKernelExecutor& executor, bool useCclBuffer)
{
    for (uint32_t rank = 0; rank < executor.GetRankSize(); ++rank) {
        const Mem buffer = useCclBuffer ? executor.GetCclBuffer(rank) : executor.GetAivCommInfoBuffer(rank);
        if (buffer.size != 0) {
            return buffer.size;
        }
    }
    return 0;
}

bool WriteJsonAtomically(const json& content, const std::string& filePath)
{
    const fs::path outputPath(filePath);
    const fs::path tempPath = outputPath.string() + ".tmp";

    std::error_code ec;
    fs::create_directories(outputPath.parent_path(), ec);
    if (ec) {
        return false;
    }

    std::ofstream ofs(tempPath, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        return false;
    }
    ofs << content.dump(2) << '\n';
    ofs.close();
    if (!ofs) {
        return false;
    }

    fs::rename(tempPath, outputPath, ec);
    if (!ec) {
        return true;
    }

    fs::remove(outputPath, ec);
    ec.clear();
    fs::rename(tempPath, outputPath, ec);
    return !ec;
}

bool ResolveExecutorJsonFilePath(const AivKernelExecutor& executor, uint32_t launchIndex, std::string& filePath)
{
    const std::string dataRelPath = "data/hcclvm_aiv_rank" + std::to_string(executor.GetRankId()) + "_launch"
                                    + std::to_string(launchIndex) + "_task.json";
    filePath = InstallPath::ResolveToInstallRoot(dataRelPath);
    return true;
}

json SerializeExecutor(const AivKernelExecutor& executor, uint32_t launchIndex)
{
    json rankJson = json::object();
    rankJson["mode"] = "aiv";
    rankJson["rank"] = executor.GetRankId();
    rankJson["launchIndex"] = launchIndex;
    rankJson["rankSize"] = executor.GetRankSize();
    rankJson["inputGlobalOffsetBase"] = executor.GetInputGlobalOffsetBase();
    rankJson["outputGlobalOffsetBase"] = executor.GetOutputGlobalOffsetBase();
    rankJson["inBufferSize"] = executor.GetInBuffer().size;
    rankJson["outBufferSize"] = executor.GetOutBuffer().size;
    rankJson["cclBufferSize"] = ResolveBufferSize(executor, true);
    rankJson["aivCommInfoSize"] = ResolveBufferSize(executor, false);
    rankJson["ubBufferSize"] = executor.GetUbBufferSize();
    rankJson["curOp"] = SerializeOpParam(executor.GetCurOp());

    json coreArray = json::array();
    for (const auto& core : executor.GetAivCores()) {
        if (core != nullptr) {
            coreArray.push_back(SerializeCore(executor, *core));
        }
    }
    rankJson["aivCores"] = std::move(coreArray);
    return rankJson;
}

bool DumpExecutorToJsonFile(const AivKernelExecutor& executor, uint32_t launchIndex)
{
    std::string filePath;
    if (!ResolveExecutorJsonFilePath(executor, launchIndex, filePath)) {
        return false;
    }
    return WriteJsonAtomically(SerializeExecutor(executor, launchIndex), filePath);
}
} // namespace AivSim
