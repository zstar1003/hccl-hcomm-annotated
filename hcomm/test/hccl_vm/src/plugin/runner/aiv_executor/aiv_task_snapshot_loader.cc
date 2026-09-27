/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_task_snapshot_loader.h"
#include "sim_common_api.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann_json/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr char AIV_RANK_TASK_FILE_PREFIX[] = "hcclvm_aiv_rank";
constexpr char AIV_RANK_TASK_FILE_SUFFIX[] = "_task.json";
constexpr char AIV_RANK_TASK_FILE_LAUNCH_MARKER[] = "_launch";

static void SetCommonError(std::string* errorMessage, const std::string& message)
{
    if (errorMessage != nullptr) {
        *errorMessage = message;
    }
}

static bool ResolveAivSnapshotDirectory(fs::path& snapshotDir, std::string* errorMessage)
{
    snapshotDir = fs::path(InstallPath::ResolveToInstallRoot("data"));
    std::error_code ec;
    if (!fs::exists(snapshotDir, ec) || ec || !fs::is_directory(snapshotDir, ec)) {
        SetCommonError(errorMessage, "AIV data directory does not exist: " + snapshotDir.string());
        return false;
    }

    return true;
}

static bool
ResolveRankTaskFilePath(uint32_t rankId, uint32_t launchIndex, std::string& filePath, std::string* errorMessage)
{
    fs::path snapshotDir;
    if (!ResolveAivSnapshotDirectory(snapshotDir, errorMessage)) {
        return false;
    }

    std::error_code ec;
    filePath = (snapshotDir
                / (std::string(AIV_RANK_TASK_FILE_PREFIX) + std::to_string(rankId)
                   + std::string(AIV_RANK_TASK_FILE_LAUNCH_MARKER) + std::to_string(launchIndex)
                   + std::string(AIV_RANK_TASK_FILE_SUFFIX)))
                   .string();
    if (!fs::exists(filePath, ec) || ec) {
        SetCommonError(errorMessage, "AIV task json file does not exist: " + filePath);
        return false;
    }

    return true;
}

static std::vector<uint32_t> LoadTaskIdsFromPayload(const json& payload, const char* fieldName)
{
    std::vector<uint32_t> taskIds;
    if (!payload.is_object() || !payload.contains(fieldName) || !payload[fieldName].is_array()) {
        return taskIds;
    }
    for (const auto& taskIdJson : payload[fieldName]) {
        taskIds.push_back(taskIdJson.get<uint32_t>());
    }
    return taskIds;
}

static bool
ParseDataSlice(const json& payload, const char* pipeName, AivSim::AivDataSlice& slice, std::string* errorMessage)
{
    if (!payload.is_object() || !payload.contains(pipeName) || !payload[pipeName].is_object()) {
        SetCommonError(errorMessage, std::string("task payload missing data slice pipe: ") + pipeName);
        return false;
    }

    const auto& sliceJson = payload[pipeName];
    slice = AivSim::AivDataSlice(
        static_cast<AivSim::AivBufferType>(sliceJson.value("bufferType", 0U)), sliceJson.value("offset", 0ULL),
        sliceJson.value("size", 0ULL));
    return true;
}

static bool ParseRuntimeTaskArray(
    const json& taskArray, std::vector<std::shared_ptr<AivSim::AivTask>>& tasks,
    std::map<uint32_t, std::shared_ptr<AivSim::AivTaskPipeBarrier>>& barrierTaskMap,
    std::vector<std::pair<std::shared_ptr<AivSim::AivTaskPipeBarrier>, std::vector<uint32_t>>>& pendingBarrierGroups,
    std::string* errorMessage)
{
    tasks.clear();
    if (!taskArray.is_array()) {
        SetCommonError(errorMessage, "runtime task array is not array");
        return false;
    }

    for (const auto& taskJson : taskArray) {
        const auto taskType = static_cast<AivSim::AivTaskType>(
            taskJson.value("taskType", static_cast<uint32_t>(AivSim::AivTaskType::INVALID_TYPE)));
        const json payload = taskJson.value("payload", json::object());

        std::shared_ptr<AivSim::AivTask> task;
        switch (taskType) {
            case AivSim::AivTaskType::MEM_COPY: {
                AivSim::AivDataSlice src;
                AivSim::AivDataSlice dst;
                if (!ParseDataSlice(payload, "src", src, errorMessage)
                    || !ParseDataSlice(payload, "dst", dst, errorMessage)) {
                    return false;
                }
                task = std::make_shared<AivSim::AivTaskMemCopy>(
                    payload.value("srcRank", UINT32_MAX), src, payload.value("dstRank", UINT32_MAX), dst);
                break;
            }
            case AivSim::AivTaskType::REDUCE: {
                AivSim::AivDataSlice src;
                AivSim::AivDataSlice dst;
                if (!ParseDataSlice(payload, "src", src, errorMessage)
                    || !ParseDataSlice(payload, "dst", dst, errorMessage)) {
                    return false;
                }
                task = std::make_shared<AivSim::AivTaskReduce>(
                    payload.value("srcRank", UINT32_MAX), src, payload.value("dstRank", UINT32_MAX), dst,
                    payload.value("dataType", 0U), payload.value("reduceOp", 0U));
                break;
            }
            case AivSim::AivTaskType::SET_FLAG:
                task = std::make_shared<AivSim::AivTaskSetFlag>(
                    static_cast<AscendC::pipe_t>(payload.value("srcPipe", static_cast<uint32_t>(AscendC::PIPE_ALL))),
                    static_cast<AscendC::pipe_t>(payload.value("dstPipe", static_cast<uint32_t>(AscendC::PIPE_ALL))),
                    payload.value("eventId", 0));
                break;
            case AivSim::AivTaskType::WAIT_FLAG:
                task = std::make_shared<AivSim::AivTaskWaitFlag>(
                    static_cast<AscendC::pipe_t>(payload.value("srcPipe", static_cast<uint32_t>(AscendC::PIPE_ALL))),
                    static_cast<AscendC::pipe_t>(payload.value("dstPipe", static_cast<uint32_t>(AscendC::PIPE_ALL))),
                    payload.value("eventId", 0));
                break;
            case AivSim::AivTaskType::PIPE_BARRIER:
                task = std::make_shared<AivSim::AivTaskPipeBarrier>(
                    static_cast<AscendC::pipe_t>(payload.value("pipeType", static_cast<uint32_t>(AscendC::PIPE_ALL))));
                break;
            case AivSim::AivTaskType::SYNC_ALL:
                task = std::make_shared<AivSim::AivTaskSyncAll>(payload.value("syncRound", UINT32_MAX));
                break;
            case AivSim::AivTaskType::SEND_FLAG:
                task = std::make_shared<AivSim::AivTaskSendFlag>(
                    payload.value("rank", UINT32_MAX), payload.value("commInfoOffset", 0ULL),
                    payload.value("flagValue", 0));
                break;
            case AivSim::AivTaskType::RECV_FLAG:
                task = std::make_shared<AivSim::AivTaskRecvFlag>(
                    payload.value("rank", UINT32_MAX), payload.value("commInfoOffset", 0ULL),
                    payload.value("targetValue", 0));
                break;
            default:
                SetCommonError(
                    errorMessage,
                    "unsupported AIV runtime task type: " + std::to_string(static_cast<uint32_t>(taskType)));
                return false;
        }

        task->SetTaskId(taskJson.value("taskId", 0U));
        task->SetRankId(taskJson.value("rankId", 0U));
        task->SetBlockId(taskJson.value("blockId", 0U));
        task->SetCurPipe(
            static_cast<AscendC::pipe_t>(taskJson.value("curPipe", static_cast<uint32_t>(AscendC::PIPE_ALL))));
        tasks.push_back(task);

        if (taskType == AivSim::AivTaskType::PIPE_BARRIER) {
            auto barrierTask = std::dynamic_pointer_cast<AivSim::AivTaskPipeBarrier>(task);
            barrierTaskMap[barrierTask->GetTaskId()] = barrierTask;
            pendingBarrierGroups.push_back({barrierTask, LoadTaskIdsFromPayload(payload, "barrierGroupTaskIds")});
        }
    }

    return true;
}

static bool ResolveBarrierGroups(
    const std::map<uint32_t, std::shared_ptr<AivSim::AivTaskPipeBarrier>>& barrierTaskMap,
    const std::vector<std::pair<std::shared_ptr<AivSim::AivTaskPipeBarrier>, std::vector<uint32_t>>>&
        pendingBarrierGroups,
    std::string* errorMessage)
{
    for (const auto& pendingBarrierGroup : pendingBarrierGroups) {
        const auto& barrierTask = pendingBarrierGroup.first;
        for (const auto barrierTaskId : pendingBarrierGroup.second) {
            auto barrierIt = barrierTaskMap.find(barrierTaskId);
            if (barrierIt == barrierTaskMap.end()) {
                SetCommonError(errorMessage, "failed to resolve barrier group taskId " + std::to_string(barrierTaskId));
                return false;
            }
            barrierTask->AddBarrierGroup(barrierIt->second);
        }
    }
    return true;
}

static bool ParseRuntimeRank(const json& rankJson, AivRuntimeTaskSnapshot& taskSnapshot, std::string* errorMessage)
{
    taskSnapshot.rankId = rankJson.value("rank", UINT32_MAX);
    taskSnapshot.rankSize = rankJson.value("rankSize", 0U);
    taskSnapshot.launchIndex = rankJson.value("launchIndex", 0U);
    taskSnapshot.blocks.clear();

    if (!rankJson.contains("aivCores") || !rankJson["aivCores"].is_array()) {
        SetCommonError(errorMessage, "runtime snapshot missing aivCores array");
        return false;
    }

    std::map<uint32_t, std::shared_ptr<AivSim::AivTaskPipeBarrier>> barrierTaskMap;
    std::vector<std::pair<std::shared_ptr<AivSim::AivTaskPipeBarrier>, std::vector<uint32_t>>> pendingBarrierGroups;
    for (const auto& blockJson : rankJson["aivCores"]) {
        AivRuntimeBlockTaskSnapshot block;
        block.blockIdx = blockJson.value("blockIdx", 0U);

        if (!ParseRuntimeTaskArray(
                blockJson.value("scalarTasks", json::array()), block.scalarTasks, barrierTaskMap, pendingBarrierGroups,
                errorMessage)) {
            return false;
        }
        if (!ParseRuntimeTaskArray(
                blockJson.value("mte2Tasks", json::array()), block.mte2Tasks, barrierTaskMap, pendingBarrierGroups,
                errorMessage)) {
            return false;
        }
        if (!ParseRuntimeTaskArray(
                blockJson.value("mte3Tasks", json::array()), block.mte3Tasks, barrierTaskMap, pendingBarrierGroups,
                errorMessage)) {
            return false;
        }

        taskSnapshot.blocks.push_back(std::move(block));
    }

    return ResolveBarrierGroups(barrierTaskMap, pendingBarrierGroups, errorMessage);
}

bool IsAivExpansionModeEnabled()
{
    const char* expansionMode = std::getenv("HCCL_OP_EXPANSION_MODE");
    return expansionMode != nullptr && std::string(expansionMode) == "AIV";
}

void AivTaskSnapshotLoader::SetError(std::string* errorMessage, const std::string& message)
{
    if (errorMessage != nullptr) {
        *errorMessage = message;
    }
}

bool AivTaskSnapshotLoader::LoadRuntimeTaskSnapshotByLaunchDirect(
    uint32_t rankId, uint32_t launchIndex, AivRuntimeTaskSnapshot& taskSnapshot, std::string* errorMessage)
{
    std::string filePath;
    if (!ResolveRankTaskFilePath(rankId, launchIndex, filePath, errorMessage)) {
        return false;
    }

    taskSnapshot.rankId = rankId;
    taskSnapshot.rankSize = 0;
    taskSnapshot.launchIndex = launchIndex;
    taskSnapshot.filePath = filePath;
    taskSnapshot.blocks.clear();

    std::ifstream ifs(filePath);
    if (!ifs.is_open()) {
        SetError(errorMessage, "failed to open file: " + filePath);
        return false;
    }

    const json rankJson = json::parse(ifs, nullptr, false);
    if (rankJson.is_discarded()) {
        SetError(errorMessage, "failed to parse file: " + filePath);
        return false;
    }

    const uint32_t fileRankId = rankJson.value("rank", UINT32_MAX);
    if (fileRankId != rankId) {
        SetError(
            errorMessage, "rank id mismatch in file: " + filePath + ", file rank=" + std::to_string(fileRankId)
                              + ", path rank=" + std::to_string(rankId));
        return false;
    }

    const uint32_t fileLaunchIndex = rankJson.value("launchIndex", UINT32_MAX);
    if (fileLaunchIndex != launchIndex) {
        SetError(
            errorMessage, "launchIndex mismatch in file: " + filePath + ", file launchIndex="
                              + std::to_string(fileLaunchIndex) + ", path launchIndex=" + std::to_string(launchIndex));
        return false;
    }

    if (!ParseRuntimeRank(rankJson, taskSnapshot, errorMessage)) {
        return false;
    }
    taskSnapshot.filePath = filePath;
    taskSnapshot.launchIndex = fileLaunchIndex;
    return true;
}
