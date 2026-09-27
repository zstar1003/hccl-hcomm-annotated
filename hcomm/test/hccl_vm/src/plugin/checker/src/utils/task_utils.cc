/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_utils.h"

#include <cstdint>
#include <iostream>
#include <queue>
#include <set>

#include "ccu_instr_info.h"
#include "checker_def.h"
#include "data_slice.h"
#include "error_codes.h"
#include "hccl_types.h"
#include "sim_log.h"
#include "log.h"
#include "storage_manager.h"
#include "task_ccu.h"
#include "task_meta_defs.h"

namespace HcclSim {
namespace {
    bool IsDigitString(const std::string& value, size_t start, size_t end)
    {
        if (start >= end || end > value.size()) {
            return false;
        }
        for (size_t index = start; index < end; ++index) {
            if (value[index] < '0' || value[index] > '9') {
                return false;
            }
        }
        return true;
    }

    bool TryParseMainTaskId(const std::string& taskId, RankId& rankId, uint64_t& nodeIndex)
    {
        if (taskId.size() < 4 || taskId[0] != 'r') {
            return false;
        }
        const size_t splitPos = taskId.find('n', 1);
        if (splitPos == std::string::npos || splitPos == 1 || splitPos + 1 >= taskId.size()) {
            return false;
        }
        if (!IsDigitString(taskId, 1, splitPos) || !IsDigitString(taskId, splitPos + 1, taskId.size())) {
            return false;
        }

        rankId = static_cast<RankId>(std::stoul(taskId.substr(1, splitPos - 1)));
        nodeIndex = std::stoull(taskId.substr(splitPos + 1));
        return true;
    }

    bool TryParseDerivedTaskId(const std::string& taskId, const std::string& baseTaskId, uint64_t& subIndex)
    {
        if (baseTaskId.empty() || taskId.size() <= baseTaskId.size() + 1) {
            return false;
        }
        if (taskId.compare(0, baseTaskId.size(), baseTaskId) != 0 || taskId[baseTaskId.size()] != '_') {
            return false;
        }

        const size_t suffixStart = baseTaskId.size() + 1;
        if (!IsDigitString(taskId, suffixStart, taskId.size())) {
            return false;
        }

        subIndex = std::stoull(taskId.substr(suffixStart));
        return true;
    }

    bool IsValidTaskNode(const TaskNode* node)
    {
        return node != nullptr && node->task != nullptr && node->rankIdx != static_cast<RankId>(-1);
    }

    void SeedMainGraphRankCounters(TaskNode* dummyStart, std::map<RankId, uint64_t>& rankNodeCounters)
    {
        if (dummyStart == nullptr) {
            return;
        }

        std::queue<TaskNode*> visitQueue;
        std::set<TaskNode*> visited;
        visitQueue.push(dummyStart);
        visited.insert(dummyStart);

        while (!visitQueue.empty()) {
            TaskNode* currentNode = visitQueue.front();
            visitQueue.pop();

            for (auto* child : currentNode->children) {
                if (child != nullptr && visited.insert(child).second) {
                    visitQueue.push(child);
                }
            }

            if (!IsValidTaskNode(currentNode)) {
                continue;
            }

            RankId rankId = static_cast<RankId>(-1);
            uint64_t nodeIndex = 0;
            if (TryParseMainTaskId(currentNode->task->GetTaskId(), rankId, nodeIndex)
                && rankId == currentNode->rankIdx) {
                rankNodeCounters[rankId] = std::max(rankNodeCounters[rankId], nodeIndex + 1);
            }
        }
    }

    void EnsureCcuSubGraphTaskIds(TaskStubCcuGraph* ccuGraphTask)
    {
        if (ccuGraphTask == nullptr || ccuGraphTask->ccuHeadTaskNode == nullptr) {
            return;
        }

        const std::string baseTaskId = ccuGraphTask->GetTaskId();
        if (baseTaskId.empty()) {
            return;
        }

        uint64_t nextSubIndex = 0;
        std::queue<TaskNode*> seedQueue;
        std::set<TaskNode*> seedVisited;
        seedQueue.push(ccuGraphTask->ccuHeadTaskNode);
        seedVisited.insert(ccuGraphTask->ccuHeadTaskNode);
        while (!seedQueue.empty()) {
            TaskNode* currentNode = seedQueue.front();
            seedQueue.pop();
            for (auto* child : currentNode->children) {
                if (child != nullptr && seedVisited.insert(child).second) {
                    seedQueue.push(child);
                }
            }
            if (!IsValidTaskNode(currentNode)) {
                continue;
            }

            uint64_t subIndex = 0;
            if (TryParseDerivedTaskId(currentNode->task->GetTaskId(), baseTaskId, subIndex)) {
                nextSubIndex = std::max(nextSubIndex, subIndex + 1);
            }
        }

        std::queue<TaskNode*> visitQueue;
        std::set<TaskNode*> visited;
        visitQueue.push(ccuGraphTask->ccuHeadTaskNode);
        visited.insert(ccuGraphTask->ccuHeadTaskNode);
        while (!visitQueue.empty()) {
            TaskNode* currentNode = visitQueue.front();
            visitQueue.pop();
            for (auto* child : currentNode->children) {
                if (child != nullptr && visited.insert(child).second) {
                    visitQueue.push(child);
                }
            }
            if (!IsValidTaskNode(currentNode)) {
                continue;
            }
            if (currentNode->task->GetTaskId().empty()) {
                AssignTaskId(currentNode->task, BuildDerivedTaskId(baseTaskId, nextSubIndex++));
            }
        }
    }
} // namespace

std::string BuildTaskId(RankId rankId, uint64_t nodeIndex) { return StringFormat("r%un%llu", rankId, nodeIndex); }

std::string BuildDerivedTaskId(const std::string& baseTaskId, uint64_t subIndex)
{
    if (baseTaskId.empty()) {
        return "";
    }
    return StringFormat("%s_%llu", baseTaskId.c_str(), subIndex);
}

void AssignTaskId(TaskStub* task, const std::string& taskId)
{
    if (task == nullptr || taskId.empty()) {
        return;
    }
    task->SetTaskId(taskId);
}

void EnsureTaskNodeIdsAssigned(TaskNode* dummyStart)
{
    if (dummyStart == nullptr) {
        return;
    }

    std::map<RankId, uint64_t> rankNodeCounters;
    SeedMainGraphRankCounters(dummyStart, rankNodeCounters);

    std::queue<TaskNode*> visitQueue;
    std::set<TaskNode*> visited;
    visitQueue.push(dummyStart);
    visited.insert(dummyStart);

    while (!visitQueue.empty()) {
        TaskNode* currentNode = visitQueue.front();
        visitQueue.pop();

        for (auto* child : currentNode->children) {
            if (child != nullptr && visited.insert(child).second) {
                visitQueue.push(child);
            }
        }

        if (!IsValidTaskNode(currentNode)) {
            continue;
        }

        if (currentNode->task->GetTaskId().empty()) {
            AssignTaskId(
                currentNode->task, BuildTaskId(currentNode->rankIdx, rankNodeCounters[currentNode->rankIdx]++));
        }

        if (currentNode->task->GetType() != TaskTypeStub::CCU_GRAPH) {
            continue;
        }

        TaskStubCcuGraph* ccuGraphTask = dynamic_cast<TaskStubCcuGraph*>(currentNode->task);
        EnsureCcuSubGraphTaskIds(ccuGraphTask);
    }
}

LinkProtoStub GetLinkProto(uint8_t commProtocol)
{
    if (commProtocol == CommProtocol::COMM_PROTOCOL_ROCE) {
        return LinkProtoStub::RDMA;
    }
    return LinkProtoStub::SDMA;
}

uint64_t CalcDataSize(HcclDataType dataType, uint64_t dataCount)
{
    if (dataType >= HCCL_DATA_TYPE_RESERVED) {
        // invalid data type
        HCCL_ERROR("[CalcDataSize] invalid dataType %d", dataType);
        return 0;
    }
    uint64_t dataTypeSize = CHECK_SIZE_TABLE[dataType];
    return dataTypeSize * dataCount;
}

// rankId, dieId, missionId
std::map<uint32_t, std::map<uint8_t, std::map<uint8_t, std::shared_ptr<HcclSim::TaskStubCcuGraph>>>> g_missionTask;

static HcclResult GetTaskDataSlice(uint32_t rankId, uint64_t addr, uint64_t size, DataSlice& dataSlice)
{
    uint32_t actualRank = 0;
    HcclResult ret = StorageManager::GetInstance().GetSlice(addr, size, dataSlice, &actualRank);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (actualRank != rankId) {
        HCCL_VM_ERROR(
            "{} Resolved data slice rank mismatch, expectedRank={}, actualRank={}, addr={}, size={}",
            MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID), rankId, actualRank, addr, size);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult
ConvertTask(const HcclSim::StorageManager& storage, HcclTaskMetaData hcclTask, std::shared_ptr<TaskStub>& task)
{
    task.reset();
    switch (hcclTask.taskType) {
        case HccLTaskMetaType::NOTIFY_WAIT: {
            RankId remoteRank = hcclTask.taskData.notify.srcRankId;
            if (remoteRank == hcclTask.rankId) {
                // Local
                task = std::make_shared<HcclSim::TaskStubLocalWaitFrom>(hcclTask.taskData.notify.notifyId);
            } else {
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.notify.protocol);
                HcclSim::LinkInfo link(linkType);
                task = std::make_shared<HcclSim::TaskStubWait>(remoteRank, link, hcclTask.taskData.notify.notifyId);
            }
            return HCCL_SUCCESS;
        }
        case HccLTaskMetaType::NOTIFY_RECORD: {
            RankId remoteRank = hcclTask.taskData.notify.dstRankId;
            if (remoteRank == hcclTask.rankId) {
                // Local
                task = std::make_shared<HcclSim::TaskStubLocalPostTo>(hcclTask.taskData.notify.notifyId);
            } else {
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.notify.protocol);
                HcclSim::LinkInfo link(linkType);
                task = std::make_shared<HcclSim::TaskStubPost>(remoteRank, link, hcclTask.taskData.notify.notifyId);
            }
            return HCCL_SUCCESS;
        }
        case HccLTaskMetaType::REDUCE: {
            RankId srcRank = hcclTask.taskData.reduce.srcRankId;
            RankId dstRank = hcclTask.taskData.reduce.dstRankId;
            uint64_t rankSrcOffset = hcclTask.taskData.reduce.srcOffset;
            uint64_t rankDstOffset = hcclTask.taskData.reduce.dstOffset;
            HcclDataType dataType = static_cast<HcclDataType>(hcclTask.taskData.reduce.dataType);
            uint64_t dataCount = hcclTask.taskData.reduce.dataCount;
            HcclReduceOp reduceOp = static_cast<HcclReduceOp>(hcclTask.taskData.reduce.reduceOp);

            DataSlice srcDataSlice;
            HcclResult ret = GetTaskDataSlice(srcRank, rankSrcOffset, dataCount, srcDataSlice);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            DataSlice dstDataSlice;
            ret = GetTaskDataSlice(dstRank, rankDstOffset, dataCount, dstDataSlice);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            if (srcRank == dstRank) {
                // Local
                task = std::make_shared<HcclSim::TaskStubLocalReduce>(srcDataSlice, dstDataSlice, dataType, reduceOp);
            } else if (srcRank == hcclTask.rankId) {
                // Write
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.transMem.protocol);
                HcclSim::LinkInfo link(linkType);
                task = std::make_shared<HcclSim::TaskStubWriteReduce>(
                    dstRank, link, srcDataSlice, dstDataSlice, dataType, reduceOp);
            } else if (dstRank == hcclTask.rankId) {
                // Read
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.transMem.protocol);
                HcclSim::LinkInfo link(linkType);
                task = std::make_shared<HcclSim::TaskStubReadReduce>(
                    srcRank, link, dstDataSlice, srcDataSlice, dataType, reduceOp);
            }
            return HCCL_SUCCESS;
        }
        case HccLTaskMetaType::MEM_CPY: {
            RankId srcRank = hcclTask.taskData.transMem.srcRankId;
            RankId dstRank = hcclTask.taskData.transMem.dstRankId;
            uint64_t rankSrcOffset = hcclTask.taskData.transMem.srcOffset;
            uint64_t rankDstOffset = hcclTask.taskData.transMem.dstOffset;
            uint64_t size = hcclTask.taskData.transMem.len;

            DataSlice srcDataSlice;
            HcclResult ret = GetTaskDataSlice(srcRank, rankSrcOffset, size, srcDataSlice);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            DataSlice dstDataSlice;
            ret = GetTaskDataSlice(dstRank, rankDstOffset, size, dstDataSlice);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            if (srcRank == dstRank) {
                // Local
                task = std::make_shared<HcclSim::TaskStubLocalCopy>(srcDataSlice, dstDataSlice);
            } else if (srcRank == hcclTask.rankId) {
                // Write
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.transMem.protocol);
                HcclSim::LinkInfo link(linkType);
                task = std::make_shared<HcclSim::TaskStubWrite>(dstRank, link, srcDataSlice, dstDataSlice);
            } else if (dstRank == hcclTask.rankId) {
                // Read
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.transMem.protocol);
                HcclSim::LinkInfo link(linkType);
                task = std::make_shared<HcclSim::TaskStubRead>(srcRank, link, dstDataSlice, srcDataSlice);
            }
            return HCCL_SUCCESS;
        }
        case HccLTaskMetaType::CCU_GRAPH: {
            auto dieId = hcclTask.taskData.ccu.dieId;
            auto missionId = hcclTask.taskData.ccu.missionId;
            // 组装SQE数据
            std::vector<HcclSim::CcuTaskParam> missionParam;
            HcclSim::CcuTaskParam ccuParam;
            ccuParam.dieId = dieId;
            ccuParam.missionId = missionId;
            ccuParam.timeout = hcclTask.taskData.ccu.timeout;
            ccuParam.instStartId = hcclTask.taskData.ccu.instStartId;
            ccuParam.instCnt = hcclTask.taskData.ccu.instCnt;
            ccuParam.key = hcclTask.taskData.ccu.key;
            ccuParam.argSize = hcclTask.taskData.ccu.argSize;
            memcpy(ccuParam.args, hcclTask.taskData.ccu.args, sizeof(uint64_t) * ccuParam.argSize);
            missionParam.push_back(ccuParam);
            HCCL_VM_INFO("rank {}, dieId= {}", hcclTask.rankId, static_cast<uint32_t>(dieId));
            HCCL_VM_INFO(
                "Get sqe info: missionId= {:d}, startId= {:d}, cnt= {:d}, argSize= {:d}", missionId,
                ccuParam.instStartId, ccuParam.instCnt, ccuParam.argSize);

            if (g_missionTask.find(hcclTask.rankId) != g_missionTask.end()) {
                auto rankMap = g_missionTask[hcclTask.rankId];
                if (rankMap.find(dieId) != rankMap.end()) {
                    auto dieMap = rankMap[dieId];
                    if (dieMap.find(missionId) != dieMap.end()) {
                        if (dieMap[missionId]->IsSameCcuGraph(ccuParam.instStartId) == true) {
                            dieMap[missionId]->AddCcuParams(ccuParam);
                            return HCCL_SUCCESS;
                        } else {
                            g_missionTask[hcclTask.rankId][dieId].erase(missionId);
                        }
                    }
                }
            }

            // 组装微码指令数据
            auto hvmInstrData = storage.GetHvmInstrData();
            for (auto& ccuInstr : hvmInstrData.instr_data) {
                if (ccuInstr.desc.rank_id != hcclTask.rankId || ccuInstr.desc.die_id != dieId) {
                    continue;
                }
                hcomm::CcuRep::CcuInstrInfo ccuInstrInfo;
                ccuInstrInfo.instrVec = ccuInstr.data;
                HCCL_VM_INFO(
                    "Create new ccu graph base node, rank_id= {}, die_id= {} - {}", ccuInstr.desc.rank_id,
                    static_cast<uint32_t>(ccuInstr.desc.die_id), static_cast<uint32_t>(dieId));
                auto taskPtr
                    = std::make_shared<HcclSim::TaskStubCcuGraph>(ccuInstrInfo, missionParam, ccuInstr.desc.rank_id);
                g_missionTask[hcclTask.rankId][dieId][missionId] = taskPtr;
                task = taskPtr;
                return HCCL_SUCCESS;
            }
            break;
        }
        default: {
            HCCL_ERROR("[ConvertTask] taskType unknown %d", hcclTask.taskType);
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ConvertTaskQueue(AllRankTaskQueues& allRankTaskQueues)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    g_missionTask.clear(); // 每次转换前清理之前的状态，防止重复数据干扰转换逻辑
    auto taskMetaVec = storage.GetHvmTaskMetaData().task_meta;
    uint32_t size = taskMetaVec.size();
    uint32_t outputInterval = std::max(1u, size / 10);  // 十分之一数据量
    outputInterval = std::min(outputInterval, 100000u); // 不超过10万条
    std::map<RankId, uint64_t> rankNodeCounters;
    for (uint32_t i = 0; i < size; i++) {
        std::shared_ptr<TaskStub> task;
        HcclResult ret = ConvertTask(storage, taskMetaVec[i], task);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "Failed to convert checker task metadata, taskIndex={}, rankId={}, ret={}", i, taskMetaVec[i].rankId,
                static_cast<uint32_t>(ret));
            g_missionTask.clear();
            return ret;
        }
        if (task == nullptr) {
            continue;
        }
        if (size > 10000 && i % outputInterval == 0) {
            HCCL_VM_INFO("Processing at index: {:d} / {:d} ({:f}% complete)", i, size, i * 100.0 / size);
        }
        uint32_t rankId = taskMetaVec[i].rankId;
        uint32_t streamId = (taskMetaVec[i].streamId);

        if (allRankTaskQueues.find(rankId) == allRankTaskQueues.end()) {
            allRankTaskQueues[rankId] = std::vector<std::vector<std::shared_ptr<HcclSim::TaskStub>>>{};
        }

        if (allRankTaskQueues[rankId].size() <= streamId) {
            allRankTaskQueues[rankId].resize(streamId + 1);
        }

        AssignTaskId(task.get(), BuildTaskId(rankId, rankNodeCounters[rankId]++));
        allRankTaskQueues[rankId][streamId].push_back(task);
        // 如果是CCU子图节点，则在其后加上一个子图分隔节点：防止后续广度优先搜索错误，生成错误语义
        if (taskMetaVec[i].taskType == HccLTaskMetaType::CCU_GRAPH) {
            auto graphSeparate = std::make_shared<TaskStubGraphSeparate>();
            AssignTaskId(graphSeparate.get(), BuildTaskId(rankId, rankNodeCounters[rankId]++));
            allRankTaskQueues[rankId][streamId].push_back(graphSeparate);
        }
    }
    return HCCL_SUCCESS;
}
} // namespace HcclSim
