/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_generator_v3.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <sstream>
#include <utility>

#include "aiv_graph_generator_v3/aiv_task_transform_v3.h"
#include "ccu_graph_generator_v3/ccu_task_transform_v3.h"
#include "sim_log.h"
#include "storage_manager.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        using SeenLocalRecords = std::vector<NodeId>;
        using SeenInterRankRecords = std::map<RankId, std::map<RankId, std::vector<NodeId>>>;

        std::string NodeIdsToString(const std::vector<TaskNode*>& nodes)
        {
            std::ostringstream os;
            os << "[";
            bool first = true;
            for (const TaskNode* node : nodes) {
                if (!first) {
                    os << ",";
                }
                first = false;
                os << ((node == nullptr) ? "null" : std::to_string(node->GetNodeId()));
            }
            os << "]";
            return os.str();
        }

        bool IsLocalNotify(const AicpuNotify& notify) { return notify.recordRankId == notify.waitRankId; }

        bool IsInterRankNotify(const AicpuNotify& notify) { return notify.recordRankId != notify.waitRankId; }

        const AicpuNotify* GetRecordNotify(const TaskNode* node)
        {
            if (node == nullptr || node->GetType() != TaskType::RECORD) {
                return nullptr;
            }
            const auto* record = dynamic_cast<const TaskRecordAICPU*>(node);
            return (record == nullptr) ? nullptr : &record->GetNotify();
        }

        const AicpuNotify* GetWaitNotify(const TaskNode* node)
        {
            if (node == nullptr || node->GetType() != TaskType::WAIT) {
                return nullptr;
            }
            const auto* wait = dynamic_cast<const TaskWaitAICPU*>(node);
            return (wait == nullptr) ? nullptr : &wait->GetNotify();
        }

        bool IsNotifyIdPeer(const AicpuNotify& recordNotify, const AicpuNotify& waitNotify)
        {
            return recordNotify.notifyId == waitNotify.notifyId;
        }

        bool StreamHasAivGraph(const TaskGraphGeneratorV3* graph, const std::vector<NodeId>& stream)
        {
            if (graph == nullptr) {
                return false;
            }
            return std::any_of(stream.begin(), stream.end(), [graph](NodeId nodeId) {
                const TaskNode* node = graph->GetNode(nodeId);
                return node != nullptr && node->GetType() == TaskType::AIV_GRAPH;
            });
        }
    } // namespace

    bool TaskGraphGeneratorV3::IsValidNodeId(NodeId nodeId) const
    {
        return nodeId >= 0 && static_cast<size_t>(nodeId) < nodes_.size();
    }

    bool TaskGraphGeneratorV3::IsMainStartNodeId(NodeId nodeId) const
    {
        return mainStart_ != nullptr && nodeId == mainStartNodeId_;
    }

    StorageManager& TaskGraphGeneratorV3::GetStorageManager() const
    {
        return storage_ == nullptr ? StorageManager::GetInstance() : *storage_;
    }

    TaskNode* TaskGraphGeneratorV3::GetNode(NodeId nodeId)
    {
        if (IsMainStartNodeId(nodeId)) {
            return mainStart_.get();
        }
        if (!IsValidNodeId(nodeId)) {
            return nullptr;
        }
        return nodes_[static_cast<size_t>(nodeId)].get();
    }

    const TaskNode* TaskGraphGeneratorV3::GetNode(NodeId nodeId) const
    {
        if (IsMainStartNodeId(nodeId)) {
            return mainStart_.get();
        }
        if (!IsValidNodeId(nodeId)) {
            return nullptr;
        }
        return nodes_[static_cast<size_t>(nodeId)].get();
    }

    HcclResult TaskGraphGeneratorV3::AddEdge(NodeId parentNodeId, NodeId childNodeId)
    {
        TaskNode* childNode = GetNode(childNodeId);
        if (childNode == nullptr || IsMainStartNodeId(childNodeId)) {
            return HCCL_E_PTR;
        }

        TaskNode* parentNode = GetNode(parentNodeId);
        if (parentNode == nullptr) {
            return HCCL_E_PTR;
        }

        parentNode->AddChild(childNode);
        childNode->AddParent(parentNode);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::RemoveEdge(NodeId parentNodeId, NodeId childNodeId)
    {
        TaskNode* parentNode = GetNode(parentNodeId);
        TaskNode* childNode = GetNode(childNodeId);
        if (parentNode == nullptr || childNode == nullptr) {
            HCCL_VM_ERROR(
                "{} Failed to remove one graph edge because the parent or child node does not "
                "exist, parentNodeId={}, childNodeId={}, parentNode={}, childNode={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), parentNodeId, childNodeId,
                parentNode == nullptr ? "null" : parentNode->Describe(),
                childNode == nullptr ? "null" : childNode->Describe());
            return HCCL_E_PARA;
        }

        (void)parentNode->RemoveChild(childNode);
        (void)childNode->RemoveParent(parentNode);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::AppendGeneratedNode(
        std::unique_ptr<TaskNode> node, const TaskPosition& position, NodeId& nodeId)
    {
        if (node == nullptr) {
            return HCCL_E_MEMORY;
        }
        if (nodes_.size() >= MAX_NODE_COUNT) {
            return HCCL_E_MEMORY;
        }

        nodeId = static_cast<NodeId>(nodes_.size());
        node->SetNodeId(nodeId);
        node->SetPosition(position);
        nodes_.emplace_back(std::move(node));
        return HCCL_SUCCESS;
    }

    bool TaskGraphGeneratorV3::HasPath(NodeId fromNodeId, NodeId toNodeId) const
    {
        if (fromNodeId == toNodeId) {
            return true;
        }
        const TaskNode* fromNode = GetNode(fromNodeId);
        const TaskNode* toNode = GetNode(toNodeId);
        if (fromNode == nullptr || toNode == nullptr) {
            return false;
        }

        std::vector<const TaskNode*> nodeStack;
        std::vector<uint8_t> visitedNodes(nodes_.size(), 0);
        bool mainStartVisited = false;
        auto markVisited = [this, &visitedNodes, &mainStartVisited](const TaskNode* node) {
            if (node == nullptr) {
                return false;
            }

            const NodeId nodeId = node->GetNodeId();
            if (IsMainStartNodeId(nodeId)) {
                if (mainStartVisited) {
                    return false;
                }
                mainStartVisited = true;
                return true;
            }

            if (!IsValidNodeId(nodeId)) {
                return false;
            }

            const size_t index = static_cast<size_t>(nodeId);
            if (visitedNodes[index] != 0) {
                return false;
            }

            visitedNodes[index] = 1;
            return true;
        };

        nodeStack.push_back(fromNode);
        (void)markVisited(fromNode);
        while (!nodeStack.empty()) {
            const TaskNode* currNode = nodeStack.back();
            nodeStack.pop_back();
            for (const TaskNode* childNode : currNode->GetChildren()) {
                if (childNode == nullptr) {
                    continue;
                }
                if (childNode == toNode) {
                    return true;
                }
                if (markVisited(childNode)) {
                    nodeStack.push_back(childNode);
                }
            }
        }
        return false;
    }

    const std::vector<TaskNode*>& TaskGraphGeneratorV3::GetParents(NodeId nodeId) const
    {
        static const std::vector<TaskNode*> EMPTY_NODES;
        const TaskNode* node = GetNode(nodeId);
        return (node == nullptr) ? EMPTY_NODES : node->GetParents();
    }

    const std::vector<TaskNode*>& TaskGraphGeneratorV3::GetChildren(NodeId nodeId) const
    {
        static const std::vector<TaskNode*> EMPTY_NODES;
        const TaskNode* node = GetNode(nodeId);
        return (node == nullptr) ? EMPTY_NODES : node->GetChildren();
    }

    size_t TaskGraphGeneratorV3::CountEdges() const
    {
        size_t edgeCount = (mainStart_ == nullptr) ? 0U : mainStart_->GetChildren().size();
        for (const auto& nodeOwner : nodes_) {
            if (nodeOwner != nullptr) {
                edgeCount += nodeOwner->GetChildren().size();
            }
        }
        return edgeCount;
    }

    HcclResult TaskGraphGeneratorV3::PushNextNode(
        const RankNodeQueues& rankTaskQueues, NodeId currNodeId, std::vector<NodeId>& nodeQue) const
    {
        const TaskNode* node = GetNode(currNodeId);
        if (node == nullptr) {
            return HCCL_E_PTR;
        }

        const StreamId streamId = node->GetPosition().streamId;
        if (streamId >= rankTaskQueues.size()) {
            return HCCL_E_PARA;
        }

        const auto& stream = rankTaskQueues[streamId];
        const auto iter = std::find(stream.begin(), stream.end(), currNodeId);
        if (iter == stream.end()) {
            return HCCL_E_PARA;
        }

        const auto nextIter = iter + 1;
        if (nextIter != stream.end()) {
            nodeQue.push_back(*nextIter);
        }
        return HCCL_SUCCESS;
    }

    bool TaskGraphGeneratorV3::IsExecutable(NodeId nodeId, const std::vector<uint8_t>& execFlags) const
    {
        if (!IsValidNodeId(nodeId) || static_cast<size_t>(nodeId) >= execFlags.size()) {
            return false;
        }

        for (const TaskNode* parentNode : GetParents(nodeId)) {
            if (parentNode == nullptr) {
                return false;
            }
            const NodeId parentNodeId = parentNode->GetNodeId();
            if (IsMainStartNodeId(parentNodeId)) {
                continue;
            }
            if (!IsValidNodeId(parentNodeId) || static_cast<size_t>(parentNodeId) >= execFlags.size()
                || execFlags[static_cast<size_t>(parentNodeId)] == 0) {
                return false;
            }
        }
        return true;
    }

    HcclResult TaskGraphGeneratorV3::ExecuteNode(
        NodeId nodeId, std::vector<NodeId>& graphNodeQue, std::vector<uint8_t>& execFlags,
        std::vector<uint8_t>& traverseFlags) const
    {
        if (IsMainStartNodeId(nodeId)) {
            for (const TaskNode* childNode : GetChildren(nodeId)) {
                if (childNode == nullptr) {
                    return HCCL_E_PTR;
                }
                const NodeId childNodeId = childNode->GetNodeId();
                if (!IsValidNodeId(childNodeId) || static_cast<size_t>(childNodeId) >= traverseFlags.size()) {
                    return HCCL_E_PTR;
                }
                if (traverseFlags[static_cast<size_t>(childNodeId)] == 0) {
                    traverseFlags[static_cast<size_t>(childNodeId)] = 1;
                    graphNodeQue.push_back(childNodeId);
                }
            }
            return HCCL_SUCCESS;
        }

        if (!IsValidNodeId(nodeId) || static_cast<size_t>(nodeId) >= execFlags.size()
            || static_cast<size_t>(nodeId) >= traverseFlags.size()) {
            return HCCL_E_PTR;
        }

        execFlags[static_cast<size_t>(nodeId)] = 1;
        for (const TaskNode* childNode : GetChildren(nodeId)) {
            if (childNode == nullptr) {
                return HCCL_E_PTR;
            }
            const NodeId childNodeId = childNode->GetNodeId();
            if (!IsValidNodeId(childNodeId) || static_cast<size_t>(childNodeId) >= traverseFlags.size()) {
                return HCCL_E_PTR;
            }
            if (traverseFlags[static_cast<size_t>(childNodeId)] == 0) {
                traverseFlags[static_cast<size_t>(childNodeId)] = 1;
                graphNodeQue.push_back(childNodeId);
            }
        }
        return HCCL_SUCCESS;
    }

    void TaskGraphGeneratorV3::Reset()
    {
        mainStartNodeId_ = INVALID_NODE_ID;
        mainStart_.reset();
        taskQueues_.clear();
        nodes_.clear();
        hasAiv_ = false;
        hasCcu_ = false;
        aivExpandStats_ = AivExpandStats{};
        ccuExpandStats_ = CcuExpandStats{};
        g_checkerAivUbBufferSize = 0;
        g_checkerAivCommInfoSize = 0;
    }

    HcclResult TaskGraphGeneratorV3::GenGraph(
        std::vector<std::unique_ptr<TaskNode>> translatedNodes, AllRankNodeQueues translatedTaskQueues)
    {
        Reset();
        HCCL_VM_INFO(
            "Start building the CheckerV3 graph from translated nodes, rankCount={}, nodeCount={}",
            translatedTaskQueues.size(), translatedNodes.size());
        if (translatedNodes.empty() || translatedTaskQueues.empty()) {
            HCCL_VM_ERROR(
                "{} Checker get empty task queue, please check if the HCCL-VM end normally, rankCount={}, nodeCount={}",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), translatedTaskQueues.size(),
                translatedNodes.size());
            return HCCL_E_PARA;
        }
        nodes_ = std::move(translatedNodes);
        taskQueues_ = std::move(translatedTaskQueues);
        hasAiv_ = std::any_of(nodes_.begin(), nodes_.end(), [](const std::unique_ptr<TaskNode>& node) {
            return node != nullptr && node->GetType() == TaskType::AIV_GRAPH;
        });
        hasCcu_ = std::any_of(nodes_.begin(), nodes_.end(), [](const std::unique_ptr<TaskNode>& node) {
            return node != nullptr && node->GetType() == TaskType::CCU_GRAPH;
        });

        HcclResult ret = CreateMainStartNode();
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        return BuildDagEdges();
    }

    HcclResult TaskGraphGeneratorV3::CreateMainStartNode()
    {
        auto mainStart = std::make_unique<TaskStart>(BoundaryType::MAIN_GRAPH);
        TaskPosition startPosition;
        mainStart->SetNodeId(MAIN_START_NODE_ID);
        mainStart->SetPosition(startPosition);
        mainStartNodeId_ = MAIN_START_NODE_ID;
        mainStart_ = std::move(mainStart);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::BuildDagEdges()
    {
        HCCL_VM_INFO(
            "Start building graph edges, rankCount={}, nodeCount={}, mainStartNodeId={}", taskQueues_.size(),
            nodes_.size(), mainStartNodeId_);

        HcclResult ret = HCCL_SUCCESS;
        for (const auto& rankEntry : taskQueues_) {
            ret = GenGraph4Rank(rankEntry.first, rankEntry.second);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        for (const auto& rankEntry : taskQueues_) {
            ret = AddLocalNotifyEdges(rankEntry.first, rankEntry.second);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        ret = AddInterRankNotifyEdges();
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        if (hasAiv_) {
            ret = ExpandAivSubGraphs();
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        if (hasCcu_) {
            ret = ExpandCcuSubGraphs();
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        HCCL_VM_INFO(
            "Finished building graph edges, nodeCount={}, edgeCount={}, mainStartNodeId={}", nodes_.size(),
            CountEdges(), mainStartNodeId_);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::ExpandAivSubGraphs()
    {
        const size_t originalNodeCount = nodes_.size();
        AivExpandStats stats;
        StorageManager& storage = GetStorageManager();
        std::vector<TaskAivGraph*> aivGraphs;

        for (size_t nodeIndex = 0; nodeIndex < originalNodeCount; ++nodeIndex) {
            TaskNode* node = nodes_[nodeIndex].get();
            if (node == nullptr || node->GetType() != TaskType::AIV_GRAPH) {
                continue;
            }

            auto* aivGraph = dynamic_cast<TaskAivGraph*>(node);
            if (aivGraph == nullptr) {
                HCCL_VM_ERROR(
                    "{} One node expected to be an AIV subgraph entry is actually another node "
                    "type, nodeId={}, node={}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR),
                    node == nullptr ? std::string("null") : std::to_string(node->GetNodeId()),
                    node == nullptr ? "node=null" : node->Describe());
                return HCCL_E_INTERNAL;
            }
            aivGraphs.push_back(aivGraph);
        }

        AivGraphsGenerateInputV3 input;
        input.graph = this;
        input.aivGraphs = aivGraphs;
        input.storage = &storage;
        AivGraphGenerateOutputV3 result;
        HcclResult ret = ExpandAivGraphsV3(input, result);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to expand AIV Graph nodes, "
                "aivGraphCount={}, ret={}",
                MakeErrorCodeText(ErrorCode::GRAPH_TRANSLATE_FAILED), aivGraphs.size(), static_cast<uint32_t>(ret));
            return ret;
        }

        stats.graphCount = aivGraphs.size();
        stats.internalNodeCount = result.internalNodeCount;
        stats.setWaitEdgeCount = result.setWaitEdgeCount;
        stats.pipeBarrierMergeCount = result.pipeBarrierMergeCount;
        stats.syncAllMergeCount = result.syncAllMergeCount;
        stats.sendRecvEdgeCount = result.sendRecvEdgeCount;
        stats.taskJsonTotalTaskCount = result.taskJsonTotalTaskCount;
        stats.dagNodeCountBeforeCpGmMerge = result.dagNodeCountBeforeCpGmMerge;
        stats.dagNodeCountAfterCpGmMerge = result.dagNodeCountAfterCpGmMerge;
        stats.cpGmLoopMergeCount = result.cpGmLoopMergeCount;
        stats.cpGmMergedIterationCount = result.cpGmMergedIterationCount;
        stats.cpGmMergedOriginalNodeCount = result.cpGmMergedOriginalNodeCount;
        stats.cpGmGeneratedNodeCount = result.cpGmGeneratedNodeCount;
        stats.cpGmInactiveNodeCount = result.cpGmInactiveNodeCount;
        stats.totalExpandNs = result.expandNs;
        aivExpandStats_ = stats;
        HCCL_VM_INFO(
            "Finished expanding AIV Graph nodes:\n"
            "  graph: aivGraphCount={}, internalNodeCount={}\n"
            "  edges: setWaitEdgeCount={}, pipeBarrierMergeCount={}, syncAllMergeCount={}, sendRecvEdgeCount={}\n"
            "  tasks: taskJsonTotalTaskCount={}, dagNodeCountBeforeCpGmMerge={}, dagNodeCountAfterCpGmMerge={}\n"
            "  merge: cpGmLoopMergeCount={}, cpGmMergedIterationCount={}, cpGmMergedOriginalNodeCount={}\n"
            "  nodes: cpGmGeneratedNodeCount={}, cpGmInactiveNodeCount={}\n"
            "  buf: ubBufferSize={}, aivCommInfoSize={}\n"
            "  time: expandTotalMs={}",
            stats.graphCount, stats.internalNodeCount, stats.setWaitEdgeCount, stats.pipeBarrierMergeCount,
            stats.syncAllMergeCount, stats.sendRecvEdgeCount, stats.taskJsonTotalTaskCount,
            stats.dagNodeCountBeforeCpGmMerge, stats.dagNodeCountAfterCpGmMerge, stats.cpGmLoopMergeCount,
            stats.cpGmMergedIterationCount, stats.cpGmMergedOriginalNodeCount, stats.cpGmGeneratedNodeCount,
            stats.cpGmInactiveNodeCount, g_checkerAivUbBufferSize, g_checkerAivCommInfoSize,
            stats.totalExpandNs / 1000000ULL);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::ExpandCcuSubGraphs()
    {
        const size_t originalNodeCount = nodes_.size();
        CcuExpandStats stats;
        StorageManager& storage = GetStorageManager();
        std::vector<TaskCcuGraph*> ccuGraphs;

        for (size_t nodeIndex = 0; nodeIndex < originalNodeCount; ++nodeIndex) {
            TaskNode* node = nodes_[nodeIndex].get();
            if (node == nullptr || node->GetType() != TaskType::CCU_GRAPH) {
                continue;
            }

            auto* ccuGraph = dynamic_cast<TaskCcuGraph*>(node);
            if (ccuGraph == nullptr) {
                HCCL_VM_ERROR(
                    "{} One node expected to be a CCU subgraph entry is actually another node "
                    "type, nodeId={}, node={}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR),
                    node == nullptr ? std::string("null") : std::to_string(node->GetNodeId()),
                    node == nullptr ? "node=null" : node->Describe());
                return HCCL_E_INTERNAL;
            }

            ccuGraphs.push_back(ccuGraph);
        }

        CcuGraphsGenerateInputV3 input;
        input.graph = this;
        input.ccuGraphs = ccuGraphs;
        input.storage = &storage;
        CcuGraphGenerateOutputV3 result;
        HcclResult ret = ExpandCcuGraphsV3(input, result);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to expand CCU Graph nodes, "
                "ccuGraphCount={}, ret={}",
                MakeErrorCodeText(ErrorCode::GRAPH_TRANSLATE_FAILED), ccuGraphs.size(), static_cast<uint32_t>(ret));
            return ret;
        }
        stats.graphCount = ccuGraphs.size();
        stats.internalNodeCount = result.internalNodeCount;
        stats.recordWaitEdgeCount = result.recordWaitEdgeCount;
        stats.totalExpandNs = result.expandNs;

        ccuExpandStats_ = stats;
        HCCL_VM_INFO(
            "Finished expanding CCU Graph nodes, ccuGraphCount={}, internalNodeCount={}, "
            "recordWaitEdgeCount={}, expandTotalMs={}",
            stats.graphCount, stats.internalNodeCount, stats.recordWaitEdgeCount, stats.totalExpandNs / 1000000ULL);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::ValidateTranslatedNodes() const
    {
        if (nodes_.size() >= MAX_NODE_COUNT) {
            return HCCL_E_MEMORY;
        }

        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (nodes_[i] == nullptr || nodes_[i]->GetNodeId() != static_cast<NodeId>(i)) {
                return HCCL_E_PARA;
            }
        }

        for (const auto& rankEntry : taskQueues_) {
            for (const auto& stream : rankEntry.second) {
                for (const NodeId nodeId : stream) {
                    if (!IsValidNodeId(nodeId)) {
                        return HCCL_E_PARA;
                    }
                }
            }
        }

        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::GenGraph4Rank(RankId rankId, const RankNodeQueues& rankTaskQueues)
    {
        size_t nonEmptyStreamCount = 0;
        size_t taskNodeCount = 0;
        size_t startEdgeCount = 0;
        size_t streamOrderEdgeCount = 0;
        // mainStart连接每个rank的主流的第一个taskNode
        if (!rankTaskQueues.empty() && !rankTaskQueues[0].empty()) {
            HcclResult ret = AddEdge(mainStartNodeId_, rankTaskQueues[0].front());
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            startEdgeCount = 1;
        }

        for (size_t streamIndex = 0; streamIndex < rankTaskQueues.size(); ++streamIndex) {
            const auto& stream = rankTaskQueues[streamIndex];
            if (!stream.empty()) {
                ++nonEmptyStreamCount;
            }
            taskNodeCount += stream.size();
            if (hasAiv_ && streamIndex != 0 && !stream.empty() && StreamHasAivGraph(this, stream)) {
                HcclResult ret = AddEdge(mainStartNodeId_, stream.front());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ++startEdgeCount;
            }
            for (size_t i = 1; i < stream.size(); ++i) {
                HcclResult ret = AddEdge(stream[i - 1], stream[i]);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ++streamOrderEdgeCount;
            }
        }
        HCCL_VM_DEBUG(
            "Built per-rank skeleton edges, rankId={}, nonEmptyStreamCount={}, taskNodeCount={}, "
            "startEdgeCount={}, streamOrderEdgeCount={}",
            rankId, nonEmptyStreamCount, taskNodeCount, startEdgeCount, streamOrderEdgeCount);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::AddLocalNotifyEdges(RankId rankId, const RankNodeQueues& rankTaskQueues)
    {
        std::vector<NodeId> rankNodeQue;
        SeenLocalRecords seenLocalRecords;
        uint64_t unmatchedCnt = 0;
        size_t matchedEdgeCount = 0;

        if (!rankTaskQueues.empty() && !rankTaskQueues[0].empty()) {
            rankNodeQue.push_back(rankTaskQueues[0].front());
        }
        for (size_t streamIndex = 1; streamIndex < rankTaskQueues.size(); ++streamIndex) {
            if (!rankTaskQueues[streamIndex].empty()) {
                rankNodeQue.push_back(rankTaskQueues[streamIndex].front());
            }
        }

        while (!rankNodeQue.empty()) {
            if (unmatchedCnt >= rankNodeQue.size()) {
                const TaskNode* node = GetNode(rankNodeQue.front());
                HCCL_VM_ERROR(
                    "{} Local Record/Wait matching is stuck on this rank. Some Wait tasks are "
                    "still blocked, but no new local Record task can unblock them, rankId={}, firstBlockedWaitNode={}, "
                    "blockedWaitNodeCount={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_DEADLOCK), rankId, node == nullptr ? "null" : node->Describe(),
                    rankNodeQue.size());
                return HCCL_E_INTERNAL;
            }

            const NodeId currNodeId = rankNodeQue.front();
            rankNodeQue.erase(rankNodeQue.begin());
            const TaskNode* currNode = GetNode(currNodeId);
            if (currNode == nullptr) {
                return HCCL_E_PTR;
            }

            const AicpuNotify* recordNotify = GetRecordNotify(currNode);
            if (recordNotify != nullptr && IsLocalNotify(*recordNotify)) {
                HcclResult ret = PushNextNode(rankTaskQueues, currNodeId, rankNodeQue);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                seenLocalRecords.push_back(currNodeId);
                unmatchedCnt = 0;
                continue;
            }

            const AicpuNotify* waitNotify = GetWaitNotify(currNode);
            if (waitNotify != nullptr && IsLocalNotify(*waitNotify)) {
                bool matched = false;
                auto recordIter = seenLocalRecords.begin();
                for (; recordIter != seenLocalRecords.end(); ++recordIter) {
                    const AicpuNotify* seenRecordNotify = GetRecordNotify(GetNode(*recordIter));
                    if (seenRecordNotify != nullptr && IsNotifyIdPeer(*seenRecordNotify, *waitNotify)) {
                        HcclResult ret = AddEdge(*recordIter, currNodeId);
                        if (ret != HCCL_SUCCESS) {
                            return ret;
                        }
                        seenLocalRecords.erase(recordIter);
                        ret = PushNextNode(rankTaskQueues, currNodeId, rankNodeQue);
                        if (ret != HCCL_SUCCESS) {
                            return ret;
                        }
                        ++matchedEdgeCount;
                        unmatchedCnt = 0;
                        matched = true;
                        break;
                    }
                }

                if (!matched) {
                    rankNodeQue.push_back(currNodeId);
                    ++unmatchedCnt;
                }
                continue;
            }

            HcclResult ret = PushNextNode(rankTaskQueues, currNodeId, rankNodeQue);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            unmatchedCnt = 0;
        }

        if (!seenLocalRecords.empty()) {
            const TaskNode* node = GetNode(seenLocalRecords.front());
            HCCL_VM_ERROR(
                "{} Found local Record tasks that were never consumed by any local Wait task, "
                "rankId={}, firstUnconsumedRecordNode={}, unconsumedRecordCount={}",
                MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED), rankId, node == nullptr ? "node=null" : node->Describe(),
                seenLocalRecords.size());
            return HCCL_E_INTERNAL;
        }

        HCCL_VM_DEBUG(
            "Finished matching local Record/Wait edges on one rank, rankId={}, "
            "matchedRecordWaitEdgeCount={}",
            rankId, matchedEdgeCount);
        return HCCL_SUCCESS;
    }

    HcclResult TaskGraphGeneratorV3::AddInterRankNotifyEdges()
    {
        SeenInterRankRecords seenInterRankRecords;
        std::vector<NodeId> graphNodeQue;
        std::vector<uint8_t> execFlags(nodes_.size(), 0);
        std::vector<uint8_t> traverseFlags(nodes_.size(), 0);
        uint64_t unmatchedCnt = 0;
        size_t matchedEdgeCount = 0;

        HcclResult ret = ExecuteNode(mainStartNodeId_, graphNodeQue, execFlags, traverseFlags);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        while (!graphNodeQue.empty()) {
            if (unmatchedCnt >= graphNodeQue.size()) {
                const TaskNode* node = GetNode(graphNodeQue.front());
                HCCL_VM_ERROR(
                    "{} Cross-rank Record/Wait matching is stuck. Some Wait tasks are still "
                    "blocked, but no new cross-rank Record task can unblock them, firstBlockedWaitNode={}, "
                    "blockedWaitNodeCount={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_DEADLOCK), node == nullptr ? "node=null" : node->Describe(),
                    graphNodeQue.size());
                return HCCL_E_INTERNAL;
            }

            const NodeId currNodeId = graphNodeQue.front();
            graphNodeQue.erase(graphNodeQue.begin());
            const TaskNode* currNode = GetNode(currNodeId);
            if (currNode == nullptr) {
                return HCCL_E_PTR;
            }

            if (!IsExecutable(currNodeId, execFlags)) {
                graphNodeQue.push_back(currNodeId);
                ++unmatchedCnt;
                continue;
            }

            const AicpuNotify* recordNotify = GetRecordNotify(currNode);
            if (recordNotify != nullptr && IsInterRankNotify(*recordNotify)) {
                seenInterRankRecords[recordNotify->recordRankId][recordNotify->waitRankId].push_back(currNodeId);
                ret = ExecuteNode(currNodeId, graphNodeQue, execFlags, traverseFlags);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                unmatchedCnt = 0;
                continue;
            }

            const AicpuNotify* waitNotify = GetWaitNotify(currNode);
            if (waitNotify != nullptr && IsInterRankNotify(*waitNotify)) {
                bool matched = false;
                auto rankIter = seenInterRankRecords.find(waitNotify->recordRankId);
                if (rankIter != seenInterRankRecords.end()) {
                    auto peerIter = rankIter->second.find(waitNotify->waitRankId);
                    if (peerIter != rankIter->second.end()) {
                        auto recordIter = peerIter->second.begin();
                        for (; recordIter != peerIter->second.end(); ++recordIter) {
                            const AicpuNotify* seenRecordNotify = GetRecordNotify(GetNode(*recordIter));
                            if (seenRecordNotify != nullptr && IsNotifyIdPeer(*seenRecordNotify, *waitNotify)) {
                                ret = AddEdge(*recordIter, currNodeId);
                                if (ret != HCCL_SUCCESS) {
                                    return ret;
                                }
                                peerIter->second.erase(recordIter);
                                ret = ExecuteNode(currNodeId, graphNodeQue, execFlags, traverseFlags);
                                if (ret != HCCL_SUCCESS) {
                                    return ret;
                                }
                                ++matchedEdgeCount;
                                unmatchedCnt = 0;
                                matched = true;
                                break;
                            }
                        }
                        if (matched) {
                            continue;
                        }
                    }
                }

                graphNodeQue.push_back(currNodeId);
                ++unmatchedCnt;
                continue;
            }

            ret = ExecuteNode(currNodeId, graphNodeQue, execFlags, traverseFlags);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            unmatchedCnt = 0;
        }

        for (const auto& rankEntry : seenInterRankRecords) {
            for (const auto& peerEntry : rankEntry.second) {
                if (!peerEntry.second.empty()) {
                    const TaskNode* node = GetNode(peerEntry.second.front());
                    const AicpuNotify* recordNotify = GetRecordNotify(node);
                    HCCL_VM_ERROR(
                        "{} Found cross-rank Record tasks that were never consumed by any matching "
                        "Wait task, recordRankId={}, waitRankId={}, notifyId={}, firstUnconsumedRecordNode={}, "
                        "unconsumedRecordCount={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED), rankEntry.first, peerEntry.first,
                        recordNotify == nullptr ? std::string("null") : std::to_string(recordNotify->notifyId),
                        node == nullptr ? "null" : node->Describe(), peerEntry.second.size());
                    return HCCL_E_INTERNAL;
                }
            }
        }

        HCCL_VM_DEBUG(
            "Finished matching cross-rank Record/Wait edges, "
            "matchedCrossRankRecordWaitEdgeCount={}",
            matchedEdgeCount);
        return HCCL_SUCCESS;
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
