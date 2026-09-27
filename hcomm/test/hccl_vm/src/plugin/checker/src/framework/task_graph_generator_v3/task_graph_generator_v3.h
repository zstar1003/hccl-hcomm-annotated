/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_GENERATOR_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_GENERATOR_V3_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "hccl_types.h"
#include "task_def_v3.h"

namespace HcclSim {
class StorageManager;
namespace TaskGraphGeneratorV3 {
    using RankNodeQueues = std::vector<std::vector<NodeId>>;
    using AllRankNodeQueues = std::map<RankId, RankNodeQueues>;

    struct CcuExpandStats {
        size_t graphCount{0};
        size_t internalNodeCount{0};
        size_t recordWaitEdgeCount{0};
        uint64_t totalExpandNs{0};
    };

    struct AivExpandStats {
        size_t graphCount{0};
        size_t internalNodeCount{0};
        size_t setWaitEdgeCount{0};
        size_t pipeBarrierMergeCount{0};
        size_t syncAllMergeCount{0};
        size_t sendRecvEdgeCount{0};
        size_t taskJsonTotalTaskCount{0};
        size_t dagNodeCountBeforeCpGmMerge{0};
        size_t dagNodeCountAfterCpGmMerge{0};
        size_t cpGmLoopMergeCount{0};
        size_t cpGmMergedIterationCount{0};
        size_t cpGmMergedOriginalNodeCount{0};
        size_t cpGmGeneratedNodeCount{0};
        size_t cpGmInactiveNodeCount{0};
        uint64_t totalExpandNs{0};
    };

    class TaskGraphGeneratorV3 {
    public:
        TaskGraphGeneratorV3() = default;
        ~TaskGraphGeneratorV3() = default;

        TaskGraphGeneratorV3(TaskGraphGeneratorV3&&) = default;
        TaskGraphGeneratorV3& operator=(TaskGraphGeneratorV3&&) = default;
        TaskGraphGeneratorV3(const TaskGraphGeneratorV3&) = delete;
        TaskGraphGeneratorV3& operator=(const TaskGraphGeneratorV3&) = delete;

        HcclResult
        GenGraph(std::vector<std::unique_ptr<TaskNode>> translatedNodes, AllRankNodeQueues translatedTaskQueues);
        void Reset();
        void SetStorageManager(StorageManager* storage) { storage_ = storage; }
        StorageManager& GetStorageManager() const;

        NodeId GetMainStartNodeId() const { return mainStartNodeId_; }
        TaskNode* GetMainStartNode() { return mainStart_.get(); }
        const TaskNode* GetMainStartNode() const { return mainStart_.get(); }
        const AllRankNodeQueues& GetTaskQueues() const { return taskQueues_; }

        TaskNode* GetNode(NodeId nodeId);
        const TaskNode* GetNode(NodeId nodeId) const;
        const std::vector<std::unique_ptr<TaskNode>>& GetNodes() const { return nodes_; }

        const std::vector<TaskNode*>& GetParents(NodeId nodeId) const;
        const std::vector<TaskNode*>& GetChildren(NodeId nodeId) const;
        const CcuExpandStats& GetCcuExpandStats() const { return ccuExpandStats_; }
        const AivExpandStats& GetAivExpandStats() const { return aivExpandStats_; }

        HcclResult AddEdge(NodeId parentNodeId, NodeId childNodeId);
        HcclResult AppendGeneratedNode(std::unique_ptr<TaskNode> node, const TaskPosition& position, NodeId& nodeId);
        HcclResult RemoveEdge(NodeId parentNodeId, NodeId childNodeId);
        bool HasPath(NodeId fromNodeId, NodeId toNodeId) const;

    private:
        bool IsValidNodeId(NodeId nodeId) const;
        bool IsMainStartNodeId(NodeId nodeId) const;
        HcclResult CreateMainStartNode();
        HcclResult ValidateTranslatedNodes() const;
        HcclResult BuildDagEdges();

        HcclResult GenGraph4Rank(RankId rankId, const RankNodeQueues& rankTaskQueues);
        HcclResult AddLocalNotifyEdges(RankId rankId, const RankNodeQueues& rankTaskQueues);
        HcclResult AddInterRankNotifyEdges();
        HcclResult ExpandAivSubGraphs();
        HcclResult ExpandCcuSubGraphs();
        HcclResult
        PushNextNode(const RankNodeQueues& rankTaskQueues, NodeId currNodeId, std::vector<NodeId>& nodeQue) const;
        bool IsExecutable(NodeId nodeId, const std::vector<uint8_t>& execFlags) const;
        HcclResult ExecuteNode(
            NodeId nodeId, std::vector<NodeId>& graphNodeQue, std::vector<uint8_t>& execFlags,
            std::vector<uint8_t>& traverseFlags) const;
        size_t CountEdges() const;

        NodeId mainStartNodeId_{INVALID_NODE_ID};
        std::unique_ptr<TaskStart> mainStart_;
        AllRankNodeQueues taskQueues_;
        std::vector<std::unique_ptr<TaskNode>> nodes_;
        bool hasAiv_{false};
        bool hasCcu_{false};
        AivExpandStats aivExpandStats_;
        CcuExpandStats ccuExpandStats_;
        StorageManager* storage_{nullptr};
    };
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_GENERATOR_V3_H
