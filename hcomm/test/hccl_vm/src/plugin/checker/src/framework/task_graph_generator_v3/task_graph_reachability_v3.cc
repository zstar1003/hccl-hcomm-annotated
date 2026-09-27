/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_reachability_v3.h"

#include <cstdint>
#include <map>
#include <queue>
#include <set>
#include <utility>

#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        constexpr size_t BITS_PER_WORD = 64U;

        void SetBit(BitRow& row, size_t idx) { row[idx / BITS_PER_WORD] |= (1ULL << (idx % BITS_PER_WORD)); }

        bool TestBit(const BitRow& row, size_t idx)
        {
            return ((row[idx / BITS_PER_WORD] >> (idx % BITS_PER_WORD)) & 1ULL) != 0;
        }

        bool IsDataMoveTask(const TaskNode* node)
        {
            return node != nullptr
                   && (node->GetType() == TaskType::TRANS_MEM || node->GetType() == TaskType::BATCH_TRANS_MEM
                       || node->GetType() == TaskType::REDUCE || node->GetType() == TaskType::BATCH_REDUCE);
        }

        bool IsMainGraphStartNode(const TaskNode* node)
        {
            if (node == nullptr || node->GetType() != TaskType::START || node->GetNodeId() != MAIN_START_NODE_ID) {
                return false;
            }

            const auto* start = dynamic_cast<const TaskStart*>(node);
            return start != nullptr && start->GetBoundaryType() == BoundaryType::MAIN_GRAPH;
        }

        struct ReachableTaskNodes {
            std::vector<const TaskNode*> nodes;
            std::map<const TaskNode*, size_t> indexByNode;
            std::map<NodeId, size_t> indexByNodeId;
        };

        struct DataMoveNodeIndexes {
            std::map<const TaskNode*, size_t> dataIndexByNode;
            std::map<NodeId, size_t> dataIndexByNodeId;
            std::vector<NodeId> dataNodeIdByIndex;
        };

        struct ReachabilityBuildContext {
            ReachableTaskNodes reachable;
            DataMoveNodeIndexes dataIndexes;
            std::vector<const TaskNode*> taskTopoNodes;
            std::vector<NodeId> fullTopoOrder;
            size_t edgeCount{0};
        };

        void ClearClosure(ReachabilityClosure& closure)
        {
            closure.matrix.clear();
            closure.dataIndexByNodeId.clear();
            closure.dataNodeIdByIndex.clear();
        }

        HcclResult AddReachableTaskNode(const TaskNode* node, ReachableTaskNodes& reachable)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            if (reachable.nodes.size() >= MAX_NODE_COUNT) {
                return HCCL_E_MEMORY;
            }

            const NodeId nodeId = node->GetNodeId();
            if (nodeId < 0) {
                HCCL_VM_ERROR(
                    "{} Task node id is invalid, nodeId={}, node={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), nodeId, node->Describe());
                return HCCL_E_PARA;
            }

            const size_t nodeIndex = reachable.nodes.size();
            const auto idResult = reachable.indexByNodeId.insert(std::make_pair(nodeId, nodeIndex));
            if (!idResult.second) {
                const size_t previousIndex = idResult.first->second;
                const TaskNode* previousNode
                    = (previousIndex < reachable.nodes.size()) ? reachable.nodes[previousIndex] : nullptr;
                HCCL_VM_ERROR(
                    "{} Two different nodes share the same task node id, nodeId={}, "
                    "firstNode={}, secondNode={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), nodeId,
                    (previousNode == nullptr) ? "null" : previousNode->Describe(), node->Describe());
                return HCCL_E_PARA;
            }

            const auto nodeResult = reachable.indexByNode.insert(std::make_pair(node, nodeIndex));
            if (!nodeResult.second) {
                HCCL_VM_ERROR(
                    "{} The same task node object was inserted more than once, node={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                return HCCL_E_INTERNAL;
            }

            reachable.nodes.push_back(node);
            return HCCL_SUCCESS;
        }

        HcclResult BuildDataMoveNodeIndexes(const ReachableTaskNodes& reachable, DataMoveNodeIndexes& dataIndexes)
        {
            dataIndexes = DataMoveNodeIndexes();
            for (const TaskNode* node : reachable.nodes) {
                if (node == nullptr) {
                    return HCCL_E_PTR;
                }
                if (!IsDataMoveTask(node)) {
                    continue;
                }

                const size_t dataIndex = dataIndexes.dataNodeIdByIndex.size();
                const NodeId nodeId = node->GetNodeId();
                dataIndexes.dataIndexByNode[node] = dataIndex;
                dataIndexes.dataIndexByNodeId[nodeId] = dataIndex;
                dataIndexes.dataNodeIdByIndex.push_back(nodeId);
            }
            return HCCL_SUCCESS;
        }

        HcclResult CollectReachableTaskNodes(const TaskNode* start, ReachableTaskNodes& reachable)
        {
            reachable = ReachableTaskNodes();
            if (!IsMainGraphStartNode(start)) {
                HCCL_VM_ERROR(
                    "{} Reachability analysis cannot start because the main start node is "
                    "invalid, mainStartNode={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID),
                    start == nullptr ? "null" : start->Describe());
                return HCCL_E_PARA;
            }

            std::queue<const TaskNode*> nodeQue;
            std::set<const TaskNode*> visited;
            nodeQue.push(start);
            visited.insert(start);
            while (!nodeQue.empty()) {
                const TaskNode* node = nodeQue.front();
                nodeQue.pop();
                if (node != start) {
                    HcclResult ret = AddReachableTaskNode(node, reachable);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }

                for (const TaskNode* childNode : node->GetChildren()) {
                    if (childNode == nullptr) {
                        HCCL_VM_ERROR(
                            "{} Graph structure is broken because one child node is null, "
                            "parent={}",
                            MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                        return HCCL_E_PTR;
                    }
                    if (visited.count(childNode) != 0) {
                        continue;
                    }
                    visited.insert(childNode);
                    nodeQue.push(childNode);
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult GetValidChildIndex(
            const TaskNode* parent, const TaskNode* child, const ReachableTaskNodes& reachable, size_t& childIndex)
        {
            if (parent == nullptr) {
                return HCCL_E_PTR;
            }
            if (child == nullptr) {
                HCCL_VM_ERROR(
                    "{} Graph structure is broken because one child node is null, parent={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), parent->Describe());
                return HCCL_E_PTR;
            }

            const auto iter = reachable.indexByNode.find(child);
            if (iter == reachable.indexByNode.end()) {
                HCCL_VM_ERROR(
                    "{} One graph edge is invalid, so reachability analysis cannot continue, "
                    "parent={}, child={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), parent->Describe(), child->Describe());
                return HCCL_E_PARA;
            }

            childIndex = iter->second;
            return HCCL_SUCCESS;
        }

        HcclResult AddChildrenIndegree(
            const TaskNode* parent, const ReachableTaskNodes& reachable, std::vector<uint32_t>& indegree)
        {
            for (const TaskNode* childNode : parent->GetChildren()) {
                size_t childIndex = 0;
                HcclResult ret = GetValidChildIndex(parent, childNode, reachable, childIndex);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ++indegree[childIndex];
            }
            return HCCL_SUCCESS;
        }

        size_t CountEdges(const TaskNode* start, const std::vector<const TaskNode*>& nodes)
        {
            size_t edgeCount = (start == nullptr) ? 0U : start->GetChildren().size();
            for (const TaskNode* node : nodes) {
                if (node != nullptr) {
                    edgeCount += node->GetChildren().size();
                }
            }
            return edgeCount;
        }

        HcclResult BuildReachabilityContext(const TaskNode* start, ReachabilityBuildContext& context)
        {
            context = ReachabilityBuildContext();

            HcclResult ret = CollectReachableTaskNodes(start, context.reachable);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            const size_t nodeCount = context.reachable.nodes.size();
            std::vector<uint32_t> indegree(nodeCount, 0);
            ret = AddChildrenIndegree(start, context.reachable, indegree);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            for (const TaskNode* node : context.reachable.nodes) {
                ret = AddChildrenIndegree(node, context.reachable, indegree);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }

            std::queue<const TaskNode*> topoQue;
            topoQue.push(start);
            context.fullTopoOrder.reserve(nodeCount + 1U);
            context.taskTopoNodes.reserve(nodeCount);
            while (!topoQue.empty()) {
                const TaskNode* node = topoQue.front();
                topoQue.pop();
                context.fullTopoOrder.push_back(node->GetNodeId());
                if (node != start) {
                    context.taskTopoNodes.push_back(node);
                }

                for (const TaskNode* childNode : node->GetChildren()) {
                    size_t childIndex = 0;
                    ret = GetValidChildIndex(node, childNode, context.reachable, childIndex);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    if (indegree[childIndex] == 0) {
                        return HCCL_E_PARA;
                    }
                    if (--indegree[childIndex] == 0) {
                        topoQue.push(childNode);
                    }
                }
            }

            if (context.taskTopoNodes.size() != nodeCount) {
                const TaskNode* firstUnreachedNode = nullptr;
                for (const TaskNode* node : context.reachable.nodes) {
                    if (node == nullptr) {
                        continue;
                    }
                    const bool found = std::find(context.taskTopoNodes.begin(), context.taskTopoNodes.end(), node)
                                       != context.taskTopoNodes.end();
                    if (!found) {
                        firstUnreachedNode = node;
                        break;
                    }
                }
                HCCL_VM_ERROR(
                    "{} This V3 graph is not a complete DAG from the main start node, "
                    "topoSize={}, expectedTopoSize={}, reachableTaskCount={}, taskNodeCount={}, mainStartNodeId={}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), context.fullTopoOrder.size(), nodeCount + 1U,
                    context.taskTopoNodes.size(), nodeCount, MAIN_START_NODE_ID);
                return HCCL_E_INTERNAL;
            }

            ret = BuildDataMoveNodeIndexes(context.reachable, context.dataIndexes);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            context.edgeCount = CountEdges(start, context.reachable.nodes);
            return HCCL_SUCCESS;
        }

        HcclResult BuildDenseReachabilityClosure(const ReachabilityBuildContext& context, ReachabilityClosure& closure)
        {
            ClearClosure(closure);
            const size_t nodeCount = context.reachable.nodes.size();
            const size_t dataNodeCount = context.dataIndexes.dataNodeIdByIndex.size();
            const size_t wordCount = (dataNodeCount + BITS_PER_WORD - 1U) / BITS_PER_WORD;

            std::vector<BitRow> reachableDataByNode(nodeCount, BitRow(wordCount, 0));
            for (auto iter = context.taskTopoNodes.rbegin(); iter != context.taskTopoNodes.rend(); ++iter) {
                const TaskNode* node = *iter;
                const auto nodeIter = context.reachable.indexByNode.find(node);
                if (nodeIter == context.reachable.indexByNode.end()) {
                    HCCL_VM_ERROR(
                        "{} This node is missing its topological index, node={}",
                        MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                    return HCCL_E_INTERNAL;
                }
                const size_t nodeIndex = nodeIter->second;
                for (const TaskNode* childNode : node->GetChildren()) {
                    size_t childIndex = 0;
                    HcclResult ret = GetValidChildIndex(node, childNode, context.reachable, childIndex);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    const auto childDataIter = context.dataIndexes.dataIndexByNode.find(childNode);
                    if (childDataIter != context.dataIndexes.dataIndexByNode.end()) {
                        SetBit(reachableDataByNode[nodeIndex], childDataIter->second);
                    }
                    for (size_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
                        reachableDataByNode[nodeIndex][wordIndex] |= reachableDataByNode[childIndex][wordIndex];
                    }
                }
            }

            closure.dataIndexByNodeId = context.dataIndexes.dataIndexByNodeId;
            closure.dataNodeIdByIndex = context.dataIndexes.dataNodeIdByIndex;
            closure.matrix.assign(dataNodeCount, BitRow(wordCount, 0));
            for (const auto& dataItem : context.dataIndexes.dataIndexByNode) {
                const TaskNode* node = dataItem.first;
                const size_t dataIndex = dataItem.second;
                const auto nodeIter = context.reachable.indexByNode.find(node);
                if (nodeIter == context.reachable.indexByNode.end() || dataIndex >= closure.matrix.size()) {
                    HCCL_VM_ERROR(
                        "{} This data-move node is missing its reachability index, node={}",
                        MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                    return HCCL_E_INTERNAL;
                }
                closure.matrix[dataIndex] = reachableDataByNode[nodeIter->second];
            }

            return HCCL_SUCCESS;
        }

    } // namespace

    HcclResult
    GenReachabilityClosure(const TaskNode* start, ReachabilityClosure& closure, std::vector<NodeId>* topoOrder)
    {
        ClearClosure(closure);
        if (topoOrder != nullptr) {
            topoOrder->clear();
        }

        ReachabilityBuildContext context;
        HcclResult ret = BuildReachabilityContext(start, context);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        ret = BuildDenseReachabilityClosure(context, closure);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        if (topoOrder != nullptr) {
            *topoOrder = context.fullTopoOrder;
        }
        return HCCL_SUCCESS;
    }

    HcclResult IsReachable(const ReachabilityClosure& closure, NodeId fromNodeId, NodeId toNodeId, bool& isReachable)
    {
        isReachable = false;
        const auto fromIter = closure.dataIndexByNodeId.find(fromNodeId);
        if (fromIter == closure.dataIndexByNodeId.end()) {
            HCCL_VM_ERROR(
                "{} Task dependency query only supports memory-copy or reduce tasks, "
                "but the source task is not in that set, sourceTaskId={}, targetTaskId={}, "
                "sourceReachabilityIndex=null",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), fromNodeId, toNodeId);
            return HCCL_E_PARA;
        }
        const auto toIter = closure.dataIndexByNodeId.find(toNodeId);
        if (toIter == closure.dataIndexByNodeId.end()) {
            HCCL_VM_ERROR(
                "{} Task dependency query only supports memory-copy or reduce tasks, "
                "but the target task is not in that set, sourceTaskId={}, targetTaskId={}, "
                "sourceReachabilityIndex={}, targetReachabilityIndex=null",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), fromNodeId, toNodeId, fromIter->second);
            return HCCL_E_PARA;
        }

        const size_t fromIndex = fromIter->second;
        const size_t toIndex = toIter->second;
        if (fromIndex >= closure.matrix.size() || toIndex / BITS_PER_WORD >= closure.matrix[fromIndex].size()) {
            HCCL_VM_ERROR(
                "{} Reachability index is invalid for this data-move task pair, "
                "sourceTaskId={}, targetTaskId={}, sourceIndex={}, targetIndex={}, matrixRowCount={}, "
                "matrixWordCountInSourceRow={}",
                MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), fromNodeId, toNodeId, fromIndex, toIndex,
                closure.matrix.size(), fromIndex < closure.matrix.size() ? closure.matrix[fromIndex].size() : 0U);
            return HCCL_E_INTERNAL;
        }
        isReachable = TestBit(closure.matrix[fromIndex], toIndex);
        return HCCL_SUCCESS;
    }

    // HcclResult IsReachable(const RoaringReachabilityClosure &closure, NodeId fromNodeId, NodeId toNodeId,
    //     bool &isReachable)
    // {
    //     isReachable = false;
    //     const auto fromIter = closure.dataIndexByNodeId.find(fromNodeId);
    //     if (fromIter == closure.dataIndexByNodeId.end()) {
    //         HCCL_VM_ERROR("Roaring reachability query only supports data move task nodes, "
    //             "fromNodeId={}, toNodeId={}", fromNodeId, toNodeId);
    //         return HCCL_E_PARA;
    //     }
    //     const auto toIter = closure.dataIndexByNodeId.find(toNodeId);
    //     if (toIter == closure.dataIndexByNodeId.end()) {
    //         HCCL_VM_ERROR("Roaring reachability query only supports data move task nodes, "
    //             "fromNodeId={}, toNodeId={}", fromNodeId, toNodeId);
    //         return HCCL_E_PARA;
    //     }

    //     const size_t fromIndex = fromIter->second;
    //     const size_t toIndex = toIter->second;
    //     if (fromIndex >= closure.matrix.size()) {
    //         HCCL_VM_ERROR("Invalid roaring data move reachability index, fromNodeId={}, "
    //             "toNodeId={}, fromIndex={}, toIndex={}, matrixSize={}",
    //             fromNodeId, toNodeId, fromIndex, toIndex, closure.matrix.size());
    //         return HCCL_E_INTERNAL;
    //     }
    //     isReachable = closure.matrix[fromIndex].contains(static_cast<uint32_t>(toIndex));
    //     return HCCL_SUCCESS;
    // }

    HcclResult CheckDataMoveTaskReachability(const TaskNode* start, DataTaskReachabilityStats* stats)
    {
        ReachabilityClosure closure;
        std::vector<NodeId> topoOrder;
        HcclResult ret = GenReachabilityClosure(start, closure, &topoOrder);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        ReachableTaskNodes reachable;
        ret = CollectReachableTaskNodes(start, reachable);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        std::vector<const TaskNode*> dataTaskNodes;
        dataTaskNodes.reserve(reachable.nodes.size());
        for (const TaskNode* node : reachable.nodes) {
            if (IsDataMoveTask(node)) {
                dataTaskNodes.push_back(node);
            }
        }

        DataTaskReachabilityStats localStats;
        localStats.nodeCount = reachable.nodes.size() + 1U;
        localStats.edgeCount = CountEdges(start, reachable.nodes);
        localStats.dataTaskNodeCount = dataTaskNodes.size();

        for (size_t i = 0; i < dataTaskNodes.size(); ++i) {
            const TaskNode* lhsNode = dataTaskNodes[i];
            if (lhsNode == nullptr) {
                return HCCL_E_PTR;
            }

            for (size_t j = i + 1; j < dataTaskNodes.size(); ++j) {
                const TaskNode* rhsNode = dataTaskNodes[j];
                if (rhsNode == nullptr) {
                    return HCCL_E_PTR;
                }

                ++localStats.dataTaskPairCount;
                bool lhsReachRhs = false;
                ret = IsReachable(closure, lhsNode->GetNodeId(), rhsNode->GetNodeId(), lhsReachRhs);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                bool rhsReachLhs = false;
                ret = IsReachable(closure, rhsNode->GetNodeId(), lhsNode->GetNodeId(), rhsReachLhs);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                if (lhsReachRhs || rhsReachLhs) {
                    ++localStats.reachablePairCount;
                } else {
                    ++localStats.unreachablePairCount;
                }
            }
        }

        HCCL_VM_INFO(
            "Task dependency summary generated, nodeCount={}, edgeCount={}, "
            "memoryTaskCount={}, taskPairsChecked={}, orderedTaskPairs={}, parallelTaskPairs={}",
            localStats.nodeCount, localStats.edgeCount, localStats.dataTaskNodeCount, localStats.dataTaskPairCount,
            localStats.reachablePairCount, localStats.unreachablePairCount);

        if (stats != nullptr) {
            *stats = localStats;
        }
        return HCCL_SUCCESS;
    }

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
