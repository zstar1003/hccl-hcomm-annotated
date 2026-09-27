/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUP_H
#define GROUP_H

#include <unordered_map>
#include <functional>
#include <memory>
#include <vector>
#include <algorithm>
#include "topo_common_types.h"
#include "log.h"

namespace Hccl {

template <typename NodeType, typename EdgeType>
class Graph {
public:
    bool HasNode(const NodeId nodeId) const { return nodes.find(nodeId) != nodes.end(); }

    bool HasEdge(const NodeId srcNodeId, const NodeId dstNodeId) const
    {
        return edges.find(srcNodeId) != edges.end() && edges.at(srcNodeId).find(dstNodeId) != edges.at(srcNodeId).end();
    }

    std::vector<std::shared_ptr<EdgeType>> GetEdges(const NodeId srcNodeId, const NodeId dstNodeId) const
    {
        if (!HasEdge(srcNodeId, dstNodeId)) {
            return {};
        }
        return edges.at(srcNodeId).at(dstNodeId);
    }

    std::vector<std::shared_ptr<EdgeType>> GetEdges(const NodeId srcNodeId) const
    {
        if (edges.find(srcNodeId) == edges.end()) {
            return {};
        }
        std::vector<std::shared_ptr<EdgeType>> nodeEdges;
        for (const auto& it : edges.at(srcNodeId)) {
            nodeEdges.insert(nodeEdges.end(), it.second.begin(), it.second.end());
        }
        return nodeEdges;
    }

    void TraverseNode(std::function<void(NodeId nodeId, const std::shared_ptr<NodeType>&)> func) const
    {
        for (auto& node : nodes) {
            func(node.first, node.second);
        }
    }

    void TraverseNode(std::function<void(std::shared_ptr<NodeType>)> func) const
    {
        for (auto& node : nodes) {
            func(node.second);
        }
    }

    void TraverseEdge(const NodeId srcNodeId, std::function<void(std::shared_ptr<EdgeType>)> func) const
    {
        if (edges.find(srcNodeId) == edges.end()) {
            return;
        }
        for (auto& srcEdges : edges.at(srcNodeId)) {
            for (auto& edge : srcEdges.second) {
                func(edge);
            }
        }
    }

    void TraverseEdge(
        const NodeId srcNodeId, const NodeId dstNodeId, std::function<void(std::shared_ptr<EdgeType>)> func) const
    {
        if (edges.find(srcNodeId) == edges.end() || edges.at(srcNodeId).find(dstNodeId) == edges.at(srcNodeId).end()) {
            return;
        }
        for (auto& edge : edges.at(srcNodeId).at(dstNodeId)) {
            func(edge);
        }
    }

    void AddNode(const NodeId nodeId, std::shared_ptr<NodeType> node)
    {
        nodes[nodeId] = node;
        HCCL_DEBUG("[Graph]add node [%llu] success! node number is [%zu]", nodeId, nodes.size());
    }

    void AddEdge(const NodeId srcNodeId, const NodeId dstNodeId, std::shared_ptr<EdgeType> edge)
    {
        edges[srcNodeId][dstNodeId].push_back(edge);
        HCCL_DEBUG(
            "[Graph]add edge from node [%llu] to node [%llu] success! edge number is [%zu]", srcNodeId, dstNodeId,
            edges[srcNodeId][dstNodeId].size());
    }

    void DeleteEdge(const NodeId srcNodeId, const NodeId dstNodeId)
    {
        // 在64+1中srcNode 和 dstNode代表fabric和db
        auto srcIt = edges.find(srcNodeId);
        if (srcIt == edges.end()) {
            return;
        }

        auto& dstMap = srcIt->second;
        auto dstIt = dstMap.find(dstNodeId);
        if (dstIt == dstMap.end()) {
            return;
        }

        // 删除对应的边
        dstMap.erase(dstIt);
    }

private:
    std::unordered_map<NodeId, std::unordered_map<NodeId, std::vector<std::shared_ptr<EdgeType>>>> edges;
    std::unordered_map<NodeId, std::shared_ptr<NodeType>> nodes;
};
} // namespace Hccl

#endif // GROUP_H
