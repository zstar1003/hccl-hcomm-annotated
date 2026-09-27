/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dag_graphviz_dump.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "file_utils.h"
#include "sim_log.h"

namespace HcclSim {
namespace {
    using namespace TaskGraphGeneratorV3;

    using V3TaskNode = TaskNode;
    using V3DotEdge = std::pair<size_t, size_t>;
    using V3DotEdgeSet = std::set<V3DotEdge>;
    using V3DotEdgeList = std::vector<V3DotEdge>;
    using V3DotEdgeGroupMap = std::map<std::string, V3DotEdgeList>;

    constexpr const char* V3_DOT_SOLID_EDGE_COLOR = "#2f3f4f";
    constexpr const char* V3_DOT_DASHED_EDGE_COLOR = "#1f77d0";
    constexpr const char* V3_DOT_SOLID_EDGE_PENWIDTH = "1.8";
    constexpr const char* V3_DOT_DASHED_EDGE_PENWIDTH = "1.25";
    constexpr const char* V3_DOT_EDGE_ARROWSIZE = "0.7";
    constexpr double V3_DOT_COL_GAP = 8.0;
    constexpr double V3_DOT_ROW_GAP = 3.2;
    // constexpr double V3_DOT_COL_GAP = 800;
    // constexpr double V3_DOT_ROW_GAP = 320;
    constexpr const char* DAG_GRAPHVIZ_DUMP_ANCHOR = "hccl_vm_install";
    constexpr const char* DAG_GRAPHVIZ_DUMP_SUBDIR = "data";

    struct LaneKey {
        RankId rankId{INVALID_RANK_ID};
        StreamId streamId{INVALID_STREAM_ID};
        QueueId queueId{INVALID_QUEUE_ID};

        bool operator<(const LaneKey& rhs) const
        {
            if (rankId != rhs.rankId) {
                return rankId < rhs.rankId;
            }
            if (streamId != rhs.streamId) {
                return streamId < rhs.streamId;
            }
            return queueId < rhs.queueId;
        }

        bool operator==(const LaneKey& rhs) const
        {
            return rankId == rhs.rankId && streamId == rhs.streamId && queueId == rhs.queueId;
        }
    };

    struct NodeLayoutInfo {
        const V3TaskNode* node{nullptr};
        int laneId{-1};
        uint32_t streamSeq{0};
        uint32_t semanticStep{0};
        uint32_t col{0};
    };

    struct LaneLayoutInfo {
        LaneKey laneKey;
        int laneId{-1};
        uint32_t row{0};
    };

    struct FormalGraphLayout {
        std::map<const V3TaskNode*, NodeLayoutInfo> nodeLayoutByPtr;
        std::map<NodeId, NodeLayoutInfo> nodeLayoutById;
        std::map<int, LaneLayoutInfo> laneLayoutById;
    };

    std::vector<const V3TaskNode*> BfsGraphV3(const V3TaskNode* start)
    {
        std::vector<const V3TaskNode*> result;
        if (start == nullptr) {
            return result;
        }

        std::queue<const V3TaskNode*> nodeQue;
        std::set<const V3TaskNode*> visited;
        nodeQue.push(start);
        visited.insert(start);

        while (!nodeQue.empty()) {
            const V3TaskNode* currNode = nodeQue.front();
            nodeQue.pop();
            result.push_back(currNode);

            for (const V3TaskNode* childNode : currNode->GetChildren()) {
                if (childNode == nullptr || visited.count(childNode) != 0) {
                    continue;
                }
                visited.insert(childNode);
                nodeQue.push(childNode);
            }
        }

        return result;
    }

    bool IsV3Boundary(const V3TaskNode* node, BoundaryType boundaryType)
    {
        if (node == nullptr) {
            return false;
        }
        if (node->GetType() == TaskType::START) {
            auto* start = dynamic_cast<const TaskStart*>(node);
            return start != nullptr && start->GetBoundaryType() == boundaryType;
        }
        if (node->GetType() == TaskType::END) {
            auto* end = dynamic_cast<const TaskEnd*>(node);
            return end != nullptr && end->GetBoundaryType() == boundaryType;
        }
        return false;
    }

    bool IsMainGraphStartNode(const V3TaskNode* node)
    {
        return node != nullptr && node->GetNodeId() == MAIN_START_NODE_ID
               && IsV3Boundary(node, BoundaryType::MAIN_GRAPH);
    }

    bool GetV3BoundaryType(const V3TaskNode* node, BoundaryType& boundaryType)
    {
        if (node == nullptr) {
            return false;
        }
        if (node->GetType() == TaskType::START) {
            auto* start = dynamic_cast<const TaskStart*>(node);
            if (start == nullptr) {
                return false;
            }
            boundaryType = start->GetBoundaryType();
            return true;
        }
        if (node->GetType() == TaskType::END) {
            auto* end = dynamic_cast<const TaskEnd*>(node);
            if (end == nullptr) {
                return false;
            }
            boundaryType = end->GetBoundaryType();
            return true;
        }
        return false;
    }

    bool IsCcuDagLayoutNode(const V3TaskNode* node)
    {
        if (node == nullptr) {
            return false;
        }
        return node->HasCcuTrace() || IsV3Boundary(node, BoundaryType::CCU_SUB_GRAPH);
    }

    std::string EscapeGraphvizLabel(const std::string& label)
    {
        std::string escaped;
        escaped.reserve(label.size());
        for (char ch : label) {
            switch (ch) {
                case '\\':
                    escaped += "\\\\";
                    break;
                case '"':
                    escaped += "\\\"";
                    break;
                case '\n':
                    escaped += "\\n";
                    break;
                case '\r':
                    break;
                default:
                    escaped += ch;
                    break;
            }
        }
        return escaped;
    }

    std::string FormatGraphvizIdValue(uint64_t value, uint64_t invalidValue)
    {
        if (value == invalidValue) {
            return "-";
        }
        return std::to_string(value);
    }

    std::string FormatGraphvizHexValue(uint64_t value, uint64_t invalidValue)
    {
        if (value == invalidValue) {
            return "-";
        }

        std::ostringstream os;
        os << "0x" << std::hex << value;
        return os.str();
    }

    const char* HcclDataTypeKeyName(uint8_t dataType)
    {
        switch (dataType) {
            case 0U:
                return "int8";
            case 1U:
                return "int16";
            case 2U:
                return "int32";
            case 3U:
                return "float16";
            case 4U:
                return "float32";
            case 5U:
                return "int64";
            case 6U:
                return "uint64";
            case 7U:
                return "uint8";
            case 8U:
                return "uint16";
            case 9U:
                return "uint32";
            case 10U:
                return "float64";
            case 11U:
                return "bfloat16";
            case 12U:
                return "int128";
            case 14U:
                return "hif8";
            case 15U:
                return "fp8e4m3";
            case 16U:
                return "fp8e5m2";
            case 17U:
                return "fp8e8m0";
            case 18U:
                return "reserved";
            default:
                return "unknown";
        }
    }

    const char* HcclReduceOpKeyName(uint8_t reduceOp)
    {
        switch (reduceOp) {
            case 0U:
                return "sum";
            case 1U:
                return "prod";
            case 2U:
                return "max";
            case 3U:
                return "min";
            case 4U:
                return "reserved";
            default:
                return "unknown";
        }
    }

    const char* TaskTypeKeyName(TaskType type)
    {
        switch (type) {
            case TaskType::TRANS_MEM:
                return "TRANS_MEM";
            case TaskType::BATCH_TRANS_MEM:
                return "BATCH_TRANS_MEM";
            case TaskType::REDUCE:
                return "REDUCE";
            case TaskType::BATCH_REDUCE:
                return "BATCH_REDUCE";
            case TaskType::RECORD:
                return "RECORD";
            case TaskType::WAIT:
                return "WAIT";
            case TaskType::CCU_GRAPH:
                return "CCU_GRAPH";
            case TaskType::AIV_GRAPH:
                return "AIV_GRAPH";
            case TaskType::AIV_SET_FLAG:
                return "AIV_SET_FLAG";
            case TaskType::AIV_WAIT_FLAG:
                return "AIV_WAIT_FLAG";
            case TaskType::AIV_PIPE_BARRIER:
                return "AIV_PIPE_BARRIER";
            case TaskType::AIV_SYNC_ALL:
                return "AIV_SYNC_ALL";
            case TaskType::AIV_SEND_FLAG:
                return "AIV_SEND_FLAG";
            case TaskType::AIV_RECV_FLAG:
                return "AIV_RECV_FLAG";
            case TaskType::START:
                return "START";
            case TaskType::END:
                return "END";
            default:
                return "INVALID";
        }
    }

    const char* BoundaryTypeKeyName(BoundaryType type)
    {
        switch (type) {
            case BoundaryType::MAIN_GRAPH:
                return "MAIN_GRAPH";
            case BoundaryType::CCU_SUB_GRAPH:
                return "CCU_SUB_GRAPH";
            case BoundaryType::AIV_SUB_GRAPH:
                return "AIV_SUB_GRAPH";
            case BoundaryType::LOOP:
                return "LOOP";
            default:
                return "INVALID";
        }
    }

    std::string MakeCompactMemSliceLabel(const MemSlice& slice)
    {
        std::ostringstream os;
        os << "{rank=" << FormatGraphvizIdValue(slice.rankId, INVALID_RANK_ID)
           << ", type=" << DescribeMemType(slice.memType) << ",offset=0x" << std::hex << slice.offset << ",len=0x"
           << slice.len << std::dec << "}";
        return os.str();
    }

    std::string MakeAicpuNotifyLabel(const AicpuNotify& notify)
    {
        std::ostringstream os;
        os << "notify{srcRank=" << FormatGraphvizIdValue(notify.recordRankId, INVALID_RANK_ID)
           << ", dstRank=" << FormatGraphvizIdValue(notify.waitRankId, INVALID_RANK_ID)
           << ", notifyId=" << FormatGraphvizIdValue(notify.notifyId, INVALID_NOTIFY_ID) << "}";
        return os.str();
    }

    std::string MakeCcuNotifyLabel(const CcuNotify& notify)
    {
        std::ostringstream os;
        os << "notify{srcRank=" << FormatGraphvizIdValue(notify.recordRankId, INVALID_RANK_ID)
           << ", dstRank=" << FormatGraphvizIdValue(notify.waitRankId, INVALID_RANK_ID)
           << ", cke=" << FormatGraphvizIdValue(notify.ckeId, INVALID_CCU_CKE)
           << ", mask=" << FormatGraphvizHexValue(notify.ckeMask, INVALID_CCU_CKE_MASK) << "}";
        return os.str();
    }

    QueueId GetV3GraphvizQueueId(const V3TaskNode* node)
    {
        if (node == nullptr) {
            return INVALID_QUEUE_ID;
        }
        if (node->HasCcuTrace() && node->GetCcuTrace().queueId != INVALID_QUEUE_ID) {
            return node->GetCcuTrace().queueId;
        }
        return node->GetPosition().queueId;
    }

    void AppendV3GraphvizPositionLine(std::ostringstream& os, const V3TaskNode* node)
    {
        if (node == nullptr) {
            return;
        }

        const auto& loc = node->GetPosition();
        const QueueId queueId = GetV3GraphvizQueueId(node);
        os << "\nrank=" << FormatGraphvizIdValue(loc.rankId, INVALID_RANK_ID)
           << ", stream=" << FormatGraphvizIdValue(loc.streamId, INVALID_STREAM_ID);
        if (queueId != INVALID_QUEUE_ID) {
            os << ", queue=" << queueId;
        }
    }

    bool AppendV3GraphvizBoundaryLabel(std::ostringstream& os, const V3TaskNode* node)
    {
        BoundaryType boundaryType = BoundaryType::INVALID;
        if (!GetV3BoundaryType(node, boundaryType)) {
            return false;
        }

        os << "\nboundary=" << BoundaryTypeKeyName(boundaryType);
        return true;
    }

    void AppendV3GraphvizTraceSummary(std::ostringstream& os, const V3TaskNode* node)
    {
        if (node == nullptr || !node->HasCcuTrace()) {
            return;
        }

        const auto& trace = node->GetCcuTrace();
        os << "\ninstr=" << trace.instrId << ", die=" << FormatGraphvizIdValue(trace.dieId, INVALID_DIE_ID)
           << ", mission=" << FormatGraphvizIdValue(trace.missionId, INVALID_MISSION_ID);
    }

    bool AppendV3GraphvizTaskLabel(std::ostringstream& os, const V3TaskNode* node)
    {
        if (node == nullptr) {
            return false;
        }

        switch (node->GetType()) {
            case TaskType::TRANS_MEM: {
                auto* transMem = dynamic_cast<const TaskTransMem*>(node);
                if (transMem == nullptr) {
                    return false;
                }
                os << "\nsrc" << MakeCompactMemSliceLabel(transMem->GetSrc()) << "\ndst"
                   << MakeCompactMemSliceLabel(transMem->GetDst());
                break;
            }
            case TaskType::BATCH_TRANS_MEM: {
                auto* batchTransMem = dynamic_cast<const TaskBatchTransMem*>(node);
                if (batchTransMem == nullptr) {
                    return false;
                }
                const auto& srcs = batchTransMem->GetSrcs();
                const auto& dsts = batchTransMem->GetDsts();
                if (!srcs.empty()) {
                    os << "\nsrc0" << MakeCompactMemSliceLabel(srcs.front());
                }
                if (!dsts.empty()) {
                    os << "\ndst0" << MakeCompactMemSliceLabel(dsts.front());
                }
                break;
            }
            case TaskType::REDUCE: {
                auto* reduce = dynamic_cast<const TaskReduce*>(node);
                if (reduce == nullptr) {
                    return false;
                }
                os << "\nsrc" << MakeCompactMemSliceLabel(reduce->GetSrc()) << "\ndst"
                   << MakeCompactMemSliceLabel(reduce->GetDst())
                   << "\ndtype=" << HcclDataTypeKeyName(reduce->GetDataType())
                   << ", op=" << HcclReduceOpKeyName(reduce->GetReduceOp());
                break;
            }
            case TaskType::BATCH_REDUCE: {
                auto* batchReduce = dynamic_cast<const TaskBatchReduce*>(node);
                if (batchReduce == nullptr) {
                    return false;
                }
                const auto& srcGroups = batchReduce->GetSrcs();
                const auto& dsts = batchReduce->GetDsts();
                if (!srcGroups.empty() && !srcGroups.front().empty()) {
                    os << "\nsrc0" << MakeCompactMemSliceLabel(srcGroups.front().front());
                }
                if (!dsts.empty()) {
                    os << "\ndst0" << MakeCompactMemSliceLabel(dsts.front());
                }
                os << "\ndtype=" << HcclDataTypeKeyName(batchReduce->GetDataType())
                   << ", op=" << HcclReduceOpKeyName(batchReduce->GetReduceOp());
                break;
            }
            case TaskType::RECORD: {
                if (auto* record = dynamic_cast<const TaskRecordAICPU*>(node)) {
                    os << "\n" << MakeAicpuNotifyLabel(record->GetNotify());
                    break;
                }
                if (auto* record = dynamic_cast<const TaskRecordCCU*>(node)) {
                    os << "\n" << MakeCcuNotifyLabel(record->GetNotify());
                    break;
                }
                return false;
            }
            case TaskType::WAIT: {
                if (auto* wait = dynamic_cast<const TaskWaitAICPU*>(node)) {
                    os << "\n" << MakeAicpuNotifyLabel(wait->GetNotify());
                    break;
                }
                if (auto* wait = dynamic_cast<const TaskWaitCCU*>(node)) {
                    os << "\n" << MakeCcuNotifyLabel(wait->GetNotify());
                    break;
                }
                return false;
            }
            case TaskType::CCU_GRAPH:
            case TaskType::AIV_GRAPH:
            case TaskType::AIV_SET_FLAG:
            case TaskType::AIV_WAIT_FLAG:
            case TaskType::AIV_PIPE_BARRIER:
            case TaskType::AIV_SYNC_ALL:
            case TaskType::AIV_SEND_FLAG:
            case TaskType::AIV_RECV_FLAG:
                break;
            default:
                return false;
        }
        return true;
    }

    std::string MakeV3GraphvizNodeLabel(const V3TaskNode* node)
    {
        std::ostringstream os;
        const NodeId nodeId = (node == nullptr) ? INVALID_NODE_ID : node->GetNodeId();
        os << "nodeId=" << nodeId;

        if (node != nullptr) {
            AppendV3GraphvizPositionLine(os, node);
            os << "\ntype=" << TaskTypeKeyName(node->GetType());
        }

        const bool isBoundaryNode = AppendV3GraphvizBoundaryLabel(os, node);
        if (!isBoundaryNode && !AppendV3GraphvizTaskLabel(os, node) && node == nullptr) {
            os << "\n<invalid v3 node>";
        }
        if (!isBoundaryNode) {
            AppendV3GraphvizTraceSummary(os, node);
        }
        return os.str();
    }

    QueueId NormalizeQueueId(QueueId queueId) { return queueId == INVALID_QUEUE_ID ? 0U : queueId; }

    LaneKey MakeLaneKey(const V3TaskNode* node)
    {
        LaneKey laneKey;
        if (node == nullptr) {
            return laneKey;
        }
        const TaskLocation& taskLoc = node->GetLocation();
        laneKey.rankId = taskLoc.rankId;
        laneKey.streamId = taskLoc.streamId;
        laneKey.queueId = NormalizeQueueId(taskLoc.queueId);
        return laneKey;
    }

    bool PreferRecordCandidate(const std::pair<LaneKey, NodeId>& lhs, const std::pair<LaneKey, NodeId>& rhs)
    {
        if (lhs.first.rankId != rhs.first.rankId) {
            return lhs.first.rankId < rhs.first.rankId;
        }
        if (lhs.first.streamId != rhs.first.streamId) {
            return lhs.first.streamId < rhs.first.streamId;
        }
        if (lhs.first.queueId != rhs.first.queueId) {
            return lhs.first.queueId < rhs.first.queueId;
        }
        return lhs.second < rhs.second;
    }

    bool IsNotifyEdge(const V3TaskNode* srcNode, const V3TaskNode* dstNode)
    {
        return srcNode != nullptr && dstNode != nullptr && srcNode->GetType() == TaskType::RECORD
               && dstNode->GetType() == TaskType::WAIT;
    }

    FormalGraphLayout BuildFormalGraphLayout(const std::vector<const V3TaskNode*>& nodes)
    {
        FormalGraphLayout layout;
        std::map<NodeId, const V3TaskNode*> nodeById;
        std::vector<const V3TaskNode*> layoutNodes;
        std::set<NodeId> seenNodeIds;
        layoutNodes.reserve(nodes.size());
        for (const V3TaskNode* node : nodes) {
            if (node == nullptr) {
                continue;
            }
            const NodeId nodeId = node->GetNodeId();
            if (seenNodeIds.count(nodeId) != 0) {
                continue;
            }
            seenNodeIds.insert(nodeId);
            nodeById[nodeId] = node;
            if (!IsMainGraphStartNode(node)) {
                layoutNodes.push_back(node);
            }
        }

        std::map<NodeId, uint32_t> indegreeById;
        std::map<NodeId, std::vector<NodeId>> childIdsById;
        for (const V3TaskNode* node : layoutNodes) {
            const NodeId nodeId = node->GetNodeId();
            indegreeById[nodeId] = 0U;
        }
        for (const V3TaskNode* node : layoutNodes) {
            const NodeId nodeId = node->GetNodeId();
            for (const V3TaskNode* child : node->GetChildren()) {
                if (child == nullptr || IsMainGraphStartNode(child)) {
                    continue;
                }
                const NodeId childId = child->GetNodeId();
                if (indegreeById.count(childId) == 0) {
                    continue;
                }
                childIdsById[nodeId].push_back(childId);
                ++indegreeById[childId];
            }
        }

        std::queue<NodeId> readyNodeIds;
        for (const V3TaskNode* node : layoutNodes) {
            const NodeId nodeId = node->GetNodeId();
            if (indegreeById[nodeId] == 0U) {
                readyNodeIds.push(nodeId);
            }
        }

        std::vector<const V3TaskNode*> topoNodes;
        topoNodes.reserve(layoutNodes.size());
        while (!readyNodeIds.empty()) {
            const NodeId nodeId = readyNodeIds.front();
            readyNodeIds.pop();
            const auto nodeIter = nodeById.find(nodeId);
            if (nodeIter == nodeById.end() || nodeIter->second == nullptr) {
                HCCL_VM_WARN(
                    "[TaskGraphGeneratorV3][GraphvizDot][v3] Topo node is missing in BFS map, nodeId={}", nodeId);
                continue;
            }
            topoNodes.push_back(nodeIter->second);

            auto childIter = childIdsById.find(nodeId);
            if (childIter == childIdsById.end()) {
                continue;
            }
            for (NodeId childId : childIter->second) {
                auto indegreeIter = indegreeById.find(childId);
                if (indegreeIter == indegreeById.end() || indegreeIter->second == 0U) {
                    continue;
                }
                --indegreeIter->second;
                if (indegreeIter->second == 0U) {
                    readyNodeIds.push(childId);
                }
            }
        }
        if (topoNodes.size() != layoutNodes.size()) {
            HCCL_VM_WARN(
                "[TaskGraphGeneratorV3][GraphvizDot][v3] Failed to build full topo order, use BFS order, "
                "topoCount={}, nodeCount={}",
                topoNodes.size(), layoutNodes.size());
            topoNodes = layoutNodes;
        }

        std::map<LaneKey, int> laneIdByKey;
        for (const V3TaskNode* node : topoNodes) {
            const LaneKey laneKey = MakeLaneKey(node);
            if (laneIdByKey.count(laneKey) != 0) {
                continue;
            }
            const int laneId = static_cast<int>(laneIdByKey.size());
            laneIdByKey[laneKey] = laneId;
        }

        std::map<int, uint32_t> laneStreamSeq;
        for (size_t index = 0; index < topoNodes.size(); ++index) {
            const V3TaskNode* node = topoNodes[index];
            const int laneId = laneIdByKey[MakeLaneKey(node)];
            NodeLayoutInfo nodeInfo;
            nodeInfo.node = node;
            nodeInfo.laneId = laneId;
            nodeInfo.streamSeq = laneStreamSeq[laneId]++;
            nodeInfo.semanticStep = static_cast<uint32_t>(index + 1U);
            layout.nodeLayoutByPtr[node] = nodeInfo;
            layout.nodeLayoutById[node->GetNodeId()] = nodeInfo;
        }

        std::map<NodeId, std::pair<LaneKey, NodeId>> primaryRecordByWait;
        for (const V3TaskNode* node : topoNodes) {
            for (const V3TaskNode* child : node->GetChildren()) {
                if (child == nullptr || child->GetNodeId() == MAIN_START_NODE_ID) {
                    continue;
                }
                if (layout.nodeLayoutById.count(node->GetNodeId()) == 0
                    || layout.nodeLayoutById.count(child->GetNodeId()) == 0 || !IsNotifyEdge(node, child)) {
                    continue;
                }

                const std::pair<LaneKey, NodeId> candidate(MakeLaneKey(node), node->GetNodeId());
                const auto primaryIter = primaryRecordByWait.find(child->GetNodeId());
                if (primaryIter == primaryRecordByWait.end() || PreferRecordCandidate(candidate, primaryIter->second)) {
                    primaryRecordByWait[child->GetNodeId()] = candidate;
                }
            }
        }

        std::map<NodeId, std::vector<std::pair<NodeId, uint32_t>>> incomingLayoutEdges;
        std::map<int, NodeId> lastNodeByLaneId;
        for (const V3TaskNode* node : topoNodes) {
            const NodeId nodeId = node->GetNodeId();
            const int laneId = layout.nodeLayoutById[nodeId].laneId;
            const auto lastLaneIter = lastNodeByLaneId.find(laneId);
            if (lastLaneIter != lastNodeByLaneId.end()) {
                incomingLayoutEdges[nodeId].push_back(std::make_pair(lastLaneIter->second, 1U));
            }
            lastNodeByLaneId[laneId] = nodeId;

            for (const V3TaskNode* parent : node->GetParents()) {
                if (parent == nullptr || parent->GetNodeId() == MAIN_START_NODE_ID) {
                    continue;
                }
                if (layout.nodeLayoutById.count(parent->GetNodeId()) == 0) {
                    continue;
                }

                uint32_t colStep = 1U;
                if (IsNotifyEdge(parent, node)) {
                    const auto primaryIter = primaryRecordByWait.find(nodeId);
                    if (primaryIter != primaryRecordByWait.end() && primaryIter->second.second == parent->GetNodeId()) {
                        colStep = 0U;
                    }
                }
                incomingLayoutEdges[nodeId].push_back(std::make_pair(parent->GetNodeId(), colStep));
            }
        }

        for (const V3TaskNode* node : topoNodes) {
            NodeLayoutInfo& nodeInfo = layout.nodeLayoutById[node->GetNodeId()];
            uint32_t col = 0U;
            const auto incomingIter = incomingLayoutEdges.find(node->GetNodeId());
            if (incomingIter != incomingLayoutEdges.end()) {
                for (const auto& entry : incomingIter->second) {
                    const auto parentInfoIter = layout.nodeLayoutById.find(entry.first);
                    if (parentInfoIter == layout.nodeLayoutById.end()) {
                        continue;
                    }
                    col = std::max(col, parentInfoIter->second.col + entry.second);
                }
            }
            nodeInfo.col = col;
            layout.nodeLayoutByPtr[node] = nodeInfo;
        }

        uint32_t row = 0U;
        for (const auto& lanePair : laneIdByKey) {
            LaneLayoutInfo laneInfo;
            laneInfo.laneKey = lanePair.first;
            laneInfo.laneId = lanePair.second;
            laneInfo.row = row++;
            layout.laneLayoutById[lanePair.second] = laneInfo;
        }

        return layout;
    }

    std::string FormatV3DotGroupValue(uint64_t value, uint64_t invalidValue)
    {
        if (value == invalidValue) {
            return "x";
        }
        return std::to_string(value);
    }

    std::string MakeGenericLaneGroup(const LaneKey& laneKey)
    {
        std::ostringstream os;
        os << "lane_r" << FormatV3DotGroupValue(laneKey.rankId, INVALID_RANK_ID) << "_s"
           << FormatV3DotGroupValue(laneKey.streamId, INVALID_STREAM_ID) << "_q"
           << FormatV3DotGroupValue(laneKey.queueId, INVALID_QUEUE_ID);
        return os.str();
    }

    std::string MakeV3DotNodeGroup(const V3TaskNode* node, const FormalGraphLayout& layout)
    {
        if (node == nullptr) {
            return "";
        }

        const auto layoutIter = layout.nodeLayoutByPtr.find(node);
        if (layoutIter != layout.nodeLayoutByPtr.end()) {
            const auto laneIter = layout.laneLayoutById.find(layoutIter->second.laneId);
            if (laneIter != layout.laneLayoutById.end()) {
                return MakeGenericLaneGroup(laneIter->second.laneKey);
            }
        }

        const auto& loc = node->GetPosition();
        std::ostringstream os;
        os << "node_r" << FormatV3DotGroupValue(loc.rankId, INVALID_RANK_ID) << "_s"
           << FormatV3DotGroupValue(loc.streamId, INVALID_STREAM_ID) << "_q"
           << FormatV3DotGroupValue(NormalizeQueueId(GetV3GraphvizQueueId(node)), INVALID_QUEUE_ID);
        return os.str();
    }

    double GetDefaultLaneY(const FormalGraphLayout& layout)
    {
        if (layout.laneLayoutById.empty()) {
            return 0.0;
        }
        return -static_cast<double>(layout.laneLayoutById.size() - 1U) * V3_DOT_ROW_GAP / 2.0;
    }

    std::pair<double, double>
    GetV3DotNodePosition(const V3TaskNode* node, const FormalGraphLayout& layout, size_t fallbackIndex)
    {
        const auto layoutIter = layout.nodeLayoutByPtr.find(node);
        if (layoutIter == layout.nodeLayoutByPtr.end()) {
            if (IsMainGraphStartNode(node)) {
                return {-V3_DOT_COL_GAP, GetDefaultLaneY(layout)};
            }
            return {static_cast<double>(fallbackIndex) * V3_DOT_COL_GAP, GetDefaultLaneY(layout)};
        }

        const NodeLayoutInfo& nodeInfo = layoutIter->second;
        const auto laneIter = layout.laneLayoutById.find(nodeInfo.laneId);
        const uint32_t row = (laneIter == layout.laneLayoutById.end()) ? 0U : laneIter->second.row;
        return {static_cast<double>(nodeInfo.col) * V3_DOT_COL_GAP, -static_cast<double>(row) * V3_DOT_ROW_GAP};
    }

    bool
    IsV3DotSameLane(const std::vector<const V3TaskNode*>& nodes, const V3DotEdge& edge, const FormalGraphLayout& layout)
    {
        if (edge.first >= nodes.size() || edge.second >= nodes.size()) {
            return false;
        }
        const auto fromIter = layout.nodeLayoutByPtr.find(nodes[edge.first]);
        const auto toIter = layout.nodeLayoutByPtr.find(nodes[edge.second]);
        return fromIter != layout.nodeLayoutByPtr.end() && toIter != layout.nodeLayoutByPtr.end()
               && fromIter->second.laneId == toIter->second.laneId;
    }

    bool IsV3DotSolidEdge(
        const std::vector<const V3TaskNode*>& nodes, const V3DotEdge& edge, const FormalGraphLayout& layout)
    {
        if (edge.first >= nodes.size() || edge.second >= nodes.size()) {
            return false;
        }
        return IsV3DotSameLane(nodes, edge, layout);
    }

    std::string MakeV3DotSolidEdgeGroupKey(
        const std::vector<const V3TaskNode*>& nodes, const V3DotEdge& edge, const FormalGraphLayout& layout)
    {
        const auto laneIter = (edge.first < nodes.size()) ? layout.nodeLayoutByPtr.find(nodes[edge.first]) :
                                                            layout.nodeLayoutByPtr.end();
        const auto laneLayoutIter = (laneIter == layout.nodeLayoutByPtr.end()) ?
                                        layout.laneLayoutById.end() :
                                        layout.laneLayoutById.find(laneIter->second.laneId);
        if (laneLayoutIter != layout.laneLayoutById.end()) {
            return "queue_" + MakeGenericLaneGroup(laneLayoutIter->second.laneKey);
        }
        return "queue_unknown";
    }

    void EmitV3DotSolidEdgeAttrs(std::ostringstream& os)
    {
        os << " [constraint=true, color=\"" << V3_DOT_SOLID_EDGE_COLOR << "\", penwidth=" << V3_DOT_SOLID_EDGE_PENWIDTH
           << ", arrowhead=normal, arrowsize=" << V3_DOT_EDGE_ARROWSIZE << "]";
    }

    void EmitV3DotDashedEdgeAttrs(std::ostringstream& os)
    {
        os << " [constraint=false, color=\"" << V3_DOT_DASHED_EDGE_COLOR
           << "\", penwidth=" << V3_DOT_DASHED_EDGE_PENWIDTH
           << ", arrowhead=normal, arrowsize=" << V3_DOT_EDGE_ARROWSIZE << ", style=dashed]";
    }

    void EmitV3DotSolidChain(std::ostringstream& os, const std::vector<size_t>& chain, const std::string& nodePrefix)
    {
        if (chain.size() < 2U) {
            return;
        }

        os << "  " << nodePrefix << chain.front();
        for (size_t index = 1U; index < chain.size(); ++index) {
            os << " -> " << nodePrefix << chain[index];
        }
        EmitV3DotSolidEdgeAttrs(os);
        os << ";\n";
    }

    void EmitV3DotSolidEdgeGroup(std::ostringstream& os, const V3DotEdgeList& edges, const std::string& nodePrefix)
    {
        std::map<size_t, std::vector<size_t>> childrenByNode;
        std::map<size_t, size_t> indegreeByNode;
        std::map<size_t, size_t> outdegreeByNode;
        V3DotEdgeSet pendingEdges;
        for (const V3DotEdge& edge : edges) {
            childrenByNode[edge.first].push_back(edge.second);
            ++outdegreeByNode[edge.first];
            ++indegreeByNode[edge.second];
            pendingEdges.insert(edge);
        }
        for (auto& item : childrenByNode) {
            std::sort(item.second.begin(), item.second.end());
        }

        auto emitFromEdge = [&](const V3DotEdge& startEdge) {
            std::vector<size_t> chain;
            chain.push_back(startEdge.first);
            chain.push_back(startEdge.second);
            pendingEdges.erase(startEdge);

            size_t curr = startEdge.second;
            while (outdegreeByNode[curr] == 1U && indegreeByNode[curr] == 1U) {
                const size_t next = childrenByNode[curr].front();
                const V3DotEdge nextEdge(curr, next);
                if (pendingEdges.count(nextEdge) == 0) {
                    break;
                }
                chain.push_back(next);
                pendingEdges.erase(nextEdge);
                curr = next;
            }
            EmitV3DotSolidChain(os, chain, nodePrefix);
        };

        for (const V3DotEdge& edge : edges) {
            if (pendingEdges.count(edge) == 0) {
                continue;
            }
            if (indegreeByNode[edge.first] != 1U) {
                emitFromEdge(edge);
            }
        }

        while (!pendingEdges.empty()) {
            emitFromEdge(*pendingEdges.begin());
        }
    }

    void EmitV3DotDependencyEdges(
        std::ostringstream& os, const V3DotEdgeSet& edgeIndexes, const std::vector<const V3TaskNode*>& nodes,
        const FormalGraphLayout& layout, const std::string& nodePrefix)
    {
        V3DotEdgeGroupMap solidEdgesByGroup;
        V3DotEdgeList dashedEdges;
        for (const auto& edge : edgeIndexes) {
            if (IsV3DotSolidEdge(nodes, edge, layout)) {
                solidEdgesByGroup[MakeV3DotSolidEdgeGroupKey(nodes, edge, layout)].push_back(edge);
            } else {
                dashedEdges.push_back(edge);
            }
        }

        os << "\n  // Same queue DAG edges are emitted as solid chains.\n";
        for (const auto& edgeGroup : solidEdgesByGroup) {
            os << "  // " << edgeGroup.first << "\n";
            EmitV3DotSolidEdgeGroup(os, edgeGroup.second, nodePrefix);
        }

        os << "\n  // Other real DAG edges are shown as dashed semantic dependencies.\n";
        for (const auto& edge : dashedEdges) {
            os << "  " << nodePrefix << edge.first << " -> " << nodePrefix << edge.second;
            EmitV3DotDashedEdgeAttrs(os);
            os << ";\n";
        }
    }

    std::string BuildV3GraphvizDot(const std::vector<const V3TaskNode*>& nodes)
    {
        std::map<const V3TaskNode*, size_t> nodeIndexByPtr;
        for (size_t i = 0; i < nodes.size(); ++i) {
            nodeIndexByPtr[nodes[i]] = i;
        }
        FormalGraphLayout layout = BuildFormalGraphLayout(nodes);

        V3DotEdgeSet edgeIndexes;
        for (const V3TaskNode* node : nodes) {
            const auto parentIter = nodeIndexByPtr.find(node);
            if (parentIter == nodeIndexByPtr.end() || node == nullptr) {
                continue;
            }

            for (const V3TaskNode* childNode : node->GetChildren()) {
                const auto childIter = nodeIndexByPtr.find(childNode);
                if (childIter == nodeIndexByPtr.end()) {
                    continue;
                }
                edgeIndexes.insert(std::make_pair(parentIter->second, childIter->second));
            }
        }

        std::ostringstream os;
        os << "digraph v3_task_graph {\n";
        os << "  // Coordinates come from the dump_v3 DAG lane/column layout. Render with neato -n2 to honor pos.\n";
        os << "  graph [layout=neato, label=\"v3 graph fixed-lane DAG\", labelloc=t, fontsize=20, "
           << "splines=ortho, overlap=false, outputorder=nodesfirst];\n";
        os << "  node [shape=box, style=\"rounded,filled\", fillcolor=\"#eef7ff\", fontname=\"Courier\", pin=true];\n";
        os << "  edge [fontname=\"Courier\", arrowhead=normal, arrowsize=" << V3_DOT_EDGE_ARROWSIZE << "];\n\n";
        for (size_t i = 0; i < nodes.size(); ++i) {
            const std::string group = MakeV3DotNodeGroup(nodes[i], layout);
            os << "  v3_" << i << " [label=\"" << EscapeGraphvizLabel(MakeV3GraphvizNodeLabel(nodes[i])) << "\"";
            const auto pos = GetV3DotNodePosition(nodes[i], layout, i);
            os << std::fixed << std::setprecision(3) << ", pos=\"" << pos.first << "," << pos.second << "!\"";
            if (!group.empty()) {
                os << ", group=\"" << group << "\"";
            }
            os << "];\n";
        }

        EmitV3DotDependencyEdges(os, edgeIndexes, nodes, layout, "v3_");
        os << "}\n";
        return os.str();
    }

    std::string MakeDagGraphvizTimestamp()
    {
        const auto now = std::chrono::system_clock::now();
        const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
        std::tm localTime{};
        localtime_r(&nowTime, &localTime);

        char timeBuffer[32]{};
        (void)std::strftime(timeBuffer, sizeof(timeBuffer), "%Y%m%d%H%M%S", &localTime);
        return timeBuffer;
    }

    std::string ResolveDagGraphvizDumpDir()
    {
        const std::string rootPath = GetRootPath(DAG_GRAPHVIZ_DUMP_ANCHOR, 8);
        if (!rootPath.empty()) {
            return JoinPath(JoinPath(rootPath, DAG_GRAPHVIZ_DUMP_ANCHOR), DAG_GRAPHVIZ_DUMP_SUBDIR);
        }
        return "";
    }

    std::string MakeDagGraphvizDumpPath()
    {
        return JoinPath(ResolveDagGraphvizDumpDir(), "TaskGraph_" + MakeDagGraphvizTimestamp() + ".dot");
    }

} // namespace

HcclResult DumpDagGraphvizDot(const TaskGraphGeneratorV3::TaskNode* start, std::string* dumpPath)
{
    if (start == nullptr) {
        HCCL_VM_WARN("[TaskGraphGeneratorV3][GraphvizDot][v3] Skip DAG dot dump, start is null.");
        return HCCL_E_PTR;
    }

    const std::vector<const V3TaskNode*> v3BfsNodes = BfsGraphV3(start);
    const std::string outputDir = ResolveDagGraphvizDumpDir();
    HcclResult ret = EnsureDirectory(outputDir);
    if (ret != HCCL_SUCCESS) {
        HCCL_VM_WARN(
            "[TaskGraphGeneratorV3][GraphvizDot][v3] Failed to create dump dir: {}, ret={}", outputDir,
            static_cast<uint32_t>(ret));
        return ret;
    }

    const std::string path = MakeDagGraphvizDumpPath();
    std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        HCCL_VM_WARN("[TaskGraphGeneratorV3][GraphvizDot][v3] Failed to open graph file: {}", path);
        return HCCL_E_INTERNAL;
    }

    out << BuildV3GraphvizDot(v3BfsNodes);
    out.close();
    if (!out) {
        HCCL_VM_WARN("[TaskGraphGeneratorV3][GraphvizDot][v3] Failed to write graph file: {}", path);
        return HCCL_E_INTERNAL;
    }

    if (dumpPath != nullptr) {
        *dumpPath = path;
    }
    HCCL_VM_INFO(
        "[TaskGraphGeneratorV3][GraphvizDot][v3] Graphviz DAG dumped, path={}, nodeCount={}", path, v3BfsNodes.size());
    return HCCL_SUCCESS;
}

} // namespace HcclSim
