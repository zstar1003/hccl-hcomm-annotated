/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump_v3/dump_v3_dag.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann_json/json.hpp>

#include "dump/dump_run_manifest.h"
#include "dump_v3/dump_v3_manager.h"
#include "framework/task_graph_generator_v3/task_graph_reachability_v3.h"
#include "sim_log.h"
#include "storage_manager.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace {
    using Json = nlohmann::json;
    using namespace TaskGraphGeneratorV3;

    constexpr uint32_t kSchemaVersion = 1U;
    constexpr uint64_t kMsSliceBytes = 4ULL * 1024ULL;
    const std::string kGraphModel = "v3";
    const std::string kGraphPath = "graph/graph.msgpack";
    const std::string kLayoutPath = "graph/layout.msgpack";

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
    };

    struct NodeDumpInfo {
        const TaskNode* node{nullptr};
        int laneId{-1};
        uint32_t streamSeq{0};
        uint32_t semanticStep{0};
        uint32_t col{0};
    };

    struct EdgeDumpInfo {
        int id{0};
        NodeId src{INVALID_NODE_ID};
        NodeId dst{INVALID_NODE_ID};
        std::string kind;
        bool primaryAlign{false};
    };

    bool IsMainGraphStartNode(const TaskNode* node)
    {
        if (node == nullptr || node->GetType() != TaskType::START || node->GetNodeId() != MAIN_START_NODE_ID) {
            return false;
        }
        const auto* start = dynamic_cast<const TaskStart*>(node);
        return start != nullptr && start->GetBoundaryType() == BoundaryType::MAIN_GRAPH;
    }

    QueueId NormalizeQueueId(QueueId queueId) { return queueId == INVALID_QUEUE_ID ? 0U : queueId; }

    LaneKey MakeLaneKey(const TaskNode* node)
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

    std::string ToString(TaskType type)
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

    std::string ToString(ProtocolType type)
    {
        switch (type) {
            case ProtocolType::SDMA:
                return "SDMA";
            case ProtocolType::RDMA:
                return "RDMA";
            case ProtocolType::CCU:
                return "CCU";
            default:
                return "INVALID";
        }
    }

    std::string ToString(MemType type)
    {
        switch (type) {
            case MemType::INPUT:
                return "INPUT";
            case MemType::OUTPUT:
                return "OUTPUT";
            case MemType::CCL:
                return "CCL";
            case MemType::AIV_UB:
                return "AIV_UB";
            case MemType::AIV_COMM:
                return "AIV_COMM";
            case MemType::MS_CCU:
                return "MS_CCU";
            default:
                return "INVALID";
        }
    }

    std::string ToString(BoundaryType type)
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

    bool IsNotifyEdge(const TaskNode* srcNode, const TaskNode* dstNode)
    {
        return srcNode != nullptr && dstNode != nullptr && srcNode->GetType() == TaskType::RECORD
               && dstNode->GetType() == TaskType::WAIT;
    }

    std::string GetProtocolName(const TaskNode* node)
    {
        if (node == nullptr) {
            return {};
        }
        switch (node->GetType()) {
            case TaskType::TRANS_MEM: {
                const auto* task = dynamic_cast<const TaskTransMem*>(node);
                if (task == nullptr) {
                    return {};
                }
                return task->GetProtocol() == ProtocolType::INVALID ? std::string{} : ToString(task->GetProtocol());
            }
            case TaskType::BATCH_TRANS_MEM: {
                const auto* task = dynamic_cast<const TaskBatchTransMem*>(node);
                if (task == nullptr) {
                    return {};
                }
                return task->GetProtocol() == ProtocolType::INVALID ? std::string{} : ToString(task->GetProtocol());
            }
            case TaskType::REDUCE: {
                const auto* task = dynamic_cast<const TaskReduce*>(node);
                if (task == nullptr) {
                    return {};
                }
                return task->GetProtocol() == ProtocolType::INVALID ? std::string{} : ToString(task->GetProtocol());
            }
            case TaskType::BATCH_REDUCE: {
                const auto* task = dynamic_cast<const TaskBatchReduce*>(node);
                if (task == nullptr) {
                    return {};
                }
                return task->GetProtocol() == ProtocolType::INVALID ? std::string{} : ToString(task->GetProtocol());
            }
            case TaskType::RECORD: {
                if (const auto* task = dynamic_cast<const TaskRecordAICPU*>(node)) {
                    return task->GetProtocol() == ProtocolType::INVALID ? std::string{} : ToString(task->GetProtocol());
                }
                if (const auto* task = dynamic_cast<const TaskRecordCCU*>(node)) {
                    return task->GetProtocol() == ProtocolType::INVALID ? std::string{} : ToString(task->GetProtocol());
                }
                return {};
            }
            case TaskType::WAIT: {
                if (const auto* task = dynamic_cast<const TaskWaitAICPU*>(node)) {
                    return task->GetProtocol() == ProtocolType::INVALID ? std::string{} : ToString(task->GetProtocol());
                }
                if (const auto* task = dynamic_cast<const TaskWaitCCU*>(node)) {
                    return task->GetProtocol() == ProtocolType::INVALID ? std::string{} : ToString(task->GetProtocol());
                }
                return {};
            }
            default:
                return {};
        }
    }

    Json ToJson(const MemSlice& slice)
    {
        Json json = Json::object();
        json["rank_id"] = slice.rankId;
        json["mem_type"] = ToString(slice.memType);
        json["offset"] = slice.offset;
        json["len"] = slice.len;
        if (slice.memType == MemType::MS_CCU && slice.offset % kMsSliceBytes == 0ULL) {
            json["ms_id"] = slice.offset / kMsSliceBytes;
        }
        return json;
    }

    template <typename ValueType>
    Json ToRegisterEntries(const std::map<uint16_t, ValueType>& registers)
    {
        Json entries = Json::array();
        for (const auto& entry : registers) {
            Json compactEntry = Json::object();
            compactEntry[std::to_string(entry.first)] = entry.second;
            entries.push_back(std::move(compactEntry));
        }
        return entries;
    }

    Json ToJson(const std::vector<MemSlice>& slices)
    {
        Json json = Json::array();
        for (const auto& slice : slices) {
            json.push_back(ToJson(slice));
        }
        return json;
    }

    Json ToJson(const std::vector<std::vector<MemSlice>>& sliceGroups)
    {
        Json json = Json::array();
        for (const auto& group : sliceGroups) {
            json.push_back(ToJson(group));
        }
        return json;
    }

    Json ToJson(const CcuRegisterSnapshot& snapshot)
    {
        Json json = Json::object();
        if (!snapshot.xn.empty()) {
            json["xn"] = ToRegisterEntries(snapshot.xn);
        }
        if (!snapshot.gsa.empty()) {
            json["gsa"] = ToRegisterEntries(snapshot.gsa);
        }
        if (!snapshot.cke.empty()) {
            json["cke"] = ToRegisterEntries(snapshot.cke);
        }
        return json;
    }

    Json ToJson(const CcuTraceInfo& trace)
    {
        Json json = Json::object();
        json["op_name"] = trace.opName;
        json["node_role"] = trace.nodeRole;
        json["instr_id"] = trace.instrId;
        json["queue_id"] = NormalizeQueueId(trace.queueId);
        if (trace.dieId != INVALID_DIE_ID) {
            json["die_id"] = trace.dieId;
        }
        if (trace.missionId != INVALID_MISSION_ID) {
            json["mission_id"] = trace.missionId;
        }

        Json jsonLoc = Json::object();
        jsonLoc["rank_id"] = trace.taskLoc.rankId;
        jsonLoc["stream_id"] = trace.taskLoc.streamId;
        jsonLoc["queue_id"] = NormalizeQueueId(trace.taskLoc.queueId);
        json["location"] = std::move(jsonLoc);

        Json registerState = ToJson(trace.registerState);
        if (!registerState.empty()) {
            json["register_state"] = std::move(registerState);
        }

        // loop start 节点的元数据
        if (trace.loopInstrIdStart != UINT16_MAX) {
            json["loop_instr_id_start"] = trace.loopInstrIdStart;
            json["loop_instr_id_end"] = trace.loopInstrIdEnd;
        }
        if (trace.loopCnt != UINT64_MAX) {
            json["loop_cnt"] = trace.loopCnt;
        }
        if (trace.loopExpandCnt != UINT64_MAX) {
            json["loop_expand_cnt"] = trace.loopExpandCnt;
        }
        if (trace.loopEndNodeId != INVALID_NODE_ID) {
            json["loop_end_node_id"] = trace.loopEndNodeId;
        }

        // loop 体内节点：所属 loop start 的 nodeId
        if (trace.loopStartNodeId != INVALID_NODE_ID) {
            json["loop_start_node_id"] = trace.loopStartNodeId;
        }

        return json;
    }

    void PopulateMemorySliceFields(const TaskNode* node, Json& nodeJson)
    {
        if (node == nullptr) {
            return;
        }
        switch (node->GetType()) {
            case TaskType::TRANS_MEM: {
                const auto* task = dynamic_cast<const TaskTransMem*>(node);
                if (task != nullptr) {
                    nodeJson["src_slice"] = ToJson(task->GetSrc());
                    nodeJson["dst_slice"] = ToJson(task->GetDst());
                }
                return;
            }
            case TaskType::BATCH_TRANS_MEM: {
                const auto* task = dynamic_cast<const TaskBatchTransMem*>(node);
                if (task != nullptr) {
                    nodeJson["src_slices"] = ToJson(task->GetSrcs());
                    nodeJson["dst_slices"] = ToJson(task->GetDsts());
                    nodeJson["merged_src_slices"] = ToJson(task->GetMergedSrcs());
                    nodeJson["merged_dst_slices"] = ToJson(task->GetMergedDsts());
                }
                return;
            }
            case TaskType::REDUCE: {
                const auto* task = dynamic_cast<const TaskReduce*>(node);
                if (task != nullptr) {
                    const auto& srcs = task->GetSrcs();
                    if (!srcs.empty()) {
                        nodeJson["src_slices"] = ToJson(srcs);
                    } else {
                        nodeJson["src_slice"] = ToJson(task->GetSrc());
                    }
                    nodeJson["dst_slice"] = ToJson(task->GetDst());
                }
                return;
            }
            case TaskType::BATCH_REDUCE: {
                const auto* task = dynamic_cast<const TaskBatchReduce*>(node);
                if (task != nullptr) {
                    nodeJson["src_groups"] = ToJson(task->GetSrcs());
                    nodeJson["dst_slices"] = ToJson(task->GetDsts());
                    nodeJson["merged_src_groups"] = ToJson(task->GetMergedSrcs());
                    nodeJson["merged_dst_slices"] = ToJson(task->GetMergedDsts());
                }
                return;
            }
            default:
                return;
        }
    }

    Json MakeNotifyJson(const TaskNode* node)
    {
        if (const auto* record = dynamic_cast<const TaskRecordAICPU*>(node)) {
            const auto& notify = record->GetNotify();
            return Json::object({
                {"kind", "aicpu"},
                {"record_rank_id", notify.recordRankId},
                {"wait_rank_id", notify.waitRankId},
                {"notify_id", notify.notifyId},
            });
        }
        if (const auto* wait = dynamic_cast<const TaskWaitAICPU*>(node)) {
            const auto& notify = wait->GetNotify();
            return Json::object({
                {"kind", "aicpu"},
                {"record_rank_id", notify.recordRankId},
                {"wait_rank_id", notify.waitRankId},
                {"notify_id", notify.notifyId},
            });
        }
        if (const auto* record = dynamic_cast<const TaskRecordCCU*>(node)) {
            const auto& notify = record->GetNotify();
            return Json::object({
                {"kind", "ccu"},
                {"record_rank_id", notify.recordRankId},
                {"wait_rank_id", notify.waitRankId},
                {"channel_id", notify.channelId},
                {"die_id", notify.dieId},
                {"cke_id", notify.ckeId},
                {"cke_mask", notify.ckeMask},
            });
        }
        if (const auto* wait = dynamic_cast<const TaskWaitCCU*>(node)) {
            const auto& notify = wait->GetNotify();
            return Json::object({
                {"kind", "ccu"},
                {"record_rank_id", notify.recordRankId},
                {"wait_rank_id", notify.waitRankId},
                {"channel_id", notify.channelId},
                {"die_id", notify.dieId},
                {"cke_id", notify.ckeId},
                {"cke_mask", notify.ckeMask},
            });
        }
        return Json();
    }

    Json MakeNodeJson(const NodeDumpInfo& nodeInfo)
    {
        const TaskNode* node = nodeInfo.node;
        Json nodeJson = Json::object();
        nodeJson["id"] = node->GetNodeId();
        nodeJson["task_type"] = ToString(node->GetType());
        nodeJson["lane_id"] = nodeInfo.laneId;
        nodeJson["stream_seq"] = nodeInfo.streamSeq;
        nodeJson["semantic_step"] = nodeInfo.semanticStep;

        const std::string protocolName = GetProtocolName(node);
        if (!protocolName.empty()) {
            nodeJson["protocol_type"] = protocolName;
        }

        PopulateMemorySliceFields(node, nodeJson);

        if (node->GetType() == TaskType::RECORD) {
            nodeJson["notify_role"] = "record";
            nodeJson["notify"] = MakeNotifyJson(node);
        } else if (node->GetType() == TaskType::WAIT) {
            nodeJson["notify_role"] = "wait";
            nodeJson["notify"] = MakeNotifyJson(node);
        }

        if (const auto* reduce = dynamic_cast<const TaskReduce*>(node)) {
            nodeJson["reduce_op"] = reduce->GetReduceOp();
            nodeJson["data_type"] = reduce->GetDataType();
        } else if (const auto* batchReduce = dynamic_cast<const TaskBatchReduce*>(node)) {
            nodeJson["reduce_op"] = batchReduce->GetReduceOp();
            nodeJson["data_type"] = batchReduce->GetDataType();
        }

        if (const auto* start = dynamic_cast<const TaskStart*>(node)) {
            nodeJson["boundary_type"] = ToString(start->GetBoundaryType());
        } else if (const auto* end = dynamic_cast<const TaskEnd*>(node)) {
            nodeJson["boundary_type"] = ToString(end->GetBoundaryType());
        }

        if (node->HasCcuTrace()) {
            nodeJson["ccu_trace"] = ToJson(node->GetCcuTrace());
        }

        return nodeJson;
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

    HcclResult
    CollectTopoNodes(const TaskGraphGeneratorV3::TaskGraphGeneratorV3& graph, std::vector<const TaskNode*>& topoNodes)
    {
        topoNodes.clear();
        const TaskNode* mainStart = graph.GetMainStartNode();
        if (!IsMainGraphStartNode(mainStart)) {
            HCCL_VM_ERROR(
                "{} Main start node of the V3 graph is invalid, mainStartNodeId={}",
                MakeErrorCodeText(ErrorCode::DUMP_FAILED),
                mainStart == nullptr ? MAIN_START_NODE_ID : mainStart->GetNodeId());
            return HCCL_E_PARA;
        }

        ReachabilityClosure closure;
        std::vector<NodeId> topoOrder;
        HcclResult ret = GenReachabilityClosure(mainStart, closure, &topoOrder);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to build the V3 graph topological order for dump output, "
                "mainStartNodeId={}, ret={}",
                MakeErrorCodeText(ErrorCode::DUMP_FAILED), mainStart->GetNodeId(), static_cast<uint32_t>(ret));
            return ret;
        }

        topoNodes.reserve(topoOrder.size());
        for (NodeId nodeId : topoOrder) {
            if (nodeId == MAIN_START_NODE_ID) {
                continue;
            }
            const TaskNode* node = graph.GetNode(nodeId);
            if (node == nullptr) {
                HCCL_VM_ERROR(
                    "{} One node in topological order is missing, nodeId={}, "
                    "processedTopoNodeCount={}",
                    MakeErrorCodeText(ErrorCode::DUMP_FAILED), nodeId, topoNodes.size());
                return HCCL_E_PTR;
            }
            topoNodes.push_back(node);
        }
        return HCCL_SUCCESS;
    }

    Json MakeStageStats(
        const std::vector<Json>& laneDocs, const std::vector<Json>& nodeDocs, const std::vector<Json>& edgeDocs)
    {
        Json stageStats = Json::object();
        std::map<RankId, size_t> laneCountByRank;
        std::map<RankId, size_t> nodeCountByRank;
        std::map<RankId, size_t> edgeCountByRank;
        std::map<int, RankId> rankByLaneId;

        for (const auto& laneDoc : laneDocs) {
            const RankId rankId = laneDoc.value("rank_id", INVALID_RANK_ID);
            const int laneId = laneDoc.value("id", -1);
            if (laneId >= 0) {
                rankByLaneId[laneId] = rankId;
            }
            if (rankId != INVALID_RANK_ID) {
                ++laneCountByRank[rankId];
            }
        }

        std::map<NodeId, RankId> rankByNodeId;
        for (const auto& nodeDoc : nodeDocs) {
            const int laneId = nodeDoc.value("lane_id", -1);
            const auto rankIter = rankByLaneId.find(laneId);
            if (rankIter == rankByLaneId.end()) {
                continue;
            }
            const NodeId nodeId = nodeDoc.value("id", INVALID_NODE_ID);
            rankByNodeId[nodeId] = rankIter->second;
            ++nodeCountByRank[rankIter->second];
        }

        for (const auto& edgeDoc : edgeDocs) {
            const NodeId src = edgeDoc.value("src", INVALID_NODE_ID);
            const auto rankIter = rankByNodeId.find(src);
            if (rankIter != rankByNodeId.end()) {
                ++edgeCountByRank[rankIter->second];
            }
        }

        stageStats["rank_count"] = laneCountByRank.size();
        stageStats["lane_count"] = laneDocs.size();
        stageStats["queue_count"] = laneDocs.size();
        stageStats["node_count"] = nodeDocs.size();
        stageStats["task_count"] = nodeDocs.size();
        stageStats["edge_count"] = edgeDocs.size();
        stageStats["ranks"] = Json::array();
        for (const auto& laneItem : laneCountByRank) {
            Json rankStats = Json::object();
            rankStats["rank_id"] = laneItem.first;
            rankStats["queue_count"] = laneItem.second;
            rankStats["node_count"] = nodeCountByRank[laneItem.first];
            rankStats["task_count"] = nodeCountByRank[laneItem.first];
            rankStats["edge_count"] = edgeCountByRank[laneItem.first];
            stageStats["ranks"].push_back(std::move(rankStats));
        }
        return stageStats;
    }

} // namespace

HcclResult DumpV3Dag(const TaskGraphGeneratorV3::TaskGraphGeneratorV3& graph, const std::string& stage)
{
    DumpV3Manager& dumpManager = DumpV3Manager::GetInstance();
    if (!dumpManager.IsEnabled()) {
        return HCCL_SUCCESS;
    }
    if (stage.empty()) {
        HCCL_VM_ERROR(
            "{} Dump stage name is empty, dataId={}, graphOutputPath={}, layoutOutputPath={}",
            MakeErrorCodeText(ErrorCode::DUMP_FAILED), StorageManager::GetInstance().GetDataId(), kGraphPath,
            kLayoutPath);
        return HCCL_E_PARA;
    }

    std::vector<const TaskNode*> topoNodes;
    HcclResult ret = CollectTopoNodes(graph, topoNodes);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    std::map<LaneKey, int> laneIdByKey;
    std::map<RankId, int> laneOrderByRank;
    std::set<RankId> rankIds;
    for (const TaskNode* node : topoNodes) {
        const LaneKey laneKey = MakeLaneKey(node);
        rankIds.insert(laneKey.rankId);
        if (laneIdByKey.count(laneKey) != 0) {
            continue;
        }
        const int laneId = static_cast<int>(laneIdByKey.size());
        laneIdByKey[laneKey] = laneId;
    }

    std::map<NodeId, NodeDumpInfo> nodeInfoById;
    std::map<int, uint32_t> laneStreamSeq;
    for (size_t index = 0; index < topoNodes.size(); ++index) {
        const TaskNode* node = topoNodes[index];
        const int laneId = laneIdByKey[MakeLaneKey(node)];
        NodeDumpInfo nodeInfo;
        nodeInfo.node = node;
        nodeInfo.laneId = laneId;
        nodeInfo.streamSeq = laneStreamSeq[laneId]++;
        nodeInfo.semanticStep = static_cast<uint32_t>(index + 1U);
        nodeInfoById[node->GetNodeId()] = nodeInfo;
    }

    std::map<NodeId, std::pair<LaneKey, NodeId>> primaryRecordByWait;
    std::vector<EdgeDumpInfo> edgeInfos;
    size_t estimatedEdgeCount = 0;
    for (const TaskNode* node : topoNodes) {
        estimatedEdgeCount += node->GetChildren().size();
    }
    edgeInfos.reserve(estimatedEdgeCount);
    for (const TaskNode* node : topoNodes) {
        const auto& children = node->GetChildren();
        for (const TaskNode* child : children) {
            if (child == nullptr || child->GetNodeId() == MAIN_START_NODE_ID) {
                continue;
            }
            const auto srcIter = nodeInfoById.find(node->GetNodeId());
            const auto dstIter = nodeInfoById.find(child->GetNodeId());
            if (srcIter == nodeInfoById.end() || dstIter == nodeInfoById.end()) {
                continue;
            }

            EdgeDumpInfo edgeInfo;
            edgeInfo.id = static_cast<int>(edgeInfos.size());
            edgeInfo.src = node->GetNodeId();
            edgeInfo.dst = child->GetNodeId();
            edgeInfo.kind = IsNotifyEdge(node, child) ?
                                "notify" :
                                ((srcIter->second.laneId == dstIter->second.laneId) ? "lane_order" : "dependency");
            edgeInfos.push_back(edgeInfo);

            if (!IsNotifyEdge(node, child)) {
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
    for (const TaskNode* node : topoNodes) {
        const NodeId nodeId = node->GetNodeId();
        const int laneId = nodeInfoById[nodeId].laneId;
        const auto lastLaneIter = lastNodeByLaneId.find(laneId);
        if (lastLaneIter != lastNodeByLaneId.end()) {
            incomingLayoutEdges[nodeId].push_back(std::make_pair(lastLaneIter->second, 1U));
        }
        lastNodeByLaneId[laneId] = nodeId;

        for (const TaskNode* parent : node->GetParents()) {
            if (parent == nullptr || parent->GetNodeId() == MAIN_START_NODE_ID) {
                continue;
            }
            if (nodeInfoById.count(parent->GetNodeId()) == 0) {
                continue;
            }

            uint32_t weight = 1U;
            if (IsNotifyEdge(parent, node)) {
                const auto primaryIter = primaryRecordByWait.find(nodeId);
                if (primaryIter != primaryRecordByWait.end() && primaryIter->second.second == parent->GetNodeId()) {
                    weight = 0U;
                }
            }
            incomingLayoutEdges[nodeId].push_back(std::make_pair(parent->GetNodeId(), weight));
        }
    }

    for (const TaskNode* node : topoNodes) {
        NodeDumpInfo& nodeInfo = nodeInfoById[node->GetNodeId()];
        uint32_t col = 0U;
        const auto incomingIter = incomingLayoutEdges.find(node->GetNodeId());
        if (incomingIter != incomingLayoutEdges.end()) {
            for (const auto& entry : incomingIter->second) {
                const auto parentInfoIter = nodeInfoById.find(entry.first);
                if (parentInfoIter == nodeInfoById.end()) {
                    continue;
                }
                col = std::max(col, parentInfoIter->second.col + entry.second);
            }
        }
        nodeInfo.col = col;
    }

    std::vector<Json> laneDocs;
    laneDocs.reserve(laneIdByKey.size());
    std::vector<Json> laneLayoutDocs;
    laneLayoutDocs.reserve(laneIdByKey.size());
    uint32_t row = 0U;
    for (const auto& lanePair : laneIdByKey) {
        const LaneKey& laneKey = lanePair.first;
        const int laneId = lanePair.second;
        const int orderInRank = laneOrderByRank[laneKey.rankId]++;

        Json laneDoc = Json::object();
        laneDoc["id"] = laneId;
        laneDoc["rank_id"] = laneKey.rankId;
        laneDoc["stream_id"] = laneKey.streamId;
        laneDoc["queue_id"] = laneKey.queueId;
        laneDoc["order_in_rank"] = orderInRank;
        laneDocs.push_back(std::move(laneDoc));

        Json laneLayoutDoc = Json::object();
        laneLayoutDoc["lane_id"] = laneId;
        laneLayoutDoc["row"] = row++;
        laneLayoutDoc["rank_order"] = laneKey.rankId;
        laneLayoutDoc["stream_order"] = laneKey.streamId;
        laneLayoutDocs.push_back(std::move(laneLayoutDoc));
    }

    std::vector<Json> nodeDocs;
    nodeDocs.reserve(topoNodes.size());
    std::vector<Json> nodeLayoutDocs;
    nodeLayoutDocs.reserve(topoNodes.size());
    std::map<NodeId, uint32_t> colByNodeId;
    for (const TaskNode* node : topoNodes) {
        const NodeDumpInfo& nodeInfo = nodeInfoById[node->GetNodeId()];
        nodeDocs.push_back(MakeNodeJson(nodeInfo));

        Json nodeLayout = Json::object();
        nodeLayout["node_id"] = node->GetNodeId();
        nodeLayout["col"] = nodeInfo.col;
        nodeLayoutDocs.push_back(std::move(nodeLayout));
        colByNodeId[node->GetNodeId()] = nodeInfo.col;
    }

    std::vector<Json> edgeDocs;
    edgeDocs.reserve(edgeInfos.size());
    std::vector<Json> notifyPairDocs;
    for (auto& edgeInfo : edgeInfos) {
        if (edgeInfo.kind == "notify") {
            const auto primaryIter = primaryRecordByWait.find(edgeInfo.dst);
            if (primaryIter != primaryRecordByWait.end() && primaryIter->second.second == edgeInfo.src
                && colByNodeId[edgeInfo.src] == colByNodeId[edgeInfo.dst]) {
                edgeInfo.primaryAlign = true;
            }

            Json notifyPair = Json::object();
            notifyPair["record"] = edgeInfo.src;
            notifyPair["wait"] = edgeInfo.dst;
            notifyPair["align_mode"] = edgeInfo.primaryAlign ? "primary_same_col" : "dependency_only";
            notifyPairDocs.push_back(std::move(notifyPair));
        }

        Json edgeDoc = Json::object();
        edgeDoc["id"] = edgeInfo.id;
        edgeDoc["src"] = edgeInfo.src;
        edgeDoc["dst"] = edgeInfo.dst;
        edgeDoc["kind"] = edgeInfo.kind;
        edgeDoc["primary_align"] = edgeInfo.primaryAlign;
        edgeDocs.push_back(std::move(edgeDoc));
    }

    Json graphDoc = Json::object();
    graphDoc["schema_version"] = kSchemaVersion;
    graphDoc["graph_model"] = kGraphModel;
    graphDoc["stage"] = stage;
    graphDoc["data_id"] = StorageManager::GetInstance().GetDataId();
    graphDoc["ranks"] = Json::array();
    for (RankId rankId : rankIds) {
        graphDoc["ranks"].push_back(rankId);
    }
    graphDoc["lanes"] = std::move(laneDocs);
    graphDoc["nodes"] = std::move(nodeDocs);
    graphDoc["edges"] = std::move(edgeDocs);
    graphDoc["notify_pairs"] = std::move(notifyPairDocs);
    graphDoc["stats"] = Json::object();
    graphDoc["stats"]["rank_count"] = rankIds.size();
    graphDoc["stats"]["lane_count"] = laneIdByKey.size();
    graphDoc["stats"]["node_count"] = topoNodes.size();
    graphDoc["stats"]["edge_count"] = edgeInfos.size();
    graphDoc["stats"]["semantic_step_count"] = topoNodes.size();

    Json layoutDoc = Json::object();
    layoutDoc["schema_version"] = kSchemaVersion;
    layoutDoc["graph_model"] = kGraphModel;
    layoutDoc["stage"] = stage;
    layoutDoc["lane_layout"] = std::move(laneLayoutDocs);
    layoutDoc["node_layout"] = std::move(nodeLayoutDocs);
    layoutDoc["edge_layout"] = Json::array();

    ret = dumpManager.WriteMsgpack(kGraphPath, graphDoc);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    ret = dumpManager.WriteMsgpack(kLayoutPath, layoutDoc);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    DumpRunManifest::GetInstance().SetGraphStageStats(
        stage, MakeStageStats(
                   graphDoc["lanes"].get<std::vector<Json>>(), graphDoc["nodes"].get<std::vector<Json>>(),
                   graphDoc["edges"].get<std::vector<Json>>()));

    HCCL_VM_INFO(
        "V3 DAG dump finished, laneCount={}, nodeCount={}, edgeCount={}", laneIdByKey.size(), topoNodes.size(),
        edgeInfos.size());
    return HCCL_SUCCESS;
}
} // namespace HcclSim
