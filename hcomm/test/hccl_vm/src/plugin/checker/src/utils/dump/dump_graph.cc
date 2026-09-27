/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump/dump_graph.h"
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <queue>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#include "dump/dump_json_utils.h"
#include "dump/dump_manager.h"
#include "dump/dump_run_manifest.h"
#include "sim_log.h"
#include "task_ccu.h"
#include "task_utils.h"

namespace HcclSim {
using Json = nlohmann::json;

static const std::string DUMP_GRAPH_ROOT_DIR = "graph";
static const std::string INPUT_TASK_QUEUE_STAGE = "input_task_queues";
static const std::string TASK_GRAPH_DUMP_TYPE = "task_graph";
static const std::string INPUT_QUEUE_DUMP_TYPE = "input_task_queues";

static std::string BuildStageRankDumpPath(const std::string& stage, RankId rankId)
{
    return StringFormat("%s/%s/rank_%u.msgpack", DUMP_GRAPH_ROOT_DIR.c_str(), stage.c_str(), rankId);
}

static std::string TaskMetaTypeToString(HccLTaskMetaType taskType)
{
    switch (taskType) {
        case HccLTaskMetaType::NOTIFY_WAIT:
            return "NOTIFY_WAIT";
        case HccLTaskMetaType::NOTIFY_RECORD:
            return "NOTIFY_RECORD";
        case HccLTaskMetaType::REDUCE:
            return "REDUCE";
        case HccLTaskMetaType::MEM_CPY:
            return "MEM_CPY";
        case HccLTaskMetaType::CCU_GRAPH:
            return "CCU_GRAPH";
        case HccLTaskMetaType::AIV_GRAPH:
            return "AIV_GRAPH";
        case HccLTaskMetaType::EVENT_WAIT:
            return "EVENT_WAIT";
        case HccLTaskMetaType::EVENT_RECORD:
            return "EVENT_RECORD";
        default:
            return "UNKNOWN";
    }
}

static std::string DumpHcclDataTypeToString(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
            return "INT8";
        case HCCL_DATA_TYPE_INT16:
            return "INT16";
        case HCCL_DATA_TYPE_INT32:
            return "INT32";
        case HCCL_DATA_TYPE_FP16:
            return "FP16";
        case HCCL_DATA_TYPE_FP32:
            return "FP32";
        case HCCL_DATA_TYPE_INT64:
            return "INT64";
        case HCCL_DATA_TYPE_UINT64:
            return "UINT64";
        case HCCL_DATA_TYPE_UINT8:
            return "UINT8";
        case HCCL_DATA_TYPE_UINT16:
            return "UINT16";
        case HCCL_DATA_TYPE_UINT32:
            return "UINT32";
        case HCCL_DATA_TYPE_FP64:
            return "FP64";
        case HCCL_DATA_TYPE_BFP16:
            return "BFP16";
        case HCCL_DATA_TYPE_INT128:
            return "INT128";
        case HCCL_DATA_TYPE_HIF8:
            return "HIF8";
        case HCCL_DATA_TYPE_FP8E4M3:
            return "FP8E4M3";
        case HCCL_DATA_TYPE_FP8E5M2:
            return "FP8E5M2";
        case HCCL_DATA_TYPE_FP8E8M0:
            return "FP8E8M0";
        case HCCL_DATA_TYPE_RESERVED:
            return "RESERVED";
        default:
            return "UNKNOWN";
    }
}

static Json DumpLinkProtoStubToJson(LinkProtoStub linkProto)
{
    switch (linkProto) {
        case LinkProtoStub::SDMA:
            return "SDMA";
        case LinkProtoStub::RDMA:
            return "RDMA";
        case LinkProtoStub::CCU:
            return "CCU";
        case LinkProtoStub::INVALID_A:
            return "INVALID";
        default:
            return "UNKNOWN";
    }
}

static Json MakeTaskMetaFields(const HcclTaskMetaData& taskMeta)
{
    Json fields = Json::object();
    switch (taskMeta.taskType) {
        case HccLTaskMetaType::NOTIFY_WAIT:
        case HccLTaskMetaType::NOTIFY_RECORD:
            fields["src_rank_id"] = taskMeta.taskData.notify.srcRankId;
            fields["notify_id"] = taskMeta.taskData.notify.notifyId;
            fields["dst_rank_id"] = taskMeta.taskData.notify.dstRankId;
            fields["notify_count"] = taskMeta.taskData.notify.notifyCount;
            fields["protocol"] = taskMeta.taskData.notify.protocol;
            break;
        case HccLTaskMetaType::REDUCE:
            fields["src_rank_id"] = taskMeta.taskData.reduce.srcRankId;
            fields["src_offset"] = taskMeta.taskData.reduce.srcOffset;
            fields["dst_rank_id"] = taskMeta.taskData.reduce.dstRankId;
            fields["dst_offset"] = taskMeta.taskData.reduce.dstOffset;
            fields["data_type"]
                = DumpHcclDataTypeToString(static_cast<HcclDataType>(taskMeta.taskData.reduce.dataType));
            fields["data_count"] = taskMeta.taskData.reduce.dataCount;
            fields["reduce_op"] = DumpReduceOpToString(static_cast<HcclReduceOp>(taskMeta.taskData.reduce.reduceOp));
            fields["protocol"] = taskMeta.taskData.reduce.protocol;
            break;
        case HccLTaskMetaType::MEM_CPY:
            fields["src_rank_id"] = taskMeta.taskData.transMem.srcRankId;
            fields["src_offset"] = taskMeta.taskData.transMem.srcOffset;
            fields["dst_rank_id"] = taskMeta.taskData.transMem.dstRankId;
            fields["dst_offset"] = taskMeta.taskData.transMem.dstOffset;
            fields["len"] = taskMeta.taskData.transMem.len;
            fields["protocol"] = taskMeta.taskData.transMem.protocol;
            break;
        case HccLTaskMetaType::CCU_GRAPH:
            fields["die_id"] = taskMeta.taskData.ccu.dieId;
            fields["mission_id"] = taskMeta.taskData.ccu.missionId;
            fields["timeout"] = taskMeta.taskData.ccu.timeout;
            fields["inst_start_id"] = taskMeta.taskData.ccu.instStartId;
            fields["inst_cnt"] = taskMeta.taskData.ccu.instCnt;
            fields["key"] = taskMeta.taskData.ccu.key;
            fields["arg_size"] = taskMeta.taskData.ccu.argSize;
            fields["args"] = Json::array();
            for (uint32_t i = 0; i < RT_CCU_SQE_ARGS_LEN; ++i) {
                fields["args"].push_back(taskMeta.taskData.ccu.args[i]);
            }
            break;
        case HccLTaskMetaType::AIV_GRAPH:
            break;
        case HccLTaskMetaType::EVENT_WAIT:
            break;
        case HccLTaskMetaType::EVENT_RECORD:
            break;
        default:
            break;
    }
    return fields;
}

static Json MakeTaskStubFields(TaskStub* task, const std::map<TaskNode*, std::string>* nodeIdMap = nullptr);
static Json
DumpCcuSingleQueue(TaskNode* headNode, RankId rankId, u32 queueIdx, const std::map<TaskNode*, std::string>* nodeIdMap);
static Json DumpCcuSubGraph(TaskStubCcuGraph* ccuGraphTask, const std::map<TaskNode*, std::string>* nodeIdMap);
static Json DumpTaskStubToJsonWithNodeMap(TaskStub* task, const std::map<TaskNode*, std::string>* nodeIdMap);

static Json MakeTaskStubFields(TaskStub* task, const std::map<TaskNode*, std::string>* nodeIdMap)
{
    Json fields = Json::object();
    if (task == nullptr) {
        return fields;
    }
    const std::string taskType = DumpTaskTypeToString(task->GetType());

    switch (task->GetType()) {
        case TaskTypeStub::LOCAL_COPY: {
            TaskStubLocalCopy* localCopyTask = dynamic_cast<TaskStubLocalCopy*>(task);
            if (localCopyTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubLocalCopy.", taskType);
                break;
            }
            fields["src_slice"] = DumpDataSliceToJson(localCopyTask->GetSrcSlice());
            fields["dst_slice"] = DumpDataSliceToJson(localCopyTask->GetDstSlice());
            fields["is_gen_from_sync"] = localCopyTask->IsGenFromSync();
            break;
        }
        case TaskTypeStub::LOCAL_REDUCE: {
            TaskStubLocalReduce* localReduceTask = dynamic_cast<TaskStubLocalReduce*>(task);
            if (localReduceTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubLocalReduce.", taskType);
                break;
            }
            fields["src_slice"] = DumpDataSliceToJson(localReduceTask->GetSrcSlice());
            fields["dst_slice"] = DumpDataSliceToJson(localReduceTask->GetDstSlice());
            fields["data_type"] = DumpHcclDataTypeToString(localReduceTask->GetDataType());
            fields["reduce_op"] = DumpReduceOpToString(localReduceTask->GetReduceOp());
            fields["is_gen_from_sync"] = localReduceTask->IsGenFromSync();
            break;
        }
        case TaskTypeStub::LOCAL_BATCH_REDUCE: {
            TaskStubLocalBatchReduce* localBatchReduceTask = dynamic_cast<TaskStubLocalBatchReduce*>(task);
            if (localBatchReduceTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubLocalBatchReduce.", taskType);
                break;
            }
            fields["src_slices"] = Json::array();
            for (const auto& srcSlice : localBatchReduceTask->GetSrcSlices()) {
                fields["src_slices"].push_back(DumpDataSliceToJson(srcSlice));
            }
            fields["dst_slice"] = DumpDataSliceToJson(localBatchReduceTask->GetDstSlice());
            fields["data_type"] = DumpHcclDataTypeToString(localBatchReduceTask->GetDataType());
            fields["reduce_op"] = DumpReduceOpToString(localBatchReduceTask->GetReduceOp());
            break;
        }
        case TaskTypeStub::LOCAL_POST_TO: {
            TaskStubLocalPostTo* localPostToTask = dynamic_cast<TaskStubLocalPostTo*>(task);
            if (localPostToTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubLocalPostTo.", taskType);
                break;
            }
            fields["notify_id"] = localPostToTask->GetNotifyId();
            fields["notify_id_back"] = localPostToTask->GetNotifyIdBack();
            fields["topic_id"] = localPostToTask->GetTopicId();
            fields["topic_id_back"] = localPostToTask->GetTopicIdBack();
            fields["post_qid"] = localPostToTask->GetPostQid();
            fields["wait_qid"] = localPostToTask->GetWaitQid();
            fields["invalid_post"] = localPostToTask->IsInvalidPost();
            break;
        }
        case TaskTypeStub::LOCAL_WAIT_FROM: {
            TaskStubLocalWaitFrom* localWaitFromTask = dynamic_cast<TaskStubLocalWaitFrom*>(task);
            if (localWaitFromTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubLocalWaitFrom.", taskType);
                break;
            }
            fields["notify_id"] = localWaitFromTask->GetNotifyId();
            break;
        }
        case TaskTypeStub::POST: {
            TaskStubPost* postTask = dynamic_cast<TaskStubPost*>(task);
            if (postTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubPost.", taskType);
                break;
            }
            fields["remote_rank"] = postTask->GetRemoteRank();
            fields["link_type"] = DumpLinkProtoStubToJson(postTask->GetLinkType());
            fields["notify_id"] = postTask->GetNotifyId();
            fields["topic_id"] = postTask->GetTopicId();
            fields["ccu_task_ptr"] = postTask->ccuTaskPtr_;
            break;
        }
        case TaskTypeStub::WAIT: {
            TaskStubWait* waitTask = dynamic_cast<TaskStubWait*>(task);
            if (waitTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubWait.", taskType);
                break;
            }
            fields["remote_rank"] = waitTask->GetRemoteRank();
            fields["link_type"] = DumpLinkProtoStubToJson(waitTask->GetLinkType());
            fields["notify_id"] = waitTask->GetNotifyId();
            fields["ccu_task_ptr"] = waitTask->ccuTaskPtr_;
            break;
        }
        case TaskTypeStub::READ: {
            TaskStubRead* readTask = dynamic_cast<TaskStubRead*>(task);
            if (readTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubRead.", taskType);
                break;
            }
            fields["remote_rank"] = readTask->GetRemoteRank();
            fields["link_type"] = DumpLinkProtoStubToJson(readTask->GetLinkType());
            fields["local_slice"] = DumpDataSliceToJson(readTask->GetLocalSlice());
            fields["remote_slice"] = DumpDataSliceToJson(readTask->GetRemoteSlice());
            fields["is_gen_from_sync"] = readTask->IsGenFromSync();
            break;
        }
        case TaskTypeStub::READ_REDUCE: {
            TaskStubReadReduce* readReduceTask = dynamic_cast<TaskStubReadReduce*>(task);
            if (readReduceTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubReadReduce.", taskType);
                break;
            }
            fields["remote_rank"] = readReduceTask->GetRemoteRank();
            fields["link_type"] = DumpLinkProtoStubToJson(readReduceTask->GetLinkType());
            fields["local_slice"] = DumpDataSliceToJson(readReduceTask->GetLocalSlice());
            fields["remote_slice"] = DumpDataSliceToJson(readReduceTask->GetRemoteSlice());
            fields["data_type"] = DumpHcclDataTypeToString(readReduceTask->GetDataType());
            fields["reduce_op"] = DumpReduceOpToString(readReduceTask->GetReduceOp());
            fields["is_gen_from_sync"] = readReduceTask->IsGenFromSync();
            break;
        }
        case TaskTypeStub::WRITE: {
            TaskStubWrite* writeTask = dynamic_cast<TaskStubWrite*>(task);
            if (writeTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubWrite.", taskType);
                break;
            }
            fields["remote_rank"] = writeTask->GetRemoteRank();
            fields["link_type"] = DumpLinkProtoStubToJson(writeTask->GetLinkType());
            fields["local_slice"] = DumpDataSliceToJson(writeTask->GetLocalSlice());
            fields["remote_slice"] = DumpDataSliceToJson(writeTask->GetRemoteSlice());
            fields["is_gen_from_sync"] = writeTask->IsGenFromSync();
            break;
        }
        case TaskTypeStub::WRITE_REDUCE: {
            TaskStubWriteReduce* writeReduceTask = dynamic_cast<TaskStubWriteReduce*>(task);
            if (writeReduceTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubWriteReduce.", taskType);
                break;
            }
            fields["remote_rank"] = writeReduceTask->GetRemoteRank();
            fields["link_type"] = DumpLinkProtoStubToJson(writeReduceTask->GetLinkType());
            fields["local_slice"] = DumpDataSliceToJson(writeReduceTask->GetLocalSlice());
            fields["remote_slice"] = DumpDataSliceToJson(writeReduceTask->GetRemoteSlice());
            fields["data_type"] = DumpHcclDataTypeToString(writeReduceTask->GetDataType());
            fields["reduce_op"] = DumpReduceOpToString(writeReduceTask->GetReduceOp());
            fields["is_gen_from_sync"] = writeReduceTask->IsGenFromSync();
            break;
        }
        case TaskTypeStub::LOCAL_POST_TO_SHADOW: {
            TaskStubLocalPostToShadow* localPostToShadowTask = dynamic_cast<TaskStubLocalPostToShadow*>(task);
            if (localPostToShadowTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubLocalPostToShadow.", taskType);
                break;
            }
            fields["neighbor_rank"] = localPostToShadowTask->GetNeighborRank();
            break;
        }
        case TaskTypeStub::LOCAL_WAIT_FROM_SHADOW: {
            TaskStubLocalWaitFromShadow* localWaitFromShadowTask = dynamic_cast<TaskStubLocalWaitFromShadow*>(task);
            if (localWaitFromShadowTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubLocalWaitFromShadow.", taskType);
                break;
            }
            fields["neighbor_rank"] = localWaitFromShadowTask->GetNeighborRank();
            break;
        }
        case TaskTypeStub::BEING_READ: {
            TaskStubBeingRead* beingReadTask = dynamic_cast<TaskStubBeingRead*>(task);
            if (beingReadTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubBeingRead.", taskType);
                break;
            }
            fields["remote_rank"] = beingReadTask->GetRemoteRank();
            fields["link_type"] = DumpLinkProtoStubToJson(beingReadTask->GetLinkType());
            fields["local_slice"] = DumpDataSliceToJson(beingReadTask->GetLocalSlice());
            fields["remote_slice"] = DumpDataSliceToJson(beingReadTask->GetRemoteSlice());
            fields["is_gen_from_sync"] = beingReadTask->IsGenFromSync();
            break;
        }
        case TaskTypeStub::BEING_READ_REDUCE: {
            TaskStubBeingReadReduce* beingReadReduceTask = dynamic_cast<TaskStubBeingReadReduce*>(task);
            if (beingReadReduceTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubBeingReadReduce.", taskType);
                break;
            }
            fields["remote_rank"] = beingReadReduceTask->GetRemoteRank();
            fields["link_type"] = DumpLinkProtoStubToJson(beingReadReduceTask->GetLinkType());
            fields["local_slice"] = DumpDataSliceToJson(beingReadReduceTask->GetLocalSlice());
            fields["remote_slice"] = DumpDataSliceToJson(beingReadReduceTask->GetRemoteSlice());
            fields["data_type"] = DumpHcclDataTypeToString(beingReadReduceTask->GetDataType());
            fields["reduce_op"] = DumpReduceOpToString(beingReadReduceTask->GetReduceOp());
            fields["is_gen_from_sync"] = beingReadReduceTask->IsGenFromSync();
            break;
        }
        case TaskTypeStub::BEING_WRITTEN: {
            TaskStubBeingWritten* beingWrittenTask = dynamic_cast<TaskStubBeingWritten*>(task);
            if (beingWrittenTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubBeingWritten.", taskType);
                break;
            }
            fields["remote_rank"] = beingWrittenTask->GetRemoteRank();
            fields["link_type"] = DumpLinkProtoStubToJson(beingWrittenTask->GetLinkType());
            fields["local_slice"] = DumpDataSliceToJson(beingWrittenTask->GetLocalSlice());
            fields["remote_slice"] = DumpDataSliceToJson(beingWrittenTask->GetRemoteSlice());
            fields["is_gen_from_sync"] = beingWrittenTask->IsGenFromSync();
            break;
        }
        case TaskTypeStub::BEING_WRITTEN_REDUCE: {
            TaskStubBeingWrittenReduce* beingWrittenReduceTask = dynamic_cast<TaskStubBeingWrittenReduce*>(task);
            if (beingWrittenReduceTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubBeingWrittenReduce.", taskType);
                break;
            }
            fields["remote_rank"] = beingWrittenReduceTask->GetRemoteRank();
            fields["link_type"] = DumpLinkProtoStubToJson(beingWrittenReduceTask->GetLinkType());
            fields["local_slice"] = DumpDataSliceToJson(beingWrittenReduceTask->GetLocalSlice());
            fields["remote_slice"] = DumpDataSliceToJson(beingWrittenReduceTask->GetRemoteSlice());
            fields["data_type"] = DumpHcclDataTypeToString(beingWrittenReduceTask->GetDataType());
            fields["reduce_op"] = DumpReduceOpToString(beingWrittenReduceTask->GetReduceOp());
            fields["is_gen_from_sync"] = beingWrittenReduceTask->IsGenFromSync();
            break;
        }
        case TaskTypeStub::CCU_GRAPH: {
            TaskStubCcuGraph* ccuGraphTask = dynamic_cast<TaskStubCcuGraph*>(task);
            if (ccuGraphTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubCcuGraph.", taskType);
                break;
            }
            fields["rank_id"] = ccuGraphTask->rankId;
            fields["queue_num"] = ccuGraphTask->queueNum_;
            fields["sub_graph"] = DumpCcuSubGraph(ccuGraphTask, nodeIdMap);
            break;
        }
        case TaskTypeStub::LOOP_START: {
            TaskStubLoopStart* loopStartTask = dynamic_cast<TaskStubLoopStart*>(task);
            if (loopStartTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubLoopStart.", taskType);
                break;
            }
            fields["loop_idx"] = loopStartTask->loopIdx;
            fields["loop_group_idx"] = loopStartTask->loopGroupIdx;
            break;
        }
        case TaskTypeStub::LOOP_END: {
            TaskStubLoopEnd* loopEndTask = dynamic_cast<TaskStubLoopEnd*>(task);
            if (loopEndTask == nullptr) {
                HCCL_VM_WARN("failed to cast task [{}] to TaskStubLoopEnd.", taskType);
                break;
            }
            fields["loop_idx"] = loopEndTask->loopIdx;
            fields["loop_group_idx"] = loopEndTask->loopGroupIdx;
            break;
        }
        case TaskTypeStub::AIV_TASK:
        case TaskTypeStub::SET_VALUE:
        case TaskTypeStub::SET_FLAG:
        case TaskTypeStub::WAIT_FLAG:
        case TaskTypeStub::SEND_SYNC:
        case TaskTypeStub::RECV_SYNC:
        case TaskTypeStub::SEND_SYNC_REDUCE:
        case TaskTypeStub::COMP_VALUE:
        case TaskTypeStub::PIPE_BARRIER:
        case TaskTypeStub::SUB_GRAPH_END:
        case TaskTypeStub::SET_FLAG_SHADOW:
        case TaskTypeStub::WAIT_FLAG_SHADOW:
        case TaskTypeStub::AIV_START:
        case TaskTypeStub::BLOCK_START:
        case TaskTypeStub::AIV_END:
        case TaskTypeStub::VIRTUAL_RANK_START:
        case TaskTypeStub::GRAPH_SEPARATE:
            break;
        default:
            break;
    }

    return fields;
}

static Json MakeGraphRankJson()
{
    Json rankJson = Json::object();
    rankJson["nodes"] = Json::array();
    rankJson["edges"] = Json::array();
    return rankJson;
}

static bool IsValidRegisteredNode(const TaskNode* node);

static void CollectMainGraphNodes(TaskNode* dummyStart, std::vector<TaskNode*>& graphNodes)
{
    graphNodes.clear();
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

        if (currentNode == dummyStart) {
            continue;
        }
        if (!IsValidRegisteredNode(currentNode)) {
            continue;
        }
        graphNodes.push_back(currentNode);
    }
}

static std::string BuildNodeId(RankId rankId, uint64_t nodeIndex) { return BuildTaskId(rankId, nodeIndex); }

static bool IsValidRegisteredNode(const TaskNode* node)
{
    if (node == nullptr) {
        return false;
    }
    if (node->task == nullptr) {
        return false;
    }
    if (node->rankIdx == static_cast<RankId>(-1)) {
        return false;
    }
    return true;
}

void CollectTaskNodesWithGlobalIds(
    TaskNode* dummyStart, std::vector<TaskNode*>& graphNodes, std::map<TaskNode*, std::string>& nodeIdMap,
    bool includeStartNode)
{
    if (dummyStart == nullptr) {
        return;
    }

    EnsureTaskNodeIdsAssigned(dummyStart);

    std::queue<TaskNode*> visitQueue;
    std::set<TaskNode*> visited;
    std::map<RankId, uint64_t> rankNodeCounters;

    auto registerNode = [&rankNodeCounters, &graphNodes, &nodeIdMap](TaskNode* node) {
        if (!IsValidRegisteredNode(node) || nodeIdMap.count(node) != 0) {
            return;
        }
        std::string nodeId = node->task->GetTaskId();
        if (nodeId.empty()) {
            HCCL_VM_ERROR("task id not found, fallback to local node id.");
            const uint64_t nodeIndex = rankNodeCounters[node->rankIdx]++;
            nodeId = BuildNodeId(node->rankIdx, nodeIndex);
        }
        nodeIdMap[node] = nodeId;
        graphNodes.push_back(node);
    };

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

        if (currentNode == dummyStart && !includeStartNode) {
            continue;
        }

        registerNode(currentNode);

        if (currentNode->task == nullptr || currentNode->task->GetType() != TaskTypeStub::CCU_GRAPH) {
            continue;
        }
        TaskStubCcuGraph* ccuGraphTask = dynamic_cast<TaskStubCcuGraph*>(currentNode->task);
        if (ccuGraphTask == nullptr || ccuGraphTask->ccuHeadTaskNode == nullptr) {
            continue;
        }

        std::queue<TaskNode*> ccuQueue;
        std::set<TaskNode*> ccuVisited;
        ccuQueue.push(ccuGraphTask->ccuHeadTaskNode);
        ccuVisited.insert(ccuGraphTask->ccuHeadTaskNode);
        while (!ccuQueue.empty()) {
            TaskNode* ccuNode = ccuQueue.front();
            ccuQueue.pop();
            for (auto* child : ccuNode->children) {
                if (child != nullptr && ccuVisited.insert(child).second) {
                    ccuQueue.push(child);
                }
            }
            registerNode(ccuNode);
        }
    }
}

static Json TaskEdgeToJson(const std::string& srcNodeId, const std::string& dstNodeId)
{
    Json edgeJson = Json::object();
    edgeJson["src_id"] = srcNodeId;
    edgeJson["dst_id"] = dstNodeId;
    return edgeJson;
}

static void CollectCcuSubGraphNodes(TaskNode* headNode, std::vector<TaskNode*>& graphNodes)
{
    graphNodes.clear();
    if (headNode == nullptr) {
        return;
    }

    std::queue<TaskNode*> visitQueue;
    std::set<TaskNode*> visited;
    visitQueue.push(headNode);
    visited.insert(headNode);

    while (!visitQueue.empty()) {
        TaskNode* currentNode = visitQueue.front();
        visitQueue.pop();

        if (IsValidRegisteredNode(currentNode)) {
            graphNodes.push_back(currentNode);
        }

        for (auto* child : currentNode->children) {
            if (child != nullptr && visited.insert(child).second) {
                visitQueue.push(child);
            }
        }
    }
}

nlohmann::json TaskMetaToJson(const HcclTaskMetaData& taskMeta)
{
    Json taskMetaJson = Json::object();
    taskMetaJson["task_type"] = TaskMetaTypeToString(taskMeta.taskType);
    taskMetaJson["comm_id"] = taskMeta.commId;
    taskMetaJson["rank_id"] = taskMeta.rankId;
    taskMetaJson["stream_id"] = taskMeta.streamId;
    taskMetaJson["task_data"] = MakeTaskMetaFields(taskMeta);
    return taskMetaJson;
}

nlohmann::json DumpTaskStubToJson(TaskStub* task) { return DumpTaskStubToJsonWithNodeMap(task, nullptr); }

static Json DumpTaskStubToJsonWithNodeMap(TaskStub* task, const std::map<TaskNode*, std::string>* nodeIdMap)
{
    Json taskJson = Json::object();
    if (task == nullptr) {
        return taskJson;
    }

    if (!task->GetTaskId().empty()) {
        taskJson["task_id"] = task->GetTaskId();
    }
    taskJson["task_type"] = DumpTaskTypeToString(task->GetType());
    taskJson["task_data"] = MakeTaskStubFields(task, nodeIdMap);
    return taskJson;
}

static nlohmann::json DumpTaskNodeToJson(
    TaskNode* node, const std::string& nodeId, const std::map<TaskNode*, std::string>& nodeIdMap,
    RankId rankFilter = static_cast<RankId>(-1))
{
    Json nodeJson = Json::object();
    if (node == nullptr) {
        return nodeJson;
    }

    if (!nodeId.empty()) {
        nodeJson["node_id"] = nodeId;
    } else if (node->task != nullptr && !node->task->GetTaskId().empty()) {
        nodeJson["node_id"] = node->task->GetTaskId();
    }
    nodeJson["rank_id"] = node->rankIdx;
    nodeJson["queue_id"] = node->queIdx;
    nodeJson["pos"] = node->pos;
    nodeJson["global_step"] = node->globalStep;
    nodeJson["local_step"] = node->localStep;
    nodeJson["is_aiv_node"] = node->isAivNode;
    nodeJson["has_aiv_task"] = node->hasAivTask;
    nodeJson["has_ccu_task"] = node->hasCcuTask;
    nodeJson["rank_pos"] = node->rankPos;
    nodeJson["block_idx"] = node->blockIdx;
    nodeJson["pipe_idx"] = node->pipeIdx;
    nodeJson["pipe_pos"] = node->pipePos;
    nodeJson["pos_info"] = node->GenPosInfo();
    nodeJson["task"] = DumpTaskStubToJsonWithNodeMap(node->task, &nodeIdMap);
    nodeJson["parents"] = Json::array();
    nodeJson["children"] = Json::array();

    for (auto* parent : node->parents) {
        if (parent == nullptr || !IsValidRegisteredNode(parent)) {
            continue;
        }
        if (rankFilter != static_cast<RankId>(-1) && parent->rankIdx != rankFilter) {
            continue;
        }
        auto iter = nodeIdMap.find(parent);
        if (iter == nodeIdMap.end()) {
            HCCL_VM_ERROR("parent node id not found.");
            continue;
        }
        nodeJson["parents"].push_back(iter->second);
    }
    for (auto* child : node->children) {
        if (child == nullptr || !IsValidRegisteredNode(child)) {
            continue;
        }
        if (rankFilter != static_cast<RankId>(-1) && child->rankIdx != rankFilter) {
            continue;
        }
        auto iter = nodeIdMap.find(child);
        if (iter == nodeIdMap.end()) {
            HCCL_VM_ERROR("child node id not found.");
            continue;
        }
        nodeJson["children"].push_back(iter->second);
    }

    return nodeJson;
}

static Json DumpCcuSubGraph(TaskStubCcuGraph* ccuGraphTask, const std::map<TaskNode*, std::string>* nodeIdMap)
{
    Json ccuGraphJson = Json::array();
    if (ccuGraphTask == nullptr || ccuGraphTask->ccuHeadTaskNode == nullptr) {
        return ccuGraphJson;
    }

    std::vector<TaskNode*> graphNodes;
    std::map<TaskNode*, std::string> localNodeIdMap;
    const std::map<TaskNode*, std::string>* resolvedNodeIdMap = nodeIdMap;
    if (resolvedNodeIdMap == nullptr) {
        resolvedNodeIdMap = &localNodeIdMap;
    }
    CollectTaskNodesWithGlobalIds(ccuGraphTask->ccuHeadTaskNode, graphNodes, localNodeIdMap, true);

    std::vector<TaskNode*> rankGraphNodes;
    rankGraphNodes.reserve(graphNodes.size());
    for (auto* node : graphNodes) {
        if (node != nullptr && node->rankIdx == ccuGraphTask->rankId) {
            rankGraphNodes.push_back(node);
        }
    }

    uint32_t maxQueueId = 0;
    for (auto* node : rankGraphNodes) {
        if (node == nullptr) {
            continue;
        }
        if (node->queIdx == static_cast<u32>(-1)) {
            continue;
        }
        maxQueueId = std::max(maxQueueId, node->queIdx);
    }

    const uint32_t queueCount = std::max<uint32_t>(ccuGraphTask->queueNum_, maxQueueId + 1);
    std::vector<std::vector<TaskNode*>> queueBuckets(queueCount);
    for (auto* node : rankGraphNodes) {
        if (node == nullptr) {
            continue;
        }
        if (node->queIdx == static_cast<u32>(-1)) {
            continue;
        }
        if (node->queIdx >= queueBuckets.size()) {
            HCCL_VM_WARN("queue id[{}] is out of range[{}], skip node.", node->queIdx, queueBuckets.size());
            continue;
        }
        queueBuckets[node->queIdx].push_back(node);
    }

    for (uint32_t queueIdx = 0; queueIdx < queueBuckets.size(); ++queueIdx) {
        Json queueJson = Json::array();
        for (auto* node : queueBuckets[queueIdx]) {
            auto nodeIdIter = resolvedNodeIdMap->find(node);
            if (nodeIdIter == resolvedNodeIdMap->end()) {
                HCCL_VM_ERROR("node id not found.");
                continue;
            }
            queueJson.push_back(DumpTaskNodeToJson(node, nodeIdIter->second, *resolvedNodeIdMap, ccuGraphTask->rankId));
        }
        ccuGraphJson.push_back(queueJson);
    }

    return ccuGraphJson;
}

static std::map<RankId, Json> BuildAllRankTaskQueuesJsonMap(const AllRankTaskQueues& allRankTaskQueues)
{
    std::map<RankId, Json> rankJsonMap;

    for (const auto& rankIter : allRankTaskQueues) {
        const RankId rankId = rankIter.first;
        Json rankJson = Json::object();
        rankJson["rank_id"] = rankId;
        rankJson["streams"] = Json::array();

        const SingleTaskQueue& taskQueue = rankIter.second;
        for (std::size_t queueIdx = 0; queueIdx < taskQueue.size(); ++queueIdx) {
            Json streamJson = Json::object();
            streamJson["queue_id"] = queueIdx;
            streamJson["tasks"] = Json::array();

            for (std::size_t taskIdx = 0; taskIdx < taskQueue[queueIdx].size(); ++taskIdx) {
                Json taskJson = DumpTaskStubToJson(taskQueue[queueIdx][taskIdx].get());
                taskJson["task_idx"] = taskIdx;
                streamJson["tasks"].push_back(taskJson);
            }

            rankJson["streams"].push_back(streamJson);
        }

        rankJsonMap[rankId] = rankJson;
    }

    return rankJsonMap;
}

static std::map<RankId, Json> BuildTaskGraphJsonMap(TaskNode* dummyStart)
{
    std::map<RankId, Json> rankJsonMap;
    std::map<RankId, std::set<std::pair<std::string, std::string>>> rankEdgeSets;
    if (dummyStart == nullptr) {
        return rankJsonMap;
    }

    // Build one global node-id registry (includes CCU subgraph nodes) for cross-dump consistency.
    std::vector<TaskNode*> allNodes;
    std::map<TaskNode*, std::string> nodeIdMap;
    CollectTaskNodesWithGlobalIds(dummyStart, allNodes, nodeIdMap);

    // Outer DAG should only contain main-graph nodes. CCU subgraph nodes are embedded under CCU task details.
    std::vector<TaskNode*> graphNodes;
    CollectMainGraphNodes(dummyStart, graphNodes);
    std::set<TaskNode*> mainNodeSet(graphNodes.begin(), graphNodes.end());

    for (auto* currentNode : graphNodes) {
        auto currentNodeIdIter = nodeIdMap.find(currentNode);
        if (currentNodeIdIter == nodeIdMap.end()) {
            HCCL_VM_ERROR("current node id not found.");
            continue;
        }

        Json& rankJson = rankJsonMap[currentNode->rankIdx];
        if (rankJson.is_null()) {
            rankJson = MakeGraphRankJson();
            rankJson["rank_id"] = currentNode->rankIdx;
        }
        rankJson["nodes"].push_back(DumpTaskNodeToJson(currentNode, currentNodeIdIter->second, nodeIdMap));

        for (auto* parent : currentNode->parents) {
            if (parent == nullptr) {
                continue;
            }
            if (mainNodeSet.count(parent) == 0) {
                continue;
            }
            auto parentIdIter = nodeIdMap.find(parent);
            if (parentIdIter == nodeIdMap.end()) {
                HCCL_VM_ERROR("parent node id not found.");
                continue;
            }

            const auto edgeKey = std::make_pair(parentIdIter->second, currentNodeIdIter->second);
            if (rankEdgeSets[currentNode->rankIdx].insert(edgeKey).second) {
                rankJson["edges"].push_back(TaskEdgeToJson(parentIdIter->second, currentNodeIdIter->second));
            }
        }

        for (auto* child : currentNode->children) {
            if (child == nullptr) {
                continue;
            }
            if (mainNodeSet.count(child) == 0) {
                continue;
            }
            auto childIdIter = nodeIdMap.find(child);
            if (childIdIter == nodeIdMap.end()) {
                HCCL_VM_ERROR("child node id not found.");
                continue;
            }

            const auto edgeKey = std::make_pair(currentNodeIdIter->second, childIdIter->second);
            if (rankEdgeSets[currentNode->rankIdx].insert(edgeKey).second) {
                rankJson["edges"].push_back(TaskEdgeToJson(currentNodeIdIter->second, childIdIter->second));
            }
        }
    }

    return rankJsonMap;
}

static HcclResult
DumpRankJsonMapByStage(const std::map<RankId, Json>& rankJsonMap, const std::string& type, const std::string& stage)
{
    if (stage.empty()) {
        HCCL_VM_WARN("stage is empty.");
        return HcclResult::HCCL_E_PARA;
    }

    DumpManager& dumpManager = DumpManager::GetInstance();
    if (!dumpManager.IsEnabled()) {
        return HcclResult::HCCL_SUCCESS;
    }

    const std::string stageDirPath = StringFormat("%s/%s", DUMP_GRAPH_ROOT_DIR.c_str(), stage.c_str());
    const std::string fullStageDirPath
        = StringFormat("%s/%s", dumpManager.GetDumpRootDir().c_str(), stageDirPath.c_str());
    auto removeDirRecursivelyBestEffort = [](const std::string& path, auto&& self) -> void {
        DIR* dir = opendir(path.c_str());
        if (dir == nullptr) {
            return;
        }

        dirent* entry = nullptr;
        while ((entry = readdir(dir)) != nullptr) {
            const std::string name = entry->d_name;
            if (name == "." || name == "..") {
                continue;
            }
            const std::string childPath = StringFormat("%s/%s", path.c_str(), name.c_str());
            struct stat st;
            if (lstat(childPath.c_str(), &st) != 0) {
                continue;
            }

            if (S_ISDIR(st.st_mode)) {
                self(childPath, self);
                continue;
            }

            if (unlink(childPath.c_str()) != 0) {
                HCCL_VM_WARN("failed to remove file[{}], errno[{}].", childPath.c_str(), errno);
            }
        }

        closedir(dir);
        if (rmdir(path.c_str()) != 0 && errno != ENOENT) {
            HCCL_VM_WARN("failed to remove dir[{}], errno[{}].", path.c_str(), errno);
        }
    };
    removeDirRecursivelyBestEffort(fullStageDirPath, removeDirRecursivelyBestEffort);

    Json stageIndexJson = Json::object();
    stageIndexJson["stage"] = stage;
    stageIndexJson["rank_count"] = rankJsonMap.size();
    stageIndexJson["ranks"] = Json::array();
    if (type != TASK_GRAPH_DUMP_TYPE) {
        stageIndexJson["type"] = type;
    }

    for (const auto& rankJsonPair : rankJsonMap) {
        Json rankDumpJson = rankJsonPair.second;
        rankDumpJson["stage"] = stage;
        rankDumpJson["rank_id"] = rankJsonPair.first;
        if (type != TASK_GRAPH_DUMP_TYPE) {
            rankDumpJson["type"] = type;
        }
        const std::string rankDumpPath = BuildStageRankDumpPath(stage, rankJsonPair.first);
        const HcclResult writeRet = dumpManager.Write(rankDumpPath, rankDumpJson);
        if (writeRet != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_WARN(
                "failed to dump rank[{}], stage[{}], ret[{}].", rankJsonPair.first, stage.c_str(),
                static_cast<u32>(writeRet));
            return writeRet;
        }
        stageIndexJson["ranks"].push_back(rankJsonPair.first);
    }

    const std::string stageIndexPath = StringFormat("%s/%s/index.msgpack", DUMP_GRAPH_ROOT_DIR.c_str(), stage.c_str());
    const HcclResult stageWriteRet = dumpManager.Write(stageIndexPath, stageIndexJson);
    if (stageWriteRet != HcclResult::HCCL_SUCCESS) {
        return stageWriteRet;
    }

    if (type == TASK_GRAPH_DUMP_TYPE) {
        Json stageStats = Json::object();
        stageStats["rank_count"] = rankJsonMap.size();
        size_t totalNodeCount = 0;
        size_t totalEdgeCount = 0;
        stageStats["ranks"] = Json::array();
        for (const auto& rankJsonPair : rankJsonMap) {
            const Json& rankJson = rankJsonPair.second;
            const size_t nodeCount
                = rankJson.contains("nodes") && rankJson["nodes"].is_array() ? rankJson["nodes"].size() : 0;
            const size_t edgeCount
                = rankJson.contains("edges") && rankJson["edges"].is_array() ? rankJson["edges"].size() : 0;
            totalNodeCount += nodeCount;
            totalEdgeCount += edgeCount;
            Json rankStats = Json::object();
            rankStats["rank_id"] = rankJsonPair.first;
            rankStats["node_count"] = nodeCount;
            rankStats["edge_count"] = edgeCount;
            stageStats["ranks"].push_back(rankStats);
        }
        stageStats["node_count"] = totalNodeCount;
        stageStats["edge_count"] = totalEdgeCount;
        DumpRunManifest::GetInstance().SetGraphStageStats(stage, stageStats);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult DumpInputTaskQueues(const AllRankTaskQueues& allRankTaskQueues)
{
    if (!DumpManager::GetInstance().IsEnabled()) {
        return HcclResult::HCCL_SUCCESS;
    }
    return DumpRankJsonMapByStage(
        BuildAllRankTaskQueuesJsonMap(allRankTaskQueues), INPUT_QUEUE_DUMP_TYPE, INPUT_TASK_QUEUE_STAGE);
}

HcclResult DumpTaskGraphByStage(TaskNode* dummyStart, const std::string& stage)
{
    if (!DumpManager::GetInstance().IsEnabled()) {
        return HcclResult::HCCL_SUCCESS;
    }
    return DumpRankJsonMapByStage(BuildTaskGraphJsonMap(dummyStart), TASK_GRAPH_DUMP_TYPE, stage);
}
} // namespace HcclSim
