/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "checker.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ccu_task_common.h"
#include "check_rank_mem.h"
#include "check_utils.h"
#include "checker_def.h"
#include "dag_graphviz_dump.h"
#include "dump/dump_graph.h"
#include "dump/validation_issue_recorder.h"
#include "dump_v3/dump_v3_dag.h"
#include "dump_v3/dump_v3_manager.h"
#include "framework/task_graph_generator_v3/task_graph_generator_v3.h"
#include "framework/task_graph_generator_v3/task_graph_mem_conflict_v3.h"
#include "framework/task_graph_generator_v3/task_graph_sync_conflict_v3.h"
#include "framework/task_graph_generator_v3/task_graph_semantic_check_v3.h"
#include "framework/task_graph_generator_v3/task_graph_single_task_check_v3.h"
#include "framework/task_graph_generator_v3/task_meta_translator_v3.h"
#include "sim_log.h"
#include "mem_conflict_check_utils.h"
#include "sim_task.h"
#include "singletask_check.h"
#include "storage_manager.h"
#include "task_ccu.h"
#include "task_check_op_semantics.h"
#include "task_def.h"
#include "task_graph_generator.h"
#include "task_graph_revamp.h"
#include "task_graph_revamp_bilateral_ccu.h"
#include "task_graph_revamp_parallel.h"
#include "task_utils.h"

using namespace std;

namespace HcclSim {
namespace {
    using V3TaskNode = TaskGraphGeneratorV3::TaskNode;
    using V3StageClock = std::chrono::steady_clock;
    constexpr uint64_t V3_NS_PER_MS = 1000000ULL;

    uint64_t ElapsedV3Ns(V3StageClock::time_point start, V3StageClock::time_point end)
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    constexpr bool ENABLE_V3_MEM_CONFLICT_CHECK = true;
    constexpr bool ENABLE_V3_SEMANTIC_CHECK = true;

    class V3StageTimer {
    public:
        explicit V3StageTimer(const char* stage) : stage_(stage), start_(V3StageClock::now()) {}

        ~V3StageTimer()
        {
            const uint64_t elapsedNs = ElapsedV3Ns(start_, V3StageClock::now());
            if (extraInfo_.empty()) {
                HCCL_VM_INFO(
                    "CheckerV3 stage finished, stage={}, status={}, costMs={}", stage_, status_,
                    elapsedNs / V3_NS_PER_MS);
                return;
            }
            HCCL_VM_INFO(
                "CheckerV3 stage finished, stage={}, status={}, costMs={}, {}", stage_, status_,
                elapsedNs / V3_NS_PER_MS, extraInfo_);
        }

        void SetStatus(const char* status) { status_ = status; }

        void SetExtraInfo(const std::string& extraInfo) { extraInfo_ = extraInfo; }

    private:
        const char* stage_{nullptr};
        const V3StageClock::time_point start_;
        const char* status_{"failed"};
        std::string extraInfo_;
    };

    bool IsV3Boundary(const V3TaskNode* node, TaskGraphGeneratorV3::BoundaryType boundaryType)
    {
        if (node == nullptr) {
            return false;
        }
        if (node->GetType() == TaskGraphGeneratorV3::TaskType::START) {
            auto* start = dynamic_cast<const TaskGraphGeneratorV3::TaskStart*>(node);
            return start != nullptr && start->GetBoundaryType() == boundaryType;
        }
        if (node->GetType() == TaskGraphGeneratorV3::TaskType::END) {
            auto* end = dynamic_cast<const TaskGraphGeneratorV3::TaskEnd*>(node);
            return end != nullptr && end->GetBoundaryType() == boundaryType;
        }
        return false;
    }

    bool IsAivTaskType(TaskGraphGeneratorV3::TaskType type)
    {
        switch (type) {
            case TaskGraphGeneratorV3::TaskType::AIV_GRAPH:
            case TaskGraphGeneratorV3::TaskType::AIV_SET_FLAG:
            case TaskGraphGeneratorV3::TaskType::AIV_WAIT_FLAG:
            case TaskGraphGeneratorV3::TaskType::AIV_PIPE_BARRIER:
            case TaskGraphGeneratorV3::TaskType::AIV_SYNC_ALL:
            case TaskGraphGeneratorV3::TaskType::AIV_SEND_FLAG:
            case TaskGraphGeneratorV3::TaskType::AIV_RECV_FLAG:
                return true;
            default:
                return false;
        }
    }

    bool IsAivDataTaskType(TaskGraphGeneratorV3::TaskType type)
    {
        switch (type) {
            case TaskGraphGeneratorV3::TaskType::TRANS_MEM:
            case TaskGraphGeneratorV3::TaskType::BATCH_TRANS_MEM:
            case TaskGraphGeneratorV3::TaskType::REDUCE:
            case TaskGraphGeneratorV3::TaskType::BATCH_REDUCE:
                return true;
            default:
                return false;
        }
    }

    bool IsAivSyntheticDataNode(const V3TaskNode* node)
    {
        if (node == nullptr || !IsAivDataTaskType(node->GetType()) || node->HasCcuTrace()) {
            return false;
        }
        const auto& loc = node->GetPosition();
        return loc.streamId == TaskGraphGeneratorV3::INVALID_STREAM_ID
               && loc.queueId != TaskGraphGeneratorV3::INVALID_QUEUE_ID;
    }

    bool IsAivDagLogNode(const V3TaskNode* node)
    {
        if (node == nullptr) {
            return false;
        }
        return IsAivTaskType(node->GetType()) || IsAivSyntheticDataNode(node)
               || IsV3Boundary(node, TaskGraphGeneratorV3::BoundaryType::AIV_SUB_GRAPH);
    }

    std::unique_ptr<TaskGraphGeneratorV3::TaskGraphGeneratorV3> GenGraphV3FromTaskMeta()
    {
        V3StageTimer timer("GenGraph");
        StorageManager& storage = StorageManager::GetInstance();
        TaskGraphGeneratorV3::TaskMetaTranslatorV3 taskMetaTranslatorV3;
        // V3 主流程入口：
        // 1. 先把 task meta 翻译成中间节点；
        // 2. 做一次从流合法性检查；
        // 3. 再生成 V3 图并在日志里记录成图/CCU 展开耗时。
        HcclResult ret = taskMetaTranslatorV3.Translate(storage, TaskGraphGeneratorV3::INVALID_OPERATOR_ID);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_WARN(
                "Failed to translate one task into the V3 internal format, V3 graph generation is "
                "stopped, ret={}",
                static_cast<uint32_t>(ret));
            return nullptr;
        }

        auto translatedNodes = taskMetaTranslatorV3.TakeNodes();
        auto translatedTaskQueues = taskMetaTranslatorV3.TakeTaskQueues();
        ret = TaskGraphGeneratorV3::CheckSlaveTaskQueue(translatedNodes, translatedTaskQueues);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_WARN(
                "Slave-stream check failed, V3 graph generation is stopped, ret={}", static_cast<uint32_t>(ret));
            return nullptr;
        }
        std::unique_ptr<TaskGraphGeneratorV3::TaskGraphGeneratorV3> graphGeneratorV3(
            new (std::nothrow) TaskGraphGeneratorV3::TaskGraphGeneratorV3);
        if (graphGeneratorV3 == nullptr) {
            HCCL_VM_WARN("Failed to create the V3 graph generator object");
            return nullptr;
        }

        ret = graphGeneratorV3->GenGraph(std::move(translatedNodes), std::move(translatedTaskQueues));
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_WARN(
                "Failed to build the graph from translated tasks, ret={}, rankCount={}, nodeCount={}",
                static_cast<uint32_t>(ret), graphGeneratorV3->GetTaskQueues().size(),
                graphGeneratorV3->GetNodes().size());
            return nullptr;
        }
        const auto& ccuExpandStats = graphGeneratorV3->GetCcuExpandStats();
        const auto& aivExpandStats = graphGeneratorV3->GetAivExpandStats();
        std::ostringstream timerExtraInfo;
        timerExtraInfo << "taskNodeCount=" << graphGeneratorV3->GetNodes().size()
                       << ", mainStartNodeId=" << graphGeneratorV3->GetMainStartNodeId()
                       << ", aivExpandTotalMs=" << (aivExpandStats.totalExpandNs / V3_NS_PER_MS)
                       << ", aivGraphCount=" << aivExpandStats.graphCount
                       << ", ccuExpandTotalMs=" << (ccuExpandStats.totalExpandNs / V3_NS_PER_MS)
                       << ", ccuGraphCount=" << ccuExpandStats.graphCount;
        timer.SetExtraInfo(timerExtraInfo.str());
        timer.SetStatus("success");
        return graphGeneratorV3;
    }

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

    const char* TaskTypeKeyName(TaskGraphGeneratorV3::TaskType type)
    {
        switch (type) {
            case TaskGraphGeneratorV3::TaskType::TRANS_MEM:
                return "TRANS_MEM";
            case TaskGraphGeneratorV3::TaskType::BATCH_TRANS_MEM:
                return "BATCH_TRANS_MEM";
            case TaskGraphGeneratorV3::TaskType::REDUCE:
                return "REDUCE";
            case TaskGraphGeneratorV3::TaskType::BATCH_REDUCE:
                return "BATCH_REDUCE";
            case TaskGraphGeneratorV3::TaskType::RECORD:
                return "RECORD";
            case TaskGraphGeneratorV3::TaskType::WAIT:
                return "WAIT";
            case TaskGraphGeneratorV3::TaskType::CCU_GRAPH:
                return "CCU_GRAPH";
            case TaskGraphGeneratorV3::TaskType::AIV_GRAPH:
                return "AIV_GRAPH";
            case TaskGraphGeneratorV3::TaskType::AIV_SET_FLAG:
                return "AIV_SET_FLAG";
            case TaskGraphGeneratorV3::TaskType::AIV_WAIT_FLAG:
                return "AIV_WAIT_FLAG";
            case TaskGraphGeneratorV3::TaskType::AIV_PIPE_BARRIER:
                return "AIV_PIPE_BARRIER";
            case TaskGraphGeneratorV3::TaskType::AIV_SYNC_ALL:
                return "AIV_SYNC_ALL";
            case TaskGraphGeneratorV3::TaskType::AIV_SEND_FLAG:
                return "AIV_SEND_FLAG";
            case TaskGraphGeneratorV3::TaskType::AIV_RECV_FLAG:
                return "AIV_RECV_FLAG";
            case TaskGraphGeneratorV3::TaskType::START:
                return "START";
            case TaskGraphGeneratorV3::TaskType::END:
                return "END";
            default:
                return "INVALID";
        }
    }

    std::string FormatV3NodeIdList(const std::vector<V3TaskNode*>& nodes)
    {
        std::ostringstream os;
        os << "[";
        bool firstNode = true;
        for (const V3TaskNode* node : nodes) {
            if (node == nullptr) {
                continue;
            }
            if (!firstNode) {
                os << ",";
            }
            firstNode = false;
            os << node->GetNodeId();
        }
        os << "]";
        return os.str();
    }

    std::string MakeAivDagTaskLogLine(const V3TaskNode* node, size_t bfsIndex)
    {
        if (node == nullptr) {
            return "node=null";
        }
        const auto& loc = node->GetPosition();
        std::ostringstream os;
        os << "bfsIndex=" << bfsIndex << ", nodeId=" << node->GetNodeId()
           << ", taskType=" << TaskTypeKeyName(node->GetType()) << ", rankId=" << loc.rankId
           << ", streamId=" << loc.streamId << ", queueId=" << loc.queueId
           << ", parentNodeIds=" << FormatV3NodeIdList(node->GetParents())
           << ", childNodeIds=" << FormatV3NodeIdList(node->GetChildren()) << ", detail=" << node->Describe();
        return os.str();
    }

    size_t LogAivExpandedDagTasks(const std::vector<const V3TaskNode*>& v3BfsNodes)
    {
        size_t aivTaskCount = 0;
        for (size_t index = 0; index < v3BfsNodes.size(); ++index) {
            const V3TaskNode* node = v3BfsNodes[index];
            if (!IsAivDagLogNode(node)) {
                continue;
            }
            HCCL_VM_DEBUG("Expanded AIV DAG node detail, {}", MakeAivDagTaskLogLine(node, index));
            ++aivTaskCount;
        }
        HCCL_VM_DEBUG("Finished dumping expanded AIV DAG node details, taskCount={}", aivTaskCount);
        return aivTaskCount;
    }

    void LogGraphV3Dag(const V3TaskNode* start)
    {
        if (start == nullptr) {
            HCCL_VM_WARN("Skip DAG dump because the graph start node is null");
            return;
        }

        const auto v3BfsNodes = BfsGraphV3(start);
        std::string dumpPath;
        const HcclResult dumpRet = DumpDagGraphvizDot(start, &dumpPath);
        if (dumpRet != HCCL_SUCCESS) {
            HCCL_VM_WARN(
                "[TaskGraphGeneratorV3][GraphvizDot][v3] Failed to dump DAG dot file, ret={}",
                static_cast<uint32_t>(dumpRet));
        } else {
            HCCL_VM_INFO(
                "[TaskGraphGeneratorV3][GraphvizDot][v3] Dump V3 DAG dot file, path={}, nodeCount={}", dumpPath,
                v3BfsNodes.size());
        }
        LogAivExpandedDagTasks(v3BfsNodes);
    }

    void LogGraphV3Sqe(const TaskGraphGeneratorV3::TaskGraphGeneratorV3& graph)
    {
        const auto& translatedTaskQueues = graph.GetTaskQueues();
        const auto& translatedNodes = graph.GetNodes();

        u32 rankIdx = 0;
        uint32_t rankNum = 0;
        for (const auto& iter : translatedTaskQueues) {
            rankIdx = iter.first;
            ++rankNum;
            HCCL_VM_INFO("=======================================================");
            HCCL_VM_INFO("rankId is : {:d}", rankIdx);
            const TaskGraphGeneratorV3::RankNodeQueues& taskQueue = iter.second;
            for (size_t i = 0; i < taskQueue.size(); ++i) {
                if (taskQueue[i].empty()) {
                    continue;
                }

                u32 taskIdx = 0;
                HCCL_VM_INFO(
                    "streamIdx : {:d}, taskNum : {:d}", static_cast<uint32_t>(i),
                    static_cast<uint32_t>(taskQueue[i].size()));
                HCCL_VM_INFO("-------------------------------------------------------");
                for (const auto& nodeId : taskQueue[i]) {
                    if (nodeId < 0 || static_cast<size_t>(nodeId) >= translatedNodes.size()
                        || translatedNodes[static_cast<size_t>(nodeId)] == nullptr) {
                        HCCL_VM_WARN("rankIdx:{:d}, taskIdx:{:d}, invalid nodeId:{:d}", rankIdx, taskIdx, nodeId);
                        ++taskIdx;
                        continue;
                    }

                    const std::unique_ptr<TaskGraphGeneratorV3::TaskNode>& task
                        = translatedNodes[static_cast<size_t>(nodeId)];
                    const std::string tempStr = task->Describe();
                    HCCL_VM_INFO("rankIdx:{:d}, taskIdx:{:d}, {:s}", rankIdx, taskIdx, tempStr);
                    ++taskIdx;
                }
            }
        }
        HCCL_VM_INFO("Finished logging V3 SQE task queues, rankNum={:d}", rankNum);
    }

    HcclResult CheckGraphV3TaskMem(const V3TaskNode* start)
    {
        V3StageTimer timer("SingleTaskCheck");
        if (start == nullptr) {
            HCCL_VM_WARN("Skip single-task memory check because the graph start node is null");
            timer.SetStatus("skipped");
            return HCCL_SUCCESS;
        }

        // 单 task 校验只看每个节点自身的 src/dst 描述是否合法，不依赖跨节点语义。
        TaskGraphGeneratorV3::SingleTaskCheckStats stats;
        HcclResult ret = TaskGraphGeneratorV3::CheckTaskMem(start, &stats);
        std::ostringstream extraInfo;
        extraInfo << "nodeCount=" << stats.nodeCount << ", dataTaskNodeCount=" << stats.dataTaskNodeCount;
        if (TaskGraphGeneratorV3::g_checkerAivUbBufferSize != 0
            || TaskGraphGeneratorV3::g_checkerAivCommInfoSize != 0) {
            extraInfo << ", aivUbBufferSize=" << TaskGraphGeneratorV3::g_checkerAivUbBufferSize
                      << ", aivCommInfoSize=" << TaskGraphGeneratorV3::g_checkerAivCommInfoSize;
        }
        timer.SetExtraInfo(extraInfo.str());
        timer.SetStatus(ret == HCCL_SUCCESS ? "success" : "failed");
        return ret;
    }

    HcclResult CheckGraphV3MemConflict(const V3TaskNode* start)
    {
        V3StageTimer timer("MemConflict");
        if (start == nullptr) {
            HCCL_VM_WARN("Skip memory-conflict check because the graph start node is null");
            timer.SetStatus("skipped");
            return HCCL_SUCCESS;
        }

        // 内存冲突校验会先收集所有数据搬运节点的访问区间，再按地址桶扫描重叠候选。
        TaskGraphGeneratorV3::MemConflictCheckStats stats;
        HcclResult ret = TaskGraphGeneratorV3::CheckDataMoveTaskMemConflict(start, &stats);
        std::ostringstream extraInfo;
        extraInfo << "nodeCount=" << stats.nodeCount << ", dataTaskNodeCount=" << stats.dataTaskNodeCount
                  << ", processedDataTaskNodeCount=" << stats.processedDataTaskNodeCount
                  << ", accessIntervalCount=" << stats.accessIntervalCount
                  << ", memoryBucketCount=" << stats.memoryBucketCount
                  << ", overlapCandidatePairs=" << stats.overlapCandidatePairCount
                  << ", orderedCandidatePairs=" << stats.orderedCandidatePairCount
                  << ", parallelCandidatePairs=" << stats.parallelCandidatePairCount;
        timer.SetExtraInfo(extraInfo.str());
        timer.SetStatus(ret == HCCL_SUCCESS ? "success" : "failed");
        return ret;
    }

    HcclResult CheckGraphV3SyncConflict(const V3TaskNode* start)
    {
        V3StageTimer timer("SyncConflict");
        if (start == nullptr) {
            HCCL_VM_WARN("Skip sync-conflict check because the graph start node is null");
            timer.SetStatus("skipped");
            return HCCL_SUCCESS;
        }

        TaskGraphGeneratorV3::SyncConflictCheckStats stats;
        HcclResult ret = TaskGraphGeneratorV3::CheckSyncResourceConflict(start, &stats);
        std::ostringstream extraInfo;
        extraInfo << "originalNodeCount=" << stats.originalNodeCount << ", copiedNodeCount=" << stats.copiedNodeCount
                  << ", copiedEdgeCount=" << stats.copiedEdgeCount
                  << ", resourceBucketCount=" << stats.resourceBucketCount << ", pairCount=" << stats.pairCount
                  << ", checkedBucketCount=" << stats.checkedBucketCount << ", conflictCount=" << stats.conflictCount;
        timer.SetExtraInfo(extraInfo.str());
        timer.SetStatus(ret == HCCL_SUCCESS ? "success" : "failed");
        return ret;
    }

    HcclResult CheckGraphV3Semantic(const V3TaskNode* start)
    {
        V3StageTimer timer("SemanticCheck");
        if (start == nullptr) {
            HCCL_VM_WARN("Skip semantic check because the graph start node is null");
            timer.SetStatus("skipped");
            return HCCL_SUCCESS;
        }

        // 语义校验按 DAG 依赖逐节点模拟内存来源，最后再校验各 rank 输出布局。
        TaskGraphGeneratorV3::SemanticCheckStats stats;
        HcclResult ret = TaskGraphGeneratorV3::CheckSemantics(start, &stats);
        std::ostringstream extraInfo;
        extraInfo << "handledNodeCount=" << stats.handledNodeCount << ", rankCount=" << stats.rankCount
                  << ", normalSemanticCount=" << stats.normalSemanticCount
                  << ", normalSemanticBytes=" << stats.normalSemanticBytes << ", msBucketCount=" << stats.msBucketCount;
        timer.SetExtraInfo(extraInfo.str());
        timer.SetStatus(ret == HCCL_SUCCESS ? "success" : "failed");
        return ret;
    }
} // namespace

Checker::~Checker()
{
    for (auto& ele : toDeleteCopyTaskNodeResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }

    for (auto& ele : toDeleteCopyTaskResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }
}

void Checker::CloseRankMemCheck() { closeRankMemCheck_ = true; }

HcclResult GenAndCheckGraphV3()
{
    auto graphGeneratorV3 = GenGraphV3FromTaskMeta();
    if (graphGeneratorV3 == nullptr) {
        return HCCL_E_PTR;
    }
    const HcclResult dumpRet = DumpV3Dag(*graphGeneratorV3);
    if (dumpRet != HCCL_SUCCESS) {
        HCCL_VM_WARN("Failed to dump the V3 DAG, ret={}", static_cast<uint32_t>(dumpRet));
    }
    const V3TaskNode* dummyStartV3 = graphGeneratorV3->GetMainStartNode();
    // 打印sqe
    LogGraphV3Sqe(*graphGeneratorV3);
    // 打印dag task任务
    LogGraphV3Dag(dummyStartV3);

    // V3 检查顺序与日志阶段保持一致：
    // 成图 -> 单 task 内存校验 -> 同步资源校验 -> 内存冲突校验 -> 语义校验。
    CHK_RET(CheckGraphV3TaskMem(dummyStartV3));
    CHK_RET(CheckGraphV3SyncConflict(dummyStartV3));
    if (ENABLE_V3_MEM_CONFLICT_CHECK) {
        CHK_RET(CheckGraphV3MemConflict(dummyStartV3));
    }
    if (ENABLE_V3_SEMANTIC_CHECK) {
        CHK_RET(CheckGraphV3Semantic(dummyStartV3));
    }
    return HCCL_SUCCESS;
}

void PrintSQEGraph(TaskNodePtr dummyStart);
HcclResult Checker::GenAndCheckGraph(AllRankTaskQueues& allRankTaskQueues, TaskCheckOpSemantics& opSemanticsChecker)
{
    ValidationIssueRecorder::GetInstance().Reset();

    u32 rankIdx = 0;
    uint32_t rankNum = 0;
    for (auto& iter : allRankTaskQueues) {
        rankIdx = iter.first;
        rankNum++;
        HCCL_VM_INFO("=======================================================");
        HCCL_VM_INFO("rankId is : {:d}", rankIdx);
        const SingleTaskQueue& taskQueue = iter.second;
        for (int i = 0; i < taskQueue.size(); i++) {
            if (taskQueue[i].size() == 0) {
                continue;
            }

            u32 taskIdx = 0;
            HCCL_VM_INFO("streamIdx : {:d}, taskNum : {:d}", i, taskQueue[i].size());
            HCCL_VM_INFO("-------------------------------------------------------");
            for (auto& task : taskQueue[i]) {
                std::string tempStr = task->Describe();
                HCCL_VM_INFO(
                    "rankIdx:{:d}, taskIdx:{:d}, {:s}, pointer: {:p}", rankIdx, taskIdx, tempStr,
                    static_cast<void*>(task.get()));
                taskIdx++;
            }
        }
    }
    {
        const HcclResult dumpRet = DumpInputTaskQueues(allRankTaskQueues);
        if (dumpRet != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_WARN("Failed to dump the old checker input task queues, ret={}", static_cast<u32>(dumpRet));
        }
    }

    // 1. 检查从流
    HCCL_VM_INFO("1. 检查从流");
    SingleTaskCheck taskChecker;
    CHK_RET(taskChecker.CheckSlaveTaskQueue(allRankTaskQueues));

    // 2. 成图
    HCCL_VM_INFO("2. 成图");
    TaskNode dummyStart = TaskNode(nullptr, -1, 0, 0);
    TaskNode dummyStartCopy = TaskNode(nullptr, -1, 0, 0);
    TaskGraphGenerator graphGenerator;
    CHK_RET(graphGenerator.GenGraph(allRankTaskQueues, &dummyStart));

    // 3. Task内存校验
    // 是否可以复用 taskChecker
    HCCL_VM_INFO("3. Task内存校验");
    CHK_RET(taskChecker.CheckTaskMem(&dummyStart));
    HCCL_VM_INFO("3. Task内存校验 END: {}", closeRankMemCheck_);

    if (!closeRankMemCheck_) {
        // 4. 图复制
        HCCL_VM_INFO("4. 图复制");
        if (dummyStart.hasCcuTask) {
            CHK_RET(CopyCcuTaskGraph(&dummyStart, &dummyStartCopy, rankNum));
            dummyStartCopy.hasCcuTask = true;
        } else {
            CopyTaskGraph(&dummyStart, &dummyStartCopy);
        }

        // 5. 图改造
        HCCL_VM_INFO("5. 图改造");
        GraphRevampBilateralSemantics graphRevamp;

        // ccu图改造
        GraphRevampParallel parallelRevamp;
        GraphRevampBilateralCcu ccuBilateralRevamp;
        if (dummyStartCopy.hasCcuTask) {
            HcclResult ret = HcclResult::HCCL_SUCCESS;
            // 并行化改造：异步节点、Loop指令块
            ret = parallelRevamp.Revamp(&dummyStartCopy);
            if (ret != HcclResult::HCCL_SUCCESS) {
                return ret;
            }

            // 单边->双边语义: CCU子图
            ret = ccuBilateralRevamp.Revamp(&dummyStartCopy);
            if (ret != HcclResult::HCCL_SUCCESS) {
                return ret;
            }
        }
        // 主图改造
        CHK_RET(graphRevamp.Revamp(&dummyStartCopy));

        // 6. Rank内存校验
        HCCL_VM_INFO("6. Rank内存校验");
        CheckRankMem checkRankmem(&dummyStartCopy);
        CHK_RET(checkRankmem.Execute());
    }

    // 7. 语义校验
    HCCL_VM_INFO("7. 语义校验");
    // PrintCcuGraph(&dummyStart);
    opSemanticsChecker.SetGraphHead(&dummyStart);
    CHK_RET(opSemanticsChecker.Execute());
    HCCL_VM_INFO("8. 语义校验 END");

    // 成图及校验成功
    return HCCL_SUCCESS;
}

void Checker::CopyTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode)
{
    // 该函数无修改
    // 遍历两遍，先将所有节点拷贝出来，再建立父子关系
    std::map<TaskNodePtr, TaskNodePtr> originNode2copyNode; // 用来收录原节点到新节点的映射
    std::vector<TaskNodePtr> candTaskNodePtr;
    std::set<TaskNodePtr> isVisited;

    originNode2copyNode[originNode] = copyNode;
    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
    }

    while (!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());

        TaskNodePtr newNodePtr = new TaskNode(curNode->task, curNode->rankIdx, curNode->queIdx, curNode->pos);
        toDeleteCopyTaskNodeResource_.push_back(newNodePtr);
        originNode2copyNode[curNode] = newNodePtr;

        for (auto& child : curNode->children) {
            if (isVisited.find(child) == isVisited.end()) {
                isVisited.insert(child);
                candTaskNodePtr.push_back(child);
            }
        }
    }

    isVisited.clear();
    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
        copyNode->children.push_back(originNode2copyNode[originNode->children[i]]);
    }
    while (!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());
        for (auto& parent : curNode->parents) {
            originNode2copyNode[curNode]->parents.push_back(originNode2copyNode[parent]);
        }
        for (auto& child : curNode->children) {
            originNode2copyNode[curNode]->children.push_back(originNode2copyNode[child]);
            if (isVisited.count(child) == 0) {
                isVisited.insert(child);
                candTaskNodePtr.push_back(child);
            }
        }
    }
}

HcclResult Checker::CopyCcuTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode, uint32_t rankNum)
{
    // 遍历两遍，先将所有节点拷贝出来，再建立父子关系
    std::map<TaskNodePtr, TaskNodePtr> originNode2copyNode; // 用来收录原节点到新节点的映射
    std::vector<TaskNodePtr> candTaskNodePtr;
    std::set<TaskNodePtr> isVisited;

    originNode2copyNode[originNode] = copyNode;
    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
    }

    // 记录ccu子图旧->新节点映射关系
    // 复用taskStub资源，新建taskNode资源
    std::vector<CcuOri2NewNodeMap> ccuOrigin2CopyNodes;
    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuOrigin2CopyNodes.resize(rankNum);
    ccuGraphs.resize(rankNum);

    while (!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());

        TaskStub* newNode = curNode->task;
        if (newNode->GetType() == TaskTypeStub::CCU_GRAPH) {
            TaskStub* newCcu = nullptr;
            CHK_RET(CopyCcuSubGraphNode(newNode, &newCcu, ccuGraphs, ccuOrigin2CopyNodes));
            newNode = newCcu;
            toDeleteCopyTaskResource_.push_back(newNode);
        }
        TaskNodePtr newNodePtr = new TaskNode(newNode, curNode->rankIdx, curNode->queIdx, curNode->pos);
        toDeleteCopyTaskNodeResource_.push_back(newNodePtr);
        originNode2copyNode[curNode] = newNodePtr;

        for (auto& child : curNode->children) {
            if (isVisited.find(child) == isVisited.end()) {
                isVisited.insert(child);
                candTaskNodePtr.push_back(child);
            }
        }
    }

    // 恢复ccu子图内部的连接关系
    CHK_RET(CopyCcuSubGraphConnection(ccuGraphs, ccuOrigin2CopyNodes));
    // 恢复外层拓扑图的连接关系
    isVisited.clear();
    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
        copyNode->children.push_back(originNode2copyNode[originNode->children[i]]);
    }
    while (!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());
        for (auto& parent : curNode->parents) {
            originNode2copyNode[curNode]->parents.push_back(originNode2copyNode[parent]);
        }
        for (auto& child : curNode->children) {
            originNode2copyNode[curNode]->children.push_back(originNode2copyNode[child]);
            if (isVisited.count(child) == 0) {
                isVisited.insert(child);
                candTaskNodePtr.push_back(child);
            }
        }
    }

    for (auto& ccuGraph : ccuGraphs) {
        for (auto ccuPair : ccuGraph) {
            g_ccuGraphTaskOri2New.insert(ccuPair);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
