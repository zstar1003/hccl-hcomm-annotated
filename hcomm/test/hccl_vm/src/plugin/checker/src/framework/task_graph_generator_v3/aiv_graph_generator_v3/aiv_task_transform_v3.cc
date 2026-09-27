/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_task_transform_v3.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "aiv_snapshot_json_loader_v3.h"
#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        constexpr uint32_t PIPE_S = 0U;
        constexpr uint32_t PIPE_MTE2 = 1U;
        constexpr uint32_t PIPE_MTE3 = 2U;
        constexpr uint32_t PIPE_ALL = 3U;
        constexpr uint32_t AIV_PIPE_NUM = 3U;

        using AivExpandClock = std::chrono::steady_clock;

        struct AivNodeRecord {
            NodeId nodeId{INVALID_NODE_ID};
            const AivRuntimeTaskV3* task{nullptr};
            uint64_t order{0};
        };

        struct AivLaunchContext {
            TaskGraphGeneratorV3* graph{nullptr};
            StorageManager* storage{nullptr};
            TaskAivGraph* placeholder{nullptr};
            AivRuntimeTaskSnapshotV3 snapshot;
            NodeId headNodeId{INVALID_NODE_ID};
            NodeId endNodeId{INVALID_NODE_ID};
            std::map<uint32_t, AivNodeRecord> taskIdToNode;
            std::vector<NodeId> internalNodeIds;
            std::set<NodeId> inactiveNodeIds;
            std::unordered_map<NodeId, std::vector<uint32_t>> cpGmMergeSourceTaskIds;
            uint32_t nextSyntheticTaskId{0};
            size_t setWaitEdgeCount{0};
            size_t pipeBarrierMergeCount{0};
            size_t syncAllMergeCount{0};
            size_t taskJsonTotalTaskCount{0};
            size_t dagNodeCountBeforeCpGmMerge{0};
            size_t dagNodeCountAfterCpGmMerge{0};
            size_t cpGmLoopMergeCount{0};
            size_t cpGmMergedIterationCount{0};
            size_t cpGmMergedOriginalNodeCount{0};
            size_t cpGmGeneratedNodeCount{0};
            size_t cpGmInactiveNodeCount{0};
            size_t cpGmLoopSkipCount{0};
        };

        struct SetWaitKey {
            RankId rankId{INVALID_RANK_ID};
            uint64_t launchIdx{0};
            uint32_t blockId{0};
            uint32_t srcPipe{0};
            uint32_t dstPipe{0};
            int32_t eventId{0};

            bool operator<(const SetWaitKey& rhs) const
            {
                if (rankId != rhs.rankId) {
                    return rankId < rhs.rankId;
                }
                if (launchIdx != rhs.launchIdx) {
                    return launchIdx < rhs.launchIdx;
                }
                if (blockId != rhs.blockId) {
                    return blockId < rhs.blockId;
                }
                if (srcPipe != rhs.srcPipe) {
                    return srcPipe < rhs.srcPipe;
                }
                if (dstPipe != rhs.dstPipe) {
                    return dstPipe < rhs.dstPipe;
                }
                return eventId < rhs.eventId;
            }
        };

        struct FlagCellKey {
            RankId flagOwnerRank{INVALID_RANK_ID};
            uint64_t launchIdx{0};
            uint64_t commInfoOffset{0};

            bool operator<(const FlagCellKey& rhs) const
            {
                if (flagOwnerRank != rhs.flagOwnerRank) {
                    return flagOwnerRank < rhs.flagOwnerRank;
                }
                if (launchIdx != rhs.launchIdx) {
                    return launchIdx < rhs.launchIdx;
                }
                return commInfoOffset < rhs.commInfoOffset;
            }
        };

        struct FlagCellState {
            int32_t currentValue{0};
            std::vector<NodeId> producerNodes;
        };

        struct MergeGroup {
            std::vector<NodeId> memberNodeIds;
            std::vector<uint32_t> memberTaskIds;
            std::vector<uint32_t> memberPipes;
            uint64_t topoOrder{0};
            uint32_t blockId{0};
            uint32_t pipeType{std::numeric_limits<uint32_t>::max()};
            uint32_t syncRound{std::numeric_limits<uint32_t>::max()};
        };

        using CpGmIter = std::array<TaskNode*, 6>;

        struct CpGmLoopGather {
            std::vector<CpGmIter> templateLoop;
            std::array<TaskNode*, 6> merged{};
        };

        uint64_t ElapsedNs(AivExpandClock::time_point start, AivExpandClock::time_point end)
        {
            return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        }

        bool IsValidPipe(uint32_t pipe) { return pipe < AIV_PIPE_NUM; }

        TaskPosition MakeAivPosition(
            RankId rankId, uint64_t launchIdx, uint32_t blockId, uint32_t pipe, uint32_t taskId, OperatorId operatorId)
        {
            TaskPosition position;
            position.operatorId = operatorId;
            position.rankId = rankId;
            position.streamId = INVALID_STREAM_ID;
            position.launchIdx = launchIdx;
            position.blockId = blockId;
            position.pipe = pipe;
            position.taskId = taskId;
            const uint32_t queueBlockId = blockId == std::numeric_limits<uint32_t>::max() ? 0U : blockId;
            const uint32_t queuePipe = IsValidPipe(pipe) ? pipe : 0U;
            position.queueId = static_cast<QueueId>(
                (launchIdx & 0xFFFFULL) * AIV_PIPE_NUM * 1024ULL + static_cast<uint64_t>(queueBlockId) * AIV_PIPE_NUM
                + queuePipe);
            return position;
        }

        MemType ConvertAivMemType(AivBufferTypeV3 bufferType)
        {
            switch (bufferType) {
                case AivBufferTypeV3::INPUT:
                    return MemType::INPUT;
                case AivBufferTypeV3::OUTPUT:
                    return MemType::OUTPUT;
                case AivBufferTypeV3::CCL:
                    return MemType::CCL;
                case AivBufferTypeV3::UB:
                    return MemType::AIV_UB;
                case AivBufferTypeV3::AIV_COMM:
                    return MemType::AIV_COMM;
                default:
                    return MemType::INVALID;
            }
        }

        MemSlice ConvertAivSlice(RankId rankId, const AivDataSliceV3& slice)
        {
            MemSlice result;
            result.rankId = rankId;
            result.memType = ConvertAivMemType(slice.type);
            result.offset = slice.offset;
            result.len = slice.size;
            return result;
        }

        TaskPosition MakeAivLocation(const AivRuntimeTaskV3& task, uint64_t launchIdx, OperatorId operatorId)
        {
            return MakeAivPosition(task.rankId, launchIdx, task.blockId, task.curPipe, task.taskId, operatorId);
        }

        SetWaitKey MakeSetWaitKey(const AivPipeEvent& event)
        {
            return SetWaitKey{event.rankId,  event.launchIdx, event.blockId,
                              event.srcPipe, event.dstPipe,   event.eventId};
        }

        FlagCellKey MakeFlagCellKey(const AivFlagSync& flag)
        {
            return FlagCellKey{flag.flagOwnerRank, flag.launchIdx, flag.commInfoOffset};
        }

        std::unique_ptr<TaskNode>
        TranslateAivRuntimeTask(const AivRuntimeTaskV3& task, uint64_t launchIdx, OperatorId operatorId)
        {
            switch (task.taskType) {
                case AivRuntimeTaskTypeV3::MEM_COPY: {
                    const MemSlice src = ConvertAivSlice(task.srcRank, task.src);
                    const MemSlice dst = ConvertAivSlice(task.dstRank, task.dst);
                    return std::make_unique<TaskTransMem>(src, dst, ProtocolType::SDMA);
                }
                case AivRuntimeTaskTypeV3::REDUCE: {
                    const MemSlice src = ConvertAivSlice(task.srcRank, task.src);
                    const MemSlice dst = ConvertAivSlice(task.dstRank, task.dst);
                    auto node = std::make_unique<TaskReduce>(
                        src, dst, static_cast<uint8_t>(task.dataType), static_cast<uint8_t>(task.reduceOp),
                        ProtocolType::SDMA);
                    return node;
                }
                case AivRuntimeTaskTypeV3::SET_FLAG: {
                    AivPipeEvent event;
                    event.rankId = task.rankId;
                    event.launchIdx = launchIdx;
                    event.blockId = task.blockId;
                    event.curPipe = task.curPipe;
                    event.srcPipe = task.srcPipe;
                    event.dstPipe = task.dstPipe;
                    event.eventId = task.eventId;
                    event.taskId = task.taskId;
                    return std::make_unique<TaskAivSetFlag>(event);
                }
                case AivRuntimeTaskTypeV3::WAIT_FLAG: {
                    AivPipeEvent event;
                    event.rankId = task.rankId;
                    event.launchIdx = launchIdx;
                    event.blockId = task.blockId;
                    event.curPipe = task.curPipe;
                    event.srcPipe = task.srcPipe;
                    event.dstPipe = task.dstPipe;
                    event.eventId = task.eventId;
                    event.taskId = task.taskId;
                    return std::make_unique<TaskAivWaitFlag>(event);
                }
                case AivRuntimeTaskTypeV3::PIPE_BARRIER: {
                    AivBarrierInfo info;
                    info.taskLoc = MakeAivLocation(task, launchIdx, operatorId);
                    info.pipeType = task.pipeType;
                    info.memberTaskIds = task.barrierGroupTaskIds;
                    return std::make_unique<TaskAivPipeBarrier>(std::move(info));
                }
                case AivRuntimeTaskTypeV3::SYNC_ALL: {
                    AivSyncAllInfo info;
                    info.taskLoc = MakeAivLocation(task, launchIdx, operatorId);
                    info.syncRound = task.syncRound;
                    return std::make_unique<TaskAivSyncAll>(std::move(info));
                }
                case AivRuntimeTaskTypeV3::SEND_FLAG: {
                    AivFlagSync flag;
                    flag.currentRank = task.rankId;
                    flag.flagOwnerRank = task.flagOwnerRank;
                    flag.launchIdx = launchIdx;
                    flag.blockId = task.blockId;
                    flag.curPipe = task.curPipe;
                    flag.taskId = task.taskId;
                    flag.commInfoOffset = task.commInfoOffset;
                    flag.value = task.flagValue;
                    return std::make_unique<TaskAivSendFlag>(flag);
                }
                case AivRuntimeTaskTypeV3::RECV_FLAG: {
                    AivFlagSync flag;
                    flag.currentRank = task.rankId;
                    flag.flagOwnerRank = task.flagOwnerRank;
                    flag.launchIdx = launchIdx;
                    flag.blockId = task.blockId;
                    flag.curPipe = task.curPipe;
                    flag.taskId = task.taskId;
                    flag.commInfoOffset = task.commInfoOffset;
                    flag.value = task.flagValue;
                    return std::make_unique<TaskAivRecvFlag>(flag);
                }
                default:
                    return nullptr;
            }
        }

        HcclResult AddEdge(TaskGraphGeneratorV3* graph, NodeId parent, NodeId child)
        {
            if (graph == nullptr) {
                return HCCL_E_PTR;
            }
            if (parent == child) {
                return HCCL_SUCCESS;
            }
            return graph->AddEdge(parent, child);
        }

        bool IsInactive(const AivLaunchContext& ctx, NodeId nodeId) { return ctx.inactiveNodeIds.count(nodeId) != 0; }

        HcclResult RewireGroupMember(TaskGraphGeneratorV3* graph, NodeId memberNodeId, NodeId mergeNodeId)
        {
            if (graph == nullptr) {
                return HCCL_E_PTR;
            }
            TaskNode* member = graph->GetNode(memberNodeId);
            if (member == nullptr) {
                return HCCL_E_PTR;
            }

            std::vector<TaskNode*> parents = member->GetParents();
            std::vector<TaskNode*> children = member->GetChildren();
            for (TaskNode* parent : parents) {
                if (parent == nullptr || parent->GetNodeId() == mergeNodeId) {
                    continue;
                }
                HcclResult ret = graph->RemoveEdge(parent->GetNodeId(), memberNodeId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = AddEdge(graph, parent->GetNodeId(), mergeNodeId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }

            for (TaskNode* child : children) {
                if (child == nullptr || child->GetNodeId() == mergeNodeId) {
                    continue;
                }
                HcclResult ret = graph->RemoveEdge(memberNodeId, child->GetNodeId());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = AddEdge(graph, mergeNodeId, child->GetNodeId());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult ValidateSnapshot(const AivLaunchContext& ctx)
        {
            if (ctx.storage == nullptr || ctx.placeholder == nullptr) {
                return HCCL_E_PTR;
            }
            const CheckerParam param = ctx.storage->GetCheckerParam(ctx.placeholder->GetOperatorId());
            if (ctx.snapshot.rankSize != 0 && param.rankSize != 0 && ctx.snapshot.rankSize != param.rankSize) {
                HCCL_VM_ERROR(
                    "{} The AIV snapshot was captured for a different rank count than the current "
                    "checker input, rankId={}, launchId={}, snapshotRankCount={}, currentRankCount={}, snapshotFile={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_SNAPSHOT_MISMATCH), ctx.placeholder->GetRankId(),
                    ctx.placeholder->GetLaunchIdx(), ctx.snapshot.rankSize, param.rankSize, ctx.snapshot.filePath);
                return HCCL_E_PARA;
            }
            return HCCL_SUCCESS;
        }

        HcclResult UpdateAivBufferSize(
            uint64_t snapshotSize, const char* fieldName, const AivLaunchContext& ctx, uint64_t& bufferSize)
        {
            if (snapshotSize == 0) {
                return HCCL_SUCCESS;
            }
            if (bufferSize == 0) {
                bufferSize = snapshotSize;
                return HCCL_SUCCESS;
            }
            if (bufferSize == snapshotSize) {
                return HCCL_SUCCESS;
            }

            HCCL_VM_ERROR(
                "{} AIV buffer size is inconsistent across snapshots, field={}, rankId={}, "
                "launchId={}, expectedSize={}, actualSize={}, expectedSource=firstNonZeroSnapshot, snapshotFile={}",
                MakeErrorCodeText(ErrorCode::GRAPH_SNAPSHOT_MISMATCH), fieldName, ctx.placeholder->GetRankId(),
                ctx.placeholder->GetLaunchIdx(), bufferSize, snapshotSize, ctx.snapshot.filePath);
            return HCCL_E_PARA;
        }

        HcclResult
        AppendRuntimeTask(AivLaunchContext& ctx, const AivRuntimeTaskV3& task, uint64_t order, NodeId& nodeId)
        {
            std::unique_ptr<TaskNode> node
                = TranslateAivRuntimeTask(task, ctx.placeholder->GetLaunchIdx(), ctx.placeholder->GetOperatorId());
            if (node == nullptr) {
                HCCL_VM_ERROR(
                    "{} One AIV runtime task type is not supported, "
                    "rankId={}, launchId={}, taskId={}, taskType={}, snapshotFile={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED), ctx.placeholder->GetRankId(),
                    ctx.placeholder->GetLaunchIdx(), task.taskId, static_cast<uint32_t>(task.taskType),
                    ctx.snapshot.filePath);
                return HCCL_E_NOT_SUPPORT;
            }

            const TaskPosition position = MakeAivPosition(
                task.rankId, ctx.placeholder->GetLaunchIdx(), task.blockId, task.curPipe, task.taskId,
                ctx.placeholder->GetOperatorId());
            HcclResult ret = ctx.graph->AppendGeneratedNode(std::move(node), position, nodeId);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            ctx.internalNodeIds.push_back(nodeId);
            ctx.taskIdToNode[task.taskId] = AivNodeRecord{nodeId, &task, order};
            return HCCL_SUCCESS;
        }

        HcclResult AppendPipeTasks(
            AivLaunchContext& ctx, const std::vector<AivRuntimeTaskV3>& tasks, NodeId headNodeId, uint64_t& order,
            std::vector<NodeId>& tailNodes)
        {
            NodeId prevNodeId = INVALID_NODE_ID;
            for (const auto& task : tasks) {
                NodeId currNodeId = INVALID_NODE_ID;
                HcclResult ret = AppendRuntimeTask(ctx, task, order++, currNodeId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = (prevNodeId == INVALID_NODE_ID) ? AddEdge(ctx.graph, headNodeId, currNodeId) :
                                                        AddEdge(ctx.graph, prevNodeId, currNodeId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                prevNodeId = currNodeId;
            }
            if (prevNodeId != INVALID_NODE_ID) {
                tailNodes.push_back(prevNodeId);
            }
            return HCCL_SUCCESS;
        }

        HcclResult BuildPipeSkeleton(AivLaunchContext& ctx, std::vector<NodeId>& tailNodes)
        {
            uint64_t order = 0;
            for (const auto& block : ctx.snapshot.blocks) {
                HcclResult ret = AppendPipeTasks(ctx, block.scalarTasks, ctx.headNodeId, order, tailNodes);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = AppendPipeTasks(ctx, block.mte2Tasks, ctx.headNodeId, order, tailNodes);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = AppendPipeTasks(ctx, block.mte3Tasks, ctx.headNodeId, order, tailNodes);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult AddSetWaitFlagEdges(AivLaunchContext& ctx)
        {
            std::set<NodeId> internalSet(ctx.internalNodeIds.begin(), ctx.internalNodeIds.end());
            std::deque<NodeId> readyQueue;
            std::set<NodeId> queued;
            std::set<NodeId> executed;
            std::map<SetWaitKey, std::deque<NodeId>> seenSetFlags;

            TaskNode* head = ctx.graph->GetNode(ctx.headNodeId);
            if (head == nullptr) {
                return HCCL_E_PTR;
            }
            for (TaskNode* child : head->GetChildren()) {
                if (child == nullptr) {
                    return HCCL_E_PTR;
                }
                if (internalSet.count(child->GetNodeId()) != 0) {
                    queued.insert(child->GetNodeId());
                    readyQueue.push_back(child->GetNodeId());
                }
            }

            auto isExecutable = [&executed](const TaskNode* node) {
                if (node == nullptr) {
                    return false;
                }
                for (const TaskNode* parent : node->GetParents()) {
                    if (parent == nullptr) {
                        return false;
                    }
                    if (parent->GetType() == TaskType::START) {
                        continue;
                    }
                    if (executed.count(parent->GetNodeId()) == 0) {
                        return false;
                    }
                }
                return true;
            };
            auto enqueueChildren = [&ctx, &internalSet, &queued, &readyQueue](const TaskNode* node) -> HcclResult {
                if (node == nullptr) {
                    return HCCL_E_PTR;
                }
                for (TaskNode* child : node->GetChildren()) {
                    if (child == nullptr) {
                        return HCCL_E_PTR;
                    }
                    const NodeId childId = child->GetNodeId();
                    if (internalSet.count(childId) == 0 || queued.count(childId) != 0) {
                        continue;
                    }
                    readyQueue.push_back(childId);
                    queued.insert(childId);
                }
                return HCCL_SUCCESS;
            };

            size_t unmatchedCnt = 0;
            while (!readyQueue.empty()) {
                if (unmatchedCnt >= readyQueue.size()) {
                    TaskNode* node = ctx.graph->GetNode(readyQueue.front());
                    size_t seenMatchingSetFlagCount = 0;
                    if (node != nullptr && node->GetType() == TaskType::AIV_WAIT_FLAG) {
                        const auto* waitFlag = dynamic_cast<const TaskAivWaitFlag*>(node);
                        if (waitFlag != nullptr) {
                            const SetWaitKey key = MakeSetWaitKey(waitFlag->GetEvent());
                            const auto iter = seenSetFlags.find(key);
                            seenMatchingSetFlagCount = (iter == seenSetFlags.end()) ? 0U : iter->second.size();
                        }
                    }
                    HCCL_VM_ERROR(
                        "{} AIV SetFlag/WaitFlag matching is stuck. Some WaitFlag tasks are still "
                        "blocked, but no matching SetFlag has become available, firstBlockedWaitFlagNode={}, "
                        "blockedWaitFlagNodeCount={}, availableSetFlagCountForThisWait={}, snapshotFile={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_DEADLOCK), node == nullptr ? "node=null" : node->Describe(),
                        readyQueue.size(), seenMatchingSetFlagCount, ctx.snapshot.filePath);
                    return HCCL_E_INTERNAL;
                }

                const NodeId nodeId = readyQueue.front();
                readyQueue.pop_front();
                queued.erase(nodeId);
                if (executed.count(nodeId) != 0) {
                    unmatchedCnt = 0;
                    continue;
                }

                TaskNode* node = ctx.graph->GetNode(nodeId);
                if (node == nullptr) {
                    return HCCL_E_PTR;
                }
                if (!isExecutable(node)) {
                    readyQueue.push_back(nodeId);
                    queued.insert(nodeId);
                    ++unmatchedCnt;
                    continue;
                }

                if (node->GetType() == TaskType::AIV_SET_FLAG) {
                    const auto* setFlag = dynamic_cast<const TaskAivSetFlag*>(node);
                    if (setFlag == nullptr) {
                        return HCCL_E_PTR;
                    }
                    seenSetFlags[MakeSetWaitKey(setFlag->GetEvent())].push_back(nodeId);
                    executed.insert(nodeId);
                    HcclResult ret = enqueueChildren(node);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    unmatchedCnt = 0;
                    continue;
                }
                if (node->GetType() != TaskType::AIV_WAIT_FLAG) {
                    executed.insert(nodeId);
                    HcclResult ret = enqueueChildren(node);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    unmatchedCnt = 0;
                    continue;
                }
                const auto* waitFlag = dynamic_cast<const TaskAivWaitFlag*>(node);
                if (waitFlag == nullptr) {
                    return HCCL_E_PTR;
                }
                const SetWaitKey key = MakeSetWaitKey(waitFlag->GetEvent());
                auto iter = seenSetFlags.find(key);
                if (iter == seenSetFlags.end() || iter->second.empty()) {
                    readyQueue.push_back(nodeId);
                    queued.insert(nodeId);
                    ++unmatchedCnt;
                    continue;
                }
                const NodeId setNodeId = iter->second.front();
                iter->second.pop_front();
                HcclResult ret = AddEdge(ctx.graph, setNodeId, nodeId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ++ctx.setWaitEdgeCount;
                executed.insert(nodeId);
                ret = enqueueChildren(node);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                unmatchedCnt = 0;
            }

            for (const auto& entry : seenSetFlags) {
                if (!entry.second.empty()) {
                    TaskNode* node = ctx.graph->GetNode(entry.second.front());
                    HCCL_VM_WARN(
                        "{} Found SetFlag tasks that were never consumed by any WaitFlag task, "
                        "firstUnconsumedSetFlagNode={}, unconsumedSetFlagCount={}, snapshotFile={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED), node == nullptr ? "node=null" : node->Describe(),
                        entry.second.size(), ctx.snapshot.filePath);
                }
            }
            return HCCL_SUCCESS;
        }

        size_t CountAivSnapshotTasks(const AivRuntimeTaskSnapshotV3& snapshot)
        {
            size_t count = 0;
            for (const auto& block : snapshot.blocks) {
                count += block.scalarTasks.size();
                count += block.mte2Tasks.size();
                count += block.mte3Tasks.size();
            }
            return count;
        }

        uint32_t FindMaxTaskId(const AivRuntimeTaskSnapshotV3& snapshot)
        {
            uint32_t maxTaskId = 0;
            for (const auto& block : snapshot.blocks) {
                auto update = [&maxTaskId](const std::vector<AivRuntimeTaskV3>& tasks) {
                    for (const auto& task : tasks) {
                        maxTaskId = std::max(maxTaskId, task.taskId);
                    }
                };
                update(block.scalarTasks);
                update(block.mte2Tasks);
                update(block.mte3Tasks);
            }
            return maxTaskId;
        }

        size_t CountReachableActiveInternalNodes(const AivLaunchContext& ctx)
        {
            if (ctx.graph == nullptr || ctx.headNodeId == INVALID_NODE_ID) {
                return 0;
            }
            std::set<NodeId> internalSet(ctx.internalNodeIds.begin(), ctx.internalNodeIds.end());
            std::set<NodeId> visited;
            std::queue<NodeId> nodeQueue;
            nodeQueue.push(ctx.headNodeId);
            visited.insert(ctx.headNodeId);

            while (!nodeQueue.empty()) {
                const NodeId nodeId = nodeQueue.front();
                nodeQueue.pop();
                const TaskNode* node = ctx.graph->GetNode(nodeId);
                if (node == nullptr) {
                    continue;
                }
                for (const TaskNode* child : node->GetChildren()) {
                    if (child == nullptr) {
                        continue;
                    }
                    const NodeId childId = child->GetNodeId();
                    if (internalSet.count(childId) == 0 || IsInactive(ctx, childId)) {
                        continue;
                    }
                    if (visited.insert(childId).second) {
                        nodeQueue.push(childId);
                    }
                }
            }
            return visited.size();
        }

        bool IsCpGmExternalMemType(MemType memType)
        {
            return memType == MemType::INPUT || memType == MemType::CCL || memType == MemType::OUTPUT
                   || memType == MemType::AIV_COMM;
        }

        bool SameSliceIdentity(const MemSlice& lhs, const MemSlice& rhs)
        {
            return lhs.rankId == rhs.rankId && lhs.memType == rhs.memType;
        }

        bool IsContinuousAfter(const MemSlice& prev, const MemSlice& next)
        {
            return SameSliceIdentity(prev, next) && next.offset == prev.offset + prev.len;
        }

        bool SliceExactEqual(const MemSlice& lhs, const MemSlice& rhs)
        {
            return lhs.rankId == rhs.rankId && lhs.memType == rhs.memType && lhs.offset == rhs.offset
                   && lhs.len == rhs.len;
        }

        bool IntervalsOverlap(const MemSlice& lhs, const MemSlice& rhs)
        {
            if (!SameSliceIdentity(lhs, rhs)) {
                return false;
            }
            return lhs.offset < rhs.offset + rhs.len && rhs.offset < lhs.offset + lhs.len;
        }

        std::vector<MemSlice> MergeMemSliceIntervals(std::vector<MemSlice> slices)
        {
            if (slices.empty()) {
                return {};
            }
            std::sort(slices.begin(), slices.end(), [](const MemSlice& lhs, const MemSlice& rhs) {
                if (lhs.rankId != rhs.rankId) {
                    return lhs.rankId < rhs.rankId;
                }
                if (lhs.memType != rhs.memType) {
                    return static_cast<uint32_t>(lhs.memType) < static_cast<uint32_t>(rhs.memType);
                }
                if (lhs.offset != rhs.offset) {
                    return lhs.offset < rhs.offset;
                }
                return lhs.len < rhs.len;
            });

            std::vector<MemSlice> merged;
            merged.push_back(slices.front());
            for (size_t index = 1; index < slices.size(); ++index) {
                MemSlice& last = merged.back();
                const MemSlice& cur = slices[index];
                if (!SameSliceIdentity(last, cur)) {
                    merged.push_back(cur);
                    continue;
                }
                const uint64_t lastEnd = last.offset + last.len;
                const uint64_t curEnd = cur.offset + cur.len;
                if (cur.offset <= lastEnd) {
                    last.len = std::max(lastEnd, curEnd) - last.offset;
                    continue;
                }
                merged.push_back(cur);
            }
            return merged;
        }

        bool BuildContinuousMergedSlices(const std::vector<MemSlice>& slices, std::vector<MemSlice>& merged)
        {
            merged.clear();
            if (slices.empty()) {
                return false;
            }
            MemSlice cur = slices.front();
            for (size_t index = 1; index < slices.size(); ++index) {
                if (!IsContinuousAfter(cur, slices[index])) {
                    return false;
                }
                cur.len += slices[index].len;
            }
            merged.push_back(cur);
            return true;
        }

        bool GetDataSlices(const TaskNode* node, MemSlice& src, MemSlice& dst, uint8_t& dataType, uint8_t& reduceOp)
        {
            if (node == nullptr) {
                return false;
            }
            if (node->GetType() == TaskType::TRANS_MEM) {
                const auto* task = dynamic_cast<const TaskTransMem*>(node);
                if (task == nullptr) {
                    return false;
                }
                src = task->GetSrc();
                dst = task->GetDst();
                dataType = 0;
                reduceOp = 0;
                return true;
            }
            if (node->GetType() == TaskType::REDUCE) {
                const auto* task = dynamic_cast<const TaskReduce*>(node);
                if (task == nullptr || task->GetSrcs().size() != 1U) {
                    return false;
                }
                src = task->GetSrc();
                dst = task->GetDst();
                dataType = task->GetDataType();
                reduceOp = task->GetReduceOp();
                return true;
            }
            return false;
        }

        bool IsValidCpGmDataNode(const TaskNode* node, uint32_t pipe, bool ubAsDst)
        {
            MemSlice src;
            MemSlice dst;
            uint8_t dataType = 0;
            uint8_t reduceOp = 0;
            if (!GetDataSlices(node, src, dst, dataType, reduceOp)) {
                return false;
            }
            if (node->GetPosition().pipe != pipe) {
                return false;
            }
            const MemSlice& ubSlice = ubAsDst ? dst : src;
            const MemSlice& externalSlice = ubAsDst ? src : dst;
            return ubSlice.memType == MemType::AIV_UB && IsCpGmExternalMemType(externalSlice.memType);
        }

        bool
        MatchPipeEventNode(const TaskNode* node, TaskType type, uint32_t curPipe, uint32_t srcPipe, uint32_t dstPipe)
        {
            if (node == nullptr || node->GetType() != type || node->GetPosition().pipe != curPipe) {
                return false;
            }
            const AivPipeEvent* event = nullptr;
            if (type == TaskType::AIV_SET_FLAG) {
                const auto* setFlag = dynamic_cast<const TaskAivSetFlag*>(node);
                event = setFlag == nullptr ? nullptr : &setFlag->GetEvent();
            } else {
                const auto* waitFlag = dynamic_cast<const TaskAivWaitFlag*>(node);
                event = waitFlag == nullptr ? nullptr : &waitFlag->GetEvent();
            }
            return event != nullptr && event->curPipe == curPipe && event->srcPipe == srcPipe
                   && event->dstPipe == dstPipe;
        }

        const AivPipeEvent* GetPipeEvent(const TaskNode* node)
        {
            if (node == nullptr) {
                return nullptr;
            }
            if (node->GetType() == TaskType::AIV_SET_FLAG) {
                const auto* setFlag = dynamic_cast<const TaskAivSetFlag*>(node);
                return setFlag == nullptr ? nullptr : &setFlag->GetEvent();
            }
            if (node->GetType() == TaskType::AIV_WAIT_FLAG) {
                const auto* waitFlag = dynamic_cast<const TaskAivWaitFlag*>(node);
                return waitFlag == nullptr ? nullptr : &waitFlag->GetEvent();
            }
            return nullptr;
        }

        bool SamePositionScope(const TaskPosition& lhs, const TaskPosition& rhs)
        {
            return lhs.rankId == rhs.rankId && lhs.launchIdx == rhs.launchIdx && lhs.blockId == rhs.blockId;
        }

        bool SameEventExceptTaskId(const TaskNode* lhsNode, const TaskNode* rhsNode)
        {
            if (lhsNode == nullptr || rhsNode == nullptr || lhsNode->GetType() != rhsNode->GetType()) {
                return false;
            }
            const AivPipeEvent* lhs = GetPipeEvent(lhsNode);
            const AivPipeEvent* rhs = GetPipeEvent(rhsNode);
            if (lhs == nullptr || rhs == nullptr) {
                return false;
            }
            return lhs->rankId == rhs->rankId && lhs->launchIdx == rhs->launchIdx && lhs->blockId == rhs->blockId
                   && lhs->curPipe == rhs->curPipe && lhs->srcPipe == rhs->srcPipe && lhs->dstPipe == rhs->dstPipe
                   && lhs->eventId == rhs->eventId;
        }

        bool SameSetWaitEvent(const TaskNode* setNode, const TaskNode* waitNode)
        {
            const AivPipeEvent* setEvent = GetPipeEvent(setNode);
            const AivPipeEvent* waitEvent = GetPipeEvent(waitNode);
            if (setEvent == nullptr || waitEvent == nullptr) {
                return false;
            }
            return setEvent->rankId == waitEvent->rankId && setEvent->launchIdx == waitEvent->launchIdx
                   && setEvent->blockId == waitEvent->blockId && setEvent->srcPipe == waitEvent->srcPipe
                   && setEvent->dstPipe == waitEvent->dstPipe && setEvent->eventId == waitEvent->eventId;
        }

        bool SameDataTypeAndOp(const TaskNode* lhsNode, const TaskNode* rhsNode)
        {
            if (lhsNode == nullptr || rhsNode == nullptr || lhsNode->GetType() != rhsNode->GetType()) {
                return false;
            }
            if (lhsNode->GetType() != TaskType::REDUCE) {
                return true;
            }
            MemSlice lhsSrc;
            MemSlice lhsDst;
            uint8_t lhsDataType = 0;
            uint8_t lhsReduceOp = 0;
            MemSlice rhsSrc;
            MemSlice rhsDst;
            uint8_t rhsDataType = 0;
            uint8_t rhsReduceOp = 0;
            return GetDataSlices(lhsNode, lhsSrc, lhsDst, lhsDataType, lhsReduceOp)
                   && GetDataSlices(rhsNode, rhsSrc, rhsDst, rhsDataType, rhsReduceOp) && lhsDataType == rhsDataType
                   && lhsReduceOp == rhsReduceOp;
        }

        bool DataSliceShapeValid(const CpGmIter& iter)
        {
            MemSlice t0Src;
            MemSlice t0Dst;
            MemSlice t3Src;
            MemSlice t3Dst;
            uint8_t dataType = 0;
            uint8_t reduceOp = 0;
            if (!GetDataSlices(iter[0], t0Src, t0Dst, dataType, reduceOp)
                || !GetDataSlices(iter[3], t3Src, t3Dst, dataType, reduceOp)) {
                return false;
            }
            if (t0Dst.memType != MemType::AIV_UB || t3Src.memType != MemType::AIV_UB) {
                return false;
            }
            if (!IsCpGmExternalMemType(t0Src.memType) || !IsCpGmExternalMemType(t3Dst.memType)) {
                return false;
            }
            if (t0Dst.len != t3Src.len) {
                return false;
            }
            return true;
        }

        bool SameUbLayout(const CpGmIter& iter)
        {
            MemSlice t0Src;
            MemSlice t0Dst;
            MemSlice t3Src;
            MemSlice t3Dst;
            uint8_t dataType = 0;
            uint8_t reduceOp = 0;
            if (!GetDataSlices(iter[0], t0Src, t0Dst, dataType, reduceOp)
                || !GetDataSlices(iter[3], t3Src, t3Dst, dataType, reduceOp)) {
                return false;
            }
            return SliceExactEqual(t0Dst, t3Src);
        }

        bool SameExternalIdentityAsFirst(const CpGmIter& first, const CpGmIter& iter)
        {
            MemSlice firstT0Src;
            MemSlice firstT0Dst;
            MemSlice firstT3Src;
            MemSlice firstT3Dst;
            MemSlice curT0Src;
            MemSlice curT0Dst;
            MemSlice curT3Src;
            MemSlice curT3Dst;
            uint8_t dataType = 0;
            uint8_t reduceOp = 0;
            if (!GetDataSlices(first[0], firstT0Src, firstT0Dst, dataType, reduceOp)
                || !GetDataSlices(first[3], firstT3Src, firstT3Dst, dataType, reduceOp)
                || !GetDataSlices(iter[0], curT0Src, curT0Dst, dataType, reduceOp)
                || !GetDataSlices(iter[3], curT3Src, curT3Dst, dataType, reduceOp)) {
                return false;
            }
            return SameSliceIdentity(firstT0Src, curT0Src) && SameSliceIdentity(firstT3Dst, curT3Dst);
        }

        bool ExternalSlicesContinuous(const CpGmLoopGather& gather)
        {
            if (gather.templateLoop.empty()) {
                return false;
            }
            std::vector<MemSlice> t0Srcs;
            std::vector<MemSlice> t3Dsts;
            t0Srcs.reserve(gather.templateLoop.size());
            t3Dsts.reserve(gather.templateLoop.size());
            for (const CpGmIter& iter : gather.templateLoop) {
                MemSlice t0Src;
                MemSlice t0Dst;
                MemSlice t3Src;
                MemSlice t3Dst;
                uint8_t dataType = 0;
                uint8_t reduceOp = 0;
                if (!GetDataSlices(iter[0], t0Src, t0Dst, dataType, reduceOp)
                    || !GetDataSlices(iter[3], t3Src, t3Dst, dataType, reduceOp)) {
                    return false;
                }
                t0Srcs.push_back(t0Src);
                t3Dsts.push_back(t3Dst);
            }
            std::vector<MemSlice> merged;
            return BuildContinuousMergedSlices(t0Srcs, merged) && BuildContinuousMergedSlices(t3Dsts, merged);
        }

        bool HasValidSyntheticTaskIdBudget(const AivLaunchContext& ctx)
        {
            const uint32_t maxTaskId = std::numeric_limits<uint32_t>::max();
            return ctx.nextSyntheticTaskId <= maxTaskId - 6U;
        }

        bool CanMergeCpGmRun(const AivLaunchContext& ctx, const CpGmLoopGather& gather)
        {
            if (!HasValidSyntheticTaskIdBudget(ctx)) {
                return false;
            }
            return ExternalSlicesContinuous(gather);
        }

        TaskNode* FindOnlyChildByType(const AivLaunchContext& ctx, const TaskNode* node, TaskType type)
        {
            if (node == nullptr) {
                return nullptr;
            }
            TaskNode* matched = nullptr;
            for (TaskNode* child : node->GetChildren()) {
                if (child == nullptr || IsInactive(ctx, child->GetNodeId()) || child->GetType() != type) {
                    continue;
                }
                if (matched != nullptr) {
                    return nullptr;
                }
                matched = child;
            }
            return matched;
        }

        TaskNode* FindOnlyWaitChildForSet(const AivLaunchContext& ctx, const TaskNode* setNode)
        {
            if (setNode == nullptr) {
                return nullptr;
            }
            TaskNode* matched = nullptr;
            for (TaskNode* child : setNode->GetChildren()) {
                if (child == nullptr || IsInactive(ctx, child->GetNodeId())
                    || child->GetType() != TaskType::AIV_WAIT_FLAG || !SameSetWaitEvent(setNode, child)) {
                    continue;
                }
                if (matched != nullptr) {
                    return nullptr;
                }
                matched = child;
            }
            return matched;
        }

        TaskNode* FindOnlySetParentForWait(const AivLaunchContext& ctx, const TaskNode* waitNode)
        {
            if (waitNode == nullptr) {
                return nullptr;
            }
            TaskNode* matched = nullptr;
            for (TaskNode* parent : waitNode->GetParents()) {
                if (parent == nullptr || IsInactive(ctx, parent->GetNodeId())
                    || parent->GetType() != TaskType::AIV_SET_FLAG || !SameSetWaitEvent(parent, waitNode)) {
                    continue;
                }
                if (matched != nullptr) {
                    return nullptr;
                }
                matched = parent;
            }
            return matched;
        }

        bool HasActiveChild(const AivLaunchContext& ctx, const TaskNode* parent, const TaskNode* target)
        {
            if (parent == nullptr || target == nullptr) {
                return false;
            }
            const NodeId targetId = target->GetNodeId();
            if (IsInactive(ctx, targetId)) {
                return false;
            }
            for (const TaskNode* child : parent->GetChildren()) {
                if (child != nullptr && child->GetNodeId() == targetId && !IsInactive(ctx, child->GetNodeId())) {
                    return true;
                }
            }
            return false;
        }

        bool MatchCpGmTemplateFromNode(TaskNode* start, const AivLaunchContext& ctx, CpGmIter& iter)
        {
            if (start == nullptr || IsInactive(ctx, start->GetNodeId())) {
                return false;
            }
            iter = CpGmIter{};
            iter[0] = start;
            if (!IsValidCpGmDataNode(iter[0], PIPE_MTE2, true)) {
                return false;
            }
            iter[1] = FindOnlyChildByType(ctx, iter[0], TaskType::AIV_SET_FLAG);
            iter[2] = FindOnlyWaitChildForSet(ctx, iter[1]);
            iter[3] = FindOnlyChildByType(ctx, iter[2], TaskType::TRANS_MEM);
            if (iter[3] == nullptr) {
                iter[3] = FindOnlyChildByType(ctx, iter[2], TaskType::REDUCE);
            }
            iter[4] = FindOnlyChildByType(ctx, iter[3], TaskType::AIV_SET_FLAG);
            iter[5] = FindOnlyWaitChildForSet(ctx, iter[4]);
            for (TaskNode* node : iter) {
                if (node == nullptr || IsInactive(ctx, node->GetNodeId())) {
                    return false;
                }
            }

            if (!HasActiveChild(ctx, iter[1], iter[5])) {
                return false;
            }

            if (FindOnlySetParentForWait(ctx, iter[2]) != iter[1]) {
                return false;
            }
            if (FindOnlySetParentForWait(ctx, iter[5]) != iter[4]) {
                return false;
            }

            if (!IsValidCpGmDataNode(iter[0], PIPE_MTE2, true)
                || !MatchPipeEventNode(iter[1], TaskType::AIV_SET_FLAG, PIPE_MTE2, PIPE_MTE2, PIPE_MTE3)
                || !MatchPipeEventNode(iter[2], TaskType::AIV_WAIT_FLAG, PIPE_MTE3, PIPE_MTE2, PIPE_MTE3)
                || !IsValidCpGmDataNode(iter[3], PIPE_MTE3, false)
                || !MatchPipeEventNode(iter[4], TaskType::AIV_SET_FLAG, PIPE_MTE3, PIPE_MTE3, PIPE_MTE2)
                || !MatchPipeEventNode(iter[5], TaskType::AIV_WAIT_FLAG, PIPE_MTE2, PIPE_MTE3, PIPE_MTE2)) {
                return false;
            }
            if (!SameSetWaitEvent(iter[1], iter[2]) || !SameSetWaitEvent(iter[4], iter[5])) {
                return false;
            }
            const TaskPosition& position = iter[0]->GetPosition();
            for (TaskNode* node : iter) {
                if (!SamePositionScope(position, node->GetPosition())) {
                    return false;
                }
            }
            return true;
        }

        bool ValidateCpGmRun(const CpGmLoopGather& gather)
        {
            if (gather.templateLoop.size() <= 1U) {
                return false;
            }
            const CpGmIter& first = gather.templateLoop.front();
            for (const CpGmIter& iter : gather.templateLoop) {
                if (iter[0]->GetType() != first[0]->GetType() || iter[3]->GetType() != first[3]->GetType()) {
                    return false;
                }
                if (!DataSliceShapeValid(iter) || !SameUbLayout(iter) || !SameExternalIdentityAsFirst(first, iter)) {
                    return false;
                }
                for (size_t idx : {1U, 2U, 4U, 5U}) {
                    if (!SameEventExceptTaskId(first[idx], iter[idx])) {
                        return false;
                    }
                }
                for (size_t idx : {0U, 3U}) {
                    if (!SameDataTypeAndOp(first[idx], iter[idx])) {
                        return false;
                    }
                }
            }
            return true;
        }

        bool CollectCpGmBlockTopo(const AivLaunchContext& ctx, uint32_t blockId, std::vector<NodeId>& topo)
        {
            topo.clear();
            std::set<NodeId> blockNodes;
            for (NodeId nodeId : ctx.internalNodeIds) {
                if (IsInactive(ctx, nodeId)) {
                    continue;
                }
                const TaskNode* node = ctx.graph->GetNode(nodeId);
                if (node == nullptr || node->GetPosition().blockId != blockId) {
                    continue;
                }
                const uint32_t pipe = node->GetPosition().pipe;
                if (pipe == PIPE_MTE2 || pipe == PIPE_MTE3) {
                    blockNodes.insert(nodeId);
                }
            }
            if (blockNodes.empty()) {
                return true;
            }

            std::map<NodeId, uint32_t> indegree;
            std::map<NodeId, std::vector<NodeId>> children;
            for (NodeId nodeId : blockNodes) {
                indegree[nodeId] = 0;
            }
            for (NodeId nodeId : blockNodes) {
                const TaskNode* node = ctx.graph->GetNode(nodeId);
                if (node == nullptr) {
                    return false;
                }
                for (const TaskNode* child : node->GetChildren()) {
                    if (child == nullptr) {
                        return false;
                    }
                    const NodeId childId = child->GetNodeId();
                    if (blockNodes.count(childId) == 0) {
                        continue;
                    }
                    children[nodeId].push_back(childId);
                    ++indegree[childId];
                }
            }

            auto lessByTaskId = [&ctx](NodeId lhs, NodeId rhs) {
                const TaskNode* lhsNode = ctx.graph->GetNode(lhs);
                const TaskNode* rhsNode = ctx.graph->GetNode(rhs);
                const uint32_t lhsTaskId
                    = lhsNode == nullptr ? std::numeric_limits<uint32_t>::max() : lhsNode->GetPosition().taskId;
                const uint32_t rhsTaskId
                    = rhsNode == nullptr ? std::numeric_limits<uint32_t>::max() : rhsNode->GetPosition().taskId;
                if (lhsTaskId != rhsTaskId) {
                    return lhsTaskId > rhsTaskId;
                }
                return lhs > rhs;
            };
            std::priority_queue<NodeId, std::vector<NodeId>, decltype(lessByTaskId)> ready(lessByTaskId);
            for (const auto& entry : indegree) {
                if (entry.second == 0) {
                    ready.push(entry.first);
                }
            }
            while (!ready.empty()) {
                const NodeId nodeId = ready.top();
                ready.pop();
                topo.push_back(nodeId);
                for (NodeId childId : children[nodeId]) {
                    auto iter = indegree.find(childId);
                    if (iter == indegree.end()) {
                        continue;
                    }
                    if (--iter->second == 0) {
                        ready.push(childId);
                    }
                }
            }
            return topo.size() == blockNodes.size();
        }

        uint32_t AllocateSyntheticTaskId(AivLaunchContext& ctx) { return ctx.nextSyntheticTaskId++; }

        TaskPosition MakeSyntheticPosition(AivLaunchContext& ctx, const TaskNode* source)
        {
            TaskPosition position = source == nullptr ? TaskPosition{} : source->GetPosition();
            position.taskId = AllocateSyntheticTaskId(ctx);
            return position;
        }

        std::unique_ptr<TaskNode> CloneSyntheticSyncNode(const TaskNode* source, const TaskPosition& position)
        {
            if (source == nullptr) {
                return nullptr;
            }
            if (source->GetType() == TaskType::AIV_SET_FLAG) {
                const auto* setFlag = dynamic_cast<const TaskAivSetFlag*>(source);
                if (setFlag == nullptr) {
                    return nullptr;
                }
                AivPipeEvent event = setFlag->GetEvent();
                event.taskId = position.taskId;
                return std::make_unique<TaskAivSetFlag>(event);
            }
            if (source->GetType() == TaskType::AIV_WAIT_FLAG) {
                const auto* waitFlag = dynamic_cast<const TaskAivWaitFlag*>(source);
                if (waitFlag == nullptr) {
                    return nullptr;
                }
                AivPipeEvent event = waitFlag->GetEvent();
                event.taskId = position.taskId;
                return std::make_unique<TaskAivWaitFlag>(event);
            }
            return nullptr;
        }

        std::vector<uint32_t> CollectSourceTaskIds(const CpGmLoopGather& gather, size_t column)
        {
            std::vector<uint32_t> taskIds;
            taskIds.reserve(gather.templateLoop.size());
            for (const CpGmIter& iter : gather.templateLoop) {
                taskIds.push_back(iter[column]->GetPosition().taskId);
            }
            return taskIds;
        }

        std::unique_ptr<TaskNode> BuildBatchNodeForColumn(const CpGmLoopGather& gather, size_t column)
        {
            std::vector<MemSlice> srcs;
            std::vector<MemSlice> dsts;
            srcs.reserve(gather.templateLoop.size());
            dsts.reserve(gather.templateLoop.size());

            const TaskType taskType = gather.templateLoop.front()[column]->GetType();
            uint8_t dataType = 0;
            uint8_t reduceOp = 0;
            for (const CpGmIter& iter : gather.templateLoop) {
                MemSlice src;
                MemSlice dst;
                uint8_t curDataType = 0;
                uint8_t curReduceOp = 0;
                if (!GetDataSlices(iter[column], src, dst, curDataType, curReduceOp)) {
                    return nullptr;
                }
                if (iter == gather.templateLoop.front()) {
                    dataType = curDataType;
                    reduceOp = curReduceOp;
                }
                srcs.push_back(src);
                dsts.push_back(dst);
            }

            std::vector<MemSlice> mergedSrcs;
            std::vector<MemSlice> mergedDsts;
            const bool srcIsUb = !srcs.empty() && srcs.front().memType == MemType::AIV_UB;
            const bool dstIsUb = !dsts.empty() && dsts.front().memType == MemType::AIV_UB;
            if (srcIsUb) {
                mergedSrcs = MergeMemSliceIntervals(srcs);
            } else if (!BuildContinuousMergedSlices(srcs, mergedSrcs)) {
                return nullptr;
            }
            if (dstIsUb) {
                mergedDsts = MergeMemSliceIntervals(dsts);
            } else if (!BuildContinuousMergedSlices(dsts, mergedDsts)) {
                return nullptr;
            }

            if (taskType == TaskType::TRANS_MEM) {
                auto batch = std::make_unique<TaskBatchTransMem>(ProtocolType::SDMA);
                batch->SetSrcMemSlices(std::move(srcs));
                batch->SetDstMemSlices(std::move(dsts));
                batch->SetMergedSrcMemSlices(std::move(mergedSrcs));
                batch->SetMergedDstMemSlices(std::move(mergedDsts));
                return batch;
            }
            if (taskType == TaskType::REDUCE) {
                auto batch = std::make_unique<TaskBatchReduce>(dataType, reduceOp, ProtocolType::SDMA);
                std::vector<std::vector<MemSlice>> srcGroups;
                srcGroups.reserve(srcs.size());
                for (const MemSlice& src : srcs) {
                    srcGroups.push_back({src});
                }
                std::vector<std::vector<MemSlice>> mergedSrcGroups;
                if (mergedDsts.size() == 1U && !mergedSrcs.empty()) {
                    mergedSrcGroups.push_back(mergedSrcs);
                } else {
                    for (const MemSlice& src : srcs) {
                        mergedSrcGroups.push_back({src});
                    }
                    mergedDsts = dsts;
                }
                batch->SetSrcMemSlices(std::move(srcGroups));
                batch->SetDstMemSlices(std::move(dsts));
                batch->SetMergedSrcMemSlices(std::move(mergedSrcGroups));
                batch->SetMergedDstMemSlices(std::move(mergedDsts));
                return batch;
            }
            return nullptr;
        }

        HcclResult AppendCpGmMergedNode(
            AivLaunchContext& ctx, CpGmLoopGather& gather, size_t column, std::unique_ptr<TaskNode> node,
            const TaskPosition& position)
        {
            if (node == nullptr) {
                return HCCL_E_INTERNAL;
            }
            NodeId nodeId = INVALID_NODE_ID;
            HcclResult ret = ctx.graph->AppendGeneratedNode(std::move(node), position, nodeId);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ctx.internalNodeIds.push_back(nodeId);
            gather.merged[column] = ctx.graph->GetNode(nodeId);
            ctx.cpGmMergeSourceTaskIds[nodeId] = CollectSourceTaskIds(gather, column);
            ++ctx.cpGmGeneratedNodeCount;
            return HCCL_SUCCESS;
        }

        bool CollectCpGmBoundaryNodes(
            const CpGmLoopGather& gather, std::set<NodeId>& runNodes,
            std::set<std::pair<NodeId, size_t>>& externalParentEdges,
            std::set<std::pair<size_t, NodeId>>& externalChildEdges)
        {
            runNodes.clear();
            externalParentEdges.clear();
            externalChildEdges.clear();
            for (const CpGmIter& iter : gather.templateLoop) {
                for (TaskNode* node : iter) {
                    runNodes.insert(node->GetNodeId());
                }
            }
            const NodeId firstNodeId = gather.templateLoop.front()[0]->GetNodeId();
            const NodeId firstMte3NodeId = gather.templateLoop.front()[2]->GetNodeId();
            const NodeId lastMte3NodeId = gather.templateLoop.back()[4]->GetNodeId();
            const NodeId lastMte2NodeId = gather.templateLoop.back()[5]->GetNodeId();
            for (const CpGmIter& iter : gather.templateLoop) {
                for (TaskNode* node : iter) {
                    for (TaskNode* parent : node->GetParents()) {
                        if (parent == nullptr) {
                            return false;
                        }
                        const NodeId parentId = parent->GetNodeId();
                        if (runNodes.count(parentId) == 0) {
                            const NodeId nodeId = node->GetNodeId();
                            if (nodeId == firstNodeId) {
                                externalParentEdges.insert({parentId, 0U});
                            } else if (nodeId == firstMte3NodeId) {
                                externalParentEdges.insert({parentId, 2U});
                            } else {
                                return false;
                            }
                        }
                    }
                    for (TaskNode* child : node->GetChildren()) {
                        if (child == nullptr) {
                            return false;
                        }
                        const NodeId childId = child->GetNodeId();
                        if (runNodes.count(childId) == 0) {
                            const NodeId nodeId = node->GetNodeId();
                            if (nodeId == lastMte3NodeId) {
                                externalChildEdges.insert({4U, childId});
                            } else if (nodeId == lastMte2NodeId) {
                                externalChildEdges.insert({5U, childId});
                            } else {
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        }

        HcclResult RemoveRunEdges(AivLaunchContext& ctx, const std::set<NodeId>& runNodes)
        {
            std::vector<std::pair<NodeId, NodeId>> edges;
            for (NodeId nodeId : runNodes) {
                TaskNode* node = ctx.graph->GetNode(nodeId);
                if (node == nullptr) {
                    return HCCL_E_PTR;
                }
                for (TaskNode* parent : node->GetParents()) {
                    if (parent != nullptr) {
                        edges.push_back({parent->GetNodeId(), nodeId});
                    }
                }
                for (TaskNode* child : node->GetChildren()) {
                    if (child != nullptr) {
                        edges.push_back({nodeId, child->GetNodeId()});
                    }
                }
            }
            std::sort(edges.begin(), edges.end());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
            for (const auto& edge : edges) {
                HcclResult ret = ctx.graph->RemoveEdge(edge.first, edge.second);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult ReplaceCpGmRun(AivLaunchContext& ctx, CpGmLoopGather& gather)
        {
            if (!ValidateCpGmRun(gather) || !CanMergeCpGmRun(ctx, gather)) {
                ++ctx.cpGmLoopSkipCount;
                return HCCL_SUCCESS;
            }

            std::set<NodeId> runNodes;
            std::set<std::pair<NodeId, size_t>> externalParentEdges;
            std::set<std::pair<size_t, NodeId>> externalChildEdges;
            if (!CollectCpGmBoundaryNodes(gather, runNodes, externalParentEdges, externalChildEdges)) {
                ++ctx.cpGmLoopSkipCount;
                return HCCL_SUCCESS;
            }

            HcclResult ret = AppendCpGmMergedNode(
                ctx, gather, 0, BuildBatchNodeForColumn(gather, 0),
                MakeSyntheticPosition(ctx, gather.templateLoop.front()[0]));
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            for (size_t column : {1U, 2U}) {
                const TaskPosition position = MakeSyntheticPosition(ctx, gather.templateLoop.front()[column]);
                ret = AppendCpGmMergedNode(
                    ctx, gather, column, CloneSyntheticSyncNode(gather.templateLoop.front()[column], position),
                    position);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            ret = AppendCpGmMergedNode(
                ctx, gather, 3, BuildBatchNodeForColumn(gather, 3),
                MakeSyntheticPosition(ctx, gather.templateLoop.front()[3]));
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            for (size_t column : {4U, 5U}) {
                const TaskPosition position = MakeSyntheticPosition(ctx, gather.templateLoop.front()[column]);
                ret = AppendCpGmMergedNode(
                    ctx, gather, column, CloneSyntheticSyncNode(gather.templateLoop.front()[column], position),
                    position);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }

            ret = RemoveRunEdges(ctx, runNodes);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            for (const auto& edge : externalParentEdges) {
                TaskNode* target = gather.merged[edge.second];
                if (target == nullptr) {
                    return HCCL_E_INTERNAL;
                }
                ret = AddEdge(ctx.graph, edge.first, target->GetNodeId());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            const std::array<std::pair<size_t, size_t>, 6> internalEdges{
                {{0U, 1U}, {1U, 5U}, {1U, 2U}, {2U, 3U}, {3U, 4U}, {4U, 5U}}};
            for (const auto& edge : internalEdges) {
                TaskNode* source = gather.merged[edge.first];
                TaskNode* target = gather.merged[edge.second];
                if (source == nullptr || target == nullptr) {
                    return HCCL_E_INTERNAL;
                }
                ret = AddEdge(ctx.graph, source->GetNodeId(), target->GetNodeId());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            for (const auto& edge : externalChildEdges) {
                TaskNode* source = gather.merged[edge.first];
                if (source == nullptr) {
                    return HCCL_E_INTERNAL;
                }
                ret = AddEdge(ctx.graph, source->GetNodeId(), edge.second);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            for (NodeId nodeId : runNodes) {
                ctx.inactiveNodeIds.insert(nodeId);
            }
            ++ctx.cpGmLoopMergeCount;
            ctx.cpGmMergedIterationCount += gather.templateLoop.size();
            ctx.cpGmMergedOriginalNodeCount += runNodes.size();
            ctx.cpGmInactiveNodeCount += runNodes.size();
            return HCCL_SUCCESS;
        }

        HcclResult FlushCpGmGather(AivLaunchContext& ctx, CpGmLoopGather& gather)
        {
            if (gather.templateLoop.size() > 1U) {
                HcclResult ret = ReplaceCpGmRun(ctx, gather);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            gather = CpGmLoopGather{};
            return HCCL_SUCCESS;
        }

        HcclResult MergeCpGM2GMLoops(AivLaunchContext& ctx)
        {
            ctx.dagNodeCountBeforeCpGmMerge = CountReachableActiveInternalNodes(ctx);
            for (const auto& block : ctx.snapshot.blocks) {
                std::vector<NodeId> topo;
                if (!CollectCpGmBlockTopo(ctx, block.blockIdx, topo)) {
                    HCCL_VM_WARN(
                        "Skip CpGM-to-GM merge for one block because the block DAG order could not "
                        "be determined, rankId={}, launchId={}, blockId={}, snapshotFile={}",
                        ctx.placeholder->GetRankId(), ctx.placeholder->GetLaunchIdx(), block.blockIdx,
                        ctx.snapshot.filePath);
                    continue;
                }

                CpGmLoopGather gather;
                std::set<NodeId> consumed; // 记录节点已经作为某个完整模板的一部分被识别过
                for (size_t index = 0; index < topo.size(); ++index) {
                    CpGmIter iter{};
                    TaskNode* node = ctx.graph->GetNode(topo[index]);
                    if (node == nullptr) {
                        continue;
                    }
                    if (consumed.count(node->GetNodeId()) != 0) {
                        continue;
                    }
                    if (MatchCpGmTemplateFromNode(node, ctx, iter)) {
                        gather.templateLoop.push_back(iter);
                        for (TaskNode* matchedNode : iter) {
                            consumed.insert(matchedNode->GetNodeId());
                        }
                        continue;
                    }
                    HcclResult ret = FlushCpGmGather(ctx, gather);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                HcclResult ret = FlushCpGmGather(ctx, gather);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            ctx.dagNodeCountAfterCpGmMerge = CountReachableActiveInternalNodes(ctx);
            HCCL_VM_INFO(
                "Finished CpGM-to-GM merge analysis:\n"
                "  ctx: rankId={}, launchId={}\n"
                "  tasks: taskJsonTotalTaskCount={}, dagNodeCountBeforeMerge={}, dagNodeCountAfterMerge={}\n"
                "  merge: cpGmLoopMergeCount={}, cpGmMergedIterationCount={}, cpGmMergedOriginalNodeCount={}\n"
                "  nodes: cpGmGeneratedNodeCount={}, cpGmInactiveNodeCount={}",
                ctx.placeholder->GetRankId(), ctx.placeholder->GetLaunchIdx(), ctx.taskJsonTotalTaskCount,
                ctx.dagNodeCountBeforeCpGmMerge, ctx.dagNodeCountAfterCpGmMerge, ctx.cpGmLoopMergeCount,
                ctx.cpGmMergedIterationCount, ctx.cpGmMergedOriginalNodeCount, ctx.cpGmGeneratedNodeCount,
                ctx.cpGmInactiveNodeCount);
            return HCCL_SUCCESS;
        }

        class DisjointSet {
        public:
            explicit DisjointSet(size_t size) : parent_(size), rank_(size, 0)
            {
                for (size_t i = 0; i < size; ++i) {
                    parent_[i] = i;
                }
            }

            size_t Find(size_t index)
            {
                if (parent_[index] != index) {
                    parent_[index] = Find(parent_[index]);
                }
                return parent_[index];
            }

            void Union(size_t lhs, size_t rhs)
            {
                size_t lhsRoot = Find(lhs);
                size_t rhsRoot = Find(rhs);
                if (lhsRoot == rhsRoot) {
                    return;
                }
                if (rank_[lhsRoot] < rank_[rhsRoot]) {
                    parent_[lhsRoot] = rhsRoot;
                } else if (rank_[lhsRoot] > rank_[rhsRoot]) {
                    parent_[rhsRoot] = lhsRoot;
                } else {
                    parent_[rhsRoot] = lhsRoot;
                    ++rank_[lhsRoot];
                }
            }

        private:
            std::vector<size_t> parent_;
            std::vector<uint8_t> rank_;
        };

        std::vector<MergeGroup> BuildPipeBarrierGroups(const AivLaunchContext& ctx)
        {
            std::vector<AivNodeRecord> barriers;
            for (const auto& entry : ctx.taskIdToNode) {
                if (entry.second.task != nullptr && entry.second.task->taskType == AivRuntimeTaskTypeV3::PIPE_BARRIER) {
                    barriers.push_back(entry.second);
                }
            }
            if (barriers.empty()) {
                return {};
            }

            std::map<uint32_t, size_t> taskIdToIndex;
            for (size_t i = 0; i < barriers.size(); ++i) {
                taskIdToIndex[barriers[i].task->taskId] = i;
            }
            DisjointSet dsu(barriers.size());
            for (size_t i = 0; i < barriers.size(); ++i) {
                dsu.Union(i, i);
                for (uint32_t peerTaskId : barriers[i].task->barrierGroupTaskIds) {
                    auto peerIter = taskIdToIndex.find(peerTaskId);
                    if (peerIter != taskIdToIndex.end()) {
                        dsu.Union(i, peerIter->second);
                    }
                }
            }

            std::map<size_t, MergeGroup> groupsByRoot;
            for (size_t i = 0; i < barriers.size(); ++i) {
                MergeGroup& group = groupsByRoot[dsu.Find(i)];
                group.memberNodeIds.push_back(barriers[i].nodeId);
                group.memberTaskIds.push_back(barriers[i].task->taskId);
                group.memberPipes.push_back(barriers[i].task->curPipe);
                group.topoOrder = group.memberNodeIds.size() == 1 ? barriers[i].order :
                                                                    std::min(group.topoOrder, barriers[i].order);
                group.blockId = barriers[i].task->blockId;
                group.pipeType = barriers[i].task->pipeType;
            }

            std::vector<MergeGroup> groups;
            groups.reserve(groupsByRoot.size());
            for (auto& entry : groupsByRoot) {
                std::sort(entry.second.memberTaskIds.begin(), entry.second.memberTaskIds.end());
                groups.push_back(std::move(entry.second));
            }
            std::sort(groups.begin(), groups.end(), [](const MergeGroup& lhs, const MergeGroup& rhs) {
                return lhs.topoOrder < rhs.topoOrder;
            });
            return groups;
        }

        std::vector<MergeGroup> BuildSyncAllGroups(const AivLaunchContext& ctx)
        {
            std::map<uint32_t, MergeGroup> groupsByRound;
            for (const auto& entry : ctx.taskIdToNode) {
                const AivNodeRecord& record = entry.second;
                if (record.task == nullptr || record.task->taskType != AivRuntimeTaskTypeV3::SYNC_ALL) {
                    continue;
                }
                MergeGroup& group = groupsByRound[record.task->syncRound];
                group.memberNodeIds.push_back(record.nodeId);
                group.memberTaskIds.push_back(record.task->taskId);
                group.memberPipes.push_back(record.task->curPipe);
                group.topoOrder
                    = group.memberNodeIds.size() == 1 ? record.order : std::min(group.topoOrder, record.order);
                group.syncRound = record.task->syncRound;
            }

            std::vector<MergeGroup> groups;
            groups.reserve(groupsByRound.size());
            for (auto& entry : groupsByRound) {
                std::sort(entry.second.memberTaskIds.begin(), entry.second.memberTaskIds.end());
                groups.push_back(std::move(entry.second));
            }
            std::sort(groups.begin(), groups.end(), [](const MergeGroup& lhs, const MergeGroup& rhs) {
                return lhs.topoOrder < rhs.topoOrder;
            });
            return groups;
        }

        HcclResult MergePipeBarrierGroups(AivLaunchContext& ctx)
        {
            const std::vector<MergeGroup> groups = BuildPipeBarrierGroups(ctx);
            for (const MergeGroup& group : groups) {
                if (group.memberNodeIds.empty()) {
                    continue;
                }
                std::set<uint32_t> memberPipes;
                for (NodeId memberNodeId : group.memberNodeIds) {
                    TaskNode* member = ctx.graph->GetNode(memberNodeId);
                    const auto* barrier = dynamic_cast<const TaskAivPipeBarrier*>(member);
                    if (barrier == nullptr) {
                        HCCL_VM_ERROR(
                            "{} One PipeBarrier group member is not a valid PipeBarrier node, "
                            "rankId={}, launchId={}, nodeId={}, snapshotFile={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), ctx.placeholder->GetRankId(),
                            ctx.placeholder->GetLaunchIdx(), memberNodeId, ctx.snapshot.filePath);
                        return HCCL_E_INTERNAL;
                    }
                    const AivBarrierInfo& memberInfo = barrier->GetInfo();
                    if (memberInfo.taskLoc.blockId != group.blockId) {
                        HCCL_VM_ERROR(
                            "{} One PipeBarrier group mixes tasks from different blocks, rankId={}, "
                            "launchId={}, expectedBlockId={}, actualBlockId={}, memberTaskId={}, snapshotFile={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), ctx.placeholder->GetRankId(),
                            ctx.placeholder->GetLaunchIdx(), group.blockId, memberInfo.taskLoc.blockId,
                            memberInfo.taskLoc.taskId, ctx.snapshot.filePath);
                        return HCCL_E_INTERNAL;
                    }
                    if (!memberPipes.insert(memberInfo.taskLoc.pipe).second) {
                        HCCL_VM_ERROR(
                            "{} One PipeBarrier group contains two tasks from the same pipe, "
                            "rankId={}, launchId={}, blockId={}, duplicatedPipeId={}, snapshotFile={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), ctx.placeholder->GetRankId(),
                            ctx.placeholder->GetLaunchIdx(), group.blockId, memberInfo.taskLoc.pipe,
                            ctx.snapshot.filePath);
                        return HCCL_E_INTERNAL;
                    }
                }
                if (group.pipeType == PIPE_ALL) {
                    for (uint32_t pipe = 0; pipe < AIV_PIPE_NUM; ++pipe) {
                        if (memberPipes.count(pipe) != 0) {
                            continue;
                        }
                        std::ostringstream existingPipeIds;
                        existingPipeIds << "[";
                        for (size_t i = 0; i < group.memberPipes.size(); ++i) {
                            if (i != 0) {
                                existingPipeIds << ",";
                            }
                            existingPipeIds << group.memberPipes[i];
                        }
                        existingPipeIds << "]";
                        std::ostringstream memberTaskIds;
                        memberTaskIds << "[";
                        for (size_t i = 0; i < group.memberTaskIds.size(); ++i) {
                            if (i != 0) {
                                memberTaskIds << ",";
                            }
                            memberTaskIds << group.memberTaskIds[i];
                        }
                        memberTaskIds << "]";
                        HCCL_VM_ERROR(
                            "{} One PIPE_ALL PipeBarrier is incomplete because one pipe is missing, "
                            "rankId={}, launchId={}, blockId={}, missingPipeId={}, existingPipeIds={}, "
                            "memberTaskIds={}, "
                            "snapshotFile={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_MEMBER_MISSING), ctx.placeholder->GetRankId(),
                            ctx.placeholder->GetLaunchIdx(), group.blockId, pipe, existingPipeIds.str(),
                            memberTaskIds.str(), ctx.snapshot.filePath);
                        return HCCL_E_INTERNAL;
                    }
                }
                AivBarrierInfo info;
                const uint32_t mergedTaskId
                    = group.memberTaskIds.empty() ? std::numeric_limits<uint32_t>::max() : group.memberTaskIds.front();
                info.taskLoc = MakeAivPosition(
                    ctx.placeholder->GetRankId(), ctx.placeholder->GetLaunchIdx(), group.blockId,
                    std::numeric_limits<uint32_t>::max(), mergedTaskId, ctx.placeholder->GetOperatorId());
                info.pipeType = group.pipeType;
                info.merged = true;
                info.memberNodeIds = group.memberNodeIds;
                info.memberTaskIds = group.memberTaskIds;
                info.memberPipes = group.memberPipes;

                NodeId mergeNodeId = INVALID_NODE_ID;
                const TaskPosition position = MakeAivPosition(
                    ctx.placeholder->GetRankId(), ctx.placeholder->GetLaunchIdx(), group.blockId,
                    std::numeric_limits<uint32_t>::max(), mergedTaskId, ctx.placeholder->GetOperatorId());
                HcclResult ret = ctx.graph->AppendGeneratedNode(
                    std::make_unique<TaskAivPipeBarrier>(std::move(info)), position, mergeNodeId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ctx.internalNodeIds.push_back(mergeNodeId);
                for (NodeId memberNodeId : group.memberNodeIds) {
                    ret = RewireGroupMember(ctx.graph, memberNodeId, mergeNodeId);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                ++ctx.pipeBarrierMergeCount;
            }
            return HCCL_SUCCESS;
        }

        HcclResult MergeSyncAllGroups(AivLaunchContext& ctx)
        {
            const std::vector<MergeGroup> groups = BuildSyncAllGroups(ctx);
            const size_t expectedMemberCount = ctx.snapshot.blocks.size() * AIV_PIPE_NUM;
            for (const MergeGroup& group : groups) {
                if (group.memberNodeIds.empty()) {
                    continue;
                }
                if (expectedMemberCount != 0 && group.memberNodeIds.size() != expectedMemberCount) {
                    HCCL_VM_ERROR(
                        "{} One SyncAll group is incomplete because it does not contain one member "
                        "for every expected block/pipe pair, rankId={}, launchId={}, syncRound={}, actualCount={}, "
                        "expectedCount={}, snapshotFile={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_MEMBER_MISSING), ctx.placeholder->GetRankId(),
                        ctx.placeholder->GetLaunchIdx(), group.syncRound, group.memberNodeIds.size(),
                        expectedMemberCount, ctx.snapshot.filePath);
                    return HCCL_E_INTERNAL;
                }
                std::set<std::pair<uint32_t, uint32_t>> members;
                for (NodeId memberNodeId : group.memberNodeIds) {
                    TaskNode* member = ctx.graph->GetNode(memberNodeId);
                    const auto* syncAll = dynamic_cast<const TaskAivSyncAll*>(member);
                    if (syncAll == nullptr) {
                        HCCL_VM_ERROR(
                            "{} One SyncAll group member is not a valid SyncAll node, rankId={}, "
                            "launchId={}, syncRound={}, nodeId={}, snapshotFile={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), ctx.placeholder->GetRankId(),
                            ctx.placeholder->GetLaunchIdx(), group.syncRound, memberNodeId, ctx.snapshot.filePath);
                        return HCCL_E_INTERNAL;
                    }
                    const TaskPosition& taskLoc = syncAll->GetInfo().taskLoc;
                    if (!members.insert({taskLoc.blockId, taskLoc.pipe}).second) {
                        HCCL_VM_ERROR(
                            "{} One SyncAll group contains duplicate members for the same "
                            "block/pipe pair, rankId={}, launchId={}, syncRound={}, blockId={}, pipeId={}, "
                            "snapshotFile={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID), ctx.placeholder->GetRankId(),
                            ctx.placeholder->GetLaunchIdx(), group.syncRound, taskLoc.blockId, taskLoc.pipe,
                            ctx.snapshot.filePath);
                        return HCCL_E_INTERNAL;
                    }
                }
                for (const auto& block : ctx.snapshot.blocks) {
                    for (uint32_t pipe = 0; pipe < AIV_PIPE_NUM; ++pipe) {
                        if (members.count({block.blockIdx, pipe}) != 0) {
                            continue;
                        }
                        HCCL_VM_ERROR(
                            "{} One SyncAll group is incomplete because a block/pipe member is "
                            "missing, rankId={}, launchId={}, syncRound={}, missingBlockId={}, missingPipeId={}, "
                            "snapshotFile={}",
                            MakeErrorCodeText(ErrorCode::GRAPH_MEMBER_MISSING), ctx.placeholder->GetRankId(),
                            ctx.placeholder->GetLaunchIdx(), group.syncRound, block.blockIdx, pipe,
                            ctx.snapshot.filePath);
                        return HCCL_E_INTERNAL;
                    }
                }
                AivSyncAllInfo info;
                const uint32_t mergedTaskId
                    = group.memberTaskIds.empty() ? std::numeric_limits<uint32_t>::max() : group.memberTaskIds.front();
                info.taskLoc = MakeAivPosition(
                    ctx.placeholder->GetRankId(), ctx.placeholder->GetLaunchIdx(), std::numeric_limits<uint32_t>::max(),
                    std::numeric_limits<uint32_t>::max(), mergedTaskId, ctx.placeholder->GetOperatorId());
                info.syncRound = group.syncRound;
                info.merged = true;
                info.memberNodeIds = group.memberNodeIds;
                info.memberTaskIds = group.memberTaskIds;
                info.memberPipes = group.memberPipes;

                NodeId mergeNodeId = INVALID_NODE_ID;
                const TaskPosition position = MakeAivPosition(
                    ctx.placeholder->GetRankId(), ctx.placeholder->GetLaunchIdx(), std::numeric_limits<uint32_t>::max(),
                    std::numeric_limits<uint32_t>::max(), mergedTaskId, ctx.placeholder->GetOperatorId());
                HcclResult ret = ctx.graph->AppendGeneratedNode(
                    std::make_unique<TaskAivSyncAll>(std::move(info)), position, mergeNodeId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ctx.internalNodeIds.push_back(mergeNodeId);
                for (NodeId memberNodeId : group.memberNodeIds) {
                    ret = RewireGroupMember(ctx.graph, memberNodeId, mergeNodeId);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                ++ctx.syncAllMergeCount;
            }
            return HCCL_SUCCESS;
        }

        HcclResult ConnectCurrentTailsToEnd(AivLaunchContext& ctx)
        {
            if (ctx.headNodeId == INVALID_NODE_ID || ctx.endNodeId == INVALID_NODE_ID) {
                return HCCL_E_PARA;
            }

            std::set<NodeId> internalSet(ctx.internalNodeIds.begin(), ctx.internalNodeIds.end());
            std::set<NodeId> reachableInternalNodes;
            std::queue<NodeId> nodeQueue;
            nodeQueue.push(ctx.headNodeId);
            reachableInternalNodes.insert(ctx.headNodeId);
            while (!nodeQueue.empty()) {
                const NodeId nodeId = nodeQueue.front();
                nodeQueue.pop();
                TaskNode* node = ctx.graph->GetNode(nodeId);
                if (node == nullptr) {
                    return HCCL_E_PTR;
                }
                for (TaskNode* child : node->GetChildren()) {
                    if (child == nullptr) {
                        return HCCL_E_PTR;
                    }
                    const NodeId childId = child->GetNodeId();
                    if (internalSet.count(childId) == 0 || IsInactive(ctx, childId)
                        || reachableInternalNodes.count(childId) != 0) {
                        continue;
                    }
                    reachableInternalNodes.insert(childId);
                    nodeQueue.push(childId);
                }
            }

            std::set<NodeId> tails;
            for (NodeId nodeId : reachableInternalNodes) {
                TaskNode* node = ctx.graph->GetNode(nodeId);
                if (node == nullptr) {
                    return HCCL_E_PTR;
                }
                if (node->GetType() == TaskType::START || node->GetType() == TaskType::END) {
                    continue;
                }
                bool hasInternalChild = false;
                for (TaskNode* child : node->GetChildren()) {
                    if (child != nullptr && internalSet.count(child->GetNodeId()) != 0) {
                        hasInternalChild = true;
                        break;
                    }
                }
                if (!hasInternalChild) {
                    tails.insert(nodeId);
                }
            }

            if (tails.empty()) {
                return AddEdge(ctx.graph, ctx.headNodeId, ctx.endNodeId);
            }
            for (NodeId tailNodeId : tails) {
                HcclResult ret = AddEdge(ctx.graph, tailNodeId, ctx.endNodeId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult ReconnectPlaceholder(AivLaunchContext& ctx)
        {
            TaskNode* placeholder = ctx.graph->GetNode(ctx.placeholder->GetNodeId());
            if (placeholder == nullptr) {
                return HCCL_E_PTR;
            }
            std::vector<TaskNode*> originalChildren = placeholder->GetChildren();
            HcclResult ret = AddEdge(ctx.graph, placeholder->GetNodeId(), ctx.headNodeId);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            for (TaskNode* child : originalChildren) {
                if (child == nullptr || child->GetNodeId() == ctx.headNodeId) {
                    continue;
                }
                ret = ctx.graph->RemoveEdge(placeholder->GetNodeId(), child->GetNodeId());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = AddEdge(ctx.graph, ctx.endNodeId, child->GetNodeId());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult ExpandOneAivGraph(
            TaskGraphGeneratorV3* graph, StorageManager* storage, TaskAivGraph* aivGraph,
            AivGraphGenerateOutputV3& output)
        {
            if (graph == nullptr || storage == nullptr || aivGraph == nullptr) {
                return HCCL_E_PTR;
            }

            AivSnapshotJsonLoaderV3 loader;
            AivLaunchContext ctx;
            ctx.graph = graph;
            ctx.storage = storage;
            ctx.placeholder = aivGraph;
            std::string errorMessage;
            HcclResult ret = loader.LoadByRankAndLaunch(
                aivGraph->GetRankId(), aivGraph->GetLaunchIdx(), ctx.snapshot, errorMessage);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Failed to load the AIV runtime snapshot file, rankId={}, launchId={}, "
                    "ret={}, loaderMessage={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_RESOURCE_NOT_FOUND), aivGraph->GetRankId(),
                    aivGraph->GetLaunchIdx(), static_cast<uint32_t>(ret), errorMessage);
                return ret;
            }
            ret = ValidateSnapshot(ctx);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ret = UpdateAivBufferSize(ctx.snapshot.ubBufferSize, "ubBufferSize", ctx, g_checkerAivUbBufferSize);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ret = UpdateAivBufferSize(ctx.snapshot.aivCommInfoSize, "aivCommInfoSize", ctx, g_checkerAivCommInfoSize);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ctx.taskJsonTotalTaskCount = CountAivSnapshotTasks(ctx.snapshot);
            const uint32_t maxTaskId = FindMaxTaskId(ctx.snapshot);
            ctx.nextSyntheticTaskId = maxTaskId == std::numeric_limits<uint32_t>::max() ?
                                          std::numeric_limits<uint32_t>::max() :
                                          maxTaskId + 1U;

            const TaskPosition basePosition = aivGraph->GetPosition();
            ret = graph->AppendGeneratedNode(
                std::make_unique<TaskStart>(BoundaryType::AIV_SUB_GRAPH), basePosition, ctx.headNodeId);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ctx.internalNodeIds.push_back(ctx.headNodeId);

            std::vector<NodeId> tailNodes;
            ret = BuildPipeSkeleton(ctx, tailNodes);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ret = AddSetWaitFlagEdges(ctx);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ret = MergeCpGM2GMLoops(ctx);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ret = MergePipeBarrierGroups(ctx);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ret = MergeSyncAllGroups(ctx);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            ret = graph->AppendGeneratedNode(
                std::make_unique<TaskEnd>(BoundaryType::AIV_SUB_GRAPH), basePosition, ctx.endNodeId);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ctx.internalNodeIds.push_back(ctx.endNodeId);
            ret = ConnectCurrentTailsToEnd(ctx);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ret = ReconnectPlaceholder(ctx);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            output.internalNodeCount += ctx.internalNodeIds.size();
            output.setWaitEdgeCount += ctx.setWaitEdgeCount;
            output.pipeBarrierMergeCount += ctx.pipeBarrierMergeCount;
            output.syncAllMergeCount += ctx.syncAllMergeCount;
            output.taskJsonTotalTaskCount += ctx.taskJsonTotalTaskCount;
            output.dagNodeCountBeforeCpGmMerge += ctx.dagNodeCountBeforeCpGmMerge;
            output.dagNodeCountAfterCpGmMerge += ctx.dagNodeCountAfterCpGmMerge;
            output.cpGmLoopMergeCount += ctx.cpGmLoopMergeCount;
            output.cpGmMergedIterationCount += ctx.cpGmMergedIterationCount;
            output.cpGmMergedOriginalNodeCount += ctx.cpGmMergedOriginalNodeCount;
            output.cpGmGeneratedNodeCount += ctx.cpGmGeneratedNodeCount;
            output.cpGmInactiveNodeCount += ctx.cpGmInactiveNodeCount;
            return HCCL_SUCCESS;
        }

        bool
        IsExecutableForFlagMatch(const TaskGraphGeneratorV3* graph, NodeId nodeId, const std::set<NodeId>& executed)
        {
            if (graph == nullptr) {
                return false;
            }
            const TaskNode* node = graph->GetNode(nodeId);
            if (node == nullptr) {
                return false;
            }
            for (const TaskNode* parent : node->GetParents()) {
                if (parent == nullptr) {
                    return false;
                }
                const NodeId parentId = parent->GetNodeId();
                if (parentId == MAIN_START_NODE_ID) {
                    continue;
                }
                if (executed.count(parentId) == 0) {
                    return false;
                }
            }
            return true;
        }

        HcclResult EnqueueChildrenForFlagMatch(
            const TaskGraphGeneratorV3* graph, NodeId nodeId, std::deque<NodeId>& readyQueue, std::set<NodeId>& queued)
        {
            const TaskNode* node = graph->GetNode(nodeId);
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            for (const TaskNode* child : node->GetChildren()) {
                if (child == nullptr) {
                    return HCCL_E_PTR;
                }
                const NodeId childId = child->GetNodeId();
                if (queued.insert(childId).second) {
                    readyQueue.push_back(childId);
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult
        MatchSendRecvFlagEdges(TaskGraphGeneratorV3* graph, const std::string& snapshotFile, size_t& matchedEdgeCount)
        {
            matchedEdgeCount = 0;
            if (graph == nullptr || graph->GetMainStartNode() == nullptr) {
                return HCCL_E_PTR;
            }

            std::map<FlagCellKey, FlagCellState> flagStates;
            std::deque<NodeId> readyQueue;
            std::set<NodeId> queued;
            std::set<NodeId> executed;

            for (const TaskNode* child : graph->GetMainStartNode()->GetChildren()) {
                if (child == nullptr) {
                    return HCCL_E_PTR;
                }
                const NodeId childId = child->GetNodeId();
                queued.insert(childId);
                readyQueue.push_back(childId);
            }

            size_t unmatchedCnt = 0;
            while (!readyQueue.empty()) {
                if (unmatchedCnt >= readyQueue.size()) {
                    const TaskNode* node = graph->GetNode(readyQueue.front());
                    std::string flagCellStateText = "{}";
                    if (node != nullptr && node->GetType() == TaskType::AIV_RECV_FLAG) {
                        const auto* recvFlag = dynamic_cast<const TaskAivRecvFlag*>(node);
                        if (recvFlag != nullptr) {
                            const auto stateIter = flagStates.find(MakeFlagCellKey(recvFlag->GetFlag()));
                            if (stateIter != flagStates.end()) {
                                std::ostringstream os;
                                os << "{currentValue=" << stateIter->second.currentValue
                                   << ", producerCount=" << stateIter->second.producerNodes.size()
                                   << ", producerNodeIds=[";
                                for (size_t i = 0; i < stateIter->second.producerNodes.size(); ++i) {
                                    if (i != 0) {
                                        os << ",";
                                    }
                                    os << stateIter->second.producerNodes[i];
                                }
                                os << "]}";
                                flagCellStateText = os.str();
                            }
                        }
                    }
                    HCCL_VM_ERROR(
                        "{} AIV SendFlag/RecvFlag matching is stuck. Some RecvFlag tasks are still "
                        "blocked, but no matching SendFlag value has become available, firstBlockedRecvFlagNode={}, "
                        "blockedRecvFlagNodeCount={}, currentFlagCellState={}, snapshotFile={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_DEADLOCK), node == nullptr ? "node=null" : node->Describe(),
                        readyQueue.size(), flagCellStateText, snapshotFile);
                    return HCCL_E_INTERNAL;
                }

                const NodeId nodeId = readyQueue.front();
                readyQueue.pop_front();
                queued.erase(nodeId);
                if (executed.count(nodeId) != 0) {
                    unmatchedCnt = 0;
                    continue;
                }
                if (!IsExecutableForFlagMatch(graph, nodeId, executed)) {
                    readyQueue.push_back(nodeId);
                    queued.insert(nodeId);
                    ++unmatchedCnt;
                    continue;
                }

                TaskNode* node = graph->GetNode(nodeId);
                if (node == nullptr) {
                    return HCCL_E_PTR;
                }
                if (node->GetType() == TaskType::AIV_SEND_FLAG) {
                    const auto* sendFlag = dynamic_cast<const TaskAivSendFlag*>(node);
                    if (sendFlag == nullptr) {
                        return HCCL_E_PTR;
                    }
                    FlagCellState& state = flagStates[MakeFlagCellKey(sendFlag->GetFlag())];
                    if (state.currentValue == sendFlag->GetFlag().value && !state.producerNodes.empty()) {
                        HCCL_VM_WARN(
                            "The same flag cell received the same SendFlag value again before any "
                            "new RecvFlag consumed it, node={}",
                            node == nullptr ? "node=null" : node->Describe());
                    }
                    state.currentValue = sendFlag->GetFlag().value;
                    state.producerNodes.clear();
                    state.producerNodes.push_back(nodeId);
                } else if (node->GetType() == TaskType::AIV_RECV_FLAG) {
                    const auto* recvFlag = dynamic_cast<const TaskAivRecvFlag*>(node);
                    if (recvFlag == nullptr) {
                        return HCCL_E_PTR;
                    }
                    FlagCellState& state = flagStates[MakeFlagCellKey(recvFlag->GetFlag())];
                    if (state.currentValue != recvFlag->GetFlag().value) {
                        readyQueue.push_back(nodeId);
                        queued.insert(nodeId);
                        ++unmatchedCnt;
                        continue;
                    }
                    for (NodeId producerNodeId : state.producerNodes) {
                        HcclResult ret = AddEdge(graph, producerNodeId, nodeId);
                        if (ret != HCCL_SUCCESS) {
                            return ret;
                        }
                        ++matchedEdgeCount;
                    }
                }

                executed.insert(nodeId);
                HcclResult ret = EnqueueChildrenForFlagMatch(graph, nodeId, readyQueue, queued);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                unmatchedCnt = 0;
            }
            return HCCL_SUCCESS;
        }
    } // namespace

    HcclResult ExpandAivGraphsV3(const AivGraphsGenerateInputV3& input, AivGraphGenerateOutputV3& output)
    {
        const auto start = AivExpandClock::now();
        output = AivGraphGenerateOutputV3{};
        if (input.graph == nullptr || input.storage == nullptr) {
            return HCCL_E_PTR;
        }

        for (TaskAivGraph* aivGraph : input.aivGraphs) {
            HcclResult ret = ExpandOneAivGraph(input.graph, input.storage, aivGraph, output);
            if (ret != HCCL_SUCCESS) {
                output.expandNs = ElapsedNs(start, AivExpandClock::now());
                return ret;
            }
        }

        const std::string snapshotFileHint
            = input.aivGraphs.empty() ? "aiv_snapshot_unavailable" : "multiple_aiv_snapshots";
        HcclResult ret = MatchSendRecvFlagEdges(input.graph, snapshotFileHint, output.sendRecvEdgeCount);
        output.expandNs = ElapsedNs(start, AivExpandClock::now());
        return ret;
    }

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
