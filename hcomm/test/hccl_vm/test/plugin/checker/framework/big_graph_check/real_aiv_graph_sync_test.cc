/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "task_graph_sync_conflict_v3.h"

namespace HcclSim {
namespace {
    using namespace TaskGraphGeneratorV3;

    class RealAivGraphSyncTest : public testing::Test {
    protected:
        struct Graph {
            TaskStart* mainStart{nullptr};
            TaskAivSetFlag* eventSet0{nullptr};
            TaskAivWaitFlag* eventWait0{nullptr};
            TaskAivSendFlag* flagSend0{nullptr};
            TaskAivRecvFlag* flagRecv0Remote{nullptr};
        };

        template <typename T>
        T* AddNode(std::unique_ptr<T> node, const TaskPosition& position)
        {
            T* raw = node.get();
            raw->SetNodeId(static_cast<NodeId>(nodes_.size()));
            raw->SetPosition(position);
            nodes_.emplace_back(std::move(node));
            return raw;
        }

        TaskStart* AddMainStart()
        {
            auto node = std::make_unique<TaskStart>(BoundaryType::MAIN_GRAPH);
            node->SetNodeId(MAIN_START_NODE_ID);
            TaskStart* raw = node.get();
            nodes_.emplace_back(std::move(node));
            return raw;
        }

        static TaskPosition Position(
            OperatorId operatorId, RankId rankId, StreamId streamId, QueueId queueId, uint32_t blockId = 0,
            uint32_t taskId = 0)
        {
            TaskPosition position;
            position.operatorId = operatorId;
            position.rankId = rankId;
            position.streamId = streamId;
            position.queueId = queueId;
            position.launchIdx = 1;
            position.blockId = blockId;
            position.taskId = taskId;
            return position;
        }

        static TaskPosition SubGraphPosition(OperatorId operatorId, RankId rankId)
        {
            return Position(operatorId, rankId, 0, INVALID_QUEUE_ID);
        }

        static void AddEdge(TaskNode* parent, TaskNode* child)
        {
            ASSERT_NE(parent, nullptr);
            ASSERT_NE(child, nullptr);
            ASSERT_TRUE(parent->AddChild(child));
            ASSERT_TRUE(child->AddParent(parent));
        }

        static AivPipeEvent
        MakeEvent(RankId rankId, uint32_t blockId, uint32_t srcPipe, uint32_t dstPipe, int32_t eventId)
        {
            AivPipeEvent event;
            event.rankId = rankId;
            event.launchIdx = 1;
            event.blockId = blockId;
            event.curPipe = dstPipe;
            event.srcPipe = srcPipe;
            event.dstPipe = dstPipe;
            event.eventId = eventId;
            return event;
        }

        static AivFlagSync MakeFlag(RankId currentRank, RankId ownerRank, uint64_t offset, int32_t value)
        {
            AivFlagSync flag;
            flag.currentRank = currentRank;
            flag.flagOwnerRank = ownerRank;
            flag.launchIdx = 1;
            flag.blockId = 0;
            flag.curPipe = 0;
            flag.taskId = 3;
            flag.commInfoOffset = offset;
            flag.value = value;
            return flag;
        }

        TaskAivPipeBarrier* AddBarrier(const TaskPosition& position, uint32_t taskId)
        {
            AivBarrierInfo info;
            info.taskLoc = position;
            info.pipeType = 3;
            info.merged = true;
            info.memberTaskIds = {taskId, taskId + 1, taskId + 2};
            return AddNode(std::make_unique<TaskAivPipeBarrier>(std::move(info)), position);
        }

        TaskAivSyncAll* AddSyncAll(const TaskPosition& position, uint32_t syncRound)
        {
            AivSyncAllInfo info;
            info.taskLoc = position;
            info.syncRound = syncRound;
            info.merged = true;
            info.memberTaskIds = {18, 19, 20, 267, 268, 269};
            return AddNode(std::make_unique<TaskAivSyncAll>(std::move(info)), position);
        }

        TaskBatchTransMem* AddBatchCopy(const TaskPosition& position, MemSlice src, MemSlice dst)
        {
            auto node = std::make_unique<TaskBatchTransMem>(ProtocolType::SDMA);
            node->AddSrcMemSlice(src);
            node->AddDstMemSlice(dst);
            node->AddMergedSrcMemSlice(src);
            node->AddMergedDstMemSlice(dst);
            return AddNode(std::move(node), position);
        }

        static MemSlice Slice(RankId rankId, MemType type, uint64_t offset, uint64_t len = 0x400)
        {
            MemSlice slice;
            slice.rankId = rankId;
            slice.memType = type;
            slice.offset = offset;
            slice.len = len;
            return slice;
        }

        TaskAivGraph* AddAivGraph(RankId rankId, const TaskPosition& position)
        {
            return AddNode(std::make_unique<TaskAivGraph>(rankId, 1, 0), position);
        }

        TaskEnd* AddSubGraphEnd(const TaskPosition& position)
        {
            return AddNode(std::make_unique<TaskEnd>(BoundaryType::AIV_SUB_GRAPH), position);
        }

        TaskAivPipeBarrier* BuildEventPipeline(
            TaskStart* subStart, RankId rankId, OperatorId operatorId, uint32_t blockId, QueueId queueId,
            TaskAivSetFlag** firstSet, TaskAivWaitFlag** firstWait)
        {
            const TaskPosition pipePosition = Position(operatorId, rankId, INVALID_STREAM_ID, queueId, blockId, 0);
            TaskAivPipeBarrier* initialBarrier = AddBarrier(pipePosition, blockId * 249);
            TaskAivSetFlag* set0 = AddNode(
                std::make_unique<TaskAivSetFlag>(MakeEvent(rankId, blockId, 1, 2, 0)),
                Position(operatorId, rankId, INVALID_STREAM_ID, queueId, blockId, 1));
            TaskAivWaitFlag* wait0 = AddNode(
                std::make_unique<TaskAivWaitFlag>(MakeEvent(rankId, blockId, 1, 2, 0)),
                Position(operatorId + 1, rankId, INVALID_STREAM_ID, queueId, blockId, 2));
            TaskBatchTransMem* copy = AddBatchCopy(
                Position(operatorId + 1, rankId, INVALID_STREAM_ID, queueId + 1, blockId, 3),
                Slice(rankId, MemType::AIV_COMM, 0xa00000 + blockId * 0x2aaaa), Slice(rankId, MemType::AIV_UB, 0));
            TaskAivSetFlag* set1 = AddNode(
                std::make_unique<TaskAivSetFlag>(MakeEvent(rankId, blockId, 2, 1, 1)),
                Position(operatorId + 1, rankId, INVALID_STREAM_ID, queueId, blockId, 4));
            TaskAivWaitFlag* wait1 = AddNode(
                std::make_unique<TaskAivWaitFlag>(MakeEvent(rankId, blockId, 2, 1, 1)),
                Position(operatorId + 2, rankId, INVALID_STREAM_ID, queueId + 1, blockId, 5));
            TaskBatchTransMem* copyOut = AddBatchCopy(
                Position(operatorId + 2, rankId, INVALID_STREAM_ID, queueId + 1, blockId, 6),
                Slice(rankId, MemType::AIV_UB, 0), Slice(rankId, MemType::AIV_COMM, 0x100000));
            TaskAivPipeBarrier* finalBarrier = AddBarrier(
                Position(operatorId + 2, rankId, INVALID_STREAM_ID, queueId, blockId, 7), blockId * 247 + 7);

            AddEdge(subStart, initialBarrier);
            AddEdge(initialBarrier, set0);
            AddEdge(set0, wait0);
            AddEdge(wait0, copy);
            AddEdge(copy, set1);
            AddEdge(set1, wait1);
            AddEdge(wait1, copyOut);
            AddEdge(copyOut, finalBarrier);
            if (firstSet != nullptr) {
                *firstSet = set0;
            }
            if (firstWait != nullptr) {
                *firstWait = wait0;
            }
            return finalBarrier;
        }

        Graph BuildRealGraph()
        {
            Graph graph;
            graph.mainStart = AddMainStart();

            TaskAivGraph* rank0Graph = AddAivGraph(0, SubGraphPosition(0, 0));
            TaskAivGraph* rank1Graph = AddAivGraph(1, SubGraphPosition(0, 1));
            TaskStart* rank0Start
                = AddNode(std::make_unique<TaskStart>(BoundaryType::AIV_SUB_GRAPH), SubGraphPosition(0, 0));
            TaskStart* rank1Start
                = AddNode(std::make_unique<TaskStart>(BoundaryType::AIV_SUB_GRAPH), SubGraphPosition(0, 1));
            AddEdge(graph.mainStart, rank0Graph);
            AddEdge(graph.mainStart, rank1Graph);
            AddEdge(rank0Graph, rank0Start);
            AddEdge(rank1Graph, rank1Start);

            // The three merged pipe queues in the example use queue IDs 3072, 3075 and 3078.
            std::vector<TaskAivPipeBarrier*> rank0Pipelines;
            std::vector<TaskAivPipeBarrier*> rank1Pipelines;
            for (uint32_t blockId = 0; blockId < 3; ++blockId) {
                TaskAivSetFlag* set = nullptr;
                TaskAivWaitFlag* wait = nullptr;
                rank0Pipelines.push_back(
                    BuildEventPipeline(rank0Start, 0, 0, blockId, 3072 + blockId * 3, &set, &wait));
                if (blockId == 0) {
                    graph.eventSet0 = set;
                    graph.eventWait0 = wait;
                }
                rank1Pipelines.push_back(
                    BuildEventPipeline(rank1Start, 1, 0, blockId, 3072 + blockId * 3, nullptr, nullptr));
            }

            // Each owner has a SendFlag fanning out to local and remote RecvFlag nodes.
            graph.flagSend0 = AddNode(
                std::make_unique<TaskAivSendFlag>(MakeFlag(0, 0, 0x900000, 2)),
                Position(0, 0, INVALID_STREAM_ID, 3072, 0, 10));
            TaskAivRecvFlag* flagRecv0Local = AddNode(
                std::make_unique<TaskAivRecvFlag>(MakeFlag(0, 0, 0x900000, 2)),
                Position(1, 0, INVALID_STREAM_ID, 3072, 0, 11));
            graph.flagRecv0Remote = AddNode(
                std::make_unique<TaskAivRecvFlag>(MakeFlag(1, 0, 0x900000, 2)),
                Position(1, 1, INVALID_STREAM_ID, 3072, 0, 12));
            TaskAivSendFlag* flagSend1 = AddNode(
                std::make_unique<TaskAivSendFlag>(MakeFlag(1, 1, 0x900000, 2)),
                Position(0, 1, INVALID_STREAM_ID, 3072, 0, 13));
            TaskAivRecvFlag* flagRecv1Local = AddNode(
                std::make_unique<TaskAivRecvFlag>(MakeFlag(1, 1, 0x900000, 2)),
                Position(1, 1, INVALID_STREAM_ID, 3072, 0, 14));
            TaskAivRecvFlag* flagRecv1Remote = AddNode(
                std::make_unique<TaskAivRecvFlag>(MakeFlag(0, 1, 0x900000, 2)),
                Position(1, 0, INVALID_STREAM_ID, 3072, 0, 15));
            AddEdge(rank0Start, graph.flagSend0);
            AddEdge(rank0Start, flagRecv0Local);
            AddEdge(rank1Start, graph.flagRecv0Remote);
            AddEdge(graph.flagSend0, flagRecv0Local);
            AddEdge(graph.flagSend0, graph.flagRecv0Remote);
            AddEdge(rank1Start, flagSend1);
            AddEdge(rank1Start, flagRecv1Local);
            AddEdge(rank0Start, flagRecv1Remote);
            AddEdge(flagSend1, flagRecv1Local);
            AddEdge(flagSend1, flagRecv1Remote);

            TaskAivSyncAll* rank0Sync = AddSyncAll(SubGraphPosition(1, 0), 0);
            TaskAivSyncAll* rank1Sync = AddSyncAll(SubGraphPosition(1, 1), 0);
            for (TaskAivPipeBarrier* pipeline : rank0Pipelines) {
                AddEdge(pipeline, rank0Sync);
            }
            for (TaskAivPipeBarrier* pipeline : rank1Pipelines) {
                AddEdge(pipeline, rank1Sync);
            }
            AddEdge(flagRecv0Local, rank0Sync);
            AddEdge(flagRecv1Remote, rank0Sync);
            AddEdge(graph.flagRecv0Remote, rank1Sync);
            AddEdge(flagRecv1Local, rank1Sync);

            TaskAivSyncAll* rank0SyncRound1 = AddSyncAll(SubGraphPosition(1, 0), 1);
            TaskAivSyncAll* rank1SyncRound1 = AddSyncAll(SubGraphPosition(1, 1), 1);
            AddEdge(rank0Sync, rank0SyncRound1);
            AddEdge(rank1Sync, rank1SyncRound1);
            AddEdge(rank0SyncRound1, AddSubGraphEnd(SubGraphPosition(1, 0)));
            AddEdge(rank1SyncRound1, AddSubGraphEnd(SubGraphPosition(1, 1)));
            return graph;
        }

        void ExpectConflict(TaskStart* start)
        {
            SyncConflictCheckStats stats;
            EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
            EXPECT_GE(stats.resourceBucketCount, 1U);
            EXPECT_GE(stats.conflictCount, 1U);
        }

        std::vector<std::unique_ptr<TaskNode>> nodes_;
    };

    TEST_F(RealAivGraphSyncTest, RealAivGraphWithPipelinesAndFlagFanoutIsAccepted)
    {
        Graph graph = BuildRealGraph();
        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(graph.mainStart, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 14U);
        EXPECT_EQ(stats.pairCount, 16U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(RealAivGraphSyncTest, AddingEventWaitAcrossOperatorIsRejected)
    {
        Graph graph = BuildRealGraph();
        TaskAivWaitFlag* extraWait = AddNode(
            std::make_unique<TaskAivWaitFlag>(MakeEvent(0, 0, 1, 2, 0)),
            Position(3, 0, INVALID_STREAM_ID, 3081, 0, 20));
        AddEdge(graph.mainStart, extraWait);
        AddEdge(graph.eventSet0, extraWait);
        ExpectConflict(graph.mainStart);
    }

    TEST_F(RealAivGraphSyncTest, AddingEventSetAcrossOperatorIsRejected)
    {
        Graph graph = BuildRealGraph();
        TaskAivSetFlag* extraSet = AddNode(
            std::make_unique<TaskAivSetFlag>(MakeEvent(0, 0, 1, 2, 0)), Position(3, 0, INVALID_STREAM_ID, 3081, 0, 21));
        AddEdge(graph.mainStart, extraSet);
        AddEdge(extraSet, graph.eventWait0);
        ExpectConflict(graph.mainStart);
    }

    TEST_F(RealAivGraphSyncTest, AddingSecondSendForFlagFanoutIsRejected)
    {
        Graph graph = BuildRealGraph();
        TaskAivSendFlag* extraSend = AddNode(
            std::make_unique<TaskAivSendFlag>(MakeFlag(0, 0, 0x900000, 2)),
            Position(3, 0, INVALID_STREAM_ID, 3081, 0, 22));
        AddEdge(graph.mainStart, extraSend);
        AddEdge(extraSend, graph.flagRecv0Remote);
        ExpectConflict(graph.mainStart);
    }

} // namespace
} // namespace HcclSim
