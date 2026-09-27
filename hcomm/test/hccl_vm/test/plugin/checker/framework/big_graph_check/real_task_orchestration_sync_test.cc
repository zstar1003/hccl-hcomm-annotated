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

    class RealTaskOrchestrationSyncTest : public testing::Test {
    protected:
        struct Graph {
            TaskStart* start{nullptr};
            TaskRecordAICPU* rank0Record12{nullptr};
            TaskWaitAICPU* rank0Wait5{nullptr};
            TaskWaitAICPU* rank0Wait12{nullptr};
            TaskRecordAICPU* rank0Record14{nullptr};
            TaskWaitAICPU* rank0Wait11{nullptr};
            TaskRecordAICPU* rank0Record5{nullptr};
            TaskWaitAICPU* rank0Wait14{nullptr};
            TaskRecordAICPU* rank0Record24{nullptr};
            TaskWaitAICPU* rank0Wait22{nullptr};
            TaskRecordAICPU* rank0Record11{nullptr};
            TaskRecordAICPU* rank1Record17{nullptr};
            TaskWaitAICPU* rank1Wait8{nullptr};
            TaskWaitAICPU* rank1Wait17{nullptr};
            TaskWaitAICPU* rank1Wait24{nullptr};
            TaskRecordAICPU* rank1Record22{nullptr};
            TaskRecordAICPU* rank1Record8{nullptr};
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

        static TaskPosition Position(OperatorId operatorId, RankId rankId, StreamId streamId)
        {
            TaskPosition position;
            position.operatorId = operatorId;
            position.rankId = rankId;
            position.streamId = streamId;
            position.queueId = streamId;
            return position;
        }

        static void AddEdge(TaskNode* parent, TaskNode* child)
        {
            ASSERT_NE(parent, nullptr);
            ASSERT_NE(child, nullptr);
            ASSERT_TRUE(parent->AddChild(child));
            ASSERT_TRUE(child->AddParent(parent));
        }

        TaskRecordAICPU* AddRecord(RankId recordRank, RankId waitRank, uint32_t notifyId, const TaskPosition& position)
        {
            AicpuNotify notify;
            notify.recordRankId = recordRank;
            notify.waitRankId = waitRank;
            notify.notifyId = notifyId;
            return AddNode(std::make_unique<TaskRecordAICPU>(notify, ProtocolType::SDMA), position);
        }

        TaskWaitAICPU* AddWait(RankId recordRank, RankId waitRank, uint32_t notifyId, const TaskPosition& position)
        {
            AicpuNotify notify;
            notify.recordRankId = recordRank;
            notify.waitRankId = waitRank;
            notify.notifyId = notifyId;
            return AddNode(std::make_unique<TaskWaitAICPU>(notify, ProtocolType::SDMA), position);
        }

        static MemSlice Slice(RankId rankId, MemType type, uint64_t offset)
        {
            MemSlice slice;
            slice.rankId = rankId;
            slice.memType = type;
            slice.offset = offset;
            slice.len = 0x100;
            return slice;
        }

        Graph BuildRealOrchestration(bool includeRecord24 = true, uint32_t record24NotifyId = 24)
        {
            Graph graph;
            graph.start = AddMainStart();

            const TaskPosition rank0Stream0 = Position(0, 0, 0);
            const TaskPosition rank0Stream11 = Position(1, 0, 11);
            const TaskPosition rank0Stream12 = Position(2, 0, 12);
            const TaskPosition rank1Stream0 = Position(0, 1, 0);
            const TaskPosition rank1Stream14 = Position(1, 1, 14);

            // Rank 0, stream 0: Record(12), Wait(5).
            graph.rank0Record12 = AddRecord(0, 0, 12, rank0Stream0);
            graph.rank0Wait5 = AddWait(0, 0, 5, rank0Stream0);
            AddEdge(graph.start, graph.rank0Record12);
            AddEdge(graph.rank0Record12, graph.rank0Wait5);

            // Rank 0, stream 11: Wait(12), Record(14), TransMem, Wait(11), Reduce, Record(5).
            graph.rank0Wait12 = AddWait(0, 0, 12, rank0Stream11);
            graph.rank0Record14 = AddRecord(0, 0, 14, rank0Stream11);
            auto rank0TransMem = AddNode(
                std::make_unique<TaskTransMem>(
                    Slice(0, MemType::INPUT, 0), Slice(0, MemType::CCL, 0x1000), ProtocolType::SDMA),
                rank0Stream11);
            graph.rank0Wait11 = AddWait(0, 0, 11, rank0Stream11);
            auto rank0Reduce = AddNode(
                std::make_unique<TaskReduce>(
                    Slice(0, MemType::CCL, 0x1000), Slice(0, MemType::OUTPUT, 0), 0, 0, ProtocolType::SDMA),
                rank0Stream11);
            graph.rank0Record5 = AddRecord(0, 0, 5, rank0Stream11);
            AddEdge(graph.start, graph.rank0Wait12);
            AddEdge(graph.rank0Wait12, graph.rank0Record14);
            AddEdge(graph.rank0Record14, rank0TransMem);
            AddEdge(rank0TransMem, graph.rank0Wait11);
            AddEdge(graph.rank0Wait11, rank0Reduce);
            AddEdge(rank0Reduce, graph.rank0Record5);

            // Rank 0, stream 12: Wait(14), Record(24), Wait(22), Record(11).
            graph.rank0Wait14 = AddWait(0, 0, 14, rank0Stream12);
            if (includeRecord24) {
                graph.rank0Record24 = AddRecord(0, 1, record24NotifyId, rank0Stream12);
            }
            graph.rank0Wait22 = AddWait(1, 0, 22, rank0Stream12);
            graph.rank0Record11 = AddRecord(0, 0, 11, rank0Stream12);
            AddEdge(graph.start, graph.rank0Wait14);
            if (includeRecord24) {
                AddEdge(graph.rank0Wait14, graph.rank0Record24);
                AddEdge(graph.rank0Record24, graph.rank0Wait22);
            } else {
                AddEdge(graph.rank0Wait14, graph.rank0Wait22);
            }
            AddEdge(graph.rank0Wait22, graph.rank0Record11);

            // Rank 1, stream 0: Record(17), Wait(8).
            graph.rank1Record17 = AddRecord(1, 1, 17, rank1Stream0);
            graph.rank1Wait8 = AddWait(1, 1, 8, rank1Stream0);
            AddEdge(graph.start, graph.rank1Record17);
            AddEdge(graph.rank1Record17, graph.rank1Wait8);

            // Rank 1, stream 14: Wait(17), Wait(24), TransMem, Record(22), Record(8).
            graph.rank1Wait17 = AddWait(1, 1, 17, rank1Stream14);
            graph.rank1Wait24 = AddWait(0, 1, 24, rank1Stream14);
            auto rank1TransMem = AddNode(
                std::make_unique<TaskTransMem>(
                    Slice(1, MemType::INPUT, 0), Slice(1, MemType::CCL, 0x2000), ProtocolType::SDMA),
                rank1Stream14);
            graph.rank1Record22 = AddRecord(1, 0, 22, rank1Stream14);
            graph.rank1Record8 = AddRecord(1, 1, 8, rank1Stream14);
            AddEdge(graph.start, graph.rank1Wait17);
            AddEdge(graph.rank1Wait17, graph.rank1Wait24);
            AddEdge(graph.rank1Wait24, rank1TransMem);
            AddEdge(rank1TransMem, graph.rank1Record22);
            AddEdge(graph.rank1Record22, graph.rank1Record8);

            // Cross-stream relationships represented by the notify records and waits in the dump.
            AddEdge(graph.rank0Record12, graph.rank0Wait12);
            AddEdge(graph.rank0Record14, graph.rank0Wait14);
            if (includeRecord24 && record24NotifyId == 24) {
                AddEdge(graph.rank0Record24, graph.rank1Wait24);
            }
            AddEdge(graph.rank1Record22, graph.rank0Wait22);
            AddEdge(graph.rank1Record17, graph.rank1Wait17);
            AddEdge(graph.rank1Record8, graph.rank1Wait8);
            AddEdge(graph.rank0Record5, graph.rank0Wait5);
            AddEdge(graph.rank0Record11, graph.rank0Wait11);
            return graph;
        }

        void ExpectConflict(TaskStart* start)
        {
            SyncConflictCheckStats stats;
            EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
            EXPECT_GE(stats.resourceBucketCount, 1U);
            EXPECT_GE(stats.conflictCount, 1U);
        }

        void ExpectSuccess(TaskStart* start)
        {
            SyncConflictCheckStats stats;
            EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
            EXPECT_GE(stats.resourceBucketCount, 8U);
            EXPECT_EQ(stats.conflictCount, 0U);
        }

        std::vector<std::unique_ptr<TaskNode>> nodes_;
    };

    TEST_F(RealTaskOrchestrationSyncTest, RealOrchestrationBaselineIsAccepted)
    {
        Graph graph = BuildRealOrchestration();
        ExpectSuccess(graph.start);
    }

    TEST_F(RealTaskOrchestrationSyncTest, AddingWaitTaskForExistingNotifyIsRejected)
    {
        Graph graph = BuildRealOrchestration();
        const TaskPosition rank0Stream12 = Position(2, 0, 12);
        TaskWaitAICPU* extraWait12 = AddWait(0, 0, 12, rank0Stream12);

        AddEdge(graph.rank0Wait14, extraWait12);
        AddEdge(extraWait12, graph.rank0Record24);
        AddEdge(graph.rank0Record12, extraWait12);

        ExpectConflict(graph.start);
    }

    TEST_F(RealTaskOrchestrationSyncTest, AddingRecordTaskForExistingNotifyIsRejected)
    {
        Graph graph = BuildRealOrchestration();
        const TaskPosition rank0Stream11 = Position(1, 0, 11);
        TaskRecordAICPU* extraRecord24 = AddRecord(0, 1, 24, rank0Stream11);

        AddEdge(graph.start, extraRecord24);
        AddEdge(extraRecord24, graph.rank1Wait24);

        ExpectConflict(graph.start);
    }

    TEST_F(RealTaskOrchestrationSyncTest, DeletingRecordTaskLeavesConsumerWithoutProducer)
    {
        Graph graph = BuildRealOrchestration(false);
        ExpectConflict(graph.start);
    }

    TEST_F(RealTaskOrchestrationSyncTest, ModifyingRecordTaskNotifyIdBreaksExistingPair)
    {
        Graph graph = BuildRealOrchestration(true, 12);
        ExpectConflict(graph.start);
    }

} // namespace
} // namespace HcclSim
