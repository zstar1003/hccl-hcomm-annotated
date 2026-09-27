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

    class RealCcuGraphSyncTest : public testing::Test {
    protected:
        struct Graph {
            TaskStart* mainStart{nullptr};
            TaskRecordCCU* rank0InitialPost{nullptr};
            TaskRecordCCU* rank0PostBits01{nullptr};
            TaskWaitCCU* rank0WaitBits01{nullptr};
            TaskRecordCCU* rank0PostBit1Local{nullptr};
            TaskWaitCCU* rank0WaitBit1Local{nullptr};
            TaskRecordCCU* rank0PostBit3{nullptr};
            TaskWaitCCU* rank0WaitBit3{nullptr};
            TaskRecordCCU* rank1PostBits01{nullptr};
            TaskWaitCCU* rank1WaitBits01{nullptr};
            TaskRecordCCU* rank1PostBit3{nullptr};
            TaskWaitCCU* rank1WaitBit3{nullptr};
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

        static TaskPosition Position(OperatorId operatorId, RankId rankId)
        {
            TaskPosition position;
            position.operatorId = operatorId;
            position.rankId = rankId;
            position.streamId = 0;
            position.queueId = 0;
            return position;
        }

        static void AddEdge(TaskNode* parent, TaskNode* child)
        {
            ASSERT_NE(parent, nullptr);
            ASSERT_NE(child, nullptr);
            ASSERT_TRUE(parent->AddChild(child));
            ASSERT_TRUE(child->AddParent(parent));
        }

        static CcuNotify
        MakeNotify(RankId recordRank, RankId waitRank, uint32_t dieId, uint16_t ckeId, uint16_t ckeMask)
        {
            CcuNotify notify;
            notify.channelId = INVALID_CHANNEL_ID;
            notify.recordRankId = recordRank;
            notify.waitRankId = waitRank;
            notify.dieId = dieId;
            notify.ckeId = ckeId;
            notify.ckeMask = ckeMask;
            return notify;
        }

        static CcuTraceInfo MakeTrace(const TaskPosition& position, const char* role, uint32_t instrId)
        {
            CcuTraceInfo trace;
            trace.valid = true;
            trace.opName = "ccu";
            trace.nodeRole = role;
            trace.instrId = instrId;
            trace.position = position;
            trace.taskLoc = position;
            trace.queueId = position.queueId;
            trace.dieId = 1;
            trace.missionId = 0;
            return trace;
        }

        TaskRecordCCU* AddRecord(const CcuNotify& notify, const TaskPosition& position, uint32_t instrId)
        {
            TaskRecordCCU* node = AddNode(std::make_unique<TaskRecordCCU>(notify, ProtocolType::CCU), position);
            node->SetCcuTrace(MakeTrace(position, "post", instrId));
            return node;
        }

        TaskWaitCCU* AddWait(const CcuNotify& notify, const TaskPosition& position, uint32_t instrId)
        {
            TaskWaitCCU* node = AddNode(std::make_unique<TaskWaitCCU>(notify, ProtocolType::CCU), position);
            node->SetCcuTrace(MakeTrace(position, "wait", instrId));
            return node;
        }

        TaskStart* AddLoopStart(const TaskPosition& position, uint32_t instrId)
        {
            TaskStart* node = AddNode(std::make_unique<TaskStart>(BoundaryType::LOOP), position);
            CcuTraceInfo trace = MakeTrace(position, "loop_start", instrId);
            trace.loopInstrIdStart = static_cast<uint16_t>(instrId);
            trace.loopInstrIdEnd = static_cast<uint16_t>(instrId);
            trace.loopCnt = 2;
            trace.loopExpandCnt = 2;
            node->SetCcuTrace(std::move(trace));
            return node;
        }

        TaskEnd* AddLoopEnd(const TaskPosition& position, uint32_t instrId)
        {
            TaskEnd* node = AddNode(std::make_unique<TaskEnd>(BoundaryType::LOOP), position);
            node->SetCcuTrace(MakeTrace(position, "loop_end", instrId));
            return node;
        }

        static MemSlice Slice(RankId rankId, MemType type, uint64_t offset)
        {
            MemSlice slice;
            slice.rankId = rankId;
            slice.memType = type;
            slice.offset = offset;
            slice.len = 0x800;
            return slice;
        }

        TaskBatchTransMem* AddBatchCopy(const TaskPosition& position, MemSlice src, MemSlice dst, uint32_t instrId)
        {
            auto node = std::make_unique<TaskBatchTransMem>(ProtocolType::CCU);
            node->AddSrcMemSlice(src);
            node->AddDstMemSlice(dst);
            node->AddMergedSrcMemSlice(src);
            node->AddMergedDstMemSlice(dst);
            TaskBatchTransMem* raw = AddNode(std::move(node), position);
            raw->SetCcuTrace(MakeTrace(position, "local_copy", instrId));
            return raw;
        }

        Graph BuildRealGraph()
        {
            Graph graph;
            graph.mainStart = AddMainStart();
            const TaskPosition rank0Operator0 = Position(0, 0);
            const TaskPosition rank0Operator1 = Position(1, 0);
            const TaskPosition rank0Operator2 = Position(2, 0);
            const TaskPosition rank1Operator0 = Position(0, 1);
            const TaskPosition rank1Operator1 = Position(1, 1);
            const TaskPosition rank1Operator2 = Position(2, 1);

            // Rank 0, ccuGraphNodeId=0. The first 0x3 wait is satisfied by rank 1.
            graph.rank0InitialPost = AddRecord(MakeNotify(0, 0, 1, 192, 0xffff), rank0Operator0, 25);
            graph.rank0PostBits01 = AddRecord(MakeNotify(0, 1, 1, 256, 0x3), rank0Operator0, 27);
            graph.rank0WaitBits01 = AddWait(MakeNotify(INVALID_RANK_ID, 0, 1, 256, 0x3), rank0Operator1, 29);
            auto rank0Write = AddNode(
                std::make_unique<TaskTransMem>(
                    Slice(0, MemType::INPUT, 0x800), Slice(1, MemType::OUTPUT, 0), ProtocolType::CCU),
                rank0Operator1);
            graph.rank0PostBit1Local = AddRecord(MakeNotify(0, 0, 1, 224, 0x2), rank0Operator1, 63);
            TaskStart* loopStart = AddLoopStart(rank0Operator1, 128);
            TaskBatchTransMem* loopCopyIn
                = AddBatchCopy(rank0Operator1, Slice(0, MemType::INPUT, 0), Slice(0, MemType::MS_CCU, 0x8000000), 128);
            TaskRecordCCU* loopPost1 = AddRecord(MakeNotify(0, 0, 1, 0, 0x1), rank0Operator1, 128);
            TaskWaitCCU* loopWait1 = AddWait(MakeNotify(0, 0, 1, 0, 0x1), rank0Operator1, 128);
            TaskBatchTransMem* loopCopyOut
                = AddBatchCopy(rank0Operator1, Slice(0, MemType::MS_CCU, 0x8000000), Slice(0, MemType::OUTPUT, 0), 128);
            TaskRecordCCU* loopPost2 = AddRecord(MakeNotify(0, 0, 1, 0, 0x1), rank0Operator1, 128);
            TaskWaitCCU* loopWait2 = AddWait(MakeNotify(0, 0, 1, 0, 0x1), rank0Operator1, 128);
            TaskEnd* loopEnd = AddLoopEnd(rank0Operator1, 128);
            TaskWaitCCU* rank0WaitBit1Local = AddWait(MakeNotify(0, 0, 1, 224, 0x2), rank0Operator1, 136);
            graph.rank0WaitBit1Local = rank0WaitBit1Local;
            graph.rank0PostBit3 = AddRecord(MakeNotify(0, 1, 1, 256, 0x8), rank0Operator2, 142);
            graph.rank0WaitBit3 = AddWait(MakeNotify(INVALID_RANK_ID, 0, 1, 256, 0x8), rank0Operator2, 143);

            AddEdge(graph.mainStart, graph.rank0InitialPost);
            AddEdge(graph.rank0InitialPost, graph.rank0PostBits01);
            AddEdge(graph.rank0PostBits01, graph.rank0WaitBits01);
            AddEdge(graph.rank0WaitBits01, rank0Write);
            AddEdge(rank0Write, graph.rank0PostBit1Local);
            AddEdge(graph.rank0PostBit1Local, loopStart);
            AddEdge(loopStart, loopCopyIn);
            AddEdge(loopCopyIn, loopPost1);
            AddEdge(loopPost1, loopWait1);
            AddEdge(loopWait1, loopCopyOut);
            AddEdge(loopCopyOut, loopPost2);
            AddEdge(loopPost2, loopWait2);
            AddEdge(loopWait2, loopEnd);
            AddEdge(loopEnd, graph.rank0WaitBit1Local);
            AddEdge(graph.rank0WaitBit1Local, graph.rank0PostBit3);
            AddEdge(graph.rank0PostBit3, graph.rank0WaitBit3);

            // Rank 1, ccuGraphNodeId=1. Its 0x3 post satisfies rank 0's wait.
            TaskRecordCCU* rank1InitialPost = AddRecord(MakeNotify(1, 1, 1, 192, 0xffff), rank1Operator0, 17);
            graph.rank1PostBits01 = AddRecord(MakeNotify(1, 0, 1, 256, 0x3), rank1Operator0, 19);
            graph.rank1WaitBits01 = AddWait(MakeNotify(INVALID_RANK_ID, 1, 1, 256, 0x3), rank1Operator1, 21);
            graph.rank1PostBit3 = AddRecord(MakeNotify(1, 0, 1, 256, 0x8), rank1Operator2, 33);
            graph.rank1WaitBit3 = AddWait(MakeNotify(INVALID_RANK_ID, 1, 1, 256, 0x8), rank1Operator2, 34);

            AddEdge(graph.mainStart, rank1InitialPost);
            AddEdge(rank1InitialPost, graph.rank1PostBits01);
            AddEdge(graph.rank1PostBits01, graph.rank1WaitBits01);
            AddEdge(graph.rank1WaitBits01, graph.rank1PostBit3);
            AddEdge(graph.rank1PostBit3, graph.rank1WaitBit3);

            // Cross-rank CCU dependencies. Each mask bit is an independent resource.
            AddEdge(graph.rank1PostBits01, graph.rank0WaitBits01);
            AddEdge(graph.rank0PostBits01, graph.rank1WaitBits01);
            AddEdge(graph.rank0PostBit1Local, graph.rank0WaitBit1Local);
            AddEdge(graph.rank0PostBit3, graph.rank1WaitBit3);
            AddEdge(graph.rank1PostBit3, graph.rank0WaitBit3);
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

    TEST_F(RealCcuGraphSyncTest, RealCcuGraphWithCrossRankMasksIsAccepted)
    {
        Graph graph = BuildRealGraph();
        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(graph.mainStart, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 40U);
        EXPECT_EQ(stats.pairCount, 41U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(RealCcuGraphSyncTest, AddingWaitForExistingCcuMaskBitIsRejected)
    {
        Graph graph = BuildRealGraph();
        TaskWaitCCU* extraWait = AddWait(MakeNotify(INVALID_RANK_ID, 1, 1, 256, 0x1), Position(3, 1), 35);
        AddEdge(graph.mainStart, extraWait);
        AddEdge(graph.rank0PostBits01, extraWait);

        ExpectConflict(graph.mainStart);
    }

    TEST_F(RealCcuGraphSyncTest, AddingRecordForExistingCcuMaskBitIsRejected)
    {
        Graph graph = BuildRealGraph();
        TaskRecordCCU* extraPost = AddRecord(MakeNotify(3, 1, 1, 256, 0x1), Position(3, 3), 36);
        AddEdge(graph.mainStart, extraPost);
        AddEdge(extraPost, graph.rank1WaitBits01);

        ExpectConflict(graph.mainStart);
    }

} // namespace
} // namespace HcclSim
