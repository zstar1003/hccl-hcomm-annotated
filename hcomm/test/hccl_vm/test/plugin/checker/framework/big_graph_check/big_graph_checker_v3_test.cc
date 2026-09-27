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

    class BigGraphCheckerV3SyncTest : public testing::Test {
    protected:
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

        static TaskPosition MakePosition(OperatorId operatorId, RankId rankId, StreamId streamId, QueueId queueId)
        {
            TaskPosition position;
            position.operatorId = operatorId;
            position.rankId = rankId;
            position.streamId = streamId;
            position.queueId = queueId;
            return position;
        }

        static void AddEdge(TaskNode* parent, TaskNode* child)
        {
            ASSERT_NE(parent, nullptr);
            ASSERT_NE(child, nullptr);
            ASSERT_TRUE(parent->AddChild(child));
            ASSERT_TRUE(child->AddParent(parent));
        }

        TaskRecordAICPU* AddAicpuRecord(uint32_t notifyId, const TaskPosition& position)
        {
            AicpuNotify notify;
            notify.recordRankId = 0;
            notify.waitRankId = 2;
            notify.notifyId = notifyId;
            return AddNode(std::make_unique<TaskRecordAICPU>(notify, ProtocolType::SDMA), position);
        }

        TaskWaitAICPU* AddAicpuWait(uint32_t notifyId, const TaskPosition& position)
        {
            AicpuNotify notify;
            notify.recordRankId = 0;
            notify.waitRankId = 2;
            notify.notifyId = notifyId;
            return AddNode(std::make_unique<TaskWaitAICPU>(notify, ProtocolType::SDMA), position);
        }

        TaskRecordCCU* AddCcuRecord(uint16_t ckeId, const TaskPosition& position)
        {
            CcuNotify notify;
            notify.channelId = INVALID_CHANNEL_ID;
            notify.recordRankId = 0;
            notify.waitRankId = 2;
            notify.dieId = 0;
            notify.ckeId = ckeId;
            notify.ckeMask = 0x1;
            return AddNode(std::make_unique<TaskRecordCCU>(notify, ProtocolType::CCU), position);
        }

        TaskWaitCCU* AddCcuWait(uint16_t ckeId, const TaskPosition& position)
        {
            CcuNotify notify;
            notify.channelId = INVALID_CHANNEL_ID;
            notify.recordRankId = 0;
            notify.waitRankId = 2;
            notify.dieId = 0;
            notify.ckeId = ckeId;
            notify.ckeMask = 0x1;
            return AddNode(std::make_unique<TaskWaitCCU>(notify, ProtocolType::CCU), position);
        }

        static AivPipeEvent MakeEvent(int32_t eventId)
        {
            AivPipeEvent event;
            event.rankId = 0;
            event.launchIdx = 7;
            event.blockId = 3;
            event.srcPipe = 1;
            event.dstPipe = 2;
            event.eventId = eventId;
            return event;
        }

        static AivFlagSync MakeFlag(uint64_t commInfoOffset, int32_t value = 5)
        {
            AivFlagSync flag;
            flag.currentRank = 0;
            flag.flagOwnerRank = 2;
            flag.launchIdx = 7;
            flag.blockId = 3;
            flag.commInfoOffset = commInfoOffset;
            flag.value = value;
            return flag;
        }

        TaskAivSetFlag* AddAivEventSet(int32_t eventId, const TaskPosition& position)
        {
            return AddNode(std::make_unique<TaskAivSetFlag>(MakeEvent(eventId)), position);
        }

        TaskAivWaitFlag* AddAivEventWait(int32_t eventId, const TaskPosition& position)
        {
            return AddNode(std::make_unique<TaskAivWaitFlag>(MakeEvent(eventId)), position);
        }

        TaskAivSendFlag* AddAivFlagSend(uint64_t commInfoOffset, const TaskPosition& position)
        {
            return AddNode(std::make_unique<TaskAivSendFlag>(MakeFlag(commInfoOffset)), position);
        }

        TaskAivRecvFlag* AddAivFlagRecv(uint64_t commInfoOffset, const TaskPosition& position)
        {
            return AddNode(std::make_unique<TaskAivRecvFlag>(MakeFlag(commInfoOffset)), position);
        }

        void ExpectConflict(TaskStart* start, size_t minimumBucketCount, size_t minimumPairCount)
        {
            SyncConflictCheckStats stats;
            EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
            EXPECT_GE(stats.resourceBucketCount, minimumBucketCount);
            EXPECT_GE(stats.pairCount, minimumPairCount);
            EXPECT_GE(stats.checkedBucketCount, 1U);
            EXPECT_GE(stats.conflictCount, 1U);
        }

        void ExpectSuccess(TaskStart* start, size_t minimumBucketCount, size_t minimumPairCount)
        {
            SyncConflictCheckStats stats;
            EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
            EXPECT_GE(stats.resourceBucketCount, minimumBucketCount);
            EXPECT_GE(stats.pairCount, minimumPairCount);
            EXPECT_EQ(stats.conflictCount, 0U);
        }

        std::vector<std::unique_ptr<TaskNode>> nodes_;
    };

    TEST_F(BigGraphCheckerV3SyncTest, CrossOperatorAicpuOnePostToManyWaitsIsRejected)
    {
        TaskStart* start = AddMainStart();
        const TaskPosition op0Rank0 = MakePosition(0, 0, 0, 0);
        const TaskPosition op1Rank1 = MakePosition(1, 1, 1, 1);
        const TaskPosition op2Rank2 = MakePosition(2, 2, 2, 2);

        TaskRecordAICPU* targetPost = AddAicpuRecord(100, op0Rank0);
        TaskWaitAICPU* targetWait1 = AddAicpuWait(100, op1Rank1);
        TaskWaitAICPU* targetWait2 = AddAicpuWait(100, op2Rank2);

        TaskRecordAICPU* validPost0 = AddAicpuRecord(200, op0Rank0);
        TaskWaitAICPU* validWait0 = AddAicpuWait(200, op1Rank1);
        TaskRecordAICPU* validPost1 = AddAicpuRecord(201, op1Rank1);
        TaskWaitAICPU* validWait1 = AddAicpuWait(201, op2Rank2);
        TaskEnd* barrier = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH), op2Rank2);

        AddEdge(start, targetPost);
        AddEdge(targetPost, targetWait1);
        AddEdge(targetPost, targetWait2);
        AddEdge(targetWait1, barrier);
        AddEdge(targetWait2, barrier);
        AddEdge(start, validPost0);
        AddEdge(validPost0, validWait0);
        AddEdge(validWait0, validPost1);
        AddEdge(validPost1, validWait1);
        AddEdge(validWait1, barrier);

        ExpectConflict(start, 3U, 4U);
    }

    TEST_F(BigGraphCheckerV3SyncTest, CrossOperatorAicpuManyPostsToOneWaitIsRejected)
    {
        TaskStart* start = AddMainStart();
        const TaskPosition op0Rank0 = MakePosition(0, 0, 0, 0);
        const TaskPosition op1Rank1 = MakePosition(1, 1, 1, 1);
        const TaskPosition op2Rank2 = MakePosition(2, 2, 2, 2);

        TaskRecordAICPU* targetPost0 = AddAicpuRecord(110, op0Rank0);
        TaskRecordAICPU* targetPost1 = AddAicpuRecord(110, op1Rank1);
        TaskWaitAICPU* targetWait = AddAicpuWait(110, op2Rank2);

        TaskRecordAICPU* validPost0 = AddAicpuRecord(210, op0Rank0);
        TaskWaitAICPU* validWait0 = AddAicpuWait(210, op1Rank1);
        TaskRecordAICPU* validPost1 = AddAicpuRecord(211, op1Rank1);
        TaskWaitAICPU* validWait1 = AddAicpuWait(211, op2Rank2);
        TaskEnd* barrier = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH), op2Rank2);

        AddEdge(start, targetPost0);
        AddEdge(start, targetPost1);
        AddEdge(targetPost0, targetWait);
        AddEdge(targetPost1, targetWait);
        AddEdge(targetWait, barrier);
        AddEdge(start, validPost0);
        AddEdge(validPost0, validWait0);
        AddEdge(validWait0, validPost1);
        AddEdge(validPost1, validWait1);
        AddEdge(validWait1, barrier);

        ExpectConflict(start, 3U, 4U);
    }

    TEST_F(BigGraphCheckerV3SyncTest, CrossOperatorAivEventOnePostToManyWaitsIsRejected)
    {
        TaskStart* start = AddMainStart();
        const TaskPosition op0Rank0 = MakePosition(0, 0, 0, 0);
        const TaskPosition op1Rank1 = MakePosition(1, 1, 1, 1);
        const TaskPosition op2Rank2 = MakePosition(2, 2, 2, 2);

        TaskAivSetFlag* targetSet = AddAivEventSet(300, op0Rank0);
        TaskAivWaitFlag* targetWait1 = AddAivEventWait(300, op1Rank1);
        TaskAivWaitFlag* targetWait2 = AddAivEventWait(300, op2Rank2);

        TaskAivSetFlag* validSet0 = AddAivEventSet(301, op0Rank0);
        TaskAivWaitFlag* validWait0 = AddAivEventWait(301, op1Rank1);
        TaskAivSetFlag* validSet1 = AddAivEventSet(302, op1Rank1);
        TaskAivWaitFlag* validWait1 = AddAivEventWait(302, op2Rank2);
        TaskEnd* barrier = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH), op2Rank2);

        AddEdge(start, targetSet);
        AddEdge(targetSet, targetWait1);
        AddEdge(targetSet, targetWait2);
        AddEdge(targetWait1, barrier);
        AddEdge(targetWait2, barrier);
        AddEdge(start, validSet0);
        AddEdge(validSet0, validWait0);
        AddEdge(validWait0, validSet1);
        AddEdge(validSet1, validWait1);
        AddEdge(validWait1, barrier);

        ExpectConflict(start, 3U, 4U);
    }

    TEST_F(BigGraphCheckerV3SyncTest, CrossOperatorAivEventManyPostsToOneWaitIsRejected)
    {
        TaskStart* start = AddMainStart();
        const TaskPosition op0Rank0 = MakePosition(0, 0, 0, 0);
        const TaskPosition op1Rank1 = MakePosition(1, 1, 1, 1);
        const TaskPosition op2Rank2 = MakePosition(2, 2, 2, 2);

        TaskAivSetFlag* targetSet0 = AddAivEventSet(310, op0Rank0);
        TaskAivSetFlag* targetSet1 = AddAivEventSet(310, op1Rank1);
        TaskAivWaitFlag* targetWait = AddAivEventWait(310, op2Rank2);

        TaskAivSetFlag* validSet0 = AddAivEventSet(311, op0Rank0);
        TaskAivWaitFlag* validWait0 = AddAivEventWait(311, op1Rank1);
        TaskAivSetFlag* validSet1 = AddAivEventSet(312, op1Rank1);
        TaskAivWaitFlag* validWait1 = AddAivEventWait(312, op2Rank2);
        TaskEnd* barrier = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH), op2Rank2);

        AddEdge(start, targetSet0);
        AddEdge(start, targetSet1);
        AddEdge(targetSet0, targetWait);
        AddEdge(targetSet1, targetWait);
        AddEdge(targetWait, barrier);
        AddEdge(start, validSet0);
        AddEdge(validSet0, validWait0);
        AddEdge(validWait0, validSet1);
        AddEdge(validSet1, validWait1);
        AddEdge(validWait1, barrier);

        ExpectConflict(start, 3U, 4U);
    }

    TEST_F(BigGraphCheckerV3SyncTest, CrossOperatorAivFlagOnePostToManyWaitsIsAllowed)
    {
        TaskStart* start = AddMainStart();
        const TaskPosition op0Rank0 = MakePosition(0, 0, 0, 0);
        const TaskPosition op1Rank1 = MakePosition(1, 1, 1, 1);
        const TaskPosition op2Rank2 = MakePosition(2, 2, 2, 2);

        TaskAivSendFlag* targetSend = AddAivFlagSend(0x1000, op0Rank0);
        TaskAivRecvFlag* targetRecv1 = AddAivFlagRecv(0x1000, op1Rank1);
        TaskAivRecvFlag* targetRecv2 = AddAivFlagRecv(0x1000, op2Rank2);

        TaskAivSendFlag* validSend0 = AddAivFlagSend(0x1100, op0Rank0);
        TaskAivRecvFlag* validRecv0 = AddAivFlagRecv(0x1100, op1Rank1);
        TaskAivSendFlag* validSend1 = AddAivFlagSend(0x1200, op1Rank1);
        TaskAivRecvFlag* validRecv1 = AddAivFlagRecv(0x1200, op2Rank2);
        TaskEnd* barrier = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH), op2Rank2);

        AddEdge(start, targetSend);
        AddEdge(targetSend, targetRecv1);
        AddEdge(targetSend, targetRecv2);
        AddEdge(targetRecv1, barrier);
        AddEdge(targetRecv2, barrier);
        AddEdge(start, validSend0);
        AddEdge(validSend0, validRecv0);
        AddEdge(validRecv0, validSend1);
        AddEdge(validSend1, validRecv1);
        AddEdge(validRecv1, barrier);

        ExpectSuccess(start, 3U, 4U);
    }

    TEST_F(BigGraphCheckerV3SyncTest, CrossOperatorAivFlagManyPostsToOneWaitIsRejected)
    {
        TaskStart* start = AddMainStart();
        const TaskPosition op0Rank0 = MakePosition(0, 0, 0, 0);
        const TaskPosition op1Rank1 = MakePosition(1, 1, 1, 1);
        const TaskPosition op2Rank2 = MakePosition(2, 2, 2, 2);

        TaskAivSendFlag* targetSend0 = AddAivFlagSend(0x1300, op0Rank0);
        TaskAivSendFlag* targetSend1 = AddAivFlagSend(0x1300, op1Rank1);
        TaskAivRecvFlag* targetRecv = AddAivFlagRecv(0x1300, op2Rank2);

        TaskAivSendFlag* validSend0 = AddAivFlagSend(0x1400, op0Rank0);
        TaskAivRecvFlag* validRecv0 = AddAivFlagRecv(0x1400, op1Rank1);
        TaskAivSendFlag* validSend1 = AddAivFlagSend(0x1500, op1Rank1);
        TaskAivRecvFlag* validRecv1 = AddAivFlagRecv(0x1500, op2Rank2);
        TaskEnd* barrier = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH), op2Rank2);

        AddEdge(start, targetSend0);
        AddEdge(start, targetSend1);
        AddEdge(targetSend0, targetRecv);
        AddEdge(targetSend1, targetRecv);
        AddEdge(targetRecv, barrier);
        AddEdge(start, validSend0);
        AddEdge(validSend0, validRecv0);
        AddEdge(validRecv0, validSend1);
        AddEdge(validSend1, validRecv1);
        AddEdge(validRecv1, barrier);

        ExpectConflict(start, 3U, 4U);
    }

    TEST_F(BigGraphCheckerV3SyncTest, CrossOperatorCcuOnePostToManyWaitsIsRejected)
    {
        TaskStart* start = AddMainStart();
        const TaskPosition op0Rank0 = MakePosition(0, 0, 0, 0);
        const TaskPosition op1Rank1 = MakePosition(1, 1, 1, 1);
        const TaskPosition op2Rank2 = MakePosition(2, 2, 2, 2);

        TaskRecordCCU* targetPost = AddCcuRecord(1, op0Rank0);
        TaskWaitCCU* targetWait1 = AddCcuWait(1, op1Rank1);
        TaskWaitCCU* targetWait2 = AddCcuWait(1, op2Rank2);

        TaskRecordCCU* validPost0 = AddCcuRecord(2, op0Rank0);
        TaskWaitCCU* validWait0 = AddCcuWait(2, op1Rank1);
        TaskRecordCCU* validPost1 = AddCcuRecord(3, op1Rank1);
        TaskWaitCCU* validWait1 = AddCcuWait(3, op2Rank2);
        TaskEnd* barrier = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH), op2Rank2);

        AddEdge(start, targetPost);
        AddEdge(targetPost, targetWait1);
        AddEdge(targetPost, targetWait2);
        AddEdge(targetWait1, barrier);
        AddEdge(targetWait2, barrier);
        AddEdge(start, validPost0);
        AddEdge(validPost0, validWait0);
        AddEdge(validWait0, validPost1);
        AddEdge(validPost1, validWait1);
        AddEdge(validWait1, barrier);

        ExpectConflict(start, 3U, 4U);
    }

    TEST_F(BigGraphCheckerV3SyncTest, CrossOperatorCcuManyPostsToOneWaitIsRejected)
    {
        TaskStart* start = AddMainStart();
        const TaskPosition op0Rank0 = MakePosition(0, 0, 0, 0);
        const TaskPosition op1Rank1 = MakePosition(1, 1, 1, 1);
        const TaskPosition op2Rank2 = MakePosition(2, 2, 2, 2);

        TaskRecordCCU* targetPost0 = AddCcuRecord(4, op0Rank0);
        TaskRecordCCU* targetPost1 = AddCcuRecord(4, op1Rank1);
        TaskWaitCCU* targetWait = AddCcuWait(4, op2Rank2);

        TaskRecordCCU* validPost0 = AddCcuRecord(5, op0Rank0);
        TaskWaitCCU* validWait0 = AddCcuWait(5, op1Rank1);
        TaskRecordCCU* validPost1 = AddCcuRecord(6, op1Rank1);
        TaskWaitCCU* validWait1 = AddCcuWait(6, op2Rank2);
        TaskEnd* barrier = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH), op2Rank2);

        AddEdge(start, targetPost0);
        AddEdge(start, targetPost1);
        AddEdge(targetPost0, targetWait);
        AddEdge(targetPost1, targetWait);
        AddEdge(targetWait, barrier);
        AddEdge(start, validPost0);
        AddEdge(validPost0, validWait0);
        AddEdge(validWait0, validPost1);
        AddEdge(validPost1, validWait1);
        AddEdge(validWait1, barrier);

        ExpectConflict(start, 3U, 4U);
    }

} // namespace
} // namespace HcclSim
