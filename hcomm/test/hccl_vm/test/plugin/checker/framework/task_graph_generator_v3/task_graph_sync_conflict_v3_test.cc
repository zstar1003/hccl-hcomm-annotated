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

    class SyncConflictTest : public testing::Test {
    protected:
        template <typename T>
        T* AddNode(std::unique_ptr<T> node)
        {
            T* raw = node.get();
            raw->SetNodeId(static_cast<NodeId>(nodes_.size()));
            nodes_.emplace_back(std::move(node));
            return raw;
        }

        template <typename T>
        T* AddNode(std::unique_ptr<T> node, const TaskPosition& position)
        {
            T* raw = AddNode(std::move(node));
            raw->SetPosition(position);
            return raw;
        }

        static TaskPosition MakePosition(RankId rankId, StreamId streamId, QueueId queueId)
        {
            TaskPosition position;
            position.rankId = rankId;
            position.streamId = streamId;
            position.queueId = queueId;
            return position;
        }

        static TaskPosition MakeLane(size_t lane)
        {
            return MakePosition(
                static_cast<RankId>(lane / 4U), static_cast<StreamId>(lane % 4U), static_cast<QueueId>(lane));
        }

        TaskStart* AddMainStart()
        {
            auto node = std::make_unique<TaskStart>(BoundaryType::MAIN_GRAPH);
            node->SetNodeId(MAIN_START_NODE_ID);
            TaskStart* raw = node.get();
            nodes_.emplace_back(std::move(node));
            return raw;
        }

        TaskRecordAICPU* AddRecord(uint32_t notifyId)
        {
            AicpuNotify notify;
            notify.recordRankId = 0;
            notify.waitRankId = 0;
            notify.notifyId = notifyId;
            return AddNode(std::make_unique<TaskRecordAICPU>(notify, ProtocolType::SDMA));
        }

        TaskWaitAICPU* AddWait(uint32_t notifyId)
        {
            AicpuNotify notify;
            notify.recordRankId = 0;
            notify.waitRankId = 0;
            notify.notifyId = notifyId;
            return AddNode(std::make_unique<TaskWaitAICPU>(notify, ProtocolType::SDMA));
        }

        TaskRecordAICPU* AddRecord(uint32_t notifyId, const TaskPosition& position)
        {
            return AddNode(std::make_unique<TaskRecordAICPU>(MakeAicpuNotify(notifyId), ProtocolType::SDMA), position);
        }

        TaskWaitAICPU* AddWait(uint32_t notifyId, const TaskPosition& position)
        {
            return AddNode(std::make_unique<TaskWaitAICPU>(MakeAicpuNotify(notifyId), ProtocolType::SDMA), position);
        }

        TaskRecordCCU* AddCcuRecord(uint16_t mask)
        {
            CcuNotify notify;
            notify.waitRankId = 0;
            notify.dieId = 0;
            notify.ckeId = 1;
            notify.ckeMask = mask;
            return AddNode(std::make_unique<TaskRecordCCU>(notify, ProtocolType::CCU));
        }

        TaskWaitCCU* AddCcuWait(uint16_t mask)
        {
            CcuNotify notify;
            notify.waitRankId = 0;
            notify.dieId = 0;
            notify.ckeId = 1;
            notify.ckeMask = mask;
            return AddNode(std::make_unique<TaskWaitCCU>(notify, ProtocolType::CCU));
        }

        static AicpuNotify MakeAicpuNotify(uint32_t notifyId)
        {
            AicpuNotify notify;
            notify.recordRankId = 0;
            notify.waitRankId = 0;
            notify.notifyId = notifyId;
            return notify;
        }

        static AivPipeEvent MakeEvent(
            int32_t eventId, RankId rankId = 0, uint64_t launchIdx = 1, uint32_t blockId = 2, uint32_t srcPipe = 3,
            uint32_t dstPipe = 4)
        {
            AivPipeEvent event;
            event.rankId = rankId;
            event.launchIdx = launchIdx;
            event.blockId = blockId;
            event.srcPipe = srcPipe;
            event.dstPipe = dstPipe;
            event.eventId = eventId;
            return event;
        }

        static AivFlagSync
        MakeFlag(uint64_t commInfoOffset, RankId flagOwnerRank = 0, uint64_t launchIdx = 1, int32_t value = 0)
        {
            AivFlagSync flag;
            flag.currentRank = 0;
            flag.flagOwnerRank = flagOwnerRank;
            flag.launchIdx = launchIdx;
            flag.blockId = 2;
            flag.commInfoOffset = commInfoOffset;
            flag.value = value;
            return flag;
        }

        static void AddEdge(TaskNode* parent, TaskNode* child)
        {
            ASSERT_NE(parent, nullptr);
            ASSERT_NE(child, nullptr);
            ASSERT_TRUE(parent->AddChild(child));
            ASSERT_TRUE(child->AddParent(parent));
        }

        std::vector<std::unique_ptr<TaskNode>> nodes_;
    };

    TEST_F(SyncConflictTest, SinglePairBucketIsSkippedAndOriginalGraphIsUnchanged)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post = AddRecord(1);
        TaskWaitAICPU* wait = AddWait(1);
        AddEdge(start, post);
        AddEdge(post, wait);

        const std::vector<TaskNode*> originalChildren = post->GetChildren();
        const std::vector<TaskNode*> originalParents = wait->GetParents();
        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 1U);
        EXPECT_EQ(stats.checkedBucketCount, 0U);
        EXPECT_EQ(stats.conflictCount, 0U);
        EXPECT_EQ(post->GetChildren(), originalChildren);
        EXPECT_EQ(wait->GetParents(), originalParents);
    }

    TEST_F(SyncConflictTest, NullStartIsRejected)
    {
        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(nullptr, &stats), HCCL_E_PTR);
        EXPECT_EQ(stats.originalNodeCount, 0U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, NullStatsArgumentIsAllowed)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post = AddRecord(1);
        TaskWaitAICPU* wait = AddWait(1);
        AddEdge(start, post);
        AddEdge(post, wait);

        EXPECT_EQ(CheckSyncResourceConflict(start), HCCL_SUCCESS);
    }

    TEST_F(SyncConflictTest, NullNotifyResourceIsIgnored)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post
            = AddNode(std::make_unique<TaskRecordAICPU>(MakeAicpuNotify(INVALID_NOTIFY_ID), ProtocolType::SDMA));
        TaskWaitAICPU* wait
            = AddNode(std::make_unique<TaskWaitAICPU>(MakeAicpuNotify(INVALID_NOTIFY_ID), ProtocolType::SDMA));
        AddEdge(start, post);
        AddEdge(post, wait);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 0U);
        EXPECT_EQ(stats.pairCount, 0U);
        EXPECT_EQ(stats.checkedBucketCount, 0U);
    }

    TEST_F(SyncConflictTest, DifferentAicpuResourcesAreCheckedIndependently)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post1 = AddRecord(1);
        TaskWaitAICPU* wait1 = AddWait(1);
        TaskRecordAICPU* post2 = AddRecord(2);
        TaskWaitAICPU* wait2 = AddWait(2);
        AddEdge(start, post1);
        AddEdge(post1, wait1);
        AddEdge(start, post2);
        AddEdge(post2, wait2);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 2U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 0U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, FewerPostsThanWaitsIsRejectedBeforePeerCheck)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post = AddRecord(1);
        TaskWaitAICPU* wait1 = AddWait(1);
        TaskWaitAICPU* wait2 = AddWait(1);
        AddEdge(start, post);
        AddEdge(post, wait1);
        AddEdge(post, wait2);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, CycleInCopiedSyncGraphIsRejected)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post = AddRecord(1);
        TaskWaitAICPU* wait = AddWait(1);
        AddEdge(start, post);
        AddEdge(post, wait);
        AddEdge(wait, post);

        EXPECT_NE(CheckSyncResourceConflict(start), HCCL_SUCCESS);
    }

    TEST_F(SyncConflictTest, OrderedReuseReachesNextPostAndStartWait)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post1 = AddRecord(1);
        TaskWaitAICPU* wait1 = AddWait(1);
        TaskRecordAICPU* post2 = AddRecord(1);
        TaskNode* order2 = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH));
        TaskWaitAICPU* wait2 = AddWait(1);
        AddEdge(start, post1);
        AddEdge(post1, wait1);
        AddEdge(wait1, post2);
        AddEdge(wait1, order2);
        AddEdge(post2, order2);
        AddEdge(post2, wait2);
        AddEdge(order2, wait2);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, MissingReceiveWaitToNextPostIsRejected)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post1 = AddRecord(1);
        TaskWaitAICPU* wait1 = AddWait(1);
        TaskRecordAICPU* post2 = AddRecord(1);
        TaskWaitAICPU* wait2 = AddWait(1);
        AddEdge(start, post1);
        AddEdge(post1, wait1);
        AddEdge(start, post2);
        AddEdge(post2, wait2);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, MissingReceiveWaitToNextStartWaitIsRejected)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post1 = AddRecord(1);
        TaskWaitAICPU* wait1 = AddWait(1);
        TaskRecordAICPU* post2 = AddRecord(1);
        TaskNode* unrelatedOrder = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH));
        TaskWaitAICPU* wait2 = AddWait(1);
        AddEdge(start, post1);
        AddEdge(post1, wait1);
        AddEdge(wait1, post2);
        AddEdge(start, unrelatedOrder);
        AddEdge(post2, wait2);
        AddEdge(unrelatedOrder, wait2);

        TaskPosition waitLane;
        waitLane.rankId = 0;
        waitLane.streamId = 0;
        waitLane.queueId = 0;
        wait2->SetPosition(waitLane);
        unrelatedOrder->SetPosition(waitLane);

        TaskPosition syncLane;
        syncLane.rankId = 1;
        syncLane.streamId = 1;
        syncLane.queueId = 1;
        post2->SetPosition(syncLane);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, PairedPostOnSameLaneAlsoReachesNextStartWait)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post1 = AddRecord(1);
        TaskWaitAICPU* wait1 = AddWait(1);
        TaskRecordAICPU* post2 = AddRecord(1);
        TaskNode* unrelatedOrder = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH));
        TaskWaitAICPU* wait2 = AddWait(1);
        AddEdge(start, post1);
        AddEdge(post1, wait1);
        AddEdge(wait1, post2);
        AddEdge(start, unrelatedOrder);
        AddEdge(post2, wait2);
        AddEdge(unrelatedOrder, wait2);

        TaskPosition waitLane;
        waitLane.rankId = 0;
        waitLane.streamId = 0;
        waitLane.queueId = 0;
        post2->SetPosition(waitLane);
        wait2->SetPosition(waitLane);

        TaskPosition unrelatedLane;
        unrelatedLane.rankId = 1;
        unrelatedLane.streamId = 1;
        unrelatedLane.queueId = 1;
        unrelatedOrder->SetPosition(unrelatedLane);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, StartWaitLaneMatchRequiresRankStreamAndQueue)
    {
        const TaskPosition waitLane = MakeLane(0);
        const std::vector<TaskPosition> mismatchedLanes{
            MakePosition(waitLane.rankId + 1U, waitLane.streamId, waitLane.queueId),
            MakePosition(waitLane.rankId, waitLane.streamId + 1U, waitLane.queueId),
            MakePosition(waitLane.rankId, waitLane.streamId, waitLane.queueId + 1U),
        };

        for (const TaskPosition& syncLane : mismatchedLanes) {
            TaskStart* start = AddMainStart();
            TaskRecordAICPU* post1 = AddRecord(1);
            TaskWaitAICPU* wait1 = AddWait(1);
            TaskRecordAICPU* post2 = AddRecord(1, syncLane);
            TaskNode* unrelatedOrder = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH), waitLane);
            TaskWaitAICPU* wait2 = AddWait(1, waitLane);
            AddEdge(start, post1);
            AddEdge(post1, wait1);
            AddEdge(wait1, post2);
            AddEdge(start, unrelatedOrder);
            AddEdge(post2, wait2);
            AddEdge(unrelatedOrder, wait2);

            SyncConflictCheckStats stats;
            EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
            EXPECT_EQ(stats.checkedBucketCount, 1U);
            EXPECT_EQ(stats.conflictCount, 1U);
        }
    }

    TEST_F(SyncConflictTest, OnePostToMultipleWaitsIsRejected)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post = AddRecord(1);
        TaskWaitAICPU* wait1 = AddWait(1);
        TaskWaitAICPU* wait2 = AddWait(1);
        AddEdge(start, post);
        AddEdge(post, wait1);
        AddEdge(post, wait2);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, MultiplePostsToOneWaitIsRejected)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* post1 = AddRecord(1);
        TaskRecordAICPU* post2 = AddRecord(1);
        TaskWaitAICPU* wait = AddWait(1);
        AddEdge(start, post1);
        AddEdge(start, post2);
        AddEdge(post1, wait);
        AddEdge(post2, wait);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, ResidualPostBeforeFirstPairIsRejected)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* residualPost = AddRecord(1);
        TaskRecordAICPU* pairedPost = AddRecord(1);
        TaskWaitAICPU* wait = AddWait(1);
        AddEdge(start, residualPost);
        AddEdge(start, pairedPost);
        AddEdge(pairedPost, wait);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, ResidualPostNotOrderedAfterReceiveWaitIsRejected)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* pairedPost = AddRecord(1);
        TaskWaitAICPU* wait = AddWait(1);
        TaskRecordAICPU* residualPost = AddRecord(1);
        AddEdge(start, pairedPost);
        AddEdge(pairedPost, wait);
        AddEdge(start, residualPost);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, MultipleResidualPostsAfterReceiveWaitAreAllowed)
    {
        TaskStart* start = AddMainStart();
        TaskRecordAICPU* pairedPost = AddRecord(1);
        TaskWaitAICPU* wait = AddWait(1);
        TaskRecordAICPU* residualPost1 = AddRecord(1);
        TaskRecordAICPU* residualPost2 = AddRecord(1);
        AddEdge(start, pairedPost);
        AddEdge(pairedPost, wait);
        AddEdge(wait, residualPost1);
        AddEdge(residualPost1, residualPost2);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.pairCount, 3U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, CcuMaskBitsUseIndependentResourceBuckets)
    {
        TaskStart* start = AddMainStart();
        TaskRecordCCU* post = AddCcuRecord(0x3);
        TaskWaitCCU* wait = AddCcuWait(0x3);
        AddEdge(start, post);
        AddEdge(post, wait);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 2U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 0U);
    }

    TEST_F(SyncConflictTest, CcuMultiplePostsToOneWaitIsRejected)
    {
        TaskStart* start = AddMainStart();
        TaskRecordCCU* post1 = AddCcuRecord(0x1);
        TaskRecordCCU* post2 = AddCcuRecord(0x1);
        TaskWaitCCU* wait = AddCcuWait(0x1);
        AddEdge(start, post1);
        AddEdge(start, post2);
        AddEdge(post1, wait);
        AddEdge(post2, wait);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, CcuOnePostToMultipleWaitsIsRejected)
    {
        TaskStart* start = AddMainStart();
        TaskRecordCCU* post = AddCcuRecord(0x1);
        TaskWaitCCU* wait1 = AddCcuWait(0x1);
        TaskWaitCCU* wait2 = AddCcuWait(0x1);
        AddEdge(start, post);
        AddEdge(post, wait1);
        AddEdge(post, wait2);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, CcuOrderedReuseAcrossMultiplePairsIsAllowed)
    {
        TaskStart* start = AddMainStart();
        const TaskPosition lane = MakeLane(0);
        TaskRecordCCU* post1 = AddNode(
            std::make_unique<TaskRecordCCU>(CcuNotify{INVALID_CHANNEL_ID, 0, 0, 0, 1, 1}, ProtocolType::CCU), lane);
        TaskWaitCCU* wait1 = AddNode(
            std::make_unique<TaskWaitCCU>(CcuNotify{INVALID_CHANNEL_ID, 0, 0, 0, 1, 1}, ProtocolType::CCU), lane);
        TaskRecordCCU* post2 = AddNode(
            std::make_unique<TaskRecordCCU>(CcuNotify{INVALID_CHANNEL_ID, 0, 0, 0, 1, 1}, ProtocolType::CCU), lane);
        TaskWaitCCU* wait2 = AddNode(
            std::make_unique<TaskWaitCCU>(CcuNotify{INVALID_CHANNEL_ID, 0, 0, 0, 1, 1}, ProtocolType::CCU), lane);
        AddEdge(start, post1);
        AddEdge(post1, wait1);
        AddEdge(wait1, post2);
        AddEdge(post2, wait2);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, CcuResourceFieldsCreateIndependentBuckets)
    {
        auto makeNotify = [](RankId waitRankId, uint32_t dieId, uint16_t ckeId) {
            CcuNotify notify;
            notify.waitRankId = waitRankId;
            notify.dieId = dieId;
            notify.ckeId = ckeId;
            notify.ckeMask = 0x1;
            return notify;
        };

        TaskStart* start = AddMainStart();
        const TaskPosition lane = MakeLane(0);
        const std::vector<CcuNotify> resources{
            makeNotify(0, 0, 1),
            makeNotify(1, 0, 1),
            makeNotify(0, 1, 1),
            makeNotify(0, 0, 2),
        };
        for (const CcuNotify& notify : resources) {
            TaskRecordCCU* post = AddNode(std::make_unique<TaskRecordCCU>(notify, ProtocolType::CCU), lane);
            TaskWaitCCU* wait = AddNode(std::make_unique<TaskWaitCCU>(notify, ProtocolType::CCU), lane);
            AddEdge(start, post);
            AddEdge(post, wait);
        }

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, resources.size());
        EXPECT_EQ(stats.pairCount, resources.size());
        EXPECT_EQ(stats.checkedBucketCount, 0U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, CcuZeroMaskIsIgnored)
    {
        TaskStart* start = AddMainStart();
        TaskRecordCCU* post = AddCcuRecord(0x0);
        TaskWaitCCU* wait = AddCcuWait(0x0);
        AddEdge(start, post);
        AddEdge(post, wait);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 0U);
        EXPECT_EQ(stats.pairCount, 0U);
        EXPECT_EQ(stats.checkedBucketCount, 0U);
    }

    TEST_F(SyncConflictTest, AivEventAndFlagResourcesAreDistinct)
    {
        TaskStart* start = AddMainStart();
        AivPipeEvent event;
        event.rankId = 0;
        event.launchIdx = 1;
        event.blockId = 2;
        event.srcPipe = 3;
        event.dstPipe = 4;
        event.eventId = 5;
        TaskAivSetFlag* setFlag = AddNode(std::make_unique<TaskAivSetFlag>(event));
        TaskAivWaitFlag* waitFlag = AddNode(std::make_unique<TaskAivWaitFlag>(event));

        AivFlagSync flag;
        flag.currentRank = 0;
        flag.flagOwnerRank = 0;
        flag.launchIdx = 1;
        flag.blockId = 2;
        flag.commInfoOffset = 0x100;
        TaskAivSendFlag* sendFlag = AddNode(std::make_unique<TaskAivSendFlag>(flag));
        TaskAivRecvFlag* recvFlag = AddNode(std::make_unique<TaskAivRecvFlag>(flag));

        AddEdge(start, setFlag);
        AddEdge(setFlag, waitFlag);
        AddEdge(start, sendFlag);
        AddEdge(sendFlag, recvFlag);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 2U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
    }

    TEST_F(SyncConflictTest, AivEventMultiplePostsToOneWaitIsRejected)
    {
        TaskStart* start = AddMainStart();
        const AivPipeEvent event = MakeEvent(7);
        TaskAivSetFlag* set1 = AddNode(std::make_unique<TaskAivSetFlag>(event));
        TaskAivSetFlag* set2 = AddNode(std::make_unique<TaskAivSetFlag>(event));
        TaskAivWaitFlag* wait = AddNode(std::make_unique<TaskAivWaitFlag>(event));
        AddEdge(start, set1);
        AddEdge(start, set2);
        AddEdge(set1, wait);
        AddEdge(set2, wait);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, AivEventOnePostToMultipleWaitsIsRejected)
    {
        TaskStart* start = AddMainStart();
        const AivPipeEvent event = MakeEvent(7);
        TaskAivSetFlag* setFlag = AddNode(std::make_unique<TaskAivSetFlag>(event));
        TaskAivWaitFlag* wait1 = AddNode(std::make_unique<TaskAivWaitFlag>(event));
        TaskAivWaitFlag* wait2 = AddNode(std::make_unique<TaskAivWaitFlag>(event));
        AddEdge(start, setFlag);
        AddEdge(setFlag, wait1);
        AddEdge(setFlag, wait2);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, AivEventOrderedReuseAcrossMultiplePairsIsAllowed)
    {
        TaskStart* start = AddMainStart();
        const AivPipeEvent event = MakeEvent(7);
        const TaskPosition lane = MakeLane(1);
        TaskAivSetFlag* set1 = AddNode(std::make_unique<TaskAivSetFlag>(event), lane);
        TaskAivWaitFlag* wait1 = AddNode(std::make_unique<TaskAivWaitFlag>(event), lane);
        TaskAivSetFlag* set2 = AddNode(std::make_unique<TaskAivSetFlag>(event), lane);
        TaskAivWaitFlag* wait2 = AddNode(std::make_unique<TaskAivWaitFlag>(event), lane);
        AddEdge(start, set1);
        AddEdge(set1, wait1);
        AddEdge(wait1, set2);
        AddEdge(set2, wait2);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, AivEventWaitWithoutMatchingPostIsRejected)
    {
        TaskStart* start = AddMainStart();
        TaskAivSetFlag* set = AddNode(std::make_unique<TaskAivSetFlag>(MakeEvent(1)));
        TaskAivWaitFlag* wait = AddNode(std::make_unique<TaskAivWaitFlag>(MakeEvent(2)));
        AddEdge(start, set);
        AddEdge(set, wait);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 2U);
        EXPECT_EQ(stats.pairCount, 1U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, AivEventFieldsCreateIndependentBuckets)
    {
        const std::vector<AivPipeEvent> resources{
            MakeEvent(1, 0, 1, 2, 3, 4), MakeEvent(1, 1, 1, 2, 3, 4), MakeEvent(1, 0, 2, 2, 3, 4),
            MakeEvent(1, 0, 1, 3, 3, 4), MakeEvent(1, 0, 1, 2, 4, 4), MakeEvent(1, 0, 1, 2, 3, 5),
            MakeEvent(2, 0, 1, 2, 3, 4),
        };

        TaskStart* start = AddMainStart();
        for (const AivPipeEvent& event : resources) {
            TaskAivSetFlag* setFlag = AddNode(std::make_unique<TaskAivSetFlag>(event));
            TaskAivWaitFlag* waitFlag = AddNode(std::make_unique<TaskAivWaitFlag>(event));
            AddEdge(start, setFlag);
            AddEdge(setFlag, waitFlag);
        }

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, resources.size());
        EXPECT_EQ(stats.pairCount, resources.size());
        EXPECT_EQ(stats.checkedBucketCount, 0U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, AivFlagMultiplePostsToOneWaitIsRejected)
    {
        TaskStart* start = AddMainStart();
        const AivFlagSync flag = MakeFlag(0x200);
        TaskAivSendFlag* send1 = AddNode(std::make_unique<TaskAivSendFlag>(flag));
        TaskAivSendFlag* send2 = AddNode(std::make_unique<TaskAivSendFlag>(flag));
        TaskAivRecvFlag* recv = AddNode(std::make_unique<TaskAivRecvFlag>(flag));
        AddEdge(start, send1);
        AddEdge(start, send2);
        AddEdge(send1, recv);
        AddEdge(send2, recv);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, AivFlagOnePostToMultipleWaitsIsAllowed)
    {
        TaskStart* start = AddMainStart();
        const AivFlagSync flag = MakeFlag(0x200);
        TaskAivSendFlag* send = AddNode(std::make_unique<TaskAivSendFlag>(flag));
        TaskAivRecvFlag* recv1 = AddNode(std::make_unique<TaskAivRecvFlag>(flag));
        TaskAivRecvFlag* recv2 = AddNode(std::make_unique<TaskAivRecvFlag>(flag));
        AddEdge(start, send);
        AddEdge(send, recv1);
        AddEdge(send, recv2);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, AivFlagOrderedReuseAcrossMultiplePairsIsAllowed)
    {
        TaskStart* start = AddMainStart();
        const AivFlagSync flag = MakeFlag(0x200);
        const TaskPosition lane = MakeLane(2);
        TaskAivSendFlag* send1 = AddNode(std::make_unique<TaskAivSendFlag>(flag), lane);
        TaskAivRecvFlag* recv1 = AddNode(std::make_unique<TaskAivRecvFlag>(flag), lane);
        TaskAivSendFlag* send2 = AddNode(std::make_unique<TaskAivSendFlag>(flag), lane);
        TaskAivRecvFlag* recv2 = AddNode(std::make_unique<TaskAivRecvFlag>(flag), lane);
        AddEdge(start, send1);
        AddEdge(send1, recv1);
        AddEdge(recv1, send2);
        AddEdge(send2, recv2);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, AivFlagFieldsCreateIndependentBuckets)
    {
        const std::vector<AivFlagSync> resources{
            MakeFlag(0x100, 0, 1),
            MakeFlag(0x100, 1, 1),
            MakeFlag(0x100, 0, 2),
            MakeFlag(0x200, 0, 1),
        };

        TaskStart* start = AddMainStart();
        for (const AivFlagSync& flag : resources) {
            TaskAivSendFlag* sendFlag = AddNode(std::make_unique<TaskAivSendFlag>(flag));
            TaskAivRecvFlag* recvFlag = AddNode(std::make_unique<TaskAivRecvFlag>(flag));
            AddEdge(start, sendFlag);
            AddEdge(sendFlag, recvFlag);
        }

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, resources.size());
        EXPECT_EQ(stats.pairCount, resources.size());
        EXPECT_EQ(stats.checkedBucketCount, resources.size());
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, AivFlagValuesCreateIndependentBuckets)
    {
        TaskStart* start = AddMainStart();
        const AivFlagSync flagValue1 = MakeFlag(0xa000, 0, 0, 1);
        const AivFlagSync flagValue2 = MakeFlag(0xa000, 0, 0, 2);
        TaskAivSendFlag* send1 = AddNode(std::make_unique<TaskAivSendFlag>(flagValue1));
        TaskAivRecvFlag* recv1 = AddNode(std::make_unique<TaskAivRecvFlag>(flagValue1));
        TaskAivSendFlag* send2 = AddNode(std::make_unique<TaskAivSendFlag>(flagValue2));
        TaskAivRecvFlag* recv2 = AddNode(std::make_unique<TaskAivRecvFlag>(flagValue2));
        AddEdge(start, send1);
        AddEdge(send1, recv1);
        AddEdge(start, send2);
        AddEdge(recv1, send2);
        AddEdge(send2, recv2);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 2U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 2U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, AivFlagDifferentValuesAllowConcurrentReceives)
    {
        TaskStart* start = AddMainStart();
        const AivFlagSync firstFlag = MakeFlag(0xa000, 0, 0, 1);
        const AivFlagSync secondFlag = MakeFlag(0xa000, 0, 0, 2);
        const TaskPosition producerLane = MakeLane(3);
        const TaskPosition firstReceiveLane = MakeLane(3);
        const TaskPosition secondReceiveLane = MakeLane(0);

        TaskAivSendFlag* send1 = AddNode(std::make_unique<TaskAivSendFlag>(firstFlag), producerLane);
        TaskAivRecvFlag* recv1 = AddNode(std::make_unique<TaskAivRecvFlag>(firstFlag), firstReceiveLane);
        TaskAivSendFlag* send2 = AddNode(std::make_unique<TaskAivSendFlag>(secondFlag), producerLane);
        TaskAivRecvFlag* recv2 = AddNode(std::make_unique<TaskAivRecvFlag>(secondFlag), secondReceiveLane);
        AddEdge(start, send1);
        AddEdge(send1, recv1);
        AddEdge(recv1, send2);
        AddEdge(send2, recv2);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 2U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 2U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, AivFlagDifferentValuesStillRequireProducerOrdering)
    {
        TaskStart* start = AddMainStart();
        const AivFlagSync firstFlag = MakeFlag(0xa000, 0, 0, 1);
        const AivFlagSync secondFlag = MakeFlag(0xa000, 0, 0, 2);
        const TaskPosition firstLane = MakeLane(3);
        const TaskPosition secondReceiveLane = MakeLane(0);

        TaskAivSendFlag* send1 = AddNode(std::make_unique<TaskAivSendFlag>(firstFlag), firstLane);
        TaskAivRecvFlag* recv1 = AddNode(std::make_unique<TaskAivRecvFlag>(firstFlag), firstLane);
        TaskAivSendFlag* send2 = AddNode(std::make_unique<TaskAivSendFlag>(secondFlag), firstLane);
        TaskAivRecvFlag* recv2 = AddNode(std::make_unique<TaskAivRecvFlag>(secondFlag), secondReceiveLane);
        AddEdge(start, send1);
        AddEdge(send1, recv1);
        AddEdge(start, send2);
        AddEdge(send2, recv2);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 2U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 2U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, AivFlagFanOutGroupsRequireAllPreviousReceivesToReachNextGroup)
    {
        TaskStart* start = AddMainStart();
        const AivFlagSync flag = MakeFlag(0xa000, 0, 1, 7);
        const TaskPosition lane = MakeLane(0);
        TaskAivSendFlag* send1 = AddNode(std::make_unique<TaskAivSendFlag>(flag), lane);
        TaskAivRecvFlag* recv1 = AddNode(std::make_unique<TaskAivRecvFlag>(flag), lane);
        TaskAivRecvFlag* recv2 = AddNode(std::make_unique<TaskAivRecvFlag>(flag), lane);
        TaskAivSendFlag* send2 = AddNode(std::make_unique<TaskAivSendFlag>(flag), lane);
        TaskAivRecvFlag* recv3 = AddNode(std::make_unique<TaskAivRecvFlag>(flag), lane);
        TaskAivRecvFlag* recv4 = AddNode(std::make_unique<TaskAivRecvFlag>(flag), lane);
        AddEdge(start, send1);
        AddEdge(send1, recv1);
        AddEdge(send1, recv2);
        AddEdge(recv1, send2);
        AddEdge(recv2, send2);
        AddEdge(send2, recv3);
        AddEdge(send2, recv4);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 4U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, AivFlagFanOutGroupsRejectMissingPreviousReceiveOrdering)
    {
        TaskStart* start = AddMainStart();
        const AivFlagSync flag = MakeFlag(0xa000, 0, 1, 7);
        const TaskPosition lane = MakeLane(0);
        TaskAivSendFlag* send1 = AddNode(std::make_unique<TaskAivSendFlag>(flag), lane);
        TaskAivRecvFlag* recv1 = AddNode(std::make_unique<TaskAivRecvFlag>(flag), lane);
        TaskAivRecvFlag* recv2 = AddNode(std::make_unique<TaskAivRecvFlag>(flag), lane);
        TaskAivSendFlag* send2 = AddNode(std::make_unique<TaskAivSendFlag>(flag), lane);
        TaskAivRecvFlag* recv3 = AddNode(std::make_unique<TaskAivRecvFlag>(flag), lane);
        TaskAivRecvFlag* recv4 = AddNode(std::make_unique<TaskAivRecvFlag>(flag), lane);
        TaskEnd* order = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH));
        AddEdge(start, send1);
        AddEdge(send1, recv1);
        AddEdge(send1, recv2);
        AddEdge(recv1, send2);
        AddEdge(start, order);
        AddEdge(order, send2);
        AddEdge(send2, recv3);
        AddEdge(send2, recv4);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 4U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, UnmatchedAivPostIsRejected)
    {
        TaskStart* start = AddMainStart();
        AivPipeEvent event;
        event.rankId = 0;
        event.launchIdx = 1;
        event.blockId = 2;
        event.srcPipe = 3;
        event.dstPipe = 4;
        event.eventId = 5;
        TaskAivSetFlag* firstSet = AddNode(std::make_unique<TaskAivSetFlag>(event));
        TaskAivSetFlag* unmatchedSet = AddNode(std::make_unique<TaskAivSetFlag>(event));
        TaskAivWaitFlag* wait = AddNode(std::make_unique<TaskAivWaitFlag>(event));
        AddEdge(start, firstSet);
        AddEdge(firstSet, unmatchedSet);
        AddEdge(firstSet, wait);

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }

    TEST_F(SyncConflictTest, SingleUnmatchedAivPostIsAllowed)
    {
        TaskStart* start = AddMainStart();
        AivPipeEvent event;
        event.rankId = 0;
        event.launchIdx = 1;
        event.blockId = 2;
        event.srcPipe = 3;
        event.dstPipe = 4;
        TaskAivSetFlag* setFlag = AddNode(std::make_unique<TaskAivSetFlag>(event));
        AddEdge(start, setFlag);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 1U);
        EXPECT_EQ(stats.checkedBucketCount, 0U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, MultipleUnmatchedAivPostsAreNotPeerConflict)
    {
        TaskStart* start = AddMainStart();
        AivPipeEvent event;
        event.rankId = 0;
        event.launchIdx = 1;
        event.blockId = 2;
        event.srcPipe = 3;
        event.dstPipe = 4;
        TaskAivSetFlag* firstSet = AddNode(std::make_unique<TaskAivSetFlag>(event));
        TaskAivSetFlag* secondSet = AddNode(std::make_unique<TaskAivSetFlag>(event));
        AddEdge(start, firstSet);
        AddEdge(firstSet, secondSet);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 2U);
        EXPECT_EQ(stats.checkedBucketCount, 0U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, UnmatchedAivPostAfterWaitIsAllowed)
    {
        TaskStart* start = AddMainStart();
        AivPipeEvent event;
        event.rankId = 0;
        event.launchIdx = 1;
        event.blockId = 2;
        event.srcPipe = 3;
        event.dstPipe = 4;
        TaskAivSetFlag* firstSet = AddNode(std::make_unique<TaskAivSetFlag>(event));
        TaskAivWaitFlag* wait = AddNode(std::make_unique<TaskAivWaitFlag>(event));
        TaskAivSetFlag* unmatchedSet = AddNode(std::make_unique<TaskAivSetFlag>(event));
        TaskAivSetFlag* secondUnmatchedSet = AddNode(std::make_unique<TaskAivSetFlag>(event));
        AddEdge(start, firstSet);
        AddEdge(firstSet, wait);
        AddEdge(wait, unmatchedSet);
        AddEdge(unmatchedSet, secondUnmatchedSet);

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.resourceBucketCount, 1U);
        EXPECT_EQ(stats.pairCount, 3U);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, LargeMultiLaneGraphWithOrderedReuseIsAccepted)
    {
        TaskStart* start = AddMainStart();
        constexpr size_t laneCount = 4U;
        constexpr size_t roundCount = 10U;
        std::vector<TaskNode*> tails(laneCount, start);

        for (size_t round = 0; round < roundCount; ++round) {
            TaskNode* barrier = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH));
            if (round == 0U) {
                AddEdge(start, barrier);
            } else {
                for (TaskNode* tail : tails) {
                    AddEdge(tail, barrier);
                }
            }

            for (size_t lane = 0; lane < laneCount; ++lane) {
                const TaskPosition position = MakeLane(lane);
                TaskRecordAICPU* post = AddRecord(static_cast<uint32_t>(lane + 1U), position);
                TaskWaitAICPU* wait = AddWait(static_cast<uint32_t>(lane + 1U), position);
                AddEdge(barrier, post);
                AddEdge(post, wait);
                tails[lane] = wait;
            }
        }

        SyncConflictCheckStats stats;
        EXPECT_EQ(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.originalNodeCount, 91U);
        EXPECT_EQ(stats.copiedNodeCount, 131U);
        EXPECT_EQ(stats.resourceBucketCount, laneCount);
        EXPECT_EQ(stats.pairCount, laneCount * roundCount);
        EXPECT_EQ(stats.checkedBucketCount, laneCount);
        EXPECT_EQ(stats.conflictCount, 0U);
    }

    TEST_F(SyncConflictTest, LargeMultiLaneGraphWithOneBrokenDependencyIsRejected)
    {
        TaskStart* start = AddMainStart();
        constexpr size_t laneCount = 4U;
        constexpr size_t roundCount = 10U;
        constexpr size_t brokenRound = 5U;
        constexpr size_t brokenLane = 0U;
        std::vector<TaskNode*> tails(laneCount, start);

        for (size_t round = 0; round < roundCount; ++round) {
            TaskNode* barrier = AddNode(std::make_unique<TaskEnd>(BoundaryType::MAIN_GRAPH));
            if (round == 0U) {
                AddEdge(start, barrier);
            } else {
                for (size_t lane = 0; lane < laneCount; ++lane) {
                    if (round == brokenRound && lane == brokenLane) {
                        continue;
                    }
                    AddEdge(tails[lane], barrier);
                }
            }

            for (size_t lane = 0; lane < laneCount; ++lane) {
                const TaskPosition position = MakeLane(lane);
                TaskRecordAICPU* post = AddRecord(static_cast<uint32_t>(lane + 1U), position);
                TaskWaitAICPU* wait = AddWait(static_cast<uint32_t>(lane + 1U), position);
                AddEdge(barrier, post);
                AddEdge(post, wait);
                tails[lane] = wait;
            }
        }

        SyncConflictCheckStats stats;
        EXPECT_NE(CheckSyncResourceConflict(start, &stats), HCCL_SUCCESS);
        EXPECT_EQ(stats.originalNodeCount, 91U);
        EXPECT_EQ(stats.copiedNodeCount, 131U);
        EXPECT_EQ(stats.resourceBucketCount, laneCount);
        EXPECT_EQ(stats.pairCount, laneCount * roundCount);
        EXPECT_EQ(stats.checkedBucketCount, 1U);
        EXPECT_EQ(stats.conflictCount, 1U);
    }
} // namespace
} // namespace HcclSim
