/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"

#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <vector>

#include "sim_task.h"
#include "task_def.h"
#include "task_graph_generator.h"

using namespace HcclSim;

class TaskGraphGeneratorTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper to create a simple local copy task
    std::shared_ptr<TaskStubLocalCopy> MakeLocalCopy()
    {
        DataSlice src(BufferType::INPUT, 0, 1024);
        DataSlice dst(BufferType::OUTPUT, 0, 1024);
        return std::make_shared<TaskStubLocalCopy>(src, dst);
    }

    // Helper to create a simple read task
    std::shared_ptr<TaskStubRead> MakeRead(RankId remoteRank)
    {
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        return std::make_shared<TaskStubRead>(remoteRank, link, localSlice, remoteSlice);
    }

    // Helper to create a simple write task
    std::shared_ptr<TaskStubWrite> MakeWrite(RankId remoteRank)
    {
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        return std::make_shared<TaskStubWrite>(remoteRank, link, localSlice, remoteSlice);
    }

    // Helper to create a post task
    std::shared_ptr<TaskStubPost> MakePost(RankId remoteRank, uint32_t topicId)
    {
        LinkInfo link(LinkProtoStub::SDMA);
        return std::make_shared<TaskStubPost>(remoteRank, link, topicId);
    }

    // Helper to create a wait task
    std::shared_ptr<TaskStubWait> MakeWait(RankId remoteRank, uint32_t topicId)
    {
        LinkInfo link(LinkProtoStub::SDMA);
        return std::make_shared<TaskStubWait>(remoteRank, link, topicId);
    }

    // Helper to create a local post task
    std::shared_ptr<TaskStubLocalPostTo> MakeLocalPost(uint32_t notifyId)
    {
        return std::make_shared<TaskStubLocalPostTo>(notifyId);
    }

    // Helper to create a local wait task
    std::shared_ptr<TaskStubLocalWaitFrom> MakeLocalWait(uint32_t notifyId)
    {
        return std::make_shared<TaskStubLocalWaitFrom>(notifyId);
    }
};

// Test GenGraph with empty task queues
TEST_F(TaskGraphGeneratorTest, GenGraph_EmptyQueues)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;
    TaskNode dummyStart(nullptr, -1, 0, 0);

    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test GenGraph with single rank, single queue with local copy
TEST_F(TaskGraphGeneratorTest, GenGraph_SingleRankSingleQueue)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    SingleTaskQueue rankQueue;
    std::vector<std::shared_ptr<TaskStub>> queue;
    queue.push_back(MakeLocalCopy());
    rankQueue.push_back(queue);

    allRankTaskQueues[0] = rankQueue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(dummyStart.children.size(), 1u);
}

// Test GenGraph with single rank, multiple tasks in queue
TEST_F(TaskGraphGeneratorTest, GenGraph_SingleRankMultipleTasks)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    SingleTaskQueue rankQueue;
    std::vector<std::shared_ptr<TaskStub>> queue;
    queue.push_back(MakeLocalCopy());
    queue.push_back(MakeLocalCopy());
    queue.push_back(MakeLocalCopy());
    rankQueue.push_back(queue);

    allRankTaskQueues[0] = rankQueue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(dummyStart.children.size(), 1u);
}

// Test GenGraph with multiple ranks
TEST_F(TaskGraphGeneratorTest, GenGraph_MultipleRanks)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    // Rank 0
    SingleTaskQueue rank0Queue;
    std::vector<std::shared_ptr<TaskStub>> queue0;
    queue0.push_back(MakeLocalCopy());
    rank0Queue.push_back(queue0);
    allRankTaskQueues[0] = rank0Queue;

    // Rank 1
    SingleTaskQueue rank1Queue;
    std::vector<std::shared_ptr<TaskStub>> queue1;
    queue1.push_back(MakeLocalCopy());
    rank1Queue.push_back(queue1);
    allRankTaskQueues[1] = rank1Queue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test GenGraph with inter-rank post/wait
TEST_F(TaskGraphGeneratorTest, GenGraph_InterRankPostWait)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    // Rank 0: posts to rank 1
    SingleTaskQueue rank0Queue;
    std::vector<std::shared_ptr<TaskStub>> queue0;
    queue0.push_back(MakePost(1, 100));
    queue0.push_back(MakeLocalCopy());
    rank0Queue.push_back(queue0);
    allRankTaskQueues[0] = rank0Queue;

    // Rank 1: waits for rank 0
    SingleTaskQueue rank1Queue;
    std::vector<std::shared_ptr<TaskStub>> queue1;
    queue1.push_back(MakeWait(0, 100));
    queue1.push_back(MakeLocalCopy());
    rank1Queue.push_back(queue1);
    allRankTaskQueues[1] = rank1Queue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test GenGraph with local post/wait matching
TEST_F(TaskGraphGeneratorTest, GenGraph_LocalPostWaitMatch)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    SingleTaskQueue rankQueue;
    std::vector<std::shared_ptr<TaskStub>> queue;
    queue.push_back(MakeLocalPost(42));
    queue.push_back(MakeLocalWait(42));
    queue.push_back(MakeLocalCopy());
    rankQueue.push_back(queue);

    allRankTaskQueues[0] = rankQueue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test GenGraph with local post/wait mismatch (different notify IDs)
TEST_F(TaskGraphGeneratorTest, GenGraph_LocalPostWaitMismatch)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    SingleTaskQueue rankQueue;
    std::vector<std::shared_ptr<TaskStub>> queue;
    queue.push_back(MakeLocalPost(42));
    queue.push_back(MakeLocalWait(99)); // Different notify ID
    queue.push_back(MakeLocalCopy());
    rankQueue.push_back(queue);

    allRankTaskQueues[0] = rankQueue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    // Should fail due to deadlock/mismatch
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

// Test GenGraph with read task
TEST_F(TaskGraphGeneratorTest, GenGraph_ReadTask)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    SingleTaskQueue rankQueue;
    std::vector<std::shared_ptr<TaskStub>> queue;
    queue.push_back(MakeRead(1));
    rankQueue.push_back(queue);

    allRankTaskQueues[0] = rankQueue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test GenGraph with write task
TEST_F(TaskGraphGeneratorTest, GenGraph_WriteTask)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    SingleTaskQueue rankQueue;
    std::vector<std::shared_ptr<TaskStub>> queue;
    queue.push_back(MakeWrite(1));
    rankQueue.push_back(queue);

    allRankTaskQueues[0] = rankQueue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test GenGraph with multiple queues per rank
TEST_F(TaskGraphGeneratorTest, GenGraph_MultipleQueuesPerRank)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    SingleTaskQueue rankQueue;
    // Master queue
    std::vector<std::shared_ptr<TaskStub>> masterQueue;
    masterQueue.push_back(MakeLocalCopy());
    rankQueue.push_back(masterQueue);

    // Slave queue (empty, should be skipped)
    std::vector<std::shared_ptr<TaskStub>> slaveQueue;
    rankQueue.push_back(slaveQueue);

    allRankTaskQueues[0] = rankQueue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test GenGraph with read and write between ranks
TEST_F(TaskGraphGeneratorTest, GenGraph_ReadWriteBetweenRanks)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    // Rank 0: writes to rank 1
    SingleTaskQueue rank0Queue;
    std::vector<std::shared_ptr<TaskStub>> queue0;
    queue0.push_back(MakeWrite(1));
    rank0Queue.push_back(queue0);
    allRankTaskQueues[0] = rank0Queue;

    // Rank 1: reads from rank 0
    SingleTaskQueue rank1Queue;
    std::vector<std::shared_ptr<TaskStub>> queue1;
    queue1.push_back(MakeRead(0));
    rank1Queue.push_back(queue1);
    allRankTaskQueues[1] = rank1Queue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test GenGraph with post/wait deadlock (unmatched post)
TEST_F(TaskGraphGeneratorTest, GenGraph_UnmatchedPost)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    // Rank 0: posts to rank 1 but rank 1 doesn't have matching wait
    SingleTaskQueue rank0Queue;
    std::vector<std::shared_ptr<TaskStub>> queue0;
    queue0.push_back(MakePost(1, 100));
    rank0Queue.push_back(queue0);
    allRankTaskQueues[0] = rank0Queue;

    // Rank 1: no wait for rank 0
    SingleTaskQueue rank1Queue;
    std::vector<std::shared_ptr<TaskStub>> queue1;
    queue1.push_back(MakeLocalCopy());
    rank1Queue.push_back(queue1);
    allRankTaskQueues[1] = rank1Queue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    // Should fail due to unmatched post
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

// Test GenGraph with local post/wait in multi-queue setup
TEST_F(TaskGraphGeneratorTest, GenGraph_LocalPostWaitMultiQueue)
{
    TaskGraphGenerator generator;
    AllRankTaskQueues allRankTaskQueues;

    SingleTaskQueue rankQueue;
    // Master queue with local post
    std::vector<std::shared_ptr<TaskStub>> masterQueue;
    masterQueue.push_back(MakeLocalPost(1));
    masterQueue.push_back(MakeLocalCopy());
    rankQueue.push_back(masterQueue);

    // Slave queue with local wait
    std::vector<std::shared_ptr<TaskStub>> slaveQueue;
    slaveQueue.push_back(MakeLocalWait(1));
    slaveQueue.push_back(MakeLocalCopy());
    rankQueue.push_back(slaveQueue);

    allRankTaskQueues[0] = rankQueue;

    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = generator.GenGraph(allRankTaskQueues, &dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
