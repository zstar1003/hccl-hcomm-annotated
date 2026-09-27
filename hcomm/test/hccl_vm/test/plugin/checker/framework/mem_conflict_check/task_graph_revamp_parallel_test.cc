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
#include <queue>
#include <set>
#include <vector>

#include "data_slice.h"
#include "sim_task.h"
#include "task_ccu.h"
#include "task_def.h"
#include "task_graph_revamp_parallel.h"

#define private public
#undef private
using namespace HcclSim;

namespace {
void BuildParentChild(TaskNodePtr parent, TaskNodePtr child)
{
    parent->children.push_back(child);
    child->parents.push_back(parent);
}
} // namespace

class GraphRevampParallelTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(GraphRevampParallelTest, Revamp_EmptyGraph)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampParallelTest, Revamp_SingleRankNoCcu)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &copyNode);

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampParallelTest, Revamp_CcuGraphEmptyLoopGroupAndParallelNodes)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampParallelTest, Revamp_CcuGraphWithSingleLoop)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    TaskStubLoopStart loopStartStub(0, 0);
    TaskStubLoopEnd loopEndStub(0, 0);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy innerCopy(srcSlice, dstSlice);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode loopStartNode(&loopStartStub, 0, 0, 1);
    TaskNode innerNode(&innerCopy, 0, 0, 2);
    TaskNode loopEndNode(&loopEndStub, 0, 0, 3);
    TaskNode afterNode(&afterCopy, 0, 0, 4);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &loopStartNode);
    BuildParentChild(&loopStartNode, &innerNode);
    BuildParentChild(&innerNode, &loopEndNode);
    BuildParentChild(&loopEndNode, &afterNode);

    ccuTask.loopGroupInfo_.push_back({LoopInfo(&loopStartNode, &loopEndNode)});
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    EXPECT_TRUE(beforeNode.loopParallel);
    EXPECT_TRUE(afterNode.loopParallel);
    EXPECT_TRUE(loopStartNode.loopParallel);
    EXPECT_TRUE(innerNode.loopParallel);
    EXPECT_TRUE(loopEndNode.loopParallel);
    EXPECT_NE(loopStartNode.queIdx, beforeNode.queIdx);
    EXPECT_NE(revamp.toDeleteTaskResource_.size(), 0u);
    EXPECT_NE(revamp.toDeleteTaskNodeResource_.size(), 0u);
}

TEST_F(GraphRevampParallelTest, Revamp_CcuGraphWithTwoLoopsInSameGroup)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    TaskStubLoopStart loopStart1(0, 0);
    TaskStubLoopEnd loopEnd1(0, 0);
    TaskStubLoopStart loopStart2(1, 0);
    TaskStubLoopEnd loopEnd2(1, 0);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy innerCopy1(srcSlice, dstSlice);
    TaskStubLocalCopy innerCopy2(srcSlice, dstSlice);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode loopStart1Node(&loopStart1, 0, 0, 1);
    TaskNode inner1Node(&innerCopy1, 0, 0, 2);
    TaskNode loopEnd1Node(&loopEnd1, 0, 0, 3);
    TaskNode loopStart2Node(&loopStart2, 0, 0, 4);
    TaskNode inner2Node(&innerCopy2, 0, 0, 5);
    TaskNode loopEnd2Node(&loopEnd2, 0, 0, 6);
    TaskNode afterNode(&afterCopy, 0, 0, 7);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &loopStart1Node);
    BuildParentChild(&loopStart1Node, &inner1Node);
    BuildParentChild(&inner1Node, &loopEnd1Node);
    BuildParentChild(&loopEnd1Node, &loopStart2Node);
    BuildParentChild(&loopStart2Node, &inner2Node);
    BuildParentChild(&inner2Node, &loopEnd2Node);
    BuildParentChild(&loopEnd2Node, &afterNode);

    ccuTask.loopGroupInfo_.push_back(
        {LoopInfo(&loopStart1Node, &loopEnd1Node), LoopInfo(&loopStart2Node, &loopEnd2Node)});
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    EXPECT_TRUE(beforeNode.loopParallel);
    EXPECT_TRUE(afterNode.loopParallel);
    EXPECT_TRUE(loopStart1Node.loopParallel);
    EXPECT_TRUE(loopStart2Node.loopParallel);
    EXPECT_NE(loopStart1Node.queIdx, beforeNode.queIdx);
    EXPECT_NE(loopStart2Node.queIdx, beforeNode.queIdx);
}

TEST_F(GraphRevampParallelTest, Revamp_CcuGraphWithSingleAsyncNode)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy asyncCopy(srcSlice, dstSlice);
    TaskStubLocalPostTo postTo(0);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode(&asyncCopy, 0, 0, 1);
    TaskNode postNode(&postTo, 0, 0, 2);
    TaskNode waitNode(&waitFrom, 0, 0, 3);
    TaskNode afterNode(&afterCopy, 0, 0, 4);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode);
    BuildParentChild(&asyncNode, &postNode);
    BuildParentChild(&postNode, &waitNode);
    BuildParentChild(&waitNode, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.localPostWaitPairs_[&postNode] = &waitNode;
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    EXPECT_TRUE(beforeNode.asynParallel);
    EXPECT_NE(asyncNode.queIdx, beforeNode.queIdx);
    EXPECT_NE(revamp.toDeleteTaskResource_.size(), 0u);
    EXPECT_NE(revamp.toDeleteTaskNodeResource_.size(), 0u);
}

TEST_F(GraphRevampParallelTest, Revamp_CcuGraphWithAsyncNodeNoLocalPost)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy asyncCopy(srcSlice, dstSlice);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode(&asyncCopy, 0, 0, 1);
    TaskNode waitNode(&waitFrom, 0, 0, 2);
    TaskNode afterNode(&afterCopy, 0, 0, 3);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode);
    BuildParentChild(&asyncNode, &waitNode);
    BuildParentChild(&waitNode, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(beforeNode.asynParallel);
}

TEST_F(GraphRevampParallelTest, Revamp_CcuGraphWithMultipleAsyncNodes)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy asyncCopy1(srcSlice, dstSlice);
    TaskStubLocalCopy asyncCopy2(srcSlice, dstSlice);
    TaskStubLocalPostTo post1(0);
    TaskStubLocalWaitFrom wait1(0);
    TaskStubLocalPostTo post2(1);
    TaskStubLocalWaitFrom wait2(1);
    TaskStubLocalWaitFrom otherWait(2);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode1(&asyncCopy1, 0, 0, 1);
    TaskNode postNode1(&post1, 0, 0, 2);
    TaskNode waitNode1(&wait1, 0, 0, 3);
    TaskNode otherWaitNode(&otherWait, 0, 0, 4);
    TaskNode asyncNode2(&asyncCopy2, 0, 0, 5);
    TaskNode postNode2(&post2, 0, 0, 6);
    TaskNode waitNode2(&wait2, 0, 0, 7);
    TaskNode afterNode(&afterCopy, 0, 0, 8);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode1);
    BuildParentChild(&asyncNode1, &postNode1);
    BuildParentChild(&postNode1, &waitNode1);
    BuildParentChild(&postNode1, &otherWaitNode);
    BuildParentChild(&waitNode1, &asyncNode2);
    BuildParentChild(&asyncNode2, &postNode2);
    BuildParentChild(&postNode2, &waitNode2);
    BuildParentChild(&waitNode2, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode1);
    ccuTask.parallelNodes_.push_back(&asyncNode2);
    ccuTask.localPostWaitPairs_[&postNode1] = &waitNode1;
    ccuTask.localPostWaitPairs_[&postNode2] = &waitNode2;
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    EXPECT_TRUE(beforeNode.asynParallel);
    EXPECT_NE(asyncNode1.queIdx, beforeNode.queIdx);
    EXPECT_NE(asyncNode2.queIdx, beforeNode.queIdx);
}

TEST_F(GraphRevampParallelTest, Revamp_MultiRankWithCcuGraph)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);

    TaskStubCcuGraph ccuTask0(static_cast<RankId>(0));
    TaskNode ccuNode0(&ccuTask0, 0, 0, 0);
    TaskStubCcuGraph ccuTask1(static_cast<RankId>(1));
    TaskNode ccuNode1(&ccuTask1, 1, 0, 0);

    BuildParentChild(&dummyStart, &ccuNode0);
    BuildParentChild(&dummyStart, &ccuNode1);

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampParallelTest, Revamp_CcuGraphWithLoopAndAsync)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    TaskStubLoopStart loopStartStub(0, 0);
    TaskStubLoopEnd loopEndStub(0, 0);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy innerCopy(srcSlice, dstSlice);
    TaskStubLocalCopy betweenCopy(srcSlice, dstSlice);
    TaskStubLocalCopy asyncCopy(srcSlice, dstSlice);
    TaskStubLocalPostTo postTo(0);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeLoopNode(&beforeCopy, 0, 0, 0);
    TaskNode loopStartNode(&loopStartStub, 0, 0, 1);
    TaskNode innerNode(&innerCopy, 0, 0, 2);
    TaskNode loopEndNode(&loopEndStub, 0, 0, 3);
    TaskNode betweenNode(&betweenCopy, 0, 0, 4);
    TaskNode asyncNode(&asyncCopy, 0, 0, 5);
    TaskNode postNode(&postTo, 0, 0, 6);
    TaskNode waitNode(&waitFrom, 0, 0, 7);
    TaskNode afterNode(&afterCopy, 0, 0, 8);

    BuildParentChild(&ccuNode, &beforeLoopNode);
    BuildParentChild(&beforeLoopNode, &loopStartNode);
    BuildParentChild(&loopStartNode, &innerNode);
    BuildParentChild(&innerNode, &loopEndNode);
    BuildParentChild(&loopEndNode, &betweenNode);
    BuildParentChild(&betweenNode, &asyncNode);
    BuildParentChild(&asyncNode, &postNode);
    BuildParentChild(&postNode, &waitNode);
    BuildParentChild(&waitNode, &afterNode);

    ccuTask.loopGroupInfo_.push_back({LoopInfo(&loopStartNode, &loopEndNode)});
    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.localPostWaitPairs_[&postNode] = &waitNode;
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    EXPECT_TRUE(loopStartNode.loopParallel);
    EXPECT_TRUE(betweenNode.asynParallel);
    EXPECT_NE(loopStartNode.queIdx, beforeLoopNode.queIdx);
    EXPECT_NE(asyncNode.queIdx, betweenNode.queIdx);
}

TEST_F(GraphRevampParallelTest, Revamp_CcuGraphWithReadAsyncNode)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubRead readStub(1, LinkInfo(LinkProtoStub::RDMA), srcSlice, dstSlice);
    TaskStubLocalPostTo postTo(0);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode(&readStub, 0, 0, 1);
    TaskNode postNode(&postTo, 0, 0, 2);
    TaskNode waitNode(&waitFrom, 0, 0, 3);
    TaskNode afterNode(&afterCopy, 0, 0, 4);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode);
    BuildParentChild(&asyncNode, &postNode);
    BuildParentChild(&postNode, &waitNode);
    BuildParentChild(&waitNode, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.localPostWaitPairs_[&postNode] = &waitNode;
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(beforeNode.asynParallel);
}

TEST_F(GraphRevampParallelTest, Revamp_CcuGraphWithWriteAsyncNode)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubWrite writeStub(1, LinkInfo(LinkProtoStub::RDMA), srcSlice, dstSlice);
    TaskStubLocalPostTo postTo(0);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode(&writeStub, 0, 0, 1);
    TaskNode postNode(&postTo, 0, 0, 2);
    TaskNode waitNode(&waitFrom, 0, 0, 3);
    TaskNode afterNode(&afterCopy, 0, 0, 4);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode);
    BuildParentChild(&asyncNode, &postNode);
    BuildParentChild(&postNode, &waitNode);
    BuildParentChild(&waitNode, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.localPostWaitPairs_[&postNode] = &waitNode;
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(beforeNode.asynParallel);
}

TEST_F(GraphRevampParallelTest, Revamp_CcuGraphWithLocalReduceAsyncNode)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalReduce reduceStub(srcSlice, dstSlice, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskStubLocalPostTo postTo(0);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode(&reduceStub, 0, 0, 1);
    TaskNode postNode(&postTo, 0, 0, 2);
    TaskNode waitNode(&waitFrom, 0, 0, 3);
    TaskNode afterNode(&afterCopy, 0, 0, 4);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode);
    BuildParentChild(&asyncNode, &postNode);
    BuildParentChild(&postNode, &waitNode);
    BuildParentChild(&waitNode, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.localPostWaitPairs_[&postNode] = &waitNode;
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(beforeNode.asynParallel);
}

TEST_F(GraphRevampParallelTest, Revamp_LoopWithMultipleInnerNodes)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    TaskStubLoopStart loopStartStub(0, 0);
    TaskStubLoopEnd loopEndStub(0, 0);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy copy1(srcSlice, dstSlice);
    TaskStubLocalCopy copy2(srcSlice, dstSlice);
    TaskStubLocalCopy copy3(srcSlice, dstSlice);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode loopStartNode(&loopStartStub, 0, 0, 1);
    TaskNode inner1(&copy1, 0, 0, 2);
    TaskNode inner2(&copy2, 0, 0, 3);
    TaskNode inner3(&copy3, 0, 0, 4);
    TaskNode loopEndNode(&loopEndStub, 0, 0, 5);
    TaskNode afterNode(&afterCopy, 0, 0, 6);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &loopStartNode);
    BuildParentChild(&loopStartNode, &inner1);
    BuildParentChild(&inner1, &inner2);
    BuildParentChild(&inner2, &inner3);
    BuildParentChild(&inner3, &loopEndNode);
    BuildParentChild(&loopEndNode, &afterNode);

    ccuTask.loopGroupInfo_.push_back({LoopInfo(&loopStartNode, &loopEndNode)});
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    EXPECT_TRUE(inner1.loopParallel);
    EXPECT_TRUE(inner2.loopParallel);
    EXPECT_TRUE(inner3.loopParallel);
    EXPECT_EQ(inner1.queIdx, loopStartNode.queIdx);
    EXPECT_EQ(inner2.queIdx, loopStartNode.queIdx);
    EXPECT_EQ(inner3.queIdx, loopStartNode.queIdx);
}

TEST_F(GraphRevampParallelTest, Revamp_TwoLoopGroups)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    TaskStubLoopStart loopStart1(0, 0);
    TaskStubLoopEnd loopEnd1(0, 0);
    TaskStubLoopStart loopStart2(0, 1);
    TaskStubLoopEnd loopEnd2(0, 1);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy1(srcSlice, dstSlice);
    TaskStubLocalCopy innerCopy1(srcSlice, dstSlice);
    TaskStubLocalCopy afterCopy1(srcSlice, dstSlice);
    TaskStubLocalCopy beforeCopy2(srcSlice, dstSlice);
    TaskStubLocalCopy innerCopy2(srcSlice, dstSlice);
    TaskStubLocalCopy afterCopy2(srcSlice, dstSlice);

    TaskNode beforeNode1(&beforeCopy1, 0, 0, 0);
    TaskNode loopStart1Node(&loopStart1, 0, 0, 1);
    TaskNode inner1Node(&innerCopy1, 0, 0, 2);
    TaskNode loopEnd1Node(&loopEnd1, 0, 0, 3);
    TaskNode after1Node(&afterCopy1, 0, 0, 4);

    TaskNode beforeNode2(&beforeCopy2, 0, 0, 5);
    TaskNode loopStart2Node(&loopStart2, 0, 0, 6);
    TaskNode inner2Node(&innerCopy2, 0, 0, 7);
    TaskNode loopEnd2Node(&loopEnd2, 0, 0, 8);
    TaskNode after2Node(&afterCopy2, 0, 0, 9);

    BuildParentChild(&ccuNode, &beforeNode1);
    BuildParentChild(&beforeNode1, &loopStart1Node);
    BuildParentChild(&loopStart1Node, &inner1Node);
    BuildParentChild(&inner1Node, &loopEnd1Node);
    BuildParentChild(&loopEnd1Node, &after1Node);

    BuildParentChild(&after1Node, &beforeNode2);
    BuildParentChild(&beforeNode2, &loopStart2Node);
    BuildParentChild(&loopStart2Node, &inner2Node);
    BuildParentChild(&inner2Node, &loopEnd2Node);
    BuildParentChild(&loopEnd2Node, &after2Node);

    ccuTask.loopGroupInfo_.push_back({LoopInfo(&loopStart1Node, &loopEnd1Node)});
    ccuTask.loopGroupInfo_.push_back({LoopInfo(&loopStart2Node, &loopEnd2Node)});
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    EXPECT_TRUE(loopStart1Node.loopParallel);
    EXPECT_TRUE(loopStart2Node.loopParallel);
    EXPECT_NE(loopStart1Node.queIdx, beforeNode1.queIdx);
    EXPECT_NE(loopStart2Node.queIdx, beforeNode2.queIdx);
}

TEST_F(GraphRevampParallelTest, Revamp_AsyncWithMultipleParents)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy parent1Copy(srcSlice, dstSlice);
    TaskStubLocalCopy parent2Copy(srcSlice, dstSlice);
    TaskStubLocalCopy asyncCopy(srcSlice, dstSlice);
    TaskStubLocalPostTo postTo(0);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode parent1(&parent1Copy, 0, 0, 0);
    TaskNode parent2(&parent2Copy, 0, 0, 1);
    TaskNode asyncNode(&asyncCopy, 0, 0, 2);
    TaskNode postNode(&postTo, 0, 0, 3);
    TaskNode waitNode(&waitFrom, 0, 0, 4);
    TaskNode afterNode(&afterCopy, 0, 0, 5);

    BuildParentChild(&ccuNode, &parent1);
    BuildParentChild(&ccuNode, &parent2);
    BuildParentChild(&parent1, &asyncNode);
    BuildParentChild(&parent2, &asyncNode);
    BuildParentChild(&asyncNode, &postNode);
    BuildParentChild(&postNode, &waitNode);
    BuildParentChild(&waitNode, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.localPostWaitPairs_[&postNode] = &waitNode;
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(parent1.asynParallel);
    EXPECT_TRUE(parent2.asynParallel);
}

TEST_F(GraphRevampParallelTest, Revamp_AsyncNodeWithChildrenOnDifferentQueue)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy asyncCopy(srcSlice, dstSlice);
    TaskStubLocalPostTo postTo(0);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy diffQueueCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode(&asyncCopy, 0, 0, 1);
    TaskNode postNode(&postTo, 0, 0, 2);
    TaskNode waitNode(&waitFrom, 0, 0, 3);
    TaskNode diffQueueChild(&diffQueueCopy, 0, 1, 0);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode);
    BuildParentChild(&asyncNode, &postNode);
    BuildParentChild(&postNode, &waitNode);
    BuildParentChild(&asyncNode, &diffQueueChild);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.localPostWaitPairs_[&postNode] = &waitNode;
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(beforeNode.asynParallel);
}

TEST_F(GraphRevampParallelTest, Destructor_CleansUpResources)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);
    TaskStubLoopStart loopStartStub(0, 0);
    TaskStubLoopEnd loopEndStub(0, 0);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode loopStartNode(&loopStartStub, 0, 0, 1);
    TaskNode loopEndNode(&loopEndStub, 0, 0, 2);
    TaskNode afterNode(&afterCopy, 0, 0, 3);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &loopStartNode);
    BuildParentChild(&loopStartNode, &loopEndNode);
    BuildParentChild(&loopEndNode, &afterNode);

    ccuTask.loopGroupInfo_.push_back({LoopInfo(&loopStartNode, &loopEndNode)});
    ccuTask.queueNum_ = 1;

    revamp.Revamp(&dummyStart);
    EXPECT_GT(revamp.toDeleteTaskResource_.size(), 0u);
    EXPECT_GT(revamp.toDeleteTaskNodeResource_.size(), 0u);
}

TEST_F(GraphRevampParallelTest, Revamp_ReadReduceAsyncNode)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubReadReduce readReduceStub(
        1, LinkInfo(LinkProtoStub::RDMA), srcSlice, dstSlice, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskStubLocalPostTo postTo(0);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode(&readReduceStub, 0, 0, 1);
    TaskNode postNode(&postTo, 0, 0, 2);
    TaskNode waitNode(&waitFrom, 0, 0, 3);
    TaskNode afterNode(&afterCopy, 0, 0, 4);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode);
    BuildParentChild(&asyncNode, &postNode);
    BuildParentChild(&postNode, &waitNode);
    BuildParentChild(&waitNode, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.localPostWaitPairs_[&postNode] = &waitNode;
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(beforeNode.asynParallel);
}

TEST_F(GraphRevampParallelTest, Revamp_WriteReduceAsyncNode)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubWriteReduce writeReduceStub(
        1, LinkInfo(LinkProtoStub::RDMA), srcSlice, dstSlice, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskStubLocalPostTo postTo(0);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode(&writeReduceStub, 0, 0, 1);
    TaskNode postNode(&postTo, 0, 0, 2);
    TaskNode waitNode(&waitFrom, 0, 0, 3);
    TaskNode afterNode(&afterCopy, 0, 0, 4);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode);
    BuildParentChild(&asyncNode, &postNode);
    BuildParentChild(&postNode, &waitNode);
    BuildParentChild(&waitNode, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.localPostWaitPairs_[&postNode] = &waitNode;
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(beforeNode.asynParallel);
}

TEST_F(GraphRevampParallelTest, Revamp_AsyncNodeWithNoPostChildrenOnSameQueue)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy asyncCopy(srcSlice, dstSlice);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode(&asyncCopy, 0, 0, 1);
    TaskNode afterNode(&afterCopy, 0, 1, 0);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode);
    BuildParentChild(&asyncNode, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.queueNum_ = 1;

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(beforeNode.asynParallel);
}

TEST_F(GraphRevampParallelTest, Revamp_VerifyLoopVirtualStreamStructure)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    TaskStubLoopStart loopStartStub(0, 0);
    TaskStubLoopEnd loopEndStub(0, 0);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy innerCopy(srcSlice, dstSlice);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode loopStartNode(&loopStartStub, 0, 0, 1);
    TaskNode innerNode(&innerCopy, 0, 0, 2);
    TaskNode loopEndNode(&loopEndStub, 0, 0, 3);
    TaskNode afterNode(&afterCopy, 0, 0, 4);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &loopStartNode);
    BuildParentChild(&loopStartNode, &innerNode);
    BuildParentChild(&innerNode, &loopEndNode);
    BuildParentChild(&loopEndNode, &afterNode);

    ccuTask.loopGroupInfo_.push_back({LoopInfo(&loopStartNode, &loopEndNode)});
    ccuTask.queueNum_ = 1;

    u32 origQueIdx = loopStartNode.queIdx;
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    EXPECT_NE(loopStartNode.queIdx, origQueIdx);
    EXPECT_NE(beforeNode.children.size(), 0u);
    bool foundLocalPost = false;
    for (auto child : beforeNode.children) {
        if (child->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
            foundLocalPost = true;
            EXPECT_TRUE(child->loopParallel);
        }
    }
    EXPECT_TRUE(foundLocalPost);
}

TEST_F(GraphRevampParallelTest, Revamp_VerifyAsyncVirtualStreamStructure)
{
    GraphRevampParallel revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy beforeCopy(srcSlice, dstSlice);
    TaskStubLocalCopy asyncCopy(srcSlice, dstSlice);
    TaskStubLocalPostTo postTo(0);
    TaskStubLocalWaitFrom waitFrom(0);
    TaskStubLocalCopy afterCopy(srcSlice, dstSlice);

    TaskNode beforeNode(&beforeCopy, 0, 0, 0);
    TaskNode asyncNode(&asyncCopy, 0, 0, 1);
    TaskNode postNode(&postTo, 0, 0, 2);
    TaskNode waitNode(&waitFrom, 0, 0, 3);
    TaskNode afterNode(&afterCopy, 0, 0, 4);

    BuildParentChild(&ccuNode, &beforeNode);
    BuildParentChild(&beforeNode, &asyncNode);
    BuildParentChild(&asyncNode, &postNode);
    BuildParentChild(&postNode, &waitNode);
    BuildParentChild(&waitNode, &afterNode);

    ccuTask.parallelNodes_.push_back(&asyncNode);
    ccuTask.localPostWaitPairs_[&postNode] = &waitNode;
    ccuTask.queueNum_ = 1;

    u32 origAsyncQueIdx = asyncNode.queIdx;
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    EXPECT_NE(asyncNode.queIdx, origAsyncQueIdx);
    EXPECT_NE(beforeNode.children.size(), 0u);
    bool foundLocalPost = false;
    for (auto child : beforeNode.children) {
        if (child->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
            foundLocalPost = true;
            EXPECT_TRUE(child->asynParallel);
        }
    }
    EXPECT_TRUE(foundLocalPost);
}
