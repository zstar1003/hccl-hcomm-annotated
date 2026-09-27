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

#include "sim_task.h"
#include "task_ccu.h"
#include "task_def.h"
#include "task_graph_revamp_base.h"

using namespace HcclSim;

namespace {
class TestGraphRevamp : public GraphRevampBase {
public:
    HcclResult Revamp(TaskNodePtr dummyStart) override { return RevampGraph(dummyStart); }
};

class TestGraphRevampNoOp : public GraphRevampBase {
public:
    HcclResult Revamp(TaskNodePtr dummyStart) override { return HcclResult::HCCL_SUCCESS; }
};

class TestGraphRevampCustom4Rank : public GraphRevampBase {
public:
    bool revampGraph4RankCalled = false;
    HcclResult Revamp(TaskNodePtr dummyStart) override { return RevampGraph(dummyStart); }
    HcclResult RevampGraph4Rank(TaskNodePtr ccuHead, RankId rankId) override
    {
        revampGraph4RankCalled = true;
        return HcclResult::HCCL_SUCCESS;
    }
};

void BuildParentChild(TaskNodePtr parent, TaskNodePtr child)
{
    parent->children.push_back(child);
    child->parents.push_back(parent);
}
} // anonymous namespace

class GraphRevampBaseTest : public testing::Test {
protected:
    void SetUp() override { revamp_ = std::make_unique<TestGraphRevamp>(); }

    void TearDown() override { revamp_.reset(); }

    std::unique_ptr<TestGraphRevamp> revamp_;
};

// ========== RemoveNodeRelation ==========

TEST_F(GraphRevampBaseTest, RemoveNodeRelation_SingleChild)
{
    TaskNode parent(nullptr, 0, 0, 0);
    TaskNode child(nullptr, 0, 0, 0);
    BuildParentChild(&parent, &child);

    EXPECT_EQ(parent.children.size(), 1);
    EXPECT_EQ(child.parents.size(), 1);

    revamp_->RemoveNodeRelation(&parent, &child);

    EXPECT_EQ(parent.children.size(), 0);
    EXPECT_EQ(child.parents.size(), 0);
}

TEST_F(GraphRevampBaseTest, RemoveNodeRelation_AmongMultiple)
{
    TaskNode parent(nullptr, 0, 0, 0);
    TaskNode child1(nullptr, 0, 0, 0);
    TaskNode child2(nullptr, 0, 0, 0);
    TaskNode child3(nullptr, 0, 0, 0);
    BuildParentChild(&parent, &child1);
    BuildParentChild(&parent, &child2);
    BuildParentChild(&parent, &child3);

    EXPECT_EQ(parent.children.size(), 3);
    EXPECT_EQ(child2.parents.size(), 1);

    revamp_->RemoveNodeRelation(&parent, &child2);

    EXPECT_EQ(parent.children.size(), 2);
    EXPECT_EQ(parent.children[0], &child1);
    EXPECT_EQ(parent.children[1], &child3);
    EXPECT_EQ(child2.parents.size(), 0);
}

// ========== AddNodeRelation ==========

TEST_F(GraphRevampBaseTest, AddNodeRelation_Basic)
{
    TaskNode parent(nullptr, 0, 0, 0);
    TaskNode child(nullptr, 0, 0, 0);

    revamp_->AddNodeRelation(&parent, &child);

    EXPECT_EQ(parent.children.size(), 1);
    EXPECT_EQ(parent.children[0], &child);
    EXPECT_EQ(child.parents.size(), 1);
    EXPECT_EQ(child.parents[0], &parent);
}

// ========== SearchGraphByRank ==========

TEST_F(GraphRevampBaseTest, SearchGraphByRank_MatchingRank)
{
    TaskNode root(nullptr, 0, 0, 0);
    TaskNode childA(nullptr, 7, 0, 0);
    TaskNode childB(nullptr, 3, 0, 0);
    BuildParentChild(&root, &childA);
    BuildParentChild(&root, &childB);

    std::queue<TaskNodePtr> graphNodeQue;
    std::set<TaskNodePtr> isVisited;

    revamp_->SearchGraphByRank(&root, graphNodeQue, isVisited, 7);

    EXPECT_EQ(graphNodeQue.size(), 1);
    EXPECT_EQ(graphNodeQue.front(), &childA);
    EXPECT_EQ(isVisited.size(), 1);
    EXPECT_TRUE(isVisited.count(&childA));
}

TEST_F(GraphRevampBaseTest, SearchGraphByRank_NoMatch)
{
    TaskNode root(nullptr, 0, 0, 0);
    TaskNode childA(nullptr, 7, 0, 0);
    BuildParentChild(&root, &childA);

    std::queue<TaskNodePtr> graphNodeQue;
    std::set<TaskNodePtr> isVisited;

    revamp_->SearchGraphByRank(&root, graphNodeQue, isVisited, 3);

    EXPECT_TRUE(graphNodeQue.empty());
    EXPECT_TRUE(isVisited.empty());
}

TEST_F(GraphRevampBaseTest, SearchGraphByRank_AlreadyVisited)
{
    TaskNode root(nullptr, 0, 0, 0);
    TaskNode childA(nullptr, 7, 0, 0);
    BuildParentChild(&root, &childA);

    std::queue<TaskNodePtr> graphNodeQue;
    std::set<TaskNodePtr> isVisited;
    isVisited.insert(&childA);

    revamp_->SearchGraphByRank(&root, graphNodeQue, isVisited, 7);

    EXPECT_TRUE(graphNodeQue.empty());
}

// ========== SearchGraphByQueueId ==========

TEST_F(GraphRevampBaseTest, SearchGraphByQueueId_Matching)
{
    TaskNode root(nullptr, 0, 0, 0);
    TaskNode childA(nullptr, 0, 5, 0);
    TaskNode childB(nullptr, 0, 9, 0);
    BuildParentChild(&root, &childA);
    BuildParentChild(&root, &childB);

    std::queue<TaskNodePtr> graphNodeQue;
    std::set<TaskNodePtr> isVisited;

    revamp_->SearchGraphByQueueId(&root, graphNodeQue, isVisited, 5);

    EXPECT_EQ(graphNodeQue.size(), 1);
    EXPECT_EQ(graphNodeQue.front(), &childA);
    EXPECT_TRUE(isVisited.count(&childA));
}

TEST_F(GraphRevampBaseTest, SearchGraphByQueueId_NoMatch)
{
    TaskNode root(nullptr, 0, 0, 0);
    TaskNode childA(nullptr, 0, 5, 0);
    BuildParentChild(&root, &childA);

    std::queue<TaskNodePtr> graphNodeQue;
    std::set<TaskNodePtr> isVisited;

    revamp_->SearchGraphByQueueId(&root, graphNodeQue, isVisited, 10);

    EXPECT_TRUE(graphNodeQue.empty());
    EXPECT_TRUE(isVisited.empty());
}

TEST_F(GraphRevampBaseTest, SearchGraphByQueueId_AlreadyVisited)
{
    TaskNode root(nullptr, 0, 0, 0);
    TaskNode childA(nullptr, 0, 5, 0);
    BuildParentChild(&root, &childA);

    std::queue<TaskNodePtr> graphNodeQue;
    std::set<TaskNodePtr> isVisited;
    isVisited.insert(&childA);

    revamp_->SearchGraphByQueueId(&root, graphNodeQue, isVisited, 5);

    EXPECT_TRUE(graphNodeQue.empty());
}

// ========== Destructor ==========

TEST_F(GraphRevampBaseTest, Destructor_CleansUpResources)
{
    auto raw = new TaskNode(nullptr, 0, 0, 0);
    revamp_->toDeleteTaskNodeResource_.push_back(raw);

    revamp_.reset();

    // If destructor didn't double-free, we're good
    SUCCEED();
}

// ========== GetPeerRankByTaskNode ==========

TEST_F(GraphRevampBaseTest, GetPeerRankByTaskNode_Read)
{
    TaskStubRead read(0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice());
    TaskNode node(&read, 0, 0, 0);

    RankId peerRank = 0;
    HcclResult ret = revamp_->GetPeerRankByTaskNode(&node, peerRank);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(peerRank, 0);
}

TEST_F(GraphRevampBaseTest, GetPeerRankByTaskNode_Write)
{
    TaskStubWrite write(0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice());
    TaskNode node(&write, 0, 0, 0);

    RankId peerRank = 0;
    HcclResult ret = revamp_->GetPeerRankByTaskNode(&node, peerRank);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(peerRank, 0);
}

TEST_F(GraphRevampBaseTest, GetPeerRankByTaskNode_ReadReduce)
{
    TaskStubReadReduce readReduce(
        0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice(), HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode node(&readReduce, 0, 0, 0);

    RankId peerRank = 0;
    HcclResult ret = revamp_->GetPeerRankByTaskNode(&node, peerRank);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(peerRank, 0);
}

TEST_F(GraphRevampBaseTest, GetPeerRankByTaskNode_WriteReduce)
{
    TaskStubWriteReduce writeReduce(
        0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice(), HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode node(&writeReduce, 0, 0, 0);

    RankId peerRank = 0;
    HcclResult ret = revamp_->GetPeerRankByTaskNode(&node, peerRank);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(peerRank, 0);
}

// ========== GetLinkProtoStubByTaskNode ==========

TEST_F(GraphRevampBaseTest, GetLinkProtoStubByTaskNode_Read)
{
    TaskStubRead read(0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice());
    TaskNode node(&read, 0, 0, 0);

    LinkProtoStub link = LinkProtoStub::INVALID_A;
    HcclResult ret = revamp_->GetLinkProtoStubByTaskNode(&node, link);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBaseTest, GetLinkProtoStubByTaskNode_Write)
{
    TaskStubWrite write(0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice());
    TaskNode node(&write, 0, 0, 0);

    LinkProtoStub link = LinkProtoStub::INVALID_A;
    HcclResult ret = revamp_->GetLinkProtoStubByTaskNode(&node, link);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBaseTest, GetLinkProtoStubByTaskNode_ReadReduce)
{
    TaskStubReadReduce readReduce(
        0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice(), HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode node(&readReduce, 0, 0, 0);

    LinkProtoStub link = LinkProtoStub::INVALID_A;
    HcclResult ret = revamp_->GetLinkProtoStubByTaskNode(&node, link);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBaseTest, GetLinkProtoStubByTaskNode_WriteReduce)
{
    TaskStubWriteReduce writeReduce(
        0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice(), HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode node(&writeReduce, 0, 0, 0);

    LinkProtoStub link = LinkProtoStub::INVALID_A;
    HcclResult ret = revamp_->GetLinkProtoStubByTaskNode(&node, link);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// ========== GenTaskStubBeingReadOrWrittern ==========

TEST_F(GraphRevampBaseTest, GenTaskStubBeingReadOrWrittern_Read)
{
    TaskStubRead read(0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice());
    TaskNode node(&read, 0, 0, 0);

    TaskStub* stub = revamp_->GenTaskStubBeingReadOrWrittern(&node);
    ASSERT_NE(stub, nullptr);
    EXPECT_EQ(stub->GetType(), TaskTypeStub::BEING_READ);
    EXPECT_NE(revamp_->toDeleteTaskResource_.empty(), true);
}

TEST_F(GraphRevampBaseTest, GenTaskStubBeingReadOrWrittern_Write)
{
    TaskStubWrite write(0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice());
    TaskNode node(&write, 0, 0, 0);

    TaskStub* stub = revamp_->GenTaskStubBeingReadOrWrittern(&node);
    ASSERT_NE(stub, nullptr);
    EXPECT_EQ(stub->GetType(), TaskTypeStub::BEING_WRITTEN);
}

TEST_F(GraphRevampBaseTest, GenTaskStubBeingReadOrWrittern_ReadReduce)
{
    TaskStubReadReduce readReduce(
        0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice(), HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode node(&readReduce, 0, 0, 0);

    TaskStub* stub = revamp_->GenTaskStubBeingReadOrWrittern(&node);
    ASSERT_NE(stub, nullptr);
    EXPECT_EQ(stub->GetType(), TaskTypeStub::BEING_READ_REDUCE);
}

TEST_F(GraphRevampBaseTest, GenTaskStubBeingReadOrWrittern_WriteReduce)
{
    TaskStubWriteReduce writeReduce(
        0, LinkInfo(LinkProtoStub::SDMA), DataSlice(), DataSlice(), HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode node(&writeReduce, 0, 0, 0);

    TaskStub* stub = revamp_->GenTaskStubBeingReadOrWrittern(&node);
    ASSERT_NE(stub, nullptr);
    EXPECT_EQ(stub->GetType(), TaskTypeStub::BEING_WRITTEN_REDUCE);
}

// ========== rank2QueSize_ static member ==========

TEST_F(GraphRevampBaseTest, StaticRank2QueSize)
{
    TestGraphRevamp::rank2QueSize_[0] = 5;
    TestGraphRevamp::rank2QueSize_[1] = 3;

    EXPECT_EQ(TestGraphRevamp::rank2QueSize_[0], 5);
    EXPECT_EQ(TestGraphRevamp::rank2QueSize_[1], 3);
    EXPECT_EQ(TestGraphRevamp::rank2QueSize_.size(), 2);

    TestGraphRevamp::rank2QueSize_.clear();
    EXPECT_TRUE(TestGraphRevamp::rank2QueSize_.empty());
}

// ========== RevampGraph ==========

TEST_F(GraphRevampBaseTest, RevampGraph_EmptyGraph)
{
    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = revamp_->RevampGraph(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBaseTest, RevampGraph_SingleRankNoCcu)
{
    TaskStubLocalCopy copyTask{DataSlice{}, DataSlice{}};
    TaskNode child(&copyTask, 0, 0, 0);
    TaskNode dummyStart(nullptr, -1, 0, 0);
    BuildParentChild(&dummyStart, &child);

    HcclResult ret = revamp_->RevampGraph(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBaseTest, RevampGraph_MultiRankNoCcu)
{
    TaskStubLocalCopy copyTask1{DataSlice{}, DataSlice{}};
    TaskNode child0(&copyTask1, 0, 0, 0);
    TaskStubLocalCopy copyTask2{DataSlice{}, DataSlice{}};
    TaskNode child1(&copyTask2, 1, 0, 0);
    TaskNode dummyStart(nullptr, -1, 0, 0);
    BuildParentChild(&dummyStart, &child0);
    BuildParentChild(&dummyStart, &child1);

    HcclResult ret = revamp_->RevampGraph(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBaseTest, RevampGraph_SingleRankWithCcuGraph)
{
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    TaskNode dummyStart(nullptr, -1, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    HcclResult ret = revamp_->RevampGraph(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBaseTest, RevampGraph_MultiRankWithCcuGraph)
{
    TaskStubCcuGraph ccuTask0(static_cast<RankId>(0));
    TaskNode ccuNode0(&ccuTask0, 0, 0, 0);
    TaskStubCcuGraph ccuTask1(static_cast<RankId>(1));
    TaskNode ccuNode1(&ccuTask1, 1, 0, 0);
    TaskNode dummyStart(nullptr, -1, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode0);
    BuildParentChild(&dummyStart, &ccuNode1);

    HcclResult ret = revamp_->RevampGraph(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// ========== RevampGraph4Rank ==========

TEST_F(GraphRevampBaseTest, RevampGraph4Rank_DefaultReturnsSuccess)
{
    TaskStubLocalCopy copyTask{DataSlice{}, DataSlice{}};
    TaskNode ccuHead(&copyTask, 0, 0, 0);
    HcclResult ret = revamp_->RevampGraph4Rank(&ccuHead, 0);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBaseTest, RevampGraph4Rank_CustomOverrideCalled)
{
    TestGraphRevampCustom4Rank customRevamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);

    HcclResult ret = customRevamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(customRevamp.revampGraph4RankCalled);
}

// ========== Destructor nullptr branches ==========

TEST_F(GraphRevampBaseTest, Destructor_NullptrTaskResource)
{
    revamp_->toDeleteTaskResource_.push_back(nullptr);
    revamp_->toDeleteTaskResource_.push_back(nullptr);
    revamp_.reset();
    SUCCEED();
}

TEST_F(GraphRevampBaseTest, Destructor_NullptrTaskNodeResource)
{
    revamp_->toDeleteTaskNodeResource_.push_back(nullptr);
    revamp_->toDeleteTaskNodeResource_.push_back(nullptr);
    revamp_.reset();
    SUCCEED();
}

TEST_F(GraphRevampBaseTest, Destructor_MixedNullptrAndValidResources)
{
    auto validTask = new TaskStubLocalCopy(DataSlice(), DataSlice());
    auto validNode = new TaskNode(nullptr, 0, 0, 0);
    revamp_->toDeleteTaskResource_.push_back(nullptr);
    revamp_->toDeleteTaskResource_.push_back(validTask);
    revamp_->toDeleteTaskResource_.push_back(nullptr);
    revamp_->toDeleteTaskNodeResource_.push_back(nullptr);
    revamp_->toDeleteTaskNodeResource_.push_back(validNode);
    revamp_->toDeleteTaskNodeResource_.push_back(nullptr);
    revamp_.reset();
    SUCCEED();
}

// ========== Revamp (calls RevampGraph) ==========

TEST_F(GraphRevampBaseTest, Revamp_CallsRevampGraph)
{
    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = revamp_->Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBaseTest, Revamp_WithCcuGraphNodes)
{
    TaskStubCcuGraph ccuTask(static_cast<RankId>(0));
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    TaskStubLocalCopy copyTask{DataSlice{}, DataSlice{}};
    TaskNode childAfter(&copyTask, 0, 0, 1);
    TaskNode dummyStart(nullptr, -1, 0, 0);
    BuildParentChild(&dummyStart, &ccuNode);
    BuildParentChild(&ccuNode, &childAfter);

    HcclResult ret = revamp_->Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBaseTest, DeletingDestructor_ViaBasePointer)
{
    GraphRevampBase* base = new TestGraphRevamp();
    base->toDeleteTaskResource_.push_back(nullptr);
    base->toDeleteTaskNodeResource_.push_back(nullptr);
    delete base;
    SUCCEED();
}
