/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <gtest/gtest.h>
#include <queue>
#include <set>

#define private public
#define protected public
#include "data_slice.h"
#include "sim_task.h"
#include "task_def.h"
#include "task_graph_revamp.h"

#undef private
#undef protected
using namespace HcclSim;

namespace {
void BuildParentChild(TaskNodePtr parent, TaskNodePtr child)
{
    parent->children.push_back(child);
    child->parents.push_back(parent);
}
} // namespace

class GraphRevampBilateralSemanticsTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(GraphRevampBilateralSemanticsTest, Destructor_EmptyResources)
{
    GraphRevampBilateralSemantics* revamp = new GraphRevampBilateralSemantics();
    EXPECT_NO_THROW(delete revamp);
}

TEST_F(GraphRevampBilateralSemanticsTest, Destructor_WithTaskResources)
{
    GraphRevampBilateralSemantics* revamp = new GraphRevampBilateralSemantics();
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStub* stub = new TaskStubLocalCopy(srcSlice, dstSlice);
    TaskNode* node = new TaskNode(stub, 0, 0, 0);
    revamp->toDeleteTaskResource_.push_back(stub);
    revamp->toDeleteTaskNodeResource_.push_back(node);
    EXPECT_NO_THROW(delete revamp);
}

TEST_F(GraphRevampBilateralSemanticsTest, Destructor_WithNullptrResources)
{
    GraphRevampBilateralSemantics* revamp = new GraphRevampBilateralSemantics();
    revamp->toDeleteTaskResource_.push_back(nullptr);
    revamp->toDeleteTaskNodeResource_.push_back(nullptr);
    EXPECT_NO_THROW(delete revamp);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_EmptyGraph)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_SingleRankWithLocalCopy)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
    TaskNode copyNode(localCopy.get(), 0, 0, 0);
    BuildParentChild(&dummyStart, &copyNode);
    copyNode.procFlag = false;
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_SingleRankWithReadSDMA)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto readTask = std::make_shared<TaskStubRead>(1, link, localSlice, remoteSlice);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    TaskNode copyNode(localCopy.get(), 0, 0, 0);
    TaskNode readNode(readTask.get(), 0, 0, 1);
    TaskNode copyNode2(localCopy2.get(), 0, 0, 2);
    BuildParentChild(&dummyStart, &copyNode);
    BuildParentChild(&copyNode, &readNode);
    BuildParentChild(&readNode, &copyNode2);
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_SingleRankWithReadReduceSDMA)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto readReduceTask = std::make_shared<TaskStubReadReduce>(
        1, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    TaskNode copyNode(localCopy.get(), 0, 0, 0);
    TaskNode readNode(readReduceTask.get(), 0, 0, 1);
    TaskNode copyNode2(localCopy2.get(), 0, 0, 2);
    BuildParentChild(&dummyStart, &copyNode);
    BuildParentChild(&copyNode, &readNode);
    BuildParentChild(&readNode, &copyNode2);
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_ReadWithNonSDMALink)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::RDMA);
    auto readTask = std::make_shared<TaskStubRead>(1, link, localSlice, remoteSlice);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    TaskNode copyNode(localCopy.get(), 0, 0, 0);
    TaskNode readNode(readTask.get(), 0, 0, 1);
    BuildParentChild(&dummyStart, &copyNode);
    BuildParentChild(&copyNode, &readNode);
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_WriteWithSDMA)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto writeTask = std::make_shared<TaskStubWrite>(1, link, localSlice, remoteSlice);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    TaskNode copyNode(localCopy.get(), 0, 0, 0);
    TaskNode writeNode(writeTask.get(), 0, 0, 1);
    TaskNode copyNode2(localCopy2.get(), 0, 0, 2);
    BuildParentChild(&dummyStart, &copyNode);
    BuildParentChild(&copyNode, &writeNode);
    BuildParentChild(&writeNode, &copyNode2);
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_WriteWithRDMA)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::RDMA);
    auto writeTask = std::make_shared<TaskStubWrite>(1, link, localSlice, remoteSlice);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    TaskNode copyNode(localCopy.get(), 0, 0, 0);
    TaskNode writeNode(writeTask.get(), 0, 0, 1);
    TaskNode copyNode2(localCopy2.get(), 0, 0, 2);
    BuildParentChild(&dummyStart, &copyNode);
    BuildParentChild(&copyNode, &writeNode);
    BuildParentChild(&writeNode, &copyNode2);
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_WriteReduceWithSDMA)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto writeReduceTask = std::make_shared<TaskStubWriteReduce>(
        1, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    TaskNode copyNode(localCopy.get(), 0, 0, 0);
    TaskNode writeNode(writeReduceTask.get(), 0, 0, 1);
    TaskNode copyNode2(localCopy2.get(), 0, 0, 2);
    BuildParentChild(&dummyStart, &copyNode);
    BuildParentChild(&copyNode, &writeNode);
    BuildParentChild(&writeNode, &copyNode2);
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_WriteReduceWithRDMA)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::RDMA);
    auto writeReduceTask = std::make_shared<TaskStubWriteReduce>(
        1, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    TaskNode copyNode(localCopy.get(), 0, 0, 0);
    TaskNode writeNode(writeReduceTask.get(), 0, 0, 1);
    TaskNode copyNode2(localCopy2.get(), 0, 0, 2);
    BuildParentChild(&dummyStart, &copyNode);
    BuildParentChild(&copyNode, &writeNode);
    BuildParentChild(&writeNode, &copyNode2);
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_WriteWithNonSupportedLink)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::CCU);
    auto writeTask = std::make_shared<TaskStubWrite>(1, link, localSlice, remoteSlice);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    TaskNode copyNode(localCopy.get(), 0, 0, 0);
    TaskNode writeNode(writeTask.get(), 0, 0, 1);
    BuildParentChild(&dummyStart, &copyNode);
    BuildParentChild(&copyNode, &writeNode);
    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_MultiRankWithReadWrite)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);

    auto localCopy1 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto writeTask = std::make_shared<TaskStubWrite>(1, link, localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    TaskNode copyNode1_r0(localCopy1.get(), 0, 0, 0);
    TaskNode writeNode(writeTask.get(), 0, 0, 1);
    TaskNode copyNode2_r0(localCopy2.get(), 0, 0, 2);
    BuildParentChild(&dummyStart, &copyNode1_r0);
    BuildParentChild(&copyNode1_r0, &writeNode);
    BuildParentChild(&writeNode, &copyNode2_r0);

    auto localCopy3 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto readTask = std::make_shared<TaskStubRead>(0, link, localSlice, remoteSlice);
    auto localCopy4 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    TaskNode copyNode1_r1(localCopy3.get(), 1, 0, 0);
    TaskNode readNode(readTask.get(), 1, 0, 1);
    TaskNode copyNode2_r1(localCopy4.get(), 1, 0, 2);
    BuildParentChild(&dummyStart, &copyNode1_r1);
    BuildParentChild(&copyNode1_r1, &readNode);
    BuildParentChild(&readNode, &copyNode2_r1);

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_ReadWithWaitBackward)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo sdmaLink(LinkProtoStub::SDMA);
    LinkInfo rdmaLink(LinkProtoStub::RDMA);

    auto waitTask = std::make_shared<TaskStubWait>(1, rdmaLink, 100);
    auto readTask = std::make_shared<TaskStubRead>(1, sdmaLink, localSlice, remoteSlice);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);

    TaskNode waitNode(waitTask.get(), 0, 0, 0);
    TaskNode copyNode(localCopy.get(), 0, 0, 1);
    TaskNode readNode(readTask.get(), 0, 0, 2);
    TaskNode copyNode2(localCopy2.get(), 0, 0, 3);

    BuildParentChild(&dummyStart, &waitNode);
    BuildParentChild(&waitNode, &copyNode);
    BuildParentChild(&copyNode, &readNode);
    BuildParentChild(&readNode, &copyNode2);

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_ReadWithPostForward)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo sdmaLink(LinkProtoStub::SDMA);
    LinkInfo rdmaLink(LinkProtoStub::RDMA);

    auto localCopy1 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto readTask = std::make_shared<TaskStubRead>(1, sdmaLink, localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto postTask = std::make_shared<TaskStubPost>(1, rdmaLink, 100);

    TaskNode copyNode1(localCopy1.get(), 0, 0, 0);
    TaskNode readNode(readTask.get(), 0, 0, 1);
    TaskNode copyNode2(localCopy2.get(), 0, 0, 2);
    TaskNode postNode(postTask.get(), 0, 0, 3);

    BuildParentChild(&dummyStart, &copyNode1);
    BuildParentChild(&copyNode1, &readNode);
    BuildParentChild(&readNode, &copyNode2);
    BuildParentChild(&copyNode2, &postNode);

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_WriteRDMAWithWaitBackward)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo rdmaLink(LinkProtoStub::RDMA);

    auto waitTask = std::make_shared<TaskStubWait>(1, rdmaLink, 100);
    auto writeTask = std::make_shared<TaskStubWrite>(1, rdmaLink, localSlice, remoteSlice);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);

    TaskNode waitNode(waitTask.get(), 0, 0, 0);
    TaskNode copyNode(localCopy.get(), 0, 0, 1);
    TaskNode writeNode(writeTask.get(), 0, 0, 2);
    TaskNode copyNode2(localCopy2.get(), 0, 0, 3);

    BuildParentChild(&dummyStart, &waitNode);
    BuildParentChild(&waitNode, &copyNode);
    BuildParentChild(&copyNode, &writeNode);
    BuildParentChild(&writeNode, &copyNode2);

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, Revamp_WriteRDMAWithReadForward)
{
    GraphRevampBilateralSemantics revamp;
    TaskNode dummyStart(nullptr, -1, 0, 0);
    DataSlice localSlice(BufferType::CCL, 0, 1024);
    DataSlice remoteSlice(BufferType::CCL, 0, 1024);
    LinkInfo rdmaLink(LinkProtoStub::RDMA);
    LinkInfo sdmaLink(LinkProtoStub::SDMA);

    auto writeTask = std::make_shared<TaskStubWrite>(1, rdmaLink, localSlice, remoteSlice);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);
    auto readTask = std::make_shared<TaskStubRead>(1, sdmaLink, localSlice, remoteSlice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(localSlice, remoteSlice);

    TaskNode copyNode(localCopy.get(), 0, 0, 0);
    TaskNode writeNode(writeTask.get(), 0, 0, 1);
    TaskNode readNode(readTask.get(), 0, 0, 2);
    TaskNode copyNode2(localCopy2.get(), 0, 0, 3);

    BuildParentChild(&dummyStart, &copyNode);
    BuildParentChild(&copyNode, &writeNode);
    BuildParentChild(&writeNode, &readNode);
    BuildParentChild(&readNode, &copyNode2);

    HcclResult ret = revamp.Revamp(&dummyStart);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, IsReadWriteWithSameRank_ReadMatch)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto readTask = std::make_shared<TaskStubRead>(1, link, slice, slice);
    TaskNode readNode(readTask.get(), 0, 0, 0);
    bool result = revamp.IsReadWriteWithSameRank(1, &readNode);
    EXPECT_TRUE(result);
}

TEST_F(GraphRevampBilateralSemanticsTest, IsReadWriteWithSameRank_ReadNoMatch)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto readTask = std::make_shared<TaskStubRead>(2, link, slice, slice);
    TaskNode readNode(readTask.get(), 0, 0, 0);
    bool result = revamp.IsReadWriteWithSameRank(1, &readNode);
    EXPECT_FALSE(result);
}

TEST_F(GraphRevampBilateralSemanticsTest, IsReadWriteWithSameRank_WriteMatch)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto writeTask = std::make_shared<TaskStubWrite>(1, link, slice, slice);
    TaskNode writeNode(writeTask.get(), 0, 0, 0);
    bool result = revamp.IsReadWriteWithSameRank(1, &writeNode);
    EXPECT_TRUE(result);
}

TEST_F(GraphRevampBilateralSemanticsTest, IsReadWriteWithSameRank_WriteNoMatch)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto writeTask = std::make_shared<TaskStubWrite>(2, link, slice, slice);
    TaskNode writeNode(writeTask.get(), 0, 0, 0);
    bool result = revamp.IsReadWriteWithSameRank(1, &writeNode);
    EXPECT_FALSE(result);
}

TEST_F(GraphRevampBilateralSemanticsTest, IsReadWriteWithSameRank_ReadReduceMatch)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto readReduceTask = std::make_shared<TaskStubReadReduce>(
        1, link, slice, slice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    TaskNode node(readReduceTask.get(), 0, 0, 0);
    bool result = revamp.IsReadWriteWithSameRank(1, &node);
    EXPECT_TRUE(result);
}

TEST_F(GraphRevampBilateralSemanticsTest, IsReadWriteWithSameRank_WriteReduceMatch)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto writeReduceTask = std::make_shared<TaskStubWriteReduce>(
        1, link, slice, slice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    TaskNode node(writeReduceTask.get(), 0, 0, 0);
    bool result = revamp.IsReadWriteWithSameRank(1, &node);
    EXPECT_TRUE(result);
}

TEST_F(GraphRevampBilateralSemanticsTest, IsReadWriteWithSameRank_NonRWType)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(slice, slice);
    TaskNode node(localCopy.get(), 0, 0, 0);
    bool result = revamp.IsReadWriteWithSameRank(1, &node);
    EXPECT_FALSE(result);
}

TEST_F(GraphRevampBilateralSemanticsTest, GetPeerRankByTaskNode_Read)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto readTask = std::make_shared<TaskStubRead>(5, link, slice, slice);
    TaskNode node(readTask.get(), 0, 0, 0);
    RankId peerRank = 0;
    HcclResult ret = revamp.GetPeerRankByTaskNode(&node, peerRank);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(peerRank, 5u);
}

TEST_F(GraphRevampBilateralSemanticsTest, GetPeerRankByTaskNode_Write)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto writeTask = std::make_shared<TaskStubWrite>(3, link, slice, slice);
    TaskNode node(writeTask.get(), 0, 0, 0);
    RankId peerRank = 0;
    HcclResult ret = revamp.GetPeerRankByTaskNode(&node, peerRank);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(peerRank, 3u);
}

TEST_F(GraphRevampBilateralSemanticsTest, GetPeerRankByTaskNode_ReadReduce)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto readReduceTask = std::make_shared<TaskStubReadReduce>(
        7, link, slice, slice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    TaskNode node(readReduceTask.get(), 0, 0, 0);
    RankId peerRank = 0;
    HcclResult ret = revamp.GetPeerRankByTaskNode(&node, peerRank);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(peerRank, 7u);
}

TEST_F(GraphRevampBilateralSemanticsTest, GetPeerRankByTaskNode_WriteReduce)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto writeReduceTask = std::make_shared<TaskStubWriteReduce>(
        9, link, slice, slice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    TaskNode node(writeReduceTask.get(), 0, 0, 0);
    RankId peerRank = 0;
    HcclResult ret = revamp.GetPeerRankByTaskNode(&node, peerRank);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(peerRank, 9u);
}

TEST_F(GraphRevampBilateralSemanticsTest, GetLinkProtoStubByTaskNode_Read)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto readTask = std::make_shared<TaskStubRead>(1, link, slice, slice);
    TaskNode node(readTask.get(), 0, 0, 0);
    LinkProtoStub result = LinkProtoStub::INVALID_A;
    HcclResult ret = revamp.GetLinkProtoStubByTaskNode(&node, result);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(result, LinkProtoStub::SDMA);
}

TEST_F(GraphRevampBilateralSemanticsTest, GetLinkProtoStubByTaskNode_Write)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::RDMA);
    auto writeTask = std::make_shared<TaskStubWrite>(1, link, slice, slice);
    TaskNode node(writeTask.get(), 0, 0, 0);
    LinkProtoStub result = LinkProtoStub::INVALID_A;
    HcclResult ret = revamp.GetLinkProtoStubByTaskNode(&node, result);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(result, LinkProtoStub::RDMA);
}

TEST_F(GraphRevampBilateralSemanticsTest, GetLinkProtoStubByTaskNode_ReadReduce)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    auto readReduceTask = std::make_shared<TaskStubReadReduce>(
        1, link, slice, slice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    TaskNode node(readReduceTask.get(), 0, 0, 0);
    LinkProtoStub result = LinkProtoStub::INVALID_A;
    HcclResult ret = revamp.GetLinkProtoStubByTaskNode(&node, result);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(result, LinkProtoStub::SDMA);
}

TEST_F(GraphRevampBilateralSemanticsTest, GetLinkProtoStubByTaskNode_WriteReduce)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    LinkInfo link(LinkProtoStub::RDMA);
    auto writeReduceTask = std::make_shared<TaskStubWriteReduce>(
        1, link, slice, slice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    TaskNode node(writeReduceTask.get(), 0, 0, 0);
    LinkProtoStub result = LinkProtoStub::INVALID_A;
    HcclResult ret = revamp.GetLinkProtoStubByTaskNode(&node, result);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(result, LinkProtoStub::RDMA);
}

TEST_F(GraphRevampBilateralSemanticsTest, InsertNode_LastNodeOfQueue)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(slice, slice);
    TaskNode headNode(localCopy.get(), 0, 0, 0);
    auto insertTask = std::make_shared<TaskStubBeingWritten>(1, LinkInfo(LinkProtoStub::SDMA), slice, slice);
    TaskNode insertNode(insertTask.get(), 0, 0, 0);
    HcclResult ret = revamp.InsertNode(&headNode, &insertNode);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(headNode.children.size(), 1u);
    EXPECT_EQ(headNode.children[0], &insertNode);
    EXPECT_EQ(insertNode.parents.size(), 1u);
    EXPECT_EQ(insertNode.parents[0], &headNode);
}

TEST_F(GraphRevampBilateralSemanticsTest, InsertNode_MiddleOfQueue)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    auto localCopy1 = std::make_shared<TaskStubLocalCopy>(slice, slice);
    auto localCopy2 = std::make_shared<TaskStubLocalCopy>(slice, slice);
    TaskNode headNode(localCopy1.get(), 0, 0, 0);
    TaskNode nextNode(localCopy2.get(), 0, 0, 1);
    BuildParentChild(&headNode, &nextNode);

    auto beingWritten = std::make_shared<TaskStubBeingWritten>(1, LinkInfo(LinkProtoStub::SDMA), slice, slice);
    TaskNode insertNode(beingWritten.get(), 0, 0, 0);
    HcclResult ret = revamp.InsertNode(&headNode, &insertNode);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(GraphRevampBilateralSemanticsTest, InsertNode_WithDifferentRankChild)
{
    GraphRevampBilateralSemantics revamp;
    DataSlice slice(BufferType::CCL, 0, 1024);
    auto localCopy = std::make_shared<TaskStubLocalCopy>(slice, slice);
    TaskNode headNode(localCopy.get(), 0, 0, 0);
    TaskNode otherRankNode(localCopy.get(), 1, 0, 0);
    BuildParentChild(&headNode, &otherRankNode);

    auto beingRead = std::make_shared<TaskStubBeingRead>(0, LinkInfo(LinkProtoStub::SDMA), slice, slice);
    TaskNode insertNode(beingRead.get(), 0, 0, 0);
    HcclResult ret = revamp.InsertNode(&headNode, &insertNode);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
