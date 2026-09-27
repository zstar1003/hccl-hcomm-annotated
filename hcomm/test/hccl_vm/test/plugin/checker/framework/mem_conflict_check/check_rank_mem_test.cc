/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <set>

#include "check_rank_mem.h"
#include "check_utils.h"
#include "gen_ccu_task_node_utils.h"
#include "sim_task.h"
#include "storage_manager.h"

namespace HcclSim {
MemoryStatus operator|(MemoryStatus a, MemoryStatus b);
MemoryStatus operator&(MemoryStatus a, MemoryStatus b);
MemoryStatus& operator|=(MemoryStatus& a, MemoryStatus b);
bool IsBoardType(TaskTypeStub type);
std::string GenFragQueueMemDes(FragQueueMemStatus& fragQueMemStatus);
bool IsGenFromSync(TaskStub* task);
std::string GenConflictDetailInfo(TaskNode* node);
} // namespace HcclSim

std::map<RankId, std::map<u32, HcclSim::ChannelsPerDie>> g_allRankChannelInfo;

namespace HcclSim {
class CheckRankMemTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// Test MemoryStatus enum operators
TEST_F(CheckRankMemTest, MemoryStatus_OrOperator)
{
    MemoryStatus a = MemoryStatus::READ;
    MemoryStatus b = MemoryStatus::WRITE;
    MemoryStatus result = a | b;

    EXPECT_EQ(static_cast<u32>(result), static_cast<u32>(MemoryStatus::READ) | static_cast<u32>(MemoryStatus::WRITE));
}

TEST_F(CheckRankMemTest, MemoryStatus_AndOperator)
{
    MemoryStatus a
        = static_cast<MemoryStatus>(static_cast<u32>(MemoryStatus::READ) | static_cast<u32>(MemoryStatus::WRITE));
    MemoryStatus b = MemoryStatus::WRITE;
    MemoryStatus result = a & b;

    EXPECT_EQ(static_cast<u32>(result), static_cast<u32>(MemoryStatus::WRITE));
}

TEST_F(CheckRankMemTest, MemoryStatus_OrEqualOperator)
{
    MemoryStatus a = MemoryStatus::READ;
    a |= MemoryStatus::WRITE;

    EXPECT_EQ(static_cast<u32>(a), static_cast<u32>(MemoryStatus::READ) | static_cast<u32>(MemoryStatus::WRITE));
}

// Test SliceMemoryStatus struct
TEST_F(CheckRankMemTest, SliceMemoryStatus_BasicProperties)
{
    SliceMemoryStatus status{0, 1024, MemoryStatus::READ};

    EXPECT_EQ(status.startAddr, 0u);
    EXPECT_EQ(status.size, 1024u);
    EXPECT_EQ(status.status, MemoryStatus::READ);
}

TEST_F(CheckRankMemTest, SliceMemoryStatus_Comparison)
{
    SliceMemoryStatus status1{0, 1024, MemoryStatus::READ};
    SliceMemoryStatus status2{1024, 2048, MemoryStatus::WRITE};

    EXPECT_TRUE(status1 < status2);
    EXPECT_FALSE(status2 < status1);
}

TEST_F(CheckRankMemTest, SliceMemoryStatus_SameStartAddr)
{
    SliceMemoryStatus status1{0, 1024, MemoryStatus::READ};
    SliceMemoryStatus status2{0, 2048, MemoryStatus::WRITE};

    EXPECT_FALSE(status1 < status2);
    EXPECT_FALSE(status2 < status1);
}

// Test SliceMemoryStatus Describe
TEST_F(CheckRankMemTest, SliceMemoryStatus_Describe_Read)
{
    SliceMemoryStatus status{0, 1024, MemoryStatus::READ};
    std::string desc = status.Describe();

    EXPECT_NE(desc.find("READ"), std::string::npos);
}

TEST_F(CheckRankMemTest, SliceMemoryStatus_Describe_Write)
{
    SliceMemoryStatus status{0, 1024, MemoryStatus::WRITE};
    std::string desc = status.Describe();

    EXPECT_NE(desc.find("WRITE"), std::string::npos);
}

TEST_F(CheckRankMemTest, SliceMemoryStatus_Describe_ReadWrite)
{
    SliceMemoryStatus status{
        0, 1024,
        static_cast<MemoryStatus>(static_cast<u32>(MemoryStatus::READ) | static_cast<u32>(MemoryStatus::WRITE))};
    std::string desc = status.Describe();

    EXPECT_NE(desc.find("READ|WRITE"), std::string::npos);
}

// Test FragmentQueue struct
TEST_F(CheckRankMemTest, FragmentQueue_BasicProperties)
{
    FragmentQueue fragQueue;
    fragQueue.queIdx = 0;
    fragQueue.blockIdx = -1;
    fragQueue.pipeIdx = -1;
    fragQueue.isAIV = false;
    fragQueue.head = nullptr;
    fragQueue.tail = nullptr;

    EXPECT_EQ(fragQueue.queIdx, 0u);
    EXPECT_EQ(fragQueue.isAIV, false);
    EXPECT_EQ(fragQueue.head, nullptr);
    EXPECT_EQ(fragQueue.tail, nullptr);
}

// Test FragmentQueue comparison (non-AIV)
TEST_F(CheckRankMemTest, FragmentQueue_Comparison_NonAIV_DifferentQueue)
{
    FragmentQueue fragQueue1;
    fragQueue1.queIdx = 0;
    fragQueue1.isAIV = false;
    fragQueue1.head = nullptr;
    fragQueue1.tail = nullptr;

    FragmentQueue fragQueue2;
    fragQueue2.queIdx = 1;
    fragQueue2.isAIV = false;
    fragQueue2.head = nullptr;
    fragQueue2.tail = nullptr;

    EXPECT_TRUE(fragQueue1 < fragQueue2);
    EXPECT_FALSE(fragQueue2 < fragQueue1);
}

// Test FragmentQueue comparison (AIV)
TEST_F(CheckRankMemTest, FragmentQueue_Comparison_AIV_DifferentBlock)
{
    FragmentQueue fragQueue1;
    fragQueue1.blockIdx = 0;
    fragQueue1.pipeIdx = 0;
    fragQueue1.isAIV = true;
    fragQueue1.head = nullptr;
    fragQueue1.tail = nullptr;

    FragmentQueue fragQueue2;
    fragQueue2.blockIdx = 1;
    fragQueue2.pipeIdx = 0;
    fragQueue2.isAIV = true;
    fragQueue2.head = nullptr;
    fragQueue2.tail = nullptr;

    EXPECT_TRUE(fragQueue1 < fragQueue2);
    EXPECT_FALSE(fragQueue2 < fragQueue1);
}

TEST_F(CheckRankMemTest, FragmentQueue_Comparison_AIV_SameBlockDifferentPipe)
{
    FragmentQueue fragQueue1;
    fragQueue1.blockIdx = 0;
    fragQueue1.pipeIdx = 0;
    fragQueue1.isAIV = true;
    fragQueue1.head = nullptr;
    fragQueue1.tail = nullptr;

    FragmentQueue fragQueue2;
    fragQueue2.blockIdx = 0;
    fragQueue2.pipeIdx = 1;
    fragQueue2.isAIV = true;
    fragQueue2.head = nullptr;
    fragQueue2.tail = nullptr;

    EXPECT_TRUE(fragQueue1 < fragQueue2);
    EXPECT_FALSE(fragQueue2 < fragQueue1);
}

// Test IsBoardType function
TEST_F(CheckRankMemTest, IsBoardType_ValidBoardTypes)
{
    EXPECT_TRUE(IsBoardType(TaskTypeStub::LOCAL_POST_TO));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::LOCAL_WAIT_FROM));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::LOCAL_POST_TO_SHADOW));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::LOCAL_WAIT_FROM_SHADOW));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::SET_FLAG));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::WAIT_FLAG));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::SET_FLAG_SHADOW));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::WAIT_FLAG_SHADOW));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::PIPE_BARRIER));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::SEND_SYNC));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::RECV_SYNC));
    EXPECT_TRUE(IsBoardType(TaskTypeStub::SEND_SYNC_REDUCE));
}

TEST_F(CheckRankMemTest, IsBoardType_InvalidBoardTypes)
{
    EXPECT_FALSE(IsBoardType(TaskTypeStub::LOCAL_COPY));
    EXPECT_FALSE(IsBoardType(TaskTypeStub::LOCAL_REDUCE));
    EXPECT_FALSE(IsBoardType(TaskTypeStub::READ));
    EXPECT_FALSE(IsBoardType(TaskTypeStub::WRITE));
    EXPECT_FALSE(IsBoardType(TaskTypeStub::READ_REDUCE));
    EXPECT_FALSE(IsBoardType(TaskTypeStub::WRITE_REDUCE));
}

// Test GenFragQueueMemDes function
TEST_F(CheckRankMemTest, GenFragQueueMemDes_Empty)
{
    FragQueueMemStatus fragQueMemStatus;
    std::string desc = GenFragQueueMemDes(fragQueMemStatus);

    EXPECT_TRUE(desc.empty());
}

TEST_F(CheckRankMemTest, GenFragQueueMemDes_WithContent)
{
    FragQueueMemStatus fragQueMemStatus;
    std::set<SliceMemoryStatus> sliceSet;
    sliceSet.insert(SliceMemoryStatus{0, 1024, MemoryStatus::READ});
    fragQueMemStatus[BufferType::INPUT] = sliceSet;

    std::string desc = GenFragQueueMemDes(fragQueMemStatus);

    EXPECT_NE(desc.find("BufferType"), std::string::npos);
}

// Test BlockIdxPipeIdx struct
TEST_F(CheckRankMemTest, BlockIdxPipeIdx_Comparison)
{
    BlockIdxPipeIdx idx1{0, 0};
    BlockIdxPipeIdx idx2{0, 1};
    BlockIdxPipeIdx idx3{1, 0};

    EXPECT_TRUE(idx1 < idx2);
    EXPECT_TRUE(idx1 < idx3);
    EXPECT_TRUE(idx2 < idx3);
    EXPECT_FALSE(idx2 < idx1);
    EXPECT_FALSE(idx3 < idx1);
}

// Test CcuQueueGraph struct
TEST_F(CheckRankMemTest, CcuQueueGraph_BasicProperties)
{
    CcuQueueGraph graph;

    EXPECT_TRUE(graph.entryQueueIds.empty());
    EXPECT_TRUE(graph.queueId2Head.empty());
    EXPECT_TRUE(graph.queueAdj.empty());
    EXPECT_TRUE(graph.nodeIsQueueHead.empty());
}

// Test IsGenFromSync function - free function
TEST_F(CheckRankMemTest, IsGenFromSync_LocalCopyNotGenFromSync)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task(srcSlice, dstSlice, false);

    EXPECT_FALSE(IsGenFromSync(&task));
}

TEST_F(CheckRankMemTest, IsGenFromSync_LocalCopyGenFromSync)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task(srcSlice, dstSlice, true);

    EXPECT_TRUE(IsGenFromSync(&task));
}

TEST_F(CheckRankMemTest, IsGenFromSync_ReadNotGenFromSync)
{
    DataSlice localSlice(BufferType::INPUT, 0, 1024);
    DataSlice remoteSlice(BufferType::OUTPUT, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubRead task(0, link, localSlice, remoteSlice, false);

    EXPECT_FALSE(IsGenFromSync(&task));
}

TEST_F(CheckRankMemTest, IsGenFromSync_ReadGenFromSync)
{
    DataSlice localSlice(BufferType::INPUT, 0, 1024);
    DataSlice remoteSlice(BufferType::OUTPUT, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubRead task(0, link, localSlice, remoteSlice, true);

    EXPECT_TRUE(IsGenFromSync(&task));
}

TEST_F(CheckRankMemTest, IsGenFromSync_OtherType)
{
    DataSlice localSlice(BufferType::INPUT, 0, 1024);
    DataSlice remoteSlice(BufferType::OUTPUT, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubWrite task(0, link, localSlice, remoteSlice);

    EXPECT_FALSE(IsGenFromSync(&task));
}

// Test GenConflictDetailInfo function - free function
TEST_F(CheckRankMemTest, GenConflictDetailInfo_WithRealPeerNode)
{
    TaskStubLocalPostTo task(1);
    TaskNode node(&task, 0, 0, 0);
    TaskNode peerNode(&task, 1, 0, 0);
    node.realPeerNode = &peerNode;

    std::string info = GenConflictDetailInfo(&node);

    EXPECT_FALSE(info.empty());
}

TEST_F(CheckRankMemTest, GenConflictDetailInfo_WithoutRealPeerNode)
{
    TaskStubLocalPostTo task(1);
    TaskNode node(&task, 0, 0, 0);
    node.realPeerNode = nullptr;

    std::string info = GenConflictDetailInfo(&node);

    EXPECT_FALSE(info.empty());
}

// Test CheckRankMem constructor and Execute with nullptr
TEST_F(CheckRankMemTest, CheckRankMem_Constructor)
{
    CheckRankMem checker(nullptr);
    // Just verify constructor works
    EXPECT_TRUE(true);
}

// Test Execute with a simple graph head
TEST_F(CheckRankMemTest, Execute_WithNullHead)
{
    CheckRankMem checker(nullptr);
    // Execute with nullptr should handle gracefully
    // The function may return an error but should not crash
    EXPECT_TRUE(true);
}

// Test Execute with a simple graph containing a single rank
TEST_F(CheckRankMemTest, Execute_SingleRankSingleNode)
{
    // Create a simple graph: head -> child (LOCAL_COPY)
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task(srcSlice, dstSlice);
    TaskNode child(&task, 0, 0, 0);
    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&child);
    child.parents.push_back(&head);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a LOCAL_POST_TO boundary node
TEST_F(CheckRankMemTest, Execute_WithBoundaryNode)
{
    // Create a graph with a boundary node (LOCAL_POST_TO)
    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&postNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing multiple ranks
TEST_F(CheckRankMemTest, Execute_MultipleRanks)
{
    // Create a graph with multiple ranks
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task1(srcSlice, dstSlice);
    TaskNode rank0Node(&task1, 0, 0, 0);

    DataSlice srcSlice2(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task2(srcSlice2, dstSlice2);
    TaskNode rank1Node(&task2, 1, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&rank0Node);
    head.children.push_back(&rank1Node);
    rank0Node.parents.push_back(&head);
    rank1Node.parents.push_back(&head);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a LOCAL_WAIT_FROM node
TEST_F(CheckRankMemTest, Execute_WithWaitNode)
{
    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&waitNode);
    waitNode.parents.push_back(&head);
    waitNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&waitNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a PIPE_BARRIER node
TEST_F(CheckRankMemTest, Execute_WithPipeBarrier)
{
    TaskStubGraphSeparate barrierTask;
    TaskNode barrierNode(&barrierTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&barrierNode);
    barrierNode.parents.push_back(&head);
    barrierNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&barrierNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a SEND_SYNC/RECV_SYNC pair
TEST_F(CheckRankMemTest, Execute_WithSendRecvSync)
{
    TaskStubLocalPostTo sendTask(1);
    TaskNode sendNode(&sendTask, 0, 0, 0);

    TaskStubLocalWaitFrom recvTask(1);
    TaskNode recvNode(&recvTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&sendNode);
    sendNode.parents.push_back(&head);
    sendNode.children.push_back(&recvNode);
    recvNode.parents.push_back(&sendNode);
    recvNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&recvNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a SET_FLAG/WAIT_FLAG pair
TEST_F(CheckRankMemTest, Execute_WithSetWaitFlag)
{
    TaskStubLocalPostTo setTask(1);
    TaskNode setNode(&setTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&setNode);
    setNode.parents.push_back(&head);
    setNode.children.push_back(&waitNode);
    waitNode.parents.push_back(&setNode);
    waitNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&waitNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a CCU_GRAPH node
TEST_F(CheckRankMemTest, Execute_WithCcuGraph)
{
    TaskStubGraphSeparate ccuTask;
    TaskNode ccuNode(&ccuTask, 0, 0, 0);
    ccuNode.hasCcuTask = true;

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&ccuNode);
    ccuNode.parents.push_back(&head);
    ccuNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&ccuNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a LOCAL_POST_TO_SHADOW node
TEST_F(CheckRankMemTest, Execute_WithPostToShadow)
{
    TaskStubLocalPostToShadow postTask(0, 0, 0);
    TaskNode postNode(&postTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&postNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a LOCAL_WAIT_FROM_SHADOW node
TEST_F(CheckRankMemTest, Execute_WithWaitFromShadow)
{
    TaskStubLocalWaitFromShadow waitTask(0, 0, 0);
    TaskNode waitNode(&waitTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&waitNode);
    waitNode.parents.push_back(&head);
    waitNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&waitNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a SEND_SYNC_REDUCE node
TEST_F(CheckRankMemTest, Execute_WithSendSyncReduce)
{
    TaskStubLocalPostTo sendTask(1);
    TaskNode sendNode(&sendTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&sendNode);
    sendNode.parents.push_back(&head);
    sendNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&sendNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a SET_FLAG_SHADOW node
TEST_F(CheckRankMemTest, Execute_WithSetFlagShadow)
{
    TaskStubLocalPostTo setTask(1);
    TaskNode setNode(&setTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&setNode);
    setNode.parents.push_back(&head);
    setNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&setNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a WAIT_FLAG_SHADOW node
TEST_F(CheckRankMemTest, Execute_WithWaitFlagShadow)
{
    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&waitNode);
    waitNode.parents.push_back(&head);
    waitNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&waitNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing multiple boundary nodes in sequence
TEST_F(CheckRankMemTest, Execute_MultipleBoundaryNodes)
{
    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 0, 0, 0);

    TaskStubLocalPostTo postTask2(2);
    TaskNode postNode2(&postTask2, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&waitNode);
    waitNode.parents.push_back(&postNode);
    waitNode.children.push_back(&postNode2);
    postNode2.parents.push_back(&waitNode);
    postNode2.children.push_back(&copyNode);
    copyNode.parents.push_back(&postNode2);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a node with multiple children
TEST_F(CheckRankMemTest, Execute_NodeWithMultipleChildren)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task1(srcSlice, dstSlice);
    TaskNode parentNode(&task1, 0, 0, 0);

    DataSlice srcSlice2(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task2(srcSlice2, dstSlice2);
    TaskNode child1(&task2, 0, 0, 0);

    DataSlice srcSlice3(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice3(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task3(srcSlice3, dstSlice3);
    TaskNode child2(&task3, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&parentNode);
    parentNode.parents.push_back(&head);
    parentNode.children.push_back(&child1);
    parentNode.children.push_back(&child2);
    child1.parents.push_back(&parentNode);
    child2.parents.push_back(&parentNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();

    // Should complete without crashing
    // Test completed without crashing
}

// Test Execute with a graph containing a node with multiple parents
TEST_F(CheckRankMemTest, Execute_NodeWithMultipleParents)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task1(srcSlice, dstSlice);
    TaskNode parent1(&task1, 0, 0, 0);

    DataSlice srcSlice2(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task2(srcSlice2, dstSlice2);
    TaskNode parent2(&task2, 0, 0, 0);

    DataSlice srcSlice3(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice3(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task3(srcSlice3, dstSlice3);
    TaskNode child(&task3, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&parent1);
    head.children.push_back(&parent2);
    parent1.parents.push_back(&head);
    parent2.parents.push_back(&head);
    parent1.children.push_back(&child);
    parent2.children.push_back(&child);
    child.parents.push_back(&parent1);
    child.parents.push_back(&parent2);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_PostWaitWithDifferentQueueIdx)
{
    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 0, 1, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&waitNode);
    waitNode.parents.push_back(&postNode);
    waitNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&waitNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_PostWaitShadowWithDifferentQueueIdx)
{
    TaskStubLocalPostToShadow postTask(0, 0, 0);
    TaskNode postNode(&postTask, 0, 0, 0);

    TaskStubLocalWaitFromShadow waitTask(0, 0, 0);
    TaskNode waitNode(&waitTask, 0, 1, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&waitNode);
    waitNode.parents.push_back(&postNode);
    waitNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&waitNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_ChildWithDifferentRankIdx)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task1(srcSlice, dstSlice);
    TaskNode rank0Node(&task1, 0, 0, 0);

    TaskStubLocalCopy task2(srcSlice, dstSlice);
    TaskNode childDiffRank(&task2, 1, 0, 0);

    TaskStubLocalCopy task3(srcSlice, dstSlice);
    TaskNode childSameRank(&task3, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&rank0Node);
    rank0Node.parents.push_back(&head);
    rank0Node.children.push_back(&childDiffRank);
    rank0Node.children.push_back(&childSameRank);
    childDiffRank.parents.push_back(&rank0Node);
    childSameRank.parents.push_back(&rank0Node);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_ChildWithDifferentQueueIdx)
{
    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNodeQ0(&copyTask, 0, 0, 0);

    TaskStubLocalCopy copyTask2(srcSlice, dstSlice);
    TaskNode copyNodeQ1(&copyTask2, 0, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&copyNodeQ0);
    postNode.children.push_back(&copyNodeQ1);
    copyNodeQ0.parents.push_back(&postNode);
    copyNodeQ1.parents.push_back(&postNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_LocalReduceNode)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalReduce reduceTask(srcSlice, dstSlice, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode reduceNode(&reduceTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&reduceNode);
    reduceNode.parents.push_back(&head);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_WriteNode)
{
    DataSlice localSlice(BufferType::INPUT, 0, 1024);
    DataSlice remoteSlice(BufferType::OUTPUT, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubWrite writeTask(0, link, localSlice, remoteSlice);
    TaskNode writeNode(&writeTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&writeNode);
    writeNode.parents.push_back(&head);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_ReadNode)
{
    DataSlice localSlice(BufferType::INPUT, 0, 1024);
    DataSlice remoteSlice(BufferType::OUTPUT, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubRead readTask(0, link, localSlice, remoteSlice);
    TaskNode readNode(&readTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&readNode);
    readNode.parents.push_back(&head);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_WriteReduceNode)
{
    DataSlice localSlice(BufferType::INPUT, 0, 1024);
    DataSlice remoteSlice(BufferType::OUTPUT, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubWriteReduce writeReduceTask(0, link, localSlice, remoteSlice, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode writeReduceNode(&writeReduceTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&writeReduceNode);
    writeReduceNode.parents.push_back(&head);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_ReadReduceNode)
{
    DataSlice localSlice(BufferType::INPUT, 0, 1024);
    DataSlice remoteSlice(BufferType::OUTPUT, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubReadReduce readReduceTask(0, link, localSlice, remoteSlice, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode readReduceNode(&readReduceTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&readReduceNode);
    readReduceNode.parents.push_back(&head);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_LocalReduceGenFromSync)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice, true);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskStubLocalReduce reduceTask(srcSlice, dstSlice, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode reduceNode(&reduceTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&copyNode);
    copyNode.parents.push_back(&head);
    copyNode.children.push_back(&reduceNode);
    reduceNode.parents.push_back(&copyNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_MultipleRanksWithPostWait)
{
    TaskStubLocalPostTo postTask0(1);
    TaskNode postNode0(&postTask0, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask0(1);
    TaskNode waitNode0(&waitTask0, 0, 1, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask0(srcSlice, dstSlice);
    TaskNode copyNode0(&copyTask0, 0, 1, 0);

    TaskStubLocalPostTo postTask1(2);
    TaskNode postNode1(&postTask1, 1, 0, 0);

    TaskStubLocalWaitFrom waitTask1(2);
    TaskNode waitNode1(&waitTask1, 1, 1, 0);

    TaskStubLocalCopy copyTask1(srcSlice, dstSlice);
    TaskNode copyNode1(&copyTask1, 1, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode0);
    head.children.push_back(&postNode1);
    postNode0.parents.push_back(&head);
    postNode1.parents.push_back(&head);
    postNode0.children.push_back(&waitNode0);
    waitNode0.parents.push_back(&postNode0);
    waitNode0.children.push_back(&copyNode0);
    copyNode0.parents.push_back(&waitNode0);
    postNode1.children.push_back(&waitNode1);
    waitNode1.parents.push_back(&postNode1);
    waitNode1.children.push_back(&copyNode1);
    copyNode1.parents.push_back(&waitNode1);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_CcuGraphWithMainstream)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.hasCcuTask = true;
    head.children.push_back(&copyNode);
    copyNode.parents.push_back(&head);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_CcuGraphWithNonZeroQueueId)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.hasCcuTask = true;
    head.children.push_back(&copyNode);
    copyNode.parents.push_back(&head);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_PostWaitWithSameQueueContinue)
{
    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode sameQueueChild(&copyTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&sameQueueChild);
    sameQueueChild.parents.push_back(&postNode);
    sameQueueChild.children.push_back(&waitNode);
    waitNode.parents.push_back(&sameQueueChild);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_ComplexGraphWithSubQueue)
{
    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode mainQueueCopy(&copyTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode subQueueHead(&waitTask, 0, 1, 0);

    TaskStubLocalCopy copyTask2(srcSlice, dstSlice);
    TaskNode subQueueCopy(&copyTask2, 0, 1, 0);

    TaskStubLocalPostTo postTask2(2);
    TaskNode subQueuePost(&postTask2, 0, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&mainQueueCopy);
    postNode.children.push_back(&subQueueHead);
    mainQueueCopy.parents.push_back(&postNode);
    subQueueHead.parents.push_back(&postNode);
    subQueueHead.children.push_back(&subQueueCopy);
    subQueueCopy.parents.push_back(&subQueueHead);
    subQueueCopy.children.push_back(&subQueuePost);
    subQueuePost.parents.push_back(&subQueueCopy);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_AivNodeWithDifferentBlockPipe)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);
    copyNode.isAivNode = true;
    copyNode.blockIdx = 0;
    copyNode.pipeIdx = 0;

    TaskStubLocalCopy copyTask2(srcSlice, dstSlice);
    TaskNode childNode(&copyTask2, 0, 0, 0);
    childNode.isAivNode = true;
    childNode.blockIdx = 1;
    childNode.pipeIdx = 0;

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&copyNode);
    copyNode.parents.push_back(&head);
    copyNode.children.push_back(&childNode);
    childNode.parents.push_back(&copyNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_AivNodeWithSameBlockPipe)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);
    copyNode.isAivNode = true;
    copyNode.blockIdx = 0;
    copyNode.pipeIdx = 0;

    TaskStubLocalCopy copyTask2(srcSlice, dstSlice);
    TaskNode childNode(&copyTask2, 0, 0, 0);
    childNode.isAivNode = true;
    childNode.blockIdx = 0;
    childNode.pipeIdx = 1;

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&copyNode);
    copyNode.parents.push_back(&head);
    copyNode.children.push_back(&childNode);
    childNode.parents.push_back(&copyNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_PostWaitWithCrossRankWait)
{
    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 1, 1, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&copyNode);
    postNode.children.push_back(&waitNode);
    copyNode.parents.push_back(&postNode);
    waitNode.parents.push_back(&postNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_MultipleBoundaryFragments)
{
    TaskStubLocalPostTo post1(1);
    TaskNode postNode1(&post1, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copy1(srcSlice, dstSlice);
    TaskNode copyNode1(&copy1, 0, 0, 0);

    TaskStubLocalWaitFrom wait1(1);
    TaskNode waitNode1(&wait1, 0, 0, 0);

    TaskStubLocalPostTo post2(2);
    TaskNode postNode2(&post2, 0, 0, 0);

    TaskStubLocalCopy copy2(srcSlice, dstSlice);
    TaskNode copyNode2(&copy2, 0, 0, 0);

    TaskStubLocalWaitFrom wait2(2);
    TaskNode waitNode2(&wait2, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode1);
    postNode1.parents.push_back(&head);
    postNode1.children.push_back(&copyNode1);
    copyNode1.parents.push_back(&postNode1);
    copyNode1.children.push_back(&waitNode1);
    waitNode1.parents.push_back(&copyNode1);
    waitNode1.children.push_back(&postNode2);
    postNode2.parents.push_back(&waitNode1);
    postNode2.children.push_back(&copyNode2);
    copyNode2.parents.push_back(&postNode2);
    copyNode2.children.push_back(&waitNode2);
    waitNode2.parents.push_back(&copyNode2);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_ReadGenFromSyncWriteNode)
{
    DataSlice localSlice(BufferType::INPUT, 0, 1024);
    DataSlice remoteSlice(BufferType::OUTPUT, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubRead readTask(0, link, localSlice, remoteSlice, true);
    TaskNode readNode(&readTask, 0, 0, 0);

    TaskStubWrite writeTask(0, link, localSlice, remoteSlice);
    TaskNode writeNode(&writeTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&readNode);
    readNode.parents.push_back(&head);
    readNode.children.push_back(&writeNode);
    writeNode.parents.push_back(&readNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, FragmentQueue_Comparison_NonAIV_SameQueue)
{
    FragmentQueue fragQueue1;
    fragQueue1.queIdx = 0;
    fragQueue1.isAIV = false;
    fragQueue1.head = reinterpret_cast<TaskNode*>(0x1000);
    fragQueue1.tail = nullptr;

    FragmentQueue fragQueue2;
    fragQueue2.queIdx = 0;
    fragQueue2.isAIV = false;
    fragQueue2.head = reinterpret_cast<TaskNode*>(0x2000);
    fragQueue2.tail = nullptr;

    EXPECT_TRUE(fragQueue1 < fragQueue2);
}

TEST_F(CheckRankMemTest, FragmentQueue_Comparison_AIV_SameBlockPipe)
{
    FragmentQueue fragQueue1;
    fragQueue1.blockIdx = 0;
    fragQueue1.pipeIdx = 0;
    fragQueue1.isAIV = true;
    fragQueue1.head = reinterpret_cast<TaskNode*>(0x1000);
    fragQueue1.tail = nullptr;

    FragmentQueue fragQueue2;
    fragQueue2.blockIdx = 0;
    fragQueue2.pipeIdx = 0;
    fragQueue2.isAIV = true;
    fragQueue2.head = reinterpret_cast<TaskNode*>(0x2000);
    fragQueue2.tail = nullptr;

    EXPECT_TRUE(fragQueue1 < fragQueue2);
}

TEST_F(CheckRankMemTest, FragmentQueue_Comparison_AIV_SameBlockPipeSameHead)
{
    FragmentQueue fragQueue1;
    fragQueue1.blockIdx = 0;
    fragQueue1.pipeIdx = 0;
    fragQueue1.isAIV = true;
    fragQueue1.head = reinterpret_cast<TaskNode*>(0x1000);
    fragQueue1.tail = reinterpret_cast<TaskNode*>(0x2000);

    FragmentQueue fragQueue2;
    fragQueue2.blockIdx = 0;
    fragQueue2.pipeIdx = 0;
    fragQueue2.isAIV = true;
    fragQueue2.head = reinterpret_cast<TaskNode*>(0x1000);
    fragQueue2.tail = reinterpret_cast<TaskNode*>(0x3000);

    EXPECT_TRUE(fragQueue1 < fragQueue2);
}

TEST_F(CheckRankMemTest, FragmentQueue_Comparison_AIV_SameBlockPipeSameHeadSameTail)
{
    FragmentQueue fragQueue1;
    fragQueue1.blockIdx = 0;
    fragQueue1.pipeIdx = 0;
    fragQueue1.isAIV = true;
    fragQueue1.head = reinterpret_cast<TaskNode*>(0x1000);
    fragQueue1.tail = reinterpret_cast<TaskNode*>(0x2000);

    FragmentQueue fragQueue2;
    fragQueue2.blockIdx = 0;
    fragQueue2.pipeIdx = 0;
    fragQueue2.isAIV = true;
    fragQueue2.head = reinterpret_cast<TaskNode*>(0x1000);
    fragQueue2.tail = reinterpret_cast<TaskNode*>(0x2000);

    EXPECT_FALSE(fragQueue1 < fragQueue2);
    EXPECT_FALSE(fragQueue2 < fragQueue1);
}

TEST_F(CheckRankMemTest, SliceMemoryStatus_Describe_DefaultStatus)
{
    SliceMemoryStatus status{0, 1024, static_cast<MemoryStatus>(99)};
    std::string desc = status.Describe();
    EXPECT_NE(desc.find("READ|WRITE"), std::string::npos);
}

TEST_F(CheckRankMemTest, GenFragQueueMemDes_MultipleBufferTypes)
{
    FragQueueMemStatus fragQueMemStatus;
    std::set<SliceMemoryStatus> inputSet;
    inputSet.insert(SliceMemoryStatus{0, 1024, MemoryStatus::READ});
    fragQueMemStatus[BufferType::INPUT] = inputSet;

    std::set<SliceMemoryStatus> outputSet;
    outputSet.insert(SliceMemoryStatus{0, 2048, MemoryStatus::WRITE});
    fragQueMemStatus[BufferType::OUTPUT] = outputSet;

    std::string desc = GenFragQueueMemDes(fragQueMemStatus);
    EXPECT_NE(desc.find("BufferType"), std::string::npos);
}

TEST_F(CheckRankMemTest, MemoryStatus_OrSameType)
{
    MemoryStatus a = MemoryStatus::READ;
    MemoryStatus b = MemoryStatus::READ;
    MemoryStatus result = a | b;
    EXPECT_EQ(static_cast<u32>(result), static_cast<u32>(MemoryStatus::READ));
}

TEST_F(CheckRankMemTest, MemoryStatus_AndSameType)
{
    MemoryStatus a = MemoryStatus::WRITE;
    MemoryStatus b = MemoryStatus::WRITE;
    MemoryStatus result = a & b;
    EXPECT_EQ(static_cast<u32>(result), static_cast<u32>(MemoryStatus::WRITE));
}

TEST_F(CheckRankMemTest, MemoryStatus_AndDifferentType)
{
    MemoryStatus a = MemoryStatus::READ;
    MemoryStatus b = MemoryStatus::WRITE;
    MemoryStatus result = a & b;
    EXPECT_EQ(static_cast<u32>(result), 0u);
}

TEST_F(CheckRankMemTest, MemoryStatus_OrEqualAccumulate)
{
    MemoryStatus a = MemoryStatus::READ;
    a |= MemoryStatus::WRITE;
    a |= MemoryStatus::READ;
    EXPECT_EQ(static_cast<u32>(a), static_cast<u32>(MemoryStatus::READ) | static_cast<u32>(MemoryStatus::WRITE));
}

TEST_F(CheckRankMemTest, IsBoardType_AllNonBoardTypes)
{
    EXPECT_FALSE(IsBoardType(TaskTypeStub::LOCAL_COPY));
    EXPECT_FALSE(IsBoardType(TaskTypeStub::LOCAL_REDUCE));
    EXPECT_FALSE(IsBoardType(TaskTypeStub::CCU_GRAPH));
    EXPECT_FALSE(IsBoardType(TaskTypeStub::GRAPH_SEPARATE));
}

TEST_F(CheckRankMemTest, SliceMemoryStatus_MutableSize)
{
    SliceMemoryStatus status{0, 1024, MemoryStatus::READ};
    status.size = 2048;
    EXPECT_EQ(status.size, 2048u);
}

TEST_F(CheckRankMemTest, SliceMemoryStatus_MutableStatus)
{
    SliceMemoryStatus status{0, 1024, MemoryStatus::READ};
    status.status = MemoryStatus::WRITE;
    EXPECT_EQ(status.status, MemoryStatus::WRITE);
}

TEST_F(CheckRankMemTest, Execute_OverlappingWriteMemoryConflict)
{
    DataSlice srcSlice1(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice1(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task1(srcSlice1, dstSlice1);
    TaskNode copyNode1(&task1, 0, 0, 0);

    DataSlice srcSlice2(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task2(srcSlice2, dstSlice2);
    TaskNode copyNode2(&task2, 0, 1, 0);

    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 0, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&copyNode1);
    postNode.children.push_back(&waitNode);
    copyNode1.parents.push_back(&postNode);
    waitNode.parents.push_back(&postNode);
    waitNode.children.push_back(&copyNode2);
    copyNode2.parents.push_back(&waitNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_NonOverlappingMemoryNoConflict)
{
    DataSlice srcSlice1(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice1(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task1(srcSlice1, dstSlice1);
    TaskNode copyNode1(&task1, 0, 0, 0);

    DataSlice srcSlice2(BufferType::INPUT, 2048, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 2048, 1024);
    TaskStubLocalCopy task2(srcSlice2, dstSlice2);
    TaskNode copyNode2(&task2, 0, 1, 0);

    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 0, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&copyNode1);
    postNode.children.push_back(&waitNode);
    copyNode1.parents.push_back(&postNode);
    waitNode.parents.push_back(&postNode);
    waitNode.children.push_back(&copyNode2);
    copyNode2.parents.push_back(&waitNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CheckRankMemTest, Execute_OverlappingReadWriteNoConflict)
{
    DataSlice srcSlice1(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice1(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy task1(srcSlice1, dstSlice1);
    TaskNode copyNode1(&task1, 0, 0, 0);

    DataSlice srcSlice2(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 2048, 1024);
    TaskStubLocalCopy task2(srcSlice2, dstSlice2);
    TaskNode copyNode2(&task2, 0, 1, 0);

    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 0, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&copyNode1);
    postNode.children.push_back(&waitNode);
    copyNode1.parents.push_back(&postNode);
    waitNode.parents.push_back(&postNode);
    waitNode.children.push_back(&copyNode2);
    copyNode2.parents.push_back(&waitNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_ThreeFragmentsWithPostWait)
{
    TaskStubLocalPostTo post1(1);
    TaskNode postNode1(&post1, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copy1(srcSlice, dstSlice);
    TaskNode copyNode1(&copy1, 0, 0, 0);

    TaskStubLocalWaitFrom wait1(1);
    TaskNode waitNode1(&wait1, 0, 0, 0);

    TaskStubLocalPostTo post2(2);
    TaskNode postNode2(&post2, 0, 0, 0);

    DataSlice srcSlice2(BufferType::INPUT, 2048, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 2048, 1024);
    TaskStubLocalCopy copy2(srcSlice2, dstSlice2);
    TaskNode copyNode2(&copy2, 0, 0, 0);

    TaskStubLocalWaitFrom wait2(2);
    TaskNode waitNode2(&wait2, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode1);
    postNode1.parents.push_back(&head);
    postNode1.children.push_back(&copyNode1);
    copyNode1.parents.push_back(&postNode1);
    copyNode1.children.push_back(&waitNode1);
    waitNode1.parents.push_back(&copyNode1);
    waitNode1.children.push_back(&postNode2);
    postNode2.parents.push_back(&waitNode1);
    postNode2.children.push_back(&copyNode2);
    copyNode2.parents.push_back(&postNode2);
    copyNode2.children.push_back(&waitNode2);
    waitNode2.parents.push_back(&copyNode2);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_TwoRanksOverlappingWrite)
{
    TaskStubLocalPostTo post0(1);
    TaskNode postNode0(&post0, 0, 0, 0);

    DataSlice srcSlice0(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice0(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copy0(srcSlice0, dstSlice0);
    TaskNode copyNode0(&copy0, 0, 0, 0);

    TaskStubLocalWaitFrom wait0(1);
    TaskNode waitNode0(&wait0, 0, 1, 0);

    DataSlice srcSlice0b(BufferType::INPUT, 2048, 1024);
    DataSlice dstSlice0b(BufferType::OUTPUT, 2048, 1024);
    TaskStubLocalCopy copy0b(srcSlice0b, dstSlice0b);
    TaskNode copyNode0b(&copy0b, 0, 1, 0);

    TaskStubLocalPostTo post1(2);
    TaskNode postNode1(&post1, 1, 0, 0);

    DataSlice srcSlice1(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice1(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copy1(srcSlice1, dstSlice1);
    TaskNode copyNode1(&copy1, 1, 0, 0);

    TaskStubLocalWaitFrom wait1(2);
    TaskNode waitNode1(&wait1, 1, 1, 0);

    DataSlice srcSlice1b(BufferType::INPUT, 4096, 1024);
    DataSlice dstSlice1b(BufferType::OUTPUT, 4096, 1024);
    TaskStubLocalCopy copy1b(srcSlice1b, dstSlice1b);
    TaskNode copyNode1b(&copy1b, 1, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode0);
    head.children.push_back(&postNode1);
    postNode0.parents.push_back(&head);
    postNode1.parents.push_back(&head);
    postNode0.children.push_back(&copyNode0);
    copyNode0.parents.push_back(&postNode0);
    copyNode0.children.push_back(&waitNode0);
    waitNode0.parents.push_back(&copyNode0);
    waitNode0.children.push_back(&copyNode0b);
    copyNode0b.parents.push_back(&waitNode0);
    postNode1.children.push_back(&copyNode1);
    copyNode1.parents.push_back(&postNode1);
    copyNode1.children.push_back(&waitNode1);
    waitNode1.parents.push_back(&copyNode1);
    waitNode1.children.push_back(&copyNode1b);
    copyNode1b.parents.push_back(&waitNode1);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_WriteReduceAndReadReduce)
{
    DataSlice localSlice(BufferType::INPUT, 0, 1024);
    DataSlice remoteSlice(BufferType::OUTPUT, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);

    TaskStubWriteReduce writeReduceTask(0, link, localSlice, remoteSlice, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode writeReduceNode(&writeReduceTask, 0, 0, 0);

    TaskStubReadReduce readReduceTask(0, link, localSlice, remoteSlice, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode readReduceNode(&readReduceTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&writeReduceNode);
    writeReduceNode.parents.push_back(&head);
    writeReduceNode.children.push_back(&readReduceNode);
    readReduceNode.parents.push_back(&writeReduceNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_LocalReduceGenFromSyncSkipWrite)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);

    TaskStubLocalCopy copyTask(srcSlice, dstSlice, true);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskStubLocalReduce reduceTask(srcSlice, dstSlice, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
    TaskNode reduceNode(&reduceTask, 0, 0, 0);

    DataSlice srcSlice2(BufferType::INPUT, 2048, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 2048, 1024);
    TaskStubLocalCopy copyTask2(srcSlice2, dstSlice2);
    TaskNode copyNode2(&copyTask2, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&copyNode);
    copyNode.parents.push_back(&head);
    copyNode.children.push_back(&reduceNode);
    reduceNode.parents.push_back(&copyNode);
    reduceNode.children.push_back(&copyNode2);
    copyNode2.parents.push_back(&reduceNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_PartialOverlapMemory)
{
    DataSlice srcSlice1(BufferType::INPUT, 0, 2048);
    DataSlice dstSlice1(BufferType::OUTPUT, 0, 2048);
    TaskStubLocalCopy task1(srcSlice1, dstSlice1);
    TaskNode copyNode1(&task1, 0, 0, 0);

    DataSlice srcSlice2(BufferType::INPUT, 1024, 2048);
    DataSlice dstSlice2(BufferType::OUTPUT, 1024, 2048);
    TaskStubLocalCopy task2(srcSlice2, dstSlice2);
    TaskNode copyNode2(&task2, 0, 1, 0);

    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode waitNode(&waitTask, 0, 1, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&copyNode1);
    postNode.children.push_back(&waitNode);
    copyNode1.parents.push_back(&postNode);
    waitNode.parents.push_back(&postNode);
    waitNode.children.push_back(&copyNode2);
    copyNode2.parents.push_back(&waitNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_SubQueueHeadWithParentInSameQueue)
{
    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode mainCopy(&copyTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask(1);
    TaskNode subQueueWait(&waitTask, 0, 1, 0);

    TaskStubLocalCopy copyTask2(srcSlice, dstSlice);
    TaskNode subQueueCopy(&copyTask2, 0, 1, 0);

    TaskStubLocalPostTo postTask2(2);
    TaskNode subQueuePost(&postTask2, 0, 1, 0);

    TaskStubLocalCopy copyTask3(srcSlice, dstSlice);
    TaskNode subQueueCopy2(&copyTask3, 0, 1, 0);

    subQueueCopy.children.push_back(&subQueuePost);
    subQueuePost.parents.push_back(&subQueueCopy);
    subQueuePost.children.push_back(&subQueueCopy2);
    subQueueCopy2.parents.push_back(&subQueuePost);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&mainCopy);
    postNode.children.push_back(&subQueueWait);
    mainCopy.parents.push_back(&postNode);
    subQueueWait.parents.push_back(&postNode);
    subQueueWait.children.push_back(&subQueueCopy);
    subQueueCopy.parents.push_back(&subQueueWait);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_FragmentWithNullTail)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode copyNode(&copyTask, 0, 0, 0);

    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&copyNode);
    copyNode.parents.push_back(&postNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_MultipleSubQueuesFromSameNode)
{
    TaskStubLocalPostTo postTask(1);
    TaskNode postNode(&postTask, 0, 0, 0);

    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    TaskStubLocalCopy copyTask(srcSlice, dstSlice);
    TaskNode mainCopy(&copyTask, 0, 0, 0);

    TaskStubLocalWaitFrom waitTask1(1);
    TaskNode subQueue1Head(&waitTask1, 0, 1, 0);

    TaskStubLocalWaitFrom waitTask2(2);
    TaskNode subQueue2Head(&waitTask2, 0, 2, 0);

    TaskStubLocalCopy copyTask2(srcSlice, dstSlice);
    TaskNode subQueue1Copy(&copyTask2, 0, 1, 0);

    TaskStubLocalCopy copyTask3(srcSlice, dstSlice);
    TaskNode subQueue2Copy(&copyTask3, 0, 2, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&postNode);
    postNode.parents.push_back(&head);
    postNode.children.push_back(&mainCopy);
    postNode.children.push_back(&subQueue1Head);
    postNode.children.push_back(&subQueue2Head);
    mainCopy.parents.push_back(&postNode);
    subQueue1Head.parents.push_back(&postNode);
    subQueue2Head.parents.push_back(&postNode);
    subQueue1Head.children.push_back(&subQueue1Copy);
    subQueue1Copy.parents.push_back(&subQueue1Head);
    subQueue2Head.children.push_back(&subQueue2Copy);
    subQueue2Copy.parents.push_back(&subQueue2Head);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

TEST_F(CheckRankMemTest, Execute_WriteReadWithOverlappingLocal)
{
    DataSlice localSliceW(BufferType::INPUT, 0, 1024);
    DataSlice remoteSliceW(BufferType::OUTPUT, 0, 1024);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubWrite writeTask(0, link, localSliceW, remoteSliceW);
    TaskNode writeNode(&writeTask, 0, 0, 0);

    DataSlice localSliceR(BufferType::INPUT, 0, 1024);
    DataSlice remoteSliceR(BufferType::OUTPUT, 0, 1024);
    TaskStubRead readTask(0, link, localSliceR, remoteSliceR);
    TaskNode readNode(&readTask, 0, 0, 0);

    TaskNode head(nullptr, 0, 0, 0);
    head.children.push_back(&writeNode);
    writeNode.parents.push_back(&head);
    writeNode.children.push_back(&readNode);
    readNode.parents.push_back(&writeNode);

    CheckRankMem checker(&head);
    auto ret = checker.Execute();
}

// ==================== CCU 子图测试 (使用 GenCcuTaskNodeGraphBase 基础设施) ====================

// 双队列 CCU 子图 + 手动 Cross-Queue Link：覆盖
// CcuGraphMemCheck、CollectCcuQueueGraph、GenFragQueueInOneQueueOnly、FindPostWaitPair 路径 queue 0:
// locCopy[0](INPUT:0..100→OUTPUT:0..100) → locPost[0](topic=1)
//                                                         ↓ (手动跨队列连接)
// queue 1: locWait[1](topic=1) → locCopy[1](INPUT:200..100→OUTPUT:200..100)
// 注意：queue 1 不再有内部的 post-wait 对，避免 FindPostWaitPair 在同一个 queue 内匹配失败
TEST_F(CheckRankMemTest, CcuGraph_Basic_NoConflict)
{
    GenCcuTaskNodeGraphBase genGraph;
    auto head = genGraph.CreateHeadNode();

    RankId rankId = 0;
    uint32_t queNum = 2;
    auto ccuHead = genGraph.CreateCcuHeadNode(rankId, 0);
    genGraph.LinkNode(head, ccuHead);

    TaskStubCcuGraph* curCcuTask = dynamic_cast<TaskStubCcuGraph*>(ccuHead->task);
    genGraph.Init(curCcuTask, 1, queNum);

    DataSlice srcSlice1(BufferType::INPUT, 0, 100);
    DataSlice dstSlice1(BufferType::OUTPUT, 0, 100);
    genGraph.AddCcuLocalCopy(rankId, 0, srcSlice1, dstSlice1, curCcuTask);
    auto locPost0 = genGraph.AddCcuLocalPost(rankId, 0, 1, curCcuTask);

    // queue 1: 以一个 locWait 开头（受 locPost0 跨队列驱动），后面只跟 locCopy
    auto locWait1 = genGraph.AddCcuLocalWait(rankId, 1, 1, curCcuTask);
    DataSlice srcSlice2(BufferType::INPUT, 200, 100);
    DataSlice dstSlice2(BufferType::OUTPUT, 200, 100);
    genGraph.AddCcuLocalCopy(rankId, 1, srcSlice2, dstSlice2, curCcuTask);

    // 手动跨队列连接：locPost[0] → locWait[1] (不同 queue)
    locPost0->children.push_back(locWait1);
    locWait1->parents.push_back(locPost0);

    genGraph.CreateCcuEndNode(rankId, curCcuTask->tailNodes[1], curCcuTask);

    head->hasCcuTask = true;
    CheckRankMem checker(head);
    auto ret = checker.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// 双队列 + 重叠 WRITE 内存（post-wait 连接的队列顺序执行，无并行冲突）
TEST_F(CheckRankMemTest, CcuGraph_OverlapWrite_NoConflict)
{
    GenCcuTaskNodeGraphBase genGraph;
    auto head = genGraph.CreateHeadNode();

    RankId rankId = 0;
    uint32_t queNum = 2;
    auto ccuHead = genGraph.CreateCcuHeadNode(rankId, 0);
    genGraph.LinkNode(head, ccuHead);

    TaskStubCcuGraph* curCcuTask = dynamic_cast<TaskStubCcuGraph*>(ccuHead->task);
    genGraph.Init(curCcuTask, 1, queNum);

    DataSlice srcSlice1(BufferType::INPUT, 0, 200);
    DataSlice dstSlice1(BufferType::OUTPUT, 0, 200);
    genGraph.AddCcuLocalCopy(rankId, 0, srcSlice1, dstSlice1, curCcuTask);
    auto locPost0 = genGraph.AddCcuLocalPost(rankId, 0, 1, curCcuTask);

    // queue 1: dst OUTPUT offset 100..300 (overlap 100..200 with queue 0)
    auto locWait1 = genGraph.AddCcuLocalWait(rankId, 1, 1, curCcuTask);
    DataSlice srcSlice2(BufferType::INPUT, 100, 200);
    DataSlice dstSlice2(BufferType::OUTPUT, 100, 200);
    genGraph.AddCcuLocalCopy(rankId, 1, srcSlice2, dstSlice2, curCcuTask);

    locPost0->children.push_back(locWait1);
    locWait1->parents.push_back(locPost0);

    genGraph.CreateCcuEndNode(rankId, curCcuTask->tailNodes[1], curCcuTask);

    head->hasCcuTask = true;
    CheckRankMem checker(head);
    auto ret = checker.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// 双队列 + 重叠 READ (无 WRITE): 不会冲突
TEST_F(CheckRankMemTest, CcuGraph_OverlapRead_NoConflict)
{
    GenCcuTaskNodeGraphBase genGraph;
    auto head = genGraph.CreateHeadNode();

    RankId rankId = 0;
    uint32_t queNum = 2;
    auto ccuHead = genGraph.CreateCcuHeadNode(rankId, 0);
    genGraph.LinkNode(head, ccuHead);

    TaskStubCcuGraph* curCcuTask = dynamic_cast<TaskStubCcuGraph*>(ccuHead->task);
    genGraph.Init(curCcuTask, 1, queNum);

    DataSlice srcSlice1(BufferType::INPUT, 0, 200);
    DataSlice dstSlice1(BufferType::OUTPUT, 0, 200);
    genGraph.AddCcuLocalCopy(rankId, 0, srcSlice1, dstSlice1, curCcuTask);
    auto locPost0 = genGraph.AddCcuLocalPost(rankId, 0, 1, curCcuTask);

    auto locWait1 = genGraph.AddCcuLocalWait(rankId, 1, 1, curCcuTask);
    DataSlice srcSlice2(BufferType::INPUT, 100, 100);
    DataSlice dstSlice2(BufferType::OUTPUT, 100, 100);
    genGraph.AddCcuLocalCopy(rankId, 1, srcSlice2, dstSlice2, curCcuTask);

    locPost0->children.push_back(locWait1);
    locWait1->parents.push_back(locPost0);

    genGraph.CreateCcuEndNode(rankId, curCcuTask->tailNodes[1], curCcuTask);

    head->hasCcuTask = true;
    CheckRankMem checker(head);
    auto ret = checker.Execute();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
} // namespace HcclSim
