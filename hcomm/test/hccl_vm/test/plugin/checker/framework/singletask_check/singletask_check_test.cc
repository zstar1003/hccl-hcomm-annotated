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
#include <map>
#include <vector>

#define private public
#include "data_slice.h"
#include "sim_task.h"
#include "singletask_check.h"
#include "storage_manager.h"
#include "task_ccu.h"
#include "task_utils.h"

#undef private
std::map<RankId, std::map<u32, HcclSim::ChannelsPerDie>> g_allRankChannelInfo;

namespace HcclSim {
class SingleTaskCheckTest : public testing::Test {
protected:
    void SetUp() override
    {
        createdNodes.clear();
        createdTasks.clear();
    }

    void TearDown() override
    {
        for (auto& n : createdNodes) {
            delete n;
        }
        createdNodes.clear();
        createdTasks.clear();
    }

    TaskNode* MakeNode(TaskStub* task, RankId rankId, u32 queIdx, u32 pos)
    {
        auto* node = new TaskNode(task, rankId, queIdx, pos);
        createdNodes.push_back(node);
        return node;
    }

    template <typename T, typename... Args>
    T* MakeTask(Args&&... args)
    {
        auto* t = new T(std::forward<Args>(args)...);
        createdTasks.push_back(std::unique_ptr<TaskStub>(t));
        return t;
    }

    std::vector<TaskNode*> createdNodes;
    std::vector<std::unique_ptr<TaskStub>> createdTasks;
};

// ==================== CheckSlaveTaskQueue Tests ====================

TEST_F(SingleTaskCheckTest, CheckSlaveTaskQueue_Empty)
{
    SingleTaskCheck checker;
    AllRankTaskQueues q;
    EXPECT_EQ(checker.CheckSlaveTaskQueue(q), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckSlaveTaskQueue_SingleQueueSizeLessThan2)
{
    SingleTaskCheck checker;
    AllRankTaskQueues allRank;
    SingleTaskQueue sq;
    sq.push_back({});
    sq.push_back({});
    allRank[0] = sq;

    EXPECT_EQ(checker.CheckSlaveTaskQueue(allRank), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckSlaveTaskQueue_ValidWaitFromAndPostTo)
{
    unsetenv("HCCLVM_TOPO_TYPE");
    SingleTaskCheck checker;
    AllRankTaskQueues allRank;

    auto* waitFrom = MakeTask<TaskStubLocalWaitFrom>(1, 0, 1);
    auto* postTo = MakeTask<TaskStubLocalPostTo>(1, 1, 0);

    SingleTaskQueue sq;
    sq.push_back({});
    sq.push_back(
        {std::shared_ptr<TaskStub>(waitFrom, [](TaskStub*) {}), std::shared_ptr<TaskStub>(postTo, [](TaskStub*) {})});
    allRank[0] = sq;

    EXPECT_EQ(checker.CheckSlaveTaskQueue(allRank), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckSlaveTaskQueue_FirstTaskNotWaitFrom_ReturnsError)
{
    unsetenv("HCCLVM_TOPO_TYPE");
    SingleTaskCheck checker;
    AllRankTaskQueues allRank;

    DataSlice sliceA(BufferType::INPUT, 0, 64);
    DataSlice sliceB(BufferType::OUTPUT, 0, 64);
    auto* localCopy = MakeTask<TaskStubLocalCopy>(sliceA, sliceB);
    auto* postTo = MakeTask<TaskStubLocalPostTo>(1, 1, 0);

    SingleTaskQueue sq;
    sq.push_back({});
    sq.push_back(
        {std::shared_ptr<TaskStub>(localCopy, [](TaskStub*) {}), std::shared_ptr<TaskStub>(postTo, [](TaskStub*) {})});
    allRank[0] = sq;

    EXPECT_EQ(checker.CheckSlaveTaskQueue(allRank), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(SingleTaskCheckTest, CheckSlaveTaskQueue_LastTaskNotPostTo_ReturnsError)
{
    unsetenv("HCCLVM_TOPO_TYPE");
    SingleTaskCheck checker;
    AllRankTaskQueues allRank;

    auto* waitFrom = MakeTask<TaskStubLocalWaitFrom>(1, 0, 1);
    DataSlice sliceA(BufferType::INPUT, 0, 64);
    DataSlice sliceB(BufferType::OUTPUT, 0, 64);
    auto* localCopy = MakeTask<TaskStubLocalCopy>(sliceA, sliceB);

    SingleTaskQueue sq;
    sq.push_back({});
    sq.push_back(
        {std::shared_ptr<TaskStub>(waitFrom, [](TaskStub*) {}),
         std::shared_ptr<TaskStub>(localCopy, [](TaskStub*) {})});
    allRank[0] = sq;

    EXPECT_EQ(checker.CheckSlaveTaskQueue(allRank), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(SingleTaskCheckTest, CheckSlaveTaskQueue_EmptyLocalCopyAtEnd_Skips)
{
    unsetenv("HCCLVM_TOPO_TYPE");
    SingleTaskCheck checker;
    AllRankTaskQueues allRank;

    auto* waitFrom = MakeTask<TaskStubLocalWaitFrom>(1, 0, 1);
    DataSlice emptySlice(BufferType::INPUT, 0, 0);
    auto* emptyCopy = MakeTask<TaskStubLocalCopy>(emptySlice, emptySlice);
    auto* nonEmptyCopy
        = MakeTask<TaskStubLocalCopy>(DataSlice(BufferType::INPUT, 0, 128), DataSlice(BufferType::OUTPUT, 0, 128));
    auto* postTo = MakeTask<TaskStubLocalPostTo>(1, 1, 0);

    SingleTaskQueue sq;
    sq.push_back({});
    sq.push_back(
        {std::shared_ptr<TaskStub>(waitFrom, [](TaskStub*) {}),
         std::shared_ptr<TaskStub>(nonEmptyCopy, [](TaskStub*) {}),
         std::shared_ptr<TaskStub>(emptyCopy, [](TaskStub*) {}), std::shared_ptr<TaskStub>(postTo, [](TaskStub*) {})});
    allRank[0] = sq;

    EXPECT_EQ(checker.CheckSlaveTaskQueue(allRank), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckSlaveTaskQueue_HFTopology_SkipsCheck)
{
    setenv("HCCLVM_TOPO_TYPE", "HF", 1);
    SingleTaskCheck checker;
    AllRankTaskQueues allRank;

    DataSlice sliceA(BufferType::INPUT, 0, 64);
    auto* localCopy = MakeTask<TaskStubLocalCopy>(sliceA, sliceA);

    SingleTaskQueue sq;
    sq.push_back({});
    sq.push_back(
        {std::shared_ptr<TaskStub>(localCopy, [](TaskStub*) {}),
         std::shared_ptr<TaskStub>(localCopy, [](TaskStub*) {})});
    allRank[0] = sq;

    EXPECT_EQ(checker.CheckSlaveTaskQueue(allRank), HcclResult::HCCL_SUCCESS);
    unsetenv("HCCLVM_TOPO_TYPE");
}

// ==================== CheckSingleSlice Tests ====================

TEST_F(SingleTaskCheckTest, CheckSingleSlice_MSType_ReturnsSuccess)
{
    SingleTaskCheck checker;
    DataSlice msSlice(BufferType::MS, 0, 1024);
    EXPECT_EQ(checker.CheckSingleSlice(0, 0, 0, msSlice, 0), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckSingleSlice_ZeroSize_ReturnsSuccess)
{
    SingleTaskCheck checker;
    DataSlice slice(BufferType::INPUT, 0, 0);
    EXPECT_EQ(checker.CheckSingleSlice(0, 0, 0, slice, 0), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckSingleSlice_OutOfBounds)
{
    SingleTaskCheck checker;
    DataSlice slice(BufferType::INPUT, UINT64_MAX - 1, 128);
    EXPECT_EQ(checker.CheckSingleSlice(0, 0, 0, slice, 0), HcclResult::HCCL_E_INTERNAL);
}

// ==================== CheckTwoSliceOverlap Tests ====================

TEST_F(SingleTaskCheckTest, CheckTwoSliceOverlap_DifferentTypes_NoConflict)
{
    SingleTaskCheck checker;
    DataSlice sliceA(BufferType::INPUT, 0, 1024);
    DataSlice sliceB(BufferType::OUTPUT, 0, 1024);
    EXPECT_EQ(checker.CheckTwoSliceOverlap(0, 0, 0, sliceA, sliceB), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckTwoSliceOverlap_ZeroSizeSlice_NoConflict)
{
    SingleTaskCheck checker;
    DataSlice sliceA(BufferType::INPUT, 0, 0);
    DataSlice sliceB(BufferType::INPUT, 0, 1024);
    EXPECT_EQ(checker.CheckTwoSliceOverlap(0, 0, 0, sliceA, sliceB), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckTwoSliceOverlap_BothZeroSize_NoConflict)
{
    SingleTaskCheck checker;
    DataSlice sliceA(BufferType::INPUT, 100, 0);
    DataSlice sliceB(BufferType::INPUT, 200, 0);
    EXPECT_EQ(checker.CheckTwoSliceOverlap(0, 0, 0, sliceA, sliceB), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckTwoSliceOverlap_NonOverlapping)
{
    SingleTaskCheck checker;
    DataSlice sliceA(BufferType::INPUT, 0, 1024);
    DataSlice sliceB(BufferType::INPUT, 2048, 1024);
    EXPECT_EQ(checker.CheckTwoSliceOverlap(0, 0, 0, sliceA, sliceB), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckTwoSliceOverlap_Adjacent_NoOverlap)
{
    SingleTaskCheck checker;
    DataSlice sliceA(BufferType::INPUT, 0, 1024);
    DataSlice sliceB(BufferType::INPUT, 1024, 1024);
    EXPECT_EQ(checker.CheckTwoSliceOverlap(0, 0, 0, sliceA, sliceB), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckTwoSliceOverlap_ExactSame_Conflict)
{
    SingleTaskCheck checker;
    DataSlice sliceA(BufferType::INPUT, 0, 1024);
    DataSlice sliceB(BufferType::INPUT, 0, 1024);
    EXPECT_EQ(checker.CheckTwoSliceOverlap(0, 0, 0, sliceA, sliceB), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(SingleTaskCheckTest, CheckTwoSliceOverlap_Contained_Conflict)
{
    SingleTaskCheck checker;
    DataSlice sliceA(BufferType::INPUT, 0, 4096);
    DataSlice sliceB(BufferType::INPUT, 1024, 1024);
    EXPECT_EQ(checker.CheckTwoSliceOverlap(0, 0, 0, sliceA, sliceB), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(SingleTaskCheckTest, CheckTwoSliceOverlap_PartialAOverlapsBStart)
{
    SingleTaskCheck checker;
    DataSlice sliceA(BufferType::INPUT, 512, 1024);
    DataSlice sliceB(BufferType::INPUT, 0, 1024);
    EXPECT_EQ(checker.CheckTwoSliceOverlap(0, 0, 0, sliceA, sliceB), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(SingleTaskCheckTest, CheckTwoSliceOverlap_OffByOne_Conflict)
{
    SingleTaskCheck checker;
    DataSlice sliceA(BufferType::INPUT, 0, 1001);
    DataSlice sliceB(BufferType::INPUT, 1000, 500);
    EXPECT_EQ(checker.CheckTwoSliceOverlap(0, 0, 0, sliceA, sliceB), HcclResult::HCCL_E_INTERNAL);
}

// ==================== AddChildrenToQueue Tests ====================

TEST_F(SingleTaskCheckTest, AddChildrenToQueue_NoChildren_EmptyAfter)
{
    SingleTaskCheck checker;
    std::set<TaskNode*> visited;
    std::queue<TaskNode*> walkQue;
    std::set<TaskNode*> simulated;

    auto* node = MakeNode(nullptr, 0, 0, 0);
    checker.AddChildrenToQueue(node, visited, walkQue, simulated);
    EXPECT_TRUE(walkQue.empty());
    EXPECT_TRUE(visited.empty());
}

TEST_F(SingleTaskCheckTest, AddChildrenToQueue_WithChildren_AddedToQueue)
{
    SingleTaskCheck checker;
    std::set<TaskNode*> visited;
    std::queue<TaskNode*> walkQue;
    std::set<TaskNode*> simulated;

    auto* parent = MakeNode(nullptr, 0, 0, 0);
    auto* child1 = MakeNode(nullptr, 0, 0, 1);
    auto* child2 = MakeNode(nullptr, 0, 0, 2);
    parent->children = {child1, child2};

    checker.AddChildrenToQueue(parent, visited, walkQue, simulated);
    EXPECT_EQ(walkQue.size(), 2u);
    EXPECT_EQ(visited.size(), 2u);
}

TEST_F(SingleTaskCheckTest, AddChildrenToQueue_AlreadyVisited_Skipped)
{
    SingleTaskCheck checker;
    std::set<TaskNode*> visited;
    std::queue<TaskNode*> walkQue;
    std::set<TaskNode*> simulated;

    auto* parent = MakeNode(nullptr, 0, 0, 0);
    auto* child1 = MakeNode(nullptr, 0, 0, 1);
    auto* child2 = MakeNode(nullptr, 0, 0, 2);
    parent->children = {child1, child2};
    visited.insert(child1);

    checker.AddChildrenToQueue(parent, visited, walkQue, simulated);
    EXPECT_EQ(walkQue.size(), 1u);
    EXPECT_EQ(visited.size(), 2u);
}

// ==================== CheckSingleTaskMem Tests ====================

TEST_F(SingleTaskCheckTest, CheckSingleTaskMem_NullTask_ReturnsSuccess)
{
    SingleTaskCheck checker;
    auto* node = MakeNode(nullptr, 0, 0, 0);
    EXPECT_EQ(checker.CheckSingleTaskMem(node), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckSingleTaskMem_LocalCopy_Success)
{
    SingleTaskCheck checker;
    DataSlice src(BufferType::INPUT, 0, 0);
    DataSlice dst(BufferType::OUTPUT, 0, 0);
    auto* taskStub = MakeTask<TaskStubLocalCopy>(src, dst);
    auto* node = MakeNode(taskStub, 0, 0, 0);
    EXPECT_EQ(checker.CheckSingleTaskMem(node), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckSingleTaskMem_LocalReduce_Success)
{
    SingleTaskCheck checker;
    DataSlice src(BufferType::INPUT, 0, 0);
    DataSlice dst(BufferType::OUTPUT, 0, 0);
    auto* taskStub
        = MakeTask<TaskStubLocalReduce>(src, dst, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
    auto* node = MakeNode(taskStub, 0, 0, 0);
    EXPECT_EQ(checker.CheckSingleTaskMem(node), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckSingleTaskMem_Read_Success)
{
    SingleTaskCheck checker;
    DataSlice local(BufferType::INPUT, 0, 0);
    DataSlice remote(BufferType::INPUT, 0, 0);
    LinkInfo link(LinkProtoStub::SDMA);
    auto* taskStub = MakeTask<TaskStubRead>(1, link, local, remote);
    auto* node = MakeNode(taskStub, 0, 0, 0);
    EXPECT_EQ(checker.CheckSingleTaskMem(node), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckSingleTaskMem_Write_Success)
{
    SingleTaskCheck checker;
    DataSlice local(BufferType::OUTPUT, 0, 0);
    DataSlice remote(BufferType::OUTPUT, 0, 0);
    LinkInfo link(LinkProtoStub::SDMA);
    auto* taskStub = MakeTask<TaskStubWrite>(1, link, local, remote);
    auto* node = MakeNode(taskStub, 0, 0, 0);
    EXPECT_EQ(checker.CheckSingleTaskMem(node), HcclResult::HCCL_SUCCESS);
}

// ==================== CheckTaskMem (public BFS) Tests ====================

TEST_F(SingleTaskCheckTest, CheckTaskMem_EmptyGraph_ReturnsSuccess)
{
    SingleTaskCheck checker;
    auto* dummyStart = MakeNode(nullptr, 0, 0, 0);
    EXPECT_EQ(checker.CheckTaskMem(dummyStart), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckTaskMem_SingleNodeGraph_Success)
{
    SingleTaskCheck checker;
    DataSlice src(BufferType::INPUT, 0, 0);
    DataSlice dst(BufferType::OUTPUT, 0, 0);
    auto* taskStub = MakeTask<TaskStubLocalCopy>(src, dst);
    auto* node = MakeNode(taskStub, 0, 0, 0);
    auto* dummyStart = MakeNode(nullptr, 0, 0, 0);
    dummyStart->children = {node};

    EXPECT_EQ(checker.CheckTaskMem(dummyStart), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckTaskMem_LinearChain_Success)
{
    SingleTaskCheck checker;
    DataSlice src(BufferType::INPUT, 0, 0);
    DataSlice dst(BufferType::OUTPUT, 0, 0);

    auto* task1 = MakeTask<TaskStubLocalCopy>(src, dst);
    auto* task2 = MakeTask<TaskStubLocalCopy>(dst, src);
    auto* node1 = MakeNode(task1, 0, 0, 0);
    auto* node2 = MakeNode(task2, 0, 0, 1);
    node1->children = {node2};
    auto* dummyStart = MakeNode(nullptr, 0, 0, 0);
    dummyStart->children = {node1};

    EXPECT_EQ(checker.CheckTaskMem(dummyStart), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckTaskMem_BranchingGraph_Success)
{
    SingleTaskCheck checker;
    DataSlice src(BufferType::INPUT, 0, 0);
    DataSlice dst(BufferType::OUTPUT, 0, 0);

    auto* t1 = MakeTask<TaskStubLocalCopy>(src, dst);
    auto* t2 = MakeTask<TaskStubLocalCopy>(dst, src);
    auto* t3 = MakeTask<TaskStubLocalCopy>(src, dst);
    auto* n1 = MakeNode(t1, 0, 0, 0);
    auto* n2 = MakeNode(t2, 0, 0, 1);
    auto* n3 = MakeNode(t3, 0, 0, 2);
    n1->children = {n2, n3};

    auto* dummyStart = MakeNode(nullptr, 0, 0, 0);
    dummyStart->children = {n1};

    EXPECT_EQ(checker.CheckTaskMem(dummyStart), HcclResult::HCCL_SUCCESS);
}

TEST_F(SingleTaskCheckTest, CheckTaskMem_LocalCopyOutOfBounds_ReturnsError)
{
    SingleTaskCheck checker;
    DataSlice src(BufferType::INPUT, 0, SIZE_MAX);
    DataSlice dst(BufferType::OUTPUT, 0, 64);
    auto* taskStub = MakeTask<TaskStubLocalCopy>(src, dst);
    auto* node = MakeNode(taskStub, 0, 0, 0);
    auto* dummyStart = MakeNode(nullptr, 0, 0, 0);
    dummyStart->children = {node};

    EXPECT_EQ(checker.CheckTaskMem(dummyStart), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(SingleTaskCheckTest, CheckTaskMem_OverlappingLocalCopy_ReturnsError)
{
    SingleTaskCheck checker;
    DataSlice src(BufferType::INPUT, 0, 1024);
    DataSlice dst(BufferType::INPUT, 512, 1024);
    auto* taskStub = MakeTask<TaskStubLocalCopy>(src, dst);
    auto* node = MakeNode(taskStub, 0, 0, 0);
    auto* dummyStart = MakeNode(nullptr, 0, 0, 0);
    dummyStart->children = {node};

    EXPECT_EQ(checker.CheckTaskMem(dummyStart), HcclResult::HCCL_E_INTERNAL);
}
} // namespace HcclSim
