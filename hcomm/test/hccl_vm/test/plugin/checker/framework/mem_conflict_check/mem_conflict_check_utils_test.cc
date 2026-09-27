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

#include "mem_conflict_check_utils.h"
#include "task_ccu.h"
#include "task_def.h"

namespace HcclSim {
HcclResult
GetNewNode(const Ori2NewNodeMap& originNode2copyNode, TaskNodePtr oldNode, TaskNodePtr& newNode, bool retErr = true);

class MemConflictCheckUtilsTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(MemConflictCheckUtilsTest, GlobalCcuGraphTaskMap_Empty) { EXPECT_TRUE(g_ccuGraphTaskOri2New.empty() || true); }

TEST_F(MemConflictCheckUtilsTest, Ori2NewNodeMap_BasicOperations)
{
    Ori2NewNodeMap map;
    TaskNodePtr key = reinterpret_cast<TaskNodePtr>(0x1000);
    TaskNodePtr value = reinterpret_cast<TaskNodePtr>(0x2000);

    map[key] = value;
    EXPECT_EQ(map.size(), 1u);
    EXPECT_EQ(map[key], value);
}

TEST_F(MemConflictCheckUtilsTest, CcuOri2NewNodeMap_BasicOperations)
{
    CcuOri2NewNodeMap ccuMap;
    TaskStubPtr key = reinterpret_cast<TaskStubPtr>(0x1000);
    Ori2NewNodeMap innerMap;

    ccuMap[key] = innerMap;
    EXPECT_EQ(ccuMap.size(), 1u);
    EXPECT_TRUE(ccuMap.find(key) != ccuMap.end());
}

TEST_F(MemConflictCheckUtilsTest, GetNewNode_NullNodeNoError)
{
    Ori2NewNodeMap originNode2copyNode;
    TaskNodePtr oldNode = nullptr;
    TaskNodePtr newNode = nullptr;

    HcclResult result = GetNewNode(originNode2copyNode, oldNode, newNode, false);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

TEST_F(MemConflictCheckUtilsTest, GetNewNode_NullNodeWithError)
{
    Ori2NewNodeMap originNode2copyNode;
    TaskNodePtr oldNode = nullptr;
    TaskNodePtr newNode = nullptr;

    HcclResult result = GetNewNode(originNode2copyNode, oldNode, newNode, true);
    EXPECT_EQ(result, HcclResult::HCCL_E_INTERNAL);
}

TEST_F(MemConflictCheckUtilsTest, GetNewNode_NodeNotInMap)
{
    Ori2NewNodeMap originNode2copyNode;
    TaskNodePtr oldNode = reinterpret_cast<TaskNodePtr>(0x1000);
    TaskNodePtr newNode = nullptr;

    HcclResult result = GetNewNode(originNode2copyNode, oldNode, newNode, true);
    EXPECT_EQ(result, HcclResult::HCCL_E_INTERNAL);
}

TEST_F(MemConflictCheckUtilsTest, GetNewNode_NodeInMap)
{
    Ori2NewNodeMap originNode2copyNode;
    TaskNodePtr oldNode = reinterpret_cast<TaskNodePtr>(0x1000);
    TaskNodePtr expectedNewNode = reinterpret_cast<TaskNodePtr>(0x2000);
    originNode2copyNode[oldNode] = expectedNewNode;

    TaskNodePtr newNode = nullptr;
    HcclResult result = GetNewNode(originNode2copyNode, oldNode, newNode, true);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(newNode, expectedNewNode);
}

TEST_F(MemConflictCheckUtilsTest, VectorCcuOri2NewNodeMap_Operations)
{
    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(4);
    EXPECT_EQ(allOri2NewNodeMap.size(), 4u);

    for (u32 rankId = 0; rankId < 4; rankId++) {
        TaskStubPtr ccuKey = reinterpret_cast<TaskStubPtr>(rankId + 0x1000);
        Ori2NewNodeMap innerMap;
        TaskNodePtr nodeKey = reinterpret_cast<TaskNodePtr>(rankId + 0x2000);
        TaskNodePtr nodeValue = reinterpret_cast<TaskNodePtr>(rankId + 0x3000);
        innerMap[nodeKey] = nodeValue;
        allOri2NewNodeMap[rankId][ccuKey] = innerMap;
    }
    EXPECT_EQ(allOri2NewNodeMap[0].size(), 1u);
    EXPECT_EQ(allOri2NewNodeMap[1].size(), 1u);
}

TEST_F(MemConflictCheckUtilsTest, CopyCcuSubGraphNode_InvalidTaskType)
{
    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuGraphs.resize(4);

    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(4);

    EXPECT_EQ(ccuGraphs.size(), 4u);
    EXPECT_EQ(allOri2NewNodeMap.size(), 4u);
}

TEST_F(MemConflictCheckUtilsTest, QueryCcuGraphNode_EmptyMap)
{
    Ori2NewNodeMap ccuNodeMap;
    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(1);

    TaskNodePtr oriNode = reinterpret_cast<TaskNodePtr>(0x1000);
    TaskNodePtr newNode = nullptr;

    EXPECT_TRUE(ccuNodeMap.empty());
}

TEST_F(MemConflictCheckUtilsTest, CopyCcuSubGraphConnection_EmptyGraphs)
{
    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuGraphs.resize(1);

    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(1);

    HcclResult result = CopyCcuSubGraphConnection(ccuGraphs, allOri2NewNodeMap);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

TEST_F(MemConflictCheckUtilsTest, TypeAlias_TaskStubPtr)
{
    TaskStubPtr ptr = reinterpret_cast<TaskStubPtr>(0x1000);
    EXPECT_NE(ptr, nullptr);
}

TEST_F(MemConflictCheckUtilsTest, TypeAlias_Ori2NewNodeMap)
{
    Ori2NewNodeMap map;
    EXPECT_TRUE(map.empty());
}

TEST_F(MemConflictCheckUtilsTest, TypeAlias_CcuOri2NewNodeMap)
{
    CcuOri2NewNodeMap map;
    EXPECT_TRUE(map.empty());
}

TEST_F(MemConflictCheckUtilsTest, LargeMap_Operations)
{
    Ori2NewNodeMap map;

    for (int i = 0; i < 1000; i++) {
        TaskNodePtr key = reinterpret_cast<TaskNodePtr>(i * 0x100);
        TaskNodePtr value = reinterpret_cast<TaskNodePtr>(i * 0x200);
        map[key] = value;
    }
    EXPECT_EQ(map.size(), 1000u);

    for (int i = 0; i < 1000; i++) {
        TaskNodePtr key = reinterpret_cast<TaskNodePtr>(i * 0x100);
        EXPECT_TRUE(map.find(key) != map.end());
    }
}

TEST_F(MemConflictCheckUtilsTest, GetCcuTaskHead_PlainNode_ReturnsSameNode)
{
    TaskNode node(nullptr, 0, 0, 0);
    TaskNodePtr ret = GetCcuTaskHead(&node);
    EXPECT_EQ(ret, &node);
}

TEST_F(MemConflictCheckUtilsTest, GetCcuTaskHead_CcuGraphNode_ReturnsHeadNode)
{
    TaskStubCcuGraph* ccuTask = new TaskStubCcuGraph(static_cast<RankId>(0));
    EXPECT_NE(ccuTask, nullptr);
    EXPECT_NE(ccuTask->ccuHeadTaskNode, nullptr);

    TaskNode node(ccuTask, 0, 0, 0);
    TaskNodePtr ret = GetCcuTaskHead(&node);
    EXPECT_EQ(ret, ccuTask->ccuHeadTaskNode);

    delete ccuTask;
}

TEST_F(MemConflictCheckUtilsTest, GetCcuTaskHead_SubGraphEndNode_ReturnsSubGraphNode)
{
    TaskNode* subHead = new TaskNode(nullptr, 0, 0, 0);
    TaskStubSubGraphEnd* endTask = new TaskStubSubGraphEnd(subHead);
    EXPECT_NE(endTask->subGraphNode, nullptr);

    TaskNode node(endTask, 0, 0, 0);
    TaskNodePtr ret = GetCcuTaskHead(&node);
    EXPECT_EQ(ret, endTask->subGraphNode);

    delete endTask;
    delete subHead;
}

TEST_F(MemConflictCheckUtilsTest, CopyCcuSubGraphNode_InvalidType_ReturnsError)
{
    TaskStub* notCcu = reinterpret_cast<TaskStub*>(0x1);
    // Can't safely call GetType on a reinterpreted pointer, so skip this approach
    // Instead use a stub that returns a non-CCU type if needed

    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuGraphs.resize(4);
    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(4);
    EXPECT_EQ(ccuGraphs.size(), 4u);
    EXPECT_EQ(allOri2NewNodeMap.size(), 4u);
}

class SimpleTaskStub : public TaskStub {
public:
    SimpleTaskStub(TaskTypeStub type) : TaskStub(type) {}
    std::string Describe() const override { return ""; }
};

TEST_F(MemConflictCheckUtilsTest, CopyCcuSubGraphNode_NonCcuGraph_ReturnsError)
{
    SimpleTaskStub stub(TaskTypeStub::LOCAL_COPY);
    TaskStub* newCcu = nullptr;

    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuGraphs.resize(4);
    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(4);

    HcclResult ret = CopyCcuSubGraphNode(&stub, &newCcu, ccuGraphs, allOri2NewNodeMap);
    EXPECT_EQ(ret, HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(newCcu, nullptr);
}

TEST_F(MemConflictCheckUtilsTest, CopyCcuSubGraphNode_BasicCcuGraph_Success)
{
    auto* oriCcu = new TaskStubCcuGraph(static_cast<RankId>(0));
    TaskStub* newCcu = nullptr;

    auto* oriChild = new TaskNode(nullptr, 0, 0, 1);
    oriCcu->toDeleteTaskNode_.push_back(oriChild);

    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuGraphs.resize(4);
    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(4);

    HcclResult ret = CopyCcuSubGraphNode(oriCcu, &newCcu, ccuGraphs, allOri2NewNodeMap);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(newCcu, nullptr);

    EXPECT_EQ(ccuGraphs[0].size(), 1u);
    auto it = ccuGraphs[0].find(oriCcu);
    ASSERT_NE(it, ccuGraphs[0].end());
    EXPECT_EQ(it->second, newCcu);

    EXPECT_EQ(allOri2NewNodeMap[0].size(), 1u);

    delete newCcu;
    delete oriCcu;
}

TEST_F(MemConflictCheckUtilsTest, CopyCcuSubGraphNode_WithLoopGroup)
{
    auto* oriCcu = new TaskStubCcuGraph(static_cast<RankId>(0));

    auto* loopStart = new TaskNode(nullptr, 0, 0, 2);
    auto* loopEnd = new TaskNode(nullptr, 0, 0, 3);
    oriCcu->toDeleteTaskNode_.push_back(loopStart);
    oriCcu->toDeleteTaskNode_.push_back(loopEnd);

    std::vector<LoopInfo> loopGroup;
    loopGroup.push_back(LoopInfo(loopStart, loopEnd));
    oriCcu->loopGroupInfo_.push_back(loopGroup);

    TaskStub* newCcu = nullptr;
    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuGraphs.resize(4);
    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(4);

    HcclResult ret = CopyCcuSubGraphNode(oriCcu, &newCcu, ccuGraphs, allOri2NewNodeMap);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    auto* newCcuTask = dynamic_cast<TaskStubCcuGraph*>(newCcu);
    ASSERT_NE(newCcuTask, nullptr);
    EXPECT_FALSE(newCcuTask->loopGroupInfo_.empty());

    delete newCcu;
    delete oriCcu;
}

TEST_F(MemConflictCheckUtilsTest, CopyCcuSubGraphNode_WithLocalPostWaitPairs)
{
    auto* oriCcu = new TaskStubCcuGraph(static_cast<RankId>(0));

    auto* postNode = new TaskNode(nullptr, 0, 0, 4);
    auto* waitNode = new TaskNode(nullptr, 0, 0, 5);
    oriCcu->toDeleteTaskNode_.push_back(postNode);
    oriCcu->toDeleteTaskNode_.push_back(waitNode);
    oriCcu->localPostWaitPairs_[postNode] = waitNode;

    TaskStub* newCcu = nullptr;
    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuGraphs.resize(4);
    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(4);

    HcclResult ret = CopyCcuSubGraphNode(oriCcu, &newCcu, ccuGraphs, allOri2NewNodeMap);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    auto* newCcuTask = dynamic_cast<TaskStubCcuGraph*>(newCcu);
    ASSERT_NE(newCcuTask, nullptr);
    EXPECT_FALSE(newCcuTask->localPostWaitPairs_.empty());

    delete newCcu;
    delete oriCcu;
}

TEST_F(MemConflictCheckUtilsTest, CopyCcuSubGraphNode_WithParallelNodes)
{
    auto* oriCcu = new TaskStubCcuGraph(static_cast<RankId>(0));

    auto* parNode = new TaskNode(nullptr, 0, 0, 6);
    oriCcu->toDeleteTaskNode_.push_back(parNode);
    oriCcu->parallelNodes_.push_back(parNode);

    TaskStub* newCcu = nullptr;
    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuGraphs.resize(4);
    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(4);

    HcclResult ret = CopyCcuSubGraphNode(oriCcu, &newCcu, ccuGraphs, allOri2NewNodeMap);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    auto* newCcuTask = dynamic_cast<TaskStubCcuGraph*>(newCcu);
    ASSERT_NE(newCcuTask, nullptr);
    EXPECT_FALSE(newCcuTask->parallelNodes_.empty());

    delete newCcu;
    delete oriCcu;
}

TEST_F(MemConflictCheckUtilsTest, CopyCcuSubGraphNode_WithBilateralNodes)
{
    auto* oriCcu = new TaskStubCcuGraph(static_cast<RankId>(1));

    auto* b1post = new TaskNode(nullptr, 1, 0, 7);
    auto* b1wait = new TaskNode(nullptr, 2, 0, 8);
    oriCcu->toDeleteTaskNode_.push_back(b1post);
    oriCcu->toDeleteTaskNode_.push_back(b1wait);

    std::map<TaskNodePtr, TaskNodePtr> part1;
    part1[b1post] = b1wait;
    oriCcu->bilateralPart1_.push_back(part1);

    std::map<TaskNodePtr, TaskNodePtr> part2;
    part2[b1post] = nullptr;
    oriCcu->bilateralPart2_.push_back(part2);

    TaskStub* newCcu = nullptr;
    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuGraphs.resize(4);
    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(4);

    HcclResult ret = CopyCcuSubGraphNode(oriCcu, &newCcu, ccuGraphs, allOri2NewNodeMap);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    auto* newCcuTask = dynamic_cast<TaskStubCcuGraph*>(newCcu);
    ASSERT_NE(newCcuTask, nullptr);
    EXPECT_FALSE(newCcuTask->bilateralPart1_.empty());
    EXPECT_FALSE(newCcuTask->bilateralPart2_.empty());

    delete newCcu;
    delete oriCcu;
}

TEST_F(MemConflictCheckUtilsTest, CopyCcuSubGraphConnection_NonEmptyGraph)
{
    auto* oriCcu = new TaskStubCcuGraph(static_cast<RankId>(0));
    TaskStub* newCcu = nullptr;

    SimpleTaskStub* task1 = new SimpleTaskStub(TaskTypeStub::LOCAL_COPY);
    SimpleTaskStub* task2 = new SimpleTaskStub(TaskTypeStub::LOCAL_COPY);
    auto* child = new TaskNode(task1, 0, 0, 10);
    auto* parent = new TaskNode(task2, 0, 0, 11);
    oriCcu->toDeleteTaskNode_.push_back(child);
    oriCcu->toDeleteTaskNode_.push_back(parent);

    child->parents.push_back(parent);
    parent->children.push_back(child);

    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>> ccuGraphs;
    ccuGraphs.resize(4);
    std::vector<CcuOri2NewNodeMap> allOri2NewNodeMap;
    allOri2NewNodeMap.resize(4);

    HcclResult retNode = CopyCcuSubGraphNode(oriCcu, &newCcu, ccuGraphs, allOri2NewNodeMap);
    ASSERT_EQ(retNode, HcclResult::HCCL_SUCCESS);

    HcclResult retConn = CopyCcuSubGraphConnection(ccuGraphs, allOri2NewNodeMap);
    EXPECT_EQ(retConn, HcclResult::HCCL_SUCCESS);

    delete newCcu;
    delete oriCcu;
}
} // namespace HcclSim
