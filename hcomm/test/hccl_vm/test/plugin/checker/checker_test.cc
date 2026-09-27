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

#include "checker.h"
#include "task_ccu.h"
#include "task_check_op_semantics.h"

namespace HcclSim {
class CheckerTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CheckerTest, Constructor_NoThrow) { EXPECT_NO_THROW(Checker checker); }

TEST_F(CheckerTest, Destructor_NoThrow)
{
    Checker* checker = new Checker();
    EXPECT_NO_THROW(delete checker);
}

TEST_F(CheckerTest, CloseRankMemCheck_NoThrow)
{
    Checker checker;
    EXPECT_NO_THROW(checker.CloseRankMemCheck());
}

TEST_F(CheckerTest, GenAndCheckGraph_EmptyQueues)
{
    Checker checker;
    AllRankTaskQueues emptyQueues;
    TaskCheckOpSemantics opSemanticsChecker(
        4, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);

    HcclResult result = checker.GenAndCheckGraph(emptyQueues, opSemanticsChecker);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

TEST_F(CheckerTest, CloseRankMemCheck_Idempotent_NoThrow)
{
    Checker checker;
    EXPECT_NO_THROW(checker.CloseRankMemCheck());
    EXPECT_NO_THROW(checker.CloseRankMemCheck());
}

TEST_F(CheckerTest, GenAndCheckGraph_EmptyQueuesWithCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues emptyQueues;
    TaskCheckOpSemantics opSemanticsChecker(
        4, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(emptyQueues, opSemanticsChecker);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLocalCopyTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto task = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
    singleQueue[0].push_back(task);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLocalReduceTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto task = std::make_shared<TaskStubLocalReduce>(
        srcSlice, dstSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    singleQueue[0].push_back(task);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLocalBatchReduceTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    std::vector<DataSlice> srcSlices;
    srcSlices.emplace_back(BufferType::INPUT, 0, 512);
    srcSlices.emplace_back(BufferType::INPUT, 512, 512);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto task = std::make_shared<TaskStubLocalBatchReduce>(
        srcSlices, dstSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    singleQueue[0].push_back(task);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithReadWriteTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        if (rank == 0) {
            auto writeTask = std::make_shared<TaskStubWrite>(1, link, localSlice, remoteSlice);
            singleQueue[0].push_back(writeTask);
        } else {
            auto readTask = std::make_shared<TaskStubRead>(0, link, localSlice, remoteSlice);
            singleQueue[0].push_back(readTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithPostWaitTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        LinkInfo link(LinkProtoStub::RDMA);
        if (rank == 0) {
            auto postTask = std::make_shared<TaskStubPost>(1, link, 100);
            singleQueue[0].push_back(postTask);
        } else {
            auto waitTask = std::make_shared<TaskStubWait>(0, link, 100);
            singleQueue[0].push_back(waitTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLocalPostWaitTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(2);
    auto postTask = std::make_shared<TaskStubLocalPostTo>(100, 0, 1);
    auto waitTask = std::make_shared<TaskStubLocalWaitFrom>(100, 1, 0);
    singleQueue[0].push_back(postTask);
    singleQueue[1].push_back(waitTask);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLoopStartEndTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    auto loopStart = std::make_shared<TaskStubLoopStart>(0, 0);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto copyTask = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
    auto loopEnd = std::make_shared<TaskStubLoopEnd>(0, 0);
    singleQueue[0].push_back(loopStart);
    singleQueue[0].push_back(copyTask);
    singleQueue[0].push_back(loopEnd);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithBeingReadBeingWrittenTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        if (rank == 0) {
            auto beingWrittenTask = std::make_shared<TaskStubBeingWritten>(1, link, localSlice, remoteSlice);
            singleQueue[0].push_back(beingWrittenTask);
        } else {
            auto beingReadTask = std::make_shared<TaskStubBeingRead>(0, link, localSlice, remoteSlice);
            singleQueue[0].push_back(beingReadTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithReadReduceWriteReduceTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        if (rank == 0) {
            auto writeReduceTask = std::make_shared<TaskStubWriteReduce>(
                1, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
            singleQueue[0].push_back(writeReduceTask);
        } else {
            auto readReduceTask = std::make_shared<TaskStubReadReduce>(
                0, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
            singleQueue[0].push_back(readReduceTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithMultiStreamTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(3);
    for (int i = 0; i < 3; i++) {
        DataSlice srcSlice(BufferType::INPUT, i * 1024, 1024);
        DataSlice dstSlice(BufferType::OUTPUT, i * 1024, 1024);
        auto task = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
        singleQueue[i].push_back(task);
    }
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLocalPostWaitShadowTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(2);
    auto postShadowTask = std::make_shared<TaskStubLocalPostToShadow>(0, 0, 1);
    auto waitShadowTask = std::make_shared<TaskStubLocalWaitFromShadow>(0, 1, 0);
    singleQueue[0].push_back(postShadowTask);
    singleQueue[1].push_back(waitShadowTask);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithBeingReadReduceBeingWrittenReduceTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        if (rank == 0) {
            auto beingWrittenReduceTask = std::make_shared<TaskStubBeingWrittenReduce>(
                1, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
            singleQueue[0].push_back(beingWrittenReduceTask);
        } else {
            auto beingReadReduceTask = std::make_shared<TaskStubBeingReadReduce>(
                0, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
            singleQueue[0].push_back(beingReadReduceTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLocalCopyTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto task = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
    singleQueue[0].push_back(task);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithReadWriteTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        if (rank == 0) {
            auto writeTask = std::make_shared<TaskStubWrite>(1, link, localSlice, remoteSlice);
            singleQueue[0].push_back(writeTask);
        } else {
            auto readTask = std::make_shared<TaskStubRead>(0, link, localSlice, remoteSlice);
            singleQueue[0].push_back(readTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithBeingReadBeingWrittenTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        if (rank == 0) {
            auto beingWrittenTask = std::make_shared<TaskStubBeingWritten>(1, link, localSlice, remoteSlice);
            singleQueue[0].push_back(beingWrittenTask);
        } else {
            auto beingReadTask = std::make_shared<TaskStubBeingRead>(0, link, localSlice, remoteSlice);
            singleQueue[0].push_back(beingReadTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithMultiStreamTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(3);
    for (int i = 0; i < 3; i++) {
        DataSlice srcSlice(BufferType::INPUT, i * 1024, 1024);
        DataSlice dstSlice(BufferType::OUTPUT, i * 1024, 1024);
        auto task = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
        singleQueue[i].push_back(task);
    }
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLocalReduceTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto task = std::make_shared<TaskStubLocalReduce>(
        srcSlice, dstSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    singleQueue[0].push_back(task);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLoopStartEndTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    auto loopStart = std::make_shared<TaskStubLoopStart>(0, 0);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto copyTask = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
    auto loopEnd = std::make_shared<TaskStubLoopEnd>(0, 0);
    singleQueue[0].push_back(loopStart);
    singleQueue[0].push_back(copyTask);
    singleQueue[0].push_back(loopEnd);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLocalPostWaitShadowTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(2);
    auto postShadowTask = std::make_shared<TaskStubLocalPostToShadow>(0, 0, 1);
    auto waitShadowTask = std::make_shared<TaskStubLocalWaitFromShadow>(0, 1, 0);
    singleQueue[0].push_back(postShadowTask);
    singleQueue[1].push_back(waitShadowTask);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithReadReduceWriteReduceTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        if (rank == 0) {
            auto writeReduceTask = std::make_shared<TaskStubWriteReduce>(
                1, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
            singleQueue[0].push_back(writeReduceTask);
        } else {
            auto readReduceTask = std::make_shared<TaskStubReadReduce>(
                0, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
            singleQueue[0].push_back(readReduceTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithBeingReadReduceBeingWrittenReduceTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        if (rank == 0) {
            auto beingWrittenReduceTask = std::make_shared<TaskStubBeingWrittenReduce>(
                1, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
            singleQueue[0].push_back(beingWrittenReduceTask);
        } else {
            auto beingReadReduceTask = std::make_shared<TaskStubBeingReadReduce>(
                0, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
            singleQueue[0].push_back(beingReadReduceTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLocalBatchReduceTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    std::vector<DataSlice> srcSlices;
    srcSlices.emplace_back(BufferType::INPUT, 0, 512);
    srcSlices.emplace_back(BufferType::INPUT, 512, 512);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto task = std::make_shared<TaskStubLocalBatchReduce>(
        srcSlices, dstSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    singleQueue[0].push_back(task);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, Destructor_AfterCloseRankMemCheckWithLocalCopyTasks)
{
    Checker* checker = new Checker();
    checker->CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto task = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
    singleQueue[0].push_back(task);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    checker->GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NO_THROW(delete checker);
}

TEST_F(CheckerTest, Destructor_AfterCloseRankMemCheckWithReadWriteTasks)
{
    Checker* checker = new Checker();
    checker->CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        DataSlice localSlice(BufferType::CCL, 0, 1024);
        DataSlice remoteSlice(BufferType::CCL, 0, 1024);
        LinkInfo link(LinkProtoStub::SDMA);
        if (rank == 0) {
            auto writeTask = std::make_shared<TaskStubWrite>(1, link, localSlice, remoteSlice);
            singleQueue[0].push_back(writeTask);
        } else {
            auto readTask = std::make_shared<TaskStubRead>(0, link, localSlice, remoteSlice);
            singleQueue[0].push_back(readTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    checker->GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NO_THROW(delete checker);
}

TEST_F(CheckerTest, CloseRankMemCheckThenCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto task = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
    singleQueue[0].push_back(task);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result1 = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result1, HcclResult::HCCL_E_PARA);

    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues2;
    SingleTaskQueue singleQueue2;
    singleQueue2.resize(1);
    DataSlice srcSlice2(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 0, 1024);
    auto task2 = std::make_shared<TaskStubLocalCopy>(srcSlice2, dstSlice2);
    singleQueue2[0].push_back(task2);
    taskQueues2[0] = singleQueue2;
    TaskCheckOpSemantics opSemanticsChecker2(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result2 = checker.GenAndCheckGraph(taskQueues2, opSemanticsChecker2);
    EXPECT_NE(result2, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithCcuGraphTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    auto ccuTask = std::make_shared<TaskStubCcuGraph>(0);
    singleQueue[0].push_back(ccuTask);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithCcuGraphAndLocalCopyTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    auto ccuTask = std::make_shared<TaskStubCcuGraph>(0);
    DataSlice srcSlice(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
    auto copyTask = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
    singleQueue[0].push_back(ccuTask);
    singleQueue[0].push_back(copyTask);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithCcuGraphMultiRankCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        auto ccuTask = std::make_shared<TaskStubCcuGraph>(rank);
        singleQueue[0].push_back(ccuTask);
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, Destructor_AfterCcuGraphCloseRankMemCheck)
{
    Checker* checker = new Checker();
    checker->CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    auto ccuTask = std::make_shared<TaskStubCcuGraph>(0);
    singleQueue[0].push_back(ccuTask);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    checker->GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NO_THROW(delete checker);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithGraphSeparateTasks)
{
    Checker checker;
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    DataSlice srcSlice1(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice1(BufferType::OUTPUT, 0, 1024);
    auto copyTask1 = std::make_shared<TaskStubLocalCopy>(srcSlice1, dstSlice1);
    auto separateTask = std::make_shared<TaskStubGraphSeparate>();
    DataSlice srcSlice2(BufferType::INPUT, 1024, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 1024, 1024);
    auto copyTask2 = std::make_shared<TaskStubLocalCopy>(srcSlice2, dstSlice2);
    singleQueue[0].push_back(copyTask1);
    singleQueue[0].push_back(separateTask);
    singleQueue[0].push_back(copyTask2);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 2048);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithGraphSeparateTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(1);
    DataSlice srcSlice1(BufferType::INPUT, 0, 1024);
    DataSlice dstSlice1(BufferType::OUTPUT, 0, 1024);
    auto copyTask1 = std::make_shared<TaskStubLocalCopy>(srcSlice1, dstSlice1);
    auto separateTask = std::make_shared<TaskStubGraphSeparate>();
    DataSlice srcSlice2(BufferType::INPUT, 1024, 1024);
    DataSlice dstSlice2(BufferType::OUTPUT, 1024, 1024);
    auto copyTask2 = std::make_shared<TaskStubLocalCopy>(srcSlice2, dstSlice2);
    singleQueue[0].push_back(copyTask1);
    singleQueue[0].push_back(separateTask);
    singleQueue[0].push_back(copyTask2);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 2048);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_MultipleRanksWithCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 4; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        DataSlice srcSlice(BufferType::INPUT, 0, 1024);
        DataSlice dstSlice(BufferType::OUTPUT, 0, 1024);
        auto task = std::make_shared<TaskStubLocalCopy>(srcSlice, dstSlice);
        singleQueue[0].push_back(task);
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        4, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_NE(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithPostWaitTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    for (u32 rank = 0; rank < 2; rank++) {
        SingleTaskQueue singleQueue;
        singleQueue.resize(1);
        LinkInfo link(LinkProtoStub::RDMA);
        if (rank == 0) {
            auto postTask = std::make_shared<TaskStubPost>(1, link, 100);
            singleQueue[0].push_back(postTask);
        } else {
            auto waitTask = std::make_shared<TaskStubWait>(0, link, 100);
            singleQueue[0].push_back(waitTask);
        }
        taskQueues[rank] = singleQueue;
    }
    TaskCheckOpSemantics opSemanticsChecker(
        2, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

TEST_F(CheckerTest, GenAndCheckGraph_WithLocalPostWaitTasksCloseRankMemCheck)
{
    Checker checker;
    checker.CloseRankMemCheck();
    AllRankTaskQueues taskQueues;
    SingleTaskQueue singleQueue;
    singleQueue.resize(2);
    auto postTask = std::make_shared<TaskStubLocalPostTo>(100, 0, 1);
    auto waitTask = std::make_shared<TaskStubLocalWaitFrom>(100, 1, 0);
    singleQueue[0].push_back(postTask);
    singleQueue[1].push_back(waitTask);
    taskQueues[0] = singleQueue;
    TaskCheckOpSemantics opSemanticsChecker(
        1, HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, 1024);
    HcclResult result = checker.GenAndCheckGraph(taskQueues, opSemanticsChecker);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}
} // namespace HcclSim
