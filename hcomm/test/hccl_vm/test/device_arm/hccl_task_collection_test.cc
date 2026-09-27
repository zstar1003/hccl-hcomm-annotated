/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <gtest/gtest.h>

#include "hccl_task_collection.h"

class HcclTaskCollectionTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(HcclTaskCollectionTest, InsertTask_WithNullptr_ShouldReturnEarly)
{
    EXPECT_NO_THROW(InsertTaskToCollectionDev(nullptr));
}

TEST_F(HcclTaskCollectionTest, InsertTask_WithValidPointer_ShouldNotThrow)
{
    HcclTaskMetaData task;
    memset(&task, 0, sizeof(task));
    EXPECT_NO_THROW(InsertTaskToCollectionDev(&task));
}

TEST_F(HcclTaskCollectionTest, InsertTask_MemCpyTask)
{
    HcclTaskMetaData task;
    memset(&task, 0, sizeof(task));
    task.taskType = HccLTaskMetaType::MEM_CPY;
    task.rankId = 0;
    task.streamId = 0;
    task.taskData.transMem.srcRankId = 0;
    task.taskData.transMem.dstRankId = 1;
    task.taskData.transMem.srcOffset = 0x1000;
    task.taskData.transMem.dstOffset = 0x2000;
    task.taskData.transMem.len = 1024;

    EXPECT_NO_THROW(InsertTaskToCollectionDev(&task));
}

TEST_F(HcclTaskCollectionTest, InsertTask_ReduceTask)
{
    HcclTaskMetaData task;
    memset(&task, 0, sizeof(task));
    task.taskType = HccLTaskMetaType::REDUCE;
    task.rankId = 0;
    task.streamId = 0;
    task.taskData.reduce.srcRankId = 0;
    task.taskData.reduce.dstRankId = 1;
    task.taskData.reduce.srcOffset = 0x1000;
    task.taskData.reduce.dstOffset = 0x2000;
    task.taskData.reduce.dataCount = 256;
    task.taskData.reduce.dataType = 0;
    task.taskData.reduce.reduceOp = 1;

    EXPECT_NO_THROW(InsertTaskToCollectionDev(&task));
}

TEST_F(HcclTaskCollectionTest, InsertTask_NotifyWaitTask)
{
    HcclTaskMetaData task;
    memset(&task, 0, sizeof(task));
    task.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    task.rankId = 0;
    task.streamId = 0;
    task.taskData.notify.srcRankId = 1;
    task.taskData.notify.notifyId = 100;

    EXPECT_NO_THROW(InsertTaskToCollectionDev(&task));
}

TEST_F(HcclTaskCollectionTest, InsertTask_NotifyRecordTask)
{
    HcclTaskMetaData task;
    memset(&task, 0, sizeof(task));
    task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    task.rankId = 0;
    task.streamId = 0;
    task.taskData.notify.dstRankId = 1;
    task.taskData.notify.notifyId = 100;

    EXPECT_NO_THROW(InsertTaskToCollectionDev(&task));
}

TEST_F(HcclTaskCollectionTest, InsertTask_MultipleTasks)
{
    for (int i = 0; i < 10; i++) {
        HcclTaskMetaData task;
        memset(&task, 0, sizeof(task));
        task.taskType = HccLTaskMetaType::MEM_CPY;
        task.rankId = i;
        task.streamId = i;
        EXPECT_NO_THROW(InsertTaskToCollectionDev(&task));
    }
}

TEST_F(HcclTaskCollectionTest, InsertTask_MaxValues)
{
    HcclTaskMetaData task;
    memset(&task, 0, sizeof(task));
    task.taskType = HccLTaskMetaType::MEM_CPY;
    task.rankId = UINT32_MAX;
    task.streamId = UINT64_MAX;
    task.taskData.transMem.srcOffset = UINT64_MAX;
    task.taskData.transMem.dstOffset = UINT64_MAX;
    task.taskData.transMem.len = UINT64_MAX;

    EXPECT_NO_THROW(InsertTaskToCollectionDev(&task));
}

TEST_F(HcclTaskCollectionTest, InsertTask_ZeroValues)
{
    HcclTaskMetaData task;
    memset(&task, 0, sizeof(task));
    task.taskType = HccLTaskMetaType::MEM_CPY;

    EXPECT_NO_THROW(InsertTaskToCollectionDev(&task));
}

TEST_F(HcclTaskCollectionTest, HcclTaskMetaData_StructSize) { EXPECT_GT(sizeof(HcclTaskMetaData), 0); }
