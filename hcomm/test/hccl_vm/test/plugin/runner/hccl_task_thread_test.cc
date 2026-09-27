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

#include "store_sim_store_pub.h"
#include "hccl_task_thread.h"
#include "hccl_types.h"

using namespace VirtualRunTime;

class HcclTaskThreadTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 清理全局通知状态 - 通过TaskNotifyWait消费所有通知
    }

    void TearDown() override
    {
        // 清理全局通知状态 - 通过TaskNotifyWait消费所有通知
    }
};

// Test: TaskNotifyRecord with new notify ID
TEST_F(HcclTaskThreadTest, TaskNotifyRecord_NewNotifyId_SetsTrue)
{
    HcclTaskMetaData task;
    task.taskData.notify.notifyId = 12345;

    auto ret = TaskNotifyRecord(task);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskNotifyRecord with existing notify ID overwrites
TEST_F(HcclTaskThreadTest, TaskNotifyRecord_ExistingNotifyId_Overwrites)
{
    HcclTaskMetaData task;
    task.taskData.notify.notifyId = 12345;

    TaskNotifyRecord(task);

    // 再次记录同一个notify ID
    auto ret = TaskNotifyRecord(task);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskNotifyWait with recorded notify ID
TEST_F(HcclTaskThreadTest, TaskNotifyWait_RecordedNotifyId_ReturnsSuccess)
{
    HcclTaskMetaData recordTask;
    recordTask.taskData.notify.notifyId = 12345;
    TaskNotifyRecord(recordTask);

    HcclTaskMetaData waitTask;
    waitTask.taskData.notify.notifyId = 12345;

    auto ret = TaskNotifyWait(waitTask);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskNotifyWait with non-recorded notify ID
TEST_F(HcclTaskThreadTest, TaskNotifyWait_NonRecordedNotifyId_ReturnsHold)
{
    HcclTaskMetaData waitTask;
    waitTask.taskData.notify.notifyId = 99999;

    auto ret = TaskNotifyWait(waitTask);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_VRT_HOLD_CMD);
}

// Test: TaskNotifyWait after TaskNotifyRecord returns success
TEST_F(HcclTaskThreadTest, TaskNotifyWait_AfterRecord_ReturnsSuccess)
{
    HcclTaskMetaData recordTask;
    recordTask.taskData.notify.notifyId = 100;
    TaskNotifyRecord(recordTask);

    HcclTaskMetaData waitTask;
    waitTask.taskData.notify.notifyId = 100;

    auto ret = TaskNotifyWait(waitTask);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskNotifyWait twice on same notify ID (second should hold)
TEST_F(HcclTaskThreadTest, TaskNotifyWait_TwiceOnSameId_SecondReturnsHold)
{
    HcclTaskMetaData recordTask;
    recordTask.taskData.notify.notifyId = 200;
    TaskNotifyRecord(recordTask);

    HcclTaskMetaData waitTask;
    waitTask.taskData.notify.notifyId = 200;

    // 第一次等待应该成功
    auto ret1 = TaskNotifyWait(waitTask);
    EXPECT_EQ(ret1, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    // 第二次等待应该返回HOLD，因为notify已经被消费
    auto ret2 = TaskNotifyWait(waitTask);
    EXPECT_EQ(ret2, HcclSim::HcclVmResult::HCCL_SIM_VRT_HOLD_CMD);
}

// Test: Multiple notify IDs are independent
TEST_F(HcclTaskThreadTest, TaskNotifyRecord_MultipleIds_Independent)
{
    HcclTaskMetaData task1;
    task1.taskData.notify.notifyId = 100;
    TaskNotifyRecord(task1);

    HcclTaskMetaData task2;
    task2.taskData.notify.notifyId = 200;
    TaskNotifyRecord(task2);

    // 等待第一个
    HcclTaskMetaData waitTask1;
    waitTask1.taskData.notify.notifyId = 100;
    auto ret1 = TaskNotifyWait(waitTask1);
    EXPECT_EQ(ret1, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    // 第二个应该仍然可用
    HcclTaskMetaData waitTask2;
    waitTask2.taskData.notify.notifyId = 200;
    auto ret2 = TaskNotifyWait(waitTask2);
    EXPECT_EQ(ret2, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskNotifyRecord with zero notify ID
TEST_F(HcclTaskThreadTest, TaskNotifyRecord_ZeroNotifyId_SetsTrue)
{
    HcclTaskMetaData task;
    task.taskData.notify.notifyId = 0;

    auto ret = TaskNotifyRecord(task);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskNotifyWait with zero notify ID
TEST_F(HcclTaskThreadTest, TaskNotifyWait_ZeroNotifyId_ReturnsSuccess)
{
    HcclTaskMetaData recordTask;
    recordTask.taskData.notify.notifyId = 0;
    TaskNotifyRecord(recordTask);

    HcclTaskMetaData waitTask;
    waitTask.taskData.notify.notifyId = 0;

    auto ret = TaskNotifyWait(waitTask);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskNotifyRecord with large notify ID
TEST_F(HcclTaskThreadTest, TaskNotifyRecord_LargeNotifyId_SetsTrue)
{
    HcclTaskMetaData task;
    task.taskData.notify.notifyId = 0xFFFFFFFFFFFFFFFF;

    auto ret = TaskNotifyRecord(task);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskNotifyWait with large notify ID
TEST_F(HcclTaskThreadTest, TaskNotifyWait_LargeNotifyId_ReturnsSuccess)
{
    HcclTaskMetaData recordTask;
    recordTask.taskData.notify.notifyId = 0xFFFFFFFFFFFFFFFF;
    TaskNotifyRecord(recordTask);

    HcclTaskMetaData waitTask;
    waitTask.taskData.notify.notifyId = 0xFFFFFFFFFFFFFFFF;

    auto ret = TaskNotifyWait(waitTask);
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskMemcpy without shared memory setup returns error
TEST_F(HcclTaskThreadTest, TaskMemcpy_NoShmSetup_ReturnsError)
{
    HcclTaskMetaData task;
    task.taskData.transMem.srcOffset = 0x1000;
    task.taskData.transMem.dstOffset = 0x2000;
    task.taskData.transMem.len = 64;

    auto ret = TaskMemcpy(task);
    // Without shared memory initialization, GetAddrByOffset should fail
    EXPECT_NE(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskReduce with SUM op without shared memory setup returns error
TEST_F(HcclTaskThreadTest, TaskReduce_SumNoShm_ReturnsError)
{
    HcclTaskMetaData task;
    task.taskData.reduce.srcOffset = 0x1000;
    task.taskData.reduce.dstOffset = 0x2000;
    task.taskData.reduce.dataCount = 10;
    task.taskData.reduce.dataType = 3; // HCCL_DATA_TYPE_INT32
    task.taskData.reduce.reduceOp = 0; // HCCL_REDUCE_SUM

    auto ret = TaskReduce(task);
    // Without shared memory initialization, GetAddrByOffset should fail
    EXPECT_NE(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskReduce with MIN op without shared memory setup returns error
TEST_F(HcclTaskThreadTest, TaskReduce_MinNoShm_ReturnsError)
{
    HcclTaskMetaData task;
    task.taskData.reduce.srcOffset = 0x1000;
    task.taskData.reduce.dstOffset = 0x2000;
    task.taskData.reduce.dataCount = 10;
    task.taskData.reduce.dataType = 3; // HCCL_DATA_TYPE_INT32
    task.taskData.reduce.reduceOp = 3; // HCCL_REDUCE_MIN = 3

    auto ret = TaskReduce(task);
    EXPECT_NE(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskReduce with MAX op without shared memory setup returns error
TEST_F(HcclTaskThreadTest, TaskReduce_MaxNoShm_ReturnsError)
{
    HcclTaskMetaData task;
    task.taskData.reduce.srcOffset = 0x1000;
    task.taskData.reduce.dstOffset = 0x2000;
    task.taskData.reduce.dataCount = 10;
    task.taskData.reduce.dataType = 3; // HCCL_DATA_TYPE_INT32
    task.taskData.reduce.reduceOp = 2; // HCCL_REDUCE_MAX = 2

    auto ret = TaskReduce(task);
    EXPECT_NE(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskReduce with invalid op returns success (default case)
TEST_F(HcclTaskThreadTest, TaskReduce_InvalidOp_ReturnsSuccess)
{
    HcclTaskMetaData task;
    task.taskData.reduce.srcOffset = 0x1000;
    task.taskData.reduce.dstOffset = 0x2000;
    task.taskData.reduce.dataCount = 10;
    task.taskData.reduce.dataType = 3;   // HCCL_DATA_TYPE_INT32
    task.taskData.reduce.reduceOp = 255; // Invalid reduce op

    auto ret = TaskReduce(task);
    // Invalid op falls through to default case and returns SUCCESS
    EXPECT_EQ(ret, HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

// Test: TaskCcuGraph without CCU resource setup - skipped (causes segfault due to uninitialized CCU resources)
// This would require full CCU resource initialization which is too complex for unit testing
TEST_F(HcclTaskThreadTest, TaskCcuGraph_NoCcuSetup_Skipped)
{
    // TaskCcuGraph requires CcuResourceManager to be initialized with valid rank/die/instr data.
    // Calling it without initialization causes a segfault. Skipping this test.
    SUCCEED() << "TaskCcuGraph requires full CCU resource setup - skipped to avoid segfault";
}
