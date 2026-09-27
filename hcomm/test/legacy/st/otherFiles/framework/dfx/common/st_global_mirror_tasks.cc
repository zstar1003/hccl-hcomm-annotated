/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/MockObject.h>
#define private public
#include "global_mirror_tasks.h"
#include "invalid_params_exception.h"
#include "internal_exception.h"
#undef private

using namespace Hccl;

class GlobalMirrorTasksTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GlobalMirrorTasks tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "GlobalMirrorTasks tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in GlobalMirrorTasks SetUP" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "A Test case in GlobalMirrorTasks TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_GetQueue_1)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    EXPECT_THROW(globalMirrorTasks.GetQueue(60, 60), InternalException);
}

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_GetQueue_2)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    EXPECT_THROW(globalMirrorTasks.GetQueue(30, 60), InternalException);
}

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_CreateQueue_1)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    EXPECT_THROW(globalMirrorTasks.CreateQueue(60, 60, QueueType::Circular_Queue), InternalException);
}

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_CreateQueue_2)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    globalMirrorTasks.CreateQueue(30, 0, QueueType::Circular_Queue);

    globalMirrorTasks.GetQueue(30, 0);

    globalMirrorTasks.CreateQueue(30, 1, QueueType::Vector_Queue);

    EXPECT_NO_THROW(globalMirrorTasks.CreateQueue(30, 0, QueueType::Vector_Queue));
}

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_DestroyQueue_1)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    EXPECT_THROW(globalMirrorTasks.DestroyQueue(60, 0), InternalException);
}

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_DestroyQueue_2)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    globalMirrorTasks.DestroyQueue(30, 0);
}

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_Begin)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    globalMirrorTasks.Begin(30);

    EXPECT_THROW(globalMirrorTasks.Begin(60), InternalException);
}

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_End)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    globalMirrorTasks.End(30);

    EXPECT_THROW(globalMirrorTasks.End(60), InternalException);
}

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_GetTaskInfo)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    globalMirrorTasks.CreateQueue(30, 0, QueueType::Circular_Queue);

    globalMirrorTasks.GetTaskInfo(30, 0, 0);
}

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_DevSize)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    globalMirrorTasks.DevSize();
}

TEST_F(GlobalMirrorTasksTest, GlobalMirrorTasks_GetAllTaskInfo)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    for (auto& taskMap : globalMirrorTasks.taskMaps_) {
        taskMap.clear();
    }
    // 枚举所有devId
    for (u32 devId = 0; devId < 32; devId++) {
        // 枚举所有streamId
        for (auto iter = globalMirrorTasks.Begin(devId); iter != globalMirrorTasks.End(devId); iter++) {
            // 获取对应streamId和任务队列的unique_ptr指针
            auto streamId = iter->first;
            auto& taskInfoQueue = iter->second;
            // 枚举所有任务信息
            for (auto taskInfoIter = taskInfoQueue->Begin(); *taskInfoIter != *taskInfoQueue->End();
                 (*taskInfoIter)++) {
                std::cout << (**taskInfoIter)->Describe().c_str() << std::endl;
            }
        }
    }
}
