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
#include "mirror_task_manager.h"
#include "invalid_params_exception.h"
#include "internal_exception.h"
#include <chrono>
#include <memory>
#undef private

using namespace Hccl;

class MirrorTaskManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MirrorTaskManager tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "MirrorTaskManager tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in MirrorTaskManager SetUP" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "A Test case in MirrorTaskManager TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

TEST_F(MirrorTaskManagerTest, MirrorTaskManager_AddTaskInfo_1)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    for (auto& taskMap : globalMirrorTasks.taskMaps_) {
        taskMap.clear();
    }
    TaskParam taskParam
        = {TaskParamType::TASK_NOTIFY_RECORD, std::chrono::high_resolution_clock::now().time_since_epoch().count(),
           std::chrono::high_resolution_clock::now().time_since_epoch().count(), 0, 0};

    CollOperator op;
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();

    op.staticAddr = false;
    dfxOpInfo->op_ = op;

    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    auto taskInfo = std::make_unique<TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);

    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));

    mirrorTaskManager.GetCurrDfxOpInfo();

    mirrorTaskManager.GetQueue(0);

    globalMirrorTasks.GetTaskInfo(0, 0, 0);
}

TEST_F(MirrorTaskManagerTest, MirrorTaskManager_AddTaskInfo_2)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);

    std::unique_ptr<TaskInfo> nullTaskInfo = nullptr;
    EXPECT_THROW(mirrorTaskManager.AddTaskInfo(nullTaskInfo), InternalException);
}

TEST_F(MirrorTaskManagerTest, MirrorTaskManager_GetQueue_1)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);

    EXPECT_THROW(mirrorTaskManager.GetQueue(0), InternalException);
}

TEST_F(MirrorTaskManagerTest, MirrorTaskManager_Iterator_1)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();

    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    for (auto& taskMap : globalMirrorTasks.taskMaps_) {
        taskMap.clear();
    }
    // 初始化TaskParam
    TaskParam taskParam
        = {.taskType = TaskParamType::TASK_NOTIFY_RECORD,
           .beginTime = 0,
           .endTime = 0,
           .taskPara = {.Notify = {.notifyID = 123, .value = 456}}};
    // 初始化dfxOpInfo
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;

    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);

    auto taskInfo1 = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    auto taskInfo2 = std::make_unique<TaskInfo>(0, 1, 1, taskParam, dfxOpInfo);

    mirrorTaskManager.AddTaskInfo(taskInfo1);
    mirrorTaskManager.AddTaskInfo(taskInfo2);

    // 枚举所有streamId
    for (auto queueIter = mirrorTaskManager.Begin(); queueIter != mirrorTaskManager.End(); queueIter++) {
        // 获取对应streamId和任务队列的指针
        auto streamId = queueIter->first;
        Queue<std::unique_ptr<TaskInfo>>* taskInfoQueue = queueIter->second;

        // 枚举所有任务信息
        for (auto taskInfoIter = taskInfoQueue->Begin(); *taskInfoIter != *taskInfoQueue->End(); (*taskInfoIter)++) {
            std::cout << (**taskInfoIter)->Describe().c_str() << std::endl;
        }
    }
}
