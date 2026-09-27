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
#include <mockcpp/mokc.h>

#include "global_mirror_tasks.h"
#include "mirror_task_manager.h"

#define private public
#define protected public
#include "task_exception_handler_lite.h"
#include "communicator_impl.h"
#include "communicator_impl_lite.h"
#undef private
#undef protected

using namespace std;
using namespace Hccl;

class TaskExceptionHandlerLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TaskExceptionHandlerLiteTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "TaskExceptionHandlerLiteTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in TaskExceptionHandlerLiteTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in TaskExceptionHandlerLiteTest TearDown" << std::endl;
    }

    unique_ptr<TaskInfo> InitTaskInfo(u32 streamId = 0, u32 taskId = 0, u32 remoteRank = 0)
    {
        TaskParam taskParam{};
        shared_ptr<DfxOpInfo> dfxOpInfo = make_shared<DfxOpInfo>();
        return make_unique<TaskInfo>(streamId, taskId, remoteRank, taskParam, dfxOpInfo);
    }
};

TEST_F(TaskExceptionHandlerLiteTest, GetInstance_ShouldReturnSameInstance_WhenCalledMultipleTimes)
{
    auto* instance1 = &TaskExceptionHandlerLite::GetInstance();
    auto* instance2 = &TaskExceptionHandlerLite::GetInstance();
    EXPECT_EQ(instance1, instance2);
}

TEST_F(TaskExceptionHandlerLiteTest, Register_ShouldRegisterCallback_WhenCalled)
{
    auto& instance = TaskExceptionHandlerLite::GetInstance();
    instance.Register();
}

TEST_F(TaskExceptionHandlerLiteTest, test_get_group_rank_info)
{
    auto taskInfo = InitTaskInfo();

    EXPECT_EQ(TaskExceptionHandlerLite::GetGroupRankInfo(*taskInfo), ""); // communicator is nullptr

    // Mock CommunicatorImplLite
    CommunicatorImplLite commImplLite{0};
    commImplLite.rankSize = 4;
    commImplLite.myRank = 1;
    commImplLite.commId = "RankGroup";
    taskInfo->dfxOpInfo_->comm_ = &commImplLite;
    EXPECT_EQ(TaskExceptionHandlerLite::GetGroupRankInfo(*taskInfo), "group:[RankGroup], rankSize[4], rankId[1]");
}

TEST_F(TaskExceptionHandlerLiteTest, test_process_when_task_more_than_50)
{
    // 打桩 GlobalMirrorTasks
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 1); // diveceId 0
    shared_ptr<DfxOpInfo> dfxOpInfo = make_shared<DfxOpInfo>();
    dfxOpInfo->commIndex_ = 3;
    dfxOpInfo->op_.dataCount = 0xff;
    dfxOpInfo->op_.reduceOp = ReduceOp::MAX;
    dfxOpInfo->op_.dataType = DataType::FP8E4M3;
    dfxOpInfo->algType_ = AlgType{AlgType::BINARY_HD}.Describe();
    dfxOpInfo->op_.inputMem = make_shared<Buffer>(0x111122223333, 0);
    dfxOpInfo->op_.outputMem = make_shared<Buffer>(0xaaaabbbbcccc, 0);
    CommunicatorImplLite communicator{0}; // Mock CommunicatorImpl
    communicator.commId = "GroupName";
    communicator.rankSize = 4;
    communicator.myRank = 1;
    dfxOpInfo->comm_ = &communicator;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    // 加入一些 Task 数据
    // 在异常 Task 前加入60个 Task
    for (uint32_t i = 0; i < 50; ++i) {
        auto preTaskInfo = InitTaskInfo(0, i); // streamId 0, taskId 0-59
        preTaskInfo->dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
        preTaskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
        preTaskInfo->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
        mirrorTaskManager.AddTaskInfo(preTaskInfo);
    }
    // 加入当前异常 Task
    auto taskInfo = InitTaskInfo(0, 60); // streamId 0, taskId 60
    taskInfo->dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    taskInfo->taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskInfo->taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    mirrorTaskManager.AddTaskInfo(std::move(taskInfo));

    // 调用 TaskExceptionHandler::Process() 打印异常DFX信息
    rtLogicCqReport_t exceptionInfo{};
    exceptionInfo.streamId = 0;
    exceptionInfo.taskId = 60; // 当前异常TaskId 60
    TaskExceptionHandlerLite::Process(&exceptionInfo);

    globalMirrorTasks.DestroyQueue(0, 0); // diveceId 0, streamId 0
}
