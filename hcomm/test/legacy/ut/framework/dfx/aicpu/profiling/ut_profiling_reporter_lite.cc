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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <stdexcept>
#include <string>
#define private public
#define protected public
#include "profiling_reporter_lite.h"
#include "communicator_impl_lite.h"
#undef private
#undef protected

using namespace Hccl;

class ProfilingReporterLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ProfilingReporterLiteTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ProfilingReporterLiteTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ProfilingReporterLiteTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ProfilingReporterLiteTest TearDown" << std::endl;
    }
};

// 测试ProfilingReporterLite类接口
TEST_F(ProfilingReporterLiteTest, Call_profilingReporterLite_api_test)
{
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManagerLite mirrorTaskManagerLite;
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
    CommunicatorImplLite* comm = new CommunicatorImplLite(0);
    dfxOpInfo->comm_ = comm;
    mirrorTaskManagerLite.SetCurrDfxOpInfo(dfxOpInfo);
    auto taskInfo1 = std::make_unique<TaskInfo>(3, 0, 0, taskParam, dfxOpInfo);
    auto taskInfo2 = std::make_unique<TaskInfo>(0, 1, 1, taskParam, dfxOpInfo);
    mirrorTaskManagerLite.AddTaskInfo(std::move(taskInfo1));
    mirrorTaskManagerLite.AddTaskInfo(std::move(taskInfo2));
    ProfilingReporterLite profilingReporter(&mirrorTaskManagerLite, &ProfilingHandlerLite::GetInstance());
    profilingReporter.ReportAllTasks();
    profilingReporter.UpdateProfStat();
    delete comm;
}

TEST_F(
    ProfilingReporterLiteTest, Ut_MirrorTaskManagerLite_GetCurrDfxOpInfo_When_DefaultConstructed_Expect_ReturnNullptr)
{
    MirrorTaskManagerLite mirrorTaskManagerLite;
    EXPECT_EQ(mirrorTaskManagerLite.GetCurrDfxOpInfo(), nullptr);
}
