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
#include <chrono>
#include <mockcpp/mockcpp.hpp>
#include <stdexcept>
#include <string>
#define private public
#define protected public
#include "profiling_handler_lite.h"
#include "task_info.h"
#include "task_param.h"
#include "mirror_task_manager.h"
#include "communicator_impl_lite.h"
#include "internal_exception.h"
#undef private
#undef protected

using namespace Hccl;
using namespace aicpu;

int32_t ableNum = 0;
class ProfilingHandlerLiteTest : public testing::Test {
protected:
    ProfilingHandlerLiteTest() : comm(0) {}
    static void SetUpTestCase() { std::cout << "ProfilingHandlerLiteTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ProfilingHandlerLiteTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ProfilingHandlerLiteTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ProfilingHandlerLiteTest TearDown" << std::endl;
    }

    CommunicatorImplLite comm;
};

namespace aicpu {
status_t GetTaskAndStreamId(uint64_t& taskId, uint32_t& streamId) { return status_t::AICPU_ERROR_NONE; }
} // namespace aicpu

extern "C" {
int32_t AdprofReportAdditionalInfo(uint32_t agingFlag, const void* data, uint32_t length)
{
    if (ableNum == 0) {
        return 0;
    } else if (ableNum == 1) {
        return 1;
    } else {
        void* mem = nullptr;
        uintptr_t value = reinterpret_cast<uintptr_t>(mem);
        return value;
    }
}

int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    if (ableNum == 0) {
        return 0;
    } else if (ableNum == 1) {
        return 1;
    } else {
        void* mem = nullptr;
        uintptr_t value = reinterpret_cast<uintptr_t>(mem);
        return value;
    }
}
}

// 全局状态为false：测试ReportHcclOpInfo接口
TEST_F(ProfilingHandlerLiteTest, ReportHcclOpInfo_test)
{
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    dfxOpInfo->tag_ = "testTag";
    dfxOpInfo->commIndex_ = 0;
    dfxOpInfo->beginTime_ = 0;
    dfxOpInfo->endTime_ = 1;
    dfxOpInfo->comm_ = &comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    ableNum = 0;
    handler.enableHcclL0_ = true;
    EXPECT_NO_THROW(handler.ReportHcclOpInfo(*dfxOpInfo));
}

// 全局状态为false：测试ReportHcclOpInfo接口
TEST_F(ProfilingHandlerLiteTest, ReportHcclOpInfo1_test)
{
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    dfxOpInfo->tag_ = "testTag";
    dfxOpInfo->commIndex_ = 0;
    dfxOpInfo->beginTime_ = 0;
    dfxOpInfo->endTime_ = 1;
    dfxOpInfo->comm_ = &comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    ableNum = 1;
    handler.enableHcclL0_ = true;
    handler.reportAdditionalInfo_ = AdprofReportAdditionalInfo;
    EXPECT_THROW(handler.ReportHcclOpInfo(*dfxOpInfo), InternalException);
}

TEST_F(ProfilingHandlerLiteTest, ReportHcclOpInfo2_test)
{
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    dfxOpInfo->tag_ = "testTag";
    dfxOpInfo->commIndex_ = 0;
    dfxOpInfo->beginTime_ = 0;
    dfxOpInfo->endTime_ = 1;
    dfxOpInfo->comm_ = &comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    ableNum = 2;
    handler.enableHcclL0_ = true;
    EXPECT_NO_THROW(handler.ReportHcclOpInfo(*dfxOpInfo));
}

// 测试ReportMainStreamTask接口
TEST_F(ProfilingHandlerLiteTest, ReportMainStreamTask_test)
{
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    FlagTaskInfo flagTaskInfo;
    flagTaskInfo.taskId = 0;
    flagTaskInfo.type = MainStreamTaskType::HEAD;
    handler.enableHcclL0_ = true;
    handler.enableHcclL1_ = true;
    handler.ReportMainStreamTask(flagTaskInfo);
}

// 测试ReporttHcclTaskDetails接口
TEST_F(ProfilingHandlerLiteTest, ReporttHcclTaskDetails_test)
{
    std::vector<TaskInfo*> taskInfo;
    u32 streamId = 1;
    u32 taskId = 1;
    u32 remoteRank = 1;
    TaskParam taskParam
        = {TaskParamType::TASK_NOTIFY_RECORD, std::chrono::high_resolution_clock::now().time_since_epoch().count(),
           std::chrono::high_resolution_clock::now().time_since_epoch().count(), 0, 0};
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opTag = "testTag";
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImplLite* comm = new CommunicatorImplLite(0);
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    TaskInfo task(streamId, taskId, remoteRank, taskParam, dfxOpInfo);
    for (int i = 0; i <= 2; i++) {
        taskInfo.push_back(&task);
    }
    handler.enableHcclL1_ = true;
    handler.ReportHcclTaskDetails(taskInfo);
    delete comm;
}

// 测试ReporttHcclTaskDetails接口
TEST_F(ProfilingHandlerLiteTest, ReporttHcclTaskDetails1_test)
{
    std::vector<TaskInfo*> taskInfo;
    u32 streamId = 1;
    u32 taskId = 1;
    u32 remoteRank = 1;
    TaskParam taskParam
        = {TaskParamType::TASK_SDMA, std::chrono::high_resolution_clock::now().time_since_epoch().count(),
           std::chrono::high_resolution_clock::now().time_since_epoch().count(), 0, 0};
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImplLite* comm = new CommunicatorImplLite(0);
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    TaskInfo task(streamId, taskId, remoteRank, taskParam, dfxOpInfo);
    taskInfo.push_back(&task);
    handler.enableHcclL1_ = true;
    handler.ReportHcclTaskDetails(taskInfo);
    delete comm;
}

TEST_F(ProfilingHandlerLiteTest, ReporttHcclTaskDetails2_test)
{
    std::vector<TaskInfo*> taskInfo;
    u32 streamId = 1;
    u32 taskId = 1;
    u32 remoteRank = 1;
    TaskParam taskParam
        = {TaskParamType::TASK_REDUCE_INLINE, std::chrono::high_resolution_clock::now().time_since_epoch().count(),
           std::chrono::high_resolution_clock::now().time_since_epoch().count(), 0, 0};
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    MirrorTaskManager mirrorTaskManager(0, &globalMirrorTasks, 0);
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    CommunicatorImplLite* comm = new CommunicatorImplLite(0);
    dfxOpInfo->comm_ = comm;
    mirrorTaskManager.SetCurrDfxOpInfo(dfxOpInfo);
    TaskInfo task(streamId, taskId, remoteRank, taskParam, dfxOpInfo);
    taskInfo.push_back(&task);
    handler.enableHcclL1_ = true;
    handler.ReportHcclTaskDetails(taskInfo);
    delete comm;
}

// 测试aicpu开关状态接口
TEST_F(ProfilingHandlerLiteTest, GetProfState_test)
{
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    uint64_t feature = 0;
    handler.enableHcclL0_ = false;
    handler.enableHcclL1_ = false;
    EXPECT_EQ(false, handler.GetProfL0State());
    EXPECT_EQ(false, handler.GetProfL1State());
    EXPECT_EQ(false, handler.IsL1fromOffToOn());
    EXPECT_EQ(false, handler.IsProfOn(feature));
    EXPECT_EQ(false, handler.IsProfSwitchOn(ProfilingLevel::L1));
    EXPECT_EQ(false, handler.IsProfSwitchOn(ProfilingLevel::L0));
    handler.UpdateProfSwitch();
}
// 测试aicpu开关状态接口
TEST_F(ProfilingHandlerLiteTest, SetProLevelOn_test)
{
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    uint64_t feature = 0;
    handler.enableHcclL0_ = false;
    handler.enableHcclL1_ = false;
    handler.SetProL1On(true);
    handler.SetProL0On(true);
    EXPECT_EQ(true, handler.enableHcclL0_);
    EXPECT_EQ(true, handler.enableHcclL1_);
}

TEST_F(ProfilingHandlerLiteTest, Ut_ReportHcclOpInfo_When_CommNullptr_Expect_NoThrow)
{
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    dfxOpInfo->tag_ = "testTag";
    dfxOpInfo->commIndex_ = 0;
    dfxOpInfo->beginTime_ = 0;
    dfxOpInfo->endTime_ = 1;
    dfxOpInfo->comm_ = nullptr;
    dfxOpInfo->isIndop_ = false;
    ableNum = 0;
    handler.enableHcclL0_ = true;
    EXPECT_NO_THROW(handler.ReportHcclOpInfo(*dfxOpInfo));
}

TEST_F(ProfilingHandlerLiteTest, Ut_ReportHcclTaskDetails_When_DfxOpInfoNullptr_Expect_NoCrash)
{
    ProfilingHandlerLite& handler = Hccl::ProfilingHandlerLite::GetInstance();
    std::vector<TaskInfo*> taskInfoVec;
    TaskParam taskParam{};
    TaskInfo taskInfo(0, 0, 0, taskParam, nullptr);
    taskInfoVec.push_back(&taskInfo);
    handler.enableHcclL1_ = true;
    EXPECT_NO_THROW(handler.ReportHcclTaskDetails(taskInfoVec));
}
