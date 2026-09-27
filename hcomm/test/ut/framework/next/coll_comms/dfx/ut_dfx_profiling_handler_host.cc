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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <cstring>

#define private public
#define protected public

// 阻止旧版 profiling_handler.h / profiling_reporter.h 被重复包含
// （stub 链接旧版，新版 dfx_profiling_handler.h 定义了相同的 struct/const）
#define HCCL_PROFILING_HANDLER_H
#define HCCL_PROFILING_REPORTER_H

#include "dfx_dlprof_function.h"
#include "dfx_profiling_handler.h"
#include "dfx_profiling_reporter.h"
#include "mirror_task_manager.h"
#undef private
#undef protected

#include "task_info.h"
#include "task_param.h"

using namespace Hccl;

// ===========================================================================
// DfxDlProfFunction UT
// ===========================================================================

class DfxDlProfFunctionTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& inst = DfxDlProfFunction::GetInstance();
        inst.initializedFlag_.store(false, std::memory_order_release);
        inst.handle_ = nullptr;
    }
    void TearDown() override { GlobalMockObject::verify(); }

    DfxDlProfFunction& inst_ = DfxDlProfFunction::GetInstance();
};

TEST_F(DfxDlProfFunctionTest, Ut_StubInit_Expect_StubFuncsNonNull)
{
    EXPECT_TRUE(inst_.dlMsprofRegisterCallback);
    EXPECT_TRUE(inst_.dlMsprofRegTypeInfo);
    EXPECT_TRUE(inst_.dlMsprofReportApi);
    EXPECT_TRUE(inst_.dlMsprofReportCompactInfo);
    EXPECT_TRUE(inst_.dlMsprofReportAdditionalInfo);
    EXPECT_TRUE(inst_.dlMsprofStr2Id);
    EXPECT_TRUE(inst_.dlMsprofSysCycleTime);
}

TEST_F(DfxDlProfFunctionTest, Ut_Init_When_AlreadyInit_Expect_Success)
{
    inst_.initializedFlag_.store(true, std::memory_order_release);
    EXPECT_EQ(inst_.DfxDlProfFunctionInit(), HCCL_SUCCESS);
}

TEST_F(DfxDlProfFunctionTest, Ut_Init_When_NotInit_NoLib_Expect_NoThrow)
{
    inst_.handle_ = nullptr;
    EXPECT_NO_THROW(inst_.DfxDlProfFunctionInit());
}

TEST_F(DfxDlProfFunctionTest, Ut_StubFuncs_Expect_NonNull)
{
    EXPECT_TRUE(inst_.dlMsprofSysCycleTime);
    EXPECT_TRUE(inst_.dlMsprofStr2Id);
    EXPECT_TRUE(inst_.dlMsprofReportApi);
}

// ===========================================================================
// DfxProfilingHandler UT
// ===========================================================================

class DfxProfilingHandlerTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& h = DfxProfilingHandler::GetInstance();
        h.initializedFlag_.store(false);
        h.enableHostApi_.store(false);
        h.enableHcclNode_.store(false);
        h.enableHcclL0_.store(false);
        h.enableHcclL1_.store(false);
        h.cachedAlgTypeHashId_.store(0);
        h.cacheTaskInfos_.clear();
        while (!h.cachedTaskApiInfo_.empty()) {
            h.cachedTaskApiInfo_.pop();
        }
        while (!h.cachedAclApiInfo_.empty()) {
            h.cachedAclApiInfo_.pop();
        }
        while (!h.cacheHcclOpInfo_.empty()) {
            h.cacheHcclOpInfo_.pop();
        }
        while (!h.cacheHcclAdditionInfo_.empty()) {
            h.cacheHcclAdditionInfo_.pop();
        }
    }
    void TearDown() override { GlobalMockObject::verify(); }

    DfxProfilingHandler& handler_ = DfxProfilingHandler::GetInstance();
};

TEST_F(DfxProfilingHandlerTest, Ut_Init_When_AlreadyInit_Expect_Success)
{
    handler_.initializedFlag_.store(true);
    EXPECT_EQ(handler_.Init(), HCCL_SUCCESS);
}

TEST_F(DfxProfilingHandlerTest, Ut_GetProfHashId_When_NameNull_Expect_Invalid)
{
    EXPECT_EQ(handler_.GetProfHashId(nullptr, 10), 0xFFFFFFFFFFFFFFFFULL);
}

TEST_F(DfxProfilingHandlerTest, Ut_GetProfHashId_When_LenZero_Expect_Invalid)
{
    EXPECT_EQ(handler_.GetProfHashId("test", 0), 0xFFFFFFFFFFFFFFFFULL);
}

TEST_F(DfxProfilingHandlerTest, Ut_GetProfHashId_When_ValidName_Expect_StubReturn0)
{
    EXPECT_EQ(handler_.GetProfHashId("allreduce", 9), 0ULL);
}

TEST_F(DfxProfilingHandlerTest, Ut_GetTaskTypeValue_When_DpuTask_Expect_DpuInfo)
{
    EXPECT_EQ(
        handler_.GetTaskTypeValue(TaskParamType::TASK_DPU_INLINE_WRITE),
        static_cast<uint32_t>(ProfTaskType::TASK_DPU_HCCL_INFO));
}

TEST_F(DfxProfilingHandlerTest, Ut_GetTaskTypeValue_When_OtherTask_Expect_HcclInfo)
{
    EXPECT_EQ(handler_.GetTaskTypeValue(TaskParamType::TASK_SDMA), static_cast<uint32_t>(ProfTaskType::TASK_HCCL_INFO));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_DpuTask_Expect_EarlyReturn)
{
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_DPU_KERNEL, 100, 200, true, false, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_AicpuKernel_Expect_EarlyReturn)
{
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_AICPU_KERNEL, 100, 200, true, false, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_CachedReq_Expect_Pushed)
{
    handler_.enableHcclNode_.store(false);
    size_t before = handler_.cachedTaskApiInfo_.size();
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_SDMA, 100, 200, true, true, false));
    EXPECT_EQ(handler_.cachedTaskApiInfo_.size(), before + 1);
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_NodeDisabled_Expect_Return)
{
    handler_.enableHcclNode_.store(false);
    handler_.enableHcclL1_.store(true);
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_SDMA, 100, 200, true, false, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_NodeEnabled_L1Disabled_IgnoreFalse_Expect_Return)
{
    handler_.enableHcclNode_.store(true);
    handler_.enableHcclL1_.store(false);
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_SDMA, 100, 200, true, false, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskApi_When_NodeDisabled_L1Enabled_IgnoreTrue_Expect_Return)
{
    handler_.enableHcclNode_.store(false);
    handler_.enableHcclL1_.store(true);
    EXPECT_NO_THROW(handler_.ReportHcclTaskApi(TaskParamType::TASK_SDMA, 100, 200, true, false, true));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetails_When_L1Off_NotCached_Expect_Return)
{
    handler_.enableHcclL1_.store(false);
    TaskInfo taskInfo(0, 0, 0, {}, nullptr, false);
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetails(taskInfo, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetails_When_DfxOpInfoNull_Expect_Return)
{
    handler_.enableHcclL1_.store(true);
    TaskInfo taskInfo(0, 0, 0, {}, nullptr, false);
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetails(taskInfo, true));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetails_When_CommNull_Expect_Return)
{
    handler_.enableHcclL1_.store(true);
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->comm_ = nullptr;
    TaskInfo taskInfo(0, 0, 0, {}, opInfo, false);
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetails(taskInfo, true));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_Empty_Expect_Return)
{
    std::vector<TaskInfo*> tasks;
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetailsBatch(tasks, false));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_DfxOpInfoNull_Expect_Return)
{
    handler_.enableHcclL1_.store(true);
    TaskInfo taskInfo(0, 0, 0, {}, nullptr, false);
    std::vector<TaskInfo*> tasks{&taskInfo};
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetailsBatch(tasks, true));
}

TEST_F(DfxProfilingHandlerTest, Ut_ReportHcclTaskDetailsBatch_When_CommNull_Expect_Return)
{
    handler_.enableHcclL1_.store(true);
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->comm_ = nullptr;
    TaskInfo taskInfo(0, 0, 0, {}, opInfo, false);
    std::vector<TaskInfo*> tasks{&taskInfo};
    EXPECT_NO_THROW(handler_.ReportHcclTaskDetailsBatch(tasks, true));
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_DataNull_Expect_Para)
{
    EXPECT_EQ(handler_.CommandHandle(0, nullptr, 100), HCCL_E_PARA);
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_LenTooSmall_Expect_Para)
{
    rtProfCommandHandle_t data{};
    EXPECT_EQ(handler_.CommandHandle(0, &data, 1), HCCL_E_PARA);
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_BadRtType_Expect_Para)
{
    rtProfCommandHandle_t data{};
    EXPECT_EQ(handler_.CommandHandle(999, &data, sizeof(data)), HCCL_E_PARA);
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_Start_Expect_NoThrow)
{
    rtProfCommandHandle_t data{};
    data.type = PROF_COMMANDHANDLE_TYPE_START;
    EXPECT_NO_THROW(
        handler_.CommandHandle(static_cast<uint32_t>(rtProfCtrlType_t::RT_PROF_CTRL_SWITCH), &data, sizeof(data)));
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_Stop_Expect_NoThrow)
{
    rtProfCommandHandle_t data{};
    data.type = PROF_COMMANDHANDLE_TYPE_STOP;
    EXPECT_NO_THROW(
        handler_.CommandHandle(static_cast<uint32_t>(rtProfCtrlType_t::RT_PROF_CTRL_SWITCH), &data, sizeof(data)));
}

TEST_F(DfxProfilingHandlerTest, Ut_CommandHandle_When_Default_Expect_NoThrow)
{
    rtProfCommandHandle_t data{};
    data.type = static_cast<decltype(data.type)>(999);
    EXPECT_NO_THROW(
        handler_.CommandHandle(static_cast<uint32_t>(rtProfCtrlType_t::RT_PROF_CTRL_SWITCH), &data, sizeof(data)));
}

TEST_F(DfxProfilingHandlerTest, Ut_GetHostApiState_Expect_MatchFlag)
{
    handler_.enableHostApi_.store(true);
    EXPECT_TRUE(handler_.GetHostApiState());
    handler_.enableHostApi_.store(false);
    EXPECT_FALSE(handler_.GetHostApiState());
}

TEST_F(DfxProfilingHandlerTest, Ut_GetHcclNodeState_Expect_MatchFlag)
{
    handler_.enableHcclNode_.store(true);
    EXPECT_TRUE(handler_.GetHcclNodeState());
}

TEST_F(DfxProfilingHandlerTest, Ut_GetHcclL0State_Expect_MatchFlag)
{
    handler_.enableHcclL0_.store(true);
    EXPECT_TRUE(handler_.GetHcclL0State());
}

TEST_F(DfxProfilingHandlerTest, Ut_GetHcclL1State_Expect_MatchFlag)
{
    handler_.enableHcclL1_.store(true);
    EXPECT_TRUE(handler_.GetHcclL1State());
}

TEST_F(DfxProfilingHandlerTest, Ut_GetCachedAlgTypeHashId_Expect_MatchStored)
{
    handler_.cachedAlgTypeHashId_.store(12345);
    EXPECT_EQ(handler_.GetCachedAlgTypeHashId(), 12345ULL);
}

// ===========================================================================
// DfxProfilingReporter UT
// ===========================================================================

class DfxProfilingReporterTest : public testing::Test {
protected:
    void SetUp() override
    {
        handlerPtr_ = &DfxProfilingHandler::GetInstance();
        mgr_ = std::make_unique<MirrorTaskManager>(0, &GlobalMirrorTasks::Instance(), false);
        reporter_ = std::make_unique<DfxProfilingReporter>(mgr_.get(), handlerPtr_);
    }
    void TearDown() override { GlobalMockObject::verify(); }

    DfxProfilingHandler* handlerPtr_ = nullptr;
    std::unique_ptr<MirrorTaskManager> mgr_;
    std::unique_ptr<DfxProfilingReporter> reporter_;
};

TEST_F(DfxProfilingReporterTest, Ut_Init_When_AlreadyInit_Expect_Success)
{
    reporter_->initializedFlag_ = true;
    EXPECT_EQ(reporter_->Init(), HCCL_SUCCESS);
}

TEST_F(DfxProfilingReporterTest, Ut_Init_When_NullMgr_Expect_NoThrow)
{
    DfxProfilingReporter rep(nullptr, handlerPtr_);
    EXPECT_NO_THROW(rep.Init());
}

TEST_F(DfxProfilingReporterTest, Ut_Init_When_NullHandler_Expect_NoThrow)
{
    DfxProfilingReporter rep(mgr_.get(), nullptr);
    EXPECT_NO_THROW(rep.Init());
}

TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_OpInfoNull_Expect_Return)
{
    EXPECT_NO_THROW(reporter_->ReportOp(1000, false, true));
}

TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_AicpuEngine_Expect_NoThrow)
{
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->op_.opType = OpType::ALLREDUCE;
    opInfo->engine = CommEngine::COMM_ENGINE_AICPU;
    mgr_->SetCurrDfxOpInfo(opInfo);
    EXPECT_NO_THROW(reporter_->ReportOp(1000, false, true));
}

TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_OtherEngine_Expect_NoThrow)
{
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->op_.opType = OpType::ALLREDUCE;
    opInfo->engine = CommEngine::COMM_ENGINE_CPU_TS;
    mgr_->SetCurrDfxOpInfo(opInfo);
    EXPECT_NO_THROW(reporter_->ReportOp(1000, false, true));
}

TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_OpBasedTrue_Expect_NoThrow)
{
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->op_.opType = OpType::ALLREDUCE;
    opInfo->engine = CommEngine::COMM_ENGINE_CPU_TS;
    mgr_->SetCurrDfxOpInfo(opInfo);
    EXPECT_NO_THROW(reporter_->ReportOp(1000, false, true));
}

TEST_F(DfxProfilingReporterTest, Ut_ReportOp_When_OpBasedFalse_Expect_NoThrow)
{
    auto opInfo = std::make_shared<DfxOpInfo>();
    opInfo->op_.opType = OpType::ALLREDUCE;
    opInfo->engine = CommEngine::COMM_ENGINE_CPU_TS;
    mgr_->SetCurrDfxOpInfo(opInfo);
    EXPECT_NO_THROW(reporter_->ReportOp(1000, false, false));
}

TEST_F(DfxProfilingReporterTest, Ut_UpdateProfStat_When_SameState_Expect_NoChange)
{
    bool state = handlerPtr_->GetHcclL1State();
    reporter_->enableHcclL1_.store(state);
    EXPECT_NO_THROW(reporter_->UpdateProfStat());
    EXPECT_EQ(reporter_->enableHcclL1_.load(), state);
}

TEST_F(DfxProfilingReporterTest, Ut_UpdateProfStat_When_StateChanged_Expect_NoThrow)
{
    reporter_->enableHcclL1_.store(false);
    handlerPtr_->enableHcclL1_.store(true);
    EXPECT_NO_THROW(reporter_->UpdateProfStat());
}
