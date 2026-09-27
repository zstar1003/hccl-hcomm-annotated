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
#include <vector>
#define private public
#define protected public
#include "dfx_profiling_reporter_lite.h"
#include "dfx_profiling_handler_lite.h"
#include "res_pub.h"
#undef private
#undef protected

using namespace Hccl;

class DfxProfilingReporterLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DfxProfilingReporterLiteTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "DfxProfilingReporterLiteTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        handler_.enableHcclL0_ = false;
        handler_.enableHcclL1_ = false;
        reporter_ = std::make_unique<DfxProfilingReporterLite>(&handler_);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in DfxProfilingReporterLiteTest TearDown" << std::endl;
    }

    DfxProfilingHandlerLite& handler_ = DfxProfilingHandlerLite::GetInstance();
    std::unique_ptr<DfxProfilingReporterLite> reporter_;
};

TEST_F(DfxProfilingReporterLiteTest, Ut_Constructor_When_Normal_Expect_HandlerSet)
{
    auto rep = std::make_unique<DfxProfilingReporterLite>(&handler_);
    EXPECT_NE(rep, nullptr);
    EXPECT_EQ(rep->profilingHandlerLite_, &handler_);
    EXPECT_EQ(rep->initializedFlag_, false);
}

TEST_F(DfxProfilingReporterLiteTest, Ut_Constructor_When_NullptrHandler_Expect_NoThrow)
{
    EXPECT_NO_THROW(std::make_unique<DfxProfilingReporterLite>(nullptr));
}

TEST_F(DfxProfilingReporterLiteTest, Ut_Init_When_Normal_Expect_ReturnSuccess)
{
    reporter_->initializedFlag_ = false;
    HcclResult ret = reporter_->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DfxProfilingReporterLiteTest, Ut_Init_When_AlreadyInitialized_Expect_ReturnSuccess)
{
    reporter_->initializedFlag_ = true;
    HcclResult ret = reporter_->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DfxProfilingReporterLiteTest, Ut_ReportStreamTask_When_NullptrQueue_Expect_NoThrow)
{
    EXPECT_NO_THROW(reporter_->ReportStreamTask(nullptr, DfxCommContext{}));
}

TEST_F(DfxProfilingReporterLiteTest, Ut_ReportStreamTask_When_L1Off_Expect_NoThrow)
{
    handler_.enableHcclL1_ = false;
    TaskInfoCircularQueue queue;
    EXPECT_NO_THROW(reporter_->ReportStreamTask(&queue, DfxCommContext{}));
}

TEST_F(DfxProfilingReporterLiteTest, Ut_ReportStreamTask_When_L1On_Expect_NoThrow)
{
    handler_.enableHcclL1_ = true;
    TaskInfoCircularQueue queue;
    EXPECT_NO_THROW(reporter_->ReportStreamTask(&queue, DfxCommContext{}));
}

TEST_F(DfxProfilingReporterLiteTest, Ut_ReportAllTasks_When_EmptyThreads_Expect_NoThrow)
{
    std::vector<hccl::Thread*> threads;
    EXPECT_NO_THROW(reporter_->ReportAllTasks(threads, DfxCommContext{}));
}

TEST_F(DfxProfilingReporterLiteTest, Ut_ReportAllTasks_When_L1Off_Expect_NoThrow)
{
    handler_.enableHcclL1_ = false;
    std::vector<hccl::Thread*> threads;
    EXPECT_NO_THROW(reporter_->ReportAllTasks(threads, DfxCommContext{}));
}

TEST_F(DfxProfilingReporterLiteTest, Ut_UpdateProfStat_Expect_NoThrow) { EXPECT_NO_THROW(reporter_->UpdateProfStat()); }

TEST_F(DfxProfilingReporterLiteTest, Ut_ReportAllTasksLog_When_EmptyThreads_Expect_NoThrow)
{
    std::vector<hccl::Thread*> threads;
    EXPECT_NO_THROW(reporter_->ReportAllTasksLog(threads));
}
