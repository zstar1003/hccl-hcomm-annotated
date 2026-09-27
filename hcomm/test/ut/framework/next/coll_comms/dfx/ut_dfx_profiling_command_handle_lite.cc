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
#define private public
#define protected public
#include "dfx_profiling_command_handle_lite.h"
#include "dfx_profiling_handler_lite.h"
#undef private
#undef protected

using namespace Hccl;

class DfxProfilingCommandHandleLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DfxProfilingCommandHandleLiteTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "DfxProfilingCommandHandleLiteTest TearDown" << std::endl; }

    virtual void SetUp() {}

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in DfxProfilingCommandHandleLiteTest TearDown" << std::endl;
    }

    DfxProfilingHandlerLite& handler_ = DfxProfilingHandlerLite::GetInstance();
};

TEST_F(DfxProfilingCommandHandleLiteTest, Ut_DfxRegisterProfCallBack_When_Normal_Expect_ReturnSuccess)
{
    HcclResult ret = DfxRegisterProfCallBack();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DfxProfilingCommandHandleLiteTest, Ut_DfxRegisterProfCallBack_When_CallbackNull_Expect_ReturnSuccess)
{
    HcclResult ret = DfxRegisterProfCallBack();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DfxProfilingCommandHandleLiteTest, Ut_DfxDeviceCommandHandle_When_DataNull_Expect_ReturnFailed)
{
    int32_t ret = DfxDeviceCommandHandle(0, nullptr, 0);
    EXPECT_EQ(ret, PROF_FAILED);
}

TEST_F(DfxProfilingCommandHandleLiteTest, Ut_DfxDeviceCommandHandle_When_StartL0_Expect_NoThrow)
{
    MsprofCommandHandle command{};
    command.type = PROF_COMMANDHANDLE_TYPE_START;
    command.profSwitch = ADPROF_TASK_TIME_L0;
    int32_t ret = DfxDeviceCommandHandle(0, &command, sizeof(command));
    EXPECT_EQ(ret, PROF_SUCCESS);
}

TEST_F(DfxProfilingCommandHandleLiteTest, Ut_DfxDeviceCommandHandle_When_StartL1_Expect_NoThrow)
{
    MsprofCommandHandle command{};
    command.type = PROF_COMMANDHANDLE_TYPE_START;
    command.profSwitch = ADPROF_TASK_TIME_L1;
    int32_t ret = DfxDeviceCommandHandle(0, &command, sizeof(command));
    EXPECT_EQ(ret, PROF_SUCCESS);
}

TEST_F(DfxProfilingCommandHandleLiteTest, Ut_DfxDeviceCommandHandle_When_Stop_Expect_NoThrow)
{
    MsprofCommandHandle command{};
    command.type = PROF_COMMANDHANDLE_TYPE_STOP;
    command.profSwitch = 0;
    int32_t ret = DfxDeviceCommandHandle(0, &command, sizeof(command));
    EXPECT_EQ(ret, PROF_SUCCESS);
}
