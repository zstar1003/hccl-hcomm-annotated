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
#define private public
#include "task_abort_handler_v2.h"
#undef private

using namespace Hccl;
class TaskAbortHandlerTest : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "TaskAbortHandlerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "TaskAbortHandlerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        TaskAbortHandler::GetInstance().commVector.clear();

        std::cout << "A Test case in TaskAbortHandlerTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in TaskAbortHandlerTest TearDown" << std::endl;
        TaskAbortHandler::GetInstance().commVector.clear();
        GlobalMockObject::verify();
    }
};

TEST_F(TaskAbortHandlerTest, test_task_abort_handle_call_back_stage_pre_success)
{
    // 构造入参
    uint32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t time = 30U;

    CommParams commParams;
    auto communicator = std::make_unique<HcclCommunicator>(commParams);
    TaskAbortHandler::GetInstance().Register(communicator.get());
    void* args = reinterpret_cast<void*>(&TaskAbortHandler::GetInstance().commVector);

    MOCKER_CPP(&HcclCommunicator::Suspend, HcclResult(HcclCommunicator::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, time, args);
    EXPECT_EQ(ret, 0);

    time = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, time, args);
    EXPECT_EQ(ret, 0);
    GlobalMockObject::verify();
    TaskAbortHandler::GetInstance().UnRegister(communicator.get());
}

TEST_F(TaskAbortHandlerTest, test_task_abort_handle_call_back_stage_pre_fail)
{
    // 构造入参
    uint32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t time = 30U;

    CommParams commParams;
    auto communicator = std::make_unique<HcclCommunicator>(commParams);
    TaskAbortHandler::GetInstance().Register(communicator.get());
    void* args = reinterpret_cast<void*>(&TaskAbortHandler::GetInstance().commVector);

    MOCKER_CPP(&HcclCommunicator::Suspend, HcclResult(HcclCommunicator::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, time, args);
    EXPECT_EQ(ret, 1);

    time = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, time, args);
    EXPECT_EQ(ret, 1);
    GlobalMockObject::verify();
    TaskAbortHandler::GetInstance().UnRegister(communicator.get());
}

TEST_F(TaskAbortHandlerTest, test_task_abort_handle_call_back_stage_post_success)
{
    // 构造入参
    uint32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t time = 30U;

    CommParams commParams;
    auto communicator = std::make_unique<HcclCommunicator>(commParams);
    TaskAbortHandler::GetInstance().Register(communicator.get());
    void* args = reinterpret_cast<void*>(&TaskAbortHandler::GetInstance().commVector);

    MOCKER(HcclCcuTaskKillPreProcess).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HcclCcuTaskKillPostProcess).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::Clean, HcclResult(HcclCommunicator::*)())
        .stubs()
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, time, args);
    EXPECT_EQ(ret, 0);

    time = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, time, args);
    EXPECT_EQ(ret, 0);
    GlobalMockObject::verify();
    TaskAbortHandler::GetInstance().UnRegister(communicator.get());
}

TEST_F(TaskAbortHandlerTest, test_task_abort_handle_call_back_stage_post_fail)
{
    // 构造入参
    uint32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t time = 30U;

    CommParams commParams;
    auto communicator = std::make_unique<HcclCommunicator>(commParams);
    TaskAbortHandler::GetInstance().Register(communicator.get());
    void* args = reinterpret_cast<void*>(&TaskAbortHandler::GetInstance().commVector);

    MOCKER(HcclCcuTaskKillPreProcess).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HcclCcuTaskKillPostProcess).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::Clean, HcclResult(HcclCommunicator::*)())
        .stubs()
        .will(returnValue(HcclResult::HCCL_E_INTERNAL));
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, time, args);
    EXPECT_EQ(ret, 1);

    time = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, time, args);
    EXPECT_EQ(ret, 1);

    TaskAbortHandler::GetInstance().UnRegister(communicator.get());
}
