/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "coll_comm.h"
#include "ns_recovery/task_abort_handler.h"
#include "ccu_device_pub.h"
#undef private

using namespace hccl;
using namespace hcomm;

class HcclTaskAbortHandlerTest : public testing::Test {
public:
    HcclTaskAbortHandler handler;

    static void SetUpTestCase() { std::cout << "HcclTaskAbortHandlerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclTaskAbortHandlerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        handler.commVector_.clear();
        std::cout << "A Test case in HcclTaskAbortHandlerTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in HcclTaskAbortHandlerTest TearDown" << std::endl;
        handler.commVector_.clear();
        GlobalMockObject::verify();
    }
};

// 构造一个 dummy communicator（空指针占位）并注册，返回回调参数指针
static void* PrepareAndRegisterComm(HcclTaskAbortHandler& handler, CollComm** comm)
{
    *comm = nullptr;
    handler.Register(*comm);
    return static_cast<void*>(&handler);
}

// 执行 POST 阶段回调：设置 CCU/Clean mock 并带/不带超时各触发一次回调，断言期望结果
void RunPostStageCallback(HcclTaskAbortHandler& handler, bool ccuInited, HcclResult cleanResult, TaskAbortResult expect)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 30U;
    CollComm* comm = nullptr;
    void* args = PrepareAndRegisterComm(handler, &comm);

    MOCKER(CcuIsInited).stubs().will(returnValue(ccuInited));
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuSetTaskKillDone).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuCleanDieCkes).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::Clean, HcclResult(CollComm::*)()).stubs().with(mockcpp::any()).will(returnValue(cleanResult));

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(expect));
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(expect));

    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_pre_success)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm* comm = nullptr;
    void* args = PrepareAndRegisterComm(handler, &comm);

    // 模拟 Suspend 方法返回成功
    MOCKER_CPP(&CollComm::Suspend, HcclResult(CollComm::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    // 清理
    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_with_null_args)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t timeout = 30U;

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, nullptr);
    EXPECT_NE(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));
}

TEST_F(HcclTaskAbortHandlerTest, test_unregister_not_found_comm)
{
    CollComm* comm1 = reinterpret_cast<CollComm*>(0x1000);
    CollComm* comm2 = reinterpret_cast<CollComm*>(0x2000);

    handler.Register(comm1);
    HcclResult ret = handler.UnRegister(comm2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    handler.UnRegister(comm1);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_pre_suspending)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t timeout = 0U;

    CollComm* comm = nullptr;
    void* args = PrepareAndRegisterComm(handler, &comm);

    MOCKER_CPP(&CollComm::Suspend, HcclResult(CollComm::*)()).stubs().will(returnValue(HCCL_E_SUSPENDING));

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_post_suspending)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 0U;

    CollComm* comm = nullptr;
    void* args = PrepareAndRegisterComm(handler, &comm);

    MOCKER(CcuIsInited).stubs().will(returnValue(true));
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuSetTaskKillDone).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuCleanDieCkes).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollComm::Clean, HcclResult(CollComm::*)()).stubs().will(returnValue(HCCL_E_SUSPENDING));

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_post_ccu_set_task_kill_fail)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 0U;

    CollComm* comm = nullptr;
    void* args = PrepareAndRegisterComm(handler, &comm);

    MOCKER(CcuIsInited).stubs().will(returnValue(true));
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_E_INTERNAL));

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_NE(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_register_and_unregister_multiple)
{
    CollComm* comm1 = reinterpret_cast<CollComm*>(0x1000);
    CollComm* comm2 = reinterpret_cast<CollComm*>(0x2000);

    HcclResult ret = handler.Register(comm1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = handler.Register(comm2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(handler.commVector_.size(), 2u);

    ret = handler.UnRegister(comm1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handler.commVector_.size(), 1u);

    ret = handler.UnRegister(comm2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handler.commVector_.size(), 0u);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_pre_fail)
{
    // 构造入参
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE;
    uint32_t timeout = 30U;

    // 使用 nullptr 作为测试 communicator 的占位符并注册
    CollComm* comm = nullptr;
    void* args = PrepareAndRegisterComm(handler, &comm);

    // 模拟 Suspend 方法返回失败
    MOCKER_CPP(&CollComm::Suspend, HcclResult(CollComm::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    // 测试带超时的情况
    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL));

    // 测试不带超时的情况
    timeout = 0U;
    ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL));

    // 清理
    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_post_success)
{
    RunPostStageCallback(handler, true, HCCL_SUCCESS, TaskAbortResult::TASK_ABORT_SUCCESS);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_post_skip_taskkill_when_ccu_not_inited)
{
    RunPostStageCallback(handler, false, HCCL_SUCCESS, TaskAbortResult::TASK_ABORT_SUCCESS);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_handle_call_back_stage_post_fail)
{
    RunPostStageCallback(handler, true, HCCL_E_INTERNAL, TaskAbortResult::TASK_ABORT_FAIL);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_post_clean_die_ckes_called_per_die)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 0U;

    CollComm* comm = nullptr;
    void* args = PrepareAndRegisterComm(handler, &comm);

    MOCKER(CcuIsInited).stubs().will(returnValue(true));
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuSetTaskKillDone).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuCleanDieCkes)
        .expects(exactly(static_cast<int>(CCU_MAX_IODIE_NUM)))
        .with(eq(deviceLogicId), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::Clean, HcclResult(CollComm::*)()).stubs().will(returnValue(HCCL_SUCCESS));

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_EQ(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    handler.UnRegister(comm);
}

TEST_F(HcclTaskAbortHandlerTest, test_task_abort_post_clean_die_ckes_fail)
{
    int32_t deviceLogicId = 0;
    aclrtDeviceTaskAbortStage stage = aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST;
    uint32_t timeout = 0U;

    CollComm* comm = nullptr;
    void* args = PrepareAndRegisterComm(handler, &comm);

    MOCKER(CcuIsInited).stubs().will(returnValue(true));
    MOCKER(CcuSetTaskKill).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuSetTaskKillDone).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CcuCleanDieCkes).stubs().will(returnValue(HCCL_E_INTERNAL));
    MOCKER_CPP(&CollComm::Clean, HcclResult(CollComm::*)()).stubs().will(returnValue(HCCL_SUCCESS));

    auto ret = ProcessTaskAbortHandleCallback(deviceLogicId, stage, timeout, args);
    EXPECT_NE(ret, static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS));

    handler.UnRegister(comm);
}
