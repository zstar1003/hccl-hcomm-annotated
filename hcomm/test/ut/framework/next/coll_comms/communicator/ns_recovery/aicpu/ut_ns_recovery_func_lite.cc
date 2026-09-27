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

#include "ns_recovery/aicpu/ns_recovery_func_lite.h"
#include "coll_comm_aicpu_mgr.h"

using namespace hccl;

class NsRecoveryFuncLiteTest : public testing::Test {
protected:
    void SetUp() override { GlobalMockObject::reset(); }
    void TearDown() override { GlobalMockObject::reset(); }

    static void MockHalTsdrvCtlReturn(int rv)
    {
        MOCKER(halTsdrvCtl).stubs().with(mockcpp::any()).will(returnValue(rv));
    }

    static drvError_t
    HalTsdrvCtlStub(uint32_t devId, int cmd, void* param, size_t paramSize, void* out, size_t* outSize)
    {
        return HalTsdrvCtlStubImpl(devId, cmd, param, paramSize, out, outSize);
    }

    static uint32_t stubStatus;
    static bool stubAckCountMismatch;
    static uint32_t stubNotReadyRounds;
    static uint32_t stubCallCount;

    static drvError_t
    HalTsdrvCtlStubImpl(uint32_t devId, int cmd, void* param, size_t paramSize, void* out, size_t* outSize)
    {
        if (stubAckCountMismatch) {
            *outSize = 0;
            return DRV_ERROR_NONE;
        }
        ts_ctrl_msg_body_t* outMsg = static_cast<ts_ctrl_msg_body_t*>(out);
        uint32_t curStatus = (stubCallCount < stubNotReadyRounds) ?
                                 static_cast<uint32_t>(APP_ABORT_STAUTS::APP_ABORT_INIT) :
                                 stubStatus;
        outMsg->u.query_task_ack_info.status = curStatus;
        *outSize = sizeof(ts_ctrl_msg_body_t);
        stubCallCount++;
        return DRV_ERROR_NONE;
    }

    void SetUpHalTsdrvCtlStub(uint32_t status, bool ackMismatch = false, uint32_t notReadyRounds = 0)
    {
        stubStatus = status;
        stubAckCountMismatch = ackMismatch;
        stubNotReadyRounds = notReadyRounds;
        stubCallCount = 0;
        MOCKER(halTsdrvCtl)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(HalTsdrvCtlStub));
    }
};

uint32_t NsRecoveryFuncLiteTest::stubStatus = 0;
bool NsRecoveryFuncLiteTest::stubAckCountMismatch = false;
uint32_t NsRecoveryFuncLiteTest::stubNotReadyRounds = 0;
uint32_t NsRecoveryFuncLiteTest::stubCallCount = 0;

TEST_F(NsRecoveryFuncLiteTest, Ut_DeviceQueryWhenDrvErrorExpectHcclEDrv)
{
    // 模拟 halTsdrvCtl 返回驱动不支持
    MockHalTsdrvCtlReturn(DRV_ERROR_NOT_SUPPORT);
    auto ret = NsRecoveryFuncLite::GetInstance().DeviceQuery(0, 0, 0);
    EXPECT_EQ(ret, HcclResult::HCCL_E_DRV);
    GlobalMockObject::verify();
}

TEST_F(NsRecoveryFuncLiteTest, Ut_DeviceQueryWhenSuccessExpectHcclSuccess)
{
    // 模拟 halTsdrvCtl 返回成功，step为0时会立即认为满足条件
    MockHalTsdrvCtlReturn(DRV_ERROR_NONE);
    auto ret = NsRecoveryFuncLite::GetInstance().DeviceQuery(0, 0, 0);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(NsRecoveryFuncLiteTest, Ut_DeviceQueryWhenTimeoutExpectHcclETimeout)
{
    // 模拟 halTsdrvCtl 返回成功，但返回的status保持小于step，设置超时时间触发超时分支
    MockHalTsdrvCtlReturn(DRV_ERROR_NONE);
    // 使用step=1，timeout设为1微秒（远小于函数内部每次循环会睡眠的5毫秒），以便快速触发超时分支
    auto ret = NsRecoveryFuncLite::GetInstance().DeviceQuery(0, 1, 1000ULL);
    EXPECT_EQ(ret, HcclResult::HCCL_E_TIMEOUT);
    GlobalMockObject::verify();
}

TEST_F(NsRecoveryFuncLiteTest, Ut_DeviceQueryWhenTimeoutZeroAndStatusMeetsExpectSuccess)
{
    SetUpHalTsdrvCtlStub(static_cast<uint32_t>(APP_ABORT_STAUTS::APP_ABORT_KILL_FINISH));
    auto ret = NsRecoveryFuncLite::GetInstance().DeviceQuery(
        0, static_cast<uint32_t>(APP_ABORT_STAUTS::APP_ABORT_KILL_FINISH), 0U);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(NsRecoveryFuncLiteTest, Ut_DeviceQueryWhenTimeoutZeroPollMultiRoundsThenMeetExpectSuccess)
{
    SetUpHalTsdrvCtlStub(static_cast<uint32_t>(APP_ABORT_STAUTS::APP_ABORT_KILL_FINISH), false, 2);
    auto ret = NsRecoveryFuncLite::GetInstance().DeviceQuery(
        0, static_cast<uint32_t>(APP_ABORT_STAUTS::APP_ABORT_KILL_FINISH), 0U);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_GE(NsRecoveryFuncLiteTest::stubCallCount, 3U);
    GlobalMockObject::verify();
}

TEST_F(NsRecoveryFuncLiteTest, Ut_DeviceQueryWhenAckCountMismatchExpectHcclEDrv)
{
    SetUpHalTsdrvCtlStub(0, true);
    auto ret = NsRecoveryFuncLite::GetInstance().DeviceQuery(0, 0, 0);
    EXPECT_EQ(ret, HcclResult::HCCL_E_DRV);
    GlobalMockObject::verify();
}

// 测试 Call() - 无通信域时不报错，覆盖 shared_lock 读路径(L25)
TEST_F(NsRecoveryFuncLiteTest, Ut_Call_When_NoComm_Expect_NoException)
{
    // Call() 内部获取 shared_lock 并调用 AicpuGetCommAll，无通信域时直接返回
    NsRecoveryFuncLite::GetInstance().Call();
    GlobalMockObject::verify();
}

// 测试 Call() - 有通信域但状态为 INVALID，覆盖 shared_lock + continue 分支
TEST_F(NsRecoveryFuncLiteTest, Ut_Call_When_CommStatusInvalid_Expect_SkipComm)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));

    CommAicpuParam commAicpuParam;
    std::string commName = "ns_recovery_test_group";
    strncpy(commAicpuParam.hcomId, commName.c_str(), HCOMID_MAX_SIZE - 1);
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().InitComm(&commAicpuParam), HCCL_SUCCESS);

    // Call() 获取 shared_lock 后遍历通信域，状态为 INVALID 则 continue
    NsRecoveryFuncLite::GetInstance().Call();

    EXPECT_EQ(CollCommAicpuMgr::GetInstance().DestroyComm(commAicpuParam.hcomId), HCCL_SUCCESS);
    GlobalMockObject::verify();
}
