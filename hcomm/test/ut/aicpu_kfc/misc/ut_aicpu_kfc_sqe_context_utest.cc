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

#ifndef private
#define private public
#define protected public
#endif
#include "profiling_manager_device.h"
#include "common/aicpu_kfc_def.h"
#include "aicpu_kfc/aicpu_kfc_interface.h"
#include "llt_aicpu_kfc_stub_mc2.h"
#undef private
#undef protected

using namespace std;
using namespace HcclApi;
using namespace hccl;

class MC2SqeContext_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MC2SqeContext_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "MC2SqeContext_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        g_stubDevType = DevType::DEV_TYPE_910B;
        MOCKER(halGetDeviceInfo).stubs().with(mockcpp::any()).will(invoke(StubhalGetDeviceInfo));
        std::cout << "MC2SqeContext_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        ResetMC2Context();
        GlobalMockObject::verify();
        std::cout << "MC2SqeContext_UT Test TearDown" << std::endl;
    }
};

TEST_F(MC2SqeContext_UT, TestInitSqeContext) { AicpuSqeContext::InitSqeContext(); }

TEST_F(MC2SqeContext_UT, TestSyncVariable)
{
    AicpuSqeContext::InitSqeContext();
    AicpuSqeContext::SyncVariable();
}

TEST_F(MC2SqeContext_UT, TestSaveVariable)
{
    AicpuSqeContext::InitSqeContext();
    AicpuSqeContext::SaveVariable();
}

TEST_F(MC2SqeContext_UT, TestGetNextSqeBufferAddr)
{
    AicpuSqeContext::InitSqeContext();

    GetSqeContext()->buffPtr[0].tailSqeIdx = 0;
    GetSqeContext()->buffPtr[0].tailSqeTaskId = 0;
    GetSqeContext()->buffPtr[0].sqeCnt = 0;

    uint8_t* sqeAddr;
    uint8_t* sqeTypeAddr;
    uint16_t taskId;
    EXPECT_EQ(AicpuSqeContext::GetNextSqeBufferAddr(0, sqeAddr, sqeTypeAddr, taskId), HCCL_SUCCESS);
    EXPECT_EQ(taskId, 0);
    EXPECT_EQ(AicpuSqeContext::GetNextSqeBufferAddr(0, sqeAddr, sqeTypeAddr, taskId), HCCL_SUCCESS);
    EXPECT_EQ(taskId, 1);
    EXPECT_EQ(AicpuSqeContext::GetNextSqeBufferAddr(0, sqeAddr, sqeTypeAddr, taskId), HCCL_SUCCESS);
    EXPECT_EQ(taskId, 2);
}

TEST_F(MC2SqeContext_UT, TestRecordAddInfo)
{
    AicpuSqeContext::InitSqeContext();
    GetSqeContext()->buffPtr[0].tailSqeIdx = 0;
    GetSqeContext()->buffPtr[0].tailSqeTaskId = 0;
    GetSqeContext()->buffPtr[0].sqeCnt = 0;

    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint16_t taskId = 0U;
    EXPECT_EQ(AicpuSqeContext::GetNextSqeBufferAddr(0, sqeBuffer, sqeTypeAddr, taskId), HCCL_SUCCESS);
    EXPECT_EQ(AicpuSqeContext::RecordAddInfo(0, 100), HCCL_SUCCESS);
    EXPECT_EQ(GetSqeContext()->buffPtr[0].addInfo[0], 100);
}

TEST_F(MC2SqeContext_UT, TestQuerySqeInfoByHeadNotSupported)
{
    AicpuSqeContext::InitSqeContext();

    AicpuGetComContext()->streamInfo[0].sqDepth = 2048;
    GetSqeContext()->buffPtr[0].sqeType[10] = SQE_TYPE_DEFAULT;
    GetSqeContext()->buffPtr[0].sqTail = 17;
    GetSqeContext()->buffPtr[0].tailSqeIdx = 17;

    SqeInfo sqeInfo;
    EXPECT_EQ(AicpuSqeContext::QuerySqeInfoByHead(0, 10, &sqeInfo), HCCL_E_NOT_SUPPORT);
}

TEST_F(MC2SqeContext_UT, TestQuerySqeInfoByHeadErrorInternal)
{
    AicpuSqeContext::InitSqeContext();

    AicpuGetComContext()->streamInfo[0].sqDepth = 2048;
    GetSqeContext()->buffPtr[0].sqeType[10] = SQE_TYPE_DEFAULT;
    GetSqeContext()->buffPtr[0].sqTail = 17;
    GetSqeContext()->buffPtr[0].tailSqeIdx = 1;

    SqeInfo sqeInfo;
    EXPECT_EQ(AicpuSqeContext::QuerySqeInfoByHead(0, 10, &sqeInfo), HCCL_E_INTERNAL);
}

TEST_F(MC2SqeContext_UT, TestQuerySqeInfoByHead)
{
    AicpuSqeContext::InitSqeContext();

    AicpuGetComContext()->streamInfo[0].sqDepth = 2048;
    GetSqeContext()->buffPtr[0].sqeType[10] = NOTIFY_SQE;
    GetSqeContext()->buffPtr[0].sqTail = 17;
    GetSqeContext()->buffPtr[0].tailSqeIdx = 17;

    SqeInfo sqeInfo;
    EXPECT_EQ(AicpuSqeContext::QuerySqeInfoByHead(0, 10, &sqeInfo), HCCL_SUCCESS);
    EXPECT_EQ(sqeInfo.valid, 1);
    EXPECT_EQ(sqeInfo.sqeHeadIdx, 10);
}

TEST_F(MC2SqeContext_UT, TestQuerySqeInfoByTaskIdErrorInternal)
{
    AicpuSqeContext::InitSqeContext();

    AicpuGetComContext()->streamInfo[0].sqDepth = 2048;
    GetSqeContext()->buffPtr[0].sqTail = 17;
    GetSqeContext()->buffPtr[0].tailSqeTaskId = 17;
    GetSqeContext()->buffPtr[0].tailSqeIdx = 1;
    GetSqeContext()->buffPtr[0].sqeType[7] = NOTIFY_SQE;

    SqeInfo sqeInfo;
    EXPECT_EQ(AicpuSqeContext::QuerySqeInfoByTaskId(0, 7, &sqeInfo), HCCL_E_INTERNAL);
}

TEST_F(MC2SqeContext_UT, TestQuerySqeInfoByTaskId)
{
    AicpuSqeContext::InitSqeContext();

    AicpuGetComContext()->streamInfo[0].sqDepth = 2048;
    GetSqeContext()->buffPtr[0].sqTail = 17;
    GetSqeContext()->buffPtr[0].tailSqeTaskId = 17;
    GetSqeContext()->buffPtr[0].tailSqeIdx = 17;
    GetSqeContext()->buffPtr[0].sqeType[7] = NOTIFY_SQE;

    SqeInfo sqeInfo;
    EXPECT_EQ(AicpuSqeContext::QuerySqeInfoByTaskId(0, 7, &sqeInfo), HCCL_SUCCESS);
    EXPECT_EQ(sqeInfo.valid, 1);
    EXPECT_EQ(sqeInfo.sqeHeadIdx, 7);
}

TEST_F(MC2SqeContext_UT, TestClearLocalBuff)
{
    AicpuSqeContext::InitSqeContext();

    GetSqeContext()->buffPtr[0].sqeCnt = 10;

    EXPECT_EQ(AicpuSqeContext::ClearLocalBuff(), HCCL_SUCCESS);
    EXPECT_EQ(GetSqeContext()->buffPtr[0].sqeCnt, 0);
}

TEST_F(MC2SqeContext_UT, TestGetString)
{
    SqeInfo sqeInfo;
    sqeInfo.sqeHeadIdx = 10;
    std::string str = AicpuSqeContext::GetString(sqeInfo);
    EXPECT_TRUE(str.find("sqeIdx:10") != std::string::npos);
}

TEST_F(MC2SqeContext_UT, TestSqeTaskIdOverFlow)
{
    MOCKER(AdprofCheckFeatureIsOn).stubs().will(returnValue(1));
    bool isL1On = dfx::ProfilingManager::IsProfL1On();
    bool isL0On = dfx::ProfilingManager::IsProfL0On();
    EXPECT_EQ(isL1On, true);
    EXPECT_EQ(isL0On, true);

    AicpuSqeContext::InitSqeContext();
    GetSqeContext()->buffPtr[0].tailSqeIdx = 0;
    GetSqeContext()->buffPtr[0].tailSqeTaskId = UINT16_MAX;
    GetSqeContext()->buffPtr[0].sqeCnt = 0;

    uint8_t* sqeAddr;
    uint8_t* sqeTypeAddr;
    uint16_t taskId;
    EXPECT_EQ(AicpuSqeContext::GetNextSqeBufferAddr(0, sqeAddr, sqeTypeAddr, taskId), HCCL_SUCCESS);
    EXPECT_EQ(taskId, UINT16_MAX);
    EXPECT_EQ(AicpuSqeContext::GetNextSqeBufferAddr(0, sqeAddr, sqeTypeAddr, taskId), HCCL_SUCCESS);
    EXPECT_EQ(taskId, 1);
}

TEST_F(MC2SqeContext_UT, TestSqeBuffOverFlow)
{
    AicpuSqeContext::InitSqeContext();

    GetSqeContext()->buffPtr[0].tailSqeIdx = 2048;
    GetSqeContext()->buffPtr[0].tailSqeTaskId = 3096;
    GetSqeContext()->buffPtr[0].sqeCnt = 10;

    uint8_t* sqeAddr;
    uint8_t* sqeTypeAddr;
    uint16_t taskId;
    EXPECT_EQ(AicpuSqeContext::GetNextSqeBufferAddr(0, sqeAddr, sqeTypeAddr, taskId), HCCL_SUCCESS);
    EXPECT_EQ(taskId, 3096);
    EXPECT_EQ(AicpuSqeContext::GetNextSqeBufferAddr(0, sqeAddr, sqeTypeAddr, taskId), HCCL_SUCCESS);
    EXPECT_EQ(taskId, 3097);
    EXPECT_EQ(AicpuSqeContext::GetNextSqeBufferAddr(0, sqeAddr, sqeTypeAddr, taskId), HCCL_SUCCESS);
    EXPECT_EQ(taskId, 3098);
}
