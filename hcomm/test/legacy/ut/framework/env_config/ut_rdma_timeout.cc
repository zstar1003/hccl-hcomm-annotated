/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <iostream>

#define private public
#define protected public
#include "base_config_legacy.h"
#include "env_func.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

class RdmaTimeoutTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "RdmaTimeoutTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "RdmaTimeoutTest TearDown" << endl; }

    virtual void SetUp() { cout << "A Test case in RdmaTimeoutTest SetUp" << endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        cout << "A Test case in RdmaTimeoutTest TearDown" << endl;
    }
};

// CheckRdmaTimeout 是空实现，测试该函数不抛出异常
TEST_F(RdmaTimeoutTest, Ut_CheckRdmaTimeout_When_ValueIsZero_Expect_NoThrow)
{
    u32 timeout = 0;
    EXPECT_NO_THROW(CheckRdmaTimeout(timeout));
}

// =0最小值时不做修改
TEST_F(RdmaTimeoutTest, Ut_ProcRdmaTimeout_When_ValueIsMin_Expect_NoChange)
{
    u32 timeout = EnvRdmaConfig::HCCL_RDMA_TIMEOUT_MIN; // 0
    ProcRdmaTimeout(timeout);
    EXPECT_EQ(timeout, EnvRdmaConfig::HCCL_RDMA_TIMEOUT_MIN);
}

// =31最大值时不做修改
TEST_F(RdmaTimeoutTest, Ut_ProcRdmaTimeout_When_ValueIsMax_Expect_NoChange)
{
    u32 timeout = EnvRdmaConfig::HCCL_RDMA_TIMEOUT_MAX; // 31
    ProcRdmaTimeout(timeout);
    EXPECT_EQ(timeout, EnvRdmaConfig::HCCL_RDMA_TIMEOUT_MAX);
}

// =20默认值时不做修改
TEST_F(RdmaTimeoutTest, Ut_ProcRdmaTimeout_When_ValueIsWithinRange_Expect_NoChange)
{
    u32 timeout = 20;
    ProcRdmaTimeout(timeout);
    EXPECT_EQ(timeout, 20);
}

// >=31时设置为0
TEST_F(RdmaTimeoutTest, Ut_ProcRdmaTimeout_When_ValueExceedsMax_Expect_SetToZero)
{
    u32 timeout = 32;
    ProcRdmaTimeout(timeout);
    EXPECT_EQ(timeout, EnvRdmaConfig::HCCL_RDMA_TIMEOUT_MIN);
}

TEST_F(RdmaTimeoutTest, Ut_RdmaTimeOut_When_ValueWithinRange_Expect_ParsedCorrectly)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("15")));
    rdmaCfg.rdmaTimeOut.Parse();
    EXPECT_EQ(rdmaCfg.rdmaTimeOut.Get(), 15);
}

TEST_F(RdmaTimeoutTest, Ut_RdmaTimeOut_When_ValueIsZero_Expect_ParsedCorrectly)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("0")));
    rdmaCfg.rdmaTimeOut.Parse();
    EXPECT_EQ(rdmaCfg.rdmaTimeOut.Get(), 0);
}

TEST_F(RdmaTimeoutTest, Ut_RdmaTimeOut_When_ValueIsMax_Expect_ParsedCorrectly)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("31")));
    rdmaCfg.rdmaTimeOut.Parse();
    EXPECT_EQ(rdmaCfg.rdmaTimeOut.Get(), 31);
}

// >=31时设置为0
TEST_F(RdmaTimeoutTest, Ut_RdmaTimeOut_When_ValueExceedsMax_Expect_SetToZero)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("32")));
    rdmaCfg.rdmaTimeOut.Parse();
    EXPECT_EQ(rdmaCfg.rdmaTimeOut.Get(), 0);
}

// 默认值场景
TEST_F(RdmaTimeoutTest, Ut_RdmaTimeOut_When_DefaultValue_Expect_ParsedCorrectly)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    rdmaCfg.rdmaTimeOut.Parse();
    EXPECT_EQ(rdmaCfg.rdmaTimeOut.Get(), 20);
}
