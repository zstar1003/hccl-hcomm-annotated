/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include <string>

static int g_count = 0;

class HcclCommRegCommStateCallbackTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_1SERVER_1RANK_AS_DEFAULT;
    }

    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

HcclResult ClearCallBack(HcclComm comm, HcclCommStatePhase phase, void* args)
{
    (void)comm;
    (void)phase;
    (void)args;
    return HCCL_SUCCESS;
}

HcclResult CountCallBack(HcclComm comm, HcclCommStatePhase phase, void* args)
{
    (void)comm;
    (void)phase;
    int* count = static_cast<int*>(args);
    (*count)++;
    return HCCL_SUCCESS;
}

TEST_F(
    HcclCommRegCommStateCallbackTest, Ut_HcclCommRegCommStateCallbackTest_When_ArgsIsNull_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);

    std::string regName = "regName";

    HcclResult ret = HcclCommRegCommStateCallback(regName.c_str(), ClearCallBack, nullptr);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    Ut_Comm_Destroy(comm);
}

TEST_F(
    HcclCommRegCommStateCallbackTest, Ut_HcclCommRegCommStateCallbackTest_When_CallbackIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);
    char* args = "userPtr";
    std::string regName = "regName";

    HcclResult ret = HcclCommRegCommStateCallback(regName.c_str(), nullptr, args);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(
    HcclCommRegCommStateCallbackTest, Ut_HcclCommRegCommStateCallbackTest_When_RegNameIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);
    char* args = "userPtr";

    HcclResult ret = HcclCommRegCommStateCallback(nullptr, ClearCallBack, args);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommRegCommStateCallbackTest, Ut_HcclCommRegCommStateCallbackTest_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    char* args = "userPtr";
    std::string regName = "regName";

    HcclResult ret1 = HcclCommRegCommStateCallback(regName.c_str(), ClearCallBack, args);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    HcclResult ret2 = HcclCommResumePostCallback(comm);
    EXPECT_EQ(ret2, HCCL_SUCCESS);

    Ut_Comm_Destroy(comm);
}

TEST_F(
    HcclCommRegCommStateCallbackTest, Ut_HcclCommRegCommStateCallbackTest_When_ArgsIsValid_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    std::string regName = "regName";

    HcclResult ret1 = HcclCommRegCommStateCallback(regName.c_str(), CountCallBack, &g_count);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    HcclResult ret2 = HcclCommResumePostCallback(comm);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_EQ(g_count, 1);

    Ut_Comm_Destroy(comm);
}
