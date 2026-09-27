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

class HcclGetCommNameTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_RANK_TABLE_910_1SERVER_1RANK;
        // 将enableEntryLog默认返回为true
        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclGetCommNameTest, Ut_HcclGetCommName_When_CommNameIsNull_Expect_ReturnIsHCCL_E_PTR) {}

TEST_F(HcclGetCommNameTest, Ut_HcclGetCommName_When_CommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    char* commName = new char[ROOTINFO_INDENTIFIER_MAX_LENGTH];

    HcclResult ret = HcclGetCommName(comm, commName);
    EXPECT_EQ(ret, HCCL_E_PTR);

    delete[] commName;
}

TEST_F(HcclGetCommNameTest, HcclGetCommName_When_InputNoInit_Expect_ReturnIsHCCL_E_PTR)
{
    char* commName = nullptr;

    HcclResult ret = HcclGetCommName(&comm, commName);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclGetCommNameTest, HcclGetCommName_When_Normal_Expect_ReturnIsHCCL_SUCCESS) {}
