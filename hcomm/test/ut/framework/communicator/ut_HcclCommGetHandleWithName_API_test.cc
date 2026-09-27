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

class HcclCommGetHandleWithNameTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_RANK_TABLE_910_1SERVER_1RANK;
        UT_COMM_CREATE_DEFAULT(comm);
    }
    void TearDown() override
    {
        Ut_Comm_Destroy(comm);
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclCommGetHandleWithNameTest, HcclGetCommName_When_ParamIsNullptr_Expect_ReturnIsHCCL_E_PTR)
{
    const char* commName = nullptr;

    HcclResult ret = HcclCommGetHandleWithName(commName, &comm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommGetHandleWithNameTest, Ut_HcclCommGetHandleWithName_When_CommNameIsNotExist_Expect_ReturnIsHCCL_E_PARA)
{
    const char commName[128] = "test";

    HcclResult ret = HcclCommGetHandleWithName(commName, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclCommGetHandleWithNameTest, Ut_HcclCommGetHandleWithName_When_CommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    const char commName[128] = "test";

    HcclResult ret = HcclCommGetHandleWithName(commName, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
