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

class HcclCommDestroyTest : public BaseInit {
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

TEST_F(HcclCommDestroyTest, Ut_HcclCommDestroy_When_CommIsNull_Expect_ReturnIsHCCL_SUCCESS)
{
    Ut_Device_Set(0);

    HcclResult ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommDestroyTest, Ut_HcclCommDestroy_When_GetDeviceError_Expect_ReturnIsHCCL_E_INTERNAL)
{
    MOCKER(aclrtGetDevice).stubs().will(returnValue(ACL_ERROR_RT_CONTEXT_NULL));

    HcclResult ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(HcclCommDestroyTest, Ut_HcclCommDestroy_When_InitSuccess_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);

    HcclResult ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
