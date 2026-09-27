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

class HcclCommDeactivateCommMemoryTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_1SERVER_1RANK_AS_DEFAULT;
        // MOCK掉对communicator层的依赖，保证分层测试
        MOCKER_CPP(&HcclCommunicator::DeactivateCommMemory)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclCommDeactivateCommMemoryTest, Ut_HcclCommDeactivateCommMemory_When_CommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    Ut_Device_Set(0);
    void* baseVirPtr = sal_malloc(10);

    HcclResult ret = HcclCommDeactivateCommMemory(comm, baseVirPtr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    sal_free(baseVirPtr);
}

TEST_F(
    HcclCommDeactivateCommMemoryTest, Ut_HcclCommDeactivateCommMemory_When_BaseVirPtrIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);
    void* baseVirPtr = nullptr;

    HcclResult ret = HcclCommDeactivateCommMemory(comm, baseVirPtr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommDeactivateCommMemoryTest, Ut_HcclCommDeactivateCommMemory_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    void* baseVirPtr = sal_malloc(10);

    HcclResult ret = HcclCommDeactivateCommMemory(comm, baseVirPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(baseVirPtr);
    Ut_Comm_Destroy(comm);
}
