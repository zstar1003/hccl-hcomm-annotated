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

class HcclCommResumeTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_1SERVER_1RANK_AS_DEFAULT;
        // MOCK掉对communicator层的依赖，保证分层测试
        HcclCommunicator commun_mock;
        MOCKER_CPP_VIRTUAL(commun_mock, &HcclCommunicator::Resume)
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

TEST_F(HcclCommResumeTest, Ut_HcclCommResume_When_CommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    Ut_Device_Set(0);

    HcclResult ret = HcclCommResume(comm);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommResumeTest, Ut_HcclCommResume_When_CommIsOk_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);

    HcclResult ret = HcclCommResume(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Ut_Comm_Destroy(comm);
}

class HcclCommResumeRealLogTest : public BaseInit {
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

TEST_F(HcclCommResumeRealLogTest, Ut_HcclCommResume_RealCall_CoverBeginLog)
{
    UT_COMM_CREATE_DEFAULT(comm);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclCommunicator* communicator = hcclComm->GetHcclCommunicator();
    ASSERT_NE(communicator, nullptr);
    communicator->identifier_ = "ut_test_comm";

    HcclResult ret = hcclComm->Resume();
    (void)ret;

    Ut_Comm_Destroy(comm);
}
