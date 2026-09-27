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

class HcclGetCommAsyncErrorTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_1SERVER_2RANK_AS_DEFAULT;
        // MOCK掉对communicator层的依赖，保证分层测试
        HcclCommunicator commun_mock;
        MOCKER_CPP_VIRTUAL(commun_mock, &HcclCommunicator::GetCqeError)
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

TEST_F(HcclGetCommAsyncErrorTest, Ut_HcclGetCommAsyncError_When_CommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    Ut_Device_Set(0);

    HcclResult asyncError;

    HcclResult ret = HcclGetCommAsyncError(comm, &asyncError);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
