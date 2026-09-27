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

class HcclCommGetStatusTest : public BaseInit {
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

    std::string GetCommId(HcclComm comm)
    {
        char commId[256] = {0};
        HcclResult ret = HcclGetCommName(comm, commId);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        return std::string(commId);
    }
};

TEST_F(HcclCommGetStatusTest, Ut_HcclCommGetStatus_When_StatusIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);

    std::string commId = GetCommId(comm);

    HcclResult ret = HcclCommGetStatus(commId.c_str(), nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommGetStatusTest, Ut_HcclCommGetStatus_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);

    MOCKER_CPP(&hcclComm::GetCommStatus).stubs().will(returnValue(HCCL_SUCCESS));

    std::string commId = GetCommId(comm);

    HcclCommStatus status = HCCL_COMM_STATUS_INVALID;
    HcclResult ret = HcclCommGetStatus(commId.c_str(), &status);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommGetStatusTest, Ut_HcclCommGetStatus_When_DevTypeIsNot950_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    UT_COMM_CREATE_DEFAULT(comm);

    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));

    std::string commId = GetCommId(comm);

    HcclCommStatus status = HCCL_COMM_STATUS_INVALID;
    HcclResult ret = HcclCommGetStatus(commId.c_str(), &status);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    Ut_Comm_Destroy(comm);
}
