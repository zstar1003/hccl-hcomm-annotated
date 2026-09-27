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

class HcclCommWorkingDevNicSetTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_1SERVER_1RANK_AS_DEFAULT;
        // MOCK掉对communicator层的依赖，保证分层测试
        MOCKER_CPP(&HcclCommunicator::SwitchNic, HcclResult(HcclCommunicator::*)(uint32_t, uint32_t*, bool*))
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

TEST_F(HcclCommWorkingDevNicSetTest, Ut_HcclCommWorkingDevNicSet_When_CommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    Ut_Device_Set(0);
    uint32_t ranks[4] = {0, 1, 2, 3};
    bool useBackup[4] = {true, true, true, true};
    uint32_t nRanks = 4;

    HcclResult ret = HcclCommWorkingDevNicSet(comm, ranks, useBackup, nRanks);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommWorkingDevNicSetTest, Ut_HcclCommWorkingDevNicSet_When_RanksIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);
    uint32_t* pRanks = nullptr;
    bool useBackup[4] = {true, true, true, true};
    uint32_t nRanks = 4;

    HcclResult ret = HcclCommWorkingDevNicSet(comm, pRanks, useBackup, nRanks);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommWorkingDevNicSetTest, Ut_HcclCommWorkingDevNicSet_When_Normal_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);
    uint32_t ranks[4] = {0, 1, 2, 3};
    bool* pUseBackup = nullptr;
    uint32_t nRanks = 4;

    HcclResult ret = HcclCommWorkingDevNicSet(comm, ranks, pUseBackup, nRanks);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommWorkingDevNicSetTest, Ut_HcclCommWorkingDevNicSet_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    uint32_t ranks[4] = {0, 1, 2, 3};
    bool useBackup[4] = {true, true, true, true};
    uint32_t nRanks = 4;

    HcclResult ret = HcclCommWorkingDevNicSet(comm, ranks, useBackup, nRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Ut_Comm_Destroy(comm);
}
