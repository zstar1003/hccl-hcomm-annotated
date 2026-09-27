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

class HcclCreateSubCommConfigTest : public BaseInit {
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

TEST_F(HcclCreateSubCommConfigTest, Ut_HcclCreateSubCommConfig_When_RankNumIsZero_Expect_ReturnIsHCCL_E_PARA)
{
    UT_COMM_CREATE_DEFAULT(comm);
    int rankNum = 0;
    uint32_t rankIds[1] = {0};
    u32 subCommId = 2;
    u32 subCommRankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    HcclComm subComm = nullptr;

    HcclResult ret = HcclCreateSubCommConfig(&comm, rankNum, rankIds, subCommId, subCommRankId, &commConfig, &subComm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(subComm);
    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCreateSubCommConfigTest, Ut_HcclCreateSubCommConfig_When_RankIdsIsNull_Expect_ReturnIsHCCL_E_PARA)
{
    UT_COMM_CREATE_DEFAULT(comm);
    int rankNum = 1;
    uint32_t* rankIds = nullptr;
    u32 subCommId = 2;
    u32 subCommRankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    HcclComm subComm = nullptr;

    HcclResult ret = HcclCreateSubCommConfig(&comm, rankNum, rankIds, subCommId, subCommRankId, &commConfig, &subComm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(subComm);
    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCreateSubCommConfigTest, Ut_HcclCreateSubCommConfig_When_subCommIdIsInvalid_Expect_ReturnIsHCCL_E_PARA)
{
    UT_COMM_CREATE_DEFAULT(comm);
    int rankNum = 1;
    uint32_t rankIds[1] = {0};
    u32 subCommId = 0xFFFFFFFF;
    u32 subCommRankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    HcclComm subComm = nullptr;

    HcclResult ret = HcclCreateSubCommConfig(&comm, rankNum, rankIds, subCommId, subCommRankId, &commConfig, &subComm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(subComm);
    Ut_Comm_Destroy(comm);
}

TEST_F(
    HcclCreateSubCommConfigTest,
    Ut_HcclCreateSubCommConfig_When_RankIdsIsNullAndSubCommIdIsInvalid_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    int rankNum = 1;
    uint32_t* rankIds = nullptr;
    u32 subCommId = 0xFFFFFFFF;
    u32 subCommRankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    HcclComm subComm = nullptr;

    HcclResult ret = HcclCreateSubCommConfig(&comm, rankNum, rankIds, subCommId, subCommRankId, &commConfig, &subComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Ut_Comm_Destroy(subComm);
    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCreateSubCommConfigTest, Ut_HcclCreateSubCommConfig_When_CommConfigIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);
    int rankNum = 1;
    uint32_t rankIds[1] = {0};
    u32 subCommId = 0;
    u32 subCommRankId = 0;
    HcclCommConfig* pCommConfig = nullptr;
    HcclComm subComm = nullptr;

    HcclResult ret = HcclCreateSubCommConfig(&comm, rankNum, rankIds, subCommId, subCommRankId, pCommConfig, &subComm);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(subComm);
    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCreateSubCommConfigTest, Ut_HcclCreateSubCommConfig_When_SubCommRankId_GE_RankNum_Expect_ReturnIsHCCL_E_PARA)
{
    UT_COMM_CREATE_DEFAULT(comm);
    int rankNum = 1;
    uint32_t rankIds[2] = {0, 1};
    u32 subCommId = 2;
    u32 subCommRankId = 2;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    HcclComm subComm = nullptr;

    HcclResult ret = HcclCreateSubCommConfig(&comm, rankNum, rankIds, subCommId, subCommRankId, &commConfig, &subComm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(subComm);
    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCreateSubCommConfigTest, Ut_HcclCreateSubCommConfig_When_CommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);
    int rankNum = 1;
    uint32_t rankIds[1] = {0};
    u32 subCommId = 2;
    u32 subCommRankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    void** pSubComm = nullptr;

    HcclResult ret = HcclCreateSubCommConfig(&comm, rankNum, rankIds, subCommId, subCommRankId, &commConfig, pSubComm);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCreateSubCommConfigTest, Ut_HcclCreateSubCommConfig_When_GetDeviceError_Expect_ReturnIsHCCL_E_RUNTIME)
{
    UT_COMM_CREATE_DEFAULT(comm);
    int rankNum = 1;
    uint32_t rankIds[1] = {0};
    u32 subCommId = 2;
    u32 subCommRankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    HcclComm subComm = nullptr;
    MOCKER(aclrtGetDevice).stubs().will(returnValue(ACL_ERROR_RT_CONTEXT_NULL));

    HcclResult ret = HcclCreateSubCommConfig(&comm, rankNum, rankIds, subCommId, subCommRankId, &commConfig, &subComm);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);

    GlobalMockObject::verify();

    Ut_Comm_Destroy(subComm);
    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCreateSubCommConfigTest, Ut_HcclCreateSubCommConfig_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    int rankNum = 1;
    uint32_t rankIds[1] = {0};
    u32 subCommId = 2;
    u32 subCommRankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    HcclComm subComm = nullptr;

    HcclResult ret = HcclCreateSubCommConfig(&comm, rankNum, rankIds, subCommId, subCommRankId, &commConfig, &subComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Ut_Comm_Destroy(subComm);
    Ut_Comm_Destroy(comm);
}
