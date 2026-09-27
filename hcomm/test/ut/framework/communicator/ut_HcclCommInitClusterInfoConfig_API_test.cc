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

class HcclCommInitClusterInfoConfigTest : public BaseInit {
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

TEST_F(HcclCommInitClusterInfoConfigTest, Ut_HcclCommInitClusterInfoConfig_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    Ut_Device_Set(0);
    const char* rankTableFile = rankTableFileName;
    u32 rankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 400;

    HcclResult ret = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Ut_Comm_Destroy(comm);
}

TEST_F(
    HcclCommInitClusterInfoConfigTest,
    Ut_HcclCommInitClusterInfoConfig_When_ConfigBufferSizeIsZero_Expect_ReturnIsHCCL_SUCCESS)
{
    Ut_Device_Set(0);
    const char* rankTableFile = rankTableFileName;
    u32 rankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 0;

    HcclResult ret = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(comm);
}

TEST_F(
    HcclCommInitClusterInfoConfigTest,
    Ut_HcclCommInitClusterInfoConfig_When_clusterInfoFileNotExist_Expect_ReturnIsHCCL_E_INTERNAL)
{
    Ut_Device_Set(0);
    const char* rankTableFile = "fake.json";
    u32 rankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 400;

    HcclResult ret = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    Ut_Comm_Destroy(comm);
}

TEST_F(
    HcclCommInitClusterInfoConfigTest,
    Ut_HcclCommInitClusterInfoConfig_When_clusterInfoIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    Ut_Device_Set(0);
    const char* rankTableFile = nullptr;
    u32 rankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 400;

    HcclResult ret = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(
    HcclCommInitClusterInfoConfigTest,
    Ut_HcclCommInitClusterInfoConfig_When_GetDeviceError_Expect_ReturnIsHCCL_E_RUNTIME)
{
    MOCKER(aclrtGetDevice).stubs().will(returnValue(ACL_ERROR_RT_CONTEXT_NULL));
    const char* rankTableFile = rankTableFileName;
    u32 rankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 400;

    HcclResult ret = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);

    GlobalMockObject::verify();
    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitClusterInfoConfigTest, Ut_HcclCommInitClusterInfoConfig_When_commIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    Ut_Device_Set(0);
    const char* rankTableFile = rankTableFileName;
    u32 rankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 400;
    void** pComm = nullptr;

    HcclResult ret = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &commConfig, pComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommInitClusterInfoConfigTest, Ut_HcclCommInitClusterInfoConfig_When_MultiInit_Expect_ReturnIsHCCL_E_UNAVAIL)
{
    Ut_Device_Set(0);
    const char* rankTableFile = rankTableFileName;
    u32 rankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 400;
    HcclComm comm1 = nullptr;
    HcclComm comm2 = nullptr;

    HcclResult ret1 = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &commConfig, &comm1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    HcclResult ret2 = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &commConfig, &comm2);
    EXPECT_EQ(ret2, HCCL_E_UNAVAIL);

    Ut_Comm_Destroy(comm1);
    Ut_Comm_Destroy(comm2);
}
