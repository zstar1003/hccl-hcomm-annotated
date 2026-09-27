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

class HcclGetConfigTest : public BaseInit {
public:
    void SetUp() override { BaseInit::SetUp(); }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclGetConfigTest, Ut_HcclGetConfig_WhenConfigValueIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    HcclResult ret = HcclGetConfig(HCCL_DETERMINISTIC, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclGetConfigTest, Ut_HcclGetConfig_When_Normal_Expect_ReturnValueIsVaild)
{
    union HcclConfigValue hcclConfigValue;
    hcclConfigValue.value = 1;
    HcclResult ret = HcclSetConfig(HCCL_DETERMINISTIC, hcclConfigValue);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    union HcclConfigValue hcclConfigValueRet;

    ret = HcclGetConfig(HCCL_DETERMINISTIC, &hcclConfigValueRet);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hcclConfigValue.value, hcclConfigValueRet.value);

    hcclConfigValue.value = 0;
    ret = HcclSetConfig(HCCL_DETERMINISTIC, hcclConfigValue);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclGetConfig(HCCL_DETERMINISTIC, &hcclConfigValueRet);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hcclConfigValue.value, hcclConfigValueRet.value);
}

TEST_F(HcclGetConfigTest, Ut_HcclGetConfig_When_SetFailed_Expect_ReturnValueIsDETERMINISTIC_DISABLE)
{
    union HcclConfigValue hcclConfigValue;
    union HcclConfigValue hcclConfigValueRet;
    DevType deviceType = DevType::DEV_TYPE_910;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    hcclConfigValue.value = 2;
    HcclResult ret = HcclSetConfig(HCCL_DETERMINISTIC, hcclConfigValue);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    ret = HcclGetConfig(HCCL_DETERMINISTIC, &hcclConfigValueRet);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hcclConfigValueRet.value, DETERMINISTIC_DISABLE);
}
