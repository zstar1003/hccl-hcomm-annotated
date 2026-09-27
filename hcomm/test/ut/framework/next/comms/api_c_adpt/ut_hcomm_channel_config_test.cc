/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "hccl/hccl_types.h"
#include "hcomm_channel.h"

class TestHcommChannelConfig : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TestHcommChannelConfig tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "TestHcommChannelConfig tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in TestHcommChannelConfig SetUp" << std::endl; }

    virtual void TearDown() { std::cout << "A Test case in TestHcommChannelConfig TearDown" << std::endl; }
};

TEST_F(TestHcommChannelConfig, Ut_HcommChannelConfigCreate_When_ConfigPtrNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommChannelConfigCreate(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannelConfig, Ut_HcommChannelConfigCreate_When_Normal_Expect_Success_And_HandleValid)
{
    HcommChannelConfig config = nullptr;
    HcommResult ret = HcommChannelConfigCreate(&config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(config, nullptr);
    HcommChannelConfigDestroy(config);
}

TEST_F(TestHcommChannelConfig, Ut_HcommChannelConfigDestroy_When_ConfigNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommChannelConfigDestroy(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannelConfig, Ut_HcommChannelConfigDestroy_When_Normal_Expect_Success)
{
    HcommChannelConfig config = nullptr;
    ASSERT_EQ(HcommChannelConfigCreate(&config), HCCL_SUCCESS);
    HcommResult ret = HcommChannelConfigDestroy(config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcommChannelConfig, Ut_HcommChannelConfigSetInt_When_ConfigNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommChannelConfigSetInt(nullptr, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannelConfig, Ut_HcommChannelConfigSetInt_When_SharedQueueTrue_Expect_Success)
{
    HcommChannelConfig config = nullptr;
    ASSERT_EQ(HcommChannelConfigCreate(&config), HCCL_SUCCESS);
    HcommResult ret = HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcommChannelConfigDestroy(config);
}

TEST_F(TestHcommChannelConfig, Ut_HcommChannelConfigSetInt_When_InvalidType_Expect_HCCL_E_PARA)
{
    HcommChannelConfig config = nullptr;
    ASSERT_EQ(HcommChannelConfigCreate(&config), HCCL_SUCCESS);
    HcommResult ret = HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_INVALID, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
    HcommChannelConfigDestroy(config);
}
