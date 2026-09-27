/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "hccl/hccl_channel.h"

class TestHcclChannelConfig : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TestHcclChannelConfig tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "TestHcclChannelConfig tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in TestHcclChannelConfig SetUp" << std::endl; }

    virtual void TearDown() { std::cout << "A Test case in TestHcclChannelConfig TearDown" << std::endl; }
};

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigCreate_When_ConfigPtrNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = HcclChannelConfigCreate(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigCreate_When_Normal_Expect_Success_And_HandleValid)
{
    HcclChannelConfig config = nullptr;
    HcclResult ret = HcclChannelConfigCreate(&config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(config, nullptr);
    HcclChannelConfigDestroy(config);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigDestroy_When_ConfigNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = HcclChannelConfigDestroy(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigDestroy_When_Normal_Expect_Success)
{
    HcclChannelConfig config = nullptr;
    ASSERT_EQ(HcclChannelConfigCreate(&config), HCCL_SUCCESS);
    HcclResult ret = HcclChannelConfigDestroy(config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigSetInt_When_ConfigNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = HcclChannelConfigSetInt(nullptr, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigSetInt_When_SharedQueueTrue_Expect_Success)
{
    HcclChannelConfig config = nullptr;
    ASSERT_EQ(HcclChannelConfigCreate(&config), HCCL_SUCCESS);
    HcclResult ret = HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclChannelConfigDestroy(config);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigSetInt_When_InvalidType_Expect_HCCL_E_PARA)
{
    HcclChannelConfig config = nullptr;
    ASSERT_EQ(HcclChannelConfigCreate(&config), HCCL_SUCCESS);
    HcclResult ret = HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_INVALID, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
    HcclChannelConfigDestroy(config);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigSetStr_When_ConfigNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = HcclChannelConfigSetStr(nullptr, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, "tag");
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigSetStr_When_ValueNullptr_Return_HCCL_E_PTR)
{
    HcclChannelConfig config = nullptr;
    ASSERT_EQ(HcclChannelConfigCreate(&config), HCCL_SUCCESS);
    HcclResult ret = HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    HcclChannelConfigDestroy(config);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigSetStr_When_TagEmpty_Expect_HCCL_E_PARA)
{
    HcclChannelConfig config = nullptr;
    ASSERT_EQ(HcclChannelConfigCreate(&config), HCCL_SUCCESS);
    HcclResult ret = HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, "");
    EXPECT_EQ(ret, HCCL_E_PARA);
    HcclChannelConfigDestroy(config);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigSetStr_When_TagValid_Expect_Success)
{
    HcclChannelConfig config = nullptr;
    ASSERT_EQ(HcclChannelConfigCreate(&config), HCCL_SUCCESS);
    HcclResult ret = HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, "sharedTag1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclChannelConfigDestroy(config);
}

TEST_F(TestHcclChannelConfig, Ut_HcclChannelConfigSetStr_When_InvalidType_Expect_HCCL_E_PARA)
{
    HcclChannelConfig config = nullptr;
    ASSERT_EQ(HcclChannelConfigCreate(&config), HCCL_SUCCESS);
    HcclResult ret = HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_INVALID, "tag");
    EXPECT_EQ(ret, HCCL_E_PARA);
    HcclChannelConfigDestroy(config);
}
