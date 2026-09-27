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
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"

class CheckUbMemAttrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CheckUbMemAttrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CheckUbMemAttrTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CheckUbMemAttrTest SetUp" << std::endl; }

    virtual void TearDown() { std::cout << "A Test case in CheckUbMemAttrTest TearDown" << std::endl; }
};

// 非UB_MEM协议不校验pathMode，直接返回成功且不修改入参
TEST_F(CheckUbMemAttrTest, Ut_CheckUbMemAttr_When_NotUbMemProtocol_Expect_ReturnHCCL_SUCCESS_And_KeepPathMode)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.ubMemAttr.pathMode = 5; // 非法值，但非UB_MEM协议不应被检查

    HcommResult ret = CheckUbMemAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubMemAttr.pathMode, 5);
}

// pathMode为0xFF表示使用默认值，应被置0并返回成功
TEST_F(CheckUbMemAttrTest, Ut_CheckUbMemAttr_When_PathModeIsDefault_Expect_ReturnHCCL_SUCCESS_And_AdjustToZero)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    channelDesc.ubMemAttr.pathMode = 0xFF;

    HcommResult ret = CheckUbMemAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubMemAttr.pathMode, 0);
}

// pathMode为0应返回成功
TEST_F(CheckUbMemAttrTest, Ut_CheckUbMemAttr_When_PathModeIsZero_Expect_ReturnHCCL_SUCCESS)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    channelDesc.ubMemAttr.pathMode = 0;

    HcommResult ret = CheckUbMemAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubMemAttr.pathMode, 0);
}

// pathMode为1应返回成功
TEST_F(CheckUbMemAttrTest, Ut_CheckUbMemAttr_When_PathModeIsOne_Expect_ReturnHCCL_SUCCESS)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    channelDesc.ubMemAttr.pathMode = 1;

    HcommResult ret = CheckUbMemAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.ubMemAttr.pathMode, 1);
}

// pathMode为3(超出合法范围)应返回HCCL_E_PARA
TEST_F(CheckUbMemAttrTest, Ut_CheckUbMemAttr_When_PathModeIsThree_Expect_ReturnHCCL_E_PARA)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    channelDesc.ubMemAttr.pathMode = 3;

    HcommResult ret = CheckUbMemAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// pathMode为4(超出合法范围)应返回HCCL_E_PARA
TEST_F(CheckUbMemAttrTest, Ut_CheckUbMemAttr_When_PathModeIsFour_Expect_ReturnHCCL_E_PARA)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    channelDesc.ubMemAttr.pathMode = 4;

    HcommResult ret = CheckUbMemAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// pathMode为254(超出合法范围且非0xFF默认值)应返回HCCL_E_PARA
TEST_F(CheckUbMemAttrTest, Ut_CheckUbMemAttr_When_PathModeIsOverRange_Expect_ReturnHCCL_E_PARA)
{
    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    channelDesc.ubMemAttr.pathMode = 254;

    HcommResult ret = CheckUbMemAttr(channelDesc);
    EXPECT_EQ(ret, HCCL_E_PARA);
}
