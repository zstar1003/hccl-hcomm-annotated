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

class HcclGetErrorStringTest : public testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_SUCCESS_Expect_ReturnIsNoError)
{
    HcclResult code = HCCL_SUCCESS;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "no error"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_PARA_Expect_ReturnIsParameterError)
{
    HcclResult code = HCCL_E_PARA;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "parameter error"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_PTR_Expect_ReturnIsEmptyPointer)
{
    HcclResult code = HCCL_E_PTR;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "empty pointer"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_MEMORY_Expect_ReturnIsMemoryError)
{
    HcclResult code = HCCL_E_MEMORY;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "memory error"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_INTERNAL_Expect_ReturnIsInternalError)
{
    HcclResult code = HCCL_E_INTERNAL;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "internal error"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_NOT_SUPPORT_Expect_ReturnIsNotSupportFeature)
{
    HcclResult code = HCCL_E_NOT_SUPPORT;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "not support feature"), 0);
}

TEST_F(
    HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_NOT_FOUND_Expect_ReturnIsNotFoundSpecificResource)
{
    HcclResult code = HCCL_E_NOT_FOUND;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "not found specific resource"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_UNAVAIL_Expect_ReturnIsResourceUnavailable)
{
    HcclResult code = HCCL_E_UNAVAIL;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "resource unavailable"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_SYSCALL_Expect_ReturnIsCallSystemInterfaceError)
{
    HcclResult code = HCCL_E_SYSCALL;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "call system interface error"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_TIMEOUT_Expect_ReturnIsTimeout)
{
    HcclResult code = HCCL_E_TIMEOUT;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "timeout"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_OPEN_FILE_FAILURE_Expect_ReturnIsOpenFileFail)
{
    HcclResult code = HCCL_E_OPEN_FILE_FAILURE;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "open file fail"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_TCP_CONNECT_Expect_ReturnIsTcpConnectFail)
{
    HcclResult code = HCCL_E_TCP_CONNECT;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "tcp connect fail"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_ROCE_CONNECT_Expect_ReturnIsRoceConnectFail)
{
    HcclResult code = HCCL_E_ROCE_CONNECT;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "roce connect fail"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_TCP_TRANSFER_Expect_ReturnIsTcpTransferFail)
{
    HcclResult code = HCCL_E_TCP_TRANSFER;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "tcp transfer fail"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_ROCE_TRANSFER_Expect_ReturnIsRoceTransferFail)
{
    HcclResult code = HCCL_E_ROCE_TRANSFER;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "roce transfer fail"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_RUNTIME_Expect_ReturnIsCallRuntimeApiFail)
{
    HcclResult code = HCCL_E_RUNTIME;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "call runtime api fail"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_DRV_Expect_ReturnIsCallDriverApiFail)
{
    HcclResult code = HCCL_E_DRV;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "call driver api fail"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_PROFILING_Expect_ReturnIsCallProfilingApiFail)
{
    HcclResult code = HCCL_E_PROFILING;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "call profiling api fail"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_CCE_Expect_ReturnIsCallCceApiFail)
{
    HcclResult code = HCCL_E_CCE;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "call cce api fail"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_NETWORK_Expect_ReturnIsCallNetworkApiFail)
{
    HcclResult code = HCCL_E_NETWORK;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "call network api fail"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_AGAIN_Expect_ReturnIsTryAgain)
{
    HcclResult code = HCCL_E_AGAIN;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "try again"), 0);
}

TEST_F(HcclGetErrorStringTest, Ut_HcclGetErrorString_When_InputIsHCCL_E_REMOTE_Expect_ReturnIsErrorCqe)
{
    HcclResult code = HCCL_E_REMOTE;

    const char* ret = HcclGetErrorString(code);
    EXPECT_EQ(strcmp(ret, "error cqe"), 0);
}
