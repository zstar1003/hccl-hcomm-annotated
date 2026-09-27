/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "aicpu/dev_rdma_connection_v2.h"
#include "socket.h"
#include "orion_adapter_hccp.h"
#include "hccp.h"
#include "hccp_common.h"
#define private public

using namespace hcomm;

class DevRdmaConnectionV2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DevRdmaConnectionV2Test tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "DevRdmaConnectionV2Test tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in DevRdmaConnectionV2Test SetUP" << std::endl;

        Hccl::IpAddress localIp("1.0.0.0");
        Hccl::IpAddress remoteIp("2.0.0.0");
        fakeSocket = new Hccl::Socket(
            nullptr, localIp, 60001, remoteIp, "_0_1_", Hccl::SocketRole::SERVER, Hccl::NicType::HOST_NIC_TYPE);

        MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
        MOCKER(Hccl::HrtRaNdaQpCreate)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        MOCKER(Hccl::HrtRaNdaCqCreate)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        MOCKER(Hccl::HrtRaNdaCqDestroy).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(Hccl::HrtRaQpDestroy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(RaGetQpAttr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete fakeSocket;
        std::cout << "A Test case in DevRdmaConnectionV2Test TearDown" << std::endl;
    }

    Hccl::Socket* fakeSocket;
};

TEST_F(DevRdmaConnectionV2Test, Ut_When_Normal_Call_Expect_Status_Consistent)
{
    std::cout << "Start Ut_When_Normal_Call_Expect_Status_Consistent" << std::endl;

    RdmaHandle rdmaHandle = (void*)0x1000000;
    DevRdmaConnectionV2 connection(fakeSocket, rdmaHandle, 0);

    EXPECT_EQ(connection.rdmaConnStatus_, DevRdmaConnectionV2::RdmaConnStatus::CLOSED);

    HcclResult ret = connection.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(connection.rdmaConnStatus_, DevRdmaConnectionV2::RdmaConnStatus::INIT);

    ret = connection.CreateQp();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(connection.rdmaConnStatus_, DevRdmaConnectionV2::RdmaConnStatus::QP_CREATED);

    std::unique_ptr<Hccl::Serializable> locDto;
    ret = connection.GetExchangeDto(locDto);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(locDto, nullptr);

    ret = connection.ParseRmtExchangeDto(*locDto);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = connection.ModifyQp();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(connection.rdmaConnStatus_, DevRdmaConnectionV2::RdmaConnStatus::QP_MODIFIED);

    std::cout << "End Ut_When_Normal_Call_Expect_Status_Consistent" << std::endl;
}

TEST_F(DevRdmaConnectionV2Test, Ut_When_QpAttrDto_IsValid_Expect_Correct)
{
    std::cout << "Start Ut_When_QpAttrDto_IsValid_Expect_Correct" << std::endl;

    DevRdmaConnectionV2::QpAttrDto validDto;
    validDto.qpn = 123;
    validDto.psn = 456;
    EXPECT_TRUE(validDto.IsValid());

    DevRdmaConnectionV2::QpAttrDto invalidDto1;
    invalidDto1.qpn = UINT32_MAX;
    invalidDto1.psn = 456;
    EXPECT_FALSE(invalidDto1.IsValid());

    DevRdmaConnectionV2::QpAttrDto invalidDto2;
    invalidDto2.qpn = 123;
    invalidDto2.psn = UINT32_MAX;
    EXPECT_FALSE(invalidDto2.IsValid());

    DevRdmaConnectionV2::QpAttrDto invalidDto3;
    invalidDto3.qpn = UINT32_MAX;
    invalidDto3.psn = UINT32_MAX;
    EXPECT_FALSE(invalidDto3.IsValid());

    std::cout << "End Ut_When_QpAttrDto_IsValid_Expect_Correct" << std::endl;
}

TEST_F(DevRdmaConnectionV2Test, Ut_When_Describe_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_Describe_Expect_NotEmpty" << std::endl;

    RdmaHandle rdmaHandle = (void*)0x1000000;
    DevRdmaConnectionV2 connection(fakeSocket, rdmaHandle, 0);

    HcclResult ret = connection.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string desc = connection.Describe();
    EXPECT_FALSE(desc.empty());

    std::cout << "End Ut_When_Describe_Expect_NotEmpty" << std::endl;
}

TEST_F(DevRdmaConnectionV2Test, Ut_When_BuildSqContext_Expect_Success)
{
    std::cout << "Start Ut_When_BuildSqContext_Expect_Success" << std::endl;

    RdmaHandle rdmaHandle = (void*)0x1000000;
    DevRdmaConnectionV2 connection(fakeSocket, rdmaHandle, 0);

    HcclResult ret = connection.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = connection.CreateQp();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<Hccl::Serializable> locDto;
    ret = connection.GetExchangeDto(locDto);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = connection.ParseRmtExchangeDto(*locDto);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    SqContext sqContext;
    ret = connection.BuildSqContext(&sqContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildSqContext_Expect_Success" << std::endl;
}

TEST_F(DevRdmaConnectionV2Test, Ut_When_BuildCqContext_Expect_Success)
{
    std::cout << "Start Ut_When_BuildCqContext_Expect_Success" << std::endl;

    RdmaHandle rdmaHandle = (void*)0x1000000;
    DevRdmaConnectionV2 connection(fakeSocket, rdmaHandle, 0);

    HcclResult ret = connection.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = connection.CreateQp();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<Hccl::Serializable> locDto;
    ret = connection.GetExchangeDto(locDto);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = connection.ParseRmtExchangeDto(*locDto);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CqContext cqContext;
    ret = connection.BuildCqContext(&cqContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildCqContext_Expect_Success" << std::endl;
}

TEST_F(DevRdmaConnectionV2Test, Ut_When_GetUniqueId_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetUniqueId_Expect_NotEmpty" << std::endl;

    RdmaHandle rdmaHandle = (void*)0x1000000;
    DevRdmaConnectionV2 connection(fakeSocket, rdmaHandle, 0);

    HcclResult ret = connection.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = connection.CreateQp();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<Hccl::Serializable> locDto;
    ret = connection.GetExchangeDto(locDto);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = connection.ParseRmtExchangeDto(*locDto);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<char> uniqueId = connection.GetUniqueId();
    EXPECT_FALSE(uniqueId.empty());

    std::cout << "End Ut_When_GetUniqueId_Expect_NotEmpty" << std::endl;
}

TEST_F(DevRdmaConnectionV2Test, Ut_When_NdaAlloc_Expect_Correct)
{
    std::cout << "Start Ut_When_NdaAlloc_Expect_Correct" << std::endl;

    RdmaHandle rdmaHandle = (void*)0x1000000;
    DevRdmaConnectionV2 connection(fakeSocket, rdmaHandle, 0);
    connection.Init();

    // success
    static void* fakePtr = reinterpret_cast<void*>(0xDEADBEEF);
    MOCKER(aclrtMalloc)
        .stubs()
        .with(outBoundP(&fakePtr, sizeof(fakePtr)), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    void* result = connection.ndaOps_.alloc(1024);
    EXPECT_EQ(result, fakePtr);

    // failure
    GlobalMockObject::verify();
    MOCKER(aclrtMalloc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_ERROR_RT_MEMORY_ALLOCATION));
    result = connection.ndaOps_.alloc(1024);
    EXPECT_EQ(result, nullptr);

    std::cout << "End Ut_When_NdaAlloc_Expect_Correct" << std::endl;
}

TEST_F(DevRdmaConnectionV2Test, Ut_When_NdaFree_Expect_Correct)
{
    std::cout << "Start Ut_When_NdaFree_Expect_Correct" << std::endl;

    RdmaHandle rdmaHandle = (void*)0x1000000;
    DevRdmaConnectionV2 connection(fakeSocket, rdmaHandle, 0);
    connection.Init();

    // null pointer: should return without calling aclrtFree
    connection.ndaOps_.free(nullptr);

    // success
    MOCKER(aclrtFree).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    void* ptr = reinterpret_cast<void*>(0xDEADBEEF);
    connection.ndaOps_.free(ptr);

    // failure
    GlobalMockObject::verify();
    MOCKER(aclrtFree).stubs().with(mockcpp::any()).will(returnValue(ACL_ERROR_RT_PARAM_INVALID));
    connection.ndaOps_.free(ptr);

    std::cout << "End Ut_When_NdaFree_Expect_Correct" << std::endl;
}

TEST_F(DevRdmaConnectionV2Test, Ut_When_NdaMemset_Expect_Correct)
{
    std::cout << "Start Ut_When_NdaMemset_Expect_Correct" << std::endl;

    RdmaHandle rdmaHandle = (void*)0x1000000;
    DevRdmaConnectionV2 connection(fakeSocket, rdmaHandle, 0);
    connection.Init();

    char buffer[64] = {};

    // success
    MOCKER(aclrtMemset)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    connection.ndaOps_.memset_s(buffer, 0, sizeof(buffer));

    // failure
    GlobalMockObject::verify();
    MOCKER(aclrtMemset)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_ERROR_RT_PARAM_INVALID));
    connection.ndaOps_.memset_s(buffer, 0, sizeof(buffer));

    std::cout << "End Ut_When_NdaMemset_Expect_Correct" << std::endl;
}

TEST_F(DevRdmaConnectionV2Test, Ut_When_NdaMemcpy_Expect_Correct)
{
    std::cout << "Start Ut_When_NdaMemcpy_Expect_Correct" << std::endl;

    RdmaHandle rdmaHandle = (void*)0x1000000;
    DevRdmaConnectionV2 connection(fakeSocket, rdmaHandle, 0);
    connection.Init();

    char src[64] = {};
    char dst[64] = {};

    // success: H2H, H2D, D2H, D2D
    MOCKER(aclrtMemcpy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    EXPECT_EQ(connection.ndaOps_.memcpy_s(dst, sizeof(dst), src, sizeof(src), MEMCPY_DIRECT_HOST_TO_HOST), 0);
    EXPECT_EQ(connection.ndaOps_.memcpy_s(dst, sizeof(dst), src, sizeof(src), MEMCPY_DIRECT_HOST_TO_DEVICE), 0);
    EXPECT_EQ(connection.ndaOps_.memcpy_s(dst, sizeof(dst), src, sizeof(src), MEMCPY_DIRECT_DEVICE_TO_HOST), 0);
    EXPECT_EQ(connection.ndaOps_.memcpy_s(dst, sizeof(dst), src, sizeof(src), MEMCPY_DIRECT_DEVICE_TO_DEVICE), 0);

    // unknown direction
    EXPECT_EQ(connection.ndaOps_.memcpy_s(dst, sizeof(dst), src, sizeof(src), 99), -1);

    // aclrtMemcpy failure
    GlobalMockObject::verify();
    MOCKER(aclrtMemcpy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_ERROR_RT_PARAM_INVALID));
    EXPECT_EQ(connection.ndaOps_.memcpy_s(dst, sizeof(dst), src, sizeof(src), MEMCPY_DIRECT_HOST_TO_DEVICE), -1);

    std::cout << "End Ut_When_NdaMemcpy_Expect_Correct" << std::endl;
}
