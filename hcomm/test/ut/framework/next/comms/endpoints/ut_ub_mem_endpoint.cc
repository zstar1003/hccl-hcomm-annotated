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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "ub_mem_endpoint.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "ip_address.h"
#include "hccp.h"
#include "buffer.h"
#include "network_api_exception.h"
#include "endpoint.h"
#include "adapter_rts.h"

class AivUbMemEndpointTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AivUbMemEndpointTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AivUbMemEndpointTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AivUbMemEndpointTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AivUbMemEndpointTest TearDown" << std::endl;
    }

    HcommResult CreateUbMemEndpoint(const std::string& ip, void** endpointHandle)
    {
        Hccl::IpAddress localIp(ip);
        EndpointDesc endpointDesc;
        endpointDesc.protocol = COMM_PROTOCOL_UB_MEM;
        endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
        endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        return HcommEndpointCreate(&endpointDesc, endpointHandle);
    }

    CommMem CreateCommMem(void* addr, size_t size, CommMemType type)
    {
        CommMem mem;
        mem.type = type;
        mem.size = size;
        mem.addr = addr;
        return mem;
    }
};

// 内存注册和解注册成功
TEST_F(AivUbMemEndpointTest, Ut_When_Register_Memory_NORMAL_Expect_Return_SUCCESS)
{
    void* endpointHandle{nullptr};
    HcommResult ret = CreateUbMemEndpoint("1.0.0.0", &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommMemUnreg(endpointHandle, memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 解注册空MemHandle
TEST_F(AivUbMemEndpointTest, Ut_When_Unregister_Without_Register_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    HcommResult ret = CreateUbMemEndpoint("1.0.0.0", &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* invalidMemHandle = nullptr;
    ret = HcommMemUnreg(endpointHandle, invalidMemHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 一次reg多次unreg
TEST_F(AivUbMemEndpointTest, Ut_When_Double_Unregister_After_Register_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    HcommResult ret = CreateUbMemEndpoint("1.0.0.0", &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    // 正常注册
    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 第一次注销，应该成功
    ret = HcommMemUnreg(endpointHandle, memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 第二次注销同一个memHandle，应该失败
    ret = HcommMemUnreg(endpointHandle, memHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

// 注册空指针内存
TEST_F(AivUbMemEndpointTest, Ut_When_Register_Null_Memory_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    HcommResult ret = CreateUbMemEndpoint("1.0.0.0", &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem(nullptr, 10, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 注册大小为0的内存
TEST_F(AivUbMemEndpointTest, Ut_When_Register_Zero_Size_Memory_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    HcommResult ret = CreateUbMemEndpoint("1.0.0.0", &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 0, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 注册无效内存类型
TEST_F(AivUbMemEndpointTest, Ut_When_Register_Invalid_MemType_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    HcommResult ret = CreateUbMemEndpoint("1.0.0.0", &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_INVALID);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 注销空指针
TEST_F(AivUbMemEndpointTest, Ut_When_Unregister_Null_Handle_Expect_Return_Error)
{
    void* endpointHandle{nullptr};
    HcommResult ret = CreateUbMemEndpoint("1.0.0.0", &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 使用nullptr注销
    ret = HcommMemUnreg(endpointHandle, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(AivUbMemEndpointTest, Ut_When_Unregister_Wrong_Handle_Expect_Return_Error)
{
    void* endpointHandle1{nullptr};
    HcommResult ret = CreateUbMemEndpoint("1.0.0.0", &endpointHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* endpointHandle2{nullptr};
    ret = CreateUbMemEndpoint("2.0.0.0", &endpointHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem1 = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    CommMem mem2 = CreateCommMem((void*)0x02, 20, COMM_MEM_TYPE_DEVICE);
    void *memHandle1, *memHandle2;

    // 在第一个endpoint上注册内存
    ret = HcommMemReg(endpointHandle1, "memTag1", &mem1, &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 在第二个endpoint上注册内存
    ret = HcommMemReg(endpointHandle2, "memTag2", &mem2, &memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 使用第二个endpoint的memHandle2去第一个endpoint注销失败
    ret = HcommMemUnreg(endpointHandle1, memHandle2);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    // 使用第一个endpoint的memHandle1去第二个endpoint注销失败
    ret = HcommMemUnreg(endpointHandle2, memHandle1);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    // 正确注销：各自在自己的endpoint上注销自己的内存
    ret = HcommMemUnreg(endpointHandle1, memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommMemUnreg(endpointHandle2, memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 重复内存注册重复销毁
TEST_F(AivUbMemEndpointTest, Ut_When_Double_Register_Unregister_Memory_Expect_Return_SUCCESS)
{
    void* endpointHandle{nullptr};
    HcommResult ret = CreateUbMemEndpoint("1.0.0.0", &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommMemUnreg(endpointHandle, memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommMemUnreg(endpointHandle, memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
