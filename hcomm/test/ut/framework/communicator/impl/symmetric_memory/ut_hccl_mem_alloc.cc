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
#include <mockcpp/mockcpp.hpp>
#include <iostream>

#define private public
#define protected public
#include "hccl_comm.h"
#include "hcomm_res.h"
#include "acl/acl_rt.h"
#undef private
#undef protected

using namespace std;

constexpr size_t TWO_M = 2097152;

class MemAllocTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MemAllocTest Testcase SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "MemAllocTest Testcase TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A MemAllocTest SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A MemAllocTest TearDown" << std::endl;
    }
};

TEST_F(MemAllocTest, ut_HcommMemAlloc_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    void* ptr = nullptr;
    size_t size = TWO_M;
    HcommResult ret = HcommMemAlloc(&ptr, size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MemAllocTest, ut_HcommMemAlloc_When_SizeIsZero_Expect_ReturnHCCL_E_PARA)
{
    void* ptr = nullptr;
    size_t size = 0;
    HcommResult ret = HcommMemAlloc(&ptr, size);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(MemAllocTest, ut_HcommMemAlloc_When_GetDeviceFailed_Expect_ReturnHCCL_E_RUNTIME)
{
    MOCKER(aclrtGetDevice).stubs().will(returnValue(500000));

    void* ptr = nullptr;
    size_t size = TWO_M;
    HcommResult ret = HcommMemAlloc(&ptr, size);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MemAllocTest, ut_HcommMemAlloc_When_GetGranularityFailed_Expect_ReturnHCCL_E_RUNTIME)
{
    MOCKER(aclrtMemGetAllocationGranularity).stubs().will(returnValue(500000));

    void* ptr = nullptr;
    size_t size = TWO_M;
    HcommResult ret = HcommMemAlloc(&ptr, size);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MemAllocTest, ut_HcommMemAlloc_When_GranularityIsZero_Expect_ReturnHCCL_E_RUNTIME)
{
    MOCKER(aclrtMemGetAllocationGranularity).stubs().will(returnValue(ACL_SUCCESS));

    void* ptr = nullptr;
    size_t size = TWO_M;
    HcommResult ret = HcommMemAlloc(&ptr, size);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MemAllocTest, ut_HcommMemAlloc_When_ReserveMemAddressFailed_Expect_ReturnHCCL_E_RUNTIME)
{
    MOCKER(aclrtReserveMemAddress).stubs().will(returnValue(500000));

    void* ptr = nullptr;
    size_t size = TWO_M + 1; // 对齐测试
    HcommResult ret = HcommMemAlloc(&ptr, size);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MemAllocTest, ut_HcommMemAlloc_When_MallocPhysicalFailed_Expect_ReturnHCCL_E_RUNTIME)
{
    MOCKER(aclrtMallocPhysical).stubs().will(returnValue(500000));

    void* ptr = nullptr;
    size_t size = TWO_M;
    HcommResult ret = HcommMemAlloc(&ptr, size);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MemAllocTest, ut_HcommMemAlloc_When_MapMemFailed_Expect_ReturnHCCL_E_RUNTIME)
{
    MOCKER(aclrtMapMem).stubs().will(returnValue(500000));

    void* ptr = nullptr;
    size_t size = TWO_M;
    HcommResult ret = HcommMemAlloc(&ptr, size);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MemAllocTest, ut_HcommMemFree_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    int temp = 0;
    void* ptr = &temp;
    HcommResult ret = HcommMemFree(ptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MemAllocTest, ut_HcommMemFree_When_PtrIsNull_Expect_ReturnHCCL_SUCCESS)
{
    void* ptr = nullptr;
    HcommResult ret = HcommMemFree(ptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MemAllocTest, ut_HcommMemFree_When_RetainAllocationHandleFailed_Expect_ReturnHCCL_E_RUNTIME)
{
    MOCKER(aclrtMemRetainAllocationHandle).stubs().will(returnValue(500000));

    int temp = 0;
    void* ptr = &temp;
    HcommResult ret = HcommMemFree(ptr);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MemAllocTest, ut_HcommMemFree_When_UnmapMemFailed_Expect_ReturnHCCL_E_RUNTIME)
{
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(500000));

    int temp = 0;
    void* ptr = &temp;
    HcommResult ret = HcommMemFree(ptr);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MemAllocTest, ut_HcommMemFree_When_FreePhysicalFailed_Expect_ReturnHCCL_E_RUNTIME)
{
    MOCKER(aclrtFreePhysical).stubs().will(returnValue(500000));

    int temp = 0;
    void* ptr = &temp;
    HcommResult ret = HcommMemFree(ptr);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MemAllocTest, ut_HcommMemFree_When_ReleaseMemAddressFailed_Expect_ReturnHCCL_E_RUNTIME)
{
    MOCKER(aclrtReleaseMemAddress).stubs().will(returnValue(500000));

    int temp = 0;
    void* ptr = &temp;
    HcommResult ret = HcommMemFree(ptr);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}
