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
#include <stdio.h>

#include "hccl/base.h"
#include <hccl/hccl_types.h>

#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "sal.h"

using namespace std;
using namespace hccl;

class WrapperTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--WrapperTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--WrapperTest TearDown--\033[0m" << std::endl; }

    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(WrapperTest, hostmem_constructor_00)
{
    s32 ret = HCCL_SUCCESS;

    HostMem mem = HostMem::alloc(8);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(WrapperTest, hostmem_constructor_01)
{
    s32 ret = HCCL_SUCCESS;

    HostMem mem1 = HostMem::alloc(8);
    HostMem mem2(mem1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(WrapperTest, hostmem_operate_equal)
{
    s32 ret = HCCL_SUCCESS;

    HostMem mem1 = HostMem::alloc(8);
    HostMem mem2 = mem1;
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(WrapperTest, hostmem_create)
{
    s32 ret = HCCL_SUCCESS;
    void* ptr = NULL;
    HostMem mem;
    mem.create(ptr, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(WrapperTest, hostmem_range)
{
    s32 ret = HCCL_SUCCESS;
    HostMem test;
    HostMem mem1 = HostMem::alloc(8);
    test = mem1.range(0, 4);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(WrapperTest, hostmem_operate_equal_copy_assignment)
{
    s32 ret = HCCL_SUCCESS;

    HostMem mem1;
    HostMem mem0 = HostMem::alloc(8);
    s32 size = mem0.size();
    EXPECT_EQ(size, 8);
    if (mem0) {
        mem1 = mem0;
    }
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
TEST_F(WrapperTest, devicemem_constructor_00)
{
    s32 ret = HCCL_SUCCESS;

    DeviceMem mem = DeviceMem::alloc(8);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(WrapperTest, devicemem_constructor_01)
{
    s32 ret = HCCL_SUCCESS;

    DeviceMem mem1 = DeviceMem::alloc(8);
    DeviceMem mem2(mem1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(WrapperTest, devicemem_operate_equal)
{
    s32 ret = HCCL_SUCCESS;

    DeviceMem mem1 = DeviceMem::alloc(8);
    DeviceMem mem2 = mem1;
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(WrapperTest, devicemem_create)
{
    s32 ret = HCCL_SUCCESS;
    void* ptr = NULL;
    DeviceMem mem;
    mem.create(ptr, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(WrapperTest, devicemem_range)
{
    s32 ret = HCCL_SUCCESS;
    DeviceMem test;
    DeviceMem mem1 = DeviceMem::alloc(8);
    test = mem1.range(0, 4);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(WrapperTest, devicemem_operate_equal_copy_assignment)
{
    s32 ret = HCCL_SUCCESS;

    DeviceMem mem1;
    DeviceMem mem0 = DeviceMem::alloc(8);
    mem1 = mem0;

    EXPECT_EQ(ret, HCCL_SUCCESS);
}
