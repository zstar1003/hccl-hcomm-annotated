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
#include "sal.h"
#include "adapter_rts.h"

using namespace std;
using namespace hccl;

class HostMemTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--HostMemTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--HostMemTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(HostMemTest, constructor_00)
{
    s32 ret = HCCL_SUCCESS;

    HostMem mem = HostMem::alloc(8);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostMemTest, constructor_01)
{
    s32 ret = HCCL_SUCCESS;

    HostMem mem1 = HostMem::alloc(8);
    HostMem mem2(mem1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostMemTest, destructor_00)
{
    s32 ret = HCCL_SUCCESS;
    HostMem mem1 = HostMem::alloc(8);
    mem1.free();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostMemTest, operate_equal)
{
    s32 ret = HCCL_SUCCESS;

    HostMem mem1 = HostMem::alloc(8);
    HostMem mem2 = mem1;
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostMemTest, create)
{
    s32 ret = HCCL_SUCCESS;
    void* ptr = NULL;
    HostMem mem;
    mem.create(ptr, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostMemTest, range)
{
    s32 ret = HCCL_SUCCESS;
    HostMem test;
    HostMem mem1 = HostMem::alloc(8);
    test = mem1.range(0, 4);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostMemTest, range_errro)
{
    s32 ret = HCCL_SUCCESS;
    HostMem test;
    HostMem mem1 = HostMem::alloc(8);
    test = mem1.range(0, 9);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
TEST_F(HostMemTest, operate_equal_copy_assignment)
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

TEST_F(HostMemTest, alloc_fail)
{
    s32 ret = HCCL_SUCCESS;

    /*构造rt_malloc异常*/
    MOCKER(hrtMallocHost).expects(atMost(1)).will(returnValue(1));
    HostMem mem = HostMem::alloc(8);
    GlobalMockObject::verify();
}

TEST_F(HostMemTest, free_fail)
{
    s32 ret = HCCL_SUCCESS;
    HostMem mem = HostMem::alloc(8);

    /*构造rt_malloc异常*/
    MOCKER(hrtFreeHost).expects(atMost(1)).will(returnValue(1));
    mem.~HostMem();
    GlobalMockObject::verify();
}
