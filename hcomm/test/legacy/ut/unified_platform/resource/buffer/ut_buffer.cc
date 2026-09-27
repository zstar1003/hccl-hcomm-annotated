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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#define protected public
#include "communicator_impl.h"
#include "buffer.h"
#include "internal_exception.h"
#undef protected
#undef private
using namespace Hccl;
class BufferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Buffer tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "Buffer tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue((void*)fakeAddr));
        MOCKER(HrtFreeHost).stubs().with(mockcpp::any());
        MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)fakeAddr));
        MOCKER(HrtFree).stubs().with(mockcpp::any());
        std::cout << "A Test case in DataBufManager SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in DataBufManager TearDown" << std::endl;
    }

    uintptr_t fakeAddr = 1000;
    size_t fakeSize = 1000;
};

TEST_F(BufferTest, HostBuffer_not_selfowned)
{
    HostBuffer buffer(fakeAddr, fakeSize);

    EXPECT_EQ(fakeAddr, buffer.GetAddr());
    EXPECT_EQ(fakeSize, buffer.GetSize());
    EXPECT_EQ(false, buffer.GetSelfOwned());

    std::cout << buffer.Describe() << std::endl;
}

TEST_F(BufferTest, HostBuffer_selfowned)
{
    EXPECT_THROW(HostBuffer(0), InternalException);

    HostBuffer buffer(fakeSize);
    EXPECT_EQ(fakeAddr, buffer.GetAddr());
    EXPECT_EQ(fakeSize, buffer.GetSize());
    EXPECT_EQ(true, buffer.GetSelfOwned());
    std::cout << buffer.Describe() << std::endl;
}

TEST_F(BufferTest, DevBuffer_not_selfowned)
{
    DevBuffer buffer(fakeAddr, fakeSize);

    EXPECT_EQ(fakeAddr, buffer.GetAddr());
    EXPECT_EQ(fakeSize, buffer.GetSize());
    EXPECT_EQ(false, buffer.GetSelfOwned());

    std::cout << buffer.Describe() << std::endl;
}

TEST_F(BufferTest, DevBuffer_selfowned)
{
    EXPECT_THROW(DevBuffer(0), InternalException);

    DevBuffer buffer(fakeSize);
    EXPECT_EQ(fakeAddr, buffer.GetAddr());
    EXPECT_EQ(fakeSize, buffer.GetSize());
    EXPECT_EQ(true, buffer.GetSelfOwned());
    std::cout << buffer.Describe() << std::endl;
}

TEST_F(BufferTest, contains_buffer_test)
{
    DevBuffer buffer1(0, 100);
    DevBuffer buffer2(10, 10);
    EXPECT_TRUE(buffer1.Contains(&buffer2));
}

TEST_F(BufferTest, contains_bufadr_bufsize_test)
{
    DevBuffer buffer(100, 100);
    EXPECT_TRUE(buffer.Contains(100, 100));
}

TEST_F(BufferTest, Buffer_test)
{
    Buffer buffer(fakeAddr, fakeSize);
    EXPECT_EQ(fakeAddr, buffer.GetAddr());
    EXPECT_EQ(fakeSize, buffer.GetSize());
    auto range1 = buffer.Range(0, fakeSize);
    EXPECT_EQ(fakeAddr, range1.GetAddr());

    auto range2 = buffer.Range(0, fakeSize + 1);
    EXPECT_EQ(0, range2.GetSize());
    std::cout << buffer.Describe() << std::endl;
}
