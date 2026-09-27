/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define protected public
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "communicator_impl.h"
#include "buffer.h"
#include "internal_exception.h"
#undef protected

using namespace Hccl;
class BufferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Buffer tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "Buffer tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in DataBufManager SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in DataBufManager TearDown" << std::endl;
    }
};

TEST_F(BufferTest, DevBuffer_not_selfowned)
{
    uintptr_t fakeAddr = 1000;
    size_t fakeSize = 1000;

    shared_ptr<DevBuffer> buffer = DevBuffer::Create(fakeAddr, fakeSize);

    EXPECT_EQ(fakeAddr, buffer->GetAddr());
    EXPECT_EQ(fakeSize, buffer->GetSize());
    EXPECT_EQ(false, buffer->GetSelfOwned());

    std::cout << buffer->Describe() << std::endl;
}

TEST_F(BufferTest, DevBuffer_selfowned)
{
    EXPECT_THROW(DevBuffer(0), InternalException);

    uintptr_t fakeAddr = 1000;
    size_t fakeSize = 1000;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)fakeAddr));
    MOCKER(HrtFree).stubs().with(mockcpp::any());

    DevBuffer buffer(fakeSize);
    EXPECT_EQ(fakeAddr, buffer.GetAddr());
    EXPECT_EQ(fakeSize, buffer.GetSize());
    EXPECT_EQ(true, buffer.GetSelfOwned());
    std::cout << buffer.Describe() << std::endl;
}

TEST_F(BufferTest, contains_buffer_test)
{
    shared_ptr<DevBuffer> buffer1 = DevBuffer::Create(1, 100);
    shared_ptr<DevBuffer> buffer2 = DevBuffer::Create(10, 10);
    EXPECT_TRUE(buffer1->Contains(buffer2.get()));
}

TEST_F(BufferTest, contains_bufadr_bufsize_test)
{
    shared_ptr<DevBuffer> buffer = DevBuffer::Create(100, 100);
    EXPECT_TRUE(buffer->Contains(100, 100));
}

TEST_F(BufferTest, Buffer_test)
{
    uintptr_t fakeAddr = 1000;
    size_t fakeSize = 1000;
    shared_ptr<DevBuffer> buffer = DevBuffer::Create(fakeAddr, fakeSize);
    EXPECT_EQ(fakeAddr, buffer->GetAddr());
    EXPECT_EQ(fakeSize, buffer->GetSize());
    auto range1 = buffer->Range(0, fakeSize);
    EXPECT_EQ(fakeAddr, range1.GetAddr());

    auto range2 = buffer->Range(0, fakeSize + 1);
    EXPECT_EQ(0, range2.GetSize());
    std::cout << buffer->Describe() << std::endl;
}
