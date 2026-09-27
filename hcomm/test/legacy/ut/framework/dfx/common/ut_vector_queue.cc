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
#include <mockcpp/MockObject.h>
#define private public
#include "vector_queue.h"
#include <chrono>
#include <memory>
#undef private

using namespace Hccl;

class VectorQueueTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "VectorQueueTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "VectorQueueTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in VectorQueueTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in VectorQueueTest TearDown" << std::endl;
    }

public:
    VectorQueue<int> fakeQueue;
};

TEST_F(VectorQueueTest, Ut_Append_When_AppendNumIs2048_Return_Ok)
{
    // When
    int appendNum = 2048;
    EXPECT_EQ(fakeQueue.Size(), 0);
    // Then
    for (int i = 0; i < appendNum; ++i) {
        fakeQueue.Append(std::move(i));
    }
    EXPECT_EQ(fakeQueue.Size(), appendNum);
}

TEST_F(VectorQueueTest, Ut_Traverse_When_PrintAllElems_ReturnIsAllElems)
{
    // When
    int appendNum = 10;
    for (int i = 0; i < appendNum; ++i) {
        fakeQueue.Append(std::move(i));
    }
    // Then
    fakeQueue.Traverse([](const int& value) {
        std::cout << value << std::endl;
    });
}

TEST_F(VectorQueueTest, Ut_IsEmpty_When_VectorIsEmpty_ReturnIsTrue) { EXPECT_EQ(fakeQueue.IsEmpty(), true); }

TEST_F(VectorQueueTest, Ut_IsEmpty_When_VectorIsNotEmpty_ReturnIsFalse)
{
    // When
    int val = 10;
    fakeQueue.Append(std::move(val));
    // Then
    EXPECT_EQ(fakeQueue.IsEmpty(), false);
}

TEST_F(VectorQueueTest, Ut_Find_When_ValueInVector_ReturnIsNeEnd)
{
    // When
    int val = 10;
    fakeQueue.Append(std::move(val));
    // Then
    auto it = fakeQueue.Find([](const int& value) {
        return value == 10;
    });
    EXPECT_NE(*it, *fakeQueue.End());
    EXPECT_EQ(**it, 10);
}

TEST_F(VectorQueueTest, Ut_Find_When_ValueNotInVector_ReturnIsEqEnd)
{
    // When
    int val = 10;
    fakeQueue.Append(std::move(val));
    // Then
    auto it = fakeQueue.Find([](const int& value) {
        return value == 100;
    });
    EXPECT_EQ(*it, *fakeQueue.End());
}

TEST_F(VectorQueueTest, Ut_Begin_When_NotEmpty_ReturnIsBegin)
{
    // When
    int val = 10;
    fakeQueue.Append(std::move(val));
    val = 20;
    fakeQueue.Append(std::move(val));
    val = 30;
    fakeQueue.Append(std::move(val));
    // Then
    auto it = fakeQueue.Begin();
    EXPECT_EQ(**it, 10);
}

TEST_F(VectorQueueTest, Ut_Begin_When_IsEmpty_ReturnIsEnd)
{
    // Then
    auto it = fakeQueue.Begin();
    EXPECT_EQ(*it, *fakeQueue.End());
}

TEST_F(VectorQueueTest, Ut_Tail_When_NotEmpty_ReturnIsBegin)
{
    // When
    int val = 10;
    fakeQueue.Append(std::move(val));
    val = 20;
    fakeQueue.Append(std::move(val));
    val = 30;
    fakeQueue.Append(std::move(val));
    // Then
    auto it = fakeQueue.Tail();
    EXPECT_EQ(**it, 30);
}

TEST_F(VectorQueueTest, Ut_Tail_When_IsEmpty_ReturnIsEnd)
{
    // Then
    auto it = fakeQueue.Tail();
    EXPECT_EQ(*it, *fakeQueue.End());
}

TEST_F(VectorQueueTest, Ut_Iterator_When_PlusAndMinus_ReturnIsOk)
{
    // When
    int val = 10;
    fakeQueue.Append(std::move(val));
    val = 20;
    fakeQueue.Append(std::move(val));
    val = 30;
    fakeQueue.Append(std::move(val));

    // Then ++it
    auto it = fakeQueue.Begin();
    EXPECT_EQ(**it, 10);
    ++(*it);
    EXPECT_EQ(**it, 20);
    ++(*it);
    EXPECT_EQ(**it, 30);

    // Then --it
    --(*it);
    EXPECT_EQ(**it, 20);
    --(*it);
    EXPECT_EQ(**it, 10);

    // Then it++
    (*it)++;
    EXPECT_EQ(**it, 20);
    (*it)++;
    EXPECT_EQ(**it, 30);

    // Then it--
    (*it)--;
    EXPECT_EQ(**it, 20);
    (*it)--;
    EXPECT_EQ(**it, 10);
}

TEST_F(VectorQueueTest, Ut_Iterator_When_ScaleUp_ReturnIsOk)
{
    // When
    int val = 10;
    fakeQueue.Append(std::move(val));
    fakeQueue.elems_.shrink_to_fit(); // 缩小空间
    auto it = fakeQueue.Begin();      // 取迭代器
    // Then
    val = 20;
    fakeQueue.Append(std::move(val)); // 触发扩容
    EXPECT_EQ(**it, 10);              // 校验
}
