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
#define private public
#include "stream.h"
#include "null_ptr_exception.h"
#undef private

using namespace Hccl;
class StreamTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Stream tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "Stream tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetStreamId).stubs().will(returnValue(fakeId));
        MOCKER(HrtGetDevice).stubs().will(returnValue(fakeDevLogId));
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(fakeDevPhyId)));
        MOCKER(HrtStreamCreateWithFlags).stubs().will(returnValue(fakePtr));
        MOCKER(HrtStreamGetSqId).stubs().will(returnValue(fakeSqId));
        MOCKER(HrtStreamDestroy).stubs();
        std::cout << "A Test case in Stream SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in Stream TearDown" << std::endl;
    }
    void* fakePtr = (void*)1;
    u32 fakeId = 1;
    s32 fakeDevLogId = 1;
    u32 fakeDevPhyId = 1;
    u32 fakeSqId = 2;
    u64 fakeStmMode = 3;
};

TEST_F(StreamTest, Stream_selfownded_false)
{
    Stream stream(fakePtr);
    stream.SetStmMode(fakeStmMode);

    EXPECT_EQ(fakeId, stream.GetId());
    EXPECT_EQ(fakePtr, stream.GetPtr());
    EXPECT_EQ(false, stream.IsSelfOwned());
    EXPECT_EQ(fakeStmMode, stream.GetMode());

    stream.Describe();
}

TEST_F(StreamTest, stream_dev_used_false)
{
    Stream stream(false);

    stream.SetStmMode(fakeStmMode);

    EXPECT_EQ(fakeId, stream.GetId());
    EXPECT_EQ(fakePtr, stream.GetPtr());
    EXPECT_EQ(true, stream.IsSelfOwned());

    std::cout << stream.Describe() << std::endl;
}

TEST_F(StreamTest, stream_dev_used_true)
{
    Stream stream(true);

    stream.SetStmMode(fakeStmMode);

    EXPECT_EQ(fakeId, stream.GetId());
    EXPECT_EQ(fakePtr, stream.GetPtr());
    EXPECT_EQ(true, stream.IsSelfOwned());

    stream.GetUniqueId();

    std::cout << stream.Describe() << std::endl;
}
