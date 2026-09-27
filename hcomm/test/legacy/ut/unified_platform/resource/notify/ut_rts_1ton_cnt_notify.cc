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
#include "test_mock_setup.h"
#define private public
#define protected public
#include "rts_1ton_cnt_notify.h"
#undef protected
#undef private

using namespace Hccl;

class Rts1ToNCntNotifyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Rts1ToNCntNotifyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "Rts1ToNCntNotifyTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in Rts1ToNCntNotifyTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in Rts1ToNCntNotifyTest TearDown" << std::endl;
    }
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u32 fakeDevPhyId = 2;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
};

TEST_F(Rts1ToNCntNotifyTest, rts1toncntnotify_construct_ok)
{
    // Given
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    // When
    Rts1ToNCntNotify rts1ToNCntNotify;
    std::string des = rts1ToNCntNotify.Describe();
    EXPECT_NE(0, des.size());
    // Then
    EXPECT_EQ(fakeNotifyId, rts1ToNCntNotify.GetId());
}

TEST_F(Rts1ToNCntNotifyTest, rts1toncntnotify_postbits_submit_test)
{
    // Given
    SETUP_CNT_NOTIFY_MOCKS();

    // When
    Rts1ToNCntNotify rts1ToNCntNotify;

    // Then
    u32 val = 1;
    Stream stream;
    rts1ToNCntNotify.PostValue(val, stream);
}

TEST_F(Rts1ToNCntNotifyTest, rts1toncntnotify_waitvalue_submit_test)
{
    // Given
    SETUP_CNT_NOTIFY_MOCKS();

    // When
    Rts1ToNCntNotify rts1ToNCntNotify;

    // Then
    u32 bitValue = 1;
    Stream stream;
    rts1ToNCntNotify.WaitBits(bitValue, 100, stream);
}

TEST_F(Rts1ToNCntNotifyTest, rts1toncntnotify_getuniqueid_test)
{
    SETUP_CNT_NOTIFY_MOCKS();

    Rts1ToNCntNotify rts1ToNCntNotify;
    rts1ToNCntNotify.id = fakeNotifyId;
    rts1ToNCntNotify.devPhyId = fakeDevPhyId;

    BinaryStream binaryStream;
    binaryStream << rts1ToNCntNotify.id;
    binaryStream << rts1ToNCntNotify.devPhyId;
    std::vector<char> res;
    binaryStream.Dump(res);

    EXPECT_EQ(rts1ToNCntNotify.GetUniqueId(), res);
}
