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
#include "queue_notify_manager_v2.h"
#include "local_notify_v2.h"
#include "communicator_impl.h"

using namespace Hccl;

class QueueNotifyManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QueueNotifyManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "QueueNotifyManagerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
        std::cout << "A Test case in QueueNotifyManagerTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();

        std::cout << "A Test case in QueueNotifyManagerTest TearDown" << std::endl;
    }
};

TEST_F(QueueNotifyManagerTest, applyfor_return_ok)
{
    CommunicatorImpl comm;
    QueueNotifyManager queueNotifyManager(comm);
    // Given
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtIpcSetNotifyName).stubs();
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1));
    MOCKER(HrtNotifyGetAddr).stubs().will(returnValue((u64)0));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(1));
    MOCKER(HrtGetSocVer).stubs();

    QId fakePostQid = 1;
    QId fakeWaitQid = 2;
    u32 fakeCount = 3;

    // When
    queueNotifyManager.ApplyFor(fakePostQid, fakeWaitQid, fakeCount);

    // When
    queueNotifyManager.ApplyFor(fakePostQid, fakeWaitQid, fakeCount); // duplicate apply
}

TEST_F(QueueNotifyManagerTest, release_return_ok)
{
    CommunicatorImpl comm;
    QueueNotifyManager queueNotifyManager(comm);
    // Given
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtIpcSetNotifyName).stubs();
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1));
    MOCKER(HrtNotifyGetAddr).stubs().will(returnValue((u64)0));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(1));
    MOCKER(HrtGetSocVer).stubs();

    QId fakePostQid = 1;
    QId fakeWaitQid = 2;
    u32 fakeCount = 3;
    queueNotifyManager.ApplyFor(fakePostQid, fakeWaitQid, fakeCount);

    // When
    auto result = queueNotifyManager.Release(fakePostQid, fakeWaitQid, fakeCount);

    // Then
    EXPECT_EQ(true, result);
}

TEST_F(QueueNotifyManagerTest, destroy_return_nok)
{
    CommunicatorImpl comm;
    QueueNotifyManager queueNotifyManager(comm);
    // Given
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtIpcSetNotifyName).stubs();
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1));
    MOCKER(HrtNotifyGetAddr).stubs().will(returnValue((u64)0));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(1));
    MOCKER(HrtGetSocVer).stubs();

    QId fakePostQid = 1;
    QId fakeWaitQid = 2;
    u32 fakeCount = 3;
    queueNotifyManager.ApplyFor(fakePostQid, fakeWaitQid, fakeCount);

    QId fakePostQid1 = 1;
    QId fakeWaitQid1 = 2;
    u32 fakeCount1 = 3;
    queueNotifyManager.ApplyFor(fakePostQid1, fakeWaitQid1, fakeCount1);

    // When
    auto result = queueNotifyManager.Destroy();

    // Then
    EXPECT_EQ(true, result);
}

TEST_F(QueueNotifyManagerTest, release_test)
{
    CommunicatorImpl comm;
    QueueNotifyManager queueNotifyManager(comm);
    // Given
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtIpcSetNotifyName).stubs();
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1));
    MOCKER(HrtNotifyGetAddr).stubs().will(returnValue((u64)0));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(1));
    MOCKER(HrtGetSocVer).stubs();
    auto result = queueNotifyManager.Release(999, 999, 399);
}
