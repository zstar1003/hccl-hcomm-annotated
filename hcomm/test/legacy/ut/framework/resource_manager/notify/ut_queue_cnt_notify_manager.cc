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
#include "queue_wait_group_cnt_notify_manager.h"
#undef private
#include "communicator_impl.h"

using namespace Hccl;

class QueueWaitGroupCntNotifyManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QueueWaitGroupCntNotifyManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "QueueWaitGroupCntNotifyManagerTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in QueueWaitGroupCntNotifyManagerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();

        std::cout << "A Test case in QueueWaitGroupCntNotifyManagerTest TearDown" << std::endl;
    }
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
};

TEST_F(QueueWaitGroupCntNotifyManagerTest, apply_for_test)
{
    QueueWaitGroupCntNotifyManager queueWaitGroupCntNotifyManager;
    // Given
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    QId qid = 1;
    u32 index = 1;
    // When
    queueWaitGroupCntNotifyManager.ApplyFor(qid, index);
}

TEST_F(QueueWaitGroupCntNotifyManagerTest, get_test)
{
    QueueWaitGroupCntNotifyManager queueWaitGroupCntNotifyManager;
    // Given
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    QId qid = 1;
    u32 index = 1;

    // When
    auto result = queueWaitGroupCntNotifyManager.Get(qid, index);
    queueWaitGroupCntNotifyManager.ApplyFor(qid, index);
    auto result1 = queueWaitGroupCntNotifyManager.Get(qid, index);

    // Then
    EXPECT_EQ(nullptr, result);
    EXPECT_NE(nullptr, result1);
}

TEST_F(QueueWaitGroupCntNotifyManagerTest, release_return_ok)
{
    QueueWaitGroupCntNotifyManager queueWaitGroupCntNotifyManager;
    // Given
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType::DEV_TYPE_910A2));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    QId qid = 1;
    u32 index = 1;

    // When
    auto result0 = queueWaitGroupCntNotifyManager.Release(qid, index);

    // Then
    EXPECT_EQ(true, result0);

    // When
    queueWaitGroupCntNotifyManager.Destroy();
}

TEST_F(QueueWaitGroupCntNotifyManagerTest, getalldtos_ok)
{
    QueueWaitGroupCntNotifyManager queueWaitGroupCntNotifyManager;
    // Given
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType::DEV_TYPE_910A2));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));
}

TEST_F(QueueWaitGroupCntNotifyManagerTest, getpackeddata_ok)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    std::pair<u32, u32> cntPairId(0, 0);
    QueueWaitGroupCntNotifyManager mgr;
    auto ptr = std::make_unique<RtsCntNotify>();
    mgr.notifyPool[cntPairId] = std::move(ptr);

    mgr.GetPackedData();
}
