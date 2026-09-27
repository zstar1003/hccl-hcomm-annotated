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
#include "queue_bcast_post_cnt_notify_manager.h"
#include "communicator_impl.h"
#undef protected
#undef private

using namespace Hccl;

class QueueBcastPostCntNotifyManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QueueBcastPostCntNotifyManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "QueueBcastPostCntNotifyManagerTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in QueueBcastPostCntNotifyManagerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();

        std::cout << "A Test case in QueueBcastPostCntNotifyManagerTest TearDown" << std::endl;
    }
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
};

TEST_F(QueueBcastPostCntNotifyManagerTest, apply_for_test)
{
    QueueBcastPostCntNotifyManager queueBcastPostCntNotifyManager;
    // Given
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    QId qid = 1;
    u32 topicId = 1;
    // When
    queueBcastPostCntNotifyManager.ApplyFor(qid, topicId);
}

TEST_F(QueueBcastPostCntNotifyManagerTest, get_test)
{
    QueueBcastPostCntNotifyManager queueBcastPostCntNotifyManager;
    // Given
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    QId qid = 1;
    u32 topicId = 1;

    // When
    auto result = queueBcastPostCntNotifyManager.Get(qid, topicId);
    queueBcastPostCntNotifyManager.ApplyFor(qid, topicId);
    auto result1 = queueBcastPostCntNotifyManager.Get(qid, topicId);

    // Then
    EXPECT_EQ(nullptr, result);
    EXPECT_NE(nullptr, result1);
}

TEST_F(QueueBcastPostCntNotifyManagerTest, release_return_ok)
{
    QueueBcastPostCntNotifyManager queueBcastPostCntNotifyManager;
    // Given
    DevType devType = DevType::DEV_TYPE_910A2;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    QId qid = 1;
    u32 topicId = 1;

    // When
    auto result0 = queueBcastPostCntNotifyManager.Release(qid, topicId);

    // Then
    EXPECT_EQ(true, result0);

    // When
    queueBcastPostCntNotifyManager.Destroy();
}

TEST_F(QueueBcastPostCntNotifyManagerTest, getpackeddata_ok)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    std::pair<u32, u32> cntPairId(0, 0);
    QueueBcastPostCntNotifyManager mgr;
    auto ptr = std::make_unique<Rts1ToNCntNotify>();
    mgr.notifyPool[cntPairId] = std::move(ptr);

    mgr.GetPackedData();
}
