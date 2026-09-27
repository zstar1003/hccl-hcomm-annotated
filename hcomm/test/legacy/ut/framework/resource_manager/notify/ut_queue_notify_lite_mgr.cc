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
#include "null_ptr_exception.h"
#define private public
#include "queue_notify_lite_mgr.h"
#include "queue_notify_manager_v2.h"
#include "communicator_impl.h"
#include "rts_notify_v2.h"
#undef private
using namespace Hccl;

class QueueNotifyLiteMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QueueNotifyLiteMgrTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "QueueNotifyLiteMgrTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in QueueNotifyLiteMgrTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in QueueNotifyLiteMgrTest TearDown" << std::endl;
    }
};

TEST_F(QueueNotifyLiteMgrTest, test_parse_packed_data)
{
    CommunicatorImpl comm;
    QueueNotifyManager queueNotifyManager(comm);
    QueueNotifyLiteMgr mgr;

    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(1));
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1)).then(returnValue(2)).then(returnValue(3));

    QId fakePostQid = 1;
    QId fakeWaitQid = 2;
    u32 fakeTopicId = 3;
    queueNotifyManager.ApplyFor(fakePostQid, fakeWaitQid, fakeTopicId);
    fakePostQid = 2;
    fakeWaitQid = 3;
    fakeTopicId = 4;
    queueNotifyManager.ApplyFor(fakePostQid, fakeWaitQid, fakeTopicId);
    fakePostQid = 4;
    fakeWaitQid = 5;
    fakeTopicId = 6;
    queueNotifyManager.ApplyFor(fakePostQid, fakeWaitQid, fakeTopicId);

    auto data = queueNotifyManager.GetPackedData();
    mgr.ParsePackedData(data);

    EXPECT_EQ(queueNotifyManager.notifyPool.size(), mgr.notifys.size());
    for (auto& it : queueNotifyManager.notifyPool) {
        auto& rtsNotify = *(it.second);
        auto& notifyLite = mgr.notifys[it.first];
        EXPECT_EQ(rtsNotify.GetId(), notifyLite->GetId());
        EXPECT_EQ(rtsNotify.GetDevPhyId(), notifyLite->GetDevPhyId());
    }
}
