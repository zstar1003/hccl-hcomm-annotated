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
#include "cnt_1ton_notify_lite_mgr.h"
#include "queue_bcast_post_cnt_notify_manager.h"
#undef private

using namespace Hccl;

class Cnt1toNNotifyLiteMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Cnt1toNNotifyLiteMgrTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "Cnt1toNNotifyLiteMgrTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in Cnt1toNNotifyLiteMgrTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in Cnt1toNNotifyLiteMgrTest TearDown" << std::endl;
    }
};

TEST_F(Cnt1toNNotifyLiteMgrTest, test_parse_packed_data)
{
    QueueBcastPostCntNotifyManager cnt1toNNotifyMgr;
    Cnt1tonNotifyLiteMgr mgr;

    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1)).then(returnValue(2)).then(returnValue(3));

    QId fakePostQid = 1;
    u32 fakeTopicId = 2;
    cnt1toNNotifyMgr.ApplyFor(fakePostQid, fakeTopicId);
    fakePostQid = 2;
    fakeTopicId = 3;
    cnt1toNNotifyMgr.ApplyFor(fakePostQid, fakeTopicId);
    fakePostQid = 3;
    fakeTopicId = 4;
    cnt1toNNotifyMgr.ApplyFor(fakePostQid, fakeTopicId);

    auto data = cnt1toNNotifyMgr.GetPackedData();
    mgr.ParsePackedData(data);

    EXPECT_EQ(cnt1toNNotifyMgr.notifyPool.size(), mgr.notifys.size());
    for (auto& it : cnt1toNNotifyMgr.notifyPool) {
        auto& rts1TONCntNotify = *(it.second);
        auto& cnt1tonNotifyLite = mgr.notifys[it.first];
        EXPECT_EQ(rts1TONCntNotify.GetId(), cnt1tonNotifyLite->GetId());
    }
}
