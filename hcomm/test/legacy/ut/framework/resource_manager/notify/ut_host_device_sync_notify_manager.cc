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
#include "host_device_sync_notify_manager.h"
#undef protected
#undef private

using namespace Hccl;

class HostDeviceSyncNotifyManagerTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "HostDeviceSyncNotifyManagerTest SetUP" << std::endl;
        u32 fakeDevPhyId = 1;
        u64 fakeNotifyHandleAddr = 100;
        u32 fakeNotifyId = 1;
        u64 fakeOffset = 200;
        char fakeName[65] = "testRtsNotify";
        SETUP_COMM_NOTIFY_MOCKS();
    }

    static void TearDownTestCase() { std::cout << "HostDeviceSyncNotifyManagerTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HostDeviceSyncNotifyManagerTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HostDeviceSyncNotifyManagerTest TearDown" << std::endl;
    }
};

TEST_F(HostDeviceSyncNotifyManagerTest, Ut_GetAiCpuCntNotifys_When_Repeat_Expect_OK)
{
    HostDeviceSyncNotifyManager manager = HostDeviceSyncNotifyManager();
    u8 cnt = 10;
    void* aicpuNotify[10];
    manager.GetMc2AiCpuNotifys(cnt, aicpuNotify);
    EXPECT_EQ(10, manager.mc2AicpuNotifys_.size());
    cnt = 8;
    manager.GetMc2AiCpuNotifys(cnt, aicpuNotify);
    EXPECT_EQ(10, manager.mc2AicpuNotifys_.size());
}
