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
#define protected public
#include "host_device_sync_notify_lite_mgr.h"
#include "host_device_sync_notify_manager.h"
#include "orion_adapter_rts.h"
#undef private
#undef protected
using namespace Hccl;

class HostDeviceSyncNotifyLiteMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HostDeviceSyncNotifyLiteMgrTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HostDeviceSyncNotifyLiteMgrTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        std::cout << "A Test case in HostDeviceSyncNotifyLiteMgrTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HostDeviceSyncNotifyLiteMgrTest TearDown" << std::endl;
    }
};

TEST_F(HostDeviceSyncNotifyLiteMgrTest, test_parse_packed_data)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A3));
    HostDeviceSyncNotifyManager mgr;
    HostDeviceSyncNotifyLiteMgr liteMgr;

    auto data = mgr.GetPackedData();
    liteMgr.ParsePackedData(data);

    EXPECT_EQ(mgr.notifys[0]->GetId(), liteMgr.notifys[0]->GetId());
    EXPECT_EQ(mgr.notifys[1]->GetDevPhyId(), liteMgr.notifys[1]->GetDevPhyId());
}
