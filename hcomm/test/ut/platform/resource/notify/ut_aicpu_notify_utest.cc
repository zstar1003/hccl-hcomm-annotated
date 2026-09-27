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
#include <stdio.h>

#include "hccl/base.h"
#include <hccl/hccl_types.h>

#ifndef private
#define private public
#define protected public
#endif

#include "local_notify.h"
#include "local_notify_impl.h"
#include "dlhal_function.h"
#include "remote_notify.h"
#include "notify_base.h"
#include "rts_notify.h"
#include "local_ipc_notify.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "sal.h"

#include "adapter_rts.h"

#undef private
#undef protected

using namespace std;
using namespace hccl;

class NotifyAiCpu_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--NotifyAiCpu_UT SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--NotifyAiCpu_UT TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        notifyInfo.addr = 100;
        notifyInfo.devId = 1;
        notifyInfo.rankId = 2;
        notifyInfo.resId = 3;
        notifyInfo.tsId = 4;
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    HcclSignalInfo notifyInfo;
};

TEST_F(NotifyAiCpu_UT, init_with_signal_info)
{
    s32 ret = HCCL_SUCCESS;

    // local notify
    LocalNotify localNotify;
    ret = localNotify.Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(localNotify.notifyOwner_);

    // remote notify
    RemoteNotify remoteNotify;
    ret = remoteNotify.Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // local ipc notify
    LocalIpcNotify localIpcNotify;
    ret = localIpcNotify.Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(localIpcNotify.notifyOwner_);
}

TEST_F(NotifyAiCpu_UT, check_signal_info)
{
    s32 ret = HCCL_SUCCESS;
    HcclSignalInfo notifyInfoGotten;

    // local notify
    LocalNotify localNotify;
    localNotify.Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
    localNotify.GetNotifyData(notifyInfoGotten);
    EXPECT_EQ(notifyInfoGotten.addr, notifyInfo.addr);
    EXPECT_EQ(notifyInfoGotten.devId, notifyInfo.devId);
    EXPECT_EQ(notifyInfoGotten.resId, notifyInfo.resId);
    EXPECT_EQ(notifyInfoGotten.tsId, notifyInfo.tsId);

    // remote notify
    RemoteNotify remoteNotify;
    remoteNotify.Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
    notifyInfo.tsId = 400;
    remoteNotify.SetNotifyData(notifyInfo);
    remoteNotify.GetNotifyData(notifyInfoGotten);
    EXPECT_EQ(notifyInfoGotten.tsId, u32(400));
}
