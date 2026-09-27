/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <sstream>
#include <cstring>

#define private public
#define protected public
#include "notify_manager.h"
#include "hccl_independent_common.h"
#include "adapter_rts_common.h"
#undef private
#undef protected

#include "manager_common.h"

using namespace hccl;

class NotifyManagerTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(NotifyManagerTest, NotifyTypeToNotifyLoadType_RtsNotify)
{
    NotifyManager mgr("test_comm", (aclrtBinHandle)0x1, ManagerCallbacks{});
    NotifyLoadType loadType;
    HcclResult ret = mgr.NotifyTypeToNotifyLoadType(NOTIFY_TYPE_RTS_NOTIFY, loadType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(loadType, NotifyLoadType::HOST_NOTIFY);
}

TEST_F(NotifyManagerTest, NotifyTypeToNotifyLoadType_RtsEvent)
{
    NotifyManager mgr("test_comm", (aclrtBinHandle)0x1, ManagerCallbacks{});
    NotifyLoadType loadType;
    HcclResult ret = mgr.NotifyTypeToNotifyLoadType(NOTIFY_TYPE_RTS_EVENT, loadType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(loadType, NotifyLoadType::HOST_NOTIFY);
}

TEST_F(NotifyManagerTest, NotifyTypeToNotifyLoadType_DeviceMem)
{
    NotifyManager mgr("test_comm", (aclrtBinHandle)0x1, ManagerCallbacks{});
    NotifyLoadType loadType;
    HcclResult ret = mgr.NotifyTypeToNotifyLoadType(NOTIFY_TYPE_DEVICE_MEM, loadType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(loadType, NotifyLoadType::DEVICE_NOTIFY);
}

TEST_F(NotifyManagerTest, NotifyTypeToNotifyLoadType_Unknown)
{
    NotifyManager mgr("test_comm", (aclrtBinHandle)0x1, ManagerCallbacks{});
    NotifyLoadType loadType;
    HcclResult ret = mgr.NotifyTypeToNotifyLoadType((::NotifyType)99, loadType);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(NotifyManagerTest, Constructor_Normal)
{
    ManagerCallbacks callbacks;
    NotifyManager mgr("myCommId", (aclrtBinHandle)0x1234, callbacks);
    EXPECT_EQ(mgr.GetNotifyNum(), 0u);
}

TEST_F(NotifyManagerTest, GetNotify_OutOfRange)
{
    NotifyManager mgr("test_comm", (aclrtBinHandle)0x1, ManagerCallbacks{});
    LocalNotify* notify = mgr.GetNotify(0);
    EXPECT_EQ(notify, nullptr);
}

TEST_F(NotifyManagerTest, GetNotify_InvalidIndex)
{
    NotifyManager mgr("test_comm", (aclrtBinHandle)0x1, ManagerCallbacks{});
    LocalNotify* notify = mgr.GetNotify(100);
    EXPECT_EQ(notify, nullptr);
}

class ParseBinNotifysTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }

    std::string BuildBinaryNotifys(NotifyLoadType loadType, size_t notifyNum)
    {
        std::ostringstream oss;
        oss.write(reinterpret_cast<const char*>(&loadType), sizeof(loadType));
        oss.write(reinterpret_cast<const char*>(&notifyNum), sizeof(notifyNum));
        for (size_t i = 0; i < notifyNum; i++) {
            HcclSignalInfo info;
            memset(&info, 0, sizeof(info));
            info.resId = static_cast<u32>(i);
            info.tsId = static_cast<s32>(i);
            info.devId = static_cast<u32>(i);
            oss.write(reinterpret_cast<const char*>(&info), sizeof(info));
        }
        return oss.str();
    }
};

TEST_F(ParseBinNotifysTest, ParseBinNotifys_EmptyString)
{
    std::vector<std::unique_ptr<LocalNotify>> newNotifys;
    HcclResult ret = NotifyManager::ParseBinNotifys("", newNotifys);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ParseBinNotifysTest, GetBinNotifys_EmptyList)
{
    std::vector<std::unique_ptr<LocalNotify>> newNotifys;
    std::string result = NotifyManager::GetBinNotifys(newNotifys, NotifyLoadType::DEVICE_NOTIFY);
    NotifyLoadType loadType;
    size_t notifyNum;
    std::istringstream iss(result);
    iss.read(reinterpret_cast<char*>(&loadType), sizeof(loadType));
    iss.read(reinterpret_cast<char*>(&notifyNum), sizeof(notifyNum));
    EXPECT_EQ(notifyNum, 0u);
}

TEST_F(NotifyManagerTest, Ut_ResetAllocLocalNotifies_When_Empty_Expect_Success)
{
    NotifyManager mgr("test_comm", (aclrtBinHandle)0x1, ManagerCallbacks{});
    EXPECT_EQ(mgr.ResetAllocLocalNotifies(), HCCL_SUCCESS);
}

TEST_F(NotifyManagerTest, Ut_ResetAllocLocalNotifies_When_HrtNotifyResetFailed_Expect_ReturnFailed)
{
    NotifyManager mgr("test_comm", (aclrtBinHandle)0x1, ManagerCallbacks{});
    auto notify = std::make_unique<LocalNotify>();
    notify->notifyPtr = reinterpret_cast<HcclRtNotify>(0x1234);
    notify->notifyOwner_ = false;
    notify->notifyId_ = 7;
    mgr.notifys_.push_back(std::move(notify));

    MOCKER(hrtNotifyReset).stubs().will(returnValue(HCCL_E_INTERNAL));
    EXPECT_EQ(mgr.ResetAllocLocalNotifies(), HCCL_E_INTERNAL);
}
