/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "device_resource_manager.h"

class DeviceResourceManagerTest : public testing::Test {
protected:
};

TEST_F(DeviceResourceManagerTest, GetInstance_ReturnsSingleton)
{
    auto& mgr1 = DeviceResourceManager::GetInstance();
    auto& mgr2 = DeviceResourceManager::GetInstance();
    EXPECT_EQ(&mgr1, &mgr2);
}

TEST_F(DeviceResourceManagerTest, Init_WithValidRankSize)
{
    DeviceResourceManager& mgr = DeviceResourceManager::GetInstance();
    EXPECT_NO_THROW(mgr.Init(4));
}

TEST_F(DeviceResourceManagerTest, InitRankRes_WithValidParams)
{
    DeviceResourceManager& mgr = DeviceResourceManager::GetInstance();
    mgr.Init(4);
    EXPECT_NO_THROW(mgr.InitRankRes(0, 2));
}

TEST_F(DeviceResourceManagerTest, NotifyRecord_BasicCall)
{
    DeviceResourceManager& mgr = DeviceResourceManager::GetInstance();
    mgr.Init(4);
    mgr.InitRankRes(0, 2);
    EXPECT_NO_THROW(mgr.NotifyRecord(0, 0, 0, 1));
}

TEST_F(DeviceResourceManagerTest, NotifyWait_AfterRecord)
{
    DeviceResourceManager& mgr = DeviceResourceManager::GetInstance();
    mgr.Init(4);
    mgr.InitRankRes(0, 2);
    mgr.NotifyRecord(0, 0, 0, 1);
    EXPECT_TRUE(mgr.NotifyWait(0, 0, 0, 1));
}

TEST_F(DeviceResourceManagerTest, CheckNotifyWait_BasicCall)
{
    DeviceResourceManager& mgr = DeviceResourceManager::GetInstance();
    mgr.Init(4);
    mgr.InitRankRes(0, 2);
    mgr.NotifyRecord(0, 0, 0, 1);
    EXPECT_TRUE(mgr.CheckNotifyWait(0, 0, 0, 1));
}
