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
#include "hccp_hdc_manager.h"
#include "orion_adapter_rts.h"
using namespace Hccl;

class HccpHdcManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HccpHdcManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HccpHdcManagerTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HccpHdcManagerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HccpHdcManagerTest TearDown" << std::endl;
    }
};

TEST_F(HccpHdcManagerTest, hccp_hdc_manager_getInstance)
{
    // Given
    DevId fakedevPhyId = 3;
    DevId fakedevPhyId1 = 4;
    MOCKER(HrtGetDevicePhyIdByUserDevId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(fakedevPhyId))
        .then(returnValue(fakedevPhyId1));
    // when
    s32 deviceLogicId = 0;
    HccpHdcManager::GetInstance().Init(deviceLogicId);
    s32 deviceLogicId1 = 1;
    HccpHdcManager::GetInstance().Init(deviceLogicId1);
    auto res = HccpHdcManager::GetInstance().GetSet();

    // then
    EXPECT_EQ(2, res.size());
}

TEST_F(HccpHdcManagerTest, hccp_hdc_manager_init)
{
    // Given
    s32 deviceLogicId = 0;
    s32 deviceLogicId1 = 1;
    s32 deviceLogicId2 = 2;
    DevId fakedevPhyId = 3;
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(fakedevPhyId));
    // when
    HccpHdcManager::GetInstance().Init(deviceLogicId);
    auto res1 = HccpHdcManager::GetInstance().GetSet();
    HccpHdcManager::GetInstance().Init(deviceLogicId);
    auto res2 = HccpHdcManager::GetInstance().GetSet();

    // then
    EXPECT_EQ(res1, res2);

    // when
    HccpHdcManager::GetInstance().Init(deviceLogicId);
    HccpHdcManager::GetInstance().Init(deviceLogicId);
    HccpHdcManager::GetInstance().Init(deviceLogicId1);
    HccpHdcManager::GetInstance().Init(deviceLogicId1);
    HccpHdcManager::GetInstance().Init(deviceLogicId2);
    HccpHdcManager::GetInstance().Init(deviceLogicId2);
    auto res = HccpHdcManager::GetInstance().GetSet();

    // then
    EXPECT_EQ(3, res.size());
}
