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
#include "hccp_peer_manager.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#undef private
using namespace Hccl;

class HccpPeerManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HccpPeerManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HccpPeerManagerTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HccpPeerManagerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HccpPeerManagerTest TearDown" << std::endl;
    }
};

TEST_F(HccpPeerManagerTest, hccp_peer_manager_getInstance)
{
    // Given
    SETUP_HCCP_PEER_MGR_MOCKS(3, 4);
    // when
    s32 deviceLogicId = 0;
    HccpPeerManager::GetInstance().Init(deviceLogicId);
    s32 deviceLogicId1 = 1;
    HccpPeerManager::GetInstance().Init(deviceLogicId1);
    auto res = HccpPeerManager::GetInstance().instances_;

    // then
    EXPECT_EQ(2, res.size());
}

TEST_F(HccpPeerManagerTest, hccp_peer_manager_init)
{
    HccpPeerManager::GetInstance().instances_.clear();
    // Given
    s32 deviceLogicId = 0;
    s32 deviceLogicId1 = 1;
    s32 deviceLogicId2 = 2;
    DevId fakedevPhyId = 3;
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(fakedevPhyId));
    MOCKER(HrtRaDeInit).stubs().with();
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));

    // when
    HccpPeerManager::GetInstance().Init(deviceLogicId);
    auto res1 = HccpPeerManager::GetInstance().instances_;
    HccpPeerManager::GetInstance().Init(deviceLogicId);
    auto res2 = HccpPeerManager::GetInstance().instances_;

    // then
    EXPECT_EQ(res1[deviceLogicId].Count() + 1, res2[deviceLogicId].Count());

    // when
    HccpPeerManager::GetInstance().Init(deviceLogicId);
    HccpPeerManager::GetInstance().Init(deviceLogicId);
    HccpPeerManager::GetInstance().Init(deviceLogicId1);
    HccpPeerManager::GetInstance().Init(deviceLogicId1);
    HccpPeerManager::GetInstance().Init(deviceLogicId2);
    HccpPeerManager::GetInstance().Init(deviceLogicId2);
    auto res = HccpPeerManager::GetInstance().instances_;

    // then
    EXPECT_EQ(3, res.size());

    EXPECT_NO_THROW(HccpPeerManager::GetInstance().DeInit(deviceLogicId1));
    EXPECT_NO_THROW(HccpPeerManager::GetInstance().DeInit(deviceLogicId1));
    EXPECT_NO_THROW(HccpPeerManager::GetInstance().DeInitAll());
}
