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
#include "communicator_impl.h"
#include "conn_local_notify_manager.h"
#include "local_notify_v2.h"
#include "rdma_handle_manager.h"
#undef protected
#undef private

using namespace Hccl;

class ConnLocalNotifyManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ConnLocalNotifyManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ConnLocalNotifyManagerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
        std::cout << "A Test case in ConnLocalNotifyManagerTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();

        std::cout << "A Test case in ConnLocalNotifyManagerTest TearDown" << std::endl;
    }
};

TEST_F(ConnLocalNotifyManagerTest, applyfor_return_ok)
{
    CommunicatorImpl comm;
    ConnLocalNotifyManager connLocalNotifyManager(&comm);
    // Given
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtIpcSetNotifyName).stubs();
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1));
    MOCKER(HrtNotifyGetAddr).stubs().will(returnValue((u64)0));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(1));
    MOCKER(HrtGetSocVer).stubs();

    RankId fakeLocalRankID = 1;
    RankId fakeRemoteRankID = 2;
    u32 fakeLocalPortId = 1;
    u32 fakeRemotePortId = 1;
    BasePortType basePortType(PortDeploymentType::P2P);
    LinkData fakeLinkData(basePortType, fakeLocalRankID, fakeRemoteRankID, fakeLocalPortId, fakeRemotePortId);

    // When
    connLocalNotifyManager.ApplyFor(fakeRemoteRankID, fakeLinkData);

    // Then

    // When
    connLocalNotifyManager.ApplyFor(fakeRemoteRankID, fakeLinkData); // duplicate apply

    // Then
}

TEST_F(ConnLocalNotifyManagerTest, release_return_ok)
{
    CommunicatorImpl comm;
    ConnLocalNotifyManager connLocalNotifyManager(&comm);
    // Given
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtIpcSetNotifyName).stubs();
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1));
    MOCKER(HrtNotifyGetAddr).stubs().will(returnValue((u64)0));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(1));
    MOCKER(HrtGetSocVer).stubs();

    RankId fakeLocalRankID = 1;
    RankId fakeRemoteRankID = 2;
    u32 fakeLocalPortId = 1;
    u32 fakeRemotePortId = 1;
    BasePortType basePortType(PortDeploymentType::P2P);
    LinkData fakeLinkData(basePortType, fakeLocalRankID, fakeRemoteRankID, fakeLocalPortId, fakeRemotePortId);
    connLocalNotifyManager.ApplyFor(fakeRemoteRankID, fakeLinkData);

    // When
    auto result = connLocalNotifyManager.Release(fakeRemoteRankID, fakeLinkData);

    // Then
    EXPECT_EQ(true, result);
}

TEST_F(ConnLocalNotifyManagerTest, destroy_return_nok)
{
    CommunicatorImpl comm;
    ConnLocalNotifyManager connLocalNotifyManager(&comm);
    // Given
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtIpcSetNotifyName).stubs();
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1));
    MOCKER(HrtNotifyGetAddr).stubs().will(returnValue((u64)0));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(1));
    MOCKER(HrtGetSocVer).stubs();

    RankId fakeLocalRankID = 1;
    RankId fakeRemoteRankID = 2;
    u32 fakeLocalPortId = 1;
    u32 fakeRemotePortId = 1;
    BasePortType basePortType(PortDeploymentType::P2P);
    LinkData fakeLinkData(basePortType, fakeLocalRankID, fakeRemoteRankID, fakeLocalPortId, fakeRemotePortId);
    connLocalNotifyManager.ApplyFor(fakeRemoteRankID, fakeLinkData);

    RankId fakeLocalRankID1 = 2;
    RankId fakeRemoteRankID1 = 3;
    u32 fakeLocalPortId1 = 2;
    u32 fakeRemotePortId1 = 2;
    BasePortType basePortType1(PortDeploymentType::P2P);
    LinkData fakeLinkData1(basePortType1, fakeLocalRankID1, fakeRemoteRankID1, fakeLocalPortId1, fakeRemotePortId1);
    connLocalNotifyManager.ApplyFor(fakeRemoteRankID1, fakeLinkData1);

    // When
    auto result = connLocalNotifyManager.Destroy();

    // Then
    EXPECT_EQ(true, result);
}

TEST_F(ConnLocalNotifyManagerTest, apply_for_ub_notify_ok)
{
    CommunicatorImpl comm{};
    comm.devPhyId = 0;
    ConnLocalNotifyManager connLocalNotifyManager(&comm);
    // Given
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtIpcSetNotifyName).stubs();
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1));
    MOCKER(HrtNotifyGetAddr).stubs().will(returnValue((u64)0));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(1));
    MOCKER(HrtGetSocVer).stubs();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));

    pair<TokenIdHandle, uint32_t> fakeTokenInfo = make_pair(0x12345678, 1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));
    RankId fakeLocalRankID = 1;
    RankId fakeRemoteRankID = 4;
    u32 fakeLocalPortId = 3;
    u32 fakeRemotePortId = 2;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData fakeLinkData(basePortType, fakeLocalRankID, fakeRemoteRankID, fakeLocalPortId, fakeRemotePortId);

    EXPECT_NO_THROW(connLocalNotifyManager.ApplyFor(fakeRemoteRankID, fakeLinkData));
}
