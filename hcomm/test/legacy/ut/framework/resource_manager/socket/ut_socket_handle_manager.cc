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
#include "socket_handle_manager.h"
#include "internal_exception.h"

using namespace Hccl;

class SocketHandleManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SocketHandleManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "SocketHandleManagerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in SocketHandleManagerTest SetUP" << std::endl;
        hccpSocketHandle = new int(0);
        MOCKER(HrtRaSocketInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(hccpSocketHandle));

        BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete hccpSocketHandle;
        std::cout << "A Test case in SocketHandleManagerTest TearDown" << std::endl;
    }

    IpAddress GetAnIpAddress()
    {
        IpAddress ipAddress("1.0.0.0");
        return ipAddress;
    }

    void* hccpSocketHandle;
};

TEST_F(SocketHandleManagerTest, should_return_valid_ptr_when_calling_create_with_valid_params)
{
    // Given
    u32 devicePhyId = 0;
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    PortData localPortData(0, basePortType, 0, IpAddress());

    // when
    auto res = SocketHandleManager::GetInstance().Create(devicePhyId, localPortData);

    // then
    EXPECT_EQ(hccpSocketHandle, res);
}

TEST_F(SocketHandleManagerTest, should_return_ptr_when_calling_create_with_valid_params)
{
    // Given
    u32 devicePhyId = 0;
    PortData localPortData(0, PortDeploymentType::P2P, LinkProtoType::INVALID, 0, IpAddress());

    // then
    EXPECT_THROW(SocketHandleManager::GetInstance().Create(devicePhyId, localPortData), InternalException);
}
