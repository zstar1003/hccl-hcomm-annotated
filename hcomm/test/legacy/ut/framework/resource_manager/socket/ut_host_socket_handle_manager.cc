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
#include "host_socket_handle_manager.h"
#undef protected
#undef private

using namespace Hccl;

class HostSocketHandleManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HostSocketHandleManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HostSocketHandleManagerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in HostSocketHandleManagerTest SetUP" << std::endl;
        socketHandle = new int(0);
        MOCKER(HrtRaSocketInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(socketHandle));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete socketHandle;
        std::cout << "A Test case in HostSocketHandleManagerTest TearDown" << std::endl;
    }

    IpAddress GetAnIpAddress()
    {
        IpAddress ipAddress("1.0.0.0");
        return ipAddress;
    }

    void* socketHandle;
    ;
};

TEST_F(HostSocketHandleManagerTest, should_return_valid_ptr_when_calling_create_with_valid_params)
{
    // Given
    u32 devicePhyId = 0;
    IpAddress ipAddress = GetAnIpAddress();

    // when
    auto res = HostSocketHandleManager::GetInstance().Create(devicePhyId, ipAddress);
    auto res1 = HostSocketHandleManager::GetInstance().Get(devicePhyId, ipAddress);

    // then
    EXPECT_EQ(socketHandle, res);
    EXPECT_EQ(socketHandle, res1);

    res = HostSocketHandleManager::GetInstance().Create(devicePhyId, ipAddress);
    res1 = HostSocketHandleManager::GetInstance().Get(1, ipAddress);
    EXPECT_EQ(socketHandle, res);

    EXPECT_NO_THROW(HostSocketHandleManager::GetInstance().Destroy(devicePhyId, ipAddress));
    EXPECT_NO_THROW(HostSocketHandleManager::GetInstance().DestroyAll());
}
