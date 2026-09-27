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
#include <memory>
#include <map>

#define private public
#define protected public
#include "opretry_base.h"
#include "opretry_agent.h"
#include "opretry_server.h"
#include "socket.h"
#include "hccl_socket.h"
#include "opretry_manager.h"
#include "externalinput_pub.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class RetryBaseTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RetryBaseTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "RetryBaseTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

class RetryBaseSon : public OpRetryBase {
public:
    RetryBaseSon() {};
    ~RetryBaseSon() {};

    HcclResult Handle(RetryContext* retryCtx) override { return HCCL_SUCCESS; }

    HcclResult ProcessEvent(RetryContext* retryCtx) override { return HCCL_SUCCESS; }

    HcclResult ProcessError(RetryContext* retryCtx) override { return HCCL_SUCCESS; }

    HcclResult TestWaitResponse(std::shared_ptr<HcclSocket> socket, RetryInfo& retryInfo)
    {
        return WaitResponse(socket, retryInfo);
    }

    HcclResult TestIssueCommandWithOpId(std::shared_ptr<HcclSocket> socket, RetryCommandInfo& commandInfo)
    {
        return IssueCommandWithOpId(socket, commandInfo);
    }

    HcclResult TestWaitCommandWithOpId(std::shared_ptr<HcclSocket> socket, RetryCommandInfo& commandInfo)
    {
        return WaitCommandWithOpId(socket, commandInfo);
    }

    HcclResult
    TestRecvActiveSwitchInfo(std::shared_ptr<HcclSocket> socket, const u32 rankId, ActiveSwitchInfo& switchInfo)
    {
        return RecvActiveSwitchInfo(socket, rankId, switchInfo);
    }

    HcclResult TestWaitActiveSwitchInfo(std::shared_ptr<HcclSocket> socket, ActiveSwitchInfo& switchInfo)
    {
        return WaitActiveSwitchInfo(socket, switchInfo);
    }
};

TEST_F(RetryBaseTest, WaitResponse_Success_Expect_HcclSuccess)
{
    RetryBaseSon retryBase;
    HcclIpAddress localIp("127.0.0.1");
    std::shared_ptr<HcclSocket> socket
        = std::make_shared<HcclSocket>("TestWaitResponse", nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);

    MOCKER_CPP(&OpRetryBase::Recv).stubs().will(returnValue(HCCL_SUCCESS));

    RetryInfo retryInfo;
    HcclResult ret = retryBase.TestWaitResponse(socket, retryInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryBaseTest, IssueCommandWithOpId_Success_Expect_HcclSuccess)
{
    RetryBaseSon retryBase;
    HcclIpAddress localIp("127.0.0.1");
    std::shared_ptr<HcclSocket> socket = std::make_shared<HcclSocket>(
        "TestIssueCommandWithOpId", nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);

    MOCKER_CPP(&OpRetryBase::Send).stubs().will(returnValue(HCCL_SUCCESS));

    RetryCommandInfo commandInfo;
    HcclResult ret = retryBase.TestIssueCommandWithOpId(socket, commandInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryBaseTest, WaitCommandWithOpId_Success_Expect_HcclSuccess)
{
    RetryBaseSon retryBase;
    HcclIpAddress localIp("127.0.0.1");
    std::shared_ptr<HcclSocket> socket = std::make_shared<HcclSocket>(
        "TestWaitCommandWithOpId", nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);

    MOCKER_CPP(&OpRetryBase::Recv).stubs().will(returnValue(HCCL_SUCCESS));

    RetryCommandInfo commandInfo;
    HcclResult ret = retryBase.TestWaitCommandWithOpId(socket, commandInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryBaseTest, RecvActiveSwitchInfo_Success_Expect_HcclSuccess)
{
    RetryBaseSon retryBase;
    HcclIpAddress localIp("127.0.0.1");
    std::shared_ptr<HcclSocket> socket = std::make_shared<HcclSocket>(
        "TestRecvActiveSwitchInfo", nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);

    MOCKER_CPP(&OpRetryBase::Recv).stubs().will(returnValue(HCCL_SUCCESS));

    ActiveSwitchInfo switchInfo;
    HcclResult ret = retryBase.TestRecvActiveSwitchInfo(socket, 0, switchInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryBaseTest, RecvActiveSwitchInfo_RecvFail_Expect_Error)
{
    RetryBaseSon retryBase;
    HcclIpAddress localIp("127.0.0.1");
    std::shared_ptr<HcclSocket> socket = std::make_shared<HcclSocket>(
        "TestRecvActiveSwitchInfoFail", nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);

    MOCKER_CPP(&OpRetryBase::Recv).stubs().will(returnValue(HCCL_E_INTERNAL));

    ActiveSwitchInfo switchInfo;
    HcclResult ret = retryBase.TestRecvActiveSwitchInfo(socket, 0, switchInfo);
    EXPECT_NE(ret, HCCL_SUCCESS);
}
