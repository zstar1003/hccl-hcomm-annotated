/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#define private public
#include "channels/aiv/aiv_ub_mem_transport.h"
#include "runtime_api_exception.h"
#include "socket_exception.h"

using namespace hcomm;

static void StubSocketSendAsyncNoop(Hccl::Socket* self, const void* sendBuf, u32 size)
{
    (void)self;
    (void)sendBuf;
    (void)size;
}

static void StubSocketRecvAsyncNoop(Hccl::Socket* self, u8* recvBuf, u32 size)
{
    (void)self;
    (void)recvBuf;
    (void)size;
}

class AivUbMemTransportTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "AivUbMemTransportTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "AivUbMemTransportTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AivUbMemTransportTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AivUbMemTransportTest TearDown" << std::endl;
    }
};

TEST_F(AivUbMemTransportTest, St_GetStatus_When_SOCKET_OK_Expect_Success)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    HcommChannelDesc desc{};
    AivUbMemTransport aivTransport(fakeSocket, desc);

    aivTransport.baseStatus_ = Hccl::TransportStatus::SOCKET_OK;
    aivTransport.aivUbStatus_ = hcomm::AivUbMemTransport::AivUbMemTransportStatus::SOCKET_OK;

    MOCKER_CPP(&AivUbMemTransport::RmtBufferUnpackProc).stubs();
    MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const void*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(StubSocketSendAsyncNoop));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(u8*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(StubSocketRecvAsyncNoop));

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::SOCKET_OK);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::SEND_DATA_SIZE);

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::SOCKET_OK);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_DATA_SIZE);

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::SOCKET_OK);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::SEND_MEM_INFO);

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::SOCKET_OK);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_MEM_INFO);

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::SOCKET_OK);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_MEM_FIN);

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::READY);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::READY);
}

TEST_F(AivUbMemTransportTest, Ut_GetStatus_When_RecvData_Fail_Expect_Status_Invalid)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    HcommChannelDesc desc{};
    AivUbMemTransport aivTransport(fakeSocket, desc);

    aivTransport.baseStatus_ = Hccl::TransportStatus::SOCKET_OK;
    aivTransport.aivUbStatus_ = hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_MEM_INFO;

    MOCKER_CPP(&AivUbMemTransport::RmtBufferUnpackProc).stubs().will(throws(Hccl::RuntimeApiException("test_fail")));

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::INVALID);
    EXPECT_EQ(aivTransport.aivUbStatus_, hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_MEM_FIN);
}

TEST_F(AivUbMemTransportTest, Ut_GetStatus_When_SocketReady_Fail_Expect_Status_Invalid)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    HcommChannelDesc desc{};
    AivUbMemTransport aivTransport(fakeSocket, desc);

    aivTransport.baseStatus_ = Hccl::TransportStatus::SOCKET_OK;
    aivTransport.aivUbStatus_ = hcomm::AivUbMemTransport::AivUbMemTransportStatus::RECV_MEM_INFO;

    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(throws(Hccl::SocketException("test_fail")));

    EXPECT_EQ(aivTransport.GetStatus(), Hccl::TransportStatus::INVALID);
}
