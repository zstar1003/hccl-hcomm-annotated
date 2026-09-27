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
#include "rma_connection.h"
#include "socket.h"
#include "not_support_exception.h"

using namespace Hccl;

class RmaConnectionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DevUbConnection tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "DevUbConnection tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in DevUbConnection SetUP" << std::endl;
        fakeSocket
            = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete fakeSocket;
        std::cout << "A Test case in DevUbConnection TearDown" << std::endl;
    }

    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    std::string tag = "test";
};

class FakeRmaConnection : public RmaConnection {
public:
    FakeRmaConnection(Socket* socket, const RmaConnType rmaConnType) : RmaConnection(socket, rmaConnType) {}

    void Connect() override {}
    string Describe() const override {}
};

TEST_F(RmaConnectionTest, test_rma_connection_inline_write)
{
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);

    SqeConfig sqeConfig;
    FakeRmaConnection rmaConn(fakeSocket, RmaConnType::UB);
    MemoryBuffer memBuffer(0, 0, 0);
    EXPECT_THROW(rmaConn.PrepareInlineWrite(memBuffer, 0, sqeConfig), NotSupportException);
}
