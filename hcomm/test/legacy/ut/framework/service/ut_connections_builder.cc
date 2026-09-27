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
#include "socket_manager.h"
#include "connections_builder.h"
#include "communicator_impl.h"
#undef private

using namespace Hccl;
using namespace std;

class ConnectionsBuilderTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ConnectionsBuilderTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ConnectionsBuilderTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ConnectionsBuilderTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "A Test case in ConnectionsBuilderTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    static std::unique_ptr<Socket> mockProducer(
        IpAddress& localIpAddress, IpAddress& remoteIpAddress, u32 listenPort, SocketHandle socketHandle,
        const std::string& tag, SocketRole socketRole, NicType nicType)
    {
        return std::make_unique<Socket>(
            socketHandle, localIpAddress, listenPort, remoteIpAddress, "testxxxxx", SocketRole::CLIENT,
            NicType::DEVICE_NIC_TYPE);
    }
};

TEST_F(ConnectionsBuilderTest, CreateRmaConnections_test)
{
    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    comm.InitRmaConnManager();
    ConnectionsBuilder connBuilder(comm);
    LinkData link(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    vector<LinkData> links = {link};
    string opTag = "test";
    DevUbConnection ubConnection((void*)0x100, link.GetLocalAddr(), link.GetRemoteAddr(), OpMode::OPBASE);
    RmaConnection* rmaConnection = &ubConnection;
    MOCKER_CPP(&RmaConnManager::Create).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rmaConnection));
    connBuilder.CreateRmaConnections(opTag, links);
}
