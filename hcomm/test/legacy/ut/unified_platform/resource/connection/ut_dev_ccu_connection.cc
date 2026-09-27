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
#include <mockcpp/MockObject.h>

#include "dev_ccu_connection.h"
#include "network_api_exception.h"
#include "communicator_impl.h"
#include "ccu_api_exception.h"
#include "rma_conn_exception.h"

using namespace Hccl;
using namespace Ccu;

class DevCcuConnectionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DevCcuConnection tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "DevCcuConnection tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType::DEV_TYPE_950));
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<s32>(0)));
        MOCKER(HrtRaUbCreateJetty)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(HrtRaUbJettyCreatedOutParam()));
        MOCKER(HraGetDieAndFuncId).stubs().with(mockcpp::any()).will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));
        MOCKER(HrtRaUbCreateJfc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(jfcHandle));
        MOCKER(RaUbImportJetty)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HrtRaUbJettyImportedOutParam()));

        MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
        MOCKER(HrtRaSocketBlockSend).stubs().will(returnValue(true));

        fakeSocket
            = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

        std::cout << "A Test case in DevCcuConnection SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete fakeSocket;
        std::cout << "A Test case in DevCcuConnection TearDown" << std::endl;
    }

    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    std::string tag = "ccu_test";
    JfcHandle jfcHandle = 1;
};

TEST_F(DevCcuConnectionTest, dev_ccu_connection_get_status_return_ok)
{
    RdmaHandle rdmaHandle = (void*)0x1000000;
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    ChannelInfo channelInfo;
    channelInfo.channelId = 0;
    channelInfo.jettys.push_back(JettyInfo());
    MOCKER_CPP(&CcuComponent::AllocChannel).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(channelInfo));
    MOCKER_CPP(&CcuComponent::CreateLocalOpChannel).stubs().will(ignoreReturnValue());
    MOCKER(HrtRaSocketBlockRecv).stubs().will(returnValue(true));
    CommunicatorImpl comm;
    CcuComponent ccuComponent(&comm);
    DevCcuConnection devCcuConnection(fakeSocket, rdmaHandle, linkData, ccuComponent);
    devCcuConnection.Connect();
    RmaConnStatus status = devCcuConnection.GetStatus();

    EXPECT_EQ(RmaConnStatus::READY, status);
}

TEST_F(DevCcuConnectionTest, dev_ccu_connection_bind_success)
{
    RdmaHandle rdmaHandle = (void*)0x1000000;
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    ChannelInfo channelInfo;
    channelInfo.channelId = 0;
    channelInfo.jettys.push_back(JettyInfo());
    MOCKER_CPP(&CcuComponent::AllocChannel).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(channelInfo));
    MOCKER_CPP(&CcuComponent::CreateLocalOpChannel).stubs().will(ignoreReturnValue());
    MOCKER(HrtRaSocketBlockRecv).stubs().will(returnValue(true));
    CommunicatorImpl comm;
    CcuComponent ccuComponent(&comm);
    DevCcuConnection devCcuConnection(fakeSocket, rdmaHandle, linkData, ccuComponent);
    devCcuConnection.Connect();
    RmaConnStatus status = devCcuConnection.GetStatus();

    MOCKER_CPP(&ChannelManager::ChannelConfig).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    RdmaHandle remoteRdmaHandle = (void*)0x2000000;

    EXPECT_NO_THROW(
        devCcuConnection.Bind(std::make_unique<RemoteUbRmaBuffer>(remoteRdmaHandle).get(), BufferType::INPUT));
    EXPECT_EQ(true, devCcuConnection.updateChannelFlag);
}

std::unique_ptr<DevCcuConnection> CounctorDevCcuCon(Socket* fakeSocket)
{
    RdmaHandle rdmaHandle = (void*)0x1000000;
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    ChannelInfo channelInfo;
    channelInfo.channelId = 0;
    channelInfo.jettys.push_back(JettyInfo());
    MOCKER_CPP(&CcuComponent::AllocChannel).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(channelInfo));
    MOCKER_CPP(&CcuComponent::CreateLocalOpChannel).stubs().will(ignoreReturnValue());

    CommunicatorImpl comm;
    CcuComponent ccuComponent(&comm);

    std::unique_ptr<DevCcuConnection> ccuConn
        = std::make_unique<DevCcuConnection>(fakeSocket, rdmaHandle, linkData, ccuComponent);
    return std::move(ccuConn);
}

TEST_F(DevCcuConnectionTest, dev_ccu_connection_prepare_read_throw_exception)
{
    auto devCcuConnection = CounctorDevCcuCon(fakeSocket);

    MemoryBuffer remoteMemBuf(0, 0, 0);
    MemoryBuffer localMemBuf(0, 0, 0);
    SqeConfig config;
    EXPECT_THROW(devCcuConnection->PrepareRead(remoteMemBuf, localMemBuf, config), NotSupportException);
}

TEST_F(DevCcuConnectionTest, dev_ccu_connection_prepare_read_reduce_throw_exception)
{
    MemoryBuffer remoteMemBuf(0, 0, 0);
    MemoryBuffer localMemBuf(0, 0, 0);
    SqeConfig config;
    DataType datatype = DataType::INT8;
    ReduceOp reduceOp = ReduceOp::SUM;

    auto devCcuConnection = CounctorDevCcuCon(fakeSocket);
    EXPECT_THROW(
        devCcuConnection->PrepareReadReduce(remoteMemBuf, localMemBuf, datatype, reduceOp, config),
        NotSupportException);
}

TEST_F(DevCcuConnectionTest, dev_ccu_connection_prepare_write_throw_exception)
{
    MemoryBuffer remoteMemBuf(0, 0, 0);
    MemoryBuffer localMemBuf(0, 0, 0);
    SqeConfig config;
    auto devCcuConnection = CounctorDevCcuCon(fakeSocket);
    EXPECT_THROW(devCcuConnection->PrepareWrite(remoteMemBuf, localMemBuf, config), NotSupportException);
}

TEST_F(DevCcuConnectionTest, dev_ccu_connection_prepare_write_reduce_throw_exception)
{
    MemoryBuffer remoteMemBuf(0, 0, 0);
    MemoryBuffer localMemBuf(0, 0, 0);
    SqeConfig config;
    DataType datatype = DataType::INT8;
    ReduceOp reduceOp = ReduceOp::SUM;

    auto devCcuConnection = CounctorDevCcuCon(fakeSocket);
    EXPECT_THROW(
        devCcuConnection->PrepareWriteReduce(remoteMemBuf, localMemBuf, datatype, reduceOp, config),
        NotSupportException);
}

TEST_F(DevCcuConnectionTest, dev_ccu_connection_suspend_change_status_suspend)
{
    // construct devCcuConnection
    auto devCcuConnection = CounctorDevCcuCon(fakeSocket);

    //  When:
    MOCKER(HrtRaUbUnimportJetty).stubs().with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtRaUbDestroyJetty).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtFree).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    devCcuConnection->status = RmaConnStatus::READY;
    devCcuConnection->ccuConnStatus = DevCcuConnection::CcuConnStatus::JETTY_IMPORTED;

    // Then
    bool ret = devCcuConnection->Suspend();
    EXPECT_EQ(false, ret);
    EXPECT_EQ(DevCcuConnection::CcuConnStatus::JETTY_UNIMPORTED, devCcuConnection->ccuConnStatus);

    devCcuConnection->ccuConnStatus = DevCcuConnection::CcuConnStatus::SEND_FINISHED;
    ret = devCcuConnection->Suspend();
    EXPECT_EQ(false, ret);
    EXPECT_EQ(DevCcuConnection::CcuConnStatus::JETTY_UNIMPORTED, devCcuConnection->ccuConnStatus);

    EXPECT_EQ(1, devCcuConnection->jettys.size());
    ret = devCcuConnection->Suspend();
    EXPECT_EQ(true, ret);
    EXPECT_EQ(RmaConnStatus::SUSPENDED, devCcuConnection->status);
    EXPECT_EQ(0, devCcuConnection->jettys.size());

    ret = devCcuConnection->Suspend();
    EXPECT_EQ(true, ret);
    EXPECT_EQ(RmaConnStatus::SUSPENDED, devCcuConnection->status);
}

TEST_F(DevCcuConnectionTest, dev_ccu_connection_suspend_fail_when_status_is_not_ready)
{
    // construct devCcuConnection
    auto devCcuConnection = CounctorDevCcuCon(fakeSocket);
    devCcuConnection->status = RmaConnStatus::INIT;
    // Then
    EXPECT_THROW(devCcuConnection->Suspend(), RmaConnException);
}
