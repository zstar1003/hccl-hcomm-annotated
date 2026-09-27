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
#include "dev_rdma_connection.h"
#include "rma_conn_manager.h"
#include "socket.h"
#include "orion_adapter_rts.h"
#include "not_support_exception.h"
#include "hccp.h"
#include "hccp_ctx.h"
#include "hccp_common.h"
#undef protected
#undef private
using namespace Hccl;

class DevRdmaConnectionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DevRdmaConnection tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "DevRdmaConnection tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in DevRdmaConnection SetUP" << std::endl;
        fakeSocket
            = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete fakeSocket;
        std::cout << "A Test case in DevRdmaConnection TearDown" << std::endl;
    }
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    std::string tag = "test";
};

TEST_F(DevRdmaConnectionTest, rma_net_connection_get_status_return_ok)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));

    RdmaHandle rdmaHandle = (void*)0x1000000;
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::string tag = "test";

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    QpHandle fakeQpHandle = (void*)0x1000000;
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));
    MOCKER(HrtRaQpConnectAsync).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    // construct DevRdmaConnection
    DevRdmaConnection devRdmaConnection(fakeSocket, rdmaHandle, OpMode::OPBASE);

    //  When: qp 建链成功
    MOCKER(RaQpConnectAsync).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(HrtGetRaQpStatus).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(1));

    RmaConnStatus status = devRdmaConnection.GetStatus();
    // Then
    EXPECT_EQ(RmaConnStatus::READY, status);
}

TEST_F(DevRdmaConnectionTest, rma_net_connection_construct_error)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));

    RdmaHandle rdmaHandle = (void*)0x1000000;
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::string tag = "test";

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_NOSOC));
    QpHandle fakeQpHandle = (void*)0x1000000;

    try {
        DevRdmaConnection devRdmaConnection(fakeSocket, rdmaHandle, OpMode::OPBASE);
    } catch (NotSupportException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_NOT_SUPPORT, e.GetErrorCode());
    }
}

TEST_F(DevRdmaConnectionTest, rma_net_connection_get_status_return_time_out)
{
    // Given: socket time out
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    // When
    DevRdmaConnection devRdmaConnection(fakeSocket, rdmaHandle, OpMode::OPBASE);

    // Then
    RmaConnStatus status = devRdmaConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::CONN_INVALID, status);
    EXPECT_EQ(DevRdmaConnection::RdmaConnStatus::SOCKET_TIMEOUT, devRdmaConnection.rdmaConnStatus);
    GlobalMockObject::verify();
}

TEST_F(DevRdmaConnectionTest, rma_net_connection_get_status_return_time_connecting)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));
    MOCKER(HrtRaQpConnectAsync).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    // When
    DevRdmaConnection devRdmaConnection(fakeSocket, rdmaHandle, OpMode::OPBASE);
    MOCKER(HrtGetRaQpStatus).stubs().with(mockcpp::any()).will(returnValue(0)).then(returnValue(1));

    // Then
    RmaConnStatus status = devRdmaConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::INIT, status);
    EXPECT_EQ(DevRdmaConnection::RdmaConnStatus::CONNECTING, devRdmaConnection.rdmaConnStatus);

    status = devRdmaConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::READY, status);
}

TEST_F(DevRdmaConnectionTest, rma_net_connection_get_handle)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    // When
    DevRdmaConnection devRdmaConnection(fakeSocket, rdmaHandle, OpMode::OPBASE);

    // Then
    QpHandle qpHandle = devRdmaConnection.GetHandle();
    EXPECT_EQ(fakeQpHandle, qpHandle);
}

TEST_F(DevRdmaConnectionTest, rma_net_connection_prepare_write_tasks)
{
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(portType, 0, 1, 0, 1);

    char targetChipVer[CHIP_VERSION_MAX_LEN] = "Ascend910B1";

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    DevRdmaConnection devRdmaConnection(fakeSocket, rdmaHandle, OpMode::OPBASE);

    // Given
    MemoryBuffer localMemBuffer1(0, 0, 0);
    MemoryBuffer remoteMemBuffer1(2000, 0, 0);
    SqeConfig config{};
    // When
    auto result1 = devRdmaConnection.PrepareWrite(remoteMemBuffer1, localMemBuffer1, config);
    // Then
    EXPECT_EQ(nullptr, result1);

    MemoryBuffer localMemBuffer2(0, 100, 0);
    MemoryBuffer remoteMemBuffer2(2000, 100, 0);
    auto result2 = devRdmaConnection.PrepareWrite(remoteMemBuffer2, localMemBuffer2, config);
    EXPECT_NE(nullptr, result2);
    EXPECT_EQ(TaskType::RDMA_SEND, result2->GetType());

    u64 size3 = 0x100000000;
    MemoryBuffer localMemBuffer3(0, size3, 0);
    MemoryBuffer remoteMemBuffer3(2000, size3, 0);
    auto result3 = devRdmaConnection.PrepareWrite(remoteMemBuffer3, localMemBuffer3, config);
    EXPECT_NE(nullptr, result3);
    EXPECT_EQ(TaskType::RDMA_SEND, result3->GetType());

    u64 size4 = 0x100000010;
    MemoryBuffer localMemBuffer4(0, size4, 0);
    MemoryBuffer remoteMemBuffer4(2000, size4, 0);
    auto result4 = devRdmaConnection.PrepareWrite(remoteMemBuffer4, localMemBuffer4, config);
    EXPECT_NE(nullptr, result4);
    EXPECT_EQ(TaskType::RDMA_SEND, result4->GetType());

    MemoryBuffer localMemBuffer10(0, 0, 0);
    MemoryBuffer remoteMemBuffer10(2000, 10, 0);
    EXPECT_THROW(devRdmaConnection.PrepareWrite(remoteMemBuffer10, localMemBuffer10, config), InvalidParamsException);
}

TEST_F(DevRdmaConnectionTest, rma_net_connection_prepare_read_tasks)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    // When
    DevRdmaConnection devRdmaConnection(fakeSocket, rdmaHandle, OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    EXPECT_THROW(devRdmaConnection.PrepareRead(remoteMemBuffer, localMemBuffer, config), NotSupportException);
}

TEST_F(DevRdmaConnectionTest, rma_net_connection_prepare_read_reduce_tasks)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    // When
    DevRdmaConnection devRdmaConnection(fakeSocket, rdmaHandle, OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    EXPECT_THROW(devRdmaConnection.PrepareRead(remoteMemBuffer, localMemBuffer, config), NotSupportException);
}

TEST_F(DevRdmaConnectionTest, rma_net_connection_prepare_write_reduce_tasks)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    // When
    DevRdmaConnection devRdmaConnection(fakeSocket, rdmaHandle, OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    EXPECT_THROW(
        devRdmaConnection.PrepareWriteReduce(remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, config),
        NotSupportException);
}

TEST_F(DevRdmaConnectionTest, rma_GetTaskNum_NOK)
{
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    // When
    DevRdmaConnection devRdmaConnection(fakeSocket, rdmaHandle, OpMode::OPBASE);

    MemoryBuffer localMemBuffer10{0, 0, 0};
    MemoryBuffer remoteMemBuffer10(2000, 10, 0);
    SqeConfig config{};
    EXPECT_THROW(devRdmaConnection.PrepareWrite(remoteMemBuffer10, localMemBuffer10, config), InvalidParamsException);
}
