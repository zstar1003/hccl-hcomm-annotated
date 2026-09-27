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
#include <mockcpp/MockObject.h>
#include "virtual_topo.h"
#include "p2p_connection.h"
#include "task.h"
#define protected public
#include "p2p_transport.h"
#undef protected
#include "data_type.h"
#include "reduce_op.h"
#include "stream.h"
#include "internal_exception.h"
#include "timeout_exception.h"
#include "socket_exception.h"
#include "local_ipc_rma_buffer_v2.h"
#include "ipc_local_notify.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
using namespace Hccl;

static int memcpy_stub(void* dest, int dest_max, const void* src, int count)
{
    memcpy(dest, src, count);
    return 0;
}

class StubP2PRmaConnection : public P2PConnection {
public:
    StubP2PRmaConnection(const LinkData& linkData) : link(linkData), P2PConnection(nullptr, linkData, "tag") {}

    unique_ptr<BaseTask>
    PrepareRead(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override
    {
        return make_unique<TaskP2pMemcpy>(localMemBuf.addr, remoteMemBuf.addr, localMemBuf.size, MemcpyKind::D2D);
    }

    unique_ptr<BaseTask> PrepareReadReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config) override
    {
        return make_unique<TaskSdmaReduce>(localMemBuf.addr, remoteMemBuf.addr, localMemBuf.size, datatype, reduceOp);
    }

    unique_ptr<BaseTask>
    PrepareWrite(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override
    {
        return make_unique<TaskP2pMemcpy>(remoteMemBuf.addr, localMemBuf.addr, localMemBuf.size, MemcpyKind::D2D);
    }

    unique_ptr<BaseTask> PrepareWriteReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config) override
    {
        return nullptr;
    }

    string Describe() const override { return "StubP2PRmaConnection"; }

    void Connect() override {}

private:
    LinkData link;
};

class StubSocket : public Socket {
public:
    StubSocket()
        : Socket(
              nullptr, IpAddress("1.0.0.0"), 0, IpAddress("1.0.0.0"), "tag", SocketRole::SERVER,
              NicType::DEVICE_NIC_TYPE)
    {
        MOCKER(HrtRaSocketBlockSend).stubs().will(invoke(Send));
        MOCKER(HrtRaSocketBlockRecv).stubs().will(invoke(Recv));
    }

    static bool Send(Socket* This, const u8* sendBuf, u32 size)
    {
        buf.resize(size);
        memcpy(buf.data(), sendBuf, size);
        // buf = const_cast<u8 *>(sendBuf);
        return true;
    }

    static bool Recv(Socket* This, u8* recvBuf, u32 size)
    {
        if (buf.size() < size) {
            return false;
        }
        memcpy(recvBuf, buf.data(), size);
        return true;
    }

private:
    static std::vector<char> buf;
};

std::vector<char> StubSocket::buf;

class P2PTransportTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "P2PTransport tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "P2PTransport tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in P2PTransport SetUP" << std::endl;
        MOCKER(HrtMemAsyncCopy).stubs().with(mockcpp::any());
        MOCKER(HrtReduceAsync).stubs().with(mockcpp::any());
        MOCKER(aclrtCreateStreamWithConfig).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)100));
        MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        MOCKER(HrtIpcOpenNotify).stubs().with(mockcpp::any()).will(returnValue((void*)fakeNotifyHandleAddr));
        MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
        MOCKER(HrtIpcSetNotifyName).stubs().with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any());
        MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));
        MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
        MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));

        MOCKER(HrtNotifyRecord).stubs().with(mockcpp::any());
        MOCKER(HrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any());
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in P2PTransport TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);

    RmaBufferSlice locSlice;
    RmtRmaBufferSlice rmtSlice;

    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
};

TEST_F(P2PTransportTest, P2PTransport_describe)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    P2PTransport transport(locRes, attr, link, fakeSocket);
    transport.Describe();
}

TEST_F(P2PTransportTest, P2PTransport_establish)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    P2PTransport transport(locRes, attr, link, fakeSocket);

    MOCKER_CPP(&P2PTransport::IsSocketReady).stubs().will(returnValue(true));
    EXPECT_NO_THROW(transport.Establish());
}

TEST_F(P2PTransportTest, P2PTransport_is_socket_ready)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    P2PTransport transport(locRes, attr, link, fakeSocket);

    SocketStatus socketStatusInit = SocketStatus::INIT;
    SocketStatus socketStatusOK = SocketStatus::OK;
    SocketStatus socketStatusTimeout = SocketStatus::TIMEOUT;
    MOCKER_CPP(&Socket::GetAsyncStatus)
        .stubs()
        .will(returnValue(socketStatusInit))
        .then(returnValue(socketStatusTimeout))
        .then(returnValue(socketStatusOK));

    EXPECT_FALSE(transport.IsSocketReady());
    EXPECT_THROW(transport.IsSocketReady(), TimeoutException);
    EXPECT_TRUE(transport.IsSocketReady());
    EXPECT_TRUE(transport.IsSocketReady()); // baseStatus 为 SOCKET_OK 时，直接返回 true

    transport.socket = nullptr;
    EXPECT_THROW(transport.IsSocketReady(), InternalException);
}

TEST_F(P2PTransportTest, P2PTransport_get_status)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    P2PTransport transport(locRes, attr, link, fakeSocket);

    MOCKER_CPP(&P2PTransport::SendPid).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&P2PTransport::RecvPid).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&P2PTransport::Grant).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&P2PTransport::SendExchangeData).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&P2PTransport::RecvExchangeData).stubs().will(ignoreReturnValue());

    SocketStatus socketStatusInit = SocketStatus::INIT;
    SocketStatus socketStatusOK = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(socketStatusInit)).then(returnValue(socketStatusOK));

    StubSocket stubSocket;
    transport.socket = &stubSocket;

    // 首次建链
    TransportStatus transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::INIT);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::INIT);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::SOCKET_OK);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::SEND_PID);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::RECV_PID);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::GRANT);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::SEND_DATA);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::READY);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::RECV_DATA);

    // 复位并重新打桩
    transport.baseStatus = TransportStatus::INIT;
    transport.p2pStatus = P2PTransport::P2PStatus::INIT;
    transport.socket = &fakeSocket;
    GlobalMockObject::verify();
    MOCKER_CPP(&P2PTransport::SendPid).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&P2PTransport::RecvPid).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&P2PTransport::Grant).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&P2PTransport::SendExchangeData).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&P2PTransport::RecvExchangeData).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue(socketStatusInit)).then(returnValue(socketStatusOK));

    // 增量建链
    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::INIT);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::INIT);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::SOCKET_OK);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::GRANT);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::SEND_DATA);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::READY);
    EXPECT_EQ(transport.p2pStatus, P2PTransport::P2PStatus::RECV_DATA);
}

TEST_F(P2PTransportTest, P2PTransport_send_recv_pid_and_grant)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubP2PRmaConnection stubRmaConnection(link);
    RmaConnection* rmaConnection = &stubRmaConnection;
    locRes.connVec.push_back(rmaConnection);
    IpcLocalNotify ipcLocalNotify;
    BaseLocalNotify* validLocalNotify = &ipcLocalNotify;
    locRes.notifyVec.push_back(validLocalNotify);
    LocalIpcRmaBuffer ipcLocalRmaBuffer(devBuf);
    LocalRmaBuffer* validLocalRmaBuffer = &ipcLocalRmaBuffer;
    locRes.bufferVec.push_back(validLocalRmaBuffer);

    P2PTransport transport(locRes, attr, link, fakeSocket);

    StubSocket stubSocket;
    transport.socket = &stubSocket;

    // transport自己给自己发送
    EXPECT_NO_THROW(transport.SendPid());
    EXPECT_NO_THROW(transport.RecvPid());
    EXPECT_EQ(transport.myPid, transport.rmtPid);

    MOCKER_CPP(&IpcLocalNotify::Grant).stubs();
    MOCKER_CPP(&LocalIpcRmaBuffer::Grant).stubs();

    EXPECT_NO_THROW(transport.Grant());
}

TEST_F(P2PTransportTest, P2PTransport_send_recv_exchange_data)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubP2PRmaConnection stubRmaConnection(link);
    RmaConnection* rmaConnection = &stubRmaConnection;
    locRes.connVec.push_back(rmaConnection);
    IpcLocalNotify ipcLocalNotify;
    BaseLocalNotify* validLocalNotify = &ipcLocalNotify;
    locRes.notifyVec.push_back(validLocalNotify);
    LocalIpcRmaBuffer ipcLocalRmaBuffer(devBuf);
    LocalRmaBuffer* validLocalRmaBuffer = &ipcLocalRmaBuffer;
    locRes.bufferVec.push_back(validLocalRmaBuffer);

    P2PTransport transport(locRes, attr, link, fakeSocket);

    MOCKER(memcpy_s).stubs().with().will(invoke(memcpy_stub));
    MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(100));

    StubSocket stubSocket;
    transport.socket = &stubSocket;

    // transport自己给自己发送
    EXPECT_NO_THROW(transport.SendExchangeData());
    EXPECT_NO_THROW(transport.RecvExchangeData());
    EXPECT_STREQ(transport.attr.handshakeMsg.data(), transport.rmtHandshakeMsg.data());
}

TEST_F(P2PTransportTest, P2PTransport_read_write_read_reduce_write_reduce)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubP2PRmaConnection stubRmaConnection(link);
    RmaConnection* rmaConnection = &stubRmaConnection;
    locRes.connVec.push_back(rmaConnection);

    Stream stream;

    P2PTransport transport(locRes, attr, link, fakeSocket);
    transport.Read(locSlice, rmtSlice, stream);
    transport.Write(locSlice, rmtSlice, stream);

    ReduceIn reduceIn(DataType::INT8, ReduceOp::MAX);
    transport.ReadReduce(locSlice, rmtSlice, reduceIn, stream);
    transport.WriteReduce(locSlice, rmtSlice, reduceIn, stream);
}

TEST_F(P2PTransportTest, P2PTransport_post_wait)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    IpcLocalNotify ipcLocalNotify;
    BaseLocalNotify* validLocalNotify = &ipcLocalNotify;
    locRes.notifyVec.push_back(validLocalNotify);

    std::unique_ptr<IpcRemoteNotify> ipcRemoteNotify = std::make_unique<IpcRemoteNotify>();
    Stream stream;

    P2PTransport transport(locRes, attr, link, fakeSocket);
    transport.rmtNotifyVec.push_back(std::move(ipcRemoteNotify));

    transport.Post(0, stream);
    transport.Wait(0, stream, 0);
}
