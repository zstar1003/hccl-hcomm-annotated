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
#include "op_type.h"
#define private public
#include "ub_memory_transport_mgr.h"
#include "timeout_exception.h"
#include "coll_operator_check.h"
#include "sal.h"
#include "network_api_exception.h"
#include "exchange_ipc_buffer_dto.h"
#include "orion_adapter_hccp.h"
#include "env_config_v2.h"
#include "ub_memory_transport.h"
#include "local_ipc_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "communicator_impl.h"
#undef private

using namespace Hccl;

class UbMemoryTransportMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UbMemoryTransportMgrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UbMemoryTransportMgrTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in UbMemoryTransportMgrTest SetUP" << std::endl;
        GlobalMockObject::verify();
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in UbMemoryTransportMgrTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

class StubAivSocket : public Socket {
public:
    StubAivSocket()
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

std::vector<char> StubAivSocket::buf;

TEST_F(UbMemoryTransportMgrTest, should_return_success_when_calling_TransportsConnect)
{
    cout << 1 << endl;
    CommunicatorImpl comm;
    comm.InitSocketManager();

    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::DEBUGCASE;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);

    UbMemoryTransportMgr transportManager(comm);

    std::string opTag = "test_tag";
    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    vector<LinkData> linkDatas;
    linkDatas.push_back(linkData);
    cout << 1 << endl;
    IpAddress ipAddress("1.0.0.0");
    shared_ptr<Socket> fakeSocket
        = make_shared<Socket>(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    SocketConfig socketConfig(linkData.GetRemoteRankId(), linkData, comm.GetEstablishLinkSocketTag());
    comm.GetSocketManager().connectedSocketMap[socketConfig] = std::move(fakeSocket);
    cout << 2 << endl;
    comm.cclBuffer = DevBuffer::Create(0x100, 10);
    comm.aivTagBuffer = DevBuffer::Create(0x100, 10);
    comm.aivOffloadTagBuffer = DevBuffer::Create(0x100, 10);
    cout << 3 << endl;
    char* testStr = "test";
    transportManager.BatchCreateTransport(linkDatas);
    cout << 4 << endl;
    MOCKER(HrtRaSocketBlockSend).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    MOCKER_CPP(&UbMemoryTransport::SendMemInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&UbMemoryTransport::RecvMemInfo).stubs().will(ignoreReturnValue());

    u64 res = 1;
    MOCKER(&CheckCollOperator).stubs().with().will(ignoreReturnValue());
    MOCKER(&HrtGetDevicePhyIdByUserDevId).stubs().with().will(returnValue(static_cast<DevId>(1)));
    ReqHandleResult result = ReqHandleResult::COMPLETED;
    MOCKER(&HrtRaGetAsyncReqResult).stubs().with().will(returnValue(result));
    transportManager.ubMemLink2TransportMap[linkData]->rmtHandshakeMsg = comm.GetCurrentCollOperator()->GetUniqueId();
    transportManager.TransportsConnect();
}

TEST_F(UbMemoryTransportMgrTest, should_return_success_when_calling_GetLocMemBuffer)
{
    std::shared_ptr<Buffer> cclBuffer = DevBuffer::Create(0x100, 10);

    unique_ptr<StubAivSocket> fakeSocket = make_unique<StubAivSocket>();

    DevId devLogicId = 0;
    std::unique_ptr<UbMemoryTransport> transport
        = make_unique<UbMemoryTransport>(cclBuffer, cclBuffer, cclBuffer, fakeSocket.get(), devLogicId);

    transport->localBufferVec.push_back(make_unique<LocalIpcRmaBuffer>(cclBuffer));

    EXPECT_NE(transport->GetLocMemBuffer(0), nullptr);

    EXPECT_THROW(transport->GetLocMemBuffer(1), InternalException);
}

TEST_F(UbMemoryTransportMgrTest, should_return_success_when_calling_GetRmtMemBuffer)
{
    std::shared_ptr<Buffer> cclBuffer = DevBuffer::Create(0x100, 10);

    unique_ptr<StubAivSocket> fakeSocket = make_unique<StubAivSocket>();

    DevId devLogicId = 0;
    std::unique_ptr<UbMemoryTransport> transport
        = make_unique<UbMemoryTransport>(cclBuffer, cclBuffer, cclBuffer, fakeSocket.get(), devLogicId);

    ExchangeIpcBufferDto dto;
    transport->rmtBufferVec.push_back(make_unique<RemoteIpcRmaBuffer>(dto, "UbMemory"));

    EXPECT_NE(transport->GetRmtMemBuffer(0), nullptr);

    EXPECT_THROW(transport->GetRmtMemBuffer(1), InternalException);
}

TEST_F(UbMemoryTransportMgrTest, should_UB_CONNECT_FAILED_when_calling_GetStatus_SOCKET_INIT)
{
    std::shared_ptr<Buffer> cclBuffer = DevBuffer::Create(0x100, 10);

    unique_ptr<StubAivSocket> fakeSocket = make_unique<StubAivSocket>();

    DevId devLogicId = 0;
    std::unique_ptr<UbMemoryTransport> transport
        = make_unique<UbMemoryTransport>(cclBuffer, cclBuffer, cclBuffer, fakeSocket.get(), devLogicId);

    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue((SocketStatus)SocketStatus::INIT));
    EXPECT_EQ(transport->GetStatus(), UbMemoryTransport::UBTransportStatus::CONNECT_FAILED);
}

TEST_F(UbMemoryTransportMgrTest, should_UB_TIMEOUT_when_calling_GetStatus_SOCKET_TIMEOUT)
{
    std::shared_ptr<Buffer> cclBuffer = DevBuffer::Create(0x100, 10);

    unique_ptr<StubAivSocket> fakeSocket = make_unique<StubAivSocket>();

    DevId devLogicId = 0;
    std::unique_ptr<UbMemoryTransport> transport
        = make_unique<UbMemoryTransport>(cclBuffer, cclBuffer, cclBuffer, fakeSocket.get(), devLogicId);

    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    EXPECT_EQ(transport->GetStatus(), UbMemoryTransport::UBTransportStatus::SOCKET_TIMEOUT);
}

TEST_F(UbMemoryTransportMgrTest, should_READY_when_calling_GetStatus_SOCKET_CONNECTING)
{
    std::shared_ptr<Buffer> cclBuffer = DevBuffer::Create(0x100, 10);

    unique_ptr<StubAivSocket> fakeSocket = make_unique<StubAivSocket>();

    DevId devLogicId = 0;
    std::unique_ptr<UbMemoryTransport> transport
        = make_unique<UbMemoryTransport>(cclBuffer, cclBuffer, cclBuffer, fakeSocket.get(), devLogicId);

    transport->ubStatus = UbMemoryTransport::UBTransportStatus::READY;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue((SocketStatus)SocketStatus::CONNECTING));
    EXPECT_EQ(transport->GetStatus(), UbMemoryTransport::UBTransportStatus::READY);
}

TEST_F(UbMemoryTransportMgrTest, ST_whenInput_param_should_create_remoteIpcRmaBuffer)
{
    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);

    u64 offset{0};
    u32 pid{0};
    char_t name[RTS_IPC_MEM_NAME_LEN]{0};
    BinaryStream binaryStream;
    binaryStream << devBuf->GetAddr() << devBuf->GetSize() << offset << pid << name;
    ExchangeIpcBufferDto dto;
    dto.Deserialize(binaryStream);
    MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(pid));
    RemoteIpcRmaBuffer remoteIpcRmaBuffer(dto);
}
