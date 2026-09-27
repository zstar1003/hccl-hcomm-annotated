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
#include "dev_ub_connection.h"
#include "task.h"
#define private public
#define protected public
#include "ub_mem_transport.h"
#include "ub_local_notify.h"
#include "local_ub_rma_buffer.h"
#include "socket_exception.h"
#include "exchange_ub_conn_dto.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
#include "internal_exception.h"
#undef protected
#undef private
#include "stub_communicator_impl_trans_mgr.h"
#include "mem_transport_callback.h"
#include "dlprof_func_v2.h"
#include "rdma_handle_manager.h"

using namespace Hccl;

static int memcpy_stub(void* dest, int dest_max, const void* src, int count)
{
    memcpy(dest, src, count);
    return 0;
}

class StubUbRmaConnection : public DevUbConnection {
public:
    StubUbRmaConnection(LinkData& linkData)
        : link(linkData),
          DevUbConnection((void*)0x100, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE)
    {
        status = RmaConnStatus::READY;
    }

    unique_ptr<BaseTask>
    PrepareRead(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override
    {
        u32 jettyId = 100;
        u64 funcId = 100;
        u32 piVal = 100;
        u64 dieId = 100;
        return make_unique<TaskUbDbSend>(jettyId, funcId, piVal, dieId);
    }

    unique_ptr<BaseTask> PrepareReadReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config) override
    {
        u8 dwqe[64]{0};
        u32 jettyId = 100;
        u64 funcId = 100;
        u32 dwqeSize = 64;
        u64 dieId = 100;
        return make_unique<TaskUbDirectSend>(funcId, dieId, jettyId, dwqeSize, dwqe);
    }

    unique_ptr<BaseTask>
    PrepareWrite(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override
    {
        u32 jettyId = 100;
        u64 funcId = 100;
        u32 piVal = 100;
        u64 dieId = 100;
        return make_unique<TaskUbDbSend>(jettyId, funcId, piVal, dieId);
    }

    unique_ptr<BaseTask> PrepareWriteWithNotify(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, u64 data,
        const MemoryBuffer& remoteNotifyMemBuf, const SqeConfig& config) override
    {
        u32 jettyId = 100;
        u64 funcId = 100;
        u32 piVal = 100;
        u64 dieId = 100;
        return make_unique<TaskUbDbSend>(jettyId, funcId, piVal, dieId);
    }

    unique_ptr<BaseTask> PrepareWriteReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config) override
    {
        u32 jettyId = 100;
        u64 funcId = 100;
        u32 piVal = 100;
        u64 dieId = 100;
        return make_unique<TaskUbDbSend>(jettyId, funcId, piVal, dieId);
    }

    unique_ptr<BaseTask> PrepareInlineWriteReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config)
    {
        u32 jettyId = 100;
        u64 funcId = 100;
        u32 piVal = 100;
        u64 dieId = 100;
        return make_unique<TaskUbDbSend>(jettyId, funcId, piVal, dieId);
    }

    unique_ptr<BaseTask>
    PrepareInlineWrite(const MemoryBuffer& remoteMemBuf, u64 data, const SqeConfig& config) override
    {
        u64 dbAddr = 100;
        u32 piVal = 100;
        return make_unique<TaskWriteValue>(dbAddr, piVal);
    }

    unique_ptr<BaseTask> PrepareWriteReduceWithNotify(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        u64 data, const MemoryBuffer& remoteNotifyMemBuf, const SqeConfig& config) override
    {
        return nullptr;
    }

    string Describe() const override { return "StubUbRmaConnection"; }

    RmaConnStatus GetStatus() override { return status; }

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
        buffer.resize(size);
        memcpy(buffer.data(), sendBuf, size);
        return true;
    }

    static bool Recv(Socket* This, u8* recvBuf, u32 size)
    {
        if (buffer.size() < size) {
            return false;
        }
        memcpy(recvBuf, buffer.data(), size);
        return true;
    }

private:
    static std::vector<char> buffer;
};

std::vector<char> StubSocket::buffer;

class UbMemTransportTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UbMemTransportTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UbMemTransportTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in UbMemTransportTest SetUP" << std::endl;
        MOCKER(HrtMemAsyncCopy).stubs().with(mockcpp::any());
        MOCKER(HrtReduceAsync).stubs().with(mockcpp::any());
        std::pair<TokenIdHandle, uint32_t> fakeTokenInfo = std::make_pair(0x12345678, 1);
        MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));
        MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        fakeLocalOutParam.handle = fakeNotifyHandleAddr;
        memcpy_s(fakeLocalOutParam.key, HRT_UB_MEM_KEY_MAX_LEN, fakeKey, HRT_UB_MEM_KEY_MAX_LEN);
        fakeLocalOutParam.tokenId = fakeTokenId;
        fakeLocalOutParam.targetSegVa = fakeTargetSegVa;
        fakeLocalOutParam.keySize = fakeKeySize;
        MOCKER(HrtRaUbLocalMemReg).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeLocalOutParam));
        fakeRemoteOutParam.handle = fakeNotifyHandleAddr;
        fakeRemoteOutParam.targetSegVa = fakeTargetSegVa;
        MOCKER(HrtRaUbRemoteMemImport)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(fakeRemoteOutParam));
        MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
        MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));
        MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
        MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));

        MOCKER(HrtNotifyRecord).stubs().with(mockcpp::any());
        MOCKER(HrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any());
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in UbMemTransportTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    RmaBufferSlice locSlice;
    RmtRmaBufferSlice rmtSlice;
    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);

    HrtRaUbLocalMemRegOutParam fakeLocalOutParam;
    HrtRaUbRemMemImportedOutParam fakeRemoteOutParam;
    u64 fakeNotifyHandleAddr = 100;
    u64 fakeTargetSegVa = 150;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    u32 fakeTokenId = 100;
    u8 fakeKey[HRT_UB_MEM_KEY_MAX_LEN] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    u32 fakeKeySize = 10;
    bool isRecvFirst = false;
};

TEST_F(UbMemTransportTest, UbMemTransport_describe)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.Describe();
}

static HcclResult RecvDataProcessNeedFinishStub(UbMemTransport*, bool& needSendFinish)
{
    needSendFinish = true;
    return HCCL_SUCCESS;
}

static HcclResult RecvDataProcessNoFinishStub(UbMemTransport*, bool& needSendFinish)
{
    needSendFinish = false;
    return HCCL_SUCCESS;
}

TEST_F(UbMemTransportTest, UbMemTransport_get_status)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    MOCKER_CPP(&UbMemTransport::SendAll).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::RecvDataSize).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::SendExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::RecvExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::RecvDataProcess).stubs().will(invoke(RecvDataProcessNeedFinishStub));
    MOCKER_CPP(&UbMemTransport::SendFinish).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::RecvFinish).stubs().will(returnValue(HCCL_SUCCESS));

    StubSocket stubSocket;
    transport.socket = &stubSocket;

    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(SocketStatus(SocketStatus::OK)));

    int fakeFdStatus = SOCKET_CONNECTED;
    FdHandle fakeFdHandle = (void*)100;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus);

    MOCKER(RaGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    TransportStatus transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::SEND_DATA);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::RECV_SIZE);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::RECV_DATA);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::PROCESS_DATA);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::SEND_FIN);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::RECV_FIN);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::SET_READY);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::READY);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::READY);

    transport.baseStatus = TransportStatus::INIT;
    transport.ubStatus = UbMemTransport::UbStatus::INIT;
    transport.socket = &fakeSocket;
    GlobalMockObject::verify();

    MOCKER_CPP(&UbMemTransport::SendAll).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::RecvDataSize).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::SendExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::RecvExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::RecvDataProcess).stubs().will(invoke(RecvDataProcessNoFinishStub));
    MOCKER_CPP(&UbMemTransport::SendFinish).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::RecvFinish).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(SocketStatus(SocketStatus::OK)));
    MOCKER(RaGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::SEND_DATA);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::RECV_SIZE);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::RECV_DATA);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::SOCKET_OK);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::PROCESS_DATA);

    transStatus = transport.GetStatus();
    EXPECT_EQ(transStatus, TransportStatus::READY);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::READY);
}

TEST_F(UbMemTransportTest, UbMemTransport_send_recv_exchange_data)
{
    std::pair<TokenIdHandle, uint32_t> retPair = {1, 1};
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(retPair));

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubUbRmaConnection stubRmaConnection(link);
    RmaConnection* rmaConnection = &stubRmaConnection;
    locRes.connVec.push_back(rmaConnection);
    UbLocalNotify ubLocalNotify(rdmaHandle);
    BaseLocalNotify* validLocalNotify = &ubLocalNotify;
    locRes.notifyVec.push_back(validLocalNotify);
    LocalUbRmaBuffer ubLocalRmaBuffer(devBuf, rdmaHandle);
    LocalRmaBuffer* validLocalRmaBuffer = &ubLocalRmaBuffer;
    locRes.bufferVec.push_back(validLocalRmaBuffer);

    RtsCntNotify rtsCntNotify;
    LocalCntNotify localCntNotify(rdmaHandle, &rtsCntNotify);
    locCntRes.vec.push_back(&localCntNotify);
    locCntRes.desc.push_back('0');
    locCntRes.desc.push_back(0);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    EXPECT_THROW(transport.GetUniqueId(), InternalException);
    MOCKER(memcpy_s).stubs().with().will(invoke(memcpy_stub));
    MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(100));
    std::string fakeKeyDesc = "fakeKeyDesc";
    MOCKER(HrtRaGetKeyDescribe).stubs().will(returnValue(fakeKeyDesc));

    StubSocket stubSocket;
    transport.socket = &stubSocket;

    int fakeFdStatus = SOCKET_CONNECTED;
    FdHandle fakeFdHandle = (void*)100;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus);

    MOCKER(RaGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    int max_times = 10;
    while (!transport.IsSocketReady()) {
        if (max_times-- <= 0) {
            std::cout << "while loop retry max times." << std::endl;
            break;
        }
    }
    EXPECT_NO_THROW(transport.SendAll());

    max_times = 10;
    while (!transport.IsSocketReady()) {
        if (max_times-- <= 0) {
            std::cout << "while loop retry max times." << std::endl;
            break;
        }
    }
    EXPECT_NO_THROW(transport.RecvDataSize());
    max_times = 10;
    while (!transport.IsSocketReady()) {
        if (max_times-- <= 0) {
            std::cout << "while loop retry max times." << std::endl;
            break;
        }
    }
}

TEST_F(UbMemTransportTest, UbMemTransport_send_recv_finish)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    MOCKER(memcpy_s).stubs().will(returnValue(-1));
    EXPECT_THROW(transport.SendFinish(), SocketException);

    GlobalMockObject::verify();
    MOCKER(memcpy_s).stubs().with().will(invoke(memcpy_stub));
    MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(100));

    StubSocket stubSocket;
    transport.socket = &stubSocket;

    int fakeFdStatus = SOCKET_CONNECTED;
    FdHandle fakeFdHandle = (void*)100;
    RaSocketFdHandleParam fakeParam(fakeFdHandle, fakeFdStatus);

    MOCKER(RaGetOneSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeParam));

    int max_times = 10;
    while (!transport.IsSocketReady()) {
        if (max_times-- <= 0) {
            std::cout << "while loop retry max times." << std::endl;
            break;
        }
    }

    EXPECT_NO_THROW(transport.SendFinish());

    max_times = 10;
    while (!transport.IsSocketReady()) {
        if (max_times-- <= 0) {
            std::cout << "while loop retry max times." << std::endl;
            break;
        }
    }

    EXPECT_NO_THROW(transport.RecvFinish());
}

TEST_F(UbMemTransportTest, UbMemTransport_read_write_read_reduce_write_reduce)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubUbRmaConnection stubRmaConnection(link);
    RmaConnection* rmaConnection = &stubRmaConnection;
    locRes.connVec.push_back(rmaConnection);

    Stream stream;

    void* rdmaHandle = (void*)0x100;
    LocalUbRmaBuffer localRmaBuffer(devBuf, rdmaHandle);
    RemoteUbRmaBuffer remoteRmaBuffer(rdmaHandle);
    ;
    locSlice.buf = &localRmaBuffer;
    rmtSlice.buf = &remoteRmaBuffer;

    StubCommunicatorImplTransMgr comm;
    MemTransportCallback callback(link, comm.GetMirrorTaskManager());
    MOCKER_CPP(&DlProfFunc::isStubMode).stubs().will(returnValue(true));

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, callback);

    EXPECT_THROW(transport.Read(locSlice, rmtSlice, stream), NotSupportException);
    EXPECT_THROW(transport.Write(locSlice, rmtSlice, stream), NotSupportException);

    ReduceIn reduceIn(DataType::INT8, ReduceOp::MAX);
    EXPECT_THROW(transport.ReadReduce(locSlice, rmtSlice, reduceIn, stream), NotSupportException);
    EXPECT_THROW(transport.WriteReduce(locSlice, rmtSlice, reduceIn, stream), NotSupportException);
}

TEST_F(UbMemTransportTest, UbMemTransport_post_wait)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubUbRmaConnection stubRmaConnection(link);
    RmaConnection* rmaConnection = &stubRmaConnection;
    locRes.connVec.push_back(rmaConnection);

    void* rdmaHandle = (void*)0x100;
    UbLocalNotify ubLocalNotify(rdmaHandle);
    BaseLocalNotify* validLocalNotify = &ubLocalNotify;
    locRes.notifyVec.push_back(validLocalNotify);

    Stream stream;

    StubCommunicatorImplTransMgr comm;
    MemTransportCallback callback(link, comm.GetMirrorTaskManager());
    MOCKER_CPP(&DlProfFunc::isStubMode).stubs().will(returnValue(true));

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, callback);

    std::unique_ptr<RemoteUbRmaBuffer> remoteUbRmaBuffer = std::make_unique<RemoteUbRmaBuffer>(rdmaHandle);
    transport.rmtNotifyVec.push_back(std::move(remoteUbRmaBuffer));
    EXPECT_THROW(transport.Post(0, stream), NotSupportException);
    transport.Wait(0, stream, 0);
}

TEST_F(UbMemTransportTest, UbMemTransport_write_with_notify_write_reduce_with_notify)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubUbRmaConnection stubRmaConnection(link);
    RmaConnection* rmaConnection = &stubRmaConnection;
    locRes.connVec.push_back(rmaConnection);

    Stream stream;

    void* rdmaHandle = (void*)0x100;
    LocalUbRmaBuffer localRmaBuffer(devBuf, rdmaHandle);
    RemoteUbRmaBuffer remoteRmaBuffer(rdmaHandle);
    ;
    locSlice.buf = &localRmaBuffer;
    rmtSlice.buf = &remoteRmaBuffer;
    locSlice.size = devBuf->GetSize();
    rmtSlice.size = devBuf->GetSize();

    RmaBufferSlice emptyLocSlice;
    emptyLocSlice.size = 0;

    StubCommunicatorImplTransMgr comm;
    MemTransportCallback callback(link, comm.GetMirrorTaskManager());
    MOCKER_CPP(&DlProfFunc::isStubMode).stubs().will(returnValue(true));

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, callback);

    std::unique_ptr<RemoteUbRmaBuffer> remoteUbRmaBuffer0 = std::make_unique<RemoteUbRmaBuffer>(rdmaHandle);
    std::unique_ptr<RemoteUbRmaBuffer> remoteUbRmaBuffer1 = std::make_unique<RemoteUbRmaBuffer>(rdmaHandle);
    transport.rmtNotifyVec.push_back(std::move(remoteUbRmaBuffer0));
    transport.rmtNotifyVec.push_back(std::move(remoteUbRmaBuffer1));

    constexpr uint32_t NOTIFY_INDEX_FIN = 1;
    WithNotifyIn withNotify(TransportNotifyType::INVALID, NOTIFY_INDEX_FIN);

    // normal notify
    withNotify.notifyType_ = TransportNotifyType::NORMAL;
    EXPECT_THROW(transport.WriteWithNotify(locSlice, rmtSlice, withNotify, stream), NotSupportException);
    // write empty case
    EXPECT_THROW(transport.WriteWithNotify(emptyLocSlice, rmtSlice, withNotify, stream), NotSupportException);

    ReduceIn reduceIn(DataType::INT8, ReduceOp::MAX);

    std::unique_ptr<RemoteUbRmaBuffer> remoteUbRmaBuffer2 = std::make_unique<RemoteUbRmaBuffer>(rdmaHandle);
    std::unique_ptr<RemoteUbRmaBuffer> remoteUbRmaBuffer3 = std::make_unique<RemoteUbRmaBuffer>(rdmaHandle);
    transport.rmtCntNotifyVec.push_back(std::move(remoteUbRmaBuffer2));
    transport.rmtCntNotifyVec.push_back(std::move(remoteUbRmaBuffer3));

    transport.WriteReduceWithNotify(locSlice, rmtSlice, reduceIn, withNotify, stream);
    // write empty case
    EXPECT_THROW(
        transport.WriteReduceWithNotify(emptyLocSlice, rmtSlice, reduceIn, withNotify, stream), NotSupportException);

    // count notify
    withNotify.notifyType_ = TransportNotifyType::COUNT;
    EXPECT_THROW(transport.WriteWithNotify(locSlice, rmtSlice, withNotify, stream), NotSupportException);
    EXPECT_THROW(transport.WriteWithNotify(emptyLocSlice, rmtSlice, withNotify, stream), NotSupportException);

    transport.WriteReduceWithNotify(locSlice, rmtSlice, reduceIn, withNotify, stream);
    // write empty case
    EXPECT_THROW(
        transport.WriteReduceWithNotify(emptyLocSlice, rmtSlice, reduceIn, withNotify, stream), NotSupportException);

    // invalid notify type
    withNotify.notifyType_ = TransportNotifyType::INVALID;
    EXPECT_THROW(transport.WriteWithNotify(locSlice, rmtSlice, withNotify, stream), InternalException);

    EXPECT_THROW(transport.WriteReduceWithNotify(locSlice, rmtSlice, reduceIn, withNotify, stream), InternalException);
}

TEST_F(UbMemTransportTest, UbMemTransport_wait)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubUbRmaConnection stubRmaConnection(link);
    RmaConnection* rmaConnection = &stubRmaConnection;
    locRes.connVec.push_back(rmaConnection);
    UbLocalNotify ubLocalNotify(rdmaHandle);
    BaseLocalNotify* validLocalNotify = &ubLocalNotify;
    locRes.notifyVec.push_back(validLocalNotify);
    LocalUbRmaBuffer ubLocalRmaBuffer(devBuf, rdmaHandle);
    LocalRmaBuffer* validLocalRmaBuffer = &ubLocalRmaBuffer;
    locRes.bufferVec.push_back(validLocalRmaBuffer);

    RtsCntNotify rtsCntNotify;
    LocalCntNotify localCntNotify(rdmaHandle, &rtsCntNotify);
    locCntRes.vec.push_back(&localCntNotify);
    locCntRes.desc.push_back('0');
    locCntRes.desc.push_back(0);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.IsResReady();
}

TEST_F(UbMemTransportTest, UbMemTransport_ConnVecUnpackProc)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    StubUbRmaConnection stubRmaConnection(link);
    RmaConnection* rmaConnection = &stubRmaConnection;
    locRes.connVec.push_back(rmaConnection);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    u32 connNum = 1;
    transport.connNum = connNum;

    BinaryStream binaryStream;
    u32 rmtConnNum = connNum;
    binaryStream << rmtConnNum;
    u32 pos = 1;
    binaryStream << pos;

    ExchangeUbConnDto dto;
    dto.Serialize(binaryStream);
    bool temp = false;
    EXPECT_NO_THROW(transport.ConnVecUnpackProc(binaryStream, temp));
}

TEST_F(UbMemTransportTest, ut_UbMemTransport_GetRemoteMems_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    std::shared_ptr<DevBuffer> buffer0 = DevBuffer::Create(0x100, 0x100);
    strcpy(buffer0->memInfo_, "cclBuffer");
    LocalUbRmaBuffer ubLocalRmaBuffer0(buffer0, rdmaHandle);
    LocalRmaBuffer* validLocalRmaBuffer0 = &ubLocalRmaBuffer0;
    locRes.bufferVec.push_back(validLocalRmaBuffer0);

    std::shared_ptr<DevBuffer> buffer1 = DevBuffer::Create(0x101, 0x101);
    strcpy(buffer1->memInfo_, "buffer1");
    buffer1->memType_ = HcclMemType::HCCL_MEM_TYPE_HOST;
    LocalUbRmaBuffer ubLocalRmaBuffer1(buffer1, rdmaHandle);
    LocalRmaBuffer* validLocalRmaBuffer1 = &ubLocalRmaBuffer1;
    locRes.bufferVec.push_back(validLocalRmaBuffer1);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    BinaryStream binaryStream;
    EXPECT_NO_THROW(transport.BufferVecPack(binaryStream, transport.commonLocRes.bufferVec));
    EXPECT_NO_THROW(transport.RmtBufferVecUnpackProc(
        2, binaryStream, transport.rmtBufferVec, UbMemTransport::UbRmtBufType::BUFFER));

    CommMem* remoteMems;
    char** memInfos;
    u32 memNum;
    HcclResult ret = transport.GetRemoteMems(&memNum, &remoteMems, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string memInfo = memInfos[1];
    EXPECT_EQ(memInfo, "buffer1");
    EXPECT_EQ(remoteMems[1].type, CommMemType::COMM_MEM_TYPE_HOST);
    EXPECT_EQ(remoteMems[1].addr, (void*)0x101);
    EXPECT_EQ(remoteMems[1].size, (uint64_t)0x101);
}

TEST_F(UbMemTransportTest, ut_UbMemTransport_GetRemoteMems_When_bufferNumIs0_Expect_ReturnIsHCCL_SUCCESS)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    CommMem* remoteMems;
    char** memInfos;
    u32 memNum;
    HcclResult ret = transport.GetRemoteMems(&memNum, &remoteMems, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(UbMemTransportTest, ut_UbMemTransport_UpdateMemInfo_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    LocalUbRmaBuffer ubLocalRmaBuffer(devBuf, rdmaHandle);
    LocalRmaBuffer* validLocalRmaBuffer = &ubLocalRmaBuffer;
    locRes.bufferVec.push_back(validLocalRmaBuffer);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    BinaryStream stream0;
    EXPECT_NO_THROW(transport.BufferVecPack(stream0, transport.commonLocRes.bufferVec));
    EXPECT_NO_THROW(
        transport.RmtBufferVecUnpackProc(1, stream0, transport.rmtBufferVec, UbMemTransport::UbRmtBufType::BUFFER));

    std::shared_ptr<DevBuffer> buffer1 = DevBuffer::Create(0x101, 0x101);
    strcpy(buffer1->memInfo_, "buffer1");
    buffer1->memType_ = HcclMemType::HCCL_MEM_TYPE_HOST;
    LocalUbRmaBuffer ubLocalRmaBuffer1(buffer1, rdmaHandle);
    LocalRmaBuffer* validLocalRmaBuffer1 = &ubLocalRmaBuffer1;
    std::vector<LocalRmaBuffer*> bufferVecTemp{};
    bufferVecTemp.push_back(validLocalRmaBuffer1);

    BinaryStream stream1;
    transport.sendData.clear();
    EXPECT_NO_THROW(transport.BufferVecPack(stream1, bufferVecTemp));
    stream1.Dump(transport.sendData);
    transport.recvData = transport.sendData;

    SocketStatus fakeSocketStatus = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(fakeSocketStatus));
    MOCKER_CPP(&Socket::SendAsync).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&UbMemTransport::RecvDataSize).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UbMemTransport::RecvExchangeData).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = transport.UpdateMemInfo(bufferVecTemp);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    auto locBuffer1 = transport.commonLocRes.bufferVec[1];
    EXPECT_EQ(locBuffer1->GetAddr(), (uintptr_t)0x101);
    EXPECT_EQ(locBuffer1->GetSize(), (size_t)0x101);
    EXPECT_EQ(locBuffer1->GetBuf()->GetMemType(), HcclMemType::HCCL_MEM_TYPE_HOST);
    EXPECT_EQ(locBuffer1->GetBuf()->GetMemInfo(), "buffer1");
    auto rmtBuffer1 = transport.rmtBufferVec[1].get();
    EXPECT_EQ(rmtBuffer1->GetAddr(), (uintptr_t)0x101);
    EXPECT_EQ(rmtBuffer1->GetSize(), (u64)0x101);
    EXPECT_EQ(rmtBuffer1->GetMemType(), HcclMemType::HCCL_MEM_TYPE_HOST);
    EXPECT_EQ(rmtBuffer1->GetMemInfo(), "buffer1");
}

TEST_F(UbMemTransportTest, ut_UbMemTransport_UpdateMemInfo_When_SocketTimeout_Expect_ReturnIsHCCL_E_INTERNAL)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    std::shared_ptr<DevBuffer> buffer1 = DevBuffer::Create(0x101, 0x101);
    strcpy(buffer1->memInfo_, "buffer1");
    buffer1->memType_ = HcclMemType::HCCL_MEM_TYPE_HOST;
    LocalUbRmaBuffer ubLocalRmaBuffer1(buffer1, rdmaHandle);
    LocalRmaBuffer* validLocalRmaBuffer1 = &ubLocalRmaBuffer1;
    std::vector<LocalRmaBuffer*> bufferVecTemp{};
    bufferVecTemp.push_back(validLocalRmaBuffer1);

    SocketStatus fakeSocketStatus = SocketStatus::TIMEOUT;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(fakeSocketStatus));
    MOCKER_CPP(&Socket::SendAsync).stubs().will(ignoreReturnValue());

    HcclResult ret = transport.UpdateMemInfo(bufferVecTemp);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(UbMemTransportTest, ut_UbMemTransport_UpdateMemInfo_When_bufferNumIs0_Expect_ReturnIsHCCL_SUCCESS)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    std::vector<LocalRmaBuffer*> bufferVecTemp{};

    HcclResult ret = transport.UpdateMemInfo(bufferVecTemp);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(UbMemTransportTest, HandleInitStatus_SetToSendDataAndSocketOk)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::INIT;

    HcclResult ret = transport.HandleInitStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::SEND_DATA);
    EXPECT_EQ(transport.baseStatus, TransportStatus::SOCKET_OK);
}

TEST_F(UbMemTransportTest, HandleSendAllStatus_WhenResReady_SetToRecvSize)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubUbRmaConnection stubRmaConnection(link);
    locRes.connVec.push_back(&stubRmaConnection);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::SEND_DATA;

    MOCKER_CPP(&UbMemTransport::SendAll).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = transport.HandleSendAllStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::RECV_SIZE);
}

TEST_F(UbMemTransportTest, HandleSendAllStatus_WhenResNotReady_StaySendData)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubUbRmaConnection stubRmaConnection(link);
    RmaConnection* conn = &stubRmaConnection;
    conn->status = RmaConnStatus::INIT;
    locRes.connVec.push_back(conn);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::SEND_DATA;

    HcclResult ret = transport.HandleSendAllStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::SEND_DATA);
}

TEST_F(UbMemTransportTest, HandleRecvSizeStatus_SetToRecvData)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, false);
    transport.ubStatus = UbMemTransport::UbStatus::RECV_SIZE;

    MOCKER_CPP(&UbMemTransport::RecvDataSize).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = transport.HandleRecvSizeStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::RECV_DATA);
}

TEST_F(UbMemTransportTest, HandleRecvSizeStatus_WhenRecvFirst_SetToRecvData)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, true);
    transport.ubStatus = UbMemTransport::UbStatus::RECV_SIZE;

    MOCKER_CPP(&UbMemTransport::RecvDataSize).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = transport.HandleRecvSizeStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::RECV_DATA);
}

TEST_F(UbMemTransportTest, HandleRecvDataStatus_WhenNotRecvFirst_SetToProcessData)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, false);
    transport.ubStatus = UbMemTransport::UbStatus::RECV_DATA;

    MOCKER_CPP(&UbMemTransport::RecvExchangeData).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = transport.HandleRecvDataStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::PROCESS_DATA);
}

TEST_F(UbMemTransportTest, HandleRecvDataStatus_WhenRecvFirst_SetToSendData)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, true);
    transport.ubStatus = UbMemTransport::UbStatus::RECV_DATA;

    MOCKER_CPP(&UbMemTransport::RecvExchangeData).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = transport.HandleRecvDataStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::SEND_DATA);
}

TEST_F(UbMemTransportTest, HandleProcessDataStatus_WhenNeedSendFinish_SetToSendFin)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::PROCESS_DATA;

    MOCKER_CPP(&UbMemTransport::RecvDataProcess).stubs().will(invoke(RecvDataProcessNeedFinishStub));

    HcclResult ret = transport.HandleProcessDataStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::SEND_FIN);
}

TEST_F(UbMemTransportTest, HandleProcessDataStatus_WhenNoSendFinish_SetToReady)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::PROCESS_DATA;

    MOCKER_CPP(&UbMemTransport::RecvDataProcess).stubs().will(invoke(RecvDataProcessNoFinishStub));

    HcclResult ret = transport.HandleProcessDataStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::READY);
    EXPECT_EQ(transport.baseStatus, TransportStatus::READY);
}

TEST_F(UbMemTransportTest, HandleSendFinStatus_WhenConnsReady_SetToRecvFin)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubUbRmaConnection stubRmaConnection(link);
    locRes.connVec.push_back(&stubRmaConnection);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::SEND_FIN;

    MOCKER_CPP(&UbMemTransport::SendFinish).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = transport.HandleSendFinStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::RECV_FIN);
}

TEST_F(UbMemTransportTest, HandleSendFinStatus_WhenConnsNotReady_StaySendFin)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubUbRmaConnection stubRmaConnection(link);
    RmaConnection* conn = &stubRmaConnection;
    conn->status = RmaConnStatus::INIT;
    locRes.connVec.push_back(conn);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::SEND_FIN;
    transport.connNum = 1;

    HcclResult ret = transport.HandleSendFinStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::SEND_FIN);
}

TEST_F(UbMemTransportTest, HandleRecvFinStatus_SetToSetReady)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::RECV_FIN;

    MOCKER_CPP(&UbMemTransport::RecvFinish).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = transport.HandleRecvFinStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::SET_READY);
}

TEST_F(UbMemTransportTest, HandleSetReadyStatus_SetToReady)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::SET_READY;

    HcclResult ret = transport.HandleSetReadyStatus();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transport.ubStatus, UbMemTransport::UbStatus::READY);
    EXPECT_EQ(transport.baseStatus, TransportStatus::READY);
}

TEST_F(UbMemTransportTest, StatusMachine_WhenAlreadyReady_ReturnSuccess)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.baseStatus = TransportStatus::READY;

    HcclResult ret = transport.StatusMachine();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(UbMemTransportTest, StatusMachine_WhenSocketNotReady_ReturnInternalError)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.baseStatus = TransportStatus::INIT;
    transport.socket = nullptr;

    HcclResult ret = transport.StatusMachine();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(UbMemTransportTest, HandleRecvSizeStatus_WhenRecvDataSizeFail_ReturnError)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::RECV_SIZE;

    MOCKER_CPP(&UbMemTransport::RecvDataSize).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = transport.HandleRecvSizeStatus();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(UbMemTransportTest, HandleRecvDataStatus_WhenRecvExchangeDataFail_ReturnError)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::RECV_DATA;

    MOCKER_CPP(&UbMemTransport::RecvExchangeData).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = transport.HandleRecvDataStatus();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(UbMemTransportTest, HandleProcessDataStatus_WhenRecvDataProcessFail_ReturnError)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::PROCESS_DATA;

    MOCKER_CPP(&UbMemTransport::RecvDataProcess).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = transport.HandleProcessDataStatus();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(UbMemTransportTest, HandleSendFinStatus_WhenSendFinishFail_ReturnError)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    StubUbRmaConnection stubRmaConnection(link);
    locRes.connVec.push_back(&stubRmaConnection);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::SEND_FIN;

    MOCKER_CPP(&UbMemTransport::SendFinish).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = transport.HandleSendFinStatus();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(UbMemTransportTest, HandleRecvFinStatus_WhenRecvFinishFail_ReturnError)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport.ubStatus = UbMemTransport::UbStatus::RECV_FIN;

    MOCKER_CPP(&UbMemTransport::RecvFinish).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = transport.HandleRecvFinStatus();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(UbMemTransportTest, UbMemTransport_GetSingleLocBufferUniqueId)
{
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    UbMemTransport transport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    u64 addr = 0x1000;
    u64 size = 0x2000;
    u32 tokenId = 100;
    u32 tokenValue = 200;

    std::vector<char> result = transport.GetSingleLocBufferUniqueId(addr, size, tokenId, tokenValue);

    EXPECT_EQ(result.size(), sizeof(addr) + sizeof(size) + sizeof(tokenId) + sizeof(tokenValue));

    u64 outAddr = 0;
    u64 outSize = 0;
    u32 outTokenId = 0;
    u32 outTokenValue = 0;
    std::copy_n(result.data(), sizeof(outAddr), reinterpret_cast<char*>(&outAddr));
    std::copy_n(result.data() + sizeof(outAddr), sizeof(outSize), reinterpret_cast<char*>(&outSize));
    std::copy_n(
        result.data() + sizeof(outAddr) + sizeof(outSize), sizeof(outTokenId), reinterpret_cast<char*>(&outTokenId));
    std::copy_n(
        result.data() + sizeof(outAddr) + sizeof(outSize) + sizeof(outTokenId), sizeof(outTokenValue),
        reinterpret_cast<char*>(&outTokenValue));

    EXPECT_EQ(outAddr, addr);
    EXPECT_EQ(outSize, size);
    EXPECT_EQ(outTokenId, tokenId);
    EXPECT_EQ(outTokenValue, tokenValue);

    std::vector<char> rmtResult = transport.GetSingleRmtBufferUniqueId(addr, size, tokenId, tokenValue, 300);
    EXPECT_EQ(rmtResult.size(), result.size() + sizeof(u32));
}
