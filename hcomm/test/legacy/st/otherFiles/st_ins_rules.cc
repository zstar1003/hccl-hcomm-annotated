/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define protected public
#define private public
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "test_mock_setup.h"
#include "ins_rules.h"
#include "interpreter.h"
#include "instruction.h"
#include "p2p_transport.h"
#include "ub_mem_transport.h"
#include "conn_local_notify_manager.h"
#include "data_buf_manager.h"
#include "rma_connection.h"
#include "rma_conn_manager.h"
#include "socket_manager.h"
#include "socket.h"
#include "orion_adapter_hccp.h"
#include "local_rma_buffer.h"
#include "remote_rma_buffer.h"
#include "null_ptr_exception.h"
#include "communicator_impl.h"
#include "ccu_ins.h"
#include "ccu_ins_group.h"
#include "ccu_instruction_all_gather_mesh1d.h"
#include "ccu_communicator.h"
#include "coll_service_device_mode.h"
#include "aiv_ins.h"
#include "aiv_ins_preprocessor.h"
#include "aiv_temp_all_reduce_mesh_1D_oneshot.h"
#undef private
#undef protected

using namespace Hccl;

class StubRemoteRmaBuffer : public RemoteRmaBuffer {
public:
    StubRemoteRmaBuffer(u64 address, u64 address_size, RmaType rmaType) : RemoteRmaBuffer(rmaType)
    {
        addr = address;
        size = address_size;
    }
    string Describe() const override { return "StubRemoteRmaBuffer"; }
};

class StubLocalRmaBuffer : public LocalRmaBuffer {
public:
    StubLocalRmaBuffer(std::shared_ptr<Buffer> buf, RmaType rmaType) : LocalRmaBuffer(buf, rmaType) {}
    string Describe() const override { return "StubLocalRmaBuffer"; }
};

class StubCommunicatorImpl : public CommunicatorImpl {
public:
    StubCommunicatorImpl()
    {
        dataBufferManager = make_unique<DataBufManager>();

        localRmaBufManager = make_unique<LocalRmaBufManager>(*this);

        remoteRmaBufManager = make_unique<RemoteRmaBufManager>(*this);

        rmaConnectionManager = make_unique<RmaConnManager>(*this);

        connLocalCntNotifyManager = make_unique<ConnLocalCntNotifyManager>(this);

        queueNotifyManager = make_unique<QueueNotifyManager>(*this);

        queueWaitGroupCntNotifyManager = make_unique<QueueWaitGroupCntNotifyManager>();

        queueBcastPostCntNotifyManager = make_unique<QueueBcastPostCntNotifyManager>();

        memTransportManager = make_unique<MemTransportManager>(*this);
        devLogicId = 0;
        this->InitMirrorTaskManager();
        this->InitProfilingReporter();

        std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
        CollOperator op;
        op.opType = OpType::ALLREDUCE;
        op.staticAddr = false;
        dfxOpInfo->op_ = op;
        this->GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);

        currentCollOperator = make_unique<CollOperator>();
        currentCollOperator->opMode = OpMode::OPBASE;
        currentCollOperator->opTag = "op_base";
        currentCollOperator->inputMem = DevBuffer::Create(0x100, 0x100);
        currentCollOperator->outputMem = DevBuffer::Create(0x100, 0x100);
        currentCollOperator->scratchMem = DevBuffer::Create(0x100, 0x100);
    }

    void SetOp(OpMode opMode, string tag)
    {
        currentCollOperator->opMode = opMode;
        currentCollOperator->opTag = tag;
    }

    DataBufManager& GetDataBufferManager() const override { return *dataBufferManager.get(); }

    LocalRmaBufManager& GetLocalRmaBufManager() const override { return *localRmaBufManager.get(); }

    RemoteRmaBufManager& GetRemoteRmaBufManager() const override { return *remoteRmaBufManager.get(); }

    QueueNotifyManager& GetAicpuQueueNotifyManager() const override { return *aicpuQueueNotifyManager_.get(); }

    QueueNotifyManager& GetCcuQueueNotifyManager() const override { return *ccuQueueNotifyManager_.get(); }

    RmaConnManager& GetRmaConnManager() const override { return *rmaConnectionManager.get(); }

    CollOperator* GetCurrentCollOperator() const override { return currentCollOperator.get(); }

    ConnLocalCntNotifyManager& GetConnLocalCntNotifyManager() const override { return *connLocalCntNotifyManager; }

    MemTransportManager* GetMemTransportManager() const override { return memTransportManager.get(); }

private:
    unique_ptr<DataBufManager> dataBufferManager;
    unique_ptr<LocalRmaBufManager> localRmaBufManager;
    unique_ptr<RemoteRmaBufManager> remoteRmaBufManager;
    unique_ptr<QueueNotifyManager> queueNotifyManager;
    unique_ptr<ConnLocalNotifyManager> connLocalNotifyManager;
    unique_ptr<ConnLocalCntNotifyManager> connLocalCntNotifyManager;
    unique_ptr<StreamManager> streamManager;
    unique_ptr<SocketManager> socketManager;
    unique_ptr<RmaConnManager> rmaConnectionManager;
    unique_ptr<CollServiceBase> collService;
    unique_ptr<CollOperator> currentCollOperator;
    unique_ptr<MemTransportManager> memTransportManager;
};

class StubP2PTransport : public P2PTransport {
public:
    StubP2PTransport(CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket)
        : P2PTransport(commonLocRes, attr, linkData, socket)
    {
        stubRemoteRmaBuffer = std::make_unique<StubRemoteRmaBuffer>(remote_addr, remote_addr_len, RmaType::IPC);
    }

    void Wait(u32 index, const Stream& stream, u32 timeout) override {}

    void Post(u32 index, const Stream& stream) override {}

    void Read(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream) override {}

    void ReadReduce(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& ReduceIn,
        const Stream& stream) override
    {}

    void Write(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream) override {}

    void WriteReduce(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& ReduceIn,
        const Stream& stream) override
    {}

    RemoteRmaBuffer* GetRmtRmaBuffer(u32 index) override { return stubRemoteRmaBuffer.get(); }

    void SetRmtRmaBuffer(std::unique_ptr<RemoteRmaBuffer> rmtRmaBuffer)
    {
        stubRemoteRmaBuffer = std::move(rmtRmaBuffer);
    }

private:
    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;
    std::unique_ptr<RemoteRmaBuffer> stubRemoteRmaBuffer;
};

class StubUbMemTransport : public UbMemTransport {
public:
    StubUbMemTransport(
        CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
        RdmaHandle rdmaHandle1, LocCntNotifyRes& locCntNotifyRes1)
        : UbMemTransport(commonLocRes, attr, linkData, socket, rdmaHandle1, locCntNotifyRes1)
    {
        stubRemoteRmaBuffer = std::make_unique<StubRemoteRmaBuffer>(remote_addr, remote_addr_len, RmaType::UB);
    }

    void Wait(u32 index, const Stream& stream, u32 timeout) override {}

    void Post(u32 index, const Stream& stream) override {}

    void Read(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream) override {}

    void ReadReduce(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& ReduceIn,
        const Stream& stream) override
    {}

    void Write(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream) override {}

    void WriteReduce(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& ReduceIn,
        const Stream& stream) override
    {}

    void WriteWithNotify(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const WithNotifyIn& withNotify,
        const Stream& stream) override
    {}

    void WriteReduceWithNotify(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& ReduceIn,
        const WithNotifyIn& withNotify, const Stream& stream) override
    {}

    RemoteRmaBuffer* GetRmtRmaBuffer(u32 index) override { return stubRemoteRmaBuffer.get(); }

    void SetRmtRmaBuffer(std::unique_ptr<RemoteRmaBuffer> rmtRmaBuffer)
    {
        stubRemoteRmaBuffer = std::move(rmtRmaBuffer);
    }

private:
    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;
    std::unique_ptr<RemoteRmaBuffer> stubRemoteRmaBuffer;
};

class InsRulesTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InsRulesTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "InsRulesTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        MOCKER(HrtIpcOpenNotify).stubs().with(mockcpp::any()).will(returnValue((void*)fakeNotifyHandleAddr));
        MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
        MOCKER(HrtIpcSetNotifyName).stubs().with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any());
        MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));
        MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
        MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));

        std::cout << "A Test case in InsRulesTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();

        std::cout << "A Test case in InsRulesTest TearDown" << std::endl;
    }

    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
};

static IpAddress ipAddress("1.0.0.0");
static Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

TEST(InsRulesTest, Interpret_local_post_to)
{
    u32 fakeNotifyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    InsLocalPostTo insLocalPostToCounter(1, NotifyType::COUNTER, 0);
    insLocalPostToCounter.SetPostQid(0);

    MOCKER(HrtGetDeviceType).stubs().will(returnValue((Hccl::DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));
    MOCKER(HrtNotifyRecord).stubs().with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());

    RtsNotify* nullLocalNotify = nullptr;

    RtsNotify ipcLocalNotify;
    RtsNotify* validLocalNotify = &ipcLocalNotify;

    StubCommunicatorImpl fakeComm;

    MOCKER_CPP(&QueueNotifyManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(nullLocalNotify))
        .then(returnValue(validLocalNotify));

    InsLocalPostTo insLocalPostTo(1, NotifyType::NORMAL, 0);
    insLocalPostTo.SetPostQid(0);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};

    EXPECT_THROW(Interpret(insLocalPostTo, fakeComm, stream, taskConfig), NullPtrException);

    Interpret(insLocalPostTo, fakeComm, stream, taskConfig);

    RtsCntNotify* nullCntNotify = nullptr;
    RtsCntNotify rtsCntNotify;
    MOCKER_CPP(&QueueWaitGroupCntNotifyManager::Get)
        .stubs()
        .with()
        .will(returnValue(nullCntNotify))
        .then(returnValue(&rtsCntNotify));

    EXPECT_THROW(Interpret(insLocalPostToCounter, fakeComm, stream, taskConfig), NullPtrException);

    Interpret(insLocalPostToCounter, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_local_wait_from)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((Hccl::DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtNotifyWaitWithTimeOut).stubs();
    RtsNotify* nullLocalNotify = nullptr;

    RtsNotify ipcLocalNotify;
    RtsNotify* validLocalNotify = &ipcLocalNotify;

    StubCommunicatorImpl fakeComm;

    MOCKER_CPP(&QueueNotifyManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(nullLocalNotify))
        .then(returnValue(validLocalNotify));

    InsLocalWaitFrom insLocalWaitFrom(0, NotifyType::NORMAL, 0);
    insLocalWaitFrom.SetWaitQid(1);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};

    EXPECT_THROW(Interpret(insLocalWaitFrom, fakeComm, stream, taskConfig), NullPtrException);

    Interpret(insLocalWaitFrom, fakeComm, stream, taskConfig);
}

TEST(InsRulesTest, Interpret_write_with_fin_dev_net_ub_slice_is_not_zero_one_task)
{
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtIpcOpenNotify).stubs().with(mockcpp::any()).will(returnValue((void*)fakeNotifyHandleAddr));
    MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtIpcSetNotifyName).stubs().with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any());
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));
    MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));

    StubCommunicatorImpl fakeComm;

    // Given
    u32 remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, 0, 1, 0, 1);

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    void* rdmaHandle = (void*)0x100;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    bool isRecvFirst = false;
    StubUbMemTransport ubTransport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    BaseMemTransport* stubTransportPtr = &ubTransport;
    MOCK_GET_OPBASED_TRANSPORT(stubTransportPtr);

    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);

    InsWriteWithFin insWriteWithFin(remoteRank, link, localSlice, remoteSlice, NotifyType::NORMAL);

    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);
    printf("devBuf addr = %p\n", devBuf.get());
    StubLocalRmaBuffer stubLocalRmaBuffer(devBuf, RmaType::UB);
    LocalRmaBuffer* localRmaBuffer = &stubLocalRmaBuffer;
    LocalRmaBufManager localRmaBufManager(fakeComm);
    MOCKER_CPP(
        &LocalRmaBufManager::Get, LocalRmaBuffer * (LocalRmaBufManager::*)(const string&, const PortData&, BufferType))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(localRmaBuffer));

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    Interpret(insWriteWithFin, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_write_reduce_dev_net_ub_slice_is_not_zero_one_task)
{
    StubCommunicatorImpl fakeComm;

    // Given
    u32 remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, 0, 1, 0, 1);

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    void* rdmaHandle = (void*)0x100;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    bool isRecvFirst = false;
    StubUbMemTransport ubTransport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    BaseMemTransport* stubTransportPtr = &ubTransport;
    MOCK_GET_OPBASED_TRANSPORT(stubTransportPtr);

    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);

    InsWriteReduce insWriteReduce(remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM);

    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);
    std::cout << "devBuf addr = " << (u64)devBuf.get() << std::endl;
    StubLocalRmaBuffer stubLocalRmaBuffer(devBuf, RmaType::UB);
    LocalRmaBuffer* localRmaBuffer = &stubLocalRmaBuffer;
    LocalRmaBufManager localRmaBufManager(fakeComm);
    MOCKER_CPP(
        &LocalRmaBufManager::Get, LocalRmaBuffer * (LocalRmaBufManager::*)(const string&, const PortData&, BufferType))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(localRmaBuffer));

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    Interpret(insWriteReduce, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_write_reduce_with_fin_dev_net_ub_slice_is_not_zero_one_task)
{
    u64 fakeNotifyHandleAddr = 100;
    u64 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    MOCKER(HrtIpcOpenNotify).stubs().with(mockcpp::any()).will(returnValue((void*)fakeNotifyHandleAddr));
    MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtIpcSetNotifyName).stubs().with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any());
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));
    MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));

    StubCommunicatorImpl fakeComm;

    // Given
    u32 remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, 0, 1, 0, 1);

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    void* rdmaHandle = (void*)0x100;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    bool isRecvFirst = false;
    StubUbMemTransport ubTransport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    BaseMemTransport* stubTransportPtr = &ubTransport;
    MOCK_GET_OPBASED_TRANSPORT(stubTransportPtr);

    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);

    InsWriteReduceWithFin insWriteReduceWithFin(
        remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM, NotifyType::NORMAL);

    SETUP_LOCAL_RMA_BUF_MOCK(fakeComm);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    Interpret(insWriteReduceWithFin, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_write_with_fin_dev_net_ub_slice_is_zero_one_task)
{
    StubCommunicatorImpl fakeComm;

    // Given
    u32 remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, 0, 1, 0, 1);

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    void* rdmaHandle = (void*)0x100;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    bool isRecvFirst = false;
    StubUbMemTransport ubTransport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    BaseMemTransport* stubTransportPtr = &ubTransport;
    MOCK_GET_OPBASED_TRANSPORT(stubTransportPtr);

    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;

    DataSlice localSlice(BufferType::SCRATCH, 0, 0);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 0);

    InsWriteWithFin insWriteWithFin(remoteRank, link, localSlice, remoteSlice, NotifyType::NORMAL);

    SETUP_LOCAL_RMA_BUF_MOCK(fakeComm);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    Interpret(insWriteWithFin, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_read_reduce_dev_net_ub_slice_is_not_zero_one_task)
{
    StubCommunicatorImpl fakeComm;

    // Given
    u32 remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, 0, 1, 0, 1);

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    void* rdmaHandle = (void*)0x100;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    bool isRecvFirst = false;
    StubUbMemTransport ubTransport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    BaseMemTransport* stubTransportPtr = &ubTransport;
    MOCK_GET_OPBASED_TRANSPORT(stubTransportPtr);

    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);

    InsReadReduce insReadReduce(remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM);

    SETUP_LOCAL_RMA_BUF_MOCK(fakeComm);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    Interpret(insReadReduce, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_read_dev_net_ub_slice_is_not_zero_one_task)
{
    StubCommunicatorImpl fakeComm;

    // Given
    u32 remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, 0, 1, 0, 1);

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    void* rdmaHandle = (void*)0x100;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    bool isRecvFirst = false;
    StubUbMemTransport ubTransport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    BaseMemTransport* stubTransportPtr = &ubTransport;
    MOCK_GET_OPBASED_TRANSPORT(stubTransportPtr);

    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);

    InsRead insRead(remoteRank, link, localSlice, remoteSlice);

    SETUP_LOCAL_RMA_BUF_MOCK(fakeComm);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    Interpret(insRead, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_wait_group_fin)
{
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));

    CommunicatorImpl comm;
    comm.devLogicId = 0;
    comm.InitMirrorTaskManager();
    comm.InitProfilingReporter();
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.staticAddr = false;
    dfxOpInfo->op_ = op;
    comm.GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);

    comm.devPhyId = 0;
    comm.rmaConnectionManager = make_unique<RmaConnManager>(comm);
    comm.connLocalNotifyManager = make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = make_unique<ConnLocalCntNotifyManager>(&comm);

    vector<LinkData> links;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    links.push_back(link);
    auto insWaitGroupFin = make_unique<InsWaitGroupFin>();
    insWaitGroupFin->Append(link);
    comm.connLocalCntNotifyManager->ApplyFor(0, links);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};

    Interpret(*insWaitGroupFin, comm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_write_with_fin_dev_net_ub_slice_is_zero_one_task_cnt_notify)
{
    StubCommunicatorImpl fakeComm;

    // Given
    u32 remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, 0, 1, 0, 1);

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    void* rdmaHandle = (void*)0x100;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    bool isRecvFirst = false;
    StubUbMemTransport ubTransport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    BaseMemTransport* stubTransportPtr = &ubTransport;
    MOCK_GET_OPBASED_TRANSPORT(stubTransportPtr);

    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;

    DataSlice localSlice(BufferType::SCRATCH, 0, 0);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 0);

    InsWriteWithFin insWriteWithFin(remoteRank, link, localSlice, remoteSlice, NotifyType::COUNTER);

    SETUP_LOCAL_RMA_BUF_MOCK(fakeComm);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    Interpret(insWriteWithFin, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_write_reduce_with_fin_dev_net_ub_slice_is_zero_one_task_cnt_notify)
{
    StubCommunicatorImpl fakeComm;

    // Given
    u32 remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, 0, 1, 0, 1);

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    void* rdmaHandle = (void*)0x100;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    bool isRecvFirst = false;
    StubUbMemTransport ubTransport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    BaseMemTransport* stubTransportPtr = &ubTransport;
    MOCK_GET_OPBASED_TRANSPORT(stubTransportPtr);

    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;

    DataSlice localSlice(BufferType::SCRATCH, 0, 0);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 0);

    InsWriteReduceWithFin insWriteReduceWithFin(
        remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM, NotifyType::COUNTER);

    SETUP_LOCAL_RMA_BUF_MOCK(fakeComm);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    Interpret(insWriteReduceWithFin, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_write_p2p)
{
    StubCommunicatorImpl fakeComm;

    // Given
    u32 remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, 0, 1, 0, 1);

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    StubP2PTransport p2pTransport(locRes, attr, link, fakeSocket);

    BaseMemTransport* stubTransportPtr = &p2pTransport;
    MOCK_GET_OPBASED_TRANSPORT(stubTransportPtr);

    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);

    InsWrite insWrite(remoteRank, link, localSlice, remoteSlice);

    void* localAddr = (void*)100;
    DevBuffer devBuffer(100, 100);
    Buffer* buffer = &devBuffer;
    SETUP_DATA_BUF_MOCK(buffer);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    Interpret(insWrite, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_local_wait_group)
{
    StubCommunicatorImpl fakeComm;
    u32 fakeNotifyId = 1;
    u64 fakeNotifyHandleAddr = 100;

    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    RtsCntNotify* nullCntNotify = nullptr;
    RtsCntNotify rtsCntNotify;
    MOCKER_CPP(&QueueWaitGroupCntNotifyManager::Get)
        .stubs()
        .with()
        .will(returnValue(nullCntNotify))
        .then(returnValue(&rtsCntNotify));

    InsLocalWaitGroup insLocalWaitGroup(0);
    insLocalWaitGroup.Append(0);
    insLocalWaitGroup.Append(1);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    EXPECT_THROW(Interpret(insLocalWaitGroup, fakeComm, stream, taskConfig), NullPtrException);

    Interpret(insLocalWaitGroup, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_local_wait_from_cnt_notify)
{
    u32 fakeNotifyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    InsLocalWaitFrom insLocalWaitFrom(0, NotifyType::COUNTER, 0);
    insLocalWaitFrom.SetWaitQid(1);

    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    StubCommunicatorImpl fakeComm;

    Rts1ToNCntNotify* null1ToNCntNotify = nullptr;
    Rts1ToNCntNotify rts1ToNCntNotify;
    MOCKER_CPP(&QueueBcastPostCntNotifyManager::Get)
        .stubs()
        .with()
        .will(returnValue(null1ToNCntNotify))
        .then(returnValue(&rts1ToNCntNotify));

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    EXPECT_THROW(Interpret(insLocalWaitFrom, fakeComm, stream, taskConfig), NullPtrException);

    Interpret(insLocalWaitFrom, fakeComm, stream, taskConfig);
    GlobalMockObject::verify();
}

TEST(InsRulesTest, Interpret_local_bcast_post)
{
    StubCommunicatorImpl fakeComm;
    u32 fakeNotifyId = 1;
    u64 fakeNotifyHandleAddr = 100;

    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    Rts1ToNCntNotify* nullCntNotify = nullptr;
    Rts1ToNCntNotify rts1toNCntNotify;
    MOCKER_CPP(&QueueBcastPostCntNotifyManager::Get)
        .stubs()
        .with()
        .will(returnValue(nullCntNotify))
        .then(returnValue(&rts1toNCntNotify));

    InsLocalBcastPost insLocalBcastPost(0);
    insLocalBcastPost.Append(0);
    insLocalBcastPost.Append(1);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    EXPECT_THROW(Interpret(insLocalBcastPost, fakeComm, stream, taskConfig), NullPtrException);

    Interpret(insLocalBcastPost, fakeComm, stream, taskConfig);
}

TEST(InsRulesTest, Interpret_LocalReduce)
{
    StubCommunicatorImpl fakeComm;
    DataSlice srcSlice(BufferType::INPUT, 0, 100);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 100);
    InsLocalReduce insLocalReduce(srcSlice, dstSlice, DataType::FP32, ReduceOp::SUM);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    MOCKER(HrtMemAsyncCopy).stubs();
    MOCKER(HrtReduceAsync).stubs().with(mockcpp::any());
    // when
    Interpret(insLocalReduce, fakeComm, stream, taskConfig);
}

TEST(InsRulesTest, Interpret_local_copy)
{
    StubCommunicatorImpl fakeComm;

    u64 size = 100;
    DataSlice srcSlice(BufferType::SCRATCH, 0, size);
    DataSlice dstSlice(BufferType::SCRATCH, 100, size);
    InsLocalCopy insLocalCopy(srcSlice, dstSlice);

    DataSlice srcSlice2(BufferType::SCRATCH, 0, 0);
    DataSlice dstSlice2(BufferType::SCRATCH, 100, 0);
    InsLocalCopy insLocalCopy2(srcSlice2, dstSlice2);

    SETUP_INS_RULES_STREAM_MOCKS();
    ;
    Stream stream;
    OpTaskConfig taskConfig{};
    MOCKER(HrtMemAsyncCopy).stubs();

    // When
    Interpret(insLocalCopy, fakeComm, stream, taskConfig);
    Interpret(insLocalCopy2, fakeComm, stream, taskConfig);
}

HcclResult GetTaskParamStub(
    s32 deviceLogicId, CcuTaskArg& ccuTaskArg, const uint64_t executorId,
    std::vector<std::vector<CcuTaskParam>>& taskParam)
{
    taskParam.resize(5);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GetProfilingInfoStub(
    s32 deviceLogicId, CcuTaskArg& ccuTaskArg, const uint64_t executorId,
    std::vector<std::vector<CcuProfilingInfo>>& ccuProfilingInfo)
{
    std::vector<CcuProfilingInfo> profilingInfo;
    CcuProfilingInfo sqeProfInfo;
    sqeProfInfo.type = 0;
    sqeProfInfo.name = "AA::BB";
    profilingInfo.push_back(sqeProfInfo);

    CcuProfilingInfo localWaitProfInfo;
    localWaitProfInfo.type = 1;
    localWaitProfInfo.name = "LocalWait";
    profilingInfo.push_back(localWaitProfInfo);

    CcuProfilingInfo remoteWaitProfInfo;
    remoteWaitProfInfo.type = 1;
    remoteWaitProfInfo.name = "RemoteWait";
    profilingInfo.push_back(remoteWaitProfInfo);

    CcuProfilingInfo groupWaitProfInfo;
    groupWaitProfInfo.type = 1;
    groupWaitProfInfo.name = "GroupWait";
    profilingInfo.push_back(groupWaitProfInfo);

    CcuProfilingInfo groupReduceProfInfo;
    groupReduceProfInfo.type = 2;
    groupReduceProfInfo.name = "GroupReduce";
    profilingInfo.push_back(groupReduceProfInfo);

    CcuProfilingInfo gbProfInfo;
    gbProfInfo.type = 2;
    gbProfInfo.name = "GroupBroadcast";
    (void)memset_s(gbProfInfo.channelId, sizeof(gbProfInfo.channelId), 0x12, sizeof(gbProfInfo.channelId));
    profilingInfo.push_back(gbProfInfo);

    ccuProfilingInfo.push_back(profilingInfo);
    ccuProfilingInfo.push_back(profilingInfo);
    ccuProfilingInfo.push_back(profilingInfo);
    ccuProfilingInfo.push_back(profilingInfo);
    ccuProfilingInfo.push_back(profilingInfo);

    return HcclResult::HCCL_SUCCESS;
}

TEST(InsRulesTest, Interpret_ccu_instruction)
{
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    CommunicatorImpl comm;
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    comm.InitMirrorTaskManager();

    CcuInsGroup insGroup;
    std::unique_ptr<CcuInstruction> ins = std::make_unique<CcuInstructionAllGatherMesh1D>();
    insGroup.Append(std::move(ins));
    Stream stream;
    OpTaskConfig taskConfig{};
    comm.streamManager = std::make_unique<StreamManager>(&comm);
    MOCKER(CcuCtxMgr::GetTaskParam).stubs().will(invoke(GetTaskParamStub));
    MOCKER(CcuCtxMgr::GetProfilingInfo).stubs().will(invoke(GetProfilingInfoStub));
    MOCKER_CPP(&Hccl::CcuJettyMgr::GetRemoteRankIdByChannelId).stubs().with(mockcpp::any()).will(returnValue(0x23));
    MOCKER_CPP(&Hccl::MirrorTaskManager::AddTaskInfo).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    comm.streamManager->opbase = make_unique<OpbaseStreamManager>(&comm);
    comm.streamManager->opbase->master = make_unique<Stream>(&comm);
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.ccuStreamSyncNotifyManager = std::make_unique<CcuStreamSyncNotifyManager>();
    Interpret(insGroup, comm, stream, taskConfig);
}

TEST(InsRulesTest, Interpret_aiv_instruction)
{
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    CommunicatorImpl comm;
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    comm.InitMirrorTaskManager();
    comm.InitUbMemoryTransportMgr();

    std::vector<LinkData> links;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    links.push_back(link);
    AivOpArgs aivOpArgs;
    AivInstruction ins(links, aivOpArgs);

    rtStream_t fakePtr = nullptr;
    MOCKER(aclrtCreateStreamWithConfig)
        .stubs()
        .with(outBoundP(&fakePtr, sizeof(fakePtr)))
        .will(returnValue(ACL_SUCCESS));

    s32 fakeStreamId = 123;
    MOCKER(aclrtStreamGetId)
        .stubs()
        .with(mockcpp::any(), outBoundP(&fakeStreamId, sizeof(fakeStreamId)))
        .will(returnValue(ACL_SUCCESS));

    Stream stream;
    OpTaskConfig taskConfig{};
    comm.streamManager = std::make_unique<StreamManager>(&comm);

    comm.streamManager->opbase = make_unique<OpbaseStreamManager>(&comm);
    comm.streamManager->opbase->master = make_unique<Stream>(&comm);
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.cclBuffer = DevBuffer::Create(0x100, 10);
    comm.aivTagBuffer = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 0x100);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 0x100);
    MOCKER(HrtMemAsyncCopy).stubs();
    MOCKER(HrtMemcpy).stubs();
    Interpret(ins, comm, stream, taskConfig);
}
