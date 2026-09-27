/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <chrono>
#include <thread>
#include "socket_manager.h"
#include "rma_conn_manager.h"
#include "rma_connection.h"
#include "p2p_connection.h"
#include "types.h"
#include "socket.h"
#include "context.h"
#include "communicator_impl.h"
#include "virtual_topo.h"
#include "op_mode.h"
#include "coll_operator.h"
#include "json_parser.h"
#include "rdma_handle_manager.h"
#include "dev_rdma_connection.h"
#include "rank_table.h"
#include "timeout_exception.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "ccu_component.h"

#undef private

using namespace Hccl;

class RmaConnManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmaConnManagerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RmaConnManagerTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RmaConnManagerTest SetUP" << std::endl;
        CommunicatorImpl comm;
        socketManager = new SocketManager(comm, 0, 1, 0, mockProducer);
        hccpSocketHandle = new int(0);
        MOCKER(HrtRaSocketInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(hccpSocketHandle));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete socketManager;
        delete hccpSocketHandle;
        std::cout << "A Test case in RmaConnManagerTest TearDown" << std::endl;
    }

    static std::unique_ptr<Socket> mockProducer(
        IpAddress& localIpAddress, IpAddress& remoteIpAddress, u32 listenPort, SocketHandle socketHandle,
        const std::string& tag, SocketRole socketRole, NicType nicType)
    {
        return std::make_unique<Socket>(
            socketHandle, localIpAddress, listenPort, remoteIpAddress, "stub", SocketRole::CLIENT,
            NicType::DEVICE_NIC_TYPE);
    }

    IpAddress GetAnIpAddress()
    {
        IpAddress ipAddress("1.0.0.0");
        return ipAddress;
    }

    Hccl::SocketManager* socketManager;
    void* hccpSocketHandle;
    IpAddress localIp;
    IpAddress remoteIp;
};

TEST_F(
    RmaConnManagerTest,
    should_return_validptr_or_nullptr_when_calling_create_get_before_after_release_destroy_with_valid_params)
{
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData* linkData = new LinkData(basePortType, 0, 1, 0, 1);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    u32 localRank = 0;
    u32 remoteRank = 1;

    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl.rankGraph = make_unique<RankGraph>(0);
    impl.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    impl.Init(commParams, "ranktable.json", config);

    RmaConnManager rmaConnManager(impl);

    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, *linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .will(returnValue((SocketStatus)SocketStatus::INIT))
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    auto res1 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, res1);
    auto res2 = rmaConnManager.Get(commParams.commId, *linkData);
    EXPECT_NE(nullptr, res2);

    rmaConnManager.Release(commParams.commId, *linkData);
    auto res3 = rmaConnManager.Get(commParams.commId, *linkData);
    EXPECT_EQ(nullptr, res3);

    uint64_t a = 10;
    uintptr_t devAddr = reinterpret_cast<uintptr_t>(&a);
    std::size_t devSize = 2;
    impl.cclBuffer = make_shared<DevBuffer>(10);
    impl.CovertToCurrentCollOperator(commParams.commId, collOpParams, OpMode::OPBASE);

    auto p2pConn2 = rmaConnManager.Create(commParams.commId, *linkData);
    auto res4 = rmaConnManager.Get(commParams.commId, *linkData);
    EXPECT_NE(nullptr, res4);
    rmaConnManager.Release(commParams.commId, *linkData);
    auto res5 = rmaConnManager.Get(commParams.commId, *linkData);
    EXPECT_EQ(nullptr, res5);

    auto p2pConn3 = rmaConnManager.Create(commParams.commId, *linkData);
    auto res6 = rmaConnManager.Get(commParams.commId, *linkData);
    EXPECT_NE(nullptr, res6);
    rmaConnManager.Release(commParams.commId, *linkData);
    auto res7 = rmaConnManager.Get(commParams.commId, *linkData);
    EXPECT_EQ(nullptr, res7);

    auto p2pConn4 = rmaConnManager.Create(commParams.commId, *linkData);
    auto res8 = rmaConnManager.Get(commParams.commId, *linkData);
    EXPECT_NE(nullptr, res8);
    rmaConnManager.Destroy();
    auto res9 = rmaConnManager.Get(commParams.commId, *linkData);
    EXPECT_EQ(nullptr, res9);

    linkData->type = PortDeploymentType::DEV_NET;
    linkData->linkProtocol_ = LinkProtocol::UB_CTP;
    auto ubConn = rmaConnManager.Create(commParams.commId, *linkData);
    auto res10 = rmaConnManager.Get(commParams.commId, *linkData);
    EXPECT_NE(nullptr, res10);

    DelRankTableFile();
    delete linkData;
    delete devPtr;
    delete socket;
}

TEST_F(RmaConnManagerTest, should_return_normally_when_calling_addwhitelist_with_valid_param)
{
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    u32 localRank = 0;
    u32 remoteRank = 1;

    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl.rankGraph = make_unique<RankGraph>(0);
    impl.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);
    MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    HcclCommConfig config;
    impl.Init(commParams, "ranktable.json", config);
    RmaConnManager rmaConnManager(impl);

    MOCKER_CPP(&SocketHandleManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(hccpSocketHandle));

    MOCKER(HrtRaSocketWhiteListAdd).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any());

    DelRankTableFile();
    delete devPtr;
}

TEST_F(RmaConnManagerTest, should_return_valid_ptr_when_calling_creatermadevnetconn_with_valid_param)
{
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData* linkData = new LinkData(basePortType, 0, 1, 0, 1);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    u32 localRank = 0;
    u32 remoteRank = 1;

    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    HcclCommConfig config;
    commParams.rankSize = 8;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    void* devMemPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devMemPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl.rankGraph = make_unique<RankGraph>(0);
    impl.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);

    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    impl.Init(commParams, "ranktable.json", config);

    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    collOpParams.sendBuf = nullptr;
    collOpParams.recvBuf = nullptr;
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    impl.cclBuffer = make_shared<DevBuffer>(10);
    impl.CovertToCurrentCollOperator(commParams.commId, collOpParams, OpMode::OPBASE);
    RmaConnManager rmaConnManager(impl);

    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, *linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .will(returnValue((SocketStatus)SocketStatus::INIT))
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    RdmaHandle rdmaHandle;
    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));

    QpHandle qpHandle;
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(qpHandle));

    DevBuffer devBuffer(100, 100);
    Buffer* buffer = &devBuffer;
    MOCKER_CPP(&DataBufManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(buffer));

    MOCKER(HrtRaMrReg).stubs().with(mockcpp::any(), mockcpp::any());

    auto res1 = rmaConnManager.CreateRdmaConn(socket, commParams.commId, *linkData);
    EXPECT_NE(nullptr, res1);

    DelRankTableFile();
    delete linkData;
    delete devMemPtr;
    delete socket;
}

TEST_F(RmaConnManagerTest, should_return_valid_ptr_when_calling_createdevubconn_with_valid_param)
{
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    u32 localRank = 0;
    u32 remoteRank = 1;

    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    void* devMemPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devMemPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl.rankGraph = make_unique<RankGraph>(0);
    impl.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);

    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    impl.Init(commParams, "ranktable.json", config);

    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    collOpParams.sendBuf = nullptr;
    collOpParams.recvBuf = nullptr;
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    impl.cclBuffer = make_shared<DevBuffer>(10);
    impl.CovertToCurrentCollOperator(commParams.commId, collOpParams, OpMode::OPBASE);
    RmaConnManager rmaConnManager(impl);

    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .will(returnValue((SocketStatus)SocketStatus::INIT))
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    RdmaHandle rdmaHandle = (void*)0x100;
    cout << 0 << endl;
    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));

    QpHandle qpHandle;
    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(qpHandle));

    auto res1 = rmaConnManager.CreateUbConn(socket, commParams.commId, linkData);
    cout << 1 << endl;
    EXPECT_NE(nullptr, res1);

    DelRankTableFile();
    delete devMemPtr;
    delete socket;
}

TEST_F(RmaConnManagerTest, should_no_throw_when_calling_BatchCreate)
{
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    LinkData linkData2(portType, 0, 2, 0, 1);
    string opTag = "test";

    // When
    auto devUbConn
        = make_unique<DevUbConnection>(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    CommunicatorImpl comm;
    comm.remoteRmaBufManager = std::make_unique<RemoteRmaBufManager>(comm);
    RmaConnManager rmaConnManager(comm);
    rmaConnManager.rmaConnectionMap[opTag][linkData] = std::move(devUbConn);
    auto devUbConn2
        = make_unique<DevUbConnection>(rdmaHandle, linkData2.GetLocalAddr(), linkData2.GetRemoteAddr(), OpMode::OPBASE);
    rmaConnManager.rmaConnectionMap[opTag][linkData2] = std::move(devUbConn2);

    auto newUbConn
        = make_unique<DevUbConnection>(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    auto newUbConn2
        = make_unique<DevUbConnection>(rdmaHandle, linkData2.GetLocalAddr(), linkData2.GetRemoteAddr(), OpMode::OPBASE);

    MOCKER_CPP_VIRTUAL(*newUbConn, &DevUbConnection::GetStatus)
        .stubs()
        .will(returnValue((RmaConnStatus)RmaConnStatus::READY));
    MOCKER_CPP_VIRTUAL(*newUbConn2, &DevUbConnection::GetStatus)
        .stubs()
        .will(returnValue((RmaConnStatus)RmaConnStatus::READY));
    MOCKER_CPP(&RmaConnManager::Create)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<RmaConnection*>(newUbConn.get())))
        .then(returnValue(static_cast<RmaConnection*>(newUbConn2.get())));

    // Then
    std::vector<LinkData> links = {linkData, linkData2};
    EXPECT_NO_THROW(rmaConnManager.BatchCreate(links));
}

TEST_F(RmaConnManagerTest, Get_allDTO_return_empty)
{
    CommunicatorImpl comm;

    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = 1;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;

    uint64_t a = 10;
    uintptr_t devAddr = reinterpret_cast<uintptr_t>(&a);
    std::size_t devSize = 2;
    comm.cclBuffer = make_shared<DevBuffer>(10);
    string tag = "optag";
    comm.CovertToCurrentCollOperator(tag, collOpParams, OpMode::OPBASE);

    RmaConnManager rmaConnManager(comm);
}

TEST_F(RmaConnManagerTest, CovertToCurrentCollOperator)
{
    CommunicatorImpl comm;

    HcclSendRecvItem sendRecvInfo;
    sendRecvInfo.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    CollOpParams collOpParams;
    collOpParams.opType = OpType::BATCHSENDRECV;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = 1;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    collOpParams.batchSendRecvDataDes.sendRecvItemsPtr = static_cast<void*>(&sendRecvInfo);

    uint64_t a = 10;
    uintptr_t devAddr = reinterpret_cast<uintptr_t>(&a);
    std::size_t devSize = 2;
    comm.cclBuffer = make_shared<DevBuffer>(10);
    string tag = "optag";
    comm.CovertToCurrentCollOperator(tag, collOpParams, OpMode::OPBASE);
}

TEST_F(RmaConnManagerTest, should_delete_all_when_calling_batchdeletejettys_with_valid_param)
{
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData* linkData = new LinkData(basePortType, 0, 1, 0, 1);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    u32 localRank = 0;
    u32 remoteRank = 1;

    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl.rankGraph = make_unique<RankGraph>(0);
    impl.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    impl.Init(commParams, "ranktable.json", config);

    RmaConnManager rmaConnManager(impl);

    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, *linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .will(returnValue((SocketStatus)SocketStatus::INIT))
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    auto p2pConn1 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn1);
    auto p2pConn2 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn2);
    auto p2pConn3 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn3);

    linkData->type = PortDeploymentType::DEV_NET;
    linkData->linkProtocol_ = LinkProtocol::UB_CTP;
    auto ubConn = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, ubConn);

    ReqHandleResult result = ReqHandleResult::COMPLETED;
    MOCKER(&HrtRaGetAsyncReqResult).stubs().with().will(returnValue(result));
    BatchDeleteJettys();
    DelRankTableFile();
    delete linkData;
    delete devPtr;
    delete socket;
}

ReqHandleResult HrtRaGetAsyncReqResult_Uncompleted(RequestHandle& reqHandle) { return ReqHandleResult::NOT_COMPLETED; }

TEST_F(RmaConnManagerTest, should_delete_all_when_calling_batchdeletejettys_with_invalid_param_2)
{
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData* linkData = new LinkData(basePortType, 0, 1, 0, 1);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    u32 localRank = 0;
    u32 remoteRank = 1;

    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl.rankGraph = make_unique<RankGraph>(0);
    impl.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    impl.Init(commParams, "ranktable.json", config);

    RmaConnManager rmaConnManager(impl);

    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, *linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .will(returnValue((SocketStatus)SocketStatus::INIT))
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    auto p2pConn1 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn1);
    auto p2pConn2 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn2);
    auto p2pConn3 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn3);

    linkData->type = PortDeploymentType::DEV_NET;
    linkData->linkProtocol_ = LinkProtocol::UB_CTP;
    auto ubConn = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, ubConn);

    ReqHandleResult result = ReqHandleResult::COMPLETED;
    MOCKER(&HrtRaGetAsyncReqResult)
        .stubs()
        .with()
        .will(invoke(HrtRaGetAsyncReqResult_Uncompleted))
        .then(returnValue(result));
    BatchDeleteJettys();

    DelRankTableFile();
    delete linkData;
    delete devPtr;
    delete socket;
}

int RaCtxQpDestroyBatchAsync_no_delete(void* ctx_handle, void* qp_handle[], unsigned int* num, void** req_handle)
{
    *num = 0;
    return 0;
}

TEST_F(RmaConnManagerTest, should_delete_part_when_calling_batchdeletejettys_with_valid_param)
{
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData* linkData = new LinkData(basePortType, 0, 1, 0, 1);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    u32 localRank = 0;
    u32 remoteRank = 1;

    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl.rankGraph = make_unique<RankGraph>(0);
    impl.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    impl.Init(commParams, "ranktable.json", config);

    RmaConnManager rmaConnManager(impl);

    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, *linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .will(returnValue((SocketStatus)SocketStatus::INIT))
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    auto p2pConn1 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn1);
    auto p2pConn2 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn2);
    auto p2pConn3 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn3);

    linkData->type = PortDeploymentType::DEV_NET;
    linkData->linkProtocol_ = LinkProtocol::UB_CTP;
    auto ubConn = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, ubConn);

    ReqHandleResult result = ReqHandleResult::COMPLETED;
    MOCKER(&HrtRaGetAsyncReqResult).stubs().with().will(returnValue(result));
    MOCKER(&RaCtxQpDestroyBatchAsync).stubs().will(invoke(RaCtxQpDestroyBatchAsync_no_delete));
    BatchDeleteJettys();

    DelRankTableFile();
    delete linkData;
    delete devPtr;
    delete socket;
}

int RaCtxQpDestroyBatchAsync_return_false(void* ctx_handle, void* qp_handle[], unsigned int* num, void** req_handle)
{
    return 1;
}

TEST_F(RmaConnManagerTest, should_failed_when_calling_batchdeletejettys_with_invalid_param)
{
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData* linkData = new LinkData(basePortType, 0, 1, 0, 1);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    u32 localRank = 0;
    u32 remoteRank = 1;

    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl.rankGraph = make_unique<RankGraph>(0);
    impl.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    impl.Init(commParams, "ranktable.json", config);

    RmaConnManager rmaConnManager(impl);

    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, *linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .will(returnValue((SocketStatus)SocketStatus::INIT))
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    auto p2pConn1 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn1);
    auto p2pConn2 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn2);
    auto p2pConn3 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn3);

    linkData->type = PortDeploymentType::DEV_NET;
    linkData->linkProtocol_ = LinkProtocol::UB_CTP;
    auto ubConn = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, ubConn);

    ReqHandleResult result = ReqHandleResult::COMPLETED;
    MOCKER(&HrtRaGetAsyncReqResult).stubs().with().will(returnValue(result));
    MOCKER(&RaCtxQpDestroyBatchAsync).stubs().will(invoke(RaCtxQpDestroyBatchAsync_no_delete));
    BatchDeleteJettys();

    DelRankTableFile();
    delete linkData;
    delete devPtr;
    delete socket;
}

int RaCtxQpDestroyBatchAsync_num_false(void* ctx_handle, void* qp_handle[], unsigned int* num, void** req_handle)
{
    *num = *num + 1;
    return 0;
}

TEST_F(RmaConnManagerTest, should_failed_when_calling_batchdeletejettys_with_invalid_param_2)
{
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData* linkData = new LinkData(basePortType, 0, 1, 0, 1);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    u32 localRank = 0;
    u32 remoteRank = 1;

    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl.rankGraph = make_unique<RankGraph>(0);
    impl.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    impl.Init(commParams, "ranktable.json", config);

    RmaConnManager rmaConnManager(impl);

    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, *linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .will(returnValue((SocketStatus)SocketStatus::INIT))
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    auto p2pConn1 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn1);
    auto p2pConn2 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn2);
    auto p2pConn3 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn3);

    linkData->type = PortDeploymentType::DEV_NET;
    linkData->linkProtocol_ = LinkProtocol::UB_CTP;
    auto ubConn = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, ubConn);

    ReqHandleResult result = ReqHandleResult::COMPLETED;
    MOCKER(&HrtRaGetAsyncReqResult).stubs().with().will(returnValue(result));
    MOCKER(&RaCtxQpDestroyBatchAsync).stubs().will(invoke(RaCtxQpDestroyBatchAsync_num_false));
    BatchDeleteJettys();

    DelRankTableFile();
    delete linkData;
    delete devPtr;
    delete socket;
}

ReqHandleResult HrtRaGetAsyncReqResult_TimeOut(RequestHandle& reqHandle)
{
    std::this_thread::sleep_for(std::chrono::seconds(15));
    return ReqHandleResult::COMPLETED;
}

TEST_F(RmaConnManagerTest, should_failed_when_calling_batchdeletejettys_with_invalid_param_3)
{
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData* linkData = new LinkData(basePortType, 0, 1, 0, 1);

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    u32 localRank = 0;
    u32 remoteRank = 1;

    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_910A;
    GenRankTableFile1Ser8Dev();

    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = remoteRank;
    u32 buffer = 10;
    collOpParams.sendBuf = static_cast<void*>(&buffer);
    collOpParams.recvBuf = static_cast<void*>(&buffer);
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(commParams.devType));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    impl.rankGraph = make_unique<RankGraph>(0);
    impl.rankGraph->peers_[0] = make_shared<NetInstance::Peer>(0, 0, 0, 0);
    MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void(CommunicatorImpl::*)(const string& rankTablePath))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CtxMgrImp::Init).stubs().will(ignoreReturnValue());
    impl.Init(commParams, "ranktable.json", config);

    RmaConnManager rmaConnManager(impl);

    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, *linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .will(returnValue((SocketStatus)SocketStatus::INIT))
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    auto p2pConn1 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn1);
    auto p2pConn2 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn2);
    auto p2pConn3 = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, p2pConn3);

    linkData->type = PortDeploymentType::DEV_NET;
    linkData->linkProtocol_ = LinkProtocol::UB_CTP;
    auto ubConn = rmaConnManager.Create(commParams.commId, *linkData);
    EXPECT_NE(nullptr, ubConn);

    ReqHandleResult result = ReqHandleResult::COMPLETED;
    MOCKER(&HrtRaGetAsyncReqResult).stubs().with().will(invoke(HrtRaGetAsyncReqResult_TimeOut));
    BatchDeleteJettys();

    DelRankTableFile();
    delete linkData;
    delete devPtr;
    delete socket;
}
