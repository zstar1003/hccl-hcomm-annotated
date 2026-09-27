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
#include "stub_communicator_impl_trans_mgr.h"
#include "virtual_topo.h"
#define private public
#define protected public
#include "mem_transport_manager.h"
#include "ub_mem_transport.h"
#include "p2p_transport.h"
#include "communicator_impl.h"
#include "notify_count.h"
#include "rdma_handle_manager.h"
#undef private
#undef protected

using namespace Hccl;

class MemTransportManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MemTransportManager tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "MemTransportManager tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in MemTransportManager SetUP" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "A Test case in MemTransportManager TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

TEST(MemTransportManagerTest, MemTransportManager_is_all_transport_ready)
{
    StubCommunicatorImplTransMgr comm;
    MemTransportManager transportManager(comm);

    std::string opTag = "test_tag";
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData linkData(portType, 0, 1, 0, 1);
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    unique_ptr<P2PTransport> transport = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    P2PTransport* transportPtr = transport.get();
    transportManager.opTagOffloadMap[opTag][linkData] = std::move(transport);

    TransportStatus fakeStatus = TransportStatus::READY;
    MOCKER_CPP_VIRTUAL(*transportPtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    EXPECT_EQ(transportManager.IsAllTransportReady(), true);
    GlobalMockObject::verify();

    fakeStatus = TransportStatus::INIT;
    MOCKER_CPP_VIRTUAL(*transportPtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    EXPECT_EQ(transportManager.IsAllTransportReady(), false);
    GlobalMockObject::verify();

    fakeStatus = TransportStatus::SOCKET_OK;
    MOCKER_CPP_VIRTUAL(*transportPtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    EXPECT_EQ(transportManager.IsAllTransportReady(), false);
    GlobalMockObject::verify();
}

TEST(MemTransportManagerTest, MemTransportManager_get_transport_success)
{
    StubCommunicatorImplTransMgr comm;
    MemTransportManager transportManager(comm);

    std::string opTag = "test_tag";
    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    void* rdmaHandle = (void*)0x100;
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    bool isRecvFirst = false;
    unique_ptr<UbMemTransport> transportOpbase
        = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    BaseMemTransport* transportOpbasePtr = transportOpbase.get();
    transportManager.opTagOpbasedMap[linkData] = std::move(transportOpbase);
    unique_ptr<UbMemTransport> transportOffload
        = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    BaseMemTransport* transportOffloadPtr = transportOffload.get();
    transportManager.opTagOffloadMap[opTag][linkData] = std::move(transportOffload);

    BaseMemTransport* transportRes = transportManager.GetOpbasedTransport(linkData);
    EXPECT_EQ(transportRes, transportOpbasePtr);

    transportRes = transportManager.GetOffloadTransport(opTag, linkData);
    EXPECT_EQ(transportRes, transportOffloadPtr);
    GlobalMockObject::verify();
}

TEST(MemTransportManagerTest, MemTransportManager_get_transport_nullptr)
{
    StubCommunicatorImplTransMgr comm;
    MemTransportManager transportManager(comm);

    std::string opTag = "test_tag";
    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    bool isRecvFirst = false;
    transportManager.opTagOpbasedMap[linkData]
        = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transportManager.opTagOffloadMap[opTag][linkData]
        = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    // opbase
    LinkData linkData1(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 2, 0, 2);
    BaseMemTransport* transport = transportManager.GetOpbasedTransport(linkData1);
    EXPECT_EQ(transport, nullptr);

    // offload
    // optag 不存在
    std::string opTag1 = "test_tag_1";
    transport = transportManager.GetOffloadTransport(opTag1, linkData);
    EXPECT_EQ(transport, nullptr);

    // optag 存在, linkdata 不存在
    transport = transportManager.GetOffloadTransport(opTag, linkData1);
    EXPECT_EQ(transport, nullptr);
    GlobalMockObject::verify();
}

TEST(MemTransportManagerTest, MemTransportManager_batch_recover_transports)
{
    s32 localId = 1;
    s32 groupLevel = 0;
    string groupId = "test";
    StubCommunicatorImplTransMgr comm;
    comm.rankGraph = std::make_unique<RankGraph>(0);
    std::shared_ptr<NetInstance> fabGroup = std::make_shared<InnerNetInstance>(0, "test");
    std::shared_ptr<NetInstance> fabGroup1 = std::make_shared<InnerNetInstance>(1, "test");
    auto peer0 = std::make_shared<NetInstance::Peer>(0, localId, localId, localId);
    auto peer1 = std::make_shared<NetInstance::Peer>(1, localId, localId, localId);
    peer0->AddNetInstance(fabGroup);
    peer1->AddNetInstance(fabGroup1);
    comm.rankGraph->AddPeer(peer0);
    comm.rankGraph->AddPeer(peer1);
    comm.rankGraph->AddNetInstance(fabGroup);
    comm.rankGraph->AddNetInstance(fabGroup1);
    comm.isWorldGroup = true;
    MemTransportManager transportManager(comm);

    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);

    // 打桩 SocketManager::GetConnectedSocket
    IpAddress ipAddress("1.0.0.0");
    shared_ptr<Socket> fakeSocket
        = make_shared<Socket>(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    SocketConfig socketConfig(linkData.GetRemoteRankId(), linkData, comm.GetEstablishLinkSocketTag());
    comm.GetSocketManager().connectedSocketMap[socketConfig] = std::move(fakeSocket);

    // 打桩 RmaConnManager::Get
    RdmaHandle rdmaHandle = (void*)0x1000000;
    MOCKER(GetUbToken).stubs().will(returnValue(1));
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OFFLOAD);
    MOCKER_CPP(&RmaConnManager::Get).stubs().will(returnValue(dynamic_cast<RmaConnection*>(&devUbConnection)));

    SocketStatus fakeSocketStatus = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue(fakeSocketStatus));

    std::string opTag = "test";
    std::vector<LinkData> links;
    links.push_back(linkData);
    transportManager.BatchRecoverOpbasedTransports(links);
    transportManager.BatchRecoverOffloadTransports(opTag, links);
    GlobalMockObject::verify();
}
