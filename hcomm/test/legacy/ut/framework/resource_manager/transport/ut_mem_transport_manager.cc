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
#include <string>
#define private public
#define protected public
#include "rdma_handle_manager.h"
#include "stub_communicator_impl_trans_mgr.h"
#include "virtual_topo.h"
#include "op_type.h"
#include "mem_transport_manager.h"
#include "ub_mem_transport.h"
#include "p2p_transport.h"
#include "notify_count.h"
#include "base_mem_transport.h"
#include "recover_info.h"
#include "rank_graph_builder.h"
#include "rank_gph.h"
#include "phy_topo_builder.h"
#include "phy_topo.h"
#include "detour_service.h"
#undef protected
#undef private

using namespace Hccl;

namespace {
const std::string RANK_TABLE_4P_FOR_TRANSPORT = R"(
{
    "version": "2.0",
    "rank_count": 4,
    "rank_list": [
        {
            "rank_id": 0,
            "device_id": 0,
            "local_id": 0,
            "level_list": [{
                "net_layer": 0,
                "net_instance_id": "az0-rack0",
                "net_type": "TOPO_FILE_DESC",
                "net_attr": "",
                "rank_addr_list": [
                    {"addr_type": "IPV4", "addr": "192.168.30.1", "ports": ["0/0"]},
                    {"addr_type": "IPV4", "addr": "192.168.20.1", "ports": ["1/0"]}
                ]
            }]
        },
        {
            "rank_id": 1,
            "device_id": 1,
            "local_id": 1,
            "level_list": [{
                "net_layer": 0,
                "net_instance_id": "az0-rack0",
                "net_type": "TOPO_FILE_DESC",
                "net_attr": "",
                "rank_addr_list": [
                    {"addr_type": "IPV4", "addr": "192.168.30.2", "ports": ["0/0"]},
                    {"addr_type": "IPV4", "addr": "192.168.20.2", "ports": ["1/0"]}
                ]
            }]
        },
        {
            "rank_id": 2,
            "device_id": 2,
            "local_id": 2,
            "level_list": [{
                "net_layer": 0,
                "net_instance_id": "az0-rack0",
                "net_type": "TOPO_FILE_DESC",
                "net_attr": "",
                "rank_addr_list": [
                    {"addr_type": "IPV4", "addr": "192.168.30.3", "ports": ["0/0"]},
                    {"addr_type": "IPV4", "addr": "192.168.20.3", "ports": ["1/0"]}
                ]
            }]
        },
        {
            "rank_id": 3,
            "device_id": 3,
            "local_id": 3,
            "level_list": [{
                "net_layer": 0,
                "net_instance_id": "az0-rack0",
                "net_type": "TOPO_FILE_DESC",
                "net_attr": "",
                "rank_addr_list": [
                    {"addr_type": "IPV4", "addr": "192.168.30.4", "ports": ["0/0"]},
                    {"addr_type": "IPV4", "addr": "192.168.20.4", "ports": ["1/0"]}
                ]
            }]
        }
    ]
}
)";
} // namespace

class MemTransportManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MemTransportManager tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "MemTransportManager tests tear down." << std::endl; }

    virtual void SetUp()
    {
        PhyTopo::GetInstance()->Clear();
        std::cout << "A Test case in MemTransportManager SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        PhyTopo::GetInstance()->Clear();
        std::cout << "A Test case in MemTransportManager TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

TEST_F(MemTransportManagerTest, MemTransportManager_batch_build_opbased_transports)
{
    StubCommunicatorImplTransMgr comm;
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
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    MOCKER_CPP(&RmaConnManager::Get).stubs().will(returnValue(dynamic_cast<RmaConnection*>(&devUbConnection)));

    SocketStatus fakeSocketStatus = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue(fakeSocketStatus));

    std::vector<LinkData> links;
    links.push_back(linkData);
    comm.GetCurrentCollOperator()->opType = OpType::ALLREDUCE;
    transportManager.BatchBuildOpbasedTransports(links);
    transportManager.Clear();
}

TEST_F(MemTransportManagerTest, MemTransportManager_batch_build_offload_transports)
{
    StubCommunicatorImplTransMgr comm;
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
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OFFLOAD);
    MOCKER_CPP(&RmaConnManager::Get).stubs().will(returnValue(dynamic_cast<RmaConnection*>(&devUbConnection)));

    SocketStatus fakeSocketStatus = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue(fakeSocketStatus));
    std::pair<TokenIdHandle, uint32_t> fakeTokenInfo = std::make_pair(0x12345678, 1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));

    std::string opTag = "test_tag";
    std::vector<LinkData> links;
    links.push_back(linkData);
    comm.GetCurrentCollOperator()->opType = OpType::ALLREDUCE;
    transportManager.BatchBuildOffloadTransports(opTag, links);
    transportManager.Clear();
}

TEST_F(MemTransportManagerTest, MemTransportManager_is_all_transport_ready)
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
}

TEST_F(MemTransportManagerTest, MemTransportManager_get_transport_success)
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
    unique_ptr<UbMemTransport> transportOpbase
        = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    BaseMemTransport* transportOpbasePtr = transportOpbase.get();
    transportManager.opTagOpbasedMap[linkData] = std::move(transportOpbase);
    unique_ptr<UbMemTransport> transportOffload
        = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    BaseMemTransport* transportOffloadPtr = transportOffload.get();
    transportManager.opTagOffloadMap[opTag][linkData] = std::move(transportOffload);
    unique_ptr<UbMemTransport> transportOneSided
        = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    BaseMemTransport* transportOneSidedPtr = transportOneSided.get();
    transportManager.oneSidedMap[linkData] = std::move(transportOneSided);

    BaseMemTransport* transportRes = transportManager.GetOpbasedTransport(linkData);
    EXPECT_EQ(transportRes, transportOpbasePtr);

    transportRes = transportManager.GetOffloadTransport(opTag, linkData);
    EXPECT_EQ(transportRes, transportOffloadPtr);

    transportRes = transportManager.GetOneSidedTransport(linkData);
    EXPECT_EQ(transportRes, transportOneSidedPtr);
}

TEST_F(MemTransportManagerTest, MemTransportManager_get_transport_nullptr)
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
}

TEST_F(MemTransportManagerTest, MemTransportManager_is_all_transport_ready_2)
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
    unique_ptr<P2PTransport> transportOpbase = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    P2PTransport* transportOpbasePtr = transportOpbase.get();
    transportManager.opTagOpbasedMap[linkData] = std::move(transportOpbase);
    unique_ptr<P2PTransport> transportOffload = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    P2PTransport* transportOffloadPtr = transportOffload.get();
    transportManager.opTagOffloadMap[opTag][linkData] = std::move(transportOffload);

    TransportStatus fakeStatus = TransportStatus::READY;
    transportManager.IsAllOpbasedTransportReady();
    transportManager.IsAllOffloadTransportReady(opTag);
    MOCKER_CPP_VIRTUAL(*transportOpbasePtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    MOCKER_CPP_VIRTUAL(*transportOffloadPtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    GlobalMockObject::verify();

    fakeStatus = TransportStatus::INIT;
    MOCKER_CPP_VIRTUAL(*transportOpbasePtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    MOCKER_CPP_VIRTUAL(*transportOffloadPtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    transportManager.IsAllOpbasedTransportReady();
    transportManager.IsAllOffloadTransportReady(opTag);
    GlobalMockObject::verify();

    fakeStatus = TransportStatus::SOCKET_OK;
    MOCKER_CPP_VIRTUAL(*transportOpbasePtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    MOCKER_CPP_VIRTUAL(*transportOffloadPtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    transportManager.IsAllOpbasedTransportReady();
    transportManager.IsAllOffloadTransportReady(opTag);
}

TEST_F(MemTransportManagerTest, MemTransportManager_is_all_transport_ready_3)
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
    unique_ptr<P2PTransport> transportOpbase = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    P2PTransport* transportOpbasePtr = transportOpbase.get();
    transportManager.opTagOpbasedMap[linkData] = std::move(transportOpbase);
    transportManager.newOpbasedTransports[linkData] = 0;
    unique_ptr<P2PTransport> transportOffload = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    P2PTransport* transportOffloadPtr = transportOffload.get();
    transportManager.opTagOffloadMap[opTag][linkData] = std::move(transportOffload);
    transportManager.newOffloadTransports[opTag][linkData] = 0;

    TransportStatus fakeStatus = TransportStatus::READY;
    MOCKER_CPP_VIRTUAL(*transportOpbasePtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    MOCKER_CPP_VIRTUAL(*transportOffloadPtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    GlobalMockObject::verify();

    fakeStatus = TransportStatus::INIT;
    MOCKER_CPP_VIRTUAL(*transportOpbasePtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    MOCKER_CPP_VIRTUAL(*transportOffloadPtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    EXPECT_EQ(transportManager.IsAllOpbasedTransportReady(), false);
    EXPECT_EQ(transportManager.IsAllOffloadTransportReady(opTag), false);
}

TEST_F(MemTransportManagerTest, MemTransportManager_batch_recover_transports)
{
    s32 localId = 1;
    s32 groupLevel = 0;
    string groupId = "test";
    StubCommunicatorImplTransMgr comm;
    comm.rankGraph = std::make_unique<RankGraph>(0);
    std::shared_ptr<NetInstance> fabGroup = std::make_shared<InnerNetInstance>(0, "test");
    std::shared_ptr<NetInstance> fabGroup1 = std::make_shared<InnerNetInstance>(1, "test");
    auto peer0 = std::make_shared<NetInstance::Peer>(0, localId, localId, 0);
    auto peer1 = std::make_shared<NetInstance::Peer>(1, localId, localId, 0);
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
}

TEST_F(MemTransportManagerTest, MemTransportManager_is_all_transport_recovered_ready_1)
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
    unique_ptr<P2PTransport> transportOpbase = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    P2PTransport* transportOpbasePtr = transportOpbase.get();
    transportManager.opTagOpbasedMap[linkData] = std::move(transportOpbase);
    transportManager.newOpbasedTransports[linkData] = 0;
    RecoverInfoData recoverInfoData{};
    RecoverInfo recoverInfo(recoverInfoData, 0);
    transportOpbasePtr->rmtHandshakeMsg = recoverInfo.GetUniqueId();
    transportOpbasePtr->attr.handshakeMsg = recoverInfo.GetUniqueId();

    TransportStatus fakeStatus = TransportStatus::READY;
    MOCKER_CPP_VIRTUAL(*transportOpbasePtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    EXPECT_EQ(transportManager.IsAllOpbasedTransportRecoveredReady(), true);
    EXPECT_EQ(transportManager.newOpbasedTransports.size(), 0);
}

TEST_F(MemTransportManagerTest, MemTransportManager_is_all_transport_recovered_ready_2)
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
    unique_ptr<P2PTransport> transportOpbase = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    P2PTransport* transportOpbasePtr = transportOpbase.get();
    transportManager.opTagOpbasedMap[linkData] = std::move(transportOpbase);
    transportManager.newOpbasedTransports[linkData] = 0;

    TransportStatus fakeStatus = TransportStatus::INIT;
    MOCKER_CPP_VIRTUAL(*transportOpbasePtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    EXPECT_EQ(transportManager.IsAllOpbasedTransportRecoveredReady(), false);

    unique_ptr<P2PTransport> transportOffload = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    P2PTransport* transportOffloadPtr = transportOffload.get();
    transportManager.opTagOffloadMap[opTag][linkData] = std::move(transportOffload);
    transportManager.newOffloadTransports[opTag][linkData] = 0;

    MOCKER_CPP_VIRTUAL(*transportOffloadPtr, &P2PTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    EXPECT_EQ(transportManager.IsAllOffloadTransportRecoveredReady(opTag), false);
}

TEST_F(MemTransportManagerTest, MemTransportManager_getpackeddata)
{
    // mock
    MOCKER_CPP(&MemTransportManager::IsAllOffloadTransportReady, bool(MemTransportManager::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(true));

    // when
    StubCommunicatorImplTransMgr comm;
    MemTransportManager transportManager(comm);
    std::string opTag = "test_tag";
    std::vector<char> res;
    BinaryStream binaryStream;
    binaryStream << 0;
    binaryStream.Dump(res);

    // check
    EXPECT_EQ(transportManager.GetOffloadPackedData(opTag), res);

    // when
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData linkData(portType, 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    unique_ptr<P2PTransport> transport = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    P2PTransport* transportPtr = transport.get();
    transportManager.opTagOffloadMap[opTag][linkData] = std::move(transport);
    transportManager.newOffloadTransports[opTag][linkData] = 0;
    RdmaHandle rdmaHandle;
    BaseMemTransport::LocCntNotifyRes locCntNotifyRes;
    std::unique_ptr<BaseMemTransport> trans = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    MOCKER_CPP_VIRTUAL(*trans, &BaseMemTransport::GetUniqueId).stubs().will(returnValue(std::vector<char>()));

    // check
    EXPECT_NE(transportManager.GetOffloadPackedData(opTag), res);
}

TEST_F(MemTransportManagerTest, MemTransportManager_update_offload_transports)
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
    transportManager.opTagOffloadMap[opTag][linkData]
        = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    // 打桩 RmaConnManager::Get
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OFFLOAD);
    MOCKER_CPP(&RmaConnManager::Get).stubs().will(returnValue(dynamic_cast<RmaConnection*>(&devUbConnection)));

    transportManager.UpdateOffloadTransports();
}

TEST_F(MemTransportManagerTest, MemTransportManager_is_all_one_sided_transport_ready_1)
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
    unique_ptr<UbMemTransport> transportOneSided
        = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    UbMemTransport* transportOneSidedPtr = transportOneSided.get();
    transportManager.oneSidedMap[linkData] = std::move(transportOneSided);
    transportManager.newOneSidedTransports[linkData] = 0;

    TransportStatus fakeStatus = TransportStatus::INIT;
    MOCKER_CPP_VIRTUAL(*transportOneSidedPtr, &UbMemTransport::GetStatus).stubs().will(returnValue(fakeStatus));
    transportManager.IsAllOneSidedTransportReady();
}

TEST_F(MemTransportManagerTest, MemTransportManager_batch_build_oneSide_transports)
{
    StubCommunicatorImplTransMgr comm;
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
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    MOCKER_CPP(&RmaConnManager::Get).stubs().will(returnValue(dynamic_cast<RmaConnection*>(&devUbConnection)));
    SocketStatus fakeSocketStatus = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue(fakeSocketStatus));

    std::vector<LinkData> links;
    links.push_back(linkData);
    comm.GetCurrentCollOperator()->opType = OpType::ALLREDUCE;
    EXPECT_NO_THROW(transportManager.BatchBuildOneSidedTransports(links));
    transportManager.Clear();
}

TEST_F(MemTransportManagerTest, MemTransportManager_UT_GetUrmaWqsAndCqs)
{
    StubCommunicatorImplTransMgr comm;
    comm.rankSize = 4;
    comm.myRank = 0;
    RankGraphBuilder rankGraphBuilder;
    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/resource_manager/transport/topo_4p.json"};
    unique_ptr<RankGraph> rankGraph = rankGraphBuilder.Build(RANK_TABLE_4P_FOR_TRANSPORT, topoFilePath, 0);
    comm.rankGraph = std::move(rankGraph);
    MemTransportManager transportManager(comm);

    MOCKER_CPP(&MemTransportManager::IsAllTransportReady).stubs().will(returnValue(true));
    EXPECT_NO_THROW(transportManager.GetUrmaWqs());
    EXPECT_NO_THROW(transportManager.GetUrmaCqs());
}
