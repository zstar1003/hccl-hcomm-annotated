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
#include "test_mock_setup.h"
#include "communicator_impl.h"
#include "rdma_handle_manager.h"
#include "ccu_transport_group_manager.h"
#include "ccu_transport_manager.h"
#include "coll_service_device_mode.h"
#include "hccp_tlv_hdc_manager.h"
#include "ccu_driver_handle.h"
#include "ccu_component.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "tp_manager.h"
#undef private

using namespace Hccl;

class CcuTransportGroupMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuTransportGroupMgrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuTransportGroupMgrTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuTransportGroupMgrTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify(); // 避免用例之间的联系，防止上一个用例的打桩函数在本用例生效
        std::cout << "A Test case in CcuTransportGroupMgrTest TearDown" << std::endl;
    }

    IpAddress GetAnIpAddress()
    {
        IpAddress ipAddress("1.0.0.0");
        return ipAddress;
    }

    Hccl::SocketManager* socketManager;
    void* hccpSocketHandle;
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
};

const char filePath[] = "ranktable.json";
const char ranktable4pPath[] = "ranktable.json";
const char topoPath[] = "topo.json";
const std::string RankTable1Ser8Dev = R"(
    {
    "server_count":"1",
    "server_list":
    [
        {
            "device":[
                        {
                        "device_id":"0",
                        "rank_id":"0"
                        },
                        {
                        "device_id":"1",
                        "rank_id":"1"
                        },
                        {
                        "device_id":"2",
                        "rank_id":"2"
                        },
                        {
                        "device_id":"3",
                        "rank_id":"3"
                        },
                        {
                        "device_id":"4",
                        "rank_id":"4"
                        },
                        {
                        "device_id":"5",
                        "rank_id":"5"
                        },
                        {
                        "device_id":"6",
                        "rank_id":"6"
                        },
                        {
                        "device_id":"7",
                        "rank_id":"7"
                        }
                    ],
            "server_id":"1"
        }
    ],
    "status":"completed",
    "version":"1.0"
    }
    )";

static void GenRankTableFile1Ser8Dev()
{
    try {
        nlohmann::json rankTableJson = nlohmann::json::parse(RankTable1Ser8Dev);
        std::ofstream out(filePath, std::ofstream::out);
        out << rankTableJson;
    } catch (...) {
        std::cout << filePath << " generate failed!" << std::endl;
        return;
    }
    std::cout << filePath << " generated." << std::endl;
}

static void MockerCcuFeature()
{
    MOCKER_CPP(&CcuComponent::Init).stubs();
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs();
    MOCKER_CPP(&CtxMgrImp::Init).stubs();
    MOCKER_CPP(&TpManager::Init).stubs();
}

// 测试Get接口，预期成功获取rankGroup对应的ccuTransportGroup
TEST_F(CcuTransportGroupMgrTest, Test_CcuTransportGroupMgr_001)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    u32 utCntCke = 3;
    vector<CcuTransport*> utTransports;

    // 创建CommunicatorImpl
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

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
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

    // 打桩CcuConnection构造函数中调用的函数
    JfcHandle jfcHandle = 1;
    RdmaHandle rdmaHandle = new int(1);
    u32 jettyNum = 1; // 当前迭代，jettyNum默认为1
    u32 sqSize = 128; // 当前迭代，默认使用MS，故sqSize固定为128。sqSize就是jetty深度

    SETUP_CCU_TRANSPORT_MOCKS();

    // 创建utConnection
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);

    // 创建utCcuTransport
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::unique_ptr<CcuTransport> utCcuTransport
        = std::make_unique<CcuTransport>(socket, std::move(connection), locCclBufInfo);
    utTransports.push_back(std::move(utCcuTransport.get()));

    // 打桩CcuTransportGroup构造函数中调用的函数
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));

    // 创建utCcuTransportGroup
    std::unique_ptr<CcuTransportGroup> utCcuTransportGroup
        = std::make_unique<CcuTransportGroup>(utTransports, utCntCke);

    // 创建utCcuTransportGroupMgr，并在linkGrp2TransportGrpMap中建立utlinkGroup与utCcuTransportGroup的映射
    LinkGroup utLinkGroup{vector<LinkInfo>{LinkInfo{linkData}}};
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);
    utCcuTransportGroupMgr.linkGrp2TransportGrpMap[utLinkGroup] = std::move(utCcuTransportGroup);

    auto res2 = utCcuTransportGroupMgr.Get(utLinkGroup);
    EXPECT_NE(nullptr, res2);

    delete socket;
    delete rdmaHandle;
}

// 测试Get接口，预期返回nullptr
TEST_F(CcuTransportGroupMgrTest, Test_CcuTransportGroupMgr_002)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    u32 utCntCke = 3;
    vector<CcuTransport*> utTransports;

    // 创建CommunicatorImpl
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

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
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

    // 创建utCcuTransportGroupMgr
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);

    LinkGroup utLinkGroup{vector<LinkInfo>{LinkInfo{linkData}}};
    auto res2 = utCcuTransportGroupMgr.Get(utLinkGroup);
    EXPECT_EQ(nullptr, res2);

    delete socket;
}

// 测试PrepareCreate接口，预期成功获取rankGroup对应的ccuTransportGroup
TEST_F(CcuTransportGroupMgrTest, Test_CcuTransportGroupMgr_003)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    u32 utCntCke = 3;

    // 创建CommunicatorImpl
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

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);
    CollServiceDeviceMode collService{&impl};
    impl.collService = &collService;

    // 创建Socket
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

    // 打桩CcuConnection构造函数中调用的函数
    JfcHandle jfcHandle = 1;
    RdmaHandle rdmaHandle = new int(1);
    u32 jettyNum = 1; // 当前迭代，jettyNum默认为1
    u32 sqSize = 128; // 当前迭代，默认使用MS，故sqSize固定为128。sqSize就是jetty深度

    SETUP_CCU_TRANSPORT_MOCKS();

    // 创建utConnection
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);

    // 创建utCcuTransport
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::unique_ptr<CcuTransport> utCcuTransport
        = make_unique<CcuTransport>(socket, std::move(connection), locCclBufInfo);

    set<CcuTransport*> utCcuTransportSet;
    utCcuTransportSet.insert(std::move(utCcuTransport.get()));

    // 打桩ccuTransportMgr.Get()
    MOCKER_CPP(&CcuTransportMgr::Get, set<CcuTransport*>(CcuTransportMgr::*)(RankId))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(utCcuTransportSet));

    // 打桩CcuTransportGroup构造函数中调用的函数
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));

    // 创建utCcuTransportGroupMgr
    LinkGroup utLinkGroup{vector<LinkInfo>{LinkInfo{linkData}}};
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);
    auto res2 = utCcuTransportGroupMgr.PrepareCreate(utLinkGroup, utCntCke);
    EXPECT_NE(nullptr, res2);

    delete socket;
    delete rdmaHandle;
}

// 测试PrepareCreate接口，预期成功获取rankGroup对应的ccuTransportGroup
TEST_F(CcuTransportGroupMgrTest, Test_CcuTransportGroupMgr_004)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    u32 utCntCke = 3;
    vector<CcuTransport*> utTransports;

    // 创建CommunicatorImpl
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

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
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

    // 打桩CcuConnection构造函数中调用的函数
    JfcHandle jfcHandle = 1;
    RdmaHandle rdmaHandle = new int(1);
    u32 jettyNum = 1; // 当前迭代，jettyNum默认为1
    u32 sqSize = 128; // 当前迭代，默认使用MS，故sqSize固定为128。sqSize就是jetty深度

    SETUP_CCU_TRANSPORT_MOCKS();

    // 创建utConnection
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);

    // 创建utCcuTransport
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::unique_ptr<CcuTransport> utCcuTransport
        = make_unique<CcuTransport>(socket, std::move(connection), locCclBufInfo);
    utTransports.push_back(std::move(utCcuTransport.get()));

    // 打桩CcuTransportGroup构造函数中调用的函数
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));

    // 创建utCcuTransportGroup
    std::unique_ptr<CcuTransportGroup> utCcuTransportGroup = make_unique<CcuTransportGroup>(utTransports, utCntCke);

    // 创建utCcuTransportGroupMgr，并在linkGrp2TransportGrpMap中建立utLinkGroup与utCcuTransportGroup的映射
    LinkGroup utLinkGroup{vector<LinkInfo>{LinkInfo{linkData}}};
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);
    utCcuTransportGroupMgr.linkGrp2TransportGrpMap[utLinkGroup] = std::move(utCcuTransportGroup);

    auto res2 = utCcuTransportGroupMgr.PrepareCreate(utLinkGroup, utCntCke);
    EXPECT_NE(nullptr, res2);

    delete socket;
    delete rdmaHandle;
}

// 测试Confirm接口，预期tempTransportGroup的size为0
TEST_F(CcuTransportGroupMgrTest, Test_CcuTransportGroupMgr_005)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    vector<RankId> utRanks = {1, 2, 3};
    RankGroup utRankGroup(utRanks);

    // 创建CommunicatorImpl
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

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
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

    // 创建utCcuTransportGroupMgr
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);
    LinkGroup utLinkGroup{vector<LinkInfo>{LinkInfo{linkData}}};
    utCcuTransportGroupMgr.tempTransportGrp.emplace_back(utLinkGroup);
    EXPECT_NE(0, utCcuTransportGroupMgr.tempTransportGrp.size());

    utCcuTransportGroupMgr.Confirm();
    EXPECT_EQ(0, utCcuTransportGroupMgr.tempTransportGrp.size());

    delete socket;
}

// 测试Fallback接口，预期tempTransportGroup的size为0
TEST_F(CcuTransportGroupMgrTest, Test_CcuTransportGroupMgr_006)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1); // linkData创建完成

    u32 utCntCke = 3;

    // 创建CommunicatorImpl
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

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
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

    // 打桩CcuConnection构造函数中调用的函数
    JfcHandle jfcHandle = 1;
    RdmaHandle rdmaHandle = new int(1);
    u32 jettyNum = 1; // 当前迭代，jettyNum默认为1
    u32 sqSize = 128; // 当前迭代，默认使用MS，故sqSize固定为128。sqSize就是jetty深度

    SETUP_CCU_TRANSPORT_MOCKS();

    // 创建utConnection
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);

    // 创建utCcuTransport
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::unique_ptr<CcuTransport> utCcuTransport
        = make_unique<CcuTransport>(socket, std::move(connection), locCclBufInfo);

    vector<CcuTransport*> utCcuTransportVec;
    utCcuTransportVec.emplace_back(std::move(utCcuTransport.get()));

    // 打桩CcuTransportGroup构造函数中调用的函数
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));

    // 创建utCcuTransportGroup
    std::unique_ptr<CcuTransportGroup> utCcuTransportGroup
        = make_unique<CcuTransportGroup>(utCcuTransportVec, utCntCke);

    // 创建utCcuTransportGroupMgr，并在linkGrp2TransportGrpMap中建立utLinkGroup与utCcuTransportGroup的映射
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);
    LinkGroup utLinkGroup{vector<LinkInfo>{LinkInfo{linkData}}};
    utCcuTransportGroupMgr.tempTransportGrp.emplace_back(utLinkGroup);
    utCcuTransportGroupMgr.linkGrp2TransportGrpMap[utLinkGroup] = std::move(utCcuTransportGroup);

    EXPECT_NE(0, utCcuTransportGroupMgr.tempTransportGrp.size());
    EXPECT_NE(0, utCcuTransportGroupMgr.linkGrp2TransportGrpMap.size());

    utCcuTransportGroupMgr.Fallback();
    EXPECT_EQ(0, utCcuTransportGroupMgr.tempTransportGrp.size());
    EXPECT_EQ(0, utCcuTransportGroupMgr.linkGrp2TransportGrpMap.size());

    delete socket;
    delete rdmaHandle;
}

// 测试Destroy接口，预期isDestroyed为true
TEST_F(CcuTransportGroupMgrTest, Test_CcuTransportGroupMgr_007)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // 创建CommunicatorImpl
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

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
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

    // 创建utCcuTransportGroupMgr
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);
    EXPECT_EQ(false, utCcuTransportGroupMgr.isDestroyed);

    utCcuTransportGroupMgr.Destroy();
    EXPECT_EQ(true, utCcuTransportGroupMgr.isDestroyed);

    delete socket;
}

TEST_F(CcuTransportGroupMgrTest, GetAllTransportGroups)
{
    GlobalMockObject::verify();

    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    u32 utCntCke = 3;
    vector<CcuTransport*> utTransports;

    // 创建CommunicatorImpl
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

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
    MOCKER(HrtMemset).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
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
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
    void* hccpSocketHandle;
    hccpSocketHandle = new int(0);
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

    // 打桩CcuConnection构造函数中调用的函数
    JfcHandle jfcHandle = 1;
    RdmaHandle rdmaHandle = new int(1);
    u32 jettyNum = 1; // 当前迭代，jettyNum默认为1
    u32 sqSize = 128; // 当前迭代，默认使用MS，故sqSize固定为128。sqSize就是jetty深度

    SETUP_CCU_TRANSPORT_MOCKS();

    // 创建utConnection
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);

    // 创建utCcuTransport
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::unique_ptr<CcuTransport> utCcuTransport
        = std::make_unique<CcuTransport>(socket, std::move(connection), locCclBufInfo);
    utTransports.push_back(std::move(utCcuTransport.get()));

    // 打桩CcuTransportGroup构造函数中调用的函数
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));

    // 创建utCcuTransportGroup
    std::unique_ptr<CcuTransportGroup> utCcuTransportGroup
        = std::make_unique<CcuTransportGroup>(utTransports, utCntCke);

    // 创建utCcuTransportGroupMgr，并在linkGrp2TransportGrpMap中建立utLinkGroup与utCcuTransportGroup的映射
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);
    LinkGroup utLinkGroup{vector<LinkInfo>{LinkInfo{linkData}}};
    EXPECT_THROW(utCcuTransportGroupMgr.GetAllTransportGroups(), InternalException);
    utCcuTransportGroupMgr.linkGrp2TransportGrpMap[utLinkGroup] = std::move(utCcuTransportGroup);
    EXPECT_NO_THROW(utCcuTransportGroupMgr.GetAllTransportGroups());

    delete socket;
    delete rdmaHandle;
    delete hccpSocketHandle;
}

// 测试Clean接口
TEST_F(CcuTransportGroupMgrTest, Test_CcuTransportGroupMgr_008)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // 创建CommunicatorImpl
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

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
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

    // 创建utCcuTransportGroupMgr
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);

    utCcuTransportGroupMgr.Clean();

    delete socket;
}

// 测试ResumeTransportGroup接口，预期成功获取rankGroup对应的ccuTransportGroup
TEST_F(CcuTransportGroupMgrTest, should_return_ccuTransportGroup_when_calling_ResumeTransportGroup)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    u32 utCntCke = 3;

    // 创建CommunicatorImpl
    u32 localRank = 0;
    u32 remoteRank = 1;
    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_950;
    GenRankTableFile1Ser8Dev();

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);
    CollServiceDeviceMode collService{&impl};
    impl.collService = &collService;

    // 创建Socket
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

    // 打桩CcuConnection构造函数中调用的函数
    JfcHandle jfcHandle = 1;
    RdmaHandle rdmaHandle = new int(1);
    u32 jettyNum = 1; // 当前迭代，jettyNum默认为1
    u32 sqSize = 128; // 当前迭代，默认使用MS，故sqSize固定为128。sqSize就是jetty深度

    MOCKER(CcuDeviceManager::AllocXn).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::AllocCke).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::ConfigChannel).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER(CcuDeviceManager::ReleaseXn).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::ReleaseCke).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER(HrtRaUbUnimportJetty).defaults().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByUserDevId).defaults().will(returnValue(static_cast<s32>(0)));
    MOCKER(HrtRaUbCreateJetty).defaults().will(returnValue(HrtRaUbJettyCreatedOutParam()));
    MOCKER(HraGetDieAndFuncId).defaults().will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));
    MOCKER(HrtRaUbCreateJfc).defaults().will(returnValue(jfcHandle));
    MOCKER(RaUbImportJetty).defaults().will(returnValue(HrtRaUbJettyImportedOutParam()));
    MOCKER(HrtRaUbLocalMemReg).defaults().will(returnValue(HrtRaUbLocalMemRegOutParam()));

    // 创建utConnection
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);

    // 创建utCcuTransport
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::unique_ptr<CcuTransport> utCcuTransport
        = make_unique<CcuTransport>(socket, std::move(connection), locCclBufInfo);

    set<CcuTransport*> utCcuTransportSet;
    utCcuTransportSet.insert(std::move(utCcuTransport.get()));

    // 打桩ccuTransportMgr.Get()
    MOCKER_CPP(&CcuTransportMgr::Get, set<CcuTransport*>(CcuTransportMgr::*)(RankId))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(utCcuTransportSet));

    // 打桩CcuTransportGroup构造函数中调用的函数
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));

    // 创建utCcuTransportGroupMgr
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);
    LinkGroup utLinkGroup{vector<LinkInfo>{LinkInfo{linkData}}};
    utCcuTransportGroupMgr.linkGrp2TransportGrpMap[utLinkGroup] = nullptr;
    EXPECT_NO_THROW(utCcuTransportGroupMgr.ResumeAll(utCntCke));

    delete socket;
    delete rdmaHandle;
}

// 测试ResumeTransportGroup接口，预期创建失败，抛出异常
TEST_F(CcuTransportGroupMgrTest, should_throw_if_transportGroup_init_fail_when_calling_ResumeTransportGroup)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    u32 utCntCke = 3;

    // 创建CommunicatorImpl
    u32 localRank = 0;
    u32 remoteRank = 1;
    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_950;
    GenRankTableFile1Ser8Dev();

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);
    CollServiceDeviceMode collService{&impl};
    impl.collService = &collService;

    // 创建Socket
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

    // 打桩CcuConnection构造函数中调用的函数
    JfcHandle jfcHandle = 1;
    RdmaHandle rdmaHandle = new int(1);
    u32 jettyNum = 1; // 当前迭代，jettyNum默认为1
    u32 sqSize = 128; // 当前迭代，默认使用MS，故sqSize固定为128。sqSize就是jetty深度

    MOCKER(CcuDeviceManager::AllocXn).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::AllocCke).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::ConfigChannel).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER(CcuDeviceManager::ReleaseXn).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::ReleaseCke).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER(HrtRaUbUnimportJetty).defaults().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByUserDevId).defaults().will(returnValue(static_cast<s32>(0)));
    MOCKER(HrtRaUbCreateJetty).defaults().will(returnValue(HrtRaUbJettyCreatedOutParam()));
    MOCKER(HraGetDieAndFuncId).defaults().will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));
    MOCKER(HrtRaUbCreateJfc).defaults().will(returnValue(jfcHandle));
    MOCKER(RaUbImportJetty).defaults().will(returnValue(HrtRaUbJettyImportedOutParam()));
    MOCKER(HrtRaUbLocalMemReg).defaults().will(returnValue(HrtRaUbLocalMemRegOutParam()));

    // 创建utConnection
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);

    // 创建utCcuTransport
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::unique_ptr<CcuTransport> utCcuTransport
        = make_unique<CcuTransport>(socket, std::move(connection), locCclBufInfo);

    set<CcuTransport*> utCcuTransportSet;
    utCcuTransportSet.insert(std::move(utCcuTransport.get()));

    // 打桩ccuTransportMgr.Get()
    MOCKER_CPP(&CcuTransportMgr::Get, set<CcuTransport*>(CcuTransportMgr::*)(RankId))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(utCcuTransportSet));

    // 打桩CcuTransportGroup构造函数中调用的函数
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_E_INTERNAL));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));

    // 创建utCcuTransportGroupMgr
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);
    LinkGroup utLinkGroup{vector<LinkInfo>{LinkInfo{linkData}}};
    utCcuTransportGroupMgr.linkGrp2TransportGrpMap[utLinkGroup] = nullptr;
    EXPECT_THROW(utCcuTransportGroupMgr.ResumeAll(utCntCke), InternalException);

    delete socket;
    delete rdmaHandle;
}

// 测试ResumeTransportGroup接口，预期创建失败
TEST_F(CcuTransportGroupMgrTest, should_no_throw_if_rankgroup_empty_when_calling_ResumeTransportGroup)
{
    // 创建linkData
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    RankId rankId = 0;
    u32 utCntCke = 3;

    // 创建CommunicatorImpl
    u32 localRank = 0;
    u32 remoteRank = 1;
    CommunicatorImpl impl;
    CommParams commParams;
    commParams.commId = "commId";
    commParams.myRank = localRank;
    commParams.rankSize = 8;
    HcclCommConfig config;
    commParams.devType = DevType::DEV_TYPE_950;
    GenRankTableFile1Ser8Dev();

    void* devPtr = nullptr;
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(devPtr));
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
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&HccpTlvHdcManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);
    CollServiceDeviceMode collService{&impl};
    impl.collService = &collService;

    // 创建Socket
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

    // 打桩CcuConnection构造函数中调用的函数
    JfcHandle jfcHandle = 1;
    RdmaHandle rdmaHandle = new int(1);
    u32 jettyNum = 1; // 当前迭代，jettyNum默认为1
    u32 sqSize = 128; // 当前迭代，默认使用MS，故sqSize固定为128。sqSize就是jetty深度

    MOCKER(CcuDeviceManager::AllocXn).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::AllocCke).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::ConfigChannel).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER(CcuDeviceManager::ReleaseXn).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::ReleaseCke).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER(HrtRaUbUnimportJetty).defaults().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByUserDevId).defaults().will(returnValue(static_cast<s32>(0)));
    MOCKER(HrtRaUbCreateJetty).defaults().will(returnValue(HrtRaUbJettyCreatedOutParam()));
    MOCKER(HraGetDieAndFuncId).defaults().will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));
    MOCKER(HrtRaUbCreateJfc).defaults().will(returnValue(jfcHandle));
    MOCKER(RaUbImportJetty).defaults().will(returnValue(HrtRaUbJettyImportedOutParam()));
    MOCKER(HrtRaUbLocalMemReg).defaults().will(returnValue(HrtRaUbLocalMemRegOutParam()));

    // 创建utConnection
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);

    // 创建utCcuTransport
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::unique_ptr<CcuTransport> utCcuTransport
        = make_unique<CcuTransport>(socket, std::move(connection), locCclBufInfo);

    set<CcuTransport*> utCcuTransportSet;
    utCcuTransportSet.insert(std::move(utCcuTransport.get()));

    // 打桩ccuTransportMgr.Get()
    MOCKER_CPP(&CcuTransportMgr::Get, set<CcuTransport*>(CcuTransportMgr::*)(RankId))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(utCcuTransportSet));

    // 打桩CcuTransportGroup构造函数中调用的函数
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_E_INTERNAL));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));

    // 创建utCcuTransportGroupMgr
    CcuTransportGroupMgr utCcuTransportGroupMgr(impl);
    utCcuTransportGroupMgr.linkGrp2TransportGrpMap.clear();
    EXPECT_NO_THROW(utCcuTransportGroupMgr.ResumeAll(utCntCke));

    delete socket;
    delete rdmaHandle;
}
