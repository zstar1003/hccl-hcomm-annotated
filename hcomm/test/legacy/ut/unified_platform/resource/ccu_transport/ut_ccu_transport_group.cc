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
#include "test_mock_setup.h"
#define private public
#include "communicator_impl.h"
#include "rdma_handle_manager.h"
#include "rank_table.h"
#include "ccu_transport_group.h"
#include "ccu_component.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "tp_manager.h"

#undef private

using namespace Hccl;

class CcuTransportGroupTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuTransportGroupTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuTransportGroupTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuTransportGroupTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify(); // 避免用例之间的联系，防止上一个用例的打桩函数在本用例生效
        std::cout << "A Test case in CcuTransportGroupTest TearDown" << std::endl;
    }

    IpAddress GetAnIpAddress()
    {
        IpAddress ipAddress("1.0.0.0");
        return ipAddress;
    }

    void* hccpSocketHandle;
    IpAddress localIp;
    IpAddress remoteIp;
};

static void MockerCcuFeature()
{
    MOCKER_CPP(&CcuComponent::Init).stubs();
    MOCKER_CPP(&CcuResBatchAllocator::Init).stubs();
    MOCKER_CPP(&CtxMgrImp::Init).stubs();
    MOCKER_CPP(&TpManager::Init).stubs();
}

// 测试GetGrpStatus接口，预期返回值TransportGrpStatus::INIT

TEST_F(CcuTransportGroupTest, Test_CcuTransportGroup_001)
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
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    SETUP_SOCKET_STATUS_MOCKS();

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
    CcuTransportGroup utCcuTransportGroup(utCcuTransportVec, utCntCke);

    auto res = utCcuTransportGroup.GetGrpStatus();
    EXPECT_EQ(TransportGrpStatus::INIT, res);

    delete socket;
    delete rdmaHandle;
}

// 测试Destroy接口，预期isDestroyed为true
TEST_F(CcuTransportGroupTest, Test_CcuTransportGroup_002)
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
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    SETUP_SOCKET_STATUS_MOCKS();

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
    CcuTransportGroup utCcuTransportGroup(utCcuTransportVec, utCntCke);
    EXPECT_EQ(false, utCcuTransportGroup.isDestroyed);

    utCcuTransportGroup.Destroy();
    EXPECT_EQ(true, utCcuTransportGroup.isDestroyed);

    delete socket;
    delete rdmaHandle;
}

// 测试GetCntCkeId接口，预期返回值不等于0
TEST_F(CcuTransportGroupTest, Test_CcuTransportGroup_003)
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
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    SETUP_SOCKET_STATUS_MOCKS();

    // 打桩CcuConnection构造函数中调用的函数
    JfcHandle jfcHandle = 1;
    RdmaHandle rdmaHandle = new int(1);
    u32 jettyNum = 1; // 当前迭代，jettyNum默认为1
    u32 sqSize = 128; // 当前迭代，默认使用MS，故sqSize固定为128。sqSize就是jetty深度
    u32 cntCkeId = 0;

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
    CcuTransportGroup utCcuTransportGroup(utCcuTransportVec, utCntCke);
    utCcuTransportGroup.cntCkesGroup.emplace_back(0);
    utCcuTransportGroup.cntCkesGroup.emplace_back(1);
    utCcuTransportGroup.cntCkesGroup.emplace_back(2);

    HcclResult res1 = utCcuTransportGroup.GetCntCkeId(0, cntCkeId);
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, res1);
    EXPECT_EQ(0, cntCkeId);

    HcclResult res2 = utCcuTransportGroup.GetCntCkeId(1, cntCkeId);
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, res2);
    EXPECT_EQ(1, cntCkeId);

    HcclResult res3 = utCcuTransportGroup.GetCntCkeId(2, cntCkeId);
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, res3);
    EXPECT_EQ(2, cntCkeId);

    delete socket;
    delete rdmaHandle;
}

// 测试CheckTransports接口，预期返回值等于true
TEST_F(CcuTransportGroupTest, Test_CcuTransportGroup_004)
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
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    SETUP_SOCKET_STATUS_MOCKS();

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
    vector<CcuTransport*> utCcuTransportVec;
    utCcuTransportVec.emplace_back(std::move(utCcuTransport.get()));

    // 创建utCcuTransportGroup
    CcuTransportGroup utCcuTransportGroup(utCcuTransportVec, utCntCke);
    auto res = utCcuTransportGroup.CheckTransports(utCcuTransportVec);

    EXPECT_EQ(true, res);

    delete socket;
    delete rdmaHandle;
}

// 测试CheckTransports接口，预期返回值等于false
TEST_F(CcuTransportGroupTest, Test_CcuTransportGroup_005)
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
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    SETUP_SOCKET_STATUS_MOCKS();

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
    vector<CcuTransport*> utCcuTransportVec;

    // 创建utCcuTransportGroup
    CcuTransportGroup utCcuTransportGroup(utCcuTransportVec, utCntCke);
    auto res = utCcuTransportGroup.CheckTransports(utCcuTransportVec);

    EXPECT_EQ(false, res);

    delete socket;
    delete rdmaHandle;
}

// 测试CheckTransportCntCke接口，预期返回值等于true
TEST_F(CcuTransportGroupTest, Test_CcuTransportGroup_006)
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
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    SETUP_SOCKET_STATUS_MOCKS();

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
    vector<CcuTransport*> utCcuTransportVec;
    utCcuTransportVec.emplace_back(std::move(utCcuTransport.get()));

    // 创建utCcuTransportGroup
    CcuTransportGroup utCcuTransportGroup(utCcuTransportVec, utCntCke);
    HcclResult res = utCcuTransportGroup.CheckTransportCntCke();

    EXPECT_EQ(HcclResult::HCCL_SUCCESS, res);

    delete socket;
    delete rdmaHandle;
}

// 测试CheckTransportCntCke接口，预期返回值等于false
TEST_F(CcuTransportGroupTest, Test_CcuTransportGroup_007)
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
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));

    MockerCcuFeature();

    impl.Init(commParams, "ranktable.json", config);

    // 创建Socket
    Socket* socket = new Socket(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    std::string socketTag = impl.GetEstablishLinkSocketTag();
    SocketConfig socketConfig(remoteRank, linkData, socketTag);
    impl.GetSocketManager().connectedSocketMap[socketConfig] = std::make_shared<Socket>(
        hccpSocketHandle, GetAnIpAddress(), 0, GetAnIpAddress(), "stub", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);

    SETUP_SOCKET_STATUS_MOCKS();

    // 打桩CcuConnection构造函数中调用的函数
    JfcHandle jfcHandle = 1;
    RdmaHandle rdmaHandle = new int(1);
    u32 jettyNum = 1; // 当前迭代，jettyNum默认为1
    u32 sqSize = 128; // 当前迭代，默认使用MS，故sqSize固定为128。sqSize就是jetty深度

    SETUP_CCU_TRANSPORT_MOCKS_EX(HcclResult::HCCL_E_RESERVED);

    // 创建utConnection
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    // 创建utCcuTransport
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::unique_ptr<CcuTransport> utCcuTransport
        = std::make_unique<CcuTransport>(socket, std::move(connection), locCclBufInfo);
    vector<CcuTransport*> utCcuTransportVec;
    utCcuTransportVec.emplace_back(std::move(utCcuTransport.get()));

    // 创建utCcuTransportGroup
    CcuTransportGroup utCcuTransportGroup(utCcuTransportVec, utCntCke);
    HcclResult res = utCcuTransportGroup.CheckTransportCntCke();

    EXPECT_EQ(HcclResult::HCCL_E_INTERNAL, res);

    delete socket;
    delete rdmaHandle;
}
